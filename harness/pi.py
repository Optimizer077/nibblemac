"""PC-side harness for the Raspberry Pi 4: check, push, run, bench, snapshot.

Usage (from the project root):
    python harness/pi.py check
    python harness/pi.py push build/aarch64/memband
    python harness/pi.py run "./memband"         # one shell string, run in remote_dir
    python harness/pi.py bench memband -- ./memband
    python harness/pi.py snapshot before-kernel  # full Pi config record, for diffing later

Every command is logged automatically (see harness/runlog.py):
    logs/harness.jsonl          one line per command: what, host, exit code, duration, outputs
    logs/runs/<ts>_run.log      full output of each `run`
    logs/snapshots/<ts>_<label>/  Pi configuration records
    bench/results/<name>_<ts>.json  benchmark results + board state before/after
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from runlog import LOG_DIR, ROOT, describe_paths, log_event, redact, rel, stamp, tee_command

CONFIG_PATH = ROOT / "harness" / "pi.toml"
RESULTS_DIR = ROOT / "bench" / "results"

# One remote shell snippet that prints KEY=VALUE lines describing the board state.
STATE_SCRIPT = r"""
echo "model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
echo "boot_time=$(uptime -s)"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "kernel=$(uname -r)"
echo "cmdline=$(cat /proc/cmdline)"
echo "page_size=$(getconf PAGESIZE)"
echo "cpu_features=$(grep -m1 Features /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "freq_khz=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | xargs)"
echo "isolated=$(cat /sys/devices/system/cpu/isolated 2>/dev/null)"
echo "nohz_full=$(cat /sys/devices/system/cpu/nohz_full 2>/dev/null)"
echo "temp=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2)"
echo "throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)"
echo "hugepages_total=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')"
echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "mem_avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
echo "python=$(python3 --version 2>&1)"
"""

# Files captured by `snapshot`: local name -> remote command producing it.
SNAPSHOT_ITEMS = {
    "packages.txt": "dpkg-query -W -f='${Package} ${Version}\\n' | sort",
    "config.txt": "cat /boot/firmware/config.txt",
    "cmdline.txt": "cat /boot/firmware/cmdline.txt",
    "services-enabled.txt": "systemctl list-unit-files --state=enabled --no-legend | sort",
    "kernel-config-hash.txt": "uname -a; sha256sum /boot/config-$(uname -r)",
    "eeprom.txt": "sudo rpi-eeprom-update 2>&1; vcgencmd bootloader_version 2>&1",
    "network.txt": "ip -br addr; nmcli -t -f NAME,TYPE,DEVICE con show",
    "disk.txt": "lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT; df -h /",
    "pip-user.txt": "python3 -m pip list --user 2>/dev/null || echo 'pip not installed'",
    "home-pinfer.txt": "ls -la ~/pinfer 2>/dev/null || echo 'no ~/pinfer'",
}


@dataclass(frozen=True)
class PiConfig:
    host: str
    user: str
    port: int = 22
    remote_dir: str = "pinfer"
    identity_file: str | None = None


def load_config() -> PiConfig:
    """Read harness/pi.toml (copy harness/pi.example.toml to create it)."""
    if not CONFIG_PATH.exists():
        sys.exit(f"Missing {CONFIG_PATH}. Copy harness/pi.example.toml to pi.toml and edit it.")
    with CONFIG_PATH.open("rb") as f:
        return PiConfig(**tomllib.load(f)["pi"])


def ssh_base(cfg: PiConfig) -> list[str]:
    # accept-new: trust a first-seen host key (fresh flash) but still reject a changed one.
    cmd = ["ssh", "-p", str(cfg.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           "-o", "StrictHostKeyChecking=accept-new"]
    if cfg.identity_file:
        cmd += ["-i", str(Path(cfg.identity_file).expanduser())]
    return cmd + [f"{cfg.user}@{cfg.host}"]


def ssh(cfg: PiConfig, command: str, capture: bool = False, stdin: bytes | None = None,
        retries: int = 0) -> subprocess.CompletedProcess:
    """Run a shell command on the Pi. Use retries only for read-only commands:
    exit 255 means "ssh failed", which can also happen after the command already started."""
    for attempt in range(retries + 1):
        proc = subprocess.run(ssh_base(cfg) + [command], input=stdin, capture_output=capture,
                              check=False)  # callers inspect returncode (255 = ssh failure)
        if proc.returncode != 255 or attempt == retries:
            return proc
        print(f"(ssh failed, retrying in 5 s — attempt {attempt + 2}/{retries + 1})", file=sys.stderr)
        time.sleep(5)
    return proc


def board_state(cfg: PiConfig) -> dict[str, str]:
    proc = ssh(cfg, STATE_SCRIPT, capture=True, retries=2)
    if proc.returncode != 0:
        sys.exit(f"SSH failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace')}")
    state = {}
    for line in proc.stdout.decode(errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key:
            state[key] = value.strip()
    return state


def throttle_flags(state: dict[str, str]) -> int | None:
    """vcgencmd throttle bits (current or since boot); None if the board could not report them."""
    try:
        return int(state.get("throttled", ""), 16)
    except ValueError:
        return None


BENCH_LOCK = "/tmp/pinfer-bench.lock"
# A locked job may hold the Pi for at most this long; a 38-min run once stalled four agents.
LOCK_MAX_MINUTES = 20


def remote_cmd(cfg: PiConfig, command: list[str], lock: bool = False,
               max_minutes: int = LOCK_MAX_MINUTES) -> str:
    """`command` is one shell string split by argparse; it runs inside the remote dir.
    lock=True serializes it with every other locked command on the Pi (all benchmarks), so
    parallel agents never measure on top of each other, and kills it after max_minutes
    (exit code 124) so one slow job cannot block the queue."""
    if not command:
        sys.exit("No command given.")
    rdir = shlex.quote(cfg.remote_dir)
    body = " ".join(command)
    if lock:
        body = (f"flock {BENCH_LOCK} timeout --kill-after=30 {max_minutes * 60} "
                f"bash -c {shlex.quote(body)}")
    # `|| exit 1;` (not `&&`) so a failed cd also stops commands after a `;` in the string.
    return f"mkdir -p {rdir} && cd {rdir} || exit 1; {body}"


def cmd_check(cfg: PiConfig, _: argparse.Namespace, extra: dict) -> int:
    state = board_state(cfg)
    extra["state"] = state
    for key, value in state.items():
        print(f"{key:16} {value}")
    flags = throttle_flags(state)
    if flags is None:
        print("\nWARNING: throttle state unknown (vcgencmd missing or user not in 'video' group).")
    elif flags != 0:
        print("\nWARNING: throttle flags set — check power supply and cooling before benchmarking.")
    if "asimddp" in state.get("cpu_features", "").split():
        print("\nNOTE: CPU reports dot-product support — this is not a Pi 4 Cortex-A72.")
    return 0


def cmd_push(cfg: PiConfig, args: argparse.Namespace, extra: dict) -> int:
    """Copy local files/dirs into the remote dir via a tar stream (no rsync needed on Windows)."""
    def executable(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        # Windows has no exec bit; without this, pushed binaries fail with "Permission denied".
        ti.mode = 0o755
        return ti

    extra["files"] = describe_paths(args.paths)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in args.paths:
            path = Path(p)
            tar.add(path, arcname=path.name, filter=executable)
    dest = f"{cfg.remote_dir}/{args.dest}".rstrip("/")
    rdir = shlex.quote(dest)
    proc = ssh(cfg, f"mkdir -p {rdir} && tar -xf - -C {rdir}", stdin=buf.getvalue())
    extra["dest"] = dest
    print(f"pushed {len(args.paths)} item(s) to {cfg.host}:{dest}/" if proc.returncode == 0
          else f"push failed (exit {proc.returncode})")
    return proc.returncode


def cmd_run(cfg: PiConfig, args: argparse.Namespace, extra: dict) -> int:
    command = remote_cmd(cfg, args.command, lock=args.lock)
    extra["locked"] = args.lock
    out = LOG_DIR / "runs" / f"{stamp()}_run.log"
    extra["output"] = rel(out)
    return tee_command(ssh_base(cfg) + [command], out,
                       header=f"# {datetime.now().isoformat(timespec='seconds')} "
                              f"{cfg.user}@{cfg.host}\n$ {' '.join(args.command)}")


def cmd_bench(cfg: PiConfig, args: argparse.Namespace, extra: dict) -> int:
    command = remote_cmd(cfg, args.command, lock=True)
    before = board_state(cfg)
    proc = ssh(cfg, command, capture=True)
    after = board_state(cfg)
    stdout = proc.stdout.decode(errors="replace")
    stderr = proc.stderr.decode(errors="replace")
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    results, unparsed = [], []
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                results.append(json.loads(line[len("RESULT "):]))
            except json.JSONDecodeError:
                unparsed.append(line)
    record = {
        "name": args.name,
        "command": " ".join(args.command),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exit_code": proc.returncode,
        # Valid only if the run succeeded and the board positively reported no throttling.
        "valid": proc.returncode == 0 and throttle_flags(after) == 0 and not unparsed,
        "results": results,
        "unparsed": unparsed,
        "stdout": stdout,
        "stderr": stderr,
        "state_before": before,
        "state_after": after,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}_{stamp()}.json"
    out.write_text(redact(json.dumps(record, indent=2)))
    extra.update(output=rel(out), valid=record["valid"], results=results)
    print(f"\nsaved {rel(out)}  valid={record['valid']}  "
          f"temp {before.get('temp')} -> {after.get('temp')}")
    return proc.returncode


def cmd_snapshot(cfg: PiConfig, args: argparse.Namespace, extra: dict) -> int:
    """Save the Pi's configuration to logs/snapshots/<ts>_<label>/ so changes can be diffed."""
    out = LOG_DIR / "snapshots" / f"{stamp()}_{args.label}"
    out.mkdir(parents=True)
    (out / "state.json").write_text(redact(json.dumps(board_state(cfg), indent=2)))
    failed = []
    for name, command in SNAPSHOT_ITEMS.items():
        proc = ssh(cfg, command, capture=True, retries=2)
        (out / name).write_text(redact(proc.stdout.decode(errors="replace")
                                       + proc.stderr.decode(errors="replace")), encoding="utf-8")
        if proc.returncode != 0:
            failed.append(name)
    extra.update(output=rel(out), failed=failed)
    print(f"saved {rel(out)} ({len(SNAPSHOT_ITEMS) + 1} files"
          + (f", non-zero exit: {', '.join(failed)})" if failed else ")"))
    return 0


PAUSE_FILE = Path(__file__).resolve().parent / "pi.pause"


def pause_message() -> str | None:
    """While harness/pi.pause exists (a reboot window, e.g. the overclock test), every command but
    `check` is refused, except for the holder, whose PINFER_PAUSE_TOKEN matches the file's token= line.
    The file's other lines say why and until when."""
    if not PAUSE_FILE.is_file():
        return None
    lines = PAUSE_FILE.read_text(encoding="utf-8").splitlines()
    token = next((ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("token=")), "")
    if token and os.environ.get("PINFER_PAUSE_TOKEN") == token:
        return None
    why = " ".join(ln for ln in lines if not ln.startswith("token=")).strip() or "no reason given"
    return f"[pi.py] the Pi is PAUSED ({why}); do PC-side work and retry later (exit 75)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="print board state")
    p = sub.add_parser("push", help="copy files to the Pi")
    p.add_argument("paths", nargs="+")
    p.add_argument("--dest", default="", help="subdirectory of remote_dir (default: remote_dir)")
    p = sub.add_parser("run", help="run a command in the remote dir")
    p.add_argument("--lock", action="store_true",
                   help="hold the Pi-wide benchmark lock (use for anything heavy or timed)")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("bench", help="run a benchmark and save results")
    p.add_argument("name")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("snapshot", help="record the Pi's full configuration")
    p.add_argument("label")
    args = parser.parse_args()
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    handlers = {"check": cmd_check, "push": cmd_push, "run": cmd_run,
                "bench": cmd_bench, "snapshot": cmd_snapshot}

    cfg = load_config()
    extra: dict = {}
    t0 = time.monotonic()
    code = 1
    try:
        if args.cmd != "check" and (msg := pause_message()):
            print(msg, file=sys.stderr)
            extra["paused"] = True
            code = 75  # EX_TEMPFAIL: retry later
            return code
        code = handlers[args.cmd](cfg, args, extra)
        return code
    finally:
        log_event({"cmd": args.cmd, "argv": sys.argv[1:], "host": cfg.host,
                   "exit": code, "duration_s": round(time.monotonic() - t0, 2), **extra})


if __name__ == "__main__":
    sys.exit(main())
