# Reproducing the results

From a blank SSD to the scoreboard in [RESULTS.md](RESULTS.md): the hardware, the stock OS, the direct
cable, the harness, the cross-builds, the models, and every benchmark command. Commands are copied from
the audit log `logs/harness.jsonl`; the few steps that were not logged are marked **not logged** and
reconstructed from the scripts and notes. Working rules (locks, snapshots, journals) are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

Stages: [1 hardware](#1-hardware) · [2 PC software](#2-pc-software) · [3 flash](#3-flash-stock-raspberry-pi-os) ·
[4 cable and harness](#4-direct-cable-and-harness) · [5 build](#5-build-wsl-as-root) ·
[6 models and data](#6-models-and-data) · [7 correctness](#7-correctness-before-speed) ·
[8 benchmarks](#8-benchmarks) · [9 reading results](#9-reading-benchresultsjson) · [10 gotchas](#10-gotchas)

## 1. Hardware

| Item | What was used | Source |
|---|---|---|
| Board | Raspberry Pi 4 Model B **Rev 1.1**, 4 GB (BCM2711, 4 × Cortex-A72, 1.5 GHz max on this revision) | `docs/pi-state.md`, `notes.md` |
| Boot disk | Crucial X6 1 TB USB SSD, booted over USB; no SD card | `docs/pi-state.md` |
| Power, cooling | The design asks for the official 5.1 V / 3 A supply and a heatsink or fan during benchmarks; every quoted run reported throttle flags `0x0` | `docs/design.md` §6, result files |
| Control link | One Ethernet cable directly between PC and Pi (1 Gbit/s, no router) | `docs/journal/2026-10-21.md` 21:14 |
| Internet for the Pi | Wi-Fi to a home router (drops every few minutes, see §10) | `docs/pi-state.md` |
| PC | Windows 11, WSL2 Ubuntu 26.04 (20 threads, 31 GB RAM), Python 3.13, uv, git, OpenSSH | `notes.md` |

No monitor or keyboard is needed: the first boot is headless.

## 2. PC software

- **Windows:** Python ≥ 3.11 for the harness (`harness/pi.py` uses `tomllib`; 3.13 was used), `uv`, git,
  OpenSSH with a key pair (`~/.ssh/id_ed25519_desktop`), WSL2.
- **WSL Ubuntu 26.04 packages** (`notes.md`): build-essential, cmake, ninja, gcc/g++-aarch64-linux-gnu
  15.2, mmdebstrap, qemu-user + qemu-user-binfmt, device-tree-compiler, bison, flex, libssl-dev,
  dosfstools, parted, kpartx. `qemu-user-static` no longer exists on 26.04.
- **WSL Python environments** (as root; versions recorded, creation commands **not logged** except the
  last one):

  | venv | Contents | Used for |
  |---|---|---|
  | `/root/venv-ref` | transformers 5.20.2, torch 2.14.1 (cpu), numpy, tokenizers | golden data, host tests, prompt files |
  | `/root/venv-convert` | torch 2.11.0+cpu, transformers 4.57.6 | llama.cpp GGUF conversion (`docs/journal/2026-10-21.md` 21:26) |
  | `/root/venv-ortgen` | onnxruntime-genai 0.18.0, onnxruntime 1.31.0, torch 2.14.1+cpu, transformers 5.20.2 | ORT export (`bench/BASELINES.md`) |
  | `/root/venv-pinfer-build` | hatchling: `python3 -m venv /root/venv-pinfer-build && .../pip install hatchling` | wheels (`engine/Makefile`) |

- **Enter WSL as root, always:** `wsl.exe -d Ubuntu -u root`. The default user changed to `optim` on
  2026-10-21 around 23:30, which made the root-owned venvs unreadable. From Git Bash, prefix WSL calls with
  `MSYS_NO_PATHCONV=1`, or `/mnt/...` paths get rewritten. Inside WSL the project lives at
  `/mnt/e/Personal_tries/Raspberry_Pi`; adjust to your checkout.

## 3. Flash stock Raspberry Pi OS

Scripts: `os/stock/`. Image: `2026-10-06-raspios-trixie-arm64-lite.img.xz`, SHA-256
`483db18a48da399b5b7022ffae9a07bc6d99daab30e593aedac09b69ff422843` (`docs/pi-state.md`).

1. Verify the download, then decompress it. `flash_disk.ps1` writes a raw `.img` (it rejects sizes that are
   not a multiple of 512 B) and compares the read-back with the SHA-256 you pass, so pass the hash of the
   **decompressed** image.
2. Render the cloud-init first-boot files (trixie images use cloud-init NoCloud, not `firstrun.sh`;
   `notes.md`). Only a WPA PSK hash and a crypt hash are written, no plaintext:
   ```
   python os/stock/render_firstboot.py --out <dir> --ssid <SSID> --psk-file <file: 64-hex WPA PSK> \
       --pubkey ~/.ssh/id_ed25519_desktop.pub --password-hash-file <file: crypt $6$ hash>
   ```
   Defaults: `--hostname pi4 --user pi --timezone Asia/Seoul --country KR`. Keep the PSK and hash files out
   of git (`*.secret` is gitignored).
3. Flash from an elevated PowerShell. The script refuses any disk whose name, size or bus does not match,
   or that is a boot/system disk:
   ```
   os/stock/flash_disk.ps1 -DiskNumber <N> -ExpectedName '<FriendlyName>' -ExpectedSize <bytes> `
       -ImagePath <image.img> -ExpectedSha256 <sha256 of image.img> -LogPath <log>
   ```
   Later whole-disk hashes will not match: Windows adds `System Volume Information` to the FAT partition
   when the disk comes back online. Compare block by block instead (differences only in the boot partition
   are expected): `os/stock/compare_disk.ps1 -DiskNumber <N> -ImagePath <image.img> -LogPath <log>`.
4. Copy `user-data`, `network-config` and `meta-data` to the boot partition. The original first boot also
   had `cfg80211.ieee80211_regdom=KR` in `cmdline.txt` and an `ssh` flag file (`docs/pi-state.md` #0).
5. Boot the Pi and find its IP from the router or ARP. **`pi4.local` does not work on this network**: a DNS
   server answers `.local` with a public IP (218.38.137.27), so mDNS is never used.

## 4. Direct cable and harness

1. `cp harness/pi.example.toml harness/pi.toml` and set `host` to the Pi's IP, `user = "pi"`,
   `identity_file`. `harness/pi.toml` is gitignored. `remote_dir = "pinfer"` means `~/pinfer` on the Pi;
   `run` and `bench` execute inside it.
2. `python harness/pi.py check` prints the board state. Expect `cpu_features = fp asimd evtstrm crc32 cpuid`
   (no `asimddp`) and `throttled = 0x0`.
3. `python harness/pi.py snapshot stock-baseline`.
4. Pi change #1, Wi-Fi power save off (logged 2026-10-21 21:09; the connection is named after the SSID):
   ```
   python harness/pi.py run 'sudo nmcli connection modify netplan-wlan0-<SSID> 802-11-wireless.powersave 2'
   python harness/pi.py run 'printf "[connection]\nwifi.powersave = 2\n" | sudo tee /etc/NetworkManager/conf.d/90-pinfer-wifi-powersave-off.conf >/dev/null && sudo nmcli general reload conf'
   ```
5. Pi change #2 and PC change P1, the direct cable (logged 21:15–21:16):
   ```
   python harness/pi.py snapshot before-eth-static
   python harness/pi.py run 'sudo nmcli connection modify netplan-eth0 ipv4.method manual ipv4.addresses 10.42.0.2/24 ipv4.gateway "" ipv4.never-default yes ipv6.method link-local && sudo nmcli connection up netplan-eth0'
   ```
   Then, elevated on the PC: `os/pc/eth_link.ps1 -LogPath logs/pc/<timestamp>_eth_link.log` (adapter
   `Ethernet` → 10.42.0.1/24, no gateway; `-Undo` restores DHCP). Set `host = "10.42.0.2"` in `pi.toml`
   and run `check` again (0.32 s over the cable).
6. Pi change #3, bootloader EEPROM (optional; logged 21:17):
   `python harness/pi.py snapshot before-eeprom`, then
   `python harness/pi.py run 'sudo rpi-eeprom-update -a 2>&1 | tail -6; sync; (sleep 2; sudo systemctl reboot) >/dev/null 2>&1 &'`.
7. Add a row with the exact undo command to `docs/pi-state.md` for every change (CONTRIBUTING.md).

The commands in steps 4–6 and in §8.8 are the logged ones without their diagnostic `echo`, `ip`, `ls`
and `pip freeze` parts. From Git Bash, put the remote command in single quotes so `$S`, `$V` and `~`
reach the Pi unexpanded.

## 5. Build (WSL, as root)

```bash
cd /mnt/e/Personal_tries/Raspberry_Pi
make -C engine pi MATVEC=fast   # build/aarch64/engine/pinfer_run, pinfer_ppl: static, NEON kernels
cp build/aarch64/engine/pinfer_run build/aarch64/engine/pinfer_run_fast
cp build/aarch64/engine/pinfer_ppl build/aarch64/engine/pinfer_ppl_fast
make -C engine                  # host build, MATVEC=ref: build/host/engine/pinfer_run (tests)
make                            # build/aarch64/memband
make -C engine/bench            # build/aarch64/kbench, build/aarch64/lbench
```

- The shared engine binary on the Pi is `engine/pinfer_run_fast`; every command below uses that name.
  Experimental builds get their own suffix (`pinfer_run_kern*`, `_lin*`, `_core2`, `_ctx`) so they never
  replace it (`engine/bench/KERNELS.md` Part 2). The copies were made by ad-hoc scripts in the gitignored
  `build/` directory. Copy right after the fast build: `make -C engine pi` relinks `pinfer_run` with
  whichever `MATVEC` was given last, and the default is `ref` (`engine/Makefile`).
- The kbench golden file comes from Windows Python with numpy:
  `python tools/quant.py golden build/kbench_golden.bin`.
- Library and wheels: `make -C engine lib`, `make -C engine check-lib` (fails if the Pi `.so` needs a glibc
  newer than 2.41), `make -C engine wheel-pi MATVEC=fast` → `build/dist/pinfer-0.2.0-py3-none-manylinux_2_34_aarch64.whl`
  (SONAME `libpinfer.so.0.2`; 0.1.0 is described in `docs/release-0.1.0.md`).
  Installation: `docs/library.md`.

Push from the project root on Windows (`push` sets mode 755 on everything and logs the size and SHA-256
of each pushed file, or file count and total size for a directory):

```bash
python harness/pi.py push build/aarch64/engine/pinfer_run_fast build/aarch64/engine/pinfer_ppl_fast engine/bench/spec_exact.sh --dest engine
python harness/pi.py push build/aarch64/memband
python harness/pi.py push build/aarch64/kbench build/aarch64/lbench build/kbench_golden.bin --dest kbench
```

## 6. Models and data

1. **Model.** `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` into
   `models/Qwen3.5-0.8B` (gitignored). The local copy has the `huggingface_hub` local-dir layout; the
   download command was **not logged**.
2. **Convert** (Windows or WSL Python with numpy):
   ```bash
   python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-q4k-headq4k.pq --quant q4k   # default, 436.5 MB
   python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq                      # tests and reference
   ```
   The other variants, **reconstructed** from their names and `notes.md` (invocations not logged):

   | File `models/pq/qwen35-0.8b-…` | Options |
   |---|---|
   | `q8-headq8.pq` | `--quant q8` |
   | `q4-headq4.pq` | `--quant q4` |
   | `q4-headq4k.pq` | `--quant q4 --head q4k` |
   | `mix-headq4k.pq` | `--quant q4k --override 'v_proj=q8' --override 'mtp\.fc=q8'` |
   | `mix-headq8.pq` | `--quant q4k --head q8 --override 'v_proj=q8' --override 'mtp\.fc=q8'` |

   `--no-mtp` drops the MTP head, which disables speculation.
3. **Prompt and text files** (all tracked): `bench/common/prompt64.{ids,json,txt}` (rebuild:
   `/root/venv-ref/bin/python bench/common/make_prompt.py models/Qwen3.5-0.8B`);
   `bench/quality/wikitext2-qwen35.tokens.bin` (32,768 tokens; rebuild: `python bench/quality/make_tokens.py`);
   `bench/common/ctx{512,2048,4096}.ids`, the first N of those tokens (`ctx64`, `ctx8192` come from the
   Phase 6 work).
4. **Push** (only while no benchmark runs: a large push slows a timed run):
   ```bash
   python harness/pi.py push models/pq/qwen35-0.8b-q4k-headq4k.pq models/pq/qwen35-0.8b-mix-headq4k.pq models/pq/qwen35-0.8b-bf16.pq --dest models
   python harness/pi.py push bench/common/prompt64.ids bench/quality/wikitext2-qwen35.tokens.bin --dest data
   python harness/pi.py push bench/common/ctx512.ids bench/common/ctx2048.ids bench/common/ctx4096.ids --dest data
   ```
5. **`data/chat27.ids`** (needed by `spec_exact.sh`) was made on the Pi with the installed Python package
   (logged 2026-10-22 00:31; venv setup in §8.8):
   ```
   python harness/pi.py run --lock 'cd ~/pinfer && ~/pinfer/venvs/pinfer/bin/python - <<"EOF"
   from pinfer.tokenizer import Tokenizer
   from pinfer.chat import apply_chat_template
   tok = Tokenizer.from_path("models/qwen35-0.8b-tokenizer")
   text = apply_chat_template([{"role": "user", "content": "Explain in three sentences why a Raspberry Pi 4 generates text slowly."}], tok.chat_template)
   ids = tok.encode(text, add_special_tokens=False)
   open("data/chat27.ids", "w").write(",".join(map(str, ids)))
   EOF'
   ```

## 7. Correctness before speed

```bash
# WSL root, project root. Golden data goes to ref/golden/ (gitignored, so a fresh clone must create it).
/root/venv-ref/bin/python ref/dump_reference.py    # fp32 hidden states, logits, 32 greedy tokens
/root/venv-ref/bin/python ref/mtp_reference.py     # MTP head reference and acceptance
make -C engine                                     # host binary, MATVEC=ref
/root/venv-ref/bin/python tests/test_golden.py     # per-layer error, logits, batched == token-by-token, greedy 32/32
/root/venv-ref/bin/python tests/test_spec.py       # MTP head vs reference; --spec k == plain for 9 prompts
/root/venv-ref/bin/python tests/test_state.py      # state save/restore (Phase 6)
make -C engine/bench test-qemu                     # kernel golden + correctness checks under qemu
make -C engine host SAN=address,undefined          # sanitizer build -> build/host-san/engine/
python3 tests/test_malformed.py                    # 43 broken .pq / state / token files fail cleanly
/root/venv-ref/bin/python tests/test_long_context.py   # logits to 8192 tokens vs fp32 reference (~45 min)
```
The three `tests/test_*.py` gates also run on the sanitizer build (`--bin build/host-san/engine/pinfer_run`,
with `UBSAN_OPTIONS=halt_on_error=1`); `engine/HARDENING.md` has the results and the long-context table.

On the Pi, speculative output must equal plain output for 2 models × 2 prompts × 5 configurations:

```bash
python harness/pi.py bench spec-exact-final2 -- bash engine/spec_exact.sh ./engine/pinfer_run_fast
python harness/pi.py bench kbench-v11-exact -- ./kbench/kbench --variant v11-exact --threads 1,4 --batch 1,2,3,4,5,8 --time 0.1 --reps 2
```

Pass criteria: 20 `RESULT` lines with `"match":true`; every kbench row `"exact":true`.

## 8. Benchmarks

Every timed command goes through `bench`, which takes the Pi-wide lock, kills the job after 20 minutes,
records the board state before and after, and writes `bench/results/<name>_<timestamp>.json`.

**8.1 Memory bandwidth:** `python harness/pi.py bench memband -- ./memband`

**8.2 llama.cpp v0.7.0** (`bench/llamacpp/`):
```bash
bash bench/llamacpp/build.sh v0.7.0                     # WSL root: static Pi binaries + host llama-quantize
bash bench/llamacpp/make_ggufs.sh Qwen3.5-0.8B v0.7.0   # WSL root, /root/venv-convert: bf16, Q8_0, Q4_0, Q4_K_M
python harness/pi.py push build/aarch64/llama/llama-bench build/aarch64/llama/llama-completion build/aarch64/llama/llama-simple build/aarch64/llama/llama-perplexity build/aarch64/llama/BUILD_INFO.txt --dest llama
python harness/pi.py push bench/llamacpp/llama_bench.sh
python harness/pi.py push models/gguf/Qwen3.5-0.8B-Q8_0.gguf models/gguf/Qwen3.5-0.8B-Q4_0.gguf models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf models/gguf/Qwen3.5-0.8B.SHA256SUMS --dest models
python harness/pi.py bench llamacpp-qwen35-0.8b-Q4_K_M -- ./llama_bench.sh models/Qwen3.5-0.8B-Q4_K_M.gguf
```
`llama_bench.sh` runs `llama-bench -t 1,2,3,4 -p 64 -n 64 -r 3 -o jsonl`. Repeat for Q4_0 and Q8_0.

**8.3 pinfer speed and speculation** (the scoreboard row, `pinfer-final2-q4k_20261022_023950`):
```bash
python harness/pi.py bench pinfer-final2-q4k -- 'for S in "--spec 0" "--spec 2 --draft-vocab 32768" "--spec 3 --draft-vocab 32768" "--spec 4 --draft-vocab 32768"; do ./engine/pinfer_run_fast -m models/qwen35-0.8b-q4k-headq4k.pq -t 4 -n 64 --ids-file data/prompt64.ids $S --bench | grep RESULT; done'
```
`--bench` first runs an untimed prefill + 4 tokens and resets, so the first run after a reboot does not
measure weights paging in from the SSD. `--spec 0` is plain greedy decoding through the same code path.

**8.4 Perplexity** (8 chunks × 512 tokens, 2,040 scored):
```bash
python harness/pi.py bench pinfer-ppl-c8-final2-q4k -- ./engine/pinfer_ppl_fast -m models/qwen35-0.8b-q4k-headq4k.pq -f data/wikitext2-qwen35.tokens.bin --chunks 8 -t 4
bash bench/quality/ppl_pc_c8.sh       # WSL root: llama.cpp on the same 8 chunks, PC
bash bench/quality/ppl_pc.sh 64 16    # WSL root: 64 chunks -> bench/quality/results_pc_c64.jsonl
```
The PC scripts call `/root/src/llama.cpp-v0.7.0/build-host/bin/llama-perplexity`, a host target added to
the tree `build.sh` creates (`bench/BASELINES.md`). The text is `bench/quality/wikitext-2-raw-test.txt`
(rebuild: `bench/quality/make_wikitext.py`).

**8.5 Context length** (`-c` = prompt + 64; 8192 used `-c 8256 --no-warmup`):
```bash
python harness/pi.py bench pinfer-ctx2048-new-q4k -- 'for S in "--spec 0" "--spec 3 --draft-vocab 32768"; do ./engine/pinfer_run_ctx -m models/qwen35-0.8b-q4k-headq4k.pq -t 4 -n 32 -c 2112 --ids-file data/ctx2048.ids $S --bench | grep RESULT; done'
```
`pinfer_run_ctx*` were the Phase 6 builds (`engine/CONTEXT.md`); the final one (`ctx7`) is merged and installed as `pinfer_run_fast`.

**8.6 MNN and ONNX Runtime GenAI** (details and failure modes: `bench/BASELINES.md`):
```bash
bash bench/mnn/build.sh 3.6.1        # WSL root -> build/baselines/mnn/
# model taobao-mnn/Qwen3.5-0.8B-MNN @ 3208b55b2e3c948016c9d380a697be881a1f12f8, staged without visual.mnn
python harness/pi.py push bench/ort bench/common bench/mnn --dest baselines/bench
python harness/pi.py push build/baselines/mnn --dest baselines
python harness/pi.py push build/baselines/stage/Qwen3.5-0.8B-MNN --dest baselines/models
bash bench/common/run_baselines.sh mnn

bash bench/ort/export.sh int4-acc4                        # WSL root, /root/venv-ortgen
python bench/ort/fix_config.py models/baselines/ort/int4-acc4   # works around an ORT GenAI 0.18 CPU bug
python harness/pi.py push models/baselines/ort/int4-acc4 --dest baselines/models/ort
bash bench/common/run_baselines.sh ort-int4
```
The ORT venv on the Pi is built offline: `python3 -m venv venvs/ort; venvs/ort/bin/pip install --no-index
--find-links wheels/ort wheels/ort/*.whl` after pushing the wheels with `--dest wheels`.

**8.7 Kernel microbenchmarks:**
```bash
python harness/pi.py bench kbench-v9-final -- ./kbench/kbench --golden kbench/kbench_golden.bin --variant v9-final --threads 1,2,3,4 --batch 1,2,4,8
python harness/pi.py bench lbench-v3-chunkmajor -- ./kbench/lbench --threads 1,4 --n 1,4,64
```
(The original run used the then-current binary name `./kbench/kbench-v9`.) Profiles of a whole step:
prefix a `pinfer_run_fast` command with `PF_KERN_STATS=1 PF_FWD_STATS=1`.

**8.8 Python package on the Pi.** Download the wheels on the PC (`docs/library.md` §1; pass every
`--platform` tag explicitly), push them with `--dest wheels/pinfer`, then (logged 2026-10-21 22:16 and
2026-10-22 01:18):
```bash
python harness/pi.py run 'python3 -m venv ~/pinfer/venvs/pinfer && V=~/pinfer/venvs/pinfer/bin; W=~/pinfer/wheels/pinfer; $V/pip install -q --no-index --find-links $W --upgrade pip && $V/pip install -q --no-index --find-links $W "pinfer[test]"'
python harness/pi.py push models/Qwen3.5-0.8B/tokenizer.json models/Qwen3.5-0.8B/tokenizer_config.json models/Qwen3.5-0.8B/chat_template.jinja --dest models/qwen35-0.8b-tokenizer
python harness/pi.py push ref/golden/qwen35-0.8b/meta.json --dest golden/qwen35-0.8b
python harness/pi.py push tests/python --dest tests
python harness/pi.py run --lock 'cd ~/pinfer && V=~/pinfer/venvs/pinfer/bin; PINFER_TEST_REQUIRE=1 PINFER_TEST_MODEL=models/qwen35-0.8b-bf16.pq PINFER_TEST_TOKENIZER=models/qwen35-0.8b-tokenizer PINFER_TEST_GOLDEN=golden/qwen35-0.8b/meta.json $V/python -m pytest -q tests/python'
python harness/pi.py run --lock 'cd ~/pinfer && ~/pinfer/venvs/pinfer/bin/pinfer bench -m models/qwen35-0.8b-q4k-headq4k.pq --tokenizer models/qwen35-0.8b-tokenizer --prompt-tokens 64 -n 64 --json --spec-k 3'
```

**8.9 Custom kernel (optional, changes the boot setup).** Build, install, test and roll back as in
`os/kernel/KERNEL.md` §4–§5. Take a snapshot first; `docs/pi-state.md` #4 has the undo
(`rollback.sh --purge`). The K1–K3 runs in RESULTS.md §10 used the same `pinfer_run_fast` commands as
§8.3 (with `--spec 0` and `--spec 3` only) after `sudo reboot "0 tryboot"`; `splitcore_cmdline.sh on`
added the split-core flags, `PINFER_HUGEPAGES=1` the THP copy.

## 9. Reading `bench/results/*.json`

One file per `bench` call (`harness/pi.py` `cmd_bench`):

| Field | Meaning |
|---|---|
| `name`, `command`, `timestamp`, `exit_code` | What ran and how it ended (124: killed by the 20-minute cap) |
| `results` | Every stdout line starting with `RESULT `, parsed as JSON |
| `unparsed` | `RESULT` lines that were not valid JSON |
| `stdout`, `stderr` | Full output (profiles such as `KSTATS` and `FSTATS` are on stderr) |
| `state_before`, `state_after` | Board state: model, kernel, cmdline, page size, CPU features, governor, clocks, isolated / nohz_full CPUs, temperature, throttle flags, huge pages, THP mode, free memory |
| `valid` | `exit_code == 0` **and** `state_after.throttled` parses as `0x0` **and** `unparsed` is empty |

- The throttle flags come from `vcgencmd get_throttled`, which also reports events since boot; if the
  board cannot report them (no `vcgencmd`, user not in `video`), the run is invalid. Quote only
  `valid=true` runs.
- `valid` says nothing about the outcome. `spec-exact-old_20261022_004457.json` is `valid=true` with 4 of
  20 `"match":false`. Read correctness from the tool's fields: `match` (spec_exact.sh), `exact`, `ok`,
  `noisy` (kbench), `identical` (ctx_check.sh).
- `valid` also cannot see interference from other users of the Pi; kbench's own foreign-load guard marks
  such repetitions `noisy`, and the lab notebooks record runs that were invalid as measurements
  (`engine/bench/KERNELS.md` v1, prefetch sweep).
- In a `for` loop the exit code is that of the last command, so count the `RESULT` lines.

| Tool | Main `RESULT` fields |
|---|---|
| `pinfer_run --bench` | `engine` (`pinfer-fast`, or `pinfer-fast-spec` with `--spec`), `model`, `threads`, `n_prompt`, `n_gen`, `prefill_tps`, `decode_tps` (tokens 2..n), `peak_rss_mb`; with `--spec`: `spec_k`, `draft_vocab`, `rounds`, `drafted`, `accepted`, `accept_rate`, `tokens_per_pass`, `mtp_ms_per_pass`, `verify_ms_per_pass`, `commit_ms_per_pass` |
| `pinfer_ppl` | `ppl`, `ppl_err`, `ctx`, `chunks`, `tokens_scored`, `seconds` |
| `llama_bench.sh` | llama-bench JSONL: `n_threads`, `n_prompt` (pp rows), `n_gen` (tg rows), `avg_ts`, `stddev_ts` |
| `memband` | `read_gbs` per thread count |
| `kbench` | `dtype`, `shape`, `threads`, `n`, `us`, `gbs`, `rel_err_kernel`, `rel_err_fp`, `exact`, `noisy`, `ok` |
| `spec_exact.sh` | `model`, `prompt`, `spec`, `md5` of the generated ids, `match` |

## 10. Gotchas

| Symptom | Cause | What to do | Source |
|---|---|---|---|
| Root-owned venvs unreadable in WSL | Default WSL user switched from root to `optim` (2026-10-21, ~23:30) | `wsl.exe -d Ubuntu -u root` | journal 2026-10-21, `engine/SPEC.md` §6 |
| `/mnt/...` paths mangled from Git Bash | MSYS path conversion | `MSYS_NO_PATHCONV=1` | `notes.md` |
| `$VAR` expanded too early in `wsl.exe -- bash -c '…'` | The outer shell expands it | Put the commands in a script file | `bench/BASELINES.md` |
| `pi4.local` unreachable | The network's DNS answers `.local` with a public IP | Use the IP: 10.42.0.2 over the cable | journal 2026-10-21 ~21:05 |
| SSH timeouts, broken downloads on the Pi | Wi-Fi drops every 2–3 minutes (`reason=7`: the AP forgets the Pi) | Control over the cable; download on the PC and push; run long jobs on the Pi | journal 2026-10-21, `CLAUDE.md` |
| First bench after a reboot: prefill ~20 instead of ~27 tok/s | The mmap'd weights fault in from the SSD | `--bench` warm-up (on by default) | `notes.md`, `engine/CORE.md` |
| Lock queue stalled for 24–42 minutes | One locked job held the Pi for 38 minutes | Locked jobs are killed after 20 minutes; keep jobs short (8-chunk perplexity) | journal 2026-10-21 23:05 |
| A timed run is slower than it should be | Another unlocked job, push or checksum on the Pi | `run --lock` for anything heavy; push only when no bench runs | `engine/bench/KERNELS.md`, `bench/BASELINES.md` |
| `.so` does not load on the Pi (`sqrtf@GLIBC_2.43`) | WSL glibc 2.43 > Pi glibc 2.41 | `-fno-math-errno` (in the Makefile); `make -C engine check-lib` | journal 2026-10-21 22:00 |
| `pip download` fetched x86_64 wheels | Platform tags passed through a shell variable | Pass each `--platform` tag explicitly | journal 2026-10-21 22:15 |
| Pi pytest: 50 tests skipped | Wrong environment variable names | `PINFER_TEST_REQUIRE=1` and `PINFER_TEST_*` as in §8.8 | journal 2026-10-22 00:06 |
| Shell scripts fail on the Pi | CRLF line endings written on Windows | `.gitattributes` forces LF; avoid `Path.write_text` for scripts | `bench/BASELINES.md` |
| A failed background build reported exit 0 | `\| tail` hid the exit status | Check the exit status explicitly | journal 2026-10-21 21:20 |
| ORT: "GetMemoryInfo for CPU should not be used" | `shared_initializers` written by the builder | `bench/ort/fix_config.py` | `bench/BASELINES.md` |
