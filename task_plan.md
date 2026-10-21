# Task plan: pinfer (Pi 4 native LLM runtime + split-core OS)

Design: [docs/design.md](docs/design.md). Findings: [notes.md](notes.md).

## Current phase: 0 — Setup

### P0 (now)
- [x] Check PC tooling (git, ssh, uv, QEMU, WSL2 Ubuntu 26.04) — 2026-10-21
- [x] Install WSL toolchain: gcc-aarch64-linux-gnu 15.2, mmdebstrap 1.5.7, qemu-user-binfmt
- [x] Write design doc, plan, notes
- [x] Harness skeleton `harness/pi.py` (check / push / run / bench)
- [x] Memory-bandwidth microbench `bench/memband/memband.c`, cross-compiled, runs under QEMU
- [x] Code review → fixed: exec bit on push, unknown throttle = invalid, stderr saved,
      bad RESULT lines kept, remote_dir quoted, threads pinned to CPUs, slice alignment
- [x] Flashed Raspberry Pi OS Lite 64-bit 2026-10-06 (trixie) to Crucial X6 1 TB USB SSD
      (disk 2) with cloud-init first boot: host pi4, user pi (SSH key only), Wi-Fi
      <wifi-ssid> (PSK hash), country KR, Ethernet DHCP fallback. Verified: rootfs
      bit-identical, boot files identical. Scripts: os/stock/*. Login password: harness/pi.secret
- [x] `harness/pi.toml` created (pi4.local, user pi)
- [x] Pi booted from SSD, joined Wi-Fi (192.168.1.166), `harness/pi.py check` passes

### P1 — Baseline (in progress)
- [x] memband on Pi: ~4.2 GB/s, saturated by 1 core (bench/results/memband_20261021_210615.json)
- [ ] Build llama.cpp on Pi (`-mcpu=cortex-a72`), Qwen small model Q8_0 + Q4_0 → tok/s
- [ ] Record baseline table in notes.md

### P2+ — see design §5

## Decisions
- 2026-10-21: Level 2 (split-core Linux), bare metal later (user).
- 2026-10-21: C11 + NEON, Debian trixie, Python harness over SSH (call).

## Blockers
- Pi not yet on network (needs user to flash SD card).
