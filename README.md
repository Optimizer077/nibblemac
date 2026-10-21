# pinfer — Pi-4-native LLM runtime + split-core OS

Goal: run Qwen models on a Raspberry Pi 4 (4 GB) faster than llama.cpp, with our own
NEON inference engine on isolated cores and a minimal custom Debian image.
Design: [docs/design.md](docs/design.md) · Plan: [task_plan.md](task_plan.md) · Findings: [notes.md](notes.md)

## Layout
```
harness/pi.py        PC-side control over SSH: check | push | run | bench | snapshot
harness/runlog.py    automatic audit logging used by pi.py
bench/memband/       DRAM bandwidth microbenchmark (the tokens/s ceiling)
bench/results/       benchmark JSON records (board state before/after included)
os/stock/            flash + first-boot config scripts for stock Raspberry Pi OS
docs/design.md       design
Makefile             cross-compile in WSL Ubuntu → build/aarch64/
```

## Records (where to look)
| Question | File |
|---|---|
| What changed, when? | `CHANGELOG.md` |
| What happened on a given day, and why? | `docs/journal/YYYY-MM-DD.md` |
| How does the Pi differ from stock, and how do I undo it? | `docs/pi-state.md` |
| Every command ever sent to the Pi | `logs/harness.jsonl` (+ full output in `logs/runs/`) |
| Full Pi configuration at a point in time | `logs/snapshots/<ts>_<label>/` |
| Benchmark numbers | `notes.md` table → `bench/results/*.json` |

## Step 1 — flash the SD card (once)
1. Install **Raspberry Pi Imager** on Windows (raspberrypi.com/software).
2. Device: *Raspberry Pi 4*. OS: *Raspberry Pi OS (other) → Raspberry Pi OS Lite (64-bit)*.
3. Storage: your microSD card. When asked, choose **Edit settings**:
   - Hostname: `pi4`
   - Username / password: e.g. `pi` / your choice
   - Services → **Enable SSH → Allow public-key authentication only**, paste the contents of
     `C:\Users\Optim\.ssh\id_ed25519_desktop.pub`
   - Wi-Fi: optional (we use Ethernet).
4. Write, insert the card, connect the Pi **by Ethernet to your router**, power it with the
   official 5.1 V / 3 A supply. Wait ~1 minute for first boot.

## Step 2 — connect the harness
```powershell
copy harness\pi.example.toml harness\pi.toml    # edit host/user if you changed them
python harness\pi.py check                       # prints model, kernel, temp, throttle flags...
```
If `pi4.local` does not resolve, use the Pi's IP from your router's device list.

## Step 3 — first measurement
```powershell
wsl -d Ubuntu -- make -C /mnt/e/Personal_tries/Raspberry_Pi
python harness\pi.py push build\aarch64\memband
python harness\pi.py bench memband -- ./memband
```
