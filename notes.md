# Notes and findings

## 2026-10-21 — PC environment
- Windows 11, PC on wired Ethernet (Realtek 2.5GbE), Wi-Fi off.
- Present: git, OpenSSH (ssh/scp), Python 3.13, uv, QEMU (`qemu-system-aarch64`).
- WSL2 Ubuntu 26.04, 20 threads, 31 GB RAM, runs as root.
- Installed in WSL: build-essential, cmake, ninja, gcc/g++-aarch64-linux-gnu 15.2,
  mmdebstrap 1.5.7, qemu-user + qemu-user-binfmt (aarch64 binfmt registered),
  device-tree-compiler, bison, flex, libssl-dev, dosfstools, parted, kpartx.
- Gotcha: `qemu-user-static` no longer exists on Ubuntu 26.04 → use `qemu-user-binfmt`.
- Gotcha: from Git Bash, call WSL with `MSYS_NO_PATHCONV=1` or `/mnt/...` paths get mangled.
- Existing SSH key: `~/.ssh/id_ed25519_desktop(.pub)`.

## 2026-10-21 — Flashing stock Raspberry Pi OS (trixie 2026-10-06)
- First boot is **cloud-init (NoCloud)** from boot partition `user-data`, `network-config`,
  `meta-data`; the old `firstrun.sh`/`custom.toml` paths are not used.
- Image ships a placeholder user `pi` (nologin). Pi's cloud-init distro class renames the
  first user in `users:` via userconf and masks `userconfig.service`.
- netplan → NetworkManager; a 64-hex PSK renders to `key-mgmt=wpa-psk`, `pmf=2` (optional).
  Router <wifi-ssid> is WPA2/WPA3 mixed; WPA3-only would need a plaintext passphrase.
- Validated offline: `cloud-init schema` + `netplan generate --root-dir` inside a qemu chroot.
- Windows raw write: take disk offline, write `\\.\PhysicalDriveN` (elevated). After the disk
  comes back online, Windows adds `System Volume Information` to the FAT partition, so a
  whole-disk hash will not match; compare per partition instead (os/stock/compare_disk.ps1).

## Pi 4 facts (to verify on device in P0/P1)
- `/proc/cpuinfo` Features should show `asimd` but not `asimddp` (no dot product).
- Page size via `getconf PAGESIZE` (expect 4096).
- Check `vcgencmd get_throttled` = `0x0` before trusting any benchmark.

## 2026-10-21 — First boot on the real Pi
- Board: Raspberry Pi 4 Model B **Rev 1.1** (early board, 1.5 GHz max), boots from USB SSD
  (`/dev/sda2`, resized to 917 GB). Kernel 6.18.50+rpt-rpi-v8. Python 3.13.5.
- Network: wlan0 on <wifi-ssid>, 192.168.1.166 (eth0 down). Internet OK (ping 1.1.1.1 ~5 ms).
- **`pi4.local` resolves to a public IP (218.38.137.27) on this PC** — a DNS server answers
  `.local`, so mDNS is never used. Harness uses the IP. Fix later: DHCP reservation on router.
- CPU features: `fp asimd evtstrm crc32 cpuid` → confirmed no dot product, no FP16 math.
- Stock kernel config: `NO_HZ_FULL` off, `HUGETLBFS` off, `TRANSPARENT_HUGEPAGE` off,
  `CPU_ISOLATION=y`, `PREEMPT=y`, 4 KB pages. → `isolcpus` works on stock, but tickless
  cores and huge pages **need our own kernel build** (design §4.2 risk confirmed).
- Bootloader EEPROM update available (not applied).

## Baseline numbers
| Date | Bench | Result | Notes |
|---|---|---|---|
| 2026-10-21 | memband read, 256 MB | 1 thr 4.20 / 2 thr 4.08 / 4 thr 4.07 GB/s | 38 °C, not throttled, governor ondemand |

**Key finding:** one core already saturates DRAM (~4.2 GB/s). Extra cores add no bandwidth;
they are only useful for the compute (dequantization, math) that must keep pace with it.
Decode ceiling ≈ 4.2 GB/s ÷ weight bytes per token, e.g. ~0.4 GB of 4-bit weights → ~10 tok/s.
