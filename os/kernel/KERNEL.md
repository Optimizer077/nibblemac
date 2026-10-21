# pinfer kernel: lab notebook

Owner: OS/kernel. Scope: `os/kernel/`, `build/kernel/`, `logs/kernel/`.
Status (2026-10-21): **built and staged, NOT installed.** Nothing on the Pi has been changed.
The lead installs it later with the procedure in §5.

## 1. What and why

Design §4.2 needs (a) tickless isolated cores 1-3 for the inference threads and
(b) huge pages for 0.5-1.5 GB of streamed weights. The stock Raspberry Pi OS kernel
`6.18.50+rpt-rpi-v8` has `CPU_ISOLATION=y` but **no** `NO_HZ_FULL`, `RCU_NOCB_CPU`,
`HUGETLBFS` or `TRANSPARENT_HUGEPAGE` (notes.md, 2026-10-21). So `isolcpus=` works on stock,
but `nohz_full=`, `rcu_nocbs=` and every huge-page path need a rebuilt kernel. This kernel
is the stock kernel plus exactly those options.

## 2. Source (exact)

| Item | Value | Evidence |
|---|---|---|
| Running kernel | `6.18.50+rpt-rpi-v8`, Debian pkg `1:6.18.50-1+rpt1` (2026-09-11), 4 KB pages | `uname -a`, `getconf PAGESIZE` via harness |
| Source repo | `https://github.com/raspberrypi/linux` | |
| **Commit** | **`cff533aec2fa601846766b32ff57204e0a61bed7`** ("configs: Enable the MXL86110 PHY driver", 2026-09-10) | Pi: `/usr/share/doc/linux-image-6.18.50+rpt-rpi-v8/changelog.Debian.gz` → "Linux commit: cff533ae…" |
| Branch | `rpi-6.18.y` (head at build time `8bdcd530`); cff533ae is an ancestor | GitHub compare cff533ae...rpi-6.18.y: `ahead 5247, behind 0` |
| Makefile version | 6.18.50 | `make kernelversion` (build log) |
| Clone | `git fetch --depth 1 origin <commit>` into WSL `/root/src/linux-rpi` (single commit, detached) | build.sh |
| Base config | Pi `/boot/config-6.18.50+rpt-rpi-v8`, sha256 `545260ef…c746`, 9896 lines | pulled read-only via `harness/pi.py run`; copy in `build/kernel/stock/` |

Residual difference from the stock binary: Raspberry Pi builds the Debian package from
vanilla 6.18.50 + "Raspberry Pi patch" (= this commit) + Debian packaging patches, with
gcc-14.2. We build the raspberrypi/linux commit directly with Ubuntu gcc 15.2. Drivers and
DT are the same source, so driver behaviour should match. Modules are not interchangeable
(different release string and MODVERSIONS CRCs), which is why our modules get their own
`/lib/modules/<krel>/`.

## 3. Config

Fragment: `os/kernel/pinfer.config`, merged with `scripts/kconfig/merge_config.sh -m` onto the
stock config, then `make olddefconfig`. `build.sh` then checks every symbol in the fragment
with `scripts/config --state` and stops on any mismatch. Result (`logs/kernel/config-state-4k.txt`):
all 17 fragment symbols OK.

Requested → final:

| Symbol | Stock | pinfer | Why |
|---|---|---|---|
| `NO_HZ_FULL` | n (NO_HZ_IDLE=y) | **y** | tickless cores (`nohz_full=1-3`) |
| `VIRT_CPU_ACCOUNTING_GEN` / `TICK_CPU_ACCOUNTING` | n / y | **y / n** | required by NO_HZ_FULL |
| `CONTEXT_TRACKING_USER` | n | **y** | required by NO_HZ_FULL |
| `RCU_NOCB_CPU` | n | **y** (`RCU_NOCB_CPU_DEFAULT_ALL=n`, `RCU_LAZY=n`) | `rcu_nocbs=1-3` |
| `CPU_ISOLATION` | y | y | `isolcpus=` (unchanged) |
| `HUGETLBFS` / `HUGETLB_PAGE` | n / n | **y / y** | 2 MB hugetlb pages, `MAP_HUGETLB` |
| `TRANSPARENT_HUGEPAGE` | n | **y**, mode **MADVISE** | THP only where `madvise(MADV_HUGEPAGE)` asks |
| `READ_ONLY_THP_FOR_FS` | n | **y** | khugepaged / `MADV_COLLAPSE` on read-only file mappings |
| `ARM64_CONTPTE` | n | y (auto with THP) | 64 KB contiguous-PTE mappings of large folios |
| `LOCALVERSION` | "-v8" | **"-pinfer"** | release string `6.18.50-pinfer` |
| `HZ`, `PREEMPT`, page size, VA bits | 250, PREEMPT=y, 4K, 39 | unchanged | |

Full diff: `logs/kernel/diffconfig-4k.txt` (52 lines, `scripts/diffconfig stock new`). Apart
from the table, the diff contains only (a) auto-derived symbols the new options pull in
(`ARCH_ENABLE_THP_MIGRATION`, `THP_SWAP`, `HUGETLB_PMD_PAGE_TABLE_SHARING`, `PADATA`, ...),
(b) toolchain-version symbols (`GCC_VERSION 140200 -> 150200`, `RUSTC_*`, `PAHOLE_VERSION`),
and (c) two drops caused by our toolchain, both harmless here:
- `COMPAT_VDSO`/`THUMB2_COMPAT_VDSO`: needs a 32-bit ARM cross-compiler (`CROSS_COMPILE_COMPAT`),
  not installed in WSL. Effect: 32-bit (armhf) processes use a syscall for `gettimeofday`.
  Pi OS Lite arm64 runs no 32-bit processes in our setup. To restore it: install
  `gcc-arm-linux-gnueabihf` in WSL and add `CROSS_COMPILE_COMPAT=arm-linux-gnueabihf-`.
- `RUST_IS_AVAILABLE`: informational; `CONFIG_RUST` is off in stock too.

Note: NO_HZ_FULL only changes behaviour on CPUs named in `nohz_full=`. Without the
split-core cmdline this kernel runs like stock (plus a small context-tracking cost, which is
why we measure "pinfer kernel, no cmdline" separately from stock; §7).

## 4. Build

`os/kernel/build.sh` (run from Git Bash in the project root; it re-runs itself inside WSL):

```
os/kernel/build.sh                    # STEPS="config build stage" by default
STEPS=config os/kernel/build.sh       # only source + config + verification
```

What it runs (WSL, `/root/src`):
```
git fetch --depth 1 origin cff533aec2fa601846766b32ff57204e0a61bed7 && git checkout --detach FETCH_HEAD
cp build/kernel/stock/config-6.18.50+rpt-rpi-v8 $O/.config
scripts/kconfig/merge_config.sh -m -O $O $O/.config os/kernel/pinfer.config
make O=$O ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- LOCALVERSION= olddefconfig
make O=$O ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- LOCALVERSION= -j20 Image modules dtbs
make ... INSTALL_MOD_PATH=<stage> INSTALL_MOD_STRIP=1 modules_install
```
`LOCALVERSION=` (empty) on the make line stops `setlocalversion` from appending `+` for an
untagged git tree, so the release is exactly `6.18.50-pinfer`. Out-of-tree object dir:
`/root/src/kbuild-pinfer-4k`.

Run log (WSL2 Ubuntu 26.04, gcc 15.2, 20 cores):

| Step | Time | Log |
|---|---|---|
| shallow fetch of cff533ae (single commit) | 73 s | `logs/kernel/build-4k-20261021-221252.log` |
| config (merge + olddefconfig + verify) | ~10 s | same, and `build-4k-20261021-221437.log` |
| `Image modules dtbs` -j20 | **1188 s (19.8 min)**, 0 compiler warnings in the log | `logs/kernel/build-4k-20261021-221437.log` |
| stage (modules_install, tarball, mirror) | ~40-55 s | latest `build-4k-*.log` |

Disk in WSL: source 2.0 GB (`/root/src/linux-rpi`), objects 1.4 GB (`/root/src/kbuild-pinfer-4k`).
Re-running `build.sh` is incremental, because the object dir is kept.

Output identity: `Linux version 6.18.50-pinfer (root@DESKTOP-DFIK5HO) (aarch64-linux-gnu-gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0, GNU ld (GNU Binutils for Ubuntu) 2.46) # SMP PREEMPT`.
Module vermagic: `6.18.50-pinfer SMP preempt mod_unload modversions aarch64`.

## 5. Install / test / rollback (for the lead; run on the Pi)

Layout on the Pi after install. **No stock file is overwritten.** The firmware `os_prefix`
setting makes the pinfer kernel load *everything* (kernel, DTB, overlays, cmdline.txt,
initramfs) from its own directory. Per the official config.txt docs, os_prefix is ignored
if the prefixed kernel and DTB are missing, and prefixed overlays are used only if
`<prefix>overlays/README` exists. We ship that README.
```
/boot/firmware/kernel8.img, *.dtb, overlays/, cmdline.txt, config.txt   (stock, untouched)
/boot/firmware/pinfer/kernel8-pinfer.img
/boot/firmware/pinfer/bcm2711-rpi-*.dtb, overlays/*.dtbo + README
/boot/firmware/pinfer/cmdline.txt        stock cmdline + panic=10 (+ split-core flags later)
/boot/firmware/pinfer/initramfs8-pinfer  (auto_initramfs=1 derives this name from kernel8-pinfer.img)
/boot/firmware/tryboot.txt               config.txt + "[all] os_prefix=pinfer/ kernel=kernel8-pinfer.img"
/lib/modules/6.18.50-pinfer/             (new dir; stock /lib/modules/6.18.50+rpt-rpi-v8 untouched)
/boot/config-6.18.50-pinfer, /boot/System.map-6.18.50-pinfer
```
The root-fs path (`PCIE_BRCMSTB`, `USB_XHCI_PCI`, `USB_STORAGE`, `USB_UAS`, `EXT4_FS`) is
built in (`=y`), so the kernel mounts the SSD root even without an initramfs.

### Proposed sequence

Stage A: plain kernel, one-time boot.
```
# PC
python harness/pi.py snapshot before-pinfer-kernel
python harness/pi.py push build/kernel/pinfer-kernel-6.18.50-pinfer.tar.gz
# Pi (in the harness remote dir)
sha256sum pinfer-kernel-6.18.50-pinfer.tar.gz      # compare with the .sha256 file on the PC
tar xzf pinfer-kernel-6.18.50-pinfer.tar.gz && cd 6.18.50-pinfer
sudo ./install_tryboot.sh --dry-run                # review
sudo ./install_tryboot.sh                          # checksums, installs, writes tryboot.txt
sudo reboot '0 tryboot'                            # ONE-TIME boot of the pinfer kernel
# after ~1-2 min, PC:
python harness/pi.py check
python harness/pi.py run 'uname -r; od -An -tu4 --endian=big /proc/device-tree/chosen/bootloader/tryboot; \
  cat /proc/cmdline; dmesg --level=err,warn | head -50; ls /sys/kernel/mm/transparent_hugepage; \
  grep -i huge /proc/meminfo; ip -br a; lsusb -t; vcgencmd get_throttled; systemctl --failed'
```
Pass criteria: `uname -r` = `6.18.50-pinfer`, tryboot = 1, SSH and Wi-Fi up, the SSD is
root, no new errors versus the stock dmesg (snapshot), `systemctl --failed` empty.
Then **reboot normally** (`sudo reboot`). The Pi must come back on the stock kernel. That
proves the fallback path before anything is made permanent.

Stage B: split-core cmdline (only after A passes). This edits only `pinfer/cmdline.txt`:
```
sudo ./splitcore_cmdline.sh on --hugepages 480     # or 'on' without hugepages
sudo reboot '0 tryboot'
# verify:
cat /sys/devices/system/cpu/isolated /sys/devices/system/cpu/nohz_full   # expect 1-3, 1-3
grep -E 'HugePages_(Total|Free)|Hugepagesize' /proc/meminfo               # expect 480, 2048 kB
cat /proc/irq/default_smp_affinity                                        # expect 1
cat /sys/kernel/mm/transparent_hugepage/enabled                           # expect always [madvise] never
```
Flags (syntax checked against `Documentation/admin-guide/kernel-parameters.txt` at cff533ae):
`isolcpus=nohz,domain,managed_irq,1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0 transparent_hugepage=madvise [hugepagesz=2M hugepages=N]`.
- `isolcpus` format `[flag-list,]<cpu-list>`; the flags `nohz`, `domain` and `managed_irq` exist in 6.18.
  The docs mark isolcpus as "Deprecated - use cpusets instead", but it still works.
  `domain` isolation cannot be undone at runtime.
- `nohz_full=` already implies rcu_nocbs for the same CPUs; we keep `rcu_nocbs=1-3` to be explicit.
  The boot CPU (0) is always kept out of nohz_full (it keeps timekeeping).
- The docs say a residual 1 Hz tick is offloaded to unbound workqueues. Pin them to core 0
  after boot: `echo 1 | sudo tee /sys/devices/virtual/workqueue/cpumask`.
- `transparent_hugepage=madvise` is already the Kconfig default. It is on the cmdline only for the record.
- Huge pages: N = **480** (960 MiB). The Q8_0 model is 834 MB → 398 × 2 MiB pages, plus headroom.
  Boot-time reservation guarantees contiguous memory. It takes 960 MiB away from the page
  cache for the whole uptime (MemTotal 3.7 GiB). Runtime alternative with no reboot, which can
  fail on fragmented memory: `echo 480 | sudo tee /proc/sys/vm/nr_hugepages`.
  `splitcore_cmdline.sh off` removes all of these flags.

Stage C: make it the default (only after A and B pass and the lead decides):
```
sudo ./make_permanent.sh     # only while running 6.18.50-pinfer; backs up config.txt,
                             # appends "# pinfer-kernel BEGIN … [all] os_prefix=pinfer/ kernel=kernel8-pinfer.img … END"
sudo reboot
```
Rollback (any time):
```
sudo ./rollback.sh           # removes the BEGIN..END block from config.txt, moves tryboot.txt aside
sudo ./rollback.sh --purge   # also deletes /boot/firmware/pinfer and /lib/modules/6.18.50-pinfer (not while running it)
```
No-SSH recovery: a hang or panic during a tryboot needs no action. `panic=10` reboots
after a panic, and the tryboot flag is one-shot, so the stock kernel comes back. A silent
hang needs a power-cycle, which also clears the flag. After `make_permanent.sh`: put the
SSD in the PC and delete the `# pinfer-kernel BEGIN..END` lines from `config.txt` on the FAT
partition, or copy back `config.txt.pre-pinfer-*`.

Testing of the scripts (no Pi changes): see §6.

Files the scripts write on the Pi (for `docs/pi-state.md`): `/boot/firmware/pinfer/` (new),
`/boot/firmware/tryboot.txt` (new; any previous one is kept as `tryboot.txt.old-<ts>`),
`/lib/modules/6.18.50-pinfer/` (new), `/boot/config-6.18.50-pinfer` and `/boot/System.map-6.18.50-pinfer` (new).
`make_permanent.sh` additionally changes `config.txt`, keeping a backup `config.txt.pre-pinfer-<ts>`.
Undo for all of it: `rollback.sh --purge` from the stock kernel.

## 6. Staged artifacts and script verification

| Artifact | Size | Note |
|---|---|---|
| `build/kernel/pinfer-kernel-6.18.50-pinfer.tar.gz` | 33 MB | **the install artifact**. Checksum in `.tar.gz.sha256`. Unpacks to `6.18.50-pinfer/` |
| `kernel8-pinfer.img` | 25.8 MB | uncompressed arm64 `Image` (header magic `ARM\x64` checked). Stock `kernel8.img` is 10.2 MB because it is gzip-compressed |
| `lib/modules/6.18.50-pinfer/` | 29 MB, 1914 `.ko.xz` | path list **identical** to stock `/lib/modules/6.18.50+rpt-rpi-v8/kernel` (diff of the 1914 paths) |
| `dtbs/` | 2.4 MB | 5 `bcm2711-*.dtb`, 383 overlays, `overlay_map.dtb`, `hat_map.dtb`, README |
| `config-*`, `System.map-*`, `BUILDINFO`, `SHA256SUMS`, scripts, this file | | `BUILDINFO` is sourced by the scripts (values are shell-quoted) |
| `build/kernel/6.18.50-pinfer/` | 61 MB | mirror for inspection on Windows. It lacks 4 netfilter modules whose names differ only in case (`xt_dscp/xt_hl/xt_rateest/xt_tcpmss`), because NTFS is case-insensitive. They are listed in `CASE_COLLISIONS_NOT_MIRRORED.txt`. **Push the tarball, not this directory.** `install_tryboot.sh`'s checksum step would refuse the mirror |

Checks run (PC only; the Pi was only read):
- Config: 17/17 fragment symbols verified (`logs/kernel/config-state-4k.txt`).
  Re-pulling the stock config from the Pi gave the identical sha256 and the identical result.
- **All 391 device-tree files** (`bcm2711-*.dtb`, `overlays/*.dtbo`, `overlay_map.dtb`,
  `hat_map.dtb`, README) are **byte-identical** (md5) to the Pi's stock `/boot/firmware` files.
  This confirms commit cff533ae is the exact source of the running kernel. (The first staging
  missed `overlay_map.dtb`. Without it the firmware cannot remap `dtoverlay=vc4-kms-v3d` to
  `vc4-kms-v3d-pi4` under os_prefix. Found by this comparison and fixed.)
- Module list identical to stock (1914 paths).
- Pi-side scripts: `os/kernel/test_fakeroot.sh` runs install → splitcore on/on/off →
  make_permanent (refused while running stock, then applied, then idempotent) → rollback →
  purge (refused while running pinfer, then done). It runs against a fake root built from the
  Pi's real `config.txt`/`cmdline.txt` (`os/kernel/testdata/`, sha256 equal to the Pi's).
  **17/17 PASS** (`logs/kernel/script-test-fakeroot.log`). Stock boot files are unchanged at
  every step, and after rollback `config.txt` is byte-identical to the original. Two bugs were
  found and fixed this way: an unquoted `BUILDINFO` value, and a blank line left behind by rollback.
- Not testable off the Pi: the firmware's os_prefix/tryboot behaviour, `mkinitramfs`, and the
  boot itself. Stage A of §5 tests those, with the one-shot fallback.

## 7. Measuring the effect (after install; all via `harness/pi.py bench`, which takes the Pi lock)

Every condition gets 3 reps, thermal state recorded, and only `valid=true` results are kept:

| # | Kernel | cmdline | Engine mapping |
|---|---|---|---|
| K0 | stock `+rpt-rpi-v8` | stock | plain mmap (current) |
| K1 | `-pinfer` | stock + panic=10 | plain mmap: cost of NO_HZ_FULL/context tracking built in |
| K2 | `-pinfer` | + split-core flags | plain mmap: effect of isolation/tickless |
| K3 | `-pinfer` | + split-core | `madvise(MADV_HUGEPAGE)` on the read-only model mmap (file THP) |
| K4 | `-pinfer` | + split-core + hugepages=480 | weights copied into `MAP_HUGETLB` (2 MB) |

Benchmarks:
1. `memband` read, 256 MB and 1 GB buffers, 1/2/3 threads, pinned to cores 1-3 (`taskset -c 1-3`),
   with buffers in 4 KB pages, THP (`madvise`) and hugetlb. Baseline 4.20 GB/s (1 thr, stock). This
   isolates the TLB effect from the engine.
2. Engine decode tok/s (tg64) and prefill (pp64), Qwen3.5-0.8B Q8 and Q4, threads 1-3 on cores 1-3.
   Primary metric: tg64 tok/s. Also report its run-to-run spread, because isolation should mainly
   cut jitter (p99 step time).
3. Jitter check: per-token latency histogram from the engine. Interrupt and tick counts on cores 1-3
   during a run: diff `/proc/interrupts` (LOC = arch timer) before and after. Expect LOC on 1-3 to
   drop to about 1/s under nohz_full.
4. Huge pages actually used: `grep -E 'AnonHugePages|FilePmdMapped|FileHugePages|HugePages_' /proc/meminfo`
   and `/proc/<pid>/smaps_rollup` during the run. A K3/K4 number counts only if these show the mapping
   really is huge.

Engine notes for K3 (file THP, from reading the cff533ae source; to be confirmed on the Pi):
ext4 enables large folios for regular files (`ext4_should_enable_large_folio`, max order 11 ≥
PMD order 9). ext4 uses `thp_get_unmapped_area`, so `mmap` returns a 2 MB-aligned address. A
page fault in a `VM_HUGEPAGE` file VMA reads ahead in PMD-order folios
(`mm/filemap.c: do_sync_mmap_readahead`). So `mmap(PROT_READ, MAP_PRIVATE)` +
`madvise(MADV_HUGEPAGE)` before the first touch should give 2 MB mappings with no copy. Pages
already in the page cache as small folios stay small. Drop them first
(`posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)`, or `echo 1 > /proc/sys/vm/drop_caches`).
`MADV_COLLAPSE` (needs `READ_ONLY_THP_FOR_FS`, enabled) can collapse them afterwards.

Expected size of the effect: the A72 has a small L1 DTLB and a 1024-entry unified L2 TLB with
4 KB pages, which covers only 4 MB. Streaming 0.85 GB costs ~208k page walks per token with 4 KB
pages and ~400 with 2 MB pages. The page-table walks mostly hit in cache, so expect a few
percent of bandwidth, not tens of percent. Isolation mostly affects multi-thread barrier
jitter. These are hypotheses until K0-K4 are measured.

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Kernel does not boot (config/toolchain) | low | tryboot is one-shot; `panic=10`; power-cycle returns to stock |
| Built with gcc 15.2 vs stock gcc 14.2: new warnings/miscompile | low | same source; boot test + dmesg diff vs snapshot |
| A firmware/overlay mismatch between stock files and ours | low | os_prefix loads our own DTB+overlays from the same commit |
| `apt upgrade` installs a new stock kernel | n/a for us | Debian hooks write `kernel8.img`/`initramfs8` only; `pinfer/` untouched. After make_permanent the Pi stays on pinfer. Firmware (start4.elf) updates still apply |
| Boot-time hugepages starve page cache / other agents' jobs | medium | 960 MiB is ~26% of RAM; use runtime `nr_hugepages` or omit `--hugepages` if other workloads suffer |
| `isolcpus=domain` leaves cores 1-3 idle for everything not pinned (llama.cpp baselines, builds) | certain | baselines must pin with `taskset -c 1-3`; turn it off with `splitcore_cmdline.sh off` + reboot |
| COMPAT_VDSO missing | negligible | only 32-bit processes |
| Out-of-tree modules / DKMS | none installed | -- |

## 9. 16 KB-page variant: verdict NOT BUILT (hardware cannot run it)

Evidence:
1. The Pi's CPU: `MIDR_EL1 = 0x410fd083` → Arm Cortex-A72 (part 0xd08) r0p3 (harness, 2026-10-21).
2. Arm Cortex-A72 MPCore TRM r0p3 (100095_0003_06_en), Table 4-35, ID_AA64MMFR0_EL1:
   "[23:20] 16KB Returns 0x0 to indicate that the 16KB granule is not supported."
   4 KB and 64 KB granules are supported.
   (PDF: https://documentation-service.arm.com/static/60368ce38f952d2e4134dc2e, page 4-116)
3. Kernel source at cff533ae, `arch/arm64/kernel/head.S`: `__enable_mmu` checks
   ID_AA64MMFR0_EL1.TGRAN and branches to `__no_granule_support`. That routine marks
   `CPU_STUCK_REASON_NO_GRAN` and loops in `wfe; wfi` forever, before any console output.
   A 16K kernel on a Pi 4 therefore hangs silently at boot. It is not a matter of driver bugs.
4. Consistent with this, Raspberry Pi ships 16K pages only in `kernel_2712.img`
   (BCM2712 = Cortex-A76, which has the 16K granule). Its config on our Pi
   (`/boot/config-6.18.50+rpt-rpi-2712`) has all the BCM2711 drivers compiled in, so missing
   drivers are not the blocker. Arch Linux ARM's `linux-rpi-16k` says "bcm2712/RPi5 ONLY".

So VARIANT=16k in build.sh is refused with this reason, and no 16K image is staged. The only
large-page alternative the A72 MMU supports is the **64 KB granule** (`ARM64_64K_PAGES`).
Raspberry Pi does not ship or test it. It changes the PMD huge page to 512 MB and breaks
userspace that assumes 4 KB pages, so it is not worth it. The contiguous-PTE path
(`ARM64_CONTPTE`, now enabled) already gives 64 KB TLB entries for large folios on the 4 KB kernel.
