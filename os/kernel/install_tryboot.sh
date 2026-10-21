#!/bin/bash
# install_tryboot.sh -- run ON THE PI (as root), from inside the unpacked staged tree:
#     tar xzf pinfer-kernel-<krel>.tar.gz && cd <krel> && sudo ./install_tryboot.sh
#
# Installs the pinfer kernel SIDE BY SIDE with the stock kernel and arms a ONE-TIME
# test boot. Nothing that the stock boot reads is modified:
#   - kernel, DTBs, overlays, cmdline.txt and initramfs go to /boot/firmware/<PREFIX_DIR>/
#     (firmware os_prefix), never over kernel8.img / *.dtb / overlays/ / cmdline.txt
#   - modules go to /lib/modules/<krel>/ (a new directory; stock modules untouched)
#   - /boot/firmware/tryboot.txt = copy of config.txt + os_prefix= + kernel=
#     The firmware reads tryboot.txt instead of config.txt ONLY on a `reboot '0 tryboot'`.
#     The flag is one-shot: the next reboot or any power-cycle boots the stock kernel.
#
# Options
#   --no-initramfs   do not build <PREFIX_DIR>/initramfs8-* (the root-fs path
#                    pcie-brcmstb, xhci-pci, usb-storage, uas, ext4 is built in, so the kernel
#                    can mount root without one)
#   --dry-run        print what would be done, change nothing
# After it succeeds:   sudo reboot '0 tryboot'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/BUILDINFO"           # KREL, IMG_NAME, PREFIX_DIR, COMMIT, VARIANT
# Test hook: PINFER_TEST_ROOT=<dir> runs against a fake root (no hardware checks, no
# mkinitramfs); PINFER_TEST_KREL fakes `uname -r`. Never set these on the Pi.
R="${PINFER_TEST_ROOT:-}"
running_krel() { echo "${PINFER_TEST_KREL:-$(uname -r)}"; }
FW=$R/boot/firmware
DEST="$FW/$PREFIX_DIR"
INITRD="initramfs${IMG_NAME#kernel}"; INITRD="${INITRD%.img}"   # kernel8-pinfer.img -> initramfs8-pinfer
DO_INITRD=1; DRY=0
for a in "$@"; do
    case "$a" in
        --no-initramfs) DO_INITRD=0 ;;
        --dry-run) DRY=1 ;;
        *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---- preflight (read-only)
[ "$(id -u)" = 0 ] || die "run as root (sudo)"
[ -n "$R" ] || grep -q "Raspberry Pi 4" /proc/device-tree/model || die "not a Raspberry Pi 4: $(tr -d '\0' < /proc/device-tree/model)"
[ -n "$R" ] || mountpoint -q "$FW" || die "$FW is not mounted"
[ -f "$FW/config.txt" ] || die "$FW/config.txt missing"
[ -f "$HERE/$IMG_NAME" ] && [ -d "$HERE/lib/modules/$KREL" ] || die "staged tree incomplete"
echo "== verifying staged files against SHA256SUMS"
(cd "$HERE" && sha256sum --quiet -c SHA256SUMS) || die "checksum mismatch in staged tree"
[ "$(running_krel)" != "$KREL" ] || echo "note: $KREL is the running kernel (re-install)"
need_kb=$(( $(du -sk "$HERE/$IMG_NAME" "$HERE/dtbs" | awk '{s+=$1} END {print s}') + 40000 ))
free_kb=$(df -Pk "$FW" | awk 'NR==2 {print $4}')
[ "$free_kb" -gt "$need_kb" ] || die "not enough space on $FW: ${free_kb} KB free, need ~${need_kb} KB"
stock_sum_before=$(sha256sum "$FW/kernel8.img" "$FW/config.txt" "$FW/cmdline.txt")
TS=$(date +%Y%m%d-%H%M%S)
echo "== installing $KREL (variant $VARIANT, commit $COMMIT) into $DEST"

# ---- 1. modules (new directory only)
run mkdir -p "$R/lib/modules" "$R/boot"
[ -e "$R/lib/modules/$KREL" ] && run mv "$R/lib/modules/$KREL" "$R/lib/modules/$KREL.old-$TS"
run cp -a "$HERE/lib/modules/$KREL" "$R/lib/modules/$KREL"
run depmod -b "${R:-/}" -a "$KREL"
run cp "$HERE/config-$KREL" "$HERE/System.map-$KREL" "$R/boot/"

# ---- 2. boot files under the os_prefix directory
[ -e "$DEST" ] && run mv "$DEST" "$DEST.old-$TS"
run mkdir -p "$DEST/overlays"
run cp "$HERE/$IMG_NAME" "$DEST/$IMG_NAME"
run cp "$HERE"/dtbs/*.dtb "$DEST/"
# README present => firmware loads overlays from $DEST/overlays (os_prefix rule), so the
# overlays match this kernel's DTBs; stock overlays/ stays as it is.
run cp "$HERE"/dtbs/overlays/* "$DEST/overlays/"
# Kernel cmdline used ONLY by this kernel: stock cmdline + panic=10, so a kernel panic
# reboots (and the one-shot tryboot flag is gone -> stock kernel comes back by itself).
cmd="$(tr -d '\n' < "$FW/cmdline.txt")"
[[ " $cmd " == *" panic="* ]] || cmd="$cmd panic=10"
if [ "$DRY" = 1 ]; then echo "+ write $DEST/cmdline.txt: $cmd"; else printf '%s\n' "$cmd" > "$DEST/cmdline.txt"; fi

# ---- 3. initramfs (optional; auto_initramfs=1 picks up $DEST/$INITRD by name)
[ -z "$R" ] || DO_INITRD=0
if [ "$DO_INITRD" = 1 ]; then
    if [ "$DRY" = 1 ]; then echo "+ mkinitramfs -o $DEST/$INITRD $KREL"
    elif mkinitramfs -o "$DEST/$INITRD.tmp" "$KREL"; then mv "$DEST/$INITRD.tmp" "$DEST/$INITRD"
    else rm -f "$DEST/$INITRD.tmp"; echo "WARNING: mkinitramfs failed; booting without initramfs (root drivers are built in)"; fi
fi

# ---- 4. tryboot.txt = config.txt + our kernel selection
if [ "$DRY" = 1 ]; then echo "+ write $FW/tryboot.txt (config.txt + os_prefix=$PREFIX_DIR/ kernel=$IMG_NAME)"
else
    [ -e "$FW/tryboot.txt" ] && cp "$FW/tryboot.txt" "$FW/tryboot.txt.old-$TS"
    { cat "$FW/config.txt"
      printf '\n# pinfer-kernel BEGIN (tryboot only; written %s by install_tryboot.sh)\n[all]\n' "$TS"
      printf 'os_prefix=%s/\nkernel=%s\n# pinfer-kernel END\n' "$PREFIX_DIR" "$IMG_NAME"
    } > "$FW/tryboot.txt.tmp"
    mv "$FW/tryboot.txt.tmp" "$FW/tryboot.txt"
fi
sync

# ---- 5. post-checks
[ "$(sha256sum "$FW/kernel8.img" "$FW/config.txt" "$FW/cmdline.txt")" = "$stock_sum_before" ] \
    || die "stock boot files changed unexpectedly (this must not happen)"
if [ "$DRY" = 0 ]; then
    for f in "$DEST/$IMG_NAME" "$DEST/bcm2711-rpi-4-b.dtb" "$DEST/overlays/README" "$DEST/cmdline.txt" "$FW/tryboot.txt"; do
        [ -s "$f" ] || die "missing after install: $f"
    done
    echo "== installed. files:"; ls -l "$DEST" | head -20; echo "   overlays: $(ls "$DEST/overlays" | wc -l)"
    echo "== tryboot.txt tail:"; tail -5 "$FW/tryboot.txt"
    echo "== pinfer cmdline:"; cat "$DEST/cmdline.txt"
fi
cat <<EOF

NEXT (one-time test boot of $KREL):
    sudo reboot '0 tryboot'
then check:   uname -r   (expect $KREL)
              od -An -tu4 --endian=big /proc/device-tree/chosen/bootloader/tryboot   (expect 1)
If it does not come back within ~3 minutes: power-cycle -> stock kernel.
Keep it:      sudo ./make_permanent.sh   (only while running $KREL)
Undo:         sudo ./rollback.sh [--purge]
Split-core:   sudo ./splitcore_cmdline.sh on   (edits $DEST/cmdline.txt only)
EOF
