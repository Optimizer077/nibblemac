#!/bin/bash
# make_permanent.sh -- run ON THE PI (as root) only after a successful tryboot of the
# pinfer kernel. Adds os_prefix=/kernel= to config.txt so every normal boot uses it.
# config.txt is backed up first; kernel8.img and the stock files stay in place, so
# rollback.sh (or deleting the marked block in config.txt from any PC) restores stock.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/BUILDINFO"
# Test hook: PINFER_TEST_ROOT=<dir> runs against a fake root (no hardware checks, no
# mkinitramfs); PINFER_TEST_KREL fakes `uname -r`. Never set these on the Pi.
R="${PINFER_TEST_ROOT:-}"
running_krel() { echo "${PINFER_TEST_KREL:-$(uname -r)}"; }
FW=$R/boot/firmware
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (sudo)"
[ "$(running_krel)" = "$KREL" ] || die "running $(running_krel), not $KREL: tryboot the pinfer kernel first (sudo reboot '0 tryboot')"
[ -s "$FW/$PREFIX_DIR/$IMG_NAME" ] && [ -s "$FW/$PREFIX_DIR/bcm2711-rpi-4-b.dtb" ] || die "$FW/$PREFIX_DIR incomplete"
tb=$(od -An -tu4 --endian=big $R/proc/device-tree/chosen/bootloader/tryboot 2>/dev/null | tr -d ' ' || echo "?")
[ "$tb" = 1 ] || echo "note: current boot was not a tryboot (tryboot=$tb); continuing because uname -r matches"
if grep -q '^# pinfer-kernel BEGIN' "$FW/config.txt"; then
    echo "config.txt already has a pinfer-kernel block:"; sed -n '/^# pinfer-kernel BEGIN/,/^# pinfer-kernel END/p' "$FW/config.txt"; exit 0
fi
TS=$(date +%Y%m%d-%H%M%S)
cp "$FW/config.txt" "$FW/config.txt.pre-pinfer-$TS"
{ cat "$FW/config.txt"
  printf '\n# pinfer-kernel BEGIN (permanent; written %s by make_permanent.sh; undo: rollback.sh)\n[all]\n' "$TS"
  printf 'os_prefix=%s/\nkernel=%s\n# pinfer-kernel END\n' "$PREFIX_DIR" "$IMG_NAME"
} > "$FW/config.txt.tmp"
mv "$FW/config.txt.tmp" "$FW/config.txt"
sync
echo "config.txt now boots $KREL by default. Backup: $FW/config.txt.pre-pinfer-$TS"
tail -5 "$FW/config.txt"
echo "Reboot normally to confirm; undo with: sudo ./rollback.sh"
