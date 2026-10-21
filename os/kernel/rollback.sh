#!/bin/bash
# rollback.sh -- run ON THE PI (as root). Returns the Pi to booting the stock kernel.
#   - removes the "# pinfer-kernel BEGIN..END" block from config.txt (backup kept)
#   - removes /boot/firmware/tryboot.txt (backup kept)
#   --purge additionally deletes /boot/firmware/<PREFIX_DIR>/, /lib/modules/<krel>/ and
#     /boot/{config,System.map}-<krel> (only run --purge while NOT running <krel>)
# kernel8.img, the stock DTBs/overlays and cmdline.txt are never touched by any pinfer
# script, so nothing needs restoring there.
#
# No-SSH recovery (Pi does not boot the pinfer kernel after make_permanent.sh): put the
# SSD in the PC, open config.txt on the FAT boot partition, delete the lines between
# "# pinfer-kernel BEGIN" and "# pinfer-kernel END" (or copy back config.txt.pre-pinfer-*).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/BUILDINFO"
# Test hook: PINFER_TEST_ROOT=<dir> runs against a fake root (no hardware checks, no
# mkinitramfs); PINFER_TEST_KREL fakes `uname -r`. Never set these on the Pi.
R="${PINFER_TEST_ROOT:-}"
running_krel() { echo "${PINFER_TEST_KREL:-$(uname -r)}"; }
FW=$R/boot/firmware
PURGE=0; [ "${1:-}" = "--purge" ] && PURGE=1
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
TS=$(date +%Y%m%d-%H%M%S)

if grep -q '^# pinfer-kernel BEGIN' "$FW/config.txt"; then
    cp "$FW/config.txt" "$FW/config.txt.rollback-$TS"
    # Drop the BEGIN..END block and the one blank line make_permanent.sh put before it, so
    # config.txt becomes byte-identical to the pre-pinfer version.
    awk '/^# pinfer-kernel BEGIN/ {skip=1; held=0; next}
         skip { if (/^# pinfer-kernel END/) skip=0; next }
         { if (held) { print ""; held=0 } if ($0 == "") { held=1; next } print }
         END { if (held) print "" }' "$FW/config.txt" > "$FW/config.txt.tmp"
    mv "$FW/config.txt.tmp" "$FW/config.txt"
    echo "removed pinfer block from config.txt (backup config.txt.rollback-$TS)"
else
    echo "config.txt has no pinfer block (already stock)"
fi
if [ -e "$FW/tryboot.txt" ]; then
    mv "$FW/tryboot.txt" "$FW/tryboot.txt.rollback-$TS"; echo "moved tryboot.txt -> tryboot.txt.rollback-$TS"
fi
if [ "$PURGE" = 1 ]; then
    [ "$(running_krel)" != "$KREL" ] || { echo "refusing --purge while running $KREL; reboot to stock first" >&2; exit 1; }
    rm -rf "${FW:?}/${PREFIX_DIR:?}" "$R/lib/modules/${KREL:?}" "$R/boot/config-$KREL" "$R/boot/System.map-$KREL"
    echo "purged $FW/$PREFIX_DIR, /lib/modules/$KREL"
fi
sync
echo "next normal reboot boots: $(grep -E '^kernel=' "$FW/config.txt" || echo 'kernel8.img (default)')"
