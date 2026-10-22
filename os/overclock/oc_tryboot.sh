#!/bin/bash
# oc_tryboot.sh -- run ON THE PI as root. A ONE-TIME overclock test boot through the firmware's
# tryboot mechanism, so config.txt is never edited (GitHub issue #10, owner-approved limits):
#
#   sudo bash os/overclock/oc_tryboot.sh arm 2000 6   # tryboot.txt = config.txt + [all] arm_freq / over_voltage
#   sudo reboot '0 tryboot'                            # ONE boot with tryboot.txt
#   sudo bash os/overclock/oc_tryboot.sh status        # clock, volts, temperature, throttle flags, tryboot flag
#   sudo bash os/overclock/oc_tryboot.sh restore       # put the previous tryboot.txt back, remove cmdline-oc.txt
#
# The firmware reads tryboot.txt instead of config.txt only on `reboot '0 tryboot'`; the flag is
# one-shot, so the next reboot or any power cycle boots the normal 1.5 GHz configuration. The test
# boot also reads cmdline-oc.txt (= cmdline.txt + panic=10): a kernel panic from an unstable clock
# reboots by itself, and that reboot is a normal one. A hard freeze still needs a power cycle.
#
# Limits approved by the owner (2026-10-22): arm_freq <= 2000, over_voltage <= 6, never force_turbo
# (force_turbo=1 with over_voltage > 0 sets the permanent warranty bit). The script refuses more.
set -euo pipefail

FW=/boot/firmware
BAK=$FW/tryboot.txt.pre-oc
MAX_FREQ=2000
MAX_OV=6
die() { echo "ERROR: $*" >&2; exit 1; }

status() {
    echo "model:    $(tr -d '\0' < /proc/device-tree/model)"
    echo "revision: $(awk '/^Revision/ {print $3}' /proc/cpuinfo)   kernel: $(uname -r)"
    echo "tryboot:  $(od -An -tu4 --endian=big /proc/device-tree/chosen/bootloader/tryboot | tr -d ' ')"
    echo "config:   arm_freq=$(vcgencmd get_config arm_freq | cut -d= -f2)" \
         "over_voltage=$(vcgencmd get_config over_voltage | cut -d= -f2)" \
         "force_turbo=$(vcgencmd get_config force_turbo | cut -d= -f2)"
    echo "now:      $(vcgencmd measure_clock arm) $(vcgencmd measure_volts core)" \
         "$(vcgencmd measure_temp) $(vcgencmd get_throttled)"
    echo "cpufreq:  max $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz," \
         "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "cmdline:  $(cat /proc/cmdline)"
}

preflight() {
    [ "$(id -u)" = 0 ] || die "run as root (sudo)"
    grep -q "Raspberry Pi 4" /proc/device-tree/model || die "not a Raspberry Pi 4"
    mountpoint -q "$FW" || die "$FW is not mounted"
    [ -f "$FW/config.txt" ] && [ -f "$FW/cmdline.txt" ] || die "config.txt / cmdline.txt missing"
}

arm() {
    local freq=$1 ov=$2
    [[ "$freq" =~ ^[0-9]+$ && "$ov" =~ ^[0-9]+$ ]] || die "usage: arm <arm_freq MHz> <over_voltage>"
    [ "$freq" -le "$MAX_FREQ" ] || die "arm_freq $freq is above the approved $MAX_FREQ"
    [ "$ov" -le "$MAX_OV" ] || die "over_voltage $ov is above the approved $MAX_OV"
    preflight
    ! grep -qiE '^\s*force_turbo\s*=\s*1' "$FW/config.txt" || die "config.txt sets force_turbo=1; refusing"
    local before; before=$(sha256sum "$FW/config.txt" "$FW/cmdline.txt" "$FW/kernel8.img")
    if [ -f "$BAK" ] || [ -f "$BAK.none" ]; then
        echo "backup already present (an earlier arm); keeping it"
    elif [ -f "$FW/tryboot.txt" ]; then
        cp -p "$FW/tryboot.txt" "$BAK"
        echo "backed up tryboot.txt -> $BAK"
    else
        touch "$BAK.none"
        echo "no tryboot.txt before; marker $BAK.none"
    fi
    local cmd; cmd=$(tr -d '\n' < "$FW/cmdline.txt")
    [[ "$cmd" == *panic=* ]] || cmd="$cmd panic=10"
    printf '%s\n' "$cmd" > "$FW/cmdline-oc.txt"
    {
        cat "$FW/config.txt"
        printf '\n# --- pinfer overclock test (os/overclock/oc_tryboot.sh, %s) ---\n' "$(date -Iseconds)"
        printf '[all]\narm_freq=%s\nover_voltage=%s\ncmdline=cmdline-oc.txt\n' "$freq" "$ov"
    } > "$FW/tryboot.txt.new"
    mv "$FW/tryboot.txt.new" "$FW/tryboot.txt"
    sync
    [ "$before" = "$(sha256sum "$FW/config.txt" "$FW/cmdline.txt" "$FW/kernel8.img")" ] \
        || die "a stock boot file changed; investigate before rebooting"
    echo "== tryboot.txt vs config.txt"
    diff "$FW/config.txt" "$FW/tryboot.txt" || true
    echo "== cmdline-oc.txt"; cat "$FW/cmdline-oc.txt"
    echo "armed. Next: sudo reboot '0 tryboot'"
}

restore() {
    preflight
    if [ -f "$BAK" ]; then
        cp -p "$BAK" "$FW/tryboot.txt" && rm -f "$BAK"
        echo "tryboot.txt restored from $BAK"
    elif [ -f "$BAK.none" ]; then
        rm -f "$FW/tryboot.txt" "$BAK.none"
        echo "tryboot.txt removed (there was none before)"
    else
        echo "no backup found; tryboot.txt left as is"
    fi
    rm -f "$FW/cmdline-oc.txt"
    sync
    ! grep -q "pinfer overclock test" "$FW/tryboot.txt" 2>/dev/null || die "tryboot.txt still holds the overclock block"
    echo "restored."
}

case "${1:-}" in
    arm) arm "${2:-}" "${3:-}" ;;
    restore) restore ;;
    status) status ;;
    *) echo "usage: $0 arm <arm_freq> <over_voltage> | restore | status" >&2; exit 2 ;;
esac
