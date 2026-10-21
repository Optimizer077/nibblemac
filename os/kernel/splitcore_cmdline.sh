#!/bin/bash
# splitcore_cmdline.sh -- run ON THE PI (as root). Separate, reversible step after the
# plain pinfer kernel has passed a tryboot. Edits ONLY /boot/firmware/<PREFIX_DIR>/cmdline.txt,
# which only the pinfer kernel reads (firmware os_prefix); the stock cmdline.txt is untouched.
#
#   sudo ./splitcore_cmdline.sh on  [--hugepages N]   # add split-core (+ 2 MB hugetlb pages)
#   sudo ./splitcore_cmdline.sh off                    # remove everything this script adds
#   ./splitcore_cmdline.sh show
#
# "on" adds (syntax checked against Documentation/admin-guide/kernel-parameters.txt @ cff533ae):
#   isolcpus=nohz,domain,managed_irq,1-3  no scheduler load balancing / unmanaged-noise on 1-3
#   nohz_full=1-3       stop the tick on 1-3 when one task runs (needs CONFIG_NO_HZ_FULL=y)
#   rcu_nocbs=1-3       RCU callbacks of 1-3 run in rcuo kthreads (implied by nohz_full; explicit)
#   irqaffinity=0       default IRQ affinity = core 0
#   transparent_hugepage=madvise   (already the Kconfig default; explicit for the record)
#   hugepagesz=2M hugepages=N      only with --hugepages N (N=0 or omitted: no reservation)
# Recommended N for a ~0.85 GB Q8 model: 480 (= 960 MiB; 0.85 GB / 2 MiB = 406 pages +
# headroom for alignment/activations). Reserved pages are taken from the page cache /
# other processes for the whole uptime; the runtime alternative needs no reboot:
#   echo 480 | sudo tee /proc/sys/vm/nr_hugepages ; grep Huge /proc/meminfo
# Then test with:  sudo reboot '0 tryboot'   (before make_permanent.sh)
#             or:  sudo reboot               (after make_permanent.sh)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/BUILDINFO"
CMDF=${PINFER_TEST_ROOT:-}/boot/firmware/$PREFIX_DIR/cmdline.txt   # PINFER_TEST_ROOT: test hook only
KEYS="isolcpus nohz_full rcu_nocbs irqaffinity transparent_hugepage hugepagesz hugepages"
[ -f "$CMDF" ] || { echo "$CMDF missing: run install_tryboot.sh first" >&2; exit 1; }
mode="${1:-show}"; shift || true
N=0
while [ $# -gt 0 ]; do
    case "$1" in --hugepages) N="$2"; shift 2 ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
done
strip() {   # drop every KEY=... we manage
    local w out=()
    for w in $(tr -d '\n' < "$CMDF"); do
        local k="${w%%=*}" keep=1 m
        for m in $KEYS; do [ "$k" = "$m" ] && keep=0; done
        [ "$keep" = 1 ] && out+=("$w")
    done
    echo "${out[*]}"
}
case "$mode" in
show) cat "$CMDF"; exit 0 ;;
on)   new="$(strip) isolcpus=nohz,domain,managed_irq,1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0 transparent_hugepage=madvise"
      if [ "$N" -gt 0 ]; then new="$new hugepagesz=2M hugepages=$N"; fi ;;
off)  new="$(strip)" ;;
*)    echo "usage: $0 on [--hugepages N] | off | show" >&2; exit 2 ;;
esac
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
cp "$CMDF" "$CMDF.bak-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$new" > "$CMDF.tmp" && mv "$CMDF.tmp" "$CMDF" && sync
echo "$CMDF:"; cat "$CMDF"
