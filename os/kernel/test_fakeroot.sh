#!/bin/bash
# test_fakeroot.sh -- run in WSL (NOT on the Pi): exercises install_tryboot.sh, splitcore_cmdline.sh,
# make_permanent.sh and rollback.sh end to end against a fake root (/root/src/pinfer-installtest),
# using the Pi's real config.txt/cmdline.txt (os/kernel/testdata, sha256-checked against the Pi
# 2026-10-21) and the real tarball. Every check must print PASS; any FAIL exits 1.
# Usage (Git Bash, project root):
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -c "cd /mnt/e/Personal_tries/Raspberry_Pi && \
#       bash os/kernel/test_fakeroot.sh \$PWD/os/kernel/testdata \
#       /root/src/stage-pinfer-4k/pinfer-kernel-6.18.50-pinfer.tar.gz 2>&1" > logs/kernel/script-test-fakeroot.log
# (the fixture dir must be an absolute path: the script cd's into the unpacked tree)
set -euo pipefail
FWSRC=$1; TARBALL=$2; KREL=6.18.50-pinfer
T=/root/src/pinfer-installtest; rm -rf $T; mkdir -p $T/boot/firmware/overlays $T/proc/device-tree/chosen/bootloader $T/work
cp "$FWSRC/config.txt" "$FWSRC/cmdline.txt" $T/boot/firmware/
head -c 1000000 /dev/urandom > $T/boot/firmware/kernel8.img; echo stockdtb > $T/boot/firmware/bcm2711-rpi-4-b.dtb; echo r > $T/boot/firmware/overlays/README
(cd $T/boot/firmware && find . -type f | sort | xargs sha256sum) > $T/stock.sums
tar xzf "$TARBALL" -C $T/work; cd $T/work/$KREL
export PINFER_TEST_ROOT=$T
pass() { echo "PASS: $*"; }; fail() { echo "FAIL: $*"; exit 1; }
echo "## dry run"; ./install_tryboot.sh --dry-run > $T/dry.log || fail "dry-run exit"; if [ ! -e $T/boot/firmware/tryboot.txt ] && [ ! -e $T/boot/firmware/pinfer ]; then pass "dry-run wrote nothing"; else fail "dry-run wrote files"; fi
echo "## install"; ./install_tryboot.sh | tail -25
if (cd $T/boot/firmware && sha256sum -c --quiet $T/stock.sums); then pass "stock files unchanged after install"; else fail "stock files unchanged after install"; fi
if [ -s $T/lib/modules/$KREL/modules.dep ]; then pass "depmod produced modules.dep ($(wc -l < $T/lib/modules/$KREL/modules.dep) lines)"; else fail "depmod produced modules.dep ($(wc -l < $T/lib/modules/$KREL/modules.dep) lines)"; fi
if diff <(head -n "$(wc -l < $T/boot/firmware/config.txt)" $T/boot/firmware/tryboot.txt) $T/boot/firmware/config.txt; then pass "tryboot.txt starts with config.txt verbatim"; else fail "tryboot.txt starts with config.txt verbatim"; fi
if grep -qx 'os_prefix=pinfer/' $T/boot/firmware/tryboot.txt && grep -qx 'kernel=kernel8-pinfer.img' $T/boot/firmware/tryboot.txt; then pass "tryboot.txt selects pinfer"; else fail "tryboot.txt selects pinfer"; fi
echo "## make_permanent while running stock -> must refuse"
if PINFER_TEST_KREL=6.18.50+rpt-rpi-v8 ./make_permanent.sh; then fail "should refuse"; else pass "refused"; fi
echo "## splitcore"
./splitcore_cmdline.sh on --hugepages 480 | tail -1
if grep -q 'isolcpus=nohz,domain,managed_irq,1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0 transparent_hugepage=madvise hugepagesz=2M hugepages=480' $T/boot/firmware/pinfer/cmdline.txt; then pass "split-core flags"; else fail "split-core flags"; fi
if ./splitcore_cmdline.sh on > /dev/null; [ "$(grep -o 'isolcpus=' $T/boot/firmware/pinfer/cmdline.txt | wc -l)" = 1 ] && ! grep -q hugepages= $T/boot/firmware/pinfer/cmdline.txt; then pass "re-run idempotent, hugepages removed"; else fail "re-run idempotent, hugepages removed"; fi
./splitcore_cmdline.sh off > /dev/null
if diff <(tr -d '\n' < $T/boot/firmware/cmdline.txt; echo " panic=10") $T/boot/firmware/pinfer/cmdline.txt; then pass "off restores stock+panic=10"; else fail "off restores stock+panic=10"; fi
if (cd $T/boot/firmware && sha256sum -c --quiet $T/stock.sums); then pass "stock cmdline untouched by splitcore"; else fail "stock cmdline untouched by splitcore"; fi
echo "## make_permanent while running pinfer"
printf '\0\0\0\1' > $T/proc/device-tree/chosen/bootloader/tryboot
PINFER_TEST_KREL=$KREL ./make_permanent.sh | tail -4
if out=$(PINFER_TEST_KREL=$KREL ./make_permanent.sh); echo "$out" | sed -n 1p; echo "$out" | grep -q "already has"; then pass "make_permanent idempotent"; else fail "make_permanent idempotent"; fi
if [ "$(grep -c '^# pinfer-kernel BEGIN' $T/boot/firmware/config.txt)" = 1 ]; then pass "permanent block added once"; else fail "permanent block added once"; fi
echo "## rollback"
./rollback.sh
if diff $T/boot/firmware/config.txt "$FWSRC/config.txt"; then pass "config.txt byte-identical to original after rollback"; else fail "config.txt byte-identical to original after rollback"; fi
if [ ! -e $T/boot/firmware/tryboot.txt ]; then pass "tryboot.txt moved aside"; else fail "tryboot.txt moved aside"; fi
if PINFER_TEST_KREL=$KREL ./rollback.sh --purge; then fail "purge should refuse"; else pass "purge refused while running pinfer"; fi
PINFER_TEST_KREL=6.18.50+rpt-rpi-v8 ./rollback.sh --purge | tail -2
if [ ! -e $T/boot/firmware/pinfer ] && [ ! -e $T/lib/modules/$KREL ]; then pass "purged"; else fail "purged"; fi
if (cd $T/boot/firmware && sha256sum -c --quiet $T/stock.sums); then pass "stock files still unchanged"; else fail "stock files still unchanged"; fi
echo ALL-TESTS-PASSED
rm -rf "$T"
