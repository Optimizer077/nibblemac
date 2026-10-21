#!/bin/sh
# Check a built libpinfer: exported symbols, NEEDED libraries, glibc symbol versions.
#   sh engine/pkg/check_lib.sh <lib dir> <binutils prefix: "" or aarch64-linux-gnu-> <max glibc>
# Fails (exit 1) if
#   - the .so or the .a exposes any global symbol not named pinfer_*,
#   - the .so needs a library other than libc / libm / the dynamic loader,
#   - the .so needs a GLIBC_x.y symbol version newer than <max glibc> (e.g. 2.41 for the Pi).
set -eu
dir=$1
pfx=$2
max=${3:-99.99}
so=$dir/libpinfer.so
a=$dir/libpinfer.a
fail=0

echo "== $so"
syms=$(${pfx}nm -D --defined-only "$so" | awk '{print $3}')
echo "exported: $(echo $syms)"
bad=$(echo "$syms" | grep -v '^pinfer_' || true)
if [ -n "$bad" ]; then echo "FAIL: unexpected exports: $(echo $bad)"; fail=1; fi

needed=$(${pfx}objdump -p "$so" | awk '/NEEDED/ {print $2}')
echo "NEEDED: $(echo $needed)"
for lib in $needed; do
    case $lib in
        libc.so.6|libm.so.6|ld-linux*) ;;
        *) echo "FAIL: unexpected dependency $lib"; fail=1 ;;
    esac
done

vers=$(${pfx}objdump -T "$so" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' \
       | sort -u -t. -k1,1n -k2,2n -k3,3n)
newest=$(echo "$vers" | tail -n 1)
echo "glibc symbol versions: $(echo $vers) -> newest $newest (limit $max)"
${pfx}objdump -T "$so" | grep "GLIBC_$newest)" | awk '{print "  " $NF " needs GLIBC_'"$newest"'"}'
over=$(printf '%s\n%s\n' "$newest" "$max" | sort -t. -k1,1n -k2,2n | tail -n 1)
if [ "$over" != "$max" ] && [ "$newest" != "$max" ]; then
    echo "FAIL: needs GLIBC_$newest > $max"; fail=1
fi

echo "== $a"
asyms=$(${pfx}nm -g --defined-only "$a" 2>/dev/null | awk 'NF==3 {print $3}')
echo "global: $(echo $asyms)"
abad=$(echo "$asyms" | grep -v '^pinfer_' || true)
if [ -n "$abad" ]; then echo "FAIL: unexpected globals: $(echo $abad)"; fail=1; fi

if [ $fail -eq 0 ]; then echo "OK"; else echo "FAILED"; fi
exit $fail
