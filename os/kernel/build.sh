#!/usr/bin/env bash
# pinfer kernel build: cross-compile the Raspberry Pi kernel for the Pi 4 (BCM2711)
# with the split-core + huge-page config, and stage an install tree + tarball.
#
# Usage (from Git Bash, project root):
#   os/kernel/build.sh                 # 4 KB-page kernel (the main deliverable)
#   VARIANT=16k os/kernel/build.sh     # HIGH-RISK experiment: 16 KB pages
#   STEPS="config" os/kernel/build.sh  # only fetch source + make/verify the config
#
# From Git Bash the script (1) makes sure the stock config pulled from the Pi is present
# (pulls it read-only via harness/pi.py if not) and (2) re-runs itself inside WSL Ubuntu,
# where all compiling happens (source in /root/src, never on /mnt).
#
# Outputs
#   logs/kernel/build-<variant>-<ts>.log      full log of the WSL part
#   logs/kernel/diffconfig-<variant>.txt      scripts/diffconfig stock -> ours
#   logs/kernel/config-state-<variant>.txt    scripts/config --state for every fragment symbol
#   build/kernel/<kernelrelease>/             install tree (see KERNEL.md)
#   build/kernel/pinfer-kernel-<kernelrelease>.tar.gz (+ .sha256)
set -euo pipefail

VARIANT="${VARIANT:-4k}"
STEPS="${STEPS:-config build stage}"
REPO_URL="https://github.com/raspberrypi/linux"
# Commit of the RUNNING kernel: /usr/share/doc/linux-image-6.18.50+rpt-rpi-v8/changelog.Debian.gz
# on the Pi says "Linux commit: cff533aec2..." for 1:6.18.50-1+rpt1. It is an ancestor of
# branch rpi-6.18.y (GitHub compare: behind_by 0).
COMMIT="cff533aec2fa601846766b32ff57204e0a61bed7"
STOCK_REL="6.18.50+rpt-rpi-v8"
STOCK_SHA256="545260efb824b8999548ce483bce856bb60e72718ced820fbd951c025689c746"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
STOCK_CFG_REL="build/kernel/stock/config-$STOCK_REL"

# ---------------------------------------------------------------- Git Bash (Windows) side
case "$(uname -s)" in
MINGW*|MSYS*|CYGWIN*)
    cd "$PROJ"
    if [ ! -f "$STOCK_CFG_REL" ]; then
        echo "[build.sh] pulling /boot/config-$STOCK_REL from the Pi (read-only)"
        mkdir -p build/kernel/stock logs/kernel
        tmp="build/kernel/stock/.pull.$$"
        python harness/pi.py run "gzip -9c /boot/config-$STOCK_REL | base64 -w0; echo" > "$tmp"
        tail -n 1 "$tmp" | tr -d '\r' | base64 -d | gunzip > "$STOCK_CFG_REL"
        rm -f "$tmp"
    fi
    got="$(sha256sum "$STOCK_CFG_REL" | cut -d' ' -f1)"
    [ "$got" = "$STOCK_SHA256" ] || { echo "stock config sha256 mismatch: $got" >&2; exit 1; }
    drive="$(pwd -W | cut -c1 | tr 'A-Z' 'a-z')"
    wproj="/mnt/$drive$(pwd -W | cut -c3-)"
    exec env MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- env VARIANT="$VARIANT" STEPS="$STEPS" \
        bash "$wproj/os/kernel/build.sh"
    ;;
esac

# ---------------------------------------------------------------- WSL (Linux) side
SRC=/root/src/linux-rpi
OBJ=/root/src/kbuild-pinfer-$VARIANT
JOBS="$(nproc)"
MK=(make -C "$SRC" O="$OBJ" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- LOCALVERSION= -j"$JOBS")
FRAGS=("$SCRIPT_DIR/pinfer.config")
case "$VARIANT" in
    4k)  IMG_NAME=kernel8-pinfer.img;    PREFIX_DIR=pinfer ;;
    16k) # Refused on evidence, see KERNEL.md section 9: the Pi 4's Cortex-A72 (MIDR 0x410fd083,
         # r0p3) does not implement the 16 KB translation granule (A72 TRM Table 4-35,
         # ID_AA64MMFR0_EL1.TGran16 = 0), so a 16K kernel spins in __no_granule_support.
         echo "VARIANT=16k refused: Cortex-A72 has no 16 KB granule (see os/kernel/KERNEL.md section 9)" >&2
         exit 1 ;;
    *)   echo "unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac

mkdir -p "$PROJ/logs/kernel"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJ/logs/kernel/build-$VARIANT-$TS.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [$(date +%T)] $*"; }
echo "pinfer kernel build  variant=$VARIANT  steps='$STEPS'  log=$LOG"
echo "host: $(uname -a)"
echo "gcc:  $(aarch64-linux-gnu-gcc --version | head -1)"
echo "jobs: $JOBS"

# ---- 1. source at the exact commit of the running kernel (shallow, single commit)
step "source: $REPO_URL @ $COMMIT -> $SRC"
if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" remote add origin "$REPO_URL"
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    t0=$SECONDS
    git -C "$SRC" fetch --depth 1 origin "$COMMIT"
    git -C "$SRC" checkout -q --detach FETCH_HEAD
    echo "fetch took $((SECONDS - t0)) s"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "HEAD != $COMMIT" >&2; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "source tree is dirty" >&2; exit 1; }
git -C "$SRC" log -1 --format='commit %H%ncommitter-date %cI%nsubject %s'
echo "Makefile version: $(make -s -C "$SRC" kernelversion)"

# ---- 2. config: stock running config + fragment(s), olddefconfig, verify
do_config() {
    step "config: stock $STOCK_REL + ${FRAGS[*]##*/}"
    local stock="$PROJ/$STOCK_CFG_REL"
    [ "$(sha256sum "$stock" | cut -d' ' -f1)" = "$STOCK_SHA256" ] || { echo "stock config hash mismatch" >&2; exit 1; }
    mkdir -p "$OBJ"
    cp "$stock" "$OBJ/.config"
    # -m: merge only; we run olddefconfig ourselves so its output is in the log.
    "$SRC/scripts/kconfig/merge_config.sh" -m -O "$OBJ" "$OBJ/.config" "${FRAGS[@]}"
    "${MK[@]}" olddefconfig

    # Expected value per symbol; a later fragment overrides an earlier one.
    declare -A want=()
    local f line sym val
    for f in "${FRAGS[@]}"; do
        while IFS= read -r line; do
            if [[ $line =~ ^(CONFIG_[A-Za-z0-9_]+)=(.*)$ ]]; then
                want[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"
            elif [[ $line =~ ^#\ (CONFIG_[A-Za-z0-9_]+)\ is\ not\ set$ ]]; then
                want[${BASH_REMATCH[1]}]="n"
            fi
        done < "$f"
    done
    local state="$PROJ/logs/kernel/config-state-$VARIANT.txt" bad=0 got
    : > "$state"
    for sym in $(printf '%s\n' "${!want[@]}" | sort); do
        got="$("$SRC/scripts/config" --file "$OBJ/.config" --state "${sym#CONFIG_}")"
        # --state prints y/m/n, the value of strings/ints, "undef" if absent.
        val="${want[$sym]}"; val="${val%\"}"; val="${val#\"}"
        if [ "$val" = "n" ] && [ "$got" = "undef" ]; then got="n"; fi
        if [ "$got" = "$val" ]; then r=OK; else r=MISMATCH; bad=1; fi
        printf '%-40s want=%-12s got=%-12s %s\n' "$sym" "$val" "$got" "$r" | tee -a "$state"
    done
    [ "$bad" = 0 ] || { echo "config verification FAILED (see $state)" >&2; exit 1; }
    "$SRC/scripts/diffconfig" "$stock" "$OBJ/.config" > "$PROJ/logs/kernel/diffconfig-$VARIANT.txt"
    echo "diffconfig: $(wc -l < "$PROJ/logs/kernel/diffconfig-$VARIANT.txt") changed symbols -> logs/kernel/diffconfig-$VARIANT.txt"
    cp "$OBJ/.config" "$PROJ/logs/kernel/config-pinfer-$VARIANT"
}

# ---- 3. build Image, modules, DTBs (+ overlays)
do_build() {
    step "build: Image modules dtbs (-j$JOBS)"
    local t0=$SECONDS
    "${MK[@]}" Image modules dtbs
    echo "BUILD_SECONDS=$((SECONDS - t0))"
    ls -l "$OBJ/arch/arm64/boot/Image"
}

# ---- 4. stage install tree + tarball
do_stage() {
    local rel; rel="$("${MK[@]}" -s kernelrelease)"
    step "stage: kernelrelease=$rel"
    local sroot=/root/src/stage-pinfer-$VARIANT
    local st="$sroot/$rel"
    rm -rf "$sroot"; mkdir -p "$st/dtbs/overlays"
    cp "$OBJ/arch/arm64/boot/Image" "$st/$IMG_NAME"
    cp "$OBJ/.config" "$st/config-$rel"
    cp "$OBJ/System.map" "$st/System.map-$rel"
    cp "$OBJ"/arch/arm64/boot/dts/broadcom/bcm2711-*.dtb "$st/dtbs/"
    cp "$OBJ"/arch/arm64/boot/dts/overlays/*.dtbo "$st/dtbs/overlays/"
    # overlay_map.dtb maps generic overlay names to per-SoC variants (vc4-kms-v3d ->
    # vc4-kms-v3d-pi4 on a Pi 4); hat_map.dtb maps HAT EEPROMs to overlays. Stock ships both.
    cp "$OBJ"/arch/arm64/boot/dts/overlays/overlay_map.dtb "$OBJ"/arch/arm64/boot/dts/overlays/hat_map.dtb "$st/dtbs/overlays/"
    cp "$SRC/arch/arm/boot/dts/overlays/README" "$st/dtbs/overlays/README"
    "${MK[@]}" INSTALL_MOD_PATH="$st" INSTALL_MOD_STRIP=1 modules_install > /dev/null
    rm -f "$st/lib/modules/$rel/build" "$st/lib/modules/$rel/source"   # point into WSL, useless on the Pi
    cp "$SCRIPT_DIR"/install_tryboot.sh "$SCRIPT_DIR"/make_permanent.sh "$SCRIPT_DIR"/rollback.sh \
       "$SCRIPT_DIR"/splitcore_cmdline.sh "$SCRIPT_DIR"/KERNEL.md "$st/"
    # BUILDINFO is sourced by the Pi-side scripts: every value must be shell-quoted.
    {   printf 'KREL=%q\nVARIANT=%q\nIMG_NAME=%q\nPREFIX_DIR=%q\n' "$rel" "$VARIANT" "$IMG_NAME" "$PREFIX_DIR"
        printf 'SOURCE=%q\nCOMMIT=%q\nSTOCK_KREL=%q\nSTOCK_CONFIG_SHA256=%q\n' "$REPO_URL" "$COMMIT" "$STOCK_REL" "$STOCK_SHA256"
        printf 'BUILT=%q\nGCC=%q\n' "$(date -Iseconds)" "$(aarch64-linux-gnu-gcc --version | head -1)"
    } > "$st/BUILDINFO"
    bash -n "$st/BUILDINFO" && (set -u; . "$st/BUILDINFO"; [ "$KREL" = "$rel" ]) || { echo "BUILDINFO not sourceable" >&2; exit 1; }
    (cd "$st" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    local tarname="pinfer-kernel-$rel.tar.gz"
    tar -C "$sroot" --owner=0 --group=0 -czf "$sroot/$tarname" "$rel"
    (cd "$sroot" && sha256sum "$tarname" > "$tarname.sha256")

    local out="$PROJ/build/kernel"
    mkdir -p "$out"
    rm -rf "${out:?}/$rel"
    mkdir -p "$out/$rel"
    # The TARBALL (built on ext4) is the install artifact. build/kernel/<rel>/ is a mirror for
    # inspection on Windows. /mnt/<drive> (NTFS) is case-insensitive and enabling per-directory
    # case sensitivity needs admin, but netfilter has module pairs that differ only in case
    # (xt_DSCP.ko.xz / xt_dscp.ko.xz, xt_HL, xt_RATEEST, xt_TCPMSS). The mirror therefore
    # skips the second file of each colliding pair and lists them; everything else is verified.
    local coll="$sroot/case-collisions.txt"
    (cd "$st" && find . -type f | sort | awk '{k=tolower($0); if (k in seen) print; else seen[k]=1}') > "$coll"
    tar -C "$st" -cf - --exclude-from="$coll" . | tar -C "$out/$rel" -xf -
    { echo "Not mirrored here (name differs only in case from another file; NTFS is case-insensitive)."
      echo "They ARE in build/kernel/$tarname, the artifact to push and install:"; cat "$coll"; } \
        > "$out/$rel/CASE_COLLISIONS_NOT_MIRRORED.txt"
    (cd "$out/$rel" && grep -v -F -f <(sed 's|^\./||' "$coll") SHA256SUMS | sha256sum --quiet -c -) \
        && echo "mirror verified against SHA256SUMS ($(wc -l < "$coll") case-colliding files only in the tarball)" \
        || { echo "mirror under build/kernel does NOT match SHA256SUMS" >&2; exit 1; }
    cp "$sroot/$tarname" "$sroot/$tarname.sha256" "$out/"
    echo "staged: build/kernel/$rel ($(du -sh "$st" | cut -f1)), image $(du -h "$st/$IMG_NAME" | cut -f1), modules $(du -sh "$st/lib/modules/$rel" | cut -f1), $(find "$st/lib/modules/$rel" -name '*.ko*' | wc -l) .ko files"
    echo "tarball: build/kernel/$tarname ($(du -h "$sroot/$tarname" | cut -f1)) sha256 $(cut -d' ' -f1 "$sroot/$tarname.sha256")"
}

for s in $STEPS; do
    case "$s" in
        config) do_config ;;
        build)  do_build ;;
        stage)  do_stage ;;
        *) echo "unknown step $s" >&2; exit 1 ;;
    esac
done
step "done ($SECONDS s total)"
