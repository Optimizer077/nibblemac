#!/usr/bin/env bash
# Cross-compile LiteRT-LM (CPU only) for the Pi 4 inside WSL Ubuntu (root), CMake super-build:
#   bash bench/litert/build.sh [tag=v0.18.1] [jobs=4]
# Prereqs installed in WSL (see bench/BASELINES.md): openjdk-17-jre-headless, rustup
# (stable + target aarch64-unknown-linux-gnu), gcc/g++-aarch64-linux-gnu 15.2, cmake 4.2, make.
# Why a source build: every prebuilt litert-lm-api aarch64 wheel (0.15.0-0.18.1) is compiled
# with aes/sha2/dotprod/fp16/lse/rcpc/rdm and aborts at start on the Cortex-A72.
# Output: build/baselines/litert/ (static binaries + BUILD_INFO.txt).
set -euo pipefail
TAG="${1:-v0.18.1}"
JOBS="${2:-4}"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=/root/src/LiteRT-LM-"$TAG"
BLD=/root/cache/litert-build-"$TAG"
EXTRA=/root/cache/aarch64-extra
OUT="$PROJ/build/baselines/litert"
export PATH="$HOME/.cargo/bin:$PATH"
# Rust crates with C parts (tokenizers-cpp -> onig_sys, cxx) build through cc-rs, which otherwise
# looks for "aarch64-unknown-linux-gnu-gcc"; point it (and cargo's linker) at the Ubuntu cross gcc.
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
export CFLAGS_aarch64_unknown_linux_gnu="-march=armv8-a+crc -mtune=cortex-a72"
export CXXFLAGS_aarch64_unknown_linux_gnu="-march=armv8-a+crc -mtune=cortex-a72"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
# tokenizers-cpp hard-codes CC_aarch64-unknown-linux-gnu=aarch64-unknown-linux-gnu-gcc (etc.):
# provide those names as aliases of the Ubuntu cross tools, in a private dir on PATH.
XBIN=/root/cache/xbin
mkdir -p "$XBIN"
for t in gcc g++ ar; do ln -sf "/usr/bin/aarch64-linux-gnu-$t" "$XBIN/aarch64-unknown-linux-gnu-$t"; done
export PATH="$XBIN:$PATH"
# LiteRT's CMake shim requires `uv` (it creates a Python 3.13 venv); use the standalone binary.
if ! command -v uv > /dev/null; then
    UVV=0.12.0
    curl -sSL "https://github.com/astral-sh/uv/releases/download/$UVV/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar xz -C "$XBIN" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
fi
uv --version
# LiteRT (>= 2026-09-23) includes SafeInt.hpp; LiteRT's own CMakeLists fetches it, but LiteRT-LM's
# patcher replaces that file with a shim that does not. Header-only: expose it to every compile.
SAFEINT=/root/cache/safeint
if [ ! -f "$SAFEINT/SafeInt.hpp" ]; then
    rm -rf "$SAFEINT" && git init -q "$SAFEINT"
    git -C "$SAFEINT" fetch -q --depth 1 https://github.com/dcleblanc/SafeInt 72f0745e26091af6d1186fd91426377f16688ba7
    git -C "$SAFEINT" checkout -q FETCH_HEAD
fi
export CPLUS_INCLUDE_PATH="$SAFEINT${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

[ -d "$SRC" ] || git clone --quiet --depth 1 --branch "$TAG" https://github.com/google-ai-edge/LiteRT-LM "$SRC"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "LiteRT-LM $TAG @ $COMMIT"

# 1. static zlib for the target (the final link uses -lz)
if [ ! -f "$EXTRA/lib/libz.a" ]; then
    ZV=1.3.1
    mkdir -p /root/cache/zlib && cd /root/cache/zlib
    [ -f zlib-$ZV.tar.gz ] || curl -sSLO https://github.com/madler/zlib/releases/download/v$ZV/zlib-$ZV.tar.gz
    rm -rf zlib-$ZV && tar xzf zlib-$ZV.tar.gz && cd zlib-$ZV
    CC=aarch64-linux-gnu-gcc CFLAGS="-O2 -march=armv8-a+crc -mtune=cortex-a72 -fPIC" \
        ./configure --static --prefix="$EXTRA" > /dev/null
    make -j"$JOBS" > /dev/null && make install > /dev/null
fi

# 1a. source patches: restore the upstream files, then re-apply (idempotent)
git -C "$SRC" checkout -- cmake/packages/litert_lm/CMakeLists.txt cmake/modules/fetch_content.cmake \
    cmake/packages/litert/litert_patcher.cmake cmake/packages/litert/litert_target_map.cmake \
    cmake/packages/litert/litert_litert_lm_target_map.cmake

# 1f. Samsung NPU vendor is not built (toolchain: LITERT_ENABLE_SAMSUNG=OFF, GCC 15.2 ICE); drop its
#     archives from LiteRT-LM's LiteRT target maps (they are only listed there, never used by name)
sed -i '/vendors\/samsung\//d' "$SRC/cmake/packages/litert/litert_target_map.cmake" \
    "$SRC/cmake/packages/litert/litert_litert_lm_target_map.cmake"

# 1e. LiteRT patch step: link KleidiAI into LiteRT's shared C API lib (see litert_patch.cmake)
cat "$PROJ/bench/litert/litert_patch.cmake" >> "$SRC/cmake/packages/litert/litert_patcher.cmake"

# 1d. TFLite builds libfarmhash.a inside the tree; LiteRT's shared libs link -lfarmhash
mkdir -p "$EXTRA/lib"
ln -sf "$BLD/litert_lm/build/external/tensorflow/src/tflite_external-build/_deps/farmhash-build/libfarmhash.a" \
    "$EXTRA/lib/libfarmhash.a"

# 1b. add a CMake target for litert_lm_advanced_main (benchmark flags) + static link of both CLIs
PKG="$SRC/cmake/packages/litert_lm/CMakeLists.txt"
cat "$PROJ/bench/litert/advanced_main.cmake" >> "$PKG"

# 1c. stb is fetched at "master"; upstream moved stb_image_resize2.h to deprecated/ (2026-10),
#     which breaks stb_impl.cc. Add that include dir.
FC="$SRC/cmake/modules/fetch_content.cmake"
sed -i \
    's|target_include_directories(stb_impl PUBLIC ${LITERTLM_STB_SRC_DIR})|target_include_directories(stb_impl PUBLIC ${LITERTLM_STB_SRC_DIR} ${LITERTLM_STB_SRC_DIR}/deprecated)|' "$FC"

# 2. configure the super-build: host gcc for protoc/flatc (prebuild), cross gcc for the target
cmake -S "$SRC" -B "$BLD" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_TOOLCHAIN_FILE="$PROJ/bench/litert/aarch64-linux-gnu.toolchain.cmake" \
    -DLITERTLM_TOOLCHAIN_SCRIPT="$PROJ/bench/litert/aarch64-linux-gnu.script.cmake" \
    -DLITERTLM_CC=/usr/bin/gcc -DLITERTLM_CXX=/usr/bin/g++ \
    -DLITERTLM_CC_NDK=/usr/bin/aarch64-linux-gnu-gcc -DLITERTLM_CXX_NDK=/usr/bin/aarch64-linux-gnu-g++ \
    -DCMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    > "$BLD.configure.log" 2>&1 || { tail -40 "$BLD.configure.log"; exit 1; }

# 3. build (all phases; the inner litert_lm project builds litert_lm_main)
cmake --build "$BLD" -j "$JOBS" > "$BLD.build.log" 2>&1 || { grep -n -m 30 -B 3 -A 12 "error" "$BLD.build.log" | tail -80; exit 1; }

BIN="$(find "$BLD" -type f -name litert_lm_main -perm -u+x | head -1)"
[ -n "$BIN" ] || { echo "litert_lm_main not found"; exit 1; }
mkdir -p "$OUT"
cp "$BIN" "$OUT/"
ADV="$(find "$BLD" -type f -name litert_lm_advanced_main -perm -u+x | head -1 || true)"
[ -n "$ADV" ] && cp "$ADV" "$OUT/"
{
    echo "LiteRT-LM tag: $TAG"
    echo "commit: $COMMIT"
    echo "built: $(date -Iseconds)"
    echo "build: CMake super-build, Release, C++20, fully static"
    echo "flags: -march=armv8-a+crc -mtune=cortex-a72 (XNNPACK/KleidiAI ISA kernels runtime-dispatched)"
    echo "cross compiler: $(aarch64-linux-gnu-gcc --version | head -1)"
    echo "host compiler (protoc/flatc): $(gcc --version | head -1)"
    echo "rust: $(rustc --version), target aarch64-unknown-linux-gnu"
    (cd "$OUT" && sha256sum litert_lm_main $( [ -f litert_lm_advanced_main ] && echo litert_lm_advanced_main ))
} > "$OUT/BUILD_INFO.txt"
file "$OUT"/litert_lm_main
cat "$OUT/BUILD_INFO.txt"
