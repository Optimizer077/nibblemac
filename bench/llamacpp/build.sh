#!/usr/bin/env bash
# Build the llama.cpp baseline (run inside WSL Ubuntu as root):
#   1. host (x86) tools: llama-quantize, for making GGUF quantizations on the PC
#   2. Pi 4 binaries: static aarch64, -mcpu=cortex-a72, no runtime CPU dispatch
# Outputs: build/aarch64/llama/ and build/host/llama/ in the project, with BUILD_INFO.txt.
set -euo pipefail

TAG="${1:-v0.7.0}"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=/root/src/llama.cpp-"$TAG"
OUT_PI="$PROJ/build/aarch64/llama"
OUT_HOST="$PROJ/build/host/llama"
JOBS="$(nproc)"

if [ ! -d "$SRC" ]; then
    git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "llama.cpp $TAG @ $COMMIT"

COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF)

echo "== host build"
cmake -S "$SRC" -B "$SRC/build-host" "${COMMON[@]}" > "$SRC/configure-host.log"
cmake --build "$SRC/build-host" -j "$JOBS" --target llama-quantize > "$SRC/build-host.log"

echo "== Pi 4 cross build"
PI_FLAGS="-mcpu=cortex-a72"
cmake -S "$SRC" -B "$SRC/build-pi4" "${COMMON[@]}" \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="$PI_FLAGS" -DCMAKE_CXX_FLAGS="$PI_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_OPENMP=ON \
    -DOpenMP_gomp_LIBRARY=/usr/lib/gcc-cross/aarch64-linux-gnu/15/libgomp.a \
    > "$SRC/configure-pi4.log"
cmake --build "$SRC/build-pi4" -j "$JOBS" --target llama-bench llama-completion llama-simple llama-perplexity \
    > "$SRC/build-pi4.log"

mkdir -p "$OUT_PI" "$OUT_HOST"
cp "$SRC"/build-pi4/bin/{llama-bench,llama-completion,llama-simple,llama-perplexity} "$OUT_PI"/
cp "$SRC"/build-host/bin/llama-quantize "$OUT_HOST"/
for d in "$OUT_PI" "$OUT_HOST"; do
    {
        echo "llama.cpp tag: $TAG"
        echo "commit: $COMMIT"
        echo "built: $(date -Iseconds)"
        echo "pi flags: $PI_FLAGS, static, GGML_NATIVE=OFF, GGML_CPU_ARM_ARCH=armv8-a, OpenMP"
        echo "cross compiler: $(aarch64-linux-gnu-gcc --version | head -1)"
        echo "host compiler: $(gcc --version | head -1)"
        (cd "$d" && sha256sum -- * | grep -v BUILD_INFO)
    } > "$d/BUILD_INFO.txt"
done
file "$OUT_PI/llama-bench"
cat "$OUT_PI/BUILD_INFO.txt"
