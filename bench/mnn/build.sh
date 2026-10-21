#!/usr/bin/env bash
# Cross-compile MNN (LLM engine) for the Pi 4 inside WSL Ubuntu (run as root):
#   bash bench/mnn/build.sh [tag]
# Static aarch64 build: llm_demo, llm_bench (MNN's own tools) and our mnn_bench wrapper
# (bench/mnn/mnn_bench.cpp: same 64-token prompt ids as every other runtime).
# Output: build/baselines/mnn/ with BUILD_INFO.txt.
set -euo pipefail

TAG="${1:-3.6.1}"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=/root/src/MNN-"$TAG"
BLD="$SRC/build-pi4"
OUT="$PROJ/build/baselines/mnn"
JOBS="$(nproc)"

if [ ! -d "$SRC" ]; then
    git clone --quiet --depth 1 --branch "$TAG" https://github.com/alibaba/MNN "$SRC"
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "MNN $TAG @ $COMMIT"

# -march=armv8-a for the base library. MNN_ARM82 stays ON (MNN's default): its fp16 kernels are
# compiled separately with armv8.2 flags and only selected at run time when the CPU reports
# asimdhp, which the Cortex-A72 does not. KleidiAI is OFF (needs dotprod/i8mm/SME; A72 has none).
PI_FLAGS="-mcpu=cortex-a72"
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="$PI_FLAGS" -DCMAKE_CXX_FLAGS="$PI_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF \
    -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_LLM_BUILD_DEMO=ON -DLLM_SUPPORT_HTTP_RESOURCE=OFF \
    -DMNN_KLEIDIAI=OFF -DMNN_ARM82=ON -DMNN_USE_THREAD_POOL=ON -DMNN_OPENCL=OFF \
    -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_CONVERTER=OFF \
    > "$SRC/configure-pi4.log" 2>&1 || { tail -40 "$SRC/configure-pi4.log"; exit 1; }
cmake --build "$BLD" -j "$JOBS" --target MNN llm llm_demo llm_bench > "$SRC/build-pi4.log" 2>&1 \
    || { grep -m20 -B2 -A8 "error" "$SRC/build-pi4.log"; exit 1; }

# Our wrapper: link against the same static objects the MNN tools use.
LIBS=$(find "$BLD" -maxdepth 3 -name "*.a" | tr '\n' ' ')
echo "static libs: $LIBS"
aarch64-linux-gnu-g++ $PI_FLAGS -O2 -std=c++17 \
    -I "$SRC/include" -I "$SRC/transformers/llm/engine/include" \
    "$PROJ/bench/mnn/mnn_bench.cpp" -o "$BLD/mnn_bench" \
    -static -Wl,--whole-archive $LIBS -Wl,--no-whole-archive -lpthread -ldl

mkdir -p "$OUT"
cp "$BLD"/llm_demo "$BLD"/llm_bench "$BLD"/mnn_bench "$OUT"/
{
    echo "MNN tag: $TAG"
    echo "commit: $COMMIT"
    echo "built: $(date -Iseconds)"
    echo "flags: $PI_FLAGS, static, MNN_LOW_MEMORY=ON, TRANSFORMER_FUSE=ON, ARM82=ON (runtime-dispatched), KLEIDIAI=OFF"
    echo "cross compiler: $(aarch64-linux-gnu-gcc --version | head -1)"
    (cd "$OUT" && sha256sum llm_demo llm_bench mnn_bench)
} > "$OUT/BUILD_INFO.txt"
file "$OUT"/mnn_bench
cat "$OUT/BUILD_INFO.txt"
