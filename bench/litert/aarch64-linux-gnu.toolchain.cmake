# CMake toolchain: cross-compile LiteRT-LM (CMake super-build) for the Raspberry Pi 4
# (Cortex-A72, ARMv8.0-A + CRC; no crypto/dotprod/fp16/LSE/RCpc/RDM) from x86_64 WSL Ubuntu.
# Modelled on LiteRT-LM's cmake/toolchains/litertlm_android.toolchain.cmake.
# Base ISA is plain armv8-a+crc; XNNPACK/KleidiAI still compile their own ISA-specific
# microkernels with per-file -march flags and select them at run time via cpuinfo.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(LITERTLM_PI4_ARCH_FLAGS "-march=armv8-a+crc -mtune=cortex-a72")
set(LITERTLM_AARCH64_EXTRA "/root/cache/aarch64-extra")  # static zlib for -lz

set(CMAKE_C_FLAGS_INIT   "${LITERTLM_PI4_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${LITERTLM_PI4_ARCH_FLAGS}")
# -L for the static zlib only. Full -static is applied to the two CLI targets alone
# (bench/litert/advanced_main.cmake): a global -static breaks helper executables that link
# shared libs (zlib's test program). Static because the cross sysroot's glibc/libstdc++ (GCC 15)
# are newer than the Pi's (Debian trixie).
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${LITERTLM_AARCH64_EXTRA}/lib")
# LiteRT also links a few shared libs (libLiteRt.so, NPU dispatch .so) that need -lz / -lfarmhash;
# build.sh links TFLite's libfarmhash.a into the same dir.
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-L${LITERTLM_AARCH64_EXTRA}/lib")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-L${LITERTLM_AARCH64_EXTRA}/lib")

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu ${LITERTLM_AARCH64_EXTRA})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(LITERTLM_TOOLCHAIN_ARGS
    "-DCMAKE_SYSTEM_NAME=Linux"
    "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
    "-DRust_CARGO_TARGET=aarch64-unknown-linux-gnu"
    "-DLITERTLM_RUST_LINKER_OVERRIDE=aarch64-linux-gnu-gcc"
    "-DLITERTLM_RUST_CARGO_ENV_VAR=CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER"
    "-DLITERTLM_PROTOBUF_CXX_FLAGS_RELEASE=-O2 -DNDEBUG"
    # Dependency pins (only this list reaches the inner litert_lm phase). Upstream CMake fetches
    # TensorFlow "master" and LiteRT "main"; TF f83e8816 (2026-09-28) deleted the profiling /
    # benchmark proto CMakeLists that v0.18.1's tflite shim still add_subdirectory()s. Pin TF to
    # that commit's parent and LiteRT to v0.18.1's own Bazel pin (WORKSPACE LITERT_REF).
    "-DLITERTLM_TFLITE_TAG=b4449a41869c3dbac7b4db333d019a941584e7a5"
    "-DLITERTLM_LITERT_TAG=ca21dc99de51c76bdcc572afeea1dd937d3c9e5c"
    # kissfft (FetchContent in the inner project) defaults to a shared lib; the CLIs link -static
    "-DKISSFFT_STATIC=ON"
    CACHE INTERNAL "String used to define parameterized CMAKE_ARGS"
)

# Skip LiteRT's Samsung NPU vendor code: aarch64-linux-gnu-g++ 15.2 hits a deterministic internal
# compiler error (segfault) in vendors/samsung/ai_litecore_manager.h:83; the CPU path does not use
# it. Set here (not in LITERTLM_TOOLCHAIN_ARGS) because the dependency externals (LiteRT, TFLite)
# only receive this toolchain file. LITECORE_GIT_REPO stops the Samsung header download that
# would otherwise force LITERT_ENABLE_SAMSUNG back ON.
set(LITERT_ENABLE_SAMSUNG OFF CACHE BOOL "pinfer: skip Samsung NPU vendor (GCC 15.2 ICE)" FORCE)
set(LITECORE_GIT_REPO "disabled" CACHE STRING "pinfer: no Samsung LiteCore header download" FORCE)
