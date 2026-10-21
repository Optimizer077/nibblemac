# Included by LiteRT-LM's top-level CMakeLists via -DLITERTLM_TOOLCHAIN_SCRIPT (orchestrator scope).
# Mirrors cmake/toolchains/litertlm_android.script.cmake for an aarch64-linux-gnu GCC target.
set(RUST_TARGET "aarch64-unknown-linux-gnu")
set(LITERTLM_CCRS_CXXFLAGS_KEY "CXXFLAGS_aarch64_unknown_linux_gnu")
set(LITERTLM_CCRS_CXXFLAGS_VAL "-std=c++20 -march=armv8-a+crc -mtune=cortex-a72")
set(LITERTLM_CCRS_CFLAGS_KEY "CFLAGS_aarch64_unknown_linux_gnu")
set(LITERTLM_CCRS_CFLAGS_VAL "-march=armv8-a+crc -mtune=cortex-a72")
set(LITERTLM_RUST_LINKER_OVERRIDE "aarch64-linux-gnu-gcc" CACHE STRING "Rust linker for aarch64")
set(LITERTLM_RUST_CARGO_ENV_VAR "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER" CACHE STRING "")
execute_process(
    COMMAND cargo fetch
    WORKING_DIRECTORY "${LITERTLM_PROJECT_ROOT}"
    COMMAND_ERROR_IS_FATAL ANY
)
