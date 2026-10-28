/* Compile the independently written DRM client into AArch64 builds only. */
#if defined(__aarch64__) && defined(__linux__)
#include "../gpu/v3d.c"
#endif
