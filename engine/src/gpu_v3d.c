/* Compile the independently written DRM client into AArch64 builds only. */
#ifdef __aarch64__
#include "../gpu/v3d.c"
#endif
