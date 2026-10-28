#include <metal_stdlib>
using namespace metal;

struct Shape { uint rows, cols, batch; };

// Four SIMD groups per threadgroup, one row per SIMD group. Keep all reductions
// in fp32. These are correctness-first baselines, not a tiled prefill GEMM.
kernel void nm_f32(device const float *w [[buffer(0)]],
                   device const float *x [[buffer(1)]],
                   device float *y [[buffer(2)]],
                   constant Shape &s [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint simd [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4 + simd;
    if (row >= s.rows) return;
    float sum = 0.0f;
    for (uint c = lane; c < s.cols; c += 32)
        sum += w[size_t(row) * s.cols + c] * x[size_t(group.y) * s.cols + c];
    sum = simd_sum(sum);
    if (lane == 0) y[size_t(group.y) * s.rows + row] = sum;
}

inline uint unpack6(device const uchar *p, uint i) {
    uint lo = (uint(p[i % 16]) >> (4 * (i / 16))) & 15;
    uint hi = (uint(p[16 + i % 8]) >> (2 * (i / 8))) & 3;
    return lo | (hi << 4);
}

// Directly consume upstream .pq Q4K tiles: no fp16 expansion or weight repack.
kernel void nm_q4k(device const uchar *w [[buffer(0)]],
                    device const float *x [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant Shape &s [[buffer(3)]],
                    uint2 group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4 + simd;
    if (row >= s.rows) return;
    float sum = 0.0f;
    for (uint b = 0; b < s.cols / 256; b++) {
        device const uchar *sb = w + (size_t(group.x) * (s.cols / 256) + b) * 576;
        float D = float(reinterpret_cast<device const half *>(sb)[simd]);
        float M = float(reinterpret_cast<device const half *>(sb + 8)[simd]);
        for (uint k = 0; k < 8; k++) {
            uint i = k * 4 + simd;
            float scale = D * float(unpack6(sb + 16, i));
            float minimum = M * float(unpack6(sb + 40, i));
            uchar packed = sb[64 + k * 64 + simd * 16 + lane % 16];
            uint code = (uint(packed) >> (4 * (lane / 16))) & 15;
            float weight = scale * float(code) - minimum;
            sum += weight * x[size_t(group.y) * s.cols + b * 256 + k * 32 + lane];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) y[size_t(group.y) * s.rows + row] = sum;
}
