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

// Q4K x upstream int8 activations. Eight lanes per row, four rows per SIMD
// group: coalesced loads follow the packed four-row tile. Exact integer dot,
// then the same chronological FMA + FMS as kern_quant.c. No fp32 reduction.
kernel void nm_q4k_a8(device const uchar *w [[buffer(0)]],
                       device const char *q [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant Shape &s [[buffer(3)]],
                       device const float *d [[buffer(4)]],
                       device const float *fs [[buffer(5)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint tile = group.x * 4 + simd, r = lane / 8, j = lane % 8;
    if (tile * 4 >= s.rows) return;
    float acc = 0.0f;
    for (uint b = 0; b < s.cols / 256; b++) {
        device const uchar *sb = w + (size_t(tile) * (s.cols / 256) + b) * 576;
        float D = float(reinterpret_cast<device const half *>(sb)[r]);
        float M = float(reinterpret_cast<device const half *>(sb + 8)[r]);
        for (uint k = 0; k < 8; k++) {
            size_t a = (size_t(b * 8 + k) * s.batch + group.y);
            device const char *x = q + a * 32;
            uchar lo = sb[64 + k * 64 + r * 16 + j];
            uchar hi = sb[64 + k * 64 + r * 16 + j + 8];
            int dot = int(lo & 15) * int(x[j]) + int(hi & 15) * int(x[j + 8]) +
                      int(lo >> 4) * int(x[j + 16]) + int(hi >> 4) * int(x[j + 24]);
            dot += simd_shuffle_down(dot, 4);
            dot += simd_shuffle_down(dot, 2);
            dot += simd_shuffle_down(dot, 1);
            if (j == 0) {
                float scale = D * float(unpack6(sb + 16, k * 4 + r));
                float offset = M * float(unpack6(sb + 40, k * 4 + r));
                acc = fma(float(dot), scale * d[a], acc);
                acc = fma(-offset, fs[a], acc);
            }
        }
    }
    if (j == 0) y[size_t(group.y) * s.rows + tile * 4 + r] = acc;
}

// One complete row per GPU lane: reuse uniform input loads, remove three
// cross-lane integer reductions and the 7/8-lane scale/FMA masking per block.
// Arithmetic remains an exact integer dot followed by ordered CPU-equivalent FMAs.
kernel void nm_q4k_a8_row(device const uchar *w [[buffer(0)]],
                           device const char *q [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant Shape &s [[buffer(3)]],
                           device const float *d [[buffer(4)]],
                           device const float *fs [[buffer(5)]],
                           uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    if (row >= s.rows) return;
    uint r = row % 4;
    float acc = 0.0f;
    for (uint b = 0; b < s.cols / 256; b++) {
        device const uchar *sb = w + (size_t(row / 4) * (s.cols / 256) + b) * 576;
        float D = float(reinterpret_cast<device const half *>(sb)[r]);
        float M = float(reinterpret_cast<device const half *>(sb + 8)[r]);
        #pragma unroll
        for (uint k = 0; k < 8; k++) {
            size_t a = size_t(b * 8 + k) * s.batch + position.y;
            device const char *x = q + a * 32;
            device const uchar *codes = sb + 64 + k * 64 + r * 16;
            int dot = 0;
            #pragma unroll
            for (uint j = 0; j < 4; j++) {
                uchar4 v = *reinterpret_cast<device const uchar4 *>(codes + j * 4);
                char4 x0 = *reinterpret_cast<device const char4 *>(x + j * 4);
                char4 x1 = *reinterpret_cast<device const char4 *>(x + 16 + j * 4);
                int4 p = int4(v & 15) * int4(x0) + int4(v >> 4) * int4(x1);
                dot += p.x + p.y + p.z + p.w;
            }
            float scale = D * float(unpack6(sb + 16, k * 4 + r));
            float offset = M * float(unpack6(sb + 40, k * 4 + r));
            acc = fma(float(dot), scale * d[a], acc);
            acc = fma(-offset, fs[a], acc);
        }
    }
    y[size_t(position.y) * s.rows + row] = acc;
}

// Same exact arithmetic as the row kernel; code words are transposed once at
// upload for coalesced 32-lane loads. Metadata and all quantized values are kept.
kernel void nm_q4k_a8_packed(device const uchar *w [[buffer(0)]],
                              device const char *q [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant Shape &s [[buffer(3)]],
                              device const float *d [[buffer(4)]],
                              device const float *fs [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    if (row >= s.rows) return;
    uint r = row % 32, ri = r % 4;
    float acc = 0.0f;
    for (uint b = 0; b < s.cols / 256; b++) {
        device const uchar *block = w + (size_t(row / 32) * (s.cols / 256) + b) * 4608;
        device const uchar *sb = block + (r / 4) * 64;
        float D = float(reinterpret_cast<device const half *>(sb)[ri]);
        float M = float(reinterpret_cast<device const half *>(sb + 8)[ri]);
        #pragma unroll
        for (uint k = 0; k < 8; k++) {
            size_t a = size_t(b * 8 + k) * s.batch + position.y;
            device const char *x = q + a * 32;
            int dot = 0;
            #pragma unroll
            for (uint j = 0; j < 4; j++) {
                uchar4 v = *reinterpret_cast<device const uchar4 *>(block + 512 + (k * 4 + j) * 128 + r * 4);
                char4 x0 = *reinterpret_cast<device const char4 *>(x + j * 4);
                char4 x1 = *reinterpret_cast<device const char4 *>(x + 16 + j * 4);
                int4 p = int4(v & 15) * int4(x0) + int4(v >> 4) * int4(x1);
                dot += p.x + p.y + p.z + p.w;
            }
            float scale = D * float(unpack6(sb + 16, k * 4 + ri));
            float offset = M * float(unpack6(sb + 40, k * 4 + ri));
            acc = fma(float(dot), scale * d[a], acc);
            acc = fma(-offset, fs[a], acc);
        }
    }
    y[size_t(position.y) * s.rows + row] = acc;
}
