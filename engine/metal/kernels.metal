#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Shape { uint rows, cols, batch; };

// Same 32-column activation quantization as kern_act.c. Reduction of integer
// sums is exact; precise divides avoid reciprocal approximations changing codes.
kernel void nm_quant_a8(device const float *x [[buffer(0)]],
                         device char *q [[buffer(1)]], device float *d [[buffer(2)]],
                         device float *fs [[buffer(3)]], constant Shape &s [[buffer(4)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    size_t base = size_t(group.y) * s.cols + group.x * 32;
    float v = x[base + lane];
    float maximum = simd_max(abs(v));
    float inverse = maximum > 0.0f ? precise::divide(127.0f, maximum) : 0.0f;
    int code = int(rint(v * inverse));
    size_t a = size_t(group.x) * s.batch + group.y;
    q[a * 32 + lane] = char(clamp(code, -127, 127));
    int total = simd_sum(code);
    if (lane == 0) {
        float scale = precise::divide(maximum, 127.0f);
        d[a] = scale;
        fs[a] = scale * float(total);
    }
}

inline float nm_exp_exact(float x) {
    x = clamp(x, -87.0f, 88.0f);
    float exponent = rint(x * 1.44269504088896341f);
    float r = x - exponent * 0.693145751953125f;
    r = r - exponent * 1.428606765330187e-06f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    float y = p * (r * r) + r;
    y = y + 1.0f;
    return y * as_type<float>(uint(int(exponent) + 127) << 23);
}

kernel void nm_silu_mul(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                         constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) gate[i] = precise::divide(gate[i], 1.0f + nm_exp_exact(-gate[i])) * up[i];
}

struct LinShape { uint n, heads, rep, dk, dv, qkv_stride, o_stride, keep, logged; };
inline size_t state_index(constant LinShape &s, uint h, uint i, uint j) {
    return size_t(h) * s.dk * s.dv + size_t(j / 32) * s.dk * 32 + i * 32 + j % 32;
}

kernel void nm_lin_rule(device float *S [[buffer(0)]], device const float *qkv [[buffer(1)]],
                         device const float *decay [[buffer(2)]], device const float *beta [[buffer(3)]],
                         device float *output [[buffer(4)]], device float *log [[buffer(5)]],
                         device const float *pend [[buffer(6)]], device float *work [[buffer(7)]],
                         constant LinShape &s [[buffer(8)]], uint2 pos [[thread_position_in_grid]]) {
    uint j = pos.x, h = pos.y;
    if (j >= s.dv) return;
    uint kh = h / s.rep, kd = (s.heads / s.rep) * s.dk;
    size_t width = s.dk + s.dv + 1;
    // Deferred speculative updates replay in the same order as kern_lin.c.
    for (uint i = 0; i < s.dk; i++) {
        size_t cell = state_index(s, h, i, j);
        float value = S[cell];
        for (uint t = 0; t < s.keep; t++) {
            size_t u = (size_t(t) * s.heads + h) * width;
            value = value * pend[u + s.dk + s.dv];
            value = value + pend[u + i] * pend[u + s.dk + j];
        }
        if (s.keep) S[cell] = value;
    }
    for (uint t = 0; t < s.n; t++) {
        device const float *in = qkv + size_t(t) * s.qkv_stride;
        device const float *q = in + kh * s.dk;
        device const float *k = in + kd + kh * s.dk;
        float v = in[2 * kd + h * s.dv + j];
        float d = decay[size_t(t) * s.heads + h];
        device float *dest = s.logged && t > 0 ? work : S;
        device const float *src = s.logged && t == 1 ? S : dest;
        float kv = 0.0f;
        for (uint i = 0; i < s.dk; i++) {
            size_t cell = state_index(s, h, i, j);
            float value = src[cell] * d;
            dest[cell] = value;
            kv = kv + value * k[i];
        }
        float delta = (v - kv) * beta[size_t(t) * s.heads + h];
        float o = 0.0f;
        for (uint i = 0; i < s.dk; i++) {
            size_t cell = state_index(s, h, i, j);
            float value = dest[cell] + k[i] * delta;
            dest[cell] = value;
            o = o + value * q[i];
        }
        output[size_t(t) * s.o_stride + h * s.dv + j] = o;
        if (s.logged && t > 0) {
            size_t u = (size_t(t) * s.heads + h) * width;
            log[u + s.dk + j] = delta;
            if (j == 0) {
                for (uint i = 0; i < s.dk; i++) log[u + i] = k[i];
                log[u + s.dk + s.dv] = d;
            }
        }
    }
}

kernel void nm_lin_apply(device float *S [[buffer(0)]], device const float *pend [[buffer(1)]],
                          constant LinShape &s [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    uint j = pos.x, h = pos.y;
    if (j >= s.dv) return;
    size_t width = s.dk + s.dv + 1;
    for (uint i = 0; i < s.dk; i++) {
        size_t cell = state_index(s, h, i, j);
        float value = S[cell];
        for (uint t = 0; t < s.keep; t++) {
            size_t u = (size_t(t) * s.heads + h) * width;
            value = value * pend[u + s.dk + s.dv];
            value = value + pend[u + i] * pend[u + s.dk + j];
        }
        S[cell] = value;
    }
}

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

// Small prefill/verification tile: one packed row is reused across four token
// vectors. Chronological block accumulation is unchanged for each output.
kernel void nm_q4k_a8_packed4(device const uchar *w [[buffer(0)]],
                               device const char *q [[buffer(1)]], device float *y [[buffer(2)]],
                               constant Shape &s [[buffer(3)]], device const float *d [[buffer(4)]],
                               device const float *fs [[buffer(5)]], uint2 position [[thread_position_in_grid]]) {
    uint row = position.x, first = position.y * 4;
    if (row >= s.rows) return;
    uint r = row % 32, ri = r % 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = 0; b < s.cols / 256; b++) {
        device const uchar *block = w + (size_t(row / 32) * (s.cols / 256) + b) * 4608;
        device const uchar *sb = block + (r / 4) * 64;
        float D = float(reinterpret_cast<device const half *>(sb)[ri]);
        float M = float(reinterpret_cast<device const half *>(sb + 8)[ri]);
        #pragma unroll
        for (uint k = 0; k < 8; k++) {
            int sums[4] = {0, 0, 0, 0};
            size_t base = size_t(b * 8 + k) * s.batch + first;
            #pragma unroll
            for (uint j = 0; j < 4; j++) {
                uchar4 v = *reinterpret_cast<device const uchar4 *>(block + 512 + (k * 4 + j) * 128 + r * 4);
                int4 lo = int4(v & 15), hi = int4(v >> 4);
                #pragma unroll
                for (uint t = 0; t < 4; t++) if (first + t < s.batch) {
                    device const char *x = q + (base + t) * 32;
                    char4 x0 = *reinterpret_cast<device const char4 *>(x + j * 4);
                    char4 x1 = *reinterpret_cast<device const char4 *>(x + 16 + j * 4);
                    int4 p = lo * int4(x0) + hi * int4(x1);
                    sums[t] += p.x + p.y + p.z + p.w;
                }
            }
            float scale = D * float(unpack6(sb + 16, k * 4 + ri));
            float offset = M * float(unpack6(sb + 40, k * 4 + ri));
            #pragma unroll
            for (uint t = 0; t < 4; t++) if (first + t < s.batch) {
                acc[t] = fma(float(sums[t]), scale * d[base + t], acc[t]);
                acc[t] = fma(-offset, fs[base + t], acc[t]);
            }
        }
    }
    for (uint t = 0; t < 4; t++) if (first + t < s.batch) y[size_t(first + t) * s.rows + row] = acc[t];
}
