#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "metal.h"

@interface NMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> f32;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k_a8;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k_a8_row;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k_a8_packed;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k_a8_packed4;
@property(nonatomic, strong) id<MTLComputePipelineState> quant_a8;
@property(nonatomic, strong) id<MTLComputePipelineState> silu_mul;
@property(nonatomic, strong) id<MTLComputePipelineState> lin_rule;
@property(nonatomic, strong) id<MTLComputePipelineState> lin_apply;
@property(nonatomic) BOOL useRowKernel;
@property(nonatomic) BOOL usePackedKernel;
@property(nonatomic) BOOL useBatchKernel;
@end
@implementation NMContext
@end

static int fail(char *error, size_t size, NSString *message) {
    if (error && size) snprintf(error, size, "%s", message.UTF8String ?: "Metal error");
    return -1;
}

nm_metal *nm_metal_create(const char *path, char *error, size_t size) {
    @autoreleasepool {
        NMContext *c = [NMContext new];
        c.device = MTLCreateSystemDefaultDevice();
        if (!c.device || !c.device.hasUnifiedMemory) {
            fail(error, size, @"An Apple Silicon Mac with a unified-memory Metal device is required");
            return NULL;
        }
        if (!path) { fail(error, size, @"Shader path is required"); return NULL; }
        NSError *err = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(path)
                                      encoding:NSUTF8StringEncoding error:&err];
        if (!source) { fail(error, size, err.localizedDescription); return NULL; }
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        options.languageVersion = MTLLanguageVersion2_4;
        id<MTLLibrary> library = [c.device newLibraryWithSource:source options:options error:&err];
        if (!library) { fail(error, size, err.localizedDescription); return NULL; }
        c.f32 = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_f32"]
                                  error:&err];
        if (!c.f32) { fail(error, size, err.localizedDescription); return NULL; }
        c.q4k = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_q4k"]
                                  error:&err];
        if (!c.q4k) { fail(error, size, err.localizedDescription); return NULL; }
        c.q4k_a8 = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_q4k_a8"] error:&err];
        if (!c.q4k_a8) { fail(error, size, err.localizedDescription); return NULL; }
        c.q4k_a8_row = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_q4k_a8_row"] error:&err];
        if (!c.q4k_a8_row) { fail(error, size, err.localizedDescription); return NULL; }
        c.q4k_a8_packed = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_q4k_a8_packed"] error:&err];
        if (!c.q4k_a8_packed) { fail(error, size, err.localizedDescription); return NULL; }
        c.q4k_a8_packed4 = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_q4k_a8_packed4"] error:&err];
        if (!c.q4k_a8_packed4) { fail(error, size, err.localizedDescription); return NULL; }
        c.quant_a8 = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_quant_a8"] error:&err];
        c.silu_mul = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_silu_mul"] error:&err];
        c.lin_rule = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_lin_rule"] error:&err];
        c.lin_apply = [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_lin_apply"] error:&err];
        if (!c.quant_a8 || !c.silu_mul || !c.lin_rule || !c.lin_apply) {
            fail(error, size, err.localizedDescription ?: @"Graph kernel compilation failed"); return NULL;
        }
        const char *variant = getenv("NM_METAL_KERNEL");
        if (!variant) variant = "packed";
        if (strcmp(variant, "row") && strcmp(variant, "simd") && strcmp(variant, "packed") && strcmp(variant, "packed4")) {
            fail(error, size, @"NM_METAL_KERNEL must be row, simd, packed or packed4"); return NULL;
        }
        c.useRowKernel = variant && strcmp(variant, "simd");
        c.usePackedKernel = !strcmp(variant, "packed") || !strcmp(variant, "packed4");
        c.useBatchKernel = !strcmp(variant, "packed4");
        if (c.f32.threadExecutionWidth != 32 || c.q4k.threadExecutionWidth != 32 ||
            c.q4k_a8.threadExecutionWidth != 32 || c.q4k_a8.maxTotalThreadsPerThreadgroup < 128 ||
            c.q4k_a8_row.threadExecutionWidth != 32 || c.q4k_a8_row.maxTotalThreadsPerThreadgroup < 128 ||
            c.q4k_a8_packed.threadExecutionWidth != 32 || c.q4k_a8_packed.maxTotalThreadsPerThreadgroup < 128 ||
            c.q4k_a8_packed4.threadExecutionWidth != 32 || c.q4k_a8_packed4.maxTotalThreadsPerThreadgroup < 128 ||
            c.quant_a8.threadExecutionWidth != 32 || c.lin_rule.maxTotalThreadsPerThreadgroup < 128 ||
            c.lin_apply.maxTotalThreadsPerThreadgroup < 128 || c.silu_mul.maxTotalThreadsPerThreadgroup < 128 ||
            c.f32.maxTotalThreadsPerThreadgroup < 128 || c.q4k.maxTotalThreadsPerThreadgroup < 128) {
            fail(error, size, @"Kernels require 32-lane SIMD groups and 128-thread threadgroups");
            return NULL;
        }
        c.queue = [c.device newCommandQueue];
        if (!c.queue) { fail(error, size, @"Unable to create Metal command queue"); return NULL; }
        return (nm_metal *)CFBridgingRetain(c);
    }
}

static void encode_a8(NMContext *c, id<MTLComputeCommandEncoder> encoder,
                      uint32_t rows, uint32_t cols, uint32_t batch,
                      id<MTLBuffer> w, id<MTLBuffer> q, id<MTLBuffer> d, id<MTLBuffer> fs, id<MTLBuffer> y) {
    BOOL tile = c.useBatchKernel && batch >= 4;
    [encoder setComputePipelineState:tile ? c.q4k_a8_packed4 : c.usePackedKernel ? c.q4k_a8_packed :
                                       c.useRowKernel ? c.q4k_a8_row : c.q4k_a8];
    [encoder setBuffer:w offset:0 atIndex:0];
    [encoder setBuffer:q offset:0 atIndex:1];
    [encoder setBuffer:y offset:0 atIndex:2];
    const uint32_t shape[] = {rows, cols, batch};
    [encoder setBytes:shape length:sizeof shape atIndex:3];
    [encoder setBuffer:d offset:0 atIndex:4];
    [encoder setBuffer:fs offset:0 atIndex:5];
    uint32_t rows_per_group = c.useRowKernel ? 128 : 16;
    [encoder dispatchThreadgroups:MTLSizeMake(((uint64_t)rows + rows_per_group - 1) / rows_per_group, tile ? (batch + 3) / 4 : batch, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static BOOL fits_weights(NMContext *c, id<MTLBuffer> w, uint32_t rows, uint32_t cols) {
    uint64_t tiles = c.usePackedKernel ? (((uint64_t)rows + 31) / 32) * 8 : rows / 4;
    return w && w.device == c.device && tiles * (cols / 256) <= w.length / 576;
}

static BOOL fits_buffer(NMContext *c, nm_buffer *handle, uint64_t elements, uint32_t width) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)handle;
    return b && b.device == c.device && elements <= b.length / width;
}

static int complete(id<MTLCommandBuffer> command, char *error, size_t size) {
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted ? 0 :
        fail(error, size, command.error.localizedDescription ?: @"GPU graph execution failed");
}

static void encode_quant(NMContext *c, id<MTLComputeCommandEncoder> encoder, uint32_t cols, uint32_t batch,
                          nm_buffer *input, const nm_mlp_buffers *b) {
    [encoder setComputePipelineState:c.quant_a8];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)b->q offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)b->d offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)b->fs offset:0 atIndex:3];
    const uint32_t shape[] = {0, cols, batch};
    [encoder setBytes:shape length:sizeof shape atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(cols / 32, batch, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

int nm_metal_mlp(nm_metal *context, uint32_t hidden, uint32_t ffn, uint32_t batch,
                  nm_buffer *wg, nm_buffer *wu, nm_buffer *wd, const nm_mlp_buffers *b,
                  char *error, size_t size) {
    @autoreleasepool {
        NMContext *c = (__bridge NMContext *)context;
        if (!c || !b || !hidden || !ffn || !batch || batch > 64 || hidden % 256 || ffn % 256)
            return fail(error, size, @"MLP needs hidden/ffn divisible by 256 and batch 1..64");
        uint64_t h = (uint64_t)hidden * batch, f = (uint64_t)ffn * batch, maximum = MAX(h, f);
        if (!fits_weights(c, (__bridge id<MTLBuffer>)wg, ffn, hidden) ||
            !fits_weights(c, (__bridge id<MTLBuffer>)wu, ffn, hidden) ||
            !fits_weights(c, (__bridge id<MTLBuffer>)wd, hidden, ffn) ||
            !fits_buffer(c, b->x, h, 4) || !fits_buffer(c, b->gate, f, 4) || !fits_buffer(c, b->up, f, 4) ||
            !fits_buffer(c, b->y, h, 4) || !fits_buffer(c, b->q, maximum, 1) ||
            !fits_buffer(c, b->d, maximum / 32, 4) || !fits_buffer(c, b->fs, maximum / 32, 4) || f > UINT32_MAX)
            return fail(error, size, @"MLP buffers are missing, wrong-device, or too small");
        nm_buffer *handles[] = {wg, wu, wd, b->x, b->gate, b->up, b->q, b->d, b->fs, b->y};
        for (unsigned i = 3; i < 10; i++) for (unsigned j = 0; j < i; j++)
            if (handles[i] == handles[j]) return fail(error, size, @"MLP workspace must not alias weights or itself");
        id<MTLCommandBuffer> command = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(error, size, @"Unable to encode MLP graph");
        encode_quant(c, encoder, hidden, batch, b->x, b);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_a8(c, encoder, ffn, hidden, batch, (__bridge id<MTLBuffer>)wg,
                  (__bridge id<MTLBuffer>)b->q, (__bridge id<MTLBuffer>)b->d,
                  (__bridge id<MTLBuffer>)b->fs, (__bridge id<MTLBuffer>)b->gate);
        encode_a8(c, encoder, ffn, hidden, batch, (__bridge id<MTLBuffer>)wu,
                  (__bridge id<MTLBuffer>)b->q, (__bridge id<MTLBuffer>)b->d,
                  (__bridge id<MTLBuffer>)b->fs, (__bridge id<MTLBuffer>)b->up);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:c.silu_mul];
        [encoder setBuffer:(__bridge id<MTLBuffer>)b->gate offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:1];
        uint32_t count = (uint32_t)f;
        [encoder setBytes:&count length:sizeof count atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((f + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_quant(c, encoder, ffn, batch, b->gate, b);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_a8(c, encoder, hidden, ffn, batch, (__bridge id<MTLBuffer>)wd,
                  (__bridge id<MTLBuffer>)b->q, (__bridge id<MTLBuffer>)b->d,
                  (__bridge id<MTLBuffer>)b->fs, (__bridge id<MTLBuffer>)b->y);
        [encoder endEncoding];
        return complete(command, error, size);
    }
}

static BOOL valid_lin(NMContext *c, const nm_lin_shape *s, const nm_lin_buffers *b, BOOL apply) {
    if (!c || !s || !b || !s->heads || s->heads > 256 || !s->rep || s->heads % s->rep ||
        !s->dk || s->dk > 256 || !s->dv || s->dv > 512 || s->dv % 32 || s->keep > 64 ||
        !fits_buffer(c, b->state, (uint64_t)s->heads * s->dk * s->dv, 4)) return NO;
    uint64_t width = s->dk + s->dv + 1;
    if (s->keep && !fits_buffer(c, b->pend, (uint64_t)s->keep * s->heads * width, 4)) return NO;
    if (!apply && (!s->n || s->n > 64 || s->logged > 1 ||
        s->qkv_stride < 2 * (s->heads / s->rep) * s->dk + s->heads * s->dv ||
        s->o_stride < s->heads * s->dv ||
        !fits_buffer(c, b->qkv, (uint64_t)s->n * s->qkv_stride, 4) ||
        !fits_buffer(c, b->decay, (uint64_t)s->n * s->heads, 4) ||
        !fits_buffer(c, b->beta, (uint64_t)s->n * s->heads, 4) ||
        !fits_buffer(c, b->output, (uint64_t)s->n * s->o_stride, 4) ||
        !fits_buffer(c, b->log, s->logged ? (uint64_t)s->n * s->heads * width : 1, 4) ||
        !fits_buffer(c, b->pend, s->keep ? (uint64_t)s->keep * s->heads * width : 1, 4) ||
        !fits_buffer(c, b->work, s->logged ? (uint64_t)s->heads * s->dk * s->dv : 1, 4))) return NO;
    if (apply) return !s->keep || b->state != b->pend;
    nm_buffer *handles[] = {b->state, b->qkv, b->decay, b->beta, b->output, b->log, b->pend, b->work};
    for (unsigned i = 0; i < 8; i++) for (unsigned j = 0; j < i; j++)
        if (handles[i] && handles[i] == handles[j]) return NO;
    return YES;
}

int nm_metal_lin_rule(nm_metal *context, const nm_lin_shape *s, const nm_lin_buffers *b,
                      char *error, size_t size) {
    @autoreleasepool {
        NMContext *c = (__bridge NMContext *)context;
        if (!valid_lin(c, s, b, NO)) return fail(error, size, @"Invalid DeltaNet buffers or shape");
        id<MTLCommandBuffer> command = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(error, size, @"Unable to encode DeltaNet");
        [encoder setComputePipelineState:c.lin_rule];
        nm_buffer *handles[] = {b->state, b->qkv, b->decay, b->beta, b->output, b->log, b->pend, b->work};
        for (unsigned i = 0; i < 8; i++) [encoder setBuffer:(__bridge id<MTLBuffer>)handles[i] offset:0 atIndex:i];
        [encoder setBytes:s length:sizeof *s atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake((s->dv + 127) / 128, s->heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        return complete(command, error, size);
    }
}

int nm_metal_lin_apply(nm_metal *context, const nm_lin_shape *s, const nm_lin_buffers *b,
                       char *error, size_t size) {
    @autoreleasepool {
        NMContext *c = (__bridge NMContext *)context;
        if (!valid_lin(c, s, b, YES)) return fail(error, size, @"Invalid DeltaNet replay buffers or shape");
        if (!s->keep) return 0;
        id<MTLCommandBuffer> command = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(error, size, @"Unable to encode DeltaNet replay");
        [encoder setComputePipelineState:c.lin_apply];
        [encoder setBuffer:(__bridge id<MTLBuffer>)b->state offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)b->pend offset:0 atIndex:1];
        [encoder setBytes:s length:sizeof *s atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((s->dv + 127) / 128, s->heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        return complete(command, error, size);
    }
}

int nm_metal_matmul_a8_begin(nm_metal *context, uint32_t rows, uint32_t cols, uint32_t batch,
                       nm_buffer *weights, nm_buffer *qbuf, nm_buffer *dbuf, nm_buffer *fsbuf,
                       nm_buffer *outputs, nm_submission **pending, char *error, size_t size) {
    @autoreleasepool {
        if (!pending) return fail(error, size, @"Submission output is required");
        *pending = NULL;
        NMContext *c = (__bridge NMContext *)context;
        id<MTLBuffer> w = (__bridge id<MTLBuffer>)weights, q = (__bridge id<MTLBuffer>)qbuf;
        id<MTLBuffer> d = (__bridge id<MTLBuffer>)dbuf, fs = (__bridge id<MTLBuffer>)fsbuf;
        id<MTLBuffer> y = (__bridge id<MTLBuffer>)outputs;
        if (!c || !w || !q || !d || !fs || !y || !rows || !cols || !batch || rows % 4 || cols % 256)
            return fail(error, size, @"Q4K/A8 needs valid buffers, rows % 4 == 0 and cols % 256 == 0");
        if (w.device != c.device || q.device != c.device || d.device != c.device ||
            fs.device != c.device || y.device != c.device || y == w || y == q || y == d || y == fs)
            return fail(error, size, @"Invalid Q4K/A8 device or aliased output");
        uint64_t units = c.usePackedKernel ? (((uint64_t)rows + 31) / 32) * 8 * (cols / 256) :
                                             (uint64_t)(rows / 4) * (cols / 256);
        uint64_t inputs = (uint64_t)batch * cols;
        if (units > w.length / 576 || inputs > q.length || inputs / 32 > d.length / 4 ||
            inputs / 32 > fs.length / 4 || (uint64_t)batch * rows > y.length / 4)
            return fail(error, size, @"Q4K/A8 buffer too small");
        id<MTLCommandBuffer> command = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(error, size, @"Unable to create Q4K/A8 encoder");
        encode_a8(c, encoder, rows, cols, batch, w, q, d, fs, y);
        [encoder endEncoding]; [command commit];
        *pending = (nm_submission *)CFBridgingRetain(command);
        return 0;
    }
}

int nm_metal_matmul_a8_finish(nm_submission *pending, char *error, size_t size) {
    @autoreleasepool {
        if (!pending) return fail(error, size, @"A pending submission is required");
        id<MTLCommandBuffer> command = CFBridgingRelease(pending);
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted ? 0 :
            fail(error, size, command.error.localizedDescription ?: @"Q4K/A8 execution failed");
    }
}

int nm_metal_matmul_a8(nm_metal *context, uint32_t rows, uint32_t cols, uint32_t batch,
                       nm_buffer *w, nm_buffer *q, nm_buffer *d, nm_buffer *fs,
                       nm_buffer *y, char *error, size_t size) {
    nm_submission *pending = NULL;
    if (nm_metal_matmul_a8_begin(context, rows, cols, batch, w, q, d, fs, y, &pending, error, size)) return -1;
    return nm_metal_matmul_a8_finish(pending, error, size);
}

void nm_metal_destroy(nm_metal *c) { if (c) (void)CFBridgingRelease(c); }
const char *nm_metal_device_name(nm_metal *context) {
    NMContext *c = (__bridge NMContext *)context;
    return c.device.name.UTF8String;
}
uint64_t nm_metal_recommended_memory(nm_metal *context) {
    NMContext *c = (__bridge NMContext *)context;
    return c.device.recommendedMaxWorkingSetSize;
}
nm_buffer *nm_metal_buffer_create(nm_metal *context, size_t bytes) {
    NMContext *c = (__bridge NMContext *)context;
    if (!c || !bytes || bytes > c.device.maxBufferLength) return NULL;
    return (nm_buffer *)CFBridgingRetain([c.device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared]);
}
void nm_metal_buffer_destroy(nm_buffer *b) { if (b) (void)CFBridgingRelease(b); }
void *nm_metal_buffer_contents(nm_buffer *buffer) {
    return [(__bridge id<MTLBuffer>)buffer contents];
}

nm_buffer *nm_metal_a8_weights(nm_metal *context, uint32_t rows, uint32_t cols,
                               const void *packed, size_t bytes) {
    NMContext *c = (__bridge NMContext *)context;
    if (!c || !packed || !rows || !cols || rows % 4 || cols % 256) return NULL;
    uint64_t units = (uint64_t)(rows / 4) * (cols / 256);
    if (units > SIZE_MAX / 576 || bytes != units * 576) return NULL;
    uint64_t padded_units = (((uint64_t)rows + 31) / 32) * 8 * (cols / 256);
    if (padded_units > SIZE_MAX / 576) return NULL;
    size_t length = c.usePackedKernel ? padded_units * 576 : bytes;
    nm_buffer *buffer = nm_metal_buffer_create(context, length);
    if (!buffer) return NULL;
    unsigned char *dst = nm_metal_buffer_contents(buffer);
    const unsigned char *src = packed;
    if (!c.usePackedKernel) { memcpy(dst, src, bytes); return buffer; }
    memset(dst, 0, length);
    // 32 rows x 256 columns: 512 metadata bytes + 4096 code bytes. The
    // codes' uint32 words are transposed across rows so a GPU SIMD group
    // reads one contiguous 128-byte span per instruction. Still 4.5 bits/weight.
    for (uint32_t t = 0; t < ((uint64_t)rows + 31) / 32; t++)
        for (uint32_t b = 0; b < cols / 256; b++) {
            unsigned char *out = dst + ((size_t)t * (cols / 256) + b) * 4608;
            for (uint32_t r = 0; r < 32 && (uint64_t)t * 32 + r < rows; r++) {
                const unsigned char *in = src + ((size_t)((t * 32 + r) / 4) * (cols / 256) + b) * 576;
                if (r % 4 == 0) memcpy(out + (r / 4) * 64, in, 64);
                for (uint32_t k = 0; k < 8; k++)
                    for (uint32_t j = 0; j < 4; j++)
                        memcpy(out + 512 + (k * 4 + j) * 128 + r * 4,
                               in + 64 + k * 64 + (r % 4) * 16 + j * 4, 4);
            }
        }
    return buffer;
}

int nm_metal_matmul(nm_metal *context, uint32_t dtype, uint32_t rows, uint32_t cols,
                    uint32_t batch, nm_buffer *weights, nm_buffer *inputs,
                    nm_buffer *outputs, uint32_t repeats, double *gpu_seconds,
                    char *error, size_t size) {
    @autoreleasepool {
        if (gpu_seconds) *gpu_seconds = 0;
        NMContext *c = (__bridge NMContext *)context;
        id<MTLBuffer> w = (__bridge id<MTLBuffer>)weights;
        id<MTLBuffer> x = (__bridge id<MTLBuffer>)inputs;
        id<MTLBuffer> y = (__bridge id<MTLBuffer>)outputs;
        if (!c || !w || !x || !y || !rows || !cols || !batch || !repeats)
            return fail(error, size, @"Context, buffers, dimensions and repeats must be nonzero");
        if (w.device != c.device || x.device != c.device || y.device != c.device || w == y || x == y)
            return fail(error, size, @"Buffers must belong to this device and output must not alias inputs");
        if (dtype != NM_F32 && dtype != NM_Q4K)
            return fail(error, size, @"Unsupported dtype (expected F32=0 or Q4K=4)");
        if (dtype == NM_Q4K && (rows % 4 || cols % 256))
            return fail(error, size, @"Q4K needs rows divisible by 4 and cols divisible by 256");
        // Division-first checks avoid overflow for malformed uint32 dimensions.
        uint64_t elements = (uint64_t)rows * cols;
        uint64_t units = dtype == NM_F32 ? elements : elements / 1024;
        uint64_t unit_bytes = dtype == NM_F32 ? 4 : 576;
        if (units > w.length / unit_bytes ||
            (uint64_t)batch * cols > x.length / 4 || (uint64_t)batch * rows > y.length / 4)
            return fail(error, size, @"Buffer is too small for the requested shape");
        id<MTLCommandBuffer> command = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(error, size, @"Unable to create command encoder");
        [encoder setComputePipelineState:dtype == NM_F32 ? c.f32 : c.q4k];
        [encoder setBuffer:w offset:0 atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:y offset:0 atIndex:2];
        const uint32_t shape[] = {rows, cols, batch};
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
        MTLSize grid = MTLSizeMake(((uint64_t)rows + 3) / 4, batch, 1);
        for (uint32_t i = 0; i < repeats; i++)
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return fail(error, size, command.error.localizedDescription ?: @"Metal execution failed");
        if (gpu_seconds) *gpu_seconds = command.GPUEndTime - command.GPUStartTime;
        return 0;
    }
}
