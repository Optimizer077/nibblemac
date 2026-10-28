#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include "metal.h"

@interface NMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> f32;
@property(nonatomic, strong) id<MTLComputePipelineState> q4k;
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
        if (c.f32.threadExecutionWidth != 32 || c.q4k.threadExecutionWidth != 32 ||
            c.f32.maxTotalThreadsPerThreadgroup < 128 || c.q4k.maxTotalThreadsPerThreadgroup < 128) {
            fail(error, size, @"Kernels require 32-lane SIMD groups and 128-thread threadgroups");
            return NULL;
        }
        c.queue = [c.device newCommandQueue];
        if (!c.queue) { fail(error, size, @"Unable to create Metal command queue"); return NULL; }
        return (nm_metal *)CFBridgingRetain(c);
    }
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
