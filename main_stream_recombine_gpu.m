/*
    main_stream_recombine_gpu.m

    Tests whether stream recombine can be made "free":
      native add
      split + add recombine
      split + OR recombine
      split + XOR recombine

    Compile:
      xcrun -sdk macosx metal -c StreamRecombineKernels.metal -o StreamRecombineKernels.air
      xcrun -sdk macosx metallib StreamRecombineKernels.air -o StreamRecombineKernels.metallib
      clang -O3 main_stream_recombine_gpu.m -framework Foundation -framework Metal -o stream_recombine_gpu

    Run:
      ./stream_recombine_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u

typedef struct {
    uint32_t result;
    uint32_t s0;
    uint32_t s1;
    uint32_t s2;
    uint32_t s3;
    uint32_t s4;
    uint32_t overlapMask;
} StreamDebug;

static uint64_t rng_state = 0x123456789abcdefULL;

static inline uint32_t xorshift32(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return (uint32_t)(x >> 32) ^ (uint32_t)x;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double run_kernel(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    NSArray<id<MTLBuffer>> *buffers,
    uint32_t count
) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pipeline];

    for (NSUInteger i = 0; i < [buffers count]; i++) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }

    id<MTLBuffer> nbuf = [device newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    [enc setBuffer:nbuf offset:0 atIndex:[buffers count]];

    NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;

    MTLSize threadsPerGroup = MTLSizeMake(tg, 1, 1);
    MTLSize grid = MTLSizeMake(count, 1, 1);

    double t0 = now_sec();

    [enc dispatchThreads:grid threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return now_sec() - t0;
}

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> device, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];

    if (!fn) {
        fprintf(stderr, "Missing function: %s\n", [name UTF8String]);
        exit(1);
    }

    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];

    if (!p) {
        fprintf(stderr, "Pipeline failed: %s\n", [[err localizedDescription] UTF8String]);
        exit(1);
    }

    return p;
}

static uint64_t check_u32(const uint32_t *expected, const uint32_t *got, uint32_t sample) {
    uint64_t ok = 0;
    for (uint32_t i = 0; i < sample; i++) {
        if (expected[i] == got[i]) ok++;
    }
    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("Stream Recombine GPU Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"StreamRecombineKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> addPipe = make_pipe(device, lib, @"streamAddRecombineKernel");
        id<MTLComputePipelineState> orPipe = make_pipe(device, lib, @"streamOrRecombineKernel");
        id<MTLComputePipelineState> xorPipe = make_pipe(device, lib, @"streamXorRecombineKernel");
        id<MTLComputePipelineState> debugPipe = make_pipe(device, lib, @"streamDebugKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *addOut = calloc(N, sizeof(uint32_t));
        uint32_t *orOut = calloc(N, sizeof(uint32_t));
        uint32_t *xorOut = calloc(N, sizeof(uint32_t));
        StreamDebug *debugOut = calloc(N, sizeof(StreamDebug));

        if (!A || !B || !nativeOut || !addOut || !orOut || !xorOut || !debugOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> nativeBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> addBuf = [device newBufferWithBytes:addOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> orBuf = [device newBufferWithBytes:orOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xorBuf = [device newBufferWithBytes:xorOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> debugBuf = [device newBufferWithBytes:debugOut length:N * sizeof(StreamDebug) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeBuf];
        NSArray *addBuffers = @[aBuf, bBuf, addBuf];
        NSArray *orBuffers = @[aBuf, bBuf, orBuf];
        NSArray *xorBuffers = @[aBuf, bBuf, xorBuf];
        NSArray *debugBuffers = @[aBuf, bBuf, debugBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, addPipe, addBuffers, N);
        (void)run_kernel(device, queue, orPipe, orBuffers, N);
        (void)run_kernel(device, queue, xorPipe, xorBuffers, N);
        (void)run_kernel(device, queue, debugPipe, debugBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double addTime = run_kernel(device, queue, addPipe, addBuffers, N);
        double orTime = run_kernel(device, queue, orPipe, orBuffers, N);
        double xorTime = run_kernel(device, queue, xorPipe, xorBuffers, N);
        double debugTime = run_kernel(device, queue, debugPipe, debugBuffers, N);

        memcpy(nativeOut, [nativeBuf contents], N * sizeof(uint32_t));
        memcpy(addOut, [addBuf contents], N * sizeof(uint32_t));
        memcpy(orOut, [orBuf contents], N * sizeof(uint32_t));
        memcpy(xorOut, [xorBuf contents], N * sizeof(uint32_t));
        memcpy(debugOut, [debugBuf contents], N * sizeof(StreamDebug));

        uint32_t sample = N < 10000u ? N : 10000u;

        uint64_t addOk = check_u32(nativeOut, addOut, sample);
        uint64_t orOk = check_u32(nativeOut, orOut, sample);
        uint64_t xorOk = check_u32(nativeOut, xorOut, sample);

        uint64_t noOverlap = 0;
        for (uint32_t i = 0; i < sample; i++) {
            if (debugOut[i].overlapMask == 0u) noOverlap++;
        }

        printf("\n--- Timing ---\n");
        printf("Native add:        %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Stream + add:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               addTime * 1000.0,
               (double)N / addTime / 1e6,
               addTime / nativeTime);

        printf("Stream + OR:       %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               orTime * 1000.0,
               (double)N / orTime / 1e6,
               orTime / nativeTime);

        printf("Stream + XOR:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               xorTime * 1000.0,
               (double)N / xorTime / 1e6,
               xorTime / nativeTime);

        printf("Stream debug:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               debugTime * 1000.0,
               (double)N / debugTime / 1e6,
               debugTime / nativeTime);

        printf("\n--- Correctness ---\n");
        printf("add recombine:     %llu/%u\n", (unsigned long long)addOk, sample);
        printf("OR recombine:      %llu/%u\n", (unsigned long long)orOk, sample);
        printf("XOR recombine:     %llu/%u\n", (unsigned long long)xorOk, sample);
        printf("no stream overlap: %llu/%u\n", (unsigned long long)noOverlap, sample);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u add=%u or=%u xor=%u overlap=%u\n",
               A[0], B[0], nativeOut[0], addOut[0], orOut[0], xorOut[0], debugOut[0].overlapMask);
        printf("streams: %u %u %u %u %u\n",
               debugOut[0].s0, debugOut[0].s1, debugOut[0].s2, debugOut[0].s3, debugOut[0].s4);

        free(A);
        free(B);
        free(nativeOut);
        free(addOut);
        free(orOut);
        free(xorOut);
        free(debugOut);
    }

    return 0;
}
