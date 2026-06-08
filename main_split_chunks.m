/*
    main_split_chunks.m

    GPU chunk-split addition experiment.

    Tests:
      1. Native GPU add
      2. Split 32-bit add into four 8-bit chunks
      3. Shifted split add, then shift result back

    Compile:
      xcrun -sdk macosx metal -c Base4Kernels_SplitChunks.metal -o Base4Kernels_SplitChunks.air
      xcrun -sdk macosx metallib Base4Kernels_SplitChunks.air -o Base4Kernels_SplitChunks.metallib
      clang -O3 main_split_chunks.m -framework Foundation -framework Metal -o base4_split_chunks

    Run:
      ./base4_split_chunks
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define SHIFT 2u

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

static double run_shifted_kernel(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    NSArray<id<MTLBuffer>> *buffers,
    uint32_t count,
    uint32_t shift
) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pipeline];

    for (NSUInteger i = 0; i < [buffers count]; i++) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }

    id<MTLBuffer> nbuf = [device newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sbuf = [device newBufferWithBytes:&shift length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    [enc setBuffer:nbuf offset:0 atIndex:[buffers count]];
    [enc setBuffer:sbuf offset:0 atIndex:[buffers count] + 1];

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

int main(void) {
    @autoreleasepool {
        printf("GPU Split-Chunk Add Experiment\n");
        printf("N=%u SHIFT=%u\n", N, SHIFT);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4Kernels_SplitChunks.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> splitPipe = make_pipe(device, lib, @"splitChunkAddKernel");
        id<MTLComputePipelineState> shiftedPipe = make_pipe(device, lib, @"shiftedSplitChunkAddKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *splitOut = calloc(N, sizeof(uint32_t));
        uint32_t *shiftedOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !splitOut || !shiftedOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        uint32_t maxv = 0xffffffffu >> (SHIFT + 1u); // avoid shifted add overflow

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32() & maxv;
            B[i] = xorshift32() & maxv;
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> splitOutBuf = [device newBufferWithBytes:splitOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> shiftedOutBuf = [device newBufferWithBytes:shiftedOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *splitBuffers = @[aBuf, bBuf, splitOutBuf];
        NSArray *shiftedBuffers = @[aBuf, bBuf, shiftedOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, splitPipe, splitBuffers, N);
        (void)run_shifted_kernel(device, queue, shiftedPipe, shiftedBuffers, N, SHIFT);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double splitTime = run_kernel(device, queue, splitPipe, splitBuffers, N);
        double shiftedTime = run_shifted_kernel(device, queue, shiftedPipe, shiftedBuffers, N, SHIFT);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(splitOut, [splitOutBuf contents], N * sizeof(uint32_t));
        memcpy(shiftedOut, [shiftedOutBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t splitOk = 0;
        uint64_t shiftedOk = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (splitOut[i] == expected) splitOk++;
            if (shiftedOut[i] == expected) shiftedOk++;
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:       %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Split chunk add:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               splitTime * 1000.0,
               (double)N / splitTime / 1e6,
               splitTime / nativeTime);

        printf("Shifted split add:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               shiftedTime * 1000.0,
               (double)N / shiftedTime / 1e6,
               shiftedTime / nativeTime);

        printf("Split correctness:    %llu/%u\n", (unsigned long long)splitOk, sampleCount);
        printf("Shift correctness:    %llu/%u\n", (unsigned long long)shiftedOk, sampleCount);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u split=%u shifted=%u\n",
               A[0], B[0], nativeOut[0], splitOut[0], shiftedOut[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(splitOut);
        free(shiftedOut);
    }

    return 0;
}
