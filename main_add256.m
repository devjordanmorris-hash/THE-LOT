/*
    main_add256.m

    256-bit addition benchmark on Metal.

    Representation:
      each 256-bit number = 8 x uint32 words
      word 0 is least significant

    Tests:
      1. native word-carry 256-bit add
      2. split-8 inside each 32-bit word
      3. dual-future 32-bit word precompute, then tiny 8-word carry resolve

    Compile:
      xcrun -sdk macosx metal -c Add256Kernels.metal -o Add256Kernels.air
      xcrun -sdk macosx metallib Add256Kernels.air -o Add256Kernels.metallib
      clang -O3 main_add256.m -framework Foundation -framework Metal -o add256_bench

    Run:
      ./add256_bench
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 500000u
#define WORDS 8u

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

    id<MTLBuffer> nbuf = [device newBufferWithBytes:&count
                                             length:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];

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

static void add256_cpu(const uint32_t *A, const uint32_t *B, uint32_t *O, uint32_t n) {
    for (uint32_t j = 0; j < n; j++) {
        uint64_t carry = 0;
        uint32_t base = j * WORDS;

        for (uint32_t i = 0; i < WORDS; i++) {
            uint64_t sum = (uint64_t)A[base + i] + (uint64_t)B[base + i] + carry;
            O[base + i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

int main(void) {
    @autoreleasepool {
        printf("256-bit Add Split-8 GPU Benchmark\n");
        printf("N=%u 256-bit pairs\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Add256Kernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"add256NativeWordKernel");
        id<MTLComputePipelineState> splitPipe = make_pipe(device, lib, @"add256Split8Kernel");
        id<MTLComputePipelineState> dualPipe = make_pipe(device, lib, @"add256DualFutureKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        size_t totalWords = (size_t)N * WORDS;
        size_t bytes = totalWords * sizeof(uint32_t);

        uint32_t *A = malloc(bytes);
        uint32_t *B = malloc(bytes);
        uint32_t *nativeOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *splitOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *dualOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *cpuOut = calloc(totalWords, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !splitOut || !dualOut || !cpuOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (size_t i = 0; i < totalWords; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> splitOutBuf = [device newBufferWithBytes:splitOut length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dualOutBuf = [device newBufferWithBytes:dualOut length:bytes options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *splitBuffers = @[aBuf, bBuf, splitOutBuf];
        NSArray *dualBuffers = @[aBuf, bBuf, dualOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, splitPipe, splitBuffers, N);
        (void)run_kernel(device, queue, dualPipe, dualBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double splitTime = run_kernel(device, queue, splitPipe, splitBuffers, N);
        double dualTime = run_kernel(device, queue, dualPipe, dualBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], bytes);
        memcpy(splitOut, [splitOutBuf contents], bytes);
        memcpy(dualOut, [dualOutBuf contents], bytes);

        add256_cpu(A, B, cpuOut, 10000u);

        uint64_t nativeOk = 0;
        uint64_t splitOk = 0;
        uint64_t dualOk = 0;

        for (uint32_t j = 0; j < 10000u; j++) {
            int no = 1, so = 1, du = 1;
            uint32_t base = j * WORDS;

            for (uint32_t i = 0; i < WORDS; i++) {
                if (nativeOut[base + i] != cpuOut[base + i]) no = 0;
                if (splitOut[base + i] != cpuOut[base + i]) so = 0;
                if (dualOut[base + i] != cpuOut[base + i]) du = 0;
            }

            nativeOk += no;
            splitOk += so;
            dualOk += du;
        }

        printf("\n--- Results ---\n");
        printf("Native word-carry 256: %.3f ms | %.2f M 256-bit adds/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Split8 per word 256:   %.3f ms | %.2f M 256-bit adds/sec | slowdown %.2fx\n",
               splitTime * 1000.0,
               (double)N / splitTime / 1e6,
               splitTime / nativeTime);

        printf("Dual-future 256:       %.3f ms | %.2f M 256-bit adds/sec | slowdown %.2fx\n",
               dualTime * 1000.0,
               (double)N / dualTime / 1e6,
               dualTime / nativeTime);

        printf("Native correctness:    %llu/10000\n", (unsigned long long)nativeOk);
        printf("Split8 correctness:    %llu/10000\n", (unsigned long long)splitOk);
        printf("Dual correctness:      %llu/10000\n", (unsigned long long)dualOk);

        printf("\nExample low words:\n");
        printf("A0=%u B0=%u native0=%u split0=%u dual0=%u\n",
               A[0], B[0], nativeOut[0], splitOut[0], dualOut[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(splitOut);
        free(dualOut);
        free(cpuOut);
    }

    return 0;
}
