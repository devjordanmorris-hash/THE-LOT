/*
    main_bitwise.m

    Metal benchmark comparing:
      1. native GPU add
      2. branchless/bitwise transform with debug outputs
      3. branchless/bitwise transform without mask/residual output

    Commands:
      xcrun -sdk macosx metal -c Base4Kernels_Bitwise.metal -o Base4Kernels_Bitwise.air
      xcrun -sdk macosx metallib Base4Kernels_Bitwise.air -o Base4Kernels_Bitwise.metallib
      clang -O3 main_bitwise.m -framework Foundation -framework Metal -o base4_metal_bitwise
      ./base4_metal_bitwise
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u

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

    id<MTLBuffer> nbuf =
        [device newBufferWithBytes:&count
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

    id<MTLComputePipelineState> p =
        [device newComputePipelineStateWithFunction:fn error:&err];

    if (!p) {
        fprintf(stderr, "Pipeline failed: %s\n", [[err localizedDescription] UTF8String]);
        exit(1);
    }

    return p;
}

int main(void) {
    @autoreleasepool {
        printf("Base-4 Metal Bitwise Transform Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4Kernels_Bitwise.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe =
            make_pipe(device, lib, @"nativeAddKernel");

        id<MTLComputePipelineState> bitwisePipe =
            make_pipe(device, lib, @"base4TransformAddBitwiseKernel");

        id<MTLComputePipelineState> noDebugPipe =
            make_pipe(device, lib, @"base4TransformAddBitwiseNoDebugKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *bitwiseOut = calloc(N, sizeof(uint32_t));
        uint32_t *noDebugOut = calloc(N, sizeof(uint32_t));
        uint32_t *masks = calloc(N, sizeof(uint32_t));
        uint32_t *residuals = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !bitwiseOut || !noDebugOut || !masks || !residuals) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bitwiseOutBuf = [device newBufferWithBytes:bitwiseOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> noDebugOutBuf = [device newBufferWithBytes:noDebugOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [device newBufferWithBytes:masks length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> residualBuf = [device newBufferWithBytes:residuals length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *bitwiseBuffers = @[aBuf, bBuf, bitwiseOutBuf, maskBuf, residualBuf];
        NSArray *noDebugBuffers = @[aBuf, bBuf, noDebugOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, bitwisePipe, bitwiseBuffers, N);
        (void)run_kernel(device, queue, noDebugPipe, noDebugBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double bitwiseTime = run_kernel(device, queue, bitwisePipe, bitwiseBuffers, N);
        double noDebugTime = run_kernel(device, queue, noDebugPipe, noDebugBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(bitwiseOut, [bitwiseOutBuf contents], N * sizeof(uint32_t));
        memcpy(noDebugOut, [noDebugOutBuf contents], N * sizeof(uint32_t));
        memcpy(masks, [maskBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t ok = 0;
        uint64_t noDebugOk = 0;
        uint64_t maskPop = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];
            if (nativeOut[i] == expected && bitwiseOut[i] == expected) ok++;
            if (noDebugOut[i] == expected) noDebugOk++;
        }

        for (uint32_t i = 0; i < N; i++) {
            maskPop += __builtin_popcount(masks[i]);
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:       %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Bitwise transform:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               bitwiseTime * 1000.0,
               (double)N / bitwiseTime / 1e6,
               bitwiseTime / nativeTime);

        printf("No-debug transform:   %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               noDebugTime * 1000.0,
               (double)N / noDebugTime / 1e6,
               noDebugTime / nativeTime);

        printf("Correct sample:       %llu/%u\n", (unsigned long long)ok, sampleCount);
        printf("No-debug sample:      %llu/%u\n", (unsigned long long)noDebugOk, sampleCount);
        printf("Avg mask popcount:    %.4f / 16 base4 digits\n", (double)maskPop / (double)N);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u bitwise=%u mask=%u\n",
               A[0], B[0], nativeOut[0], bitwiseOut[0], masks[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(bitwiseOut);
        free(noDebugOut);
        free(masks);
        free(residuals);
    }

    return 0;
}
