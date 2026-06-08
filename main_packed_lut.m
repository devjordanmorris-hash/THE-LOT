/*
    main_packed_lut.m

    Packed dual-carry LUT split-chunk GPU add benchmark.

    Compile:
      xcrun -sdk macosx metal -c Base4Kernels_PackedLUT.metal -o Base4Kernels_PackedLUT.air
      xcrun -sdk macosx metallib Base4Kernels_PackedLUT.air -o Base4Kernels_PackedLUT.metallib
      clang -O3 main_packed_lut.m -framework Foundation -framework Metal -o base4_packed_lut

    Run:
      ./base4_packed_lut
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define LUT_SIZE 65536u

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

static uint32_t pack_lut_entry(uint32_t a, uint32_t b) {
    uint32_t sum0 = a + b;
    uint32_t r0 = sum0 & 0xffu;
    uint32_t c0 = (sum0 >> 8) & 1u;

    uint32_t sum1 = a + b + 1u;
    uint32_t r1 = sum1 & 0xffu;
    uint32_t c1 = (sum1 >> 8) & 1u;

    return (r0 & 0xffu) | ((c0 & 1u) << 8) | ((r1 & 0xffu) << 9) | ((c1 & 1u) << 17);
}

static void build_lut(uint32_t *lut) {
    for (uint32_t a = 0; a < 256; a++) {
        for (uint32_t b = 0; b < 256; b++) {
            lut[(a << 8) | b] = pack_lut_entry(a, b);
        }
    }
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

int main(void) {
    @autoreleasepool {
        printf("Packed Dual-Carry LUT Add Benchmark\n");
        printf("N=%u LUT=%u entries %.2f KB\n", N, LUT_SIZE, (double)(LUT_SIZE * sizeof(uint32_t)) / 1024.0);

        uint32_t *lut = malloc(LUT_SIZE * sizeof(uint32_t));
        if (!lut) {
            fprintf(stderr, "LUT allocation failed\n");
            return 1;
        }
        build_lut(lut);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4Kernels_PackedLUT.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> splitPipe = make_pipe(device, lib, @"splitChunkAddKernel");
        id<MTLComputePipelineState> lutPipe = make_pipe(device, lib, @"packedDualCarryLUTAddKernel");
        id<MTLComputePipelineState> lutBranchlessPipe = make_pipe(device, lib, @"packedDualCarryLUTAddBranchlessKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *splitOut = calloc(N, sizeof(uint32_t));
        uint32_t *lutOut = calloc(N, sizeof(uint32_t));
        uint32_t *lutBranchlessOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !splitOut || !lutOut || !lutBranchlessOut) {
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
        id<MTLBuffer> splitOutBuf = [device newBufferWithBytes:splitOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lutOutBuf = [device newBufferWithBytes:lutOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lutBranchlessOutBuf = [device newBufferWithBytes:lutBranchlessOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lutBuf = [device newBufferWithBytes:lut length:LUT_SIZE * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *splitBuffers = @[aBuf, bBuf, splitOutBuf];
        NSArray *lutBuffers = @[aBuf, bBuf, lutOutBuf, lutBuf];
        NSArray *lutBranchlessBuffers = @[aBuf, bBuf, lutBranchlessOutBuf, lutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, splitPipe, splitBuffers, N);
        (void)run_kernel(device, queue, lutPipe, lutBuffers, N);
        (void)run_kernel(device, queue, lutBranchlessPipe, lutBranchlessBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double splitTime = run_kernel(device, queue, splitPipe, splitBuffers, N);
        double lutTime = run_kernel(device, queue, lutPipe, lutBuffers, N);
        double lutBranchlessTime = run_kernel(device, queue, lutBranchlessPipe, lutBranchlessBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(splitOut, [splitOutBuf contents], N * sizeof(uint32_t));
        memcpy(lutOut, [lutOutBuf contents], N * sizeof(uint32_t));
        memcpy(lutBranchlessOut, [lutBranchlessOutBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t splitOk = 0;
        uint64_t lutOk = 0;
        uint64_t lutBranchlessOk = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (splitOut[i] == expected) splitOk++;
            if (lutOut[i] == expected) lutOk++;
            if (lutBranchlessOut[i] == expected) lutBranchlessOk++;
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:        %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Split chunk add:       %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               splitTime * 1000.0,
               (double)N / splitTime / 1e6,
               splitTime / nativeTime);

        printf("Packed LUT add:        %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               lutTime * 1000.0,
               (double)N / lutTime / 1e6,
               lutTime / nativeTime);

        printf("Packed LUT branchless: %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               lutBranchlessTime * 1000.0,
               (double)N / lutBranchlessTime / 1e6,
               lutBranchlessTime / nativeTime);

        printf("Split correctness:     %llu/%u\n", (unsigned long long)splitOk, sampleCount);
        printf("LUT correctness:       %llu/%u\n", (unsigned long long)lutOk, sampleCount);
        printf("LUT branchless corr:   %llu/%u\n", (unsigned long long)lutBranchlessOk, sampleCount);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u split=%u lut=%u lutB=%u\n",
               A[0], B[0], nativeOut[0], splitOut[0], lutOut[0], lutBranchlessOut[0]);

        free(lut);
        free(A);
        free(B);
        free(nativeOut);
        free(splitOut);
        free(lutOut);
        free(lutBranchlessOut);
    }

    return 0;
}
