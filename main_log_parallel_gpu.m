/*
    main_log_parallel_gpu.m

    Log2 as additive correction terms, parallel GPU test.

    Tests:
      - native Metal log2
      - serial greedy additive log2
      - parallel per-term additive log2 + token reduction
      - debug accuracy for parallel version

    Compile:
      xcrun -sdk macosx metal -c LogParallelKernels.metal -o LogParallelKernels.air
      xcrun -sdk macosx metallib LogParallelKernels.air -o LogParallelKernels.metallib
      clang -O3 main_log_parallel_gpu.m -framework Foundation -framework Metal -o log_parallel_gpu

    Run:
      ./log_parallel_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 1000000u
#define GROUP_SIZE 32u

typedef struct {
    float approx;
    float exact;
    float absErr;
    uint32_t termCount;
} LogDebugOut;

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

static double run_threads_kernel(
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

static double run_groups_kernel(
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

    MTLSize threadsPerGroup = MTLSizeMake(GROUP_SIZE, 1, 1);
    MTLSize groups = MTLSizeMake(count, 1, 1);

    double t0 = now_sec();

    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return now_sec() - t0;
}

static void summarize_accuracy(const char *label, const float *ref, const float *out, uint32_t sample) {
    double mean = 0.0;
    double maxe = 0.0;
    uint64_t under1e3 = 0;
    uint64_t under1e2 = 0;
    uint64_t under1e1 = 0;

    for (uint32_t i = 0; i < sample; i++) {
        double e = fabs((double)ref[i] - (double)out[i]);
        mean += e;
        if (e > maxe) maxe = e;
        if (e < 1e-3) under1e3++;
        if (e < 1e-2) under1e2++;
        if (e < 1e-1) under1e1++;
    }

    mean /= (double)sample;

    printf("%s mean_err %.8g max_err %.8g | <1e-3 %llu/%u | <1e-2 %llu/%u | <1e-1 %llu/%u\n",
           label,
           mean,
           maxe,
           (unsigned long long)under1e3, sample,
           (unsigned long long)under1e2, sample,
           (unsigned long long)under1e1, sample);
}

int main(void) {
    @autoreleasepool {
        printf("Parallel Additive Log2 GPU Benchmark\n");
        printf("N=%u, GROUP_SIZE=%u\n", N, GROUP_SIZE);

        float *X = malloc(N * sizeof(float));
        float *nativeOut = calloc(N, sizeof(float));
        float *serialOut = calloc(N, sizeof(float));
        float *parallelOut = calloc(N, sizeof(float));
        LogDebugOut *debugOut = calloc(N, sizeof(LogDebugOut));

        if (!X || !nativeOut || !serialOut || !parallelOut || !debugOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            float u = (float)xorshift32() / (float)UINT32_MAX;
            // log-uniform-ish range [1e-6, 1e6]
            float expv = -6.0f + 12.0f * u;
            X[i] = powf(10.0f, expv);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"LogParallelKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"log2NativeKernel");
        id<MTLComputePipelineState> serialPipe = make_pipe(device, lib, @"log2AddSerialKernel");
        id<MTLComputePipelineState> parallelPipe = make_pipe(device, lib, @"log2AddParallelTermsKernel");
        id<MTLComputePipelineState> debugPipe = make_pipe(device, lib, @"log2AddParallelDebugKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> xBuf = [device newBufferWithBytes:X length:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> serialBuf = [device newBufferWithBytes:serialOut length:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> parallelBuf = [device newBufferWithBytes:parallelOut length:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> debugBuf = [device newBufferWithBytes:debugOut length:N * sizeof(LogDebugOut) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[xBuf, nativeBuf];
        NSArray *serialBuffers = @[xBuf, serialBuf];
        NSArray *parallelBuffers = @[xBuf, parallelBuf];
        NSArray *debugBuffers = @[xBuf, debugBuf];

        // warmup
        (void)run_threads_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_threads_kernel(device, queue, serialPipe, serialBuffers, N);
        (void)run_groups_kernel(device, queue, parallelPipe, parallelBuffers, N);
        (void)run_groups_kernel(device, queue, debugPipe, debugBuffers, N);

        double nativeTime = run_threads_kernel(device, queue, nativePipe, nativeBuffers, N);
        double serialTime = run_threads_kernel(device, queue, serialPipe, serialBuffers, N);
        double parallelTime = run_groups_kernel(device, queue, parallelPipe, parallelBuffers, N);
        double debugTime = run_groups_kernel(device, queue, debugPipe, debugBuffers, N);

        memcpy(nativeOut, [nativeBuf contents], N * sizeof(float));
        memcpy(serialOut, [serialBuf contents], N * sizeof(float));
        memcpy(parallelOut, [parallelBuf contents], N * sizeof(float));
        memcpy(debugOut, [debugBuf contents], N * sizeof(LogDebugOut));

        uint32_t sample = N < 10000u ? N : 10000u;

        printf("\n--- Timing ---\n");
        printf("Native log2:             %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Serial additive log2:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               serialTime * 1000.0,
               (double)N / serialTime / 1e6,
               serialTime / nativeTime);

        printf("Parallel term log2:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               parallelTime * 1000.0,
               (double)N / parallelTime / 1e6,
               parallelTime / nativeTime);

        printf("Parallel debug:          %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               debugTime * 1000.0,
               (double)N / debugTime / 1e6,
               debugTime / nativeTime);

        printf("\n--- Accuracy vs native Metal log2 ---\n");
        summarize_accuracy("Serial additive:", nativeOut, serialOut, sample);
        summarize_accuracy("Parallel terms: ", nativeOut, parallelOut, sample);

        double meanDbgErr = 0.0;
        double maxDbgErr = 0.0;
        double meanTerms = 0.0;

        for (uint32_t i = 0; i < sample; i++) {
            meanDbgErr += debugOut[i].absErr;
            if (debugOut[i].absErr > maxDbgErr) maxDbgErr = debugOut[i].absErr;
            meanTerms += debugOut[i].termCount;
        }

        meanDbgErr /= sample;
        meanTerms /= sample;

        printf("\n--- Parallel debug sample ---\n");
        printf("mean abs err %.8g | max abs err %.8g | avg accepted terms %.3f\n",
               meanDbgErr, maxDbgErr, meanTerms);

        printf("\nExample:\n");
        printf("x=%g native=%g serial=%g parallel=%g\n",
               X[0], nativeOut[0], serialOut[0], parallelOut[0]);

        printf("\nNote:\n");
        printf("Parallel term version is not greedy; it tests hardware-shaped parallel term generation.\n");
        printf("If accuracy is poor, next step is block-prefix/product correction, not abandoning the idea.\n");

        free(X);
        free(nativeOut);
        free(serialOut);
        free(parallelOut);
        free(debugOut);
    }

    return 0;
}
