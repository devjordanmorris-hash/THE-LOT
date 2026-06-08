/*
    main_32_predict.m

    32-bit byte carry predictor benchmark.

    Compile:
      xcrun -sdk macosx metal -c Add32PredictKernels.metal -o Add32PredictKernels.air
      xcrun -sdk macosx metallib Add32PredictKernels.air -o Add32PredictKernels.metallib
      clang -O3 main_32_predict.m -framework Foundation -framework Metal -o add32_predict

    Run:
      ./add32_predict
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
        printf("32-bit Byte Carry Predictor GPU Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Add32PredictKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> predictPipe = make_pipe(device, lib, @"bytePredict32AddKernel");
        id<MTLComputePipelineState> prefixPipe = make_pipe(device, lib, @"bytePrefix32AddKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *predictOut = calloc(N, sizeof(uint32_t));
        uint32_t *prefixOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !predictOut || !prefixOut) {
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
        id<MTLBuffer> predictOutBuf = [device newBufferWithBytes:predictOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> prefixOutBuf = [device newBufferWithBytes:prefixOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *predictBuffers = @[aBuf, bBuf, predictOutBuf];
        NSArray *prefixBuffers = @[aBuf, bBuf, prefixOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, predictPipe, predictBuffers, N);
        (void)run_kernel(device, queue, prefixPipe, prefixBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double predictTime = run_kernel(device, queue, predictPipe, predictBuffers, N);
        double prefixTime = run_kernel(device, queue, prefixPipe, prefixBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(predictOut, [predictOutBuf contents], N * sizeof(uint32_t));
        memcpy(prefixOut, [prefixOutBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t predictOk = 0;
        uint64_t prefixOk = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (predictOut[i] == expected) predictOk++;
            if (prefixOut[i] == expected) prefixOk++;
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:        %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Byte-predict 32 add:   %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               predictTime * 1000.0,
               (double)N / predictTime / 1e6,
               predictTime / nativeTime);

        printf("Byte-prefix 32 add:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               prefixTime * 1000.0,
               (double)N / prefixTime / 1e6,
               prefixTime / nativeTime);

        printf("Predict correctness:   %llu/%u\n", (unsigned long long)predictOk, sampleCount);
        printf("Prefix correctness:    %llu/%u\n", (unsigned long long)prefixOk, sampleCount);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u predict=%u prefix=%u\n",
               A[0], B[0], nativeOut[0], predictOut[0], prefixOut[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(predictOut);
        free(prefixOut);
    }

    return 0;
}
