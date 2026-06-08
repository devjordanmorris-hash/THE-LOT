/*
    main_mul_power_predict.m

    Scales the 32-bit byte carry predictor into:
      1. multiply by shift/add
      2. power by repeated predicted multiply

    Compile:
      xcrun -sdk macosx metal -c MulPowerPredictKernels.metal -o MulPowerPredictKernels.air
      xcrun -sdk macosx metallib MulPowerPredictKernels.air -o MulPowerPredictKernels.metallib
      clang -O3 main_mul_power_predict.m -framework Foundation -framework Metal -o mul_power_predict

    Run:
      ./mul_power_predict
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_MUL 300000u
#define N_POW 100000u

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

static uint32_t native_pow_cpu(uint32_t base, uint32_t exp) {
    uint32_t r = 1u;
    for (uint32_t i = 0; i < exp; i++) r *= base;
    return r;
}

int main(void) {
    @autoreleasepool {
        printf("Predictor Multiply/Power GPU Benchmark\n");
        printf("N_MUL=%u N_POW=%u\n", N_MUL, N_POW);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"MulPowerPredictKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativeMulPipe = make_pipe(device, lib, @"nativeMulKernel");
        id<MTLComputePipelineState> predMulPipe = make_pipe(device, lib, @"predictedMulKernel");
        id<MTLComputePipelineState> nativePowPipe = make_pipe(device, lib, @"nativePowerKernel");
        id<MTLComputePipelineState> predPowPipe = make_pipe(device, lib, @"predictedPowerKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N_MUL * sizeof(uint32_t));
        uint32_t *B = malloc(N_MUL * sizeof(uint32_t));
        uint32_t *nativeMulOut = calloc(N_MUL, sizeof(uint32_t));
        uint32_t *predMulOut = calloc(N_MUL, sizeof(uint32_t));

        uint32_t *BASE = malloc(N_POW * sizeof(uint32_t));
        uint32_t *EXP = malloc(N_POW * sizeof(uint32_t));
        uint32_t *nativePowOut = calloc(N_POW, sizeof(uint32_t));
        uint32_t *predPowOut = calloc(N_POW, sizeof(uint32_t));

        if (!A || !B || !nativeMulOut || !predMulOut || !BASE || !EXP || !nativePowOut || !predPowOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        /*
            Use 16-bit-ish values for multiply so shift/add is not dominated by useless
            high empty work but still exercises carries. Result wraps uint32 exactly.
        */
        for (uint32_t i = 0; i < N_MUL; i++) {
            A[i] = xorshift32() & 0xffffu;
            B[i] = xorshift32() & 0xffffu;
        }

        /*
            Small powers to keep runtime reasonable.
            Still wraps uint32 like native.
        */
        for (uint32_t i = 0; i < N_POW; i++) {
            BASE[i] = xorshift32() & 0xffu;
            EXP[i] = 2u + (xorshift32() % 5u); // 2..6
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N_MUL * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N_MUL * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeMulOutBuf = [device newBufferWithBytes:nativeMulOut length:N_MUL * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> predMulOutBuf = [device newBufferWithBytes:predMulOut length:N_MUL * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> baseBuf = [device newBufferWithBytes:BASE length:N_POW * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> expBuf = [device newBufferWithBytes:EXP length:N_POW * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativePowOutBuf = [device newBufferWithBytes:nativePowOut length:N_POW * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> predPowOutBuf = [device newBufferWithBytes:predPowOut length:N_POW * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeMulBuffers = @[aBuf, bBuf, nativeMulOutBuf];
        NSArray *predMulBuffers = @[aBuf, bBuf, predMulOutBuf];

        NSArray *nativePowBuffers = @[baseBuf, expBuf, nativePowOutBuf];
        NSArray *predPowBuffers = @[baseBuf, expBuf, predPowOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativeMulPipe, nativeMulBuffers, N_MUL);
        (void)run_kernel(device, queue, predMulPipe, predMulBuffers, N_MUL);
        (void)run_kernel(device, queue, nativePowPipe, nativePowBuffers, N_POW);
        (void)run_kernel(device, queue, predPowPipe, predPowBuffers, N_POW);

        double nativeMulTime = run_kernel(device, queue, nativeMulPipe, nativeMulBuffers, N_MUL);
        double predMulTime = run_kernel(device, queue, predMulPipe, predMulBuffers, N_MUL);
        double nativePowTime = run_kernel(device, queue, nativePowPipe, nativePowBuffers, N_POW);
        double predPowTime = run_kernel(device, queue, predPowPipe, predPowBuffers, N_POW);

        memcpy(nativeMulOut, [nativeMulOutBuf contents], N_MUL * sizeof(uint32_t));
        memcpy(predMulOut, [predMulOutBuf contents], N_MUL * sizeof(uint32_t));
        memcpy(nativePowOut, [nativePowOutBuf contents], N_POW * sizeof(uint32_t));
        memcpy(predPowOut, [predPowOutBuf contents], N_POW * sizeof(uint32_t));

        uint32_t mulSample = N_MUL < 10000u ? N_MUL : 10000u;
        uint32_t powSample = N_POW < 10000u ? N_POW : 10000u;

        uint64_t mulOk = 0;
        uint64_t powOk = 0;

        for (uint32_t i = 0; i < mulSample; i++) {
            uint32_t expected = A[i] * B[i];
            if (nativeMulOut[i] == expected && predMulOut[i] == expected) mulOk++;
        }

        for (uint32_t i = 0; i < powSample; i++) {
            uint32_t expected = native_pow_cpu(BASE[i], EXP[i]);
            if (nativePowOut[i] == expected && predPowOut[i] == expected) powOk++;
        }

        printf("\n--- Multiply Results ---\n");
        printf("Native GPU mul:       %.3f ms | %.2f M mul/sec\n",
               nativeMulTime * 1000.0,
               (double)N_MUL / nativeMulTime / 1e6);

        printf("Predicted shift/add:  %.3f ms | %.2f M mul/sec | slowdown %.2fx\n",
               predMulTime * 1000.0,
               (double)N_MUL / predMulTime / 1e6,
               predMulTime / nativeMulTime);

        printf("Multiply correctness: %llu/%u\n",
               (unsigned long long)mulOk, mulSample);

        printf("\n--- Power Results ---\n");
        printf("Native repeated mul:  %.3f ms | %.2f M pow/sec\n",
               nativePowTime * 1000.0,
               (double)N_POW / nativePowTime / 1e6);

        printf("Predicted pow:        %.3f ms | %.2f M pow/sec | slowdown %.2fx\n",
               predPowTime * 1000.0,
               (double)N_POW / predPowTime / 1e6,
               predPowTime / nativePowTime);

        printf("Power correctness:    %llu/%u\n",
               (unsigned long long)powOk, powSample);

        printf("\nExample:\n");
        printf("mul A=%u B=%u native=%u predicted=%u\n",
               A[0], B[0], nativeMulOut[0], predMulOut[0]);

        printf("pow base=%u exp=%u native=%u predicted=%u\n",
               BASE[0], EXP[0], nativePowOut[0], predPowOut[0]);

        free(A);
        free(B);
        free(nativeMulOut);
        free(predMulOut);
        free(BASE);
        free(EXP);
        free(nativePowOut);
        free(predPowOut);
    }

    return 0;
}
