/*
    main_power_wide_add.m

    Converts powers into long 1024-bit repeated-add workloads.

    This is NOT meant to beat native multiply/pow.
    It tests whether the winning 1024-bit carry predictor helps when
    power is expressed as many wide additions.

    Compile:
      xcrun -sdk macosx metal -c PowerWideAddKernels.metal -o PowerWideAddKernels.air
      xcrun -sdk macosx metallib PowerWideAddKernels.air -o PowerWideAddKernels.metallib
      clang -O3 main_power_wide_add.m -framework Foundation -framework Metal -o power_wide_add

    Run:
      ./power_wide_add
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 20000u
#define WORDS 32u

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

static void wide_mul_small_cpu(uint32_t *out, const uint32_t *a, uint32_t multiplier) {
    memset(out, 0, WORDS * sizeof(uint32_t));

    for (uint32_t m = 0; m < multiplier; m++) {
        uint64_t carry = 0;

        for (uint32_t i = 0; i < WORDS; i++) {
            uint64_t sum = (uint64_t)out[i] + (uint64_t)a[i] + carry;
            out[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

static void power_wide_cpu(uint32_t base, uint32_t exp, uint32_t *out) {
    uint32_t result[WORDS];
    uint32_t tmp[WORDS];

    memset(result, 0, sizeof(result));
    result[0] = 1;

    for (uint32_t e = 0; e < exp; e++) {
        wide_mul_small_cpu(tmp, result, base);
        memcpy(result, tmp, sizeof(result));
    }

    memcpy(out, result, WORDS * sizeof(uint32_t));
}

static int eq_words(const uint32_t *a, const uint32_t *b) {
    for (uint32_t i = 0; i < WORDS; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main(void) {
    @autoreleasepool {
        printf("Power as Long 1024-bit Add Benchmark\n");
        printf("N=%u, WORDS=%u\n", N, WORDS);
        printf("base range 2..15, exponent range 2..8\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"PowerWideAddKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"powerWideNativeAddKernel");
        id<MTLComputePipelineState> predictPipe = make_pipe(device, lib, @"powerWidePredictAddKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *BASE = malloc(N * sizeof(uint32_t));
        uint32_t *EXP = malloc(N * sizeof(uint32_t));

        size_t outWords = (size_t)N * WORDS;
        size_t outBytes = outWords * sizeof(uint32_t);

        uint32_t *nativeOut = calloc(outWords, sizeof(uint32_t));
        uint32_t *predictOut = calloc(outWords, sizeof(uint32_t));

        if (!BASE || !EXP || !nativeOut || !predictOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            BASE[i] = 2u + (xorshift32() % 14u); // 2..15
            EXP[i] = 2u + (xorshift32() % 7u);   // 2..8
        }

        id<MTLBuffer> baseBuf = [device newBufferWithBytes:BASE length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> expBuf = [device newBufferWithBytes:EXP length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> predictOutBuf = [device newBufferWithBytes:predictOut length:outBytes options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[baseBuf, expBuf, nativeOutBuf];
        NSArray *predictBuffers = @[baseBuf, expBuf, predictOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, predictPipe, predictBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double predictTime = run_kernel(device, queue, predictPipe, predictBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], outBytes);
        memcpy(predictOut, [predictOutBuf contents], outBytes);

        uint32_t sample = N < 2000u ? N : 2000u;
        uint64_t nativeOk = 0;
        uint64_t predictOk = 0;
        uint64_t matchEachOther = 0;

        uint32_t cpu[WORDS];

        for (uint32_t j = 0; j < sample; j++) {
            power_wide_cpu(BASE[j], EXP[j], cpu);

            uint32_t baseIdx = j * WORDS;

            nativeOk += eq_words(&nativeOut[baseIdx], cpu);
            predictOk += eq_words(&predictOut[baseIdx], cpu);
            matchEachOther += eq_words(&nativeOut[baseIdx], &predictOut[baseIdx]);
        }

        printf("\n--- Results ---\n");
        printf("Native-wide-add power:   %.3f ms | %.2f K powers/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e3);

        printf("Predict-wide-add power:  %.3f ms | %.2f K powers/sec | slowdown %.2fx\n",
               predictTime * 1000.0,
               (double)N / predictTime / 1e3,
               predictTime / nativeTime);

        printf("\n--- Correctness ---\n");
        printf("Native vs CPU:   %llu/%u\n", (unsigned long long)nativeOk, sample);
        printf("Predict vs CPU:  %llu/%u\n", (unsigned long long)predictOk, sample);
        printf("Native/Predict:  %llu/%u\n", (unsigned long long)matchEachOther, sample);

        printf("\nExample:\n");
        printf("base=%u exp=%u lowword native=%u predict=%u\n",
               BASE[0], EXP[0], nativeOut[0], predictOut[0]);

        free(BASE);
        free(EXP);
        free(nativeOut);
        free(predictOut);
    }

    return 0;
}
