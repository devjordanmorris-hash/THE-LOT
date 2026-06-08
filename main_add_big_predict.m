/*
    main_add_big_predict.m

    1024-bit carry prediction benchmark.

    1024-bit integer = 32 x uint32 words.

    Compile:
      xcrun -sdk macosx metal -c AddBigPredictKernels.metal -o AddBigPredictKernels.air
      xcrun -sdk macosx metallib AddBigPredictKernels.air -o AddBigPredictKernels.metallib
      clang -O3 main_add_big_predict.m -framework Foundation -framework Metal -o add_big_predict

    Run:
      ./add_big_predict
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 100000u
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

static void add_big_cpu(const uint32_t *A, const uint32_t *B, uint32_t *O, uint32_t n) {
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

static uint64_t check_correct(const uint32_t *got, const uint32_t *expected, uint32_t n) {
    uint64_t ok = 0;

    for (uint32_t j = 0; j < n; j++) {
        int good = 1;
        uint32_t base = j * WORDS;

        for (uint32_t i = 0; i < WORDS; i++) {
            if (got[base + i] != expected[base + i]) {
                good = 0;
                break;
            }
        }

        ok += good;
    }

    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("1024-bit Carry Predictor GPU Benchmark\n");
        printf("N=%u 1024-bit pairs, WORDS=%u\n", N, WORDS);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"AddBigPredictKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"addBigNativeWordKernel");
        id<MTLComputePipelineState> predictPipe = make_pipe(device, lib, @"addBigCarryPredictKernel");
        id<MTLComputePipelineState> prefixPipe = make_pipe(device, lib, @"addBigPrefixPredictKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        size_t totalWords = (size_t)N * WORDS;
        size_t bytes = totalWords * sizeof(uint32_t);

        uint32_t *A = malloc(bytes);
        uint32_t *B = malloc(bytes);
        uint32_t *nativeOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *predictOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *prefixOut = calloc(totalWords, sizeof(uint32_t));
        uint32_t *cpuOut = calloc(totalWords, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !predictOut || !prefixOut || !cpuOut) {
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
        id<MTLBuffer> predictOutBuf = [device newBufferWithBytes:predictOut length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> prefixOutBuf = [device newBufferWithBytes:prefixOut length:bytes options:MTLResourceStorageModeShared];

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

        memcpy(nativeOut, [nativeOutBuf contents], bytes);
        memcpy(predictOut, [predictOutBuf contents], bytes);
        memcpy(prefixOut, [prefixOutBuf contents], bytes);

        uint32_t sample = 10000u;
        if (sample > N) sample = N;

        add_big_cpu(A, B, cpuOut, sample);

        uint64_t nativeOk = check_correct(nativeOut, cpuOut, sample);
        uint64_t predictOk = check_correct(predictOut, cpuOut, sample);
        uint64_t prefixOk = check_correct(prefixOut, cpuOut, sample);

        printf("\n--- Results ---\n");
        printf("Native word-carry 1024: %.3f ms | %.2f M adds/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Carry-predict 1024:     %.3f ms | %.2f M adds/sec | slowdown %.2fx\n",
               predictTime * 1000.0,
               (double)N / predictTime / 1e6,
               predictTime / nativeTime);

        printf("Prefix-predict 1024:    %.3f ms | %.2f M adds/sec | slowdown %.2fx\n",
               prefixTime * 1000.0,
               (double)N / prefixTime / 1e6,
               prefixTime / nativeTime);

        printf("\n--- Correctness ---\n");
        printf("Native:  %llu/%u\n", (unsigned long long)nativeOk, sample);
        printf("Predict: %llu/%u\n", (unsigned long long)predictOk, sample);
        printf("Prefix:  %llu/%u\n", (unsigned long long)prefixOk, sample);

        printf("\nExample low words:\n");
        printf("A0=%u B0=%u native0=%u predict0=%u prefix0=%u\n",
               A[0], B[0], nativeOut[0], predictOut[0], prefixOut[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(predictOut);
        free(prefixOut);
        free(cpuOut);
    }

    return 0;
}
