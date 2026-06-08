/*
    main_partial_accum_predict.m

    Wide partial-product accumulation benchmark.

    This tests whether carry-predict wide-add helps in a multiply-shaped
    workload:
      many shifted partial products accumulated into a 1024-bit result.

    Compile:
      xcrun -sdk macosx metal -c PartialAccumPredictKernels.metal -o PartialAccumPredictKernels.air
      xcrun -sdk macosx metallib PartialAccumPredictKernels.air -o PartialAccumPredictKernels.metallib
      clang -O3 main_partial_accum_predict.m -framework Foundation -framework Metal -o partial_accum_predict

    Run:
      ./partial_accum_predict
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 50000u
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

static uint64_t check_match(const uint32_t *a, const uint32_t *b, uint32_t n) {
    uint64_t ok = 0;

    for (uint32_t j = 0; j < n; j++) {
        int good = 1;
        uint32_t base = j * WORDS;

        for (uint32_t i = 0; i < WORDS; i++) {
            if (a[base + i] != b[base + i]) {
                good = 0;
                break;
            }
        }

        ok += good;
    }

    return ok;
}

static void run_pair(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const char *label,
    id<MTLComputePipelineState> nativePipe,
    id<MTLComputePipelineState> predictPipe,
    NSArray *nativeBuffers,
    NSArray *predictBuffers,
    uint32_t *nativeOut,
    uint32_t *predictOut,
    id<MTLBuffer> nativeOutBuf,
    id<MTLBuffer> predictOutBuf,
    size_t outBytes
) {
    // Warmup
    (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
    (void)run_kernel(device, queue, predictPipe, predictBuffers, N);

    double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
    double predictTime = run_kernel(device, queue, predictPipe, predictBuffers, N);

    memcpy(nativeOut, [nativeOutBuf contents], outBytes);
    memcpy(predictOut, [predictOutBuf contents], outBytes);

    uint32_t sample = N < 10000u ? N : 10000u;
    uint64_t match = check_match(nativeOut, predictOut, sample);

    printf("%-24s | native %7.3f ms %8.2f K/s | predict %7.3f ms %8.2f K/s | speed %.2fx | match %llu/%u\n",
           label,
           nativeTime * 1000.0,
           (double)N / nativeTime / 1e3,
           predictTime * 1000.0,
           (double)N / predictTime / 1e3,
           nativeTime / predictTime,
           (unsigned long long)match,
           sample);
}

int main(void) {
    @autoreleasepool {
        printf("Partial Product Accumulation Predictor Benchmark\n");
        printf("N=%u, WORDS=%u, TERMS=32\n", N, WORDS);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"PartialAccumPredictKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> sparseNative = make_pipe(device, lib, @"partialAccumNativeKernel");
        id<MTLComputePipelineState> sparsePredict = make_pipe(device, lib, @"partialAccumPredictKernel");
        id<MTLComputePipelineState> denseNative = make_pipe(device, lib, @"densePartialAccumNativeKernel");
        id<MTLComputePipelineState> densePredict = make_pipe(device, lib, @"densePartialAccumPredictKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));

        size_t outWords = (size_t)N * WORDS;
        size_t outBytes = outWords * sizeof(uint32_t);

        uint32_t *nativeOut = calloc(outWords, sizeof(uint32_t));
        uint32_t *predictOut = calloc(outWords, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !predictOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> predictOutBuf = [device newBufferWithBytes:predictOut length:outBytes options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *predictBuffers = @[aBuf, bBuf, predictOutBuf];

        printf("--- Results ---\n");

        run_pair(
            device, queue,
            "sparse shifted terms",
            sparseNative, sparsePredict,
            nativeBuffers, predictBuffers,
            nativeOut, predictOut,
            nativeOutBuf, predictOutBuf,
            outBytes
        );

        run_pair(
            device, queue,
            "dense 4-word terms",
            denseNative, densePredict,
            nativeBuffers, predictBuffers,
            nativeOut, predictOut,
            nativeOutBuf, predictOutBuf,
            outBytes
        );

        printf("\nExample low word: native=%u predict=%u\n", nativeOut[0], predictOut[0]);

        free(A);
        free(B);
        free(nativeOut);
        free(predictOut);
    }

    return 0;
}
