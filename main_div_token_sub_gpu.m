/*
    main_div_token_sub_gpu.m

    Radix-4 divider with token-borrow subtractor experiment.

    Compares:
      - CPU native / %
      - GPU original radix-4 divider
      - GPU token-subtractor radix-4 divider
      - GPU token subtract microbench

    Compile:
      xcrun -sdk macosx metal -c DivTokenSubKernels.metal -o DivTokenSubKernels.air
      xcrun -sdk macosx metallib DivTokenSubKernels.air -o DivTokenSubKernels.metallib
      clang -O3 main_div_token_sub_gpu.m -framework Foundation -framework Metal -o div_token_sub_gpu

    Run:
      ./div_token_sub_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#define N 200000u

typedef struct {
    uint64_t q;
    uint64_t r;
} DivOut;

typedef struct {
    uint64_t q;
    uint64_t r;
    uint32_t waves;
} TokenDivOut;

static uint64_t rng_state = 0x123456789abcdefULL;

static inline uint64_t xrng(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
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

int main(void) {
    @autoreleasepool {
        printf("Radix-4 Divider Token-Subtractor Benchmark\n");
        printf("N=%u\n", N);

        uint64_t *X = malloc(N * sizeof(uint64_t));
        uint64_t *Y = malloc(N * sizeof(uint64_t));
        DivOut *ref = calloc(N, sizeof(DivOut));
        DivOut *orig = calloc(N, sizeof(DivOut));
        TokenDivOut *tok = calloc(N, sizeof(TokenDivOut));
        TokenDivOut *sub = calloc(N, sizeof(TokenDivOut));

        if (!X || !Y || !ref || !orig || !tok || !sub) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        // Use divisor sizes that the old GPU radix-4 path was intended for.
        // Odd/even mixed; nonzero.
        for (uint32_t i = 0; i < N; i++) {
            X[i] = xrng();

            uint64_t y;
            do {
                y = xrng() & ((1ULL << 32) - 1ULL); // small-ish divisor
            } while (y == 0);

            Y[i] = y;
        }

        double c0 = now_sec();
        for (uint32_t i = 0; i < N; i++) {
            ref[i].q = X[i] / Y[i];
            ref[i].r = X[i] % Y[i];
        }
        double c1 = now_sec();

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"DivTokenSubKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> origPipe = make_pipe(device, lib, @"div64OriginalKernel");
        id<MTLComputePipelineState> tokPipe = make_pipe(device, lib, @"div64TokenSubKernel");
        id<MTLComputePipelineState> subPipe = make_pipe(device, lib, @"sub64TokenKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> xBuf = [device newBufferWithBytes:X length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuf = [device newBufferWithBytes:Y length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> origBuf = [device newBufferWithBytes:orig length:N * sizeof(DivOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> tokBuf = [device newBufferWithBytes:tok length:N * sizeof(TokenDivOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> subBuf = [device newBufferWithBytes:sub length:N * sizeof(TokenDivOut) options:MTLResourceStorageModeShared];

        NSArray *origBuffers = @[xBuf, yBuf, origBuf];
        NSArray *tokBuffers = @[xBuf, yBuf, tokBuf];
        NSArray *subBuffers = @[xBuf, yBuf, subBuf];

        // Warmup
        (void)run_kernel(device, queue, origPipe, origBuffers, N);
        (void)run_kernel(device, queue, tokPipe, tokBuffers, N);
        (void)run_kernel(device, queue, subPipe, subBuffers, N);

        double oTime = run_kernel(device, queue, origPipe, origBuffers, N);
        double tTime = run_kernel(device, queue, tokPipe, tokBuffers, N);
        double sTime = run_kernel(device, queue, subPipe, subBuffers, N);

        memcpy(orig, [origBuf contents], N * sizeof(DivOut));
        memcpy(tok, [tokBuf contents], N * sizeof(TokenDivOut));
        memcpy(sub, [subBuf contents], N * sizeof(TokenDivOut));

        uint32_t sample = N < 10000u ? N : 10000u;

        uint64_t origOk = 0;
        uint64_t tokOk = 0;
        uint64_t subOk = 0;
        uint64_t waveSum = 0;
        uint64_t subWaveSum = 0;

        for (uint32_t i = 0; i < sample; i++) {
            if (orig[i].q == ref[i].q && orig[i].r == ref[i].r) origOk++;
            if (tok[i].q == ref[i].q && tok[i].r == ref[i].r) tokOk++;

            uint64_t diff = X[i] - Y[i];
            uint64_t borrow = (X[i] < Y[i]) ? 1ULL : 0ULL;

            if (sub[i].q == diff && sub[i].r == borrow) subOk++;

            waveSum += tok[i].waves;
            subWaveSum += sub[i].waves;
        }

        printf("\n--- Timing ---\n");
        printf("CPU native div/rem:    %.3f ms | %.2f M ops/sec\n",
               (c1 - c0) * 1000.0,
               (double)N / (c1 - c0) / 1e6);

        printf("GPU original radix-4:  %.3f ms | %.2f M ops/sec | ok %llu/%u\n",
               oTime * 1000.0,
               (double)N / oTime / 1e6,
               (unsigned long long)origOk,
               sample);

        printf("GPU token-sub radix-4: %.3f ms | %.2f M ops/sec | slowdown %.2fx vs original | ok %llu/%u\n",
               tTime * 1000.0,
               (double)N / tTime / 1e6,
               tTime / oTime,
               (unsigned long long)tokOk,
               sample);

        printf("GPU token subtract:    %.3f ms | %.2f M ops/sec | ok %llu/%u\n",
               sTime * 1000.0,
               (double)N / sTime / 1e6,
               (unsigned long long)subOk,
               sample);

        printf("\n--- Token stats sample ---\n");
        printf("avg divider token waves:  %.4f\n", (double)waveSum / (double)sample);
        printf("avg subtract token waves: %.4f\n", (double)subWaveSum / (double)sample);

        printf("\nExample:\n");
        printf("X=%" PRIu64 " Y=%" PRIu64 "\n", X[0], Y[0]);
        printf("ref q=%" PRIu64 " r=%" PRIu64 "\n", ref[0].q, ref[0].r);
        printf("orig q=%" PRIu64 " r=%" PRIu64 "\n", orig[0].q, orig[0].r);
        printf("tok  q=%" PRIu64 " r=%" PRIu64 " waves=%u\n", tok[0].q, tok[0].r, tok[0].waves);

        free(X);
        free(Y);
        free(ref);
        free(orig);
        free(tok);
        free(sub);
    }

    return 0;
}
