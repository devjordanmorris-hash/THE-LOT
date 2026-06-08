/*
    main_div_shifted_token_gpu.m

    Tests whether base-4-friendly shifting/normalization helps token-sub division.

    Compile:
      xcrun -sdk macosx metal -c DivShiftedTokenKernels.metal -o DivShiftedTokenKernels.air
      xcrun -sdk macosx metallib DivShiftedTokenKernels.air -o DivShiftedTokenKernels.metallib
      clang -O3 main_div_shifted_token_gpu.m -framework Foundation -framework Metal -o div_shifted_token_gpu

    Run:
      ./div_shifted_token_gpu
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

static void summarize_token(
    const char *label,
    double t,
    const TokenDivOut *out,
    const DivOut *ref,
    uint32_t sample,
    double refTime
) {
    uint64_t ok = 0;
    uint64_t waves = 0;

    for (uint32_t i = 0; i < sample; i++) {
        if (out[i].q == ref[i].q && out[i].r == ref[i].r) ok++;
        waves += out[i].waves;
    }

    printf("%-28s %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u | avg waves %.2f\n",
           label,
           t * 1000.0,
           (double)N / t / 1e6,
           t / refTime,
           (unsigned long long)ok,
           sample,
           (double)waves / (double)sample);
}

int main(void) {
    @autoreleasepool {
        printf("Shifted / Base4-Normalized Token Division Benchmark\n");
        printf("N=%u\n", N);

        uint64_t *X = malloc(N * sizeof(uint64_t));
        uint64_t *Y = malloc(N * sizeof(uint64_t));
        DivOut *ref = calloc(N, sizeof(DivOut));
        DivOut *orig = calloc(N, sizeof(DivOut));
        TokenDivOut *norm = calloc(N, sizeof(TokenDivOut));
        TokenDivOut *compact = calloc(N, sizeof(TokenDivOut));
        TokenDivOut *sub = calloc(N, sizeof(TokenDivOut));

        if (!X || !Y || !ref || !orig || !norm || !compact || !sub) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            X[i] = xrng();

            uint64_t y;
            do {
                y = xrng() & ((1ULL << 32) - 1ULL);
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
        NSURL *libURL = [NSURL fileURLWithPath:@"DivShiftedTokenKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> origPipe = make_pipe(device, lib, @"div64OriginalKernel");
        id<MTLComputePipelineState> normPipe = make_pipe(device, lib, @"div64TokenNormKernel");
        id<MTLComputePipelineState> compactPipe = make_pipe(device, lib, @"div64TokenBase4CompactKernel");
        id<MTLComputePipelineState> subPipe = make_pipe(device, lib, @"sub64TokenKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> xBuf = [device newBufferWithBytes:X length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuf = [device newBufferWithBytes:Y length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> origBuf = [device newBufferWithBytes:orig length:N * sizeof(DivOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> normBuf = [device newBufferWithBytes:norm length:N * sizeof(TokenDivOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> compactBuf = [device newBufferWithBytes:compact length:N * sizeof(TokenDivOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> subBuf = [device newBufferWithBytes:sub length:N * sizeof(TokenDivOut) options:MTLResourceStorageModeShared];

        NSArray *origBuffers = @[xBuf, yBuf, origBuf];
        NSArray *normBuffers = @[xBuf, yBuf, normBuf];
        NSArray *compactBuffers = @[xBuf, yBuf, compactBuf];
        NSArray *subBuffers = @[xBuf, yBuf, subBuf];

        // Warmup
        (void)run_kernel(device, queue, origPipe, origBuffers, N);
        (void)run_kernel(device, queue, normPipe, normBuffers, N);
        (void)run_kernel(device, queue, compactPipe, compactBuffers, N);
        (void)run_kernel(device, queue, subPipe, subBuffers, N);

        double oTime = run_kernel(device, queue, origPipe, origBuffers, N);
        double nTime = run_kernel(device, queue, normPipe, normBuffers, N);
        double bTime = run_kernel(device, queue, compactPipe, compactBuffers, N);
        double sTime = run_kernel(device, queue, subPipe, subBuffers, N);

        memcpy(orig, [origBuf contents], N * sizeof(DivOut));
        memcpy(norm, [normBuf contents], N * sizeof(TokenDivOut));
        memcpy(compact, [compactBuf contents], N * sizeof(TokenDivOut));
        memcpy(sub, [subBuf contents], N * sizeof(TokenDivOut));

        uint32_t sample = N < 10000u ? N : 10000u;

        uint64_t origOk = 0;
        for (uint32_t i = 0; i < sample; i++) {
            if (orig[i].q == ref[i].q && orig[i].r == ref[i].r) origOk++;
        }

        uint64_t subOk = 0, subWaves = 0;
        for (uint32_t i = 0; i < sample; i++) {
            uint64_t diff = X[i] - Y[i];
            uint64_t borrow = (X[i] < Y[i]) ? 1ULL : 0ULL;
            if (sub[i].q == diff && sub[i].r == borrow) subOk++;
            subWaves += sub[i].waves;
        }

        printf("\n--- Timing ---\n");
        printf("CPU native div/rem:        %.3f ms | %.2f M ops/sec\n",
               (c1 - c0) * 1000.0,
               (double)N / (c1 - c0) / 1e6);

        printf("GPU original radix-4:      %.3f ms | %.2f M ops/sec | ok %llu/%u\n",
               oTime * 1000.0,
               (double)N / oTime / 1e6,
               (unsigned long long)origOk,
               sample);

        summarize_token("Token norm-shift:", nTime, norm, ref, sample, oTime);
        summarize_token("Token base4-compact:", bTime, compact, ref, sample, oTime);

        printf("Token subtract standalone: %.3f ms | %.2f M ops/sec | ok %llu/%u | avg waves %.2f\n",
               sTime * 1000.0,
               (double)N / sTime / 1e6,
               (unsigned long long)subOk,
               sample,
               (double)subWaves / (double)sample);

        printf("\nExample:\n");
        printf("X=%" PRIu64 " Y=%" PRIu64 "\n", X[0], Y[0]);
        printf("ref     q=%" PRIu64 " r=%" PRIu64 "\n", ref[0].q, ref[0].r);
        printf("orig    q=%" PRIu64 " r=%" PRIu64 "\n", orig[0].q, orig[0].r);
        printf("norm    q=%" PRIu64 " r=%" PRIu64 " waves=%u\n", norm[0].q, norm[0].r, norm[0].waves);
        printf("compact q=%" PRIu64 " r=%" PRIu64 " waves=%u\n", compact[0].q, compact[0].r, compact[0].waves);

        free(X);
        free(Y);
        free(ref);
        free(orig);
        free(norm);
        free(compact);
        free(sub);
    }

    return 0;
}
