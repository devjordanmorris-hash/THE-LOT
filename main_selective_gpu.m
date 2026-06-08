/*
    main_selective_gpu.m

    Metal benchmark for selective carry preconditioning.

    Tests:
      - native add
      - selective transform with diagnostics
      - aggressive transform with diagnostics
      - selective result-only
      - aggressive result-only

    Compile:
      xcrun -sdk macosx metal -c SelectiveCarryKernels.metal -o SelectiveCarryKernels.air
      xcrun -sdk macosx metallib SelectiveCarryKernels.air -o SelectiveCarryKernels.metallib
      clang -O3 main_selective_gpu.m -framework Foundation -framework Metal -o selective_gpu

    Run:
      ./selective_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u

typedef struct {
    uint32_t result;
    uint32_t residualB;
    uint32_t mask;
    uint32_t touched;
    uint32_t origCarries;
    uint32_t residualCarries;
    uint32_t origMaxChain;
    uint32_t residualMaxChain;
} TransformOut;

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

static void summarize_transform(
    const char *label,
    const uint32_t *A,
    const uint32_t *B,
    const uint32_t *native,
    const TransformOut *T
) {
    uint64_t ok = 0;
    uint64_t fewer = 0;
    uint64_t same = 0;
    uint64_t more = 0;

    uint64_t origCarries = 0;
    uint64_t resCarries = 0;
    uint64_t origChain = 0;
    uint64_t resChain = 0;
    uint64_t touched = 0;
    uint64_t maskPop = 0;

    uint32_t sample = N < 10000u ? N : 10000u;

    for (uint32_t i = 0; i < N; i++) {
        if (i < sample && T[i].result == native[i]) ok++;

        if (T[i].residualCarries < T[i].origCarries) fewer++;
        else if (T[i].residualCarries == T[i].origCarries) same++;
        else more++;

        origCarries += T[i].origCarries;
        resCarries += T[i].residualCarries;
        origChain += T[i].origMaxChain;
        resChain += T[i].residualMaxChain;
        touched += T[i].touched;
        maskPop += __builtin_popcount(T[i].mask);
    }

    printf("\n--- %s structure ---\n", label);
    printf("correct sample:      %llu/%u\n", (unsigned long long)ok, sample);
    printf("fewer carries:       %.2f%%\n", 100.0 * (double)fewer / (double)N);
    printf("same carries:        %.2f%%\n", 100.0 * (double)same / (double)N);
    printf("more carries:        %.2f%%\n", 100.0 * (double)more / (double)N);
    printf("avg carries:         %.4f -> %.4f\n",
           (double)origCarries / (double)N,
           (double)resCarries / (double)N);
    printf("avg max chain:       %.4f -> %.4f\n",
           (double)origChain / (double)N,
           (double)resChain / (double)N);
    printf("avg touched:         %.4f / 16 digits\n", (double)touched / (double)N);
    printf("avg mask popcount:   %.4f\n", (double)maskPop / (double)N);

    printf("example: A=%u B=%u native=%u result=%u mask=%u touched=%u\n",
           A[0], B[0], native[0], T[0].result, T[0].mask, T[0].touched);
}

static void summarize_result_only(
    const char *label,
    const uint32_t *native,
    const uint32_t *out
) {
    uint32_t sample = N < 10000u ? N : 10000u;
    uint64_t ok = 0;

    for (uint32_t i = 0; i < sample; i++) {
        if (native[i] == out[i]) ok++;
    }

    printf("%s correctness: %llu/%u\n", label, (unsigned long long)ok, sample);
}

int main(void) {
    @autoreleasepool {
        printf("Selective Carry Preconditioner GPU Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"SelectiveCarryKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> selectivePipe = make_pipe(device, lib, @"selectiveAddKernel");
        id<MTLComputePipelineState> aggressivePipe = make_pipe(device, lib, @"aggressiveAddKernel");
        id<MTLComputePipelineState> selectiveNoDebugPipe = make_pipe(device, lib, @"selectiveAddNoDebugKernel");
        id<MTLComputePipelineState> aggressiveNoDebugPipe = make_pipe(device, lib, @"aggressiveAddNoDebugKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *selectiveNoDebugOut = calloc(N, sizeof(uint32_t));
        uint32_t *aggressiveNoDebugOut = calloc(N, sizeof(uint32_t));
        TransformOut *selectiveOut = calloc(N, sizeof(TransformOut));
        TransformOut *aggressiveOut = calloc(N, sizeof(TransformOut));

        if (!A || !B || !nativeOut || !selectiveNoDebugOut || !aggressiveNoDebugOut || !selectiveOut || !aggressiveOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> nativeBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> selectiveBuf = [device newBufferWithBytes:selectiveOut length:N * sizeof(TransformOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> aggressiveBuf = [device newBufferWithBytes:aggressiveOut length:N * sizeof(TransformOut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> selectiveNoDebugBuf = [device newBufferWithBytes:selectiveNoDebugOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> aggressiveNoDebugBuf = [device newBufferWithBytes:aggressiveNoDebugOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeBuf];
        NSArray *selectiveBuffers = @[aBuf, bBuf, selectiveBuf];
        NSArray *aggressiveBuffers = @[aBuf, bBuf, aggressiveBuf];
        NSArray *selectiveNoDebugBuffers = @[aBuf, bBuf, selectiveNoDebugBuf];
        NSArray *aggressiveNoDebugBuffers = @[aBuf, bBuf, aggressiveNoDebugBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, selectivePipe, selectiveBuffers, N);
        (void)run_kernel(device, queue, aggressivePipe, aggressiveBuffers, N);
        (void)run_kernel(device, queue, selectiveNoDebugPipe, selectiveNoDebugBuffers, N);
        (void)run_kernel(device, queue, aggressiveNoDebugPipe, aggressiveNoDebugBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double selectiveTime = run_kernel(device, queue, selectivePipe, selectiveBuffers, N);
        double aggressiveTime = run_kernel(device, queue, aggressivePipe, aggressiveBuffers, N);
        double selectiveNoDebugTime = run_kernel(device, queue, selectiveNoDebugPipe, selectiveNoDebugBuffers, N);
        double aggressiveNoDebugTime = run_kernel(device, queue, aggressiveNoDebugPipe, aggressiveNoDebugBuffers, N);

        memcpy(nativeOut, [nativeBuf contents], N * sizeof(uint32_t));
        memcpy(selectiveOut, [selectiveBuf contents], N * sizeof(TransformOut));
        memcpy(aggressiveOut, [aggressiveBuf contents], N * sizeof(TransformOut));
        memcpy(selectiveNoDebugOut, [selectiveNoDebugBuf contents], N * sizeof(uint32_t));
        memcpy(aggressiveNoDebugOut, [aggressiveNoDebugBuf contents], N * sizeof(uint32_t));

        printf("\n--- Timing ---\n");
        printf("Native add:            %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Selective + debug:     %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               selectiveTime * 1000.0,
               (double)N / selectiveTime / 1e6,
               selectiveTime / nativeTime);

        printf("Aggressive + debug:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               aggressiveTime * 1000.0,
               (double)N / aggressiveTime / 1e6,
               aggressiveTime / nativeTime);

        printf("Selective no-debug:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               selectiveNoDebugTime * 1000.0,
               (double)N / selectiveNoDebugTime / 1e6,
               selectiveNoDebugTime / nativeTime);

        printf("Aggressive no-debug:   %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               aggressiveNoDebugTime * 1000.0,
               (double)N / aggressiveNoDebugTime / 1e6,
               aggressiveNoDebugTime / nativeTime);

        summarize_result_only("selective no-debug", nativeOut, selectiveNoDebugOut);
        summarize_result_only("aggressive no-debug", nativeOut, aggressiveNoDebugOut);

        summarize_transform("Selective", A, B, nativeOut, selectiveOut);
        summarize_transform("Aggressive", A, B, nativeOut, aggressiveOut);

        free(A);
        free(B);
        free(nativeOut);
        free(selectiveNoDebugOut);
        free(aggressiveNoDebugOut);
        free(selectiveOut);
        free(aggressiveOut);
    }

    return 0;
}
