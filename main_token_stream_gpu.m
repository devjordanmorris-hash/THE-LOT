/*
    main_token_stream_gpu.m

    Adaptive carry-token stream GPU benchmark.

    Compile:
      xcrun -sdk macosx metal -c TokenStreamKernels.metal -o TokenStreamKernels.air
      xcrun -sdk macosx metallib TokenStreamKernels.air -o TokenStreamKernels.metallib
      clang -O3 main_token_stream_gpu.m -framework Foundation -framework Metal -o token_stream_gpu

    Run:
      ./token_stream_gpu
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
    uint32_t waves;
    uint32_t firstCarryTokens;
    uint32_t totalTokenCarries;
    uint32_t finalTokenStream;
} TokenDebug;

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

static uint64_t check_u32(const uint32_t *expected, const uint32_t *got, uint32_t sample) {
    uint64_t ok = 0;
    for (uint32_t i = 0; i < sample; i++) {
        if (expected[i] == got[i]) ok++;
    }
    return ok;
}

static void bench_u32(
    const char *label,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipe,
    NSArray *buffers,
    uint32_t *out,
    id<MTLBuffer> outBuf,
    const uint32_t *native,
    double nativeTime
) {
    (void)run_kernel(device, queue, pipe, buffers, N);
    double t = run_kernel(device, queue, pipe, buffers, N);

    memcpy(out, [outBuf contents], N * sizeof(uint32_t));

    uint32_t sample = N < 10000u ? N : 10000u;
    uint64_t ok = check_u32(native, out, sample);

    printf("%-24s %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
           label,
           t * 1000.0,
           (double)N / t / 1e6,
           t / nativeTime,
           (unsigned long long)ok,
           sample);
}

int main(void) {
    @autoreleasepool {
        printf("Adaptive Carry-Token Stream GPU Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"TokenStreamKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> serialPipe = make_pipe(device, lib, @"base4WaveSerialKernel");
        id<MTLComputePipelineState> tokenPipe = make_pipe(device, lib, @"tokenStreamKernel");
        id<MTLComputePipelineState> fixed2Pipe = make_pipe(device, lib, @"tokenStreamFixed2Kernel");
        id<MTLComputePipelineState> fixed4Pipe = make_pipe(device, lib, @"tokenStreamFixed4Kernel");
        id<MTLComputePipelineState> debugPipe = make_pipe(device, lib, @"tokenStreamDebugKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *tmpOut = calloc(N, sizeof(uint32_t));
        TokenDebug *debugOut = calloc(N, sizeof(TokenDebug));

        if (!A || !B || !nativeOut || !tmpOut || !debugOut) {
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
        id<MTLBuffer> tmpBuf = [device newBufferWithBytes:tmpOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> debugBuf = [device newBufferWithBytes:debugOut length:N * sizeof(TokenDebug) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeBuf];
        NSArray *tmpBuffers = @[aBuf, bBuf, tmpBuf];
        NSArray *debugBuffers = @[aBuf, bBuf, debugBuf];

        // Native
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        memcpy(nativeOut, [nativeBuf contents], N * sizeof(uint32_t));

        printf("--- Timing ---\n");
        printf("%-24s %.3f ms | %.2f M ops/sec\n",
               "Native add:",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        bench_u32("Wave serial:", device, queue, serialPipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_u32("Token stream:", device, queue, tokenPipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_u32("Token fixed 2 waves:", device, queue, fixed2Pipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_u32("Token fixed 4 waves:", device, queue, fixed4Pipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);

        // Debug
        (void)run_kernel(device, queue, debugPipe, debugBuffers, N);
        double debugTime = run_kernel(device, queue, debugPipe, debugBuffers, N);
        memcpy(debugOut, [debugBuf contents], N * sizeof(TokenDebug));

        uint32_t sample = N < 10000u ? N : 10000u;
        uint64_t debugOk = 0;
        uint64_t wavesTotal = 0;
        uint64_t firstTokensTotal = 0;
        uint64_t tokenCarriesTotal = 0;
        uint64_t hist[16] = {0};

        for (uint32_t i = 0; i < sample; i++) {
            if (debugOut[i].result == nativeOut[i]) debugOk++;
            wavesTotal += debugOut[i].waves;
            firstTokensTotal += debugOut[i].firstCarryTokens;
            tokenCarriesTotal += debugOut[i].totalTokenCarries;
            if (debugOut[i].waves < 16) hist[debugOut[i].waves]++;
        }

        printf("%-24s %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
               "Token debug:",
               debugTime * 1000.0,
               (double)N / debugTime / 1e6,
               debugTime / nativeTime,
               (unsigned long long)debugOk,
               sample);

        printf("\n--- Debug sample stats ---\n");
        printf("avg waves:          %.4f\n", (double)wavesTotal / (double)sample);
        printf("avg first tokens:   %.4f\n", (double)firstTokensTotal / (double)sample);
        printf("avg token carries:  %.4f\n", (double)tokenCarriesTotal / (double)sample);

        printf("wave histogram sample:\n");
        for (uint32_t i = 0; i < 12; i++) {
            if (hist[i]) {
                printf("  waves=%u : %.2f%%\n", i, 100.0 * (double)hist[i] / (double)sample);
            }
        }

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u token=%u waves=%u firstTokens=%u tokenCarries=%u\n",
               A[0], B[0], nativeOut[0], debugOut[0].result,
               debugOut[0].waves, debugOut[0].firstCarryTokens, debugOut[0].totalTokenCarries);

        free(A);
        free(B);
        free(nativeOut);
        free(tmpOut);
        free(debugOut);
    }

    return 0;
}
