/*
    main_stream_simplified_gpu.m

    Benchmarks simplified stream generation.

    Compile:
      xcrun -sdk macosx metal -c StreamSimplifiedKernels.metal -o StreamSimplifiedKernels.air
      xcrun -sdk macosx metallib StreamSimplifiedKernels.air -o StreamSimplifiedKernels.metallib
      clang -O3 main_stream_simplified_gpu.m -framework Foundation -framework Metal -o stream_simplified_gpu

    Run:
      ./stream_simplified_gpu
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

static uint64_t check(const uint32_t *expected, const uint32_t *got, uint32_t sample) {
    uint64_t ok = 0;
    for (uint32_t i = 0; i < sample; i++) {
        if (expected[i] == got[i]) ok++;
    }
    return ok;
}

static void bench_one(
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
    uint64_t ok = check(native, out, sample);

    printf("%-22s %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
           label,
           t * 1000.0,
           (double)N / t / 1e6,
           t / nativeTime,
           (unsigned long long)ok,
           sample);
}

int main(void) {
    @autoreleasepool {
        printf("Simplified Stream Generation GPU Benchmark\n");
        printf("N=%u\n", N);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"StreamSimplifiedKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> s2BranchyPipe = make_pipe(device, lib, @"stream2BranchyKernel");
        id<MTLComputePipelineState> s2BranchlessPipe = make_pipe(device, lib, @"stream2BranchlessKernel");
        id<MTLComputePipelineState> s3BranchlessPipe = make_pipe(device, lib, @"stream3BranchlessKernel");
        id<MTLComputePipelineState> stripePipe = make_pipe(device, lib, @"stripeFinalResultKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *tmpOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !tmpOut) {
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

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeBuf];
        NSArray *tmpBuffers = @[aBuf, bBuf, tmpBuf];

        // Native
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        memcpy(nativeOut, [nativeBuf contents], N * sizeof(uint32_t));

        printf("--- Timing ---\n");
        printf("%-22s %.3f ms | %.2f M ops/sec\n",
               "Native add:",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        bench_one("2-stream branchy:", device, queue, s2BranchyPipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_one("2-stream branchless:", device, queue, s2BranchlessPipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_one("3-stream branchless:", device, queue, s3BranchlessPipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);
        bench_one("stripe final result:", device, queue, stripePipe, tmpBuffers, tmpOut, tmpBuf, nativeOut, nativeTime);

        printf("\nInterpretation:\n");
        printf("  stripe final result is the lower-bound cost of stream recombine when generation is cheap.\n");
        printf("  2/3-stream kernels show whether simpler stream assignment saves enough vs 5-stream.\n");

        free(A);
        free(B);
        free(nativeOut);
        free(tmpOut);
    }

    return 0;
}
