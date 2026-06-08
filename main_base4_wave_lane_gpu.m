/*
    main_base4_wave_lane_gpu.m

    Tests whether base-4 carry waves can use GPU lanes correctly:
      - native GPU add
      - serial base-4 carry-wave, one thread per add
      - cooperative base-4 carry-wave, 16 threads per add
      - fixed-wave cooperative version

    Compile:
      xcrun -sdk macosx metal -c Base4WaveLaneKernels.metal -o Base4WaveLaneKernels.air
      xcrun -sdk macosx metallib Base4WaveLaneKernels.air -o Base4WaveLaneKernels.metallib
      clang -O3 main_base4_wave_lane_gpu.m -framework Foundation -framework Metal -o base4_wave_lane_gpu

    Run:
      ./base4_wave_lane_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define WIDTH 16u

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

static double run_threads_kernel(
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

static double run_groups_kernel(
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

    MTLSize threadsPerGroup = MTLSizeMake(WIDTH, 1, 1);
    MTLSize groups = MTLSizeMake(count, 1, 1);

    double t0 = now_sec();

    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return now_sec() - t0;
}

static uint64_t check(const uint32_t *expected, const uint32_t *got, uint32_t sample) {
    uint64_t ok = 0;
    for (uint32_t i = 0; i < sample; i++) {
        if (expected[i] == got[i]) ok++;
    }
    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("Base-4 Carry-Wave Lane GPU Benchmark\n");
        printf("N=%u, WIDTH=%u lanes per cooperative add\n", N, WIDTH);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4WaveLaneKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> serialPipe = make_pipe(device, lib, @"base4WaveSerialKernel");
        id<MTLComputePipelineState> lanePipe = make_pipe(device, lib, @"base4WaveLaneKernel");
        id<MTLComputePipelineState> fixedPipe = make_pipe(device, lib, @"base4WaveLaneFixedKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *serialOut = calloc(N, sizeof(uint32_t));
        uint32_t *laneOut = calloc(N, sizeof(uint32_t));
        uint32_t *fixedOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !serialOut || !laneOut || !fixedOut) {
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
        id<MTLBuffer> serialBuf = [device newBufferWithBytes:serialOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> laneBuf = [device newBufferWithBytes:laneOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> fixedBuf = [device newBufferWithBytes:fixedOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeBuf];
        NSArray *serialBuffers = @[aBuf, bBuf, serialBuf];
        NSArray *laneBuffers = @[aBuf, bBuf, laneBuf];
        NSArray *fixedBuffers = @[aBuf, bBuf, fixedBuf];

        // Warmup
        (void)run_threads_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_threads_kernel(device, queue, serialPipe, serialBuffers, N);
        (void)run_groups_kernel(device, queue, lanePipe, laneBuffers, N);
        (void)run_groups_kernel(device, queue, fixedPipe, fixedBuffers, N);

        double nativeTime = run_threads_kernel(device, queue, nativePipe, nativeBuffers, N);
        double serialTime = run_threads_kernel(device, queue, serialPipe, serialBuffers, N);
        double laneTime = run_groups_kernel(device, queue, lanePipe, laneBuffers, N);
        double fixedTime = run_groups_kernel(device, queue, fixedPipe, fixedBuffers, N);

        memcpy(nativeOut, [nativeBuf contents], N * sizeof(uint32_t));
        memcpy(serialOut, [serialBuf contents], N * sizeof(uint32_t));
        memcpy(laneOut, [laneBuf contents], N * sizeof(uint32_t));
        memcpy(fixedOut, [fixedBuf contents], N * sizeof(uint32_t));

        uint32_t sample = N < 10000u ? N : 10000u;

        uint64_t serialOk = check(nativeOut, serialOut, sample);
        uint64_t laneOk = check(nativeOut, laneOut, sample);
        uint64_t fixedOk = check(nativeOut, fixedOut, sample);

        printf("--- Timing ---\n");
        printf("Native add:          %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Wave serial/thread:  %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
               serialTime * 1000.0,
               (double)N / serialTime / 1e6,
               serialTime / nativeTime,
               (unsigned long long)serialOk,
               sample);

        printf("Wave 16-lane group:  %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
               laneTime * 1000.0,
               (double)N / laneTime / 1e6,
               laneTime / nativeTime,
               (unsigned long long)laneOk,
               sample);

        printf("Wave fixed 16-lane:  %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
               fixedTime * 1000.0,
               (double)N / fixedTime / 1e6,
               fixedTime / nativeTime,
               (unsigned long long)fixedOk,
               sample);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u serial=%u lane=%u fixed=%u\n",
               A[0], B[0], nativeOut[0], serialOut[0], laneOut[0], fixedOut[0]);

        printf("\nInterpretation:\n");
        printf("  If 16-lane is faster than serial/thread, lane-level parallelism is helping.\n");
        printf("  If not, threadgroup barriers/memory dominate on current GPU hardware.\n");

        free(A);
        free(B);
        free(nativeOut);
        free(serialOut);
        free(laneOut);
        free(fixedOut);
    }

    return 0;
}
