/*
    main_32_xor_memory_predict.m

    Tests smaller 32-bit predictors:
      1. native add
      2. direct byte predictor
      3. XOR-table predictor using tiny 256-entry table for propagate
      4. XOR/AND carry-logic predictor

    Compile:
      xcrun -sdk macosx metal -c Add32XorMemoryPredict.metal -o Add32XorMemoryPredict.air
      xcrun -sdk macosx metallib Add32XorMemoryPredict.air -o Add32XorMemoryPredict.metallib
      clang -O3 main_32_xor_memory_predict.m -framework Foundation -framework Metal -o add32_xor_memory_predict

    Run:
      ./add32_xor_memory_predict
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define XOR_TABLE_SIZE 256u

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

int main(void) {
    @autoreleasepool {
        printf("32-bit XOR-memory Predictor GPU Benchmark\n");
        printf("N=%u\n", N);

        uint32_t *xorTable = calloc(XOR_TABLE_SIZE, sizeof(uint32_t));
        for (uint32_t x = 0; x < XOR_TABLE_SIZE; x++) {
            xorTable[x] = (x == 0xffu) ? 1u : 0u;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Add32XorMemoryPredict.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> bytePipe = make_pipe(device, lib, @"bytePredict32AddKernel");
        id<MTLComputePipelineState> xorTablePipe = make_pipe(device, lib, @"byteXorTablePredict32AddKernel");
        id<MTLComputePipelineState> xorAndPipe = make_pipe(device, lib, @"byteXorAndPredict32AddKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *byteOut = calloc(N, sizeof(uint32_t));
        uint32_t *xorTableOut = calloc(N, sizeof(uint32_t));
        uint32_t *xorAndOut = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !byteOut || !xorTableOut || !xorAndOut) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> byteOutBuf = [device newBufferWithBytes:byteOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xorTableOutBuf = [device newBufferWithBytes:xorTableOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xorAndOutBuf = [device newBufferWithBytes:xorAndOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xorTableBuf = [device newBufferWithBytes:xorTable length:XOR_TABLE_SIZE * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *byteBuffers = @[aBuf, bBuf, byteOutBuf];
        NSArray *xorTableBuffers = @[aBuf, bBuf, xorTableOutBuf, xorTableBuf];
        NSArray *xorAndBuffers = @[aBuf, bBuf, xorAndOutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, bytePipe, byteBuffers, N);
        (void)run_kernel(device, queue, xorTablePipe, xorTableBuffers, N);
        (void)run_kernel(device, queue, xorAndPipe, xorAndBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double byteTime = run_kernel(device, queue, bytePipe, byteBuffers, N);
        double xorTableTime = run_kernel(device, queue, xorTablePipe, xorTableBuffers, N);
        double xorAndTime = run_kernel(device, queue, xorAndPipe, xorAndBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(byteOut, [byteOutBuf contents], N * sizeof(uint32_t));
        memcpy(xorTableOut, [xorTableOutBuf contents], N * sizeof(uint32_t));
        memcpy(xorAndOut, [xorAndOutBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t byteOk = 0;
        uint64_t xorTableOk = 0;
        uint64_t xorAndOk = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (byteOut[i] == expected) byteOk++;
            if (xorTableOut[i] == expected) xorTableOk++;
            if (xorAndOut[i] == expected) xorAndOk++;
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:       %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("Byte predictor:       %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               byteTime * 1000.0,
               (double)N / byteTime / 1e6,
               byteTime / nativeTime);

        printf("XOR-table predictor:  %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               xorTableTime * 1000.0,
               (double)N / xorTableTime / 1e6,
               xorTableTime / nativeTime);

        printf("XOR/AND predictor:    %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               xorAndTime * 1000.0,
               (double)N / xorAndTime / 1e6,
               xorAndTime / nativeTime);

        printf("\n--- Correctness ---\n");
        printf("Byte predictor:       %llu/%u\n", (unsigned long long)byteOk, sampleCount);
        printf("XOR-table predictor:  %llu/%u\n", (unsigned long long)xorTableOk, sampleCount);
        printf("XOR/AND predictor:    %llu/%u\n", (unsigned long long)xorAndOk, sampleCount);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u byte=%u xorTable=%u xorAnd=%u\n",
               A[0], B[0], nativeOut[0], byteOut[0], xorTableOut[0], xorAndOut[0]);

        free(xorTable);
        free(A);
        free(B);
        free(nativeOut);
        free(byteOut);
        free(xorTableOut);
        free(xorAndOut);
    }

    return 0;
}
