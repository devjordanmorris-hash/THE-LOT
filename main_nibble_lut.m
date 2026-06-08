/*
    main_nibble_lut.m

    Tests:
      native GPU add
      previous 8-bit split chunk add
      4-bit nibble direct add
      4-bit tiny LUT add

    Compile:
      xcrun -sdk macosx metal -c Base4Kernels_NibbleLUT.metal -o Base4Kernels_NibbleLUT.air
      xcrun -sdk macosx metallib Base4Kernels_NibbleLUT.air -o Base4Kernels_NibbleLUT.metallib
      clang -O3 main_nibble_lut.m -framework Foundation -framework Metal -o base4_nibble_lut

    Run:
      ./base4_nibble_lut
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define LUT_SIZE 256u

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

static uint32_t pack_nibble_entry(uint32_t a, uint32_t b) {
    uint32_t sum0 = a + b;
    uint32_t r0 = sum0 & 0xfu;
    uint32_t c0 = (sum0 >> 4) & 1u;

    uint32_t sum1 = a + b + 1u;
    uint32_t r1 = sum1 & 0xfu;
    uint32_t c1 = (sum1 >> 4) & 1u;

    return (r0 & 0xfu) | ((c0 & 1u) << 4) | ((r1 & 0xfu) << 5) | ((c1 & 1u) << 9);
}

static void build_lut(uint32_t *lut) {
    for (uint32_t a = 0; a < 16; a++) {
        for (uint32_t b = 0; b < 16; b++) {
            lut[(a << 4) | b] = pack_nibble_entry(a, b);
        }
    }
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
        printf("4-bit Nibble LUT Add Benchmark\n");
        printf("N=%u LUT=%u entries %.2f KB\n", N, LUT_SIZE, (double)(LUT_SIZE * sizeof(uint32_t)) / 1024.0);

        uint32_t *lut = malloc(LUT_SIZE * sizeof(uint32_t));
        build_lut(lut);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4Kernels_NibbleLUT.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativePipe = make_pipe(device, lib, @"nativeAddKernel");
        id<MTLComputePipelineState> split8Pipe = make_pipe(device, lib, @"split8ChunkAddKernel");
        id<MTLComputePipelineState> nibbleDirectPipe = make_pipe(device, lib, @"nibbleDirectAddKernel");
        id<MTLComputePipelineState> nibbleLUTPipe = make_pipe(device, lib, @"nibbleLUTAddKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *split8Out = calloc(N, sizeof(uint32_t));
        uint32_t *nibbleDirectOut = calloc(N, sizeof(uint32_t));
        uint32_t *nibbleLUTOut = calloc(N, sizeof(uint32_t));

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> split8OutBuf = [device newBufferWithBytes:split8Out length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nibbleDirectOutBuf = [device newBufferWithBytes:nibbleDirectOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nibbleLUTOutBuf = [device newBufferWithBytes:nibbleLUTOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lutBuf = [device newBufferWithBytes:lut length:LUT_SIZE * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *split8Buffers = @[aBuf, bBuf, split8OutBuf];
        NSArray *nibbleDirectBuffers = @[aBuf, bBuf, nibbleDirectOutBuf];
        NSArray *nibbleLUTBuffers = @[aBuf, bBuf, nibbleLUTOutBuf, lutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, split8Pipe, split8Buffers, N);
        (void)run_kernel(device, queue, nibbleDirectPipe, nibbleDirectBuffers, N);
        (void)run_kernel(device, queue, nibbleLUTPipe, nibbleLUTBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double split8Time = run_kernel(device, queue, split8Pipe, split8Buffers, N);
        double nibbleDirectTime = run_kernel(device, queue, nibbleDirectPipe, nibbleDirectBuffers, N);
        double nibbleLUTTime = run_kernel(device, queue, nibbleLUTPipe, nibbleLUTBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(split8Out, [split8OutBuf contents], N * sizeof(uint32_t));
        memcpy(nibbleDirectOut, [nibbleDirectOutBuf contents], N * sizeof(uint32_t));
        memcpy(nibbleLUTOut, [nibbleLUTOutBuf contents], N * sizeof(uint32_t));

        uint32_t sampleCount = N < 10000u ? N : 10000u;
        uint64_t split8Ok = 0;
        uint64_t nibbleDirectOk = 0;
        uint64_t nibbleLUTOk = 0;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (split8Out[i] == expected) split8Ok++;
            if (nibbleDirectOut[i] == expected) nibbleDirectOk++;
            if (nibbleLUTOut[i] == expected) nibbleLUTOk++;
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:       %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("8-bit split add:      %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               split8Time * 1000.0,
               (double)N / split8Time / 1e6,
               split8Time / nativeTime);

        printf("4-bit direct add:     %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               nibbleDirectTime * 1000.0,
               (double)N / nibbleDirectTime / 1e6,
               nibbleDirectTime / nativeTime);

        printf("4-bit tiny LUT add:   %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
               nibbleLUTTime * 1000.0,
               (double)N / nibbleLUTTime / 1e6,
               nibbleLUTTime / nativeTime);

        printf("8-bit split corr:     %llu/%u\n", (unsigned long long)split8Ok, sampleCount);
        printf("4-bit direct corr:    %llu/%u\n", (unsigned long long)nibbleDirectOk, sampleCount);
        printf("4-bit LUT corr:       %llu/%u\n", (unsigned long long)nibbleLUTOk, sampleCount);

        printf("\nExample:\n");
        printf("A=%u B=%u native=%u split8=%u direct4=%u lut4=%u\n",
               A[0], B[0], nativeOut[0], split8Out[0], nibbleDirectOut[0], nibbleLUTOut[0]);

        free(lut);
        free(A);
        free(B);
        free(nativeOut);
        free(split8Out);
        free(nibbleDirectOut);
        free(nibbleLUTOut);
    }

    return 0;
}
