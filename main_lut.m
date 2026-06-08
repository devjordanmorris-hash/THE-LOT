/*
    main_lut.m

    Metal benchmark using a global carry-chunk LUT for the base-4 add preconditioner.

    Idea:
      Instead of scanning 16 base-4 digits per value on GPU, process 4 chunks.
      Each chunk is 8 bits = 4 base-4 digits.

      LUT index:
        carry_in: 0/1
        a_byte:   0..255
        b_byte:   0..255

        index = (carry_in << 16) | (a_byte << 8) | b_byte

      LUT output packed into uint:
        bits  0..7   = mask_byte
        bits  8..15  = residual_b_byte
        bit   16     = carry_out

    Folder:
      main_lut.m
      Base4Kernels_LUT.metal

    Commands:
      xcrun -sdk macosx metal -c Base4Kernels_LUT.metal -o Base4Kernels_LUT.air
      xcrun -sdk macosx metallib Base4Kernels_LUT.air -o Base4Kernels_LUT.metallib
      clang -O3 main_lut.m -framework Foundation -framework Metal -o base4_metal_lut
      ./base4_metal_lut
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000u
#define LUT_SIZE (2u * 256u * 256u)

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

/*
    Build one 8-bit chunk:
      a_byte and b_byte each hold 4 base-4 digits.
      carry_in enters at the least significant base-4 digit of the chunk.
*/
static uint32_t build_lut_entry(uint32_t carry_in, uint32_t a_byte, uint32_t b_byte) {
    uint32_t mask_byte = 0;
    uint32_t residual_b = b_byte;
    uint32_t carry = carry_in & 1u;

    for (uint32_t shift = 0; shift < 8; shift += 2) {
        uint32_t ad = (a_byte >> shift) & 3u;
        uint32_t bd = (residual_b >> shift) & 3u;

        uint32_t sum = ad + bd + carry;

        if (sum >= 4u && bd > 0u) {
            uint32_t amount = 1u << shift;

            mask_byte |= amount;
            residual_b -= amount;

            bd -= 1u;
            sum = ad + bd + carry;
        }

        carry = (sum >= 4u) ? 1u : 0u;
    }

    return (mask_byte & 0xffu) | ((residual_b & 0xffu) << 8) | ((carry & 1u) << 16);
}

static void build_lut(uint32_t *lut) {
    for (uint32_t carry = 0; carry < 2; carry++) {
        for (uint32_t a = 0; a < 256; a++) {
            for (uint32_t b = 0; b < 256; b++) {
                uint32_t idx = (carry << 16) | (a << 8) | b;
                lut[idx] = build_lut_entry(carry, a, b);
            }
        }
    }
}

static uint32_t adaptive_add_cpu_lut(uint32_t a, uint32_t b, const uint32_t *lut) {
    uint32_t mask = 0;
    uint32_t residual = 0;
    uint32_t carry = 0;

    for (uint32_t shift = 0; shift < 32; shift += 8) {
        uint32_t a_byte = (a >> shift) & 0xffu;
        uint32_t b_byte = (b >> shift) & 0xffu;

        uint32_t idx = (carry << 16) | (a_byte << 8) | b_byte;
        uint32_t entry = lut[idx];

        uint32_t m = entry & 0xffu;
        uint32_t r = (entry >> 8) & 0xffu;
        carry = (entry >> 16) & 1u;

        mask |= m << shift;
        residual |= r << shift;
    }

    return a + residual + mask;
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

    id<MTLBuffer> nbuf =
        [device newBufferWithBytes:&count
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

int main(int argc, const char **argv) {
    @autoreleasepool {
        printf("Base-4 Metal LUT Benchmark\n");
        printf("N=%u\n", N);
        printf("LUT entries=%u, size=%.2f KB\n", LUT_SIZE, (double)(LUT_SIZE * sizeof(uint32_t)) / 1024.0);

        uint32_t *lut = malloc(LUT_SIZE * sizeof(uint32_t));
        if (!lut) {
            fprintf(stderr, "LUT allocation failed\n");
            return 1;
        }

        build_lut(lut);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device found\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"Base4Kernels_LUT.metallib"];
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&err];

        if (!library) {
            fprintf(stderr, "Failed to load Base4Kernels_LUT.metallib: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLFunction> nativeFn = [library newFunctionWithName:@"nativeAddKernel"];
        id<MTLFunction> lutFn = [library newFunctionWithName:@"base4TransformAddLUTKernel"];

        if (!nativeFn || !lutFn) {
            fprintf(stderr, "Missing kernel function(s)\n");
            return 1;
        }

        id<MTLComputePipelineState> nativePipe =
            [device newComputePipelineStateWithFunction:nativeFn error:&err];

        id<MTLComputePipelineState> lutPipe =
            [device newComputePipelineStateWithFunction:lutFn error:&err];

        if (!nativePipe || !lutPipe) {
            fprintf(stderr, "Pipeline creation failed\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        uint32_t *A = malloc(N * sizeof(uint32_t));
        uint32_t *B = malloc(N * sizeof(uint32_t));
        uint32_t *nativeOut = calloc(N, sizeof(uint32_t));
        uint32_t *lutOut = calloc(N, sizeof(uint32_t));
        uint32_t *masks = calloc(N, sizeof(uint32_t));
        uint32_t *residuals = calloc(N, sizeof(uint32_t));

        if (!A || !B || !nativeOut || !lutOut || !masks || !residuals) {
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
        id<MTLBuffer> lutOutBuf = [device newBufferWithBytes:lutOut length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [device newBufferWithBytes:masks length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> residualBuf = [device newBufferWithBytes:residuals length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lutBuf = [device newBufferWithBytes:lut length:LUT_SIZE * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        NSArray *nativeBuffers = @[aBuf, bBuf, nativeOutBuf];
        NSArray *lutBuffers = @[aBuf, bBuf, lutOutBuf, maskBuf, residualBuf, lutBuf];

        // Warmup
        (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
        (void)run_kernel(device, queue, lutPipe, lutBuffers, N);

        double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
        double lutTime = run_kernel(device, queue, lutPipe, lutBuffers, N);

        memcpy(nativeOut, [nativeOutBuf contents], N * sizeof(uint32_t));
        memcpy(lutOut, [lutOutBuf contents], N * sizeof(uint32_t));
        memcpy(masks, [maskBuf contents], N * sizeof(uint32_t));
        memcpy(residuals, [residualBuf contents], N * sizeof(uint32_t));

        uint64_t ok = 0;
        uint64_t cpuMatch = 0;
        uint64_t maskPop = 0;

        uint32_t sampleCount = N < 10000u ? N : 10000u;

        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t expected = A[i] + B[i];

            if (nativeOut[i] == expected && lutOut[i] == expected) {
                ok++;
            }

            uint32_t cpu = adaptive_add_cpu_lut(A[i], B[i], lut);
            if (cpu == lutOut[i]) {
                cpuMatch++;
            }
        }

        for (uint32_t i = 0; i < N; i++) {
            maskPop += __builtin_popcount(masks[i]);
        }

        printf("\n--- Results ---\n");
        printf("Native GPU add:       %.3f ms | %.2f M ops/sec\n",
               nativeTime * 1000.0,
               (double)N / nativeTime / 1e6);

        printf("LUT transform GPU add: %.3f ms | %.2f M ops/sec\n",
               lutTime * 1000.0,
               (double)N / lutTime / 1e6);

        printf("Slowdown:             %.2fx\n", lutTime / nativeTime);
        printf("Correct sample:       %llu/%u\n", (unsigned long long)ok, sampleCount);
        printf("CPU/GPU LUT sample:   %llu/%u\n", (unsigned long long)cpuMatch, sampleCount);
        printf("Avg mask popcount:    %.4f / 16 base4 digits\n", (double)maskPop / (double)N);

        printf("\nExample:\n");
        printf("A=%u\n", A[0]);
        printf("B=%u\n", B[0]);
        printf("native=%u\n", nativeOut[0]);
        printf("lutTransform=%u\n", lutOut[0]);
        printf("mask=%u\n", masks[0]);
        printf("residualB=%u\n", residuals[0]);

        free(lut);
        free(A);
        free(B);
        free(nativeOut);
        free(lutOut);
        free(masks);
        free(residuals);
    }

    return 0;
}
