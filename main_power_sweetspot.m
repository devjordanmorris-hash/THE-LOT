/*
    main_power_sweetspot.m

    Power as repeated wide-add workload at the observed sweet spots:
      512-bit and 1024-bit

    Compile:
      xcrun -sdk macosx metal -c PowerSweetspotKernels.metal -o PowerSweetspotKernels.air
      xcrun -sdk macosx metallib PowerSweetspotKernels.air -o PowerSweetspotKernels.metallib
      clang -O3 main_power_sweetspot.m -framework Foundation -framework Metal -o power_sweetspot

    Run:
      ./power_sweetspot
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 20000u

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

static void wide_mul_small_cpu(uint32_t *out, const uint32_t *a, uint32_t multiplier, uint32_t words) {
    memset(out, 0, words * sizeof(uint32_t));

    for (uint32_t m = 0; m < multiplier; m++) {
        uint64_t carry = 0;

        for (uint32_t i = 0; i < words; i++) {
            uint64_t sum = (uint64_t)out[i] + (uint64_t)a[i] + carry;
            out[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

static void power_wide_cpu(uint32_t base, uint32_t exp, uint32_t *out, uint32_t words) {
    uint32_t result[32];
    uint32_t tmp[32];

    memset(result, 0, sizeof(result));
    result[0] = 1;

    for (uint32_t e = 0; e < exp; e++) {
        wide_mul_small_cpu(tmp, result, base, words);
        memcpy(result, tmp, words * sizeof(uint32_t));
    }

    memcpy(out, result, words * sizeof(uint32_t));
}

static int eq_words(const uint32_t *a, const uint32_t *b, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void run_width(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    uint32_t bits,
    uint32_t words,
    id<MTLComputePipelineState> nativePipe,
    id<MTLComputePipelineState> predictPipe,
    id<MTLBuffer> baseBuf,
    id<MTLBuffer> expBuf,
    const uint32_t *BASE,
    const uint32_t *EXP
) {
    size_t outWords = (size_t)N * words;
    size_t outBytes = outWords * sizeof(uint32_t);

    uint32_t *nativeOut = calloc(outWords, sizeof(uint32_t));
    uint32_t *predictOut = calloc(outWords, sizeof(uint32_t));

    id<MTLBuffer> nativeOutBuf = [device newBufferWithBytes:nativeOut length:outBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> predictOutBuf = [device newBufferWithBytes:predictOut length:outBytes options:MTLResourceStorageModeShared];

    NSArray *nativeBuffers = @[baseBuf, expBuf, nativeOutBuf];
    NSArray *predictBuffers = @[baseBuf, expBuf, predictOutBuf];

    // Warmup
    (void)run_kernel(device, queue, nativePipe, nativeBuffers, N);
    (void)run_kernel(device, queue, predictPipe, predictBuffers, N);

    double nativeTime = run_kernel(device, queue, nativePipe, nativeBuffers, N);
    double predictTime = run_kernel(device, queue, predictPipe, predictBuffers, N);

    memcpy(nativeOut, [nativeOutBuf contents], outBytes);
    memcpy(predictOut, [predictOutBuf contents], outBytes);

    uint32_t sample = N < 2000u ? N : 2000u;
    uint64_t nativeOk = 0;
    uint64_t predictOk = 0;
    uint64_t match = 0;

    uint32_t cpu[32];

    for (uint32_t j = 0; j < sample; j++) {
        power_wide_cpu(BASE[j], EXP[j], cpu, words);

        uint32_t idx = j * words;

        nativeOk += eq_words(&nativeOut[idx], cpu, words);
        predictOk += eq_words(&predictOut[idx], cpu, words);
        match += eq_words(&nativeOut[idx], &predictOut[idx], words);
    }

    printf("%4u-bit power | native %7.3f ms %9.2f K/s | predict %7.3f ms %9.2f K/s | speed %.2fx | ok %llu/%u %llu/%u match %llu/%u\n",
           bits,
           nativeTime * 1000.0,
           (double)N / nativeTime / 1e3,
           predictTime * 1000.0,
           (double)N / predictTime / 1e3,
           nativeTime / predictTime,
           (unsigned long long)nativeOk, sample,
           (unsigned long long)predictOk, sample,
           (unsigned long long)match, sample);

    free(nativeOut);
    free(predictOut);
}

int main(void) {
    @autoreleasepool {
        printf("Power-as-Wide-Add Sweetspot Benchmark\n");
        printf("N=%u, base 2..15, exponent 2..8\n", N);

        uint32_t *BASE = malloc(N * sizeof(uint32_t));
        uint32_t *EXP = malloc(N * sizeof(uint32_t));

        for (uint32_t i = 0; i < N; i++) {
            BASE[i] = 2u + (xorshift32() % 14u);
            EXP[i] = 2u + (xorshift32() % 7u);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"PowerSweetspotKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> native512 = make_pipe(device, lib, @"power512NativeKernel");
        id<MTLComputePipelineState> predict512 = make_pipe(device, lib, @"power512PredictKernel");
        id<MTLComputePipelineState> native1024 = make_pipe(device, lib, @"power1024NativeKernel");
        id<MTLComputePipelineState> predict1024 = make_pipe(device, lib, @"power1024PredictKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> baseBuf = [device newBufferWithBytes:BASE length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> expBuf = [device newBufferWithBytes:EXP length:N * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        printf("--- Results ---\n");
        run_width(device, queue, 512, 16, native512, predict512, baseBuf, expBuf, BASE, EXP);
        run_width(device, queue, 1024, 32, native1024, predict1024, baseBuf, expBuf, BASE, EXP);

        printf("\nExample: base=%u exp=%u\n", BASE[0], EXP[0]);

        free(BASE);
        free(EXP);
    }

    return 0;
}
