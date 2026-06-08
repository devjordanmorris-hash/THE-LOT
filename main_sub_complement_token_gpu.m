/*
    main_sub_complement_token_gpu.m

    Tests subtraction methods:
      - native subtract
      - borrow-token subtract
      - complement-add carry-token subtract
      - explicit base4 complement-add carry-token subtract
      - result-only native/complement

    Compile:
      xcrun -sdk macosx metal -c SubComplementTokenKernels.metal -o SubComplementTokenKernels.air
      xcrun -sdk macosx metallib SubComplementTokenKernels.air -o SubComplementTokenKernels.metallib
      clang -O3 main_sub_complement_token_gpu.m -framework Foundation -framework Metal -o sub_complement_token_gpu

    Run:
      ./sub_complement_token_gpu
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#define N 1000000u

typedef struct {
    uint64_t result;
    uint32_t flag;
    uint32_t waves;
} SubDebug;

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

static uint64_t check_debug(const SubDebug *got, const SubDebug *ref, uint32_t sample, double *avgWaves) {
    uint64_t ok = 0;
    uint64_t waves = 0;

    for (uint32_t i = 0; i < sample; i++) {
        if (got[i].result == ref[i].result && got[i].flag == ref[i].flag) ok++;
        waves += got[i].waves;
    }

    *avgWaves = (double)waves / (double)sample;
    return ok;
}

static uint64_t check_result(const uint64_t *got, const uint64_t *ref, uint32_t sample) {
    uint64_t ok = 0;
    for (uint32_t i = 0; i < sample; i++) {
        if (got[i] == ref[i]) ok++;
    }
    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("Complement-Add Token Subtractor GPU Benchmark\n");
        printf("N=%u\n", N);

        uint64_t *A = malloc(N * sizeof(uint64_t));
        uint64_t *B = malloc(N * sizeof(uint64_t));

        SubDebug *nativeDbg = calloc(N, sizeof(SubDebug));
        SubDebug *borrowDbg = calloc(N, sizeof(SubDebug));
        SubDebug *compDbg = calloc(N, sizeof(SubDebug));
        SubDebug *b4compDbg = calloc(N, sizeof(SubDebug));

        uint64_t *nativeResult = calloc(N, sizeof(uint64_t));
        uint64_t *compResult = calloc(N, sizeof(uint64_t));

        if (!A || !B || !nativeDbg || !borrowDbg || !compDbg || !b4compDbg || !nativeResult || !compResult) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        for (uint32_t i = 0; i < N; i++) {
            A[i] = xrng();
            B[i] = xrng();
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }

        printf("Device: %s\n", [[device name] UTF8String]);

        NSError *err = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"SubComplementTokenKernels.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];

        if (!lib) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> nativeDbgPipe = make_pipe(device, lib, @"nativeSubKernel");
        id<MTLComputePipelineState> borrowPipe = make_pipe(device, lib, @"borrowTokenSubKernel");
        id<MTLComputePipelineState> compPipe = make_pipe(device, lib, @"complementAddTokenSubKernel");
        id<MTLComputePipelineState> b4compPipe = make_pipe(device, lib, @"base4ComplementAddTokenSubKernel");
        id<MTLComputePipelineState> nativeResPipe = make_pipe(device, lib, @"nativeSubResultOnlyKernel");
        id<MTLComputePipelineState> compResPipe = make_pipe(device, lib, @"complementAddResultOnlyKernel");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> nativeDbgBuf = [device newBufferWithBytes:nativeDbg length:N * sizeof(SubDebug) options:MTLResourceStorageModeShared];
        id<MTLBuffer> borrowBuf = [device newBufferWithBytes:borrowDbg length:N * sizeof(SubDebug) options:MTLResourceStorageModeShared];
        id<MTLBuffer> compBuf = [device newBufferWithBytes:compDbg length:N * sizeof(SubDebug) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b4compBuf = [device newBufferWithBytes:b4compDbg length:N * sizeof(SubDebug) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nativeResBuf = [device newBufferWithBytes:nativeResult length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> compResBuf = [device newBufferWithBytes:compResult length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];

        NSArray *nativeDbgBuffers = @[aBuf, bBuf, nativeDbgBuf];
        NSArray *borrowBuffers = @[aBuf, bBuf, borrowBuf];
        NSArray *compBuffers = @[aBuf, bBuf, compBuf];
        NSArray *b4compBuffers = @[aBuf, bBuf, b4compBuf];
        NSArray *nativeResBuffers = @[aBuf, bBuf, nativeResBuf];
        NSArray *compResBuffers = @[aBuf, bBuf, compResBuf];

        // Warmup
        (void)run_kernel(device, queue, nativeDbgPipe, nativeDbgBuffers, N);
        (void)run_kernel(device, queue, borrowPipe, borrowBuffers, N);
        (void)run_kernel(device, queue, compPipe, compBuffers, N);
        (void)run_kernel(device, queue, b4compPipe, b4compBuffers, N);
        (void)run_kernel(device, queue, nativeResPipe, nativeResBuffers, N);
        (void)run_kernel(device, queue, compResPipe, compResBuffers, N);

        double nativeDbgTime = run_kernel(device, queue, nativeDbgPipe, nativeDbgBuffers, N);
        double borrowTime = run_kernel(device, queue, borrowPipe, borrowBuffers, N);
        double compTime = run_kernel(device, queue, compPipe, compBuffers, N);
        double b4compTime = run_kernel(device, queue, b4compPipe, b4compBuffers, N);
        double nativeResTime = run_kernel(device, queue, nativeResPipe, nativeResBuffers, N);
        double compResTime = run_kernel(device, queue, compResPipe, compResBuffers, N);

        memcpy(nativeDbg, [nativeDbgBuf contents], N * sizeof(SubDebug));
        memcpy(borrowDbg, [borrowBuf contents], N * sizeof(SubDebug));
        memcpy(compDbg, [compBuf contents], N * sizeof(SubDebug));
        memcpy(b4compDbg, [b4compBuf contents], N * sizeof(SubDebug));
        memcpy(nativeResult, [nativeResBuf contents], N * sizeof(uint64_t));
        memcpy(compResult, [compResBuf contents], N * sizeof(uint64_t));

        uint32_t sample = N < 10000u ? N : 10000u;

        double borrowWaves = 0.0;
        double compWaves = 0.0;
        double b4compWaves = 0.0;

        uint64_t borrowOk = check_debug(borrowDbg, nativeDbg, sample, &borrowWaves);
        uint64_t compOk = check_debug(compDbg, nativeDbg, sample, &compWaves);
        uint64_t b4compOk = check_debug(b4compDbg, nativeDbg, sample, &b4compWaves);

        uint64_t compResOk = check_result(compResult, nativeResult, sample);

        printf("\n--- Debug-output timing ---\n");
        printf("Native subtract debug:        %.3f ms | %.2f M ops/sec\n",
               nativeDbgTime * 1000.0,
               (double)N / nativeDbgTime / 1e6);

        printf("Borrow-token subtract:        %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u | avg waves %.3f\n",
               borrowTime * 1000.0,
               (double)N / borrowTime / 1e6,
               borrowTime / nativeDbgTime,
               (unsigned long long)borrowOk,
               sample,
               borrowWaves);

        printf("Complement-add token:         %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u | avg waves %.3f\n",
               compTime * 1000.0,
               (double)N / compTime / 1e6,
               compTime / nativeDbgTime,
               (unsigned long long)compOk,
               sample,
               compWaves);

        printf("Base4 complement-add token:   %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u | avg waves %.3f\n",
               b4compTime * 1000.0,
               (double)N / b4compTime / 1e6,
               b4compTime / nativeDbgTime,
               (unsigned long long)b4compOk,
               sample,
               b4compWaves);

        printf("\n--- Result-only timing ---\n");
        printf("Native subtract result:       %.3f ms | %.2f M ops/sec\n",
               nativeResTime * 1000.0,
               (double)N / nativeResTime / 1e6);

        printf("Complement-add result:        %.3f ms | %.2f M ops/sec | slowdown %.2fx | ok %llu/%u\n",
               compResTime * 1000.0,
               (double)N / compResTime / 1e6,
               compResTime / nativeResTime,
               (unsigned long long)compResOk,
               sample);

        printf("\nExample:\n");
        printf("A=%" PRIu64 " B=%" PRIu64 "\n", A[0], B[0]);
        printf("native result=%" PRIu64 " borrow=%u\n", nativeDbg[0].result, nativeDbg[0].flag);
        printf("borrow token result=%" PRIu64 " borrow=%u waves=%u\n", borrowDbg[0].result, borrowDbg[0].flag, borrowDbg[0].waves);
        printf("comp token result=%" PRIu64 " borrow=%u waves=%u\n", compDbg[0].result, compDbg[0].flag, compDbg[0].waves);

        free(A);
        free(B);
        free(nativeDbg);
        free(borrowDbg);
        free(compDbg);
        free(b4compDbg);
        free(nativeResult);
        free(compResult);
    }

    return 0;
}
