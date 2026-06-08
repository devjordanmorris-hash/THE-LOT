#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t xrng(void){
    static uint64_t s = 0x9e3779b97f4a7c15ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t N = 10000000u; // default 10M
        if (argc > 1) {
            unsigned long v = strtoul(argv[1], NULL, 10);
            if (v > 0 && v <= 4000000000ul) N = (uint32_t)v;
        }

        // Host buffers
        size_t bytes = (size_t)N * sizeof(uint64_t);
        uint64_t *xs = (uint64_t*)malloc(bytes);
        uint64_t *ys = (uint64_t*)malloc(bytes);
        uint64_t *q_cpu = (uint64_t*)malloc(bytes);
        uint64_t *r_cpu = (uint64_t*)malloc(bytes);
        if (!xs || !ys || !q_cpu || !r_cpu) { fprintf(stderr, "alloc failed\n"); return 1; }

        for (uint32_t i = 0; i < N; ++i) {
            xs[i] = xrng();
            uint64_t y; do { y = xrng(); } while (y == 0);
            ys[i] = y;
        }

        // CPU baseline
        double t0 = now_sec();
        for (uint32_t i = 0; i < N; ++i) {
            q_cpu[i] = xs[i] / ys[i];
            r_cpu[i] = xs[i] % ys[i];
        }
        double t1 = now_sec();
        double cpu_s = t1 - t0;
        double cpu_mps = (double)N / cpu_s / 1e6;

        uint64_t cq = 0, cr = 0;
        for (uint32_t i = 0; i < N; ++i) { cq ^= q_cpu[i]; cr ^= r_cpu[i]; }
        printf("CPU baseline:  %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n",
               cpu_s, cpu_mps, cq, cr);

        // Metal setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device found.\n"); return 1; }
        NSError *err = nil;

        NSString *mslPath = @"gpu_divider_v2.metal";
        NSString *src = [NSString stringWithContentsOfFile:mslPath encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Failed to read gpu_divider_v2.metal: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fun = [lib newFunctionWithName:@"div64_evenfirst_nonrestoring"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fun error:&err];
        if (!pso) { fprintf(stderr, "Pipeline error: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLCommandQueue> q = [device newCommandQueue];
        id<MTLBuffer> bx = [device newBufferWithBytes:xs length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [device newBufferWithBytes:ys length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> br = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn = [device newBufferWithBytes:&N length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Warm-up
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bx offset:0 atIndex:0];
            [enc setBuffer:by offset:0 atIndex:1];
            [enc setBuffer:bq offset:0 atIndex:2];
            [enc setBuffer:br offset:0 atIndex:3];
            [enc setBuffer:bn offset:0 atIndex:4];

            NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
            if (w > 256) w = 256;
            MTLSize tg = MTLSizeMake(w, 1, 1);
            MTLSize grid = MTLSizeMake(N, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Timed run
        double g0 = now_sec();
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bx offset:0 atIndex:0];
            [enc setBuffer:by offset:0 atIndex:1];
            [enc setBuffer:bq offset:0 atIndex:2];
            [enc setBuffer:br offset:0 atIndex:3];
            [enc setBuffer:bn offset:0 atIndex:4];

            NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
            if (w > 256) w = 256;
            MTLSize tg = MTLSizeMake(w, 1, 1);
            MTLSize grid = MTLSizeMake(N, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double g1 = now_sec();
        double gpu_s = g1 - g0;
        double gpu_mps = (double)N / gpu_s / 1e6;

        uint64_t *qout = (uint64_t*)[bq contents];
        uint64_t *rout = (uint64_t*)[br contents];

        // Validate
        size_t mism = 0;
        for (uint32_t i = 0; i < N; ++i) {
            if (qout[i] != q_cpu[i] || rout[i] != r_cpu[i]) mism++;
        }

        uint64_t gq = 0, gr = 0;
        for (uint32_t i = 0; i < N; ++i) { gq ^= qout[i]; gr ^= rout[i]; }

        printf("GPU v2 (Metal): %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n",
               gpu_s, gpu_mps, gq, gr);
        printf("Compare: mismatches = %zu (of %u)\n", mism, N);
        printf("Speedup (CPU/GPU): %.2fx\n", (gpu_s > 0 ? cpu_s / gpu_s : 0.0));

        free(xs); free(ys); free(q_cpu); free(r_cpu);
    }
    return 0;
}
