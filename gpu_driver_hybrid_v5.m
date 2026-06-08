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

static inline uint32_t bitlen_u64(uint64_t x) {
    if (!x) return 0;
#if defined(__clang__) || defined(__GNUC__)
    return 64u - (uint32_t)__builtin_clzll(x);
#else
    uint32_t n=0; while (x) { n++; x >>= 1; } return n;
#endif
}

static void stats(const char* tag, double secs, uint64_t ops, uint64_t sinkQ, uint64_t sinkR){
    double mps = (double)ops / secs / 1e6;
    printf("%-22s %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n", tag, secs, mps, sinkQ, sinkR);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t N = 10000000u;
        uint32_t THRESH = 32u; // bitlen(y) <= THRESH -> GPU, else CPU
        if (argc > 1) {
            unsigned long v = strtoul(argv[1], NULL, 10);
            if (v > 0 && v <= 4000000000ul) N = (uint32_t)v;
        }
        if (argc > 2) {
            unsigned long t = strtoul(argv[2], NULL, 10);
            if (t >= 8 && t <= 63) THRESH = (uint32_t)t;
        }
        printf("Hybrid threshold: bitlen(y) <= %u -> GPU (radix-4), else CPU\n", THRESH);

        size_t bytes = (size_t)N * sizeof(uint64_t);
        uint64_t *xs = (uint64_t*)malloc(bytes);
        uint64_t *ys = (uint64_t*)malloc(bytes);
        uint64_t *q_out = (uint64_t*)malloc(bytes);
        uint64_t *r_out = (uint64_t*)malloc(bytes);
        uint64_t *q_cpu = (uint64_t*)malloc(bytes);
        uint64_t *r_cpu = (uint64_t*)malloc(bytes);
        if (!xs || !ys || !q_out || !r_out || !q_cpu || !r_cpu) { fprintf(stderr, "alloc failed\n"); return 1; }

        // Generate inputs
        for (uint32_t i = 0; i < N; ++i) {
            xs[i] = xrng();
            uint64_t y; do { y = xrng(); } while (y == 0);
            ys[i] = y;
        }

        // Partition into GPU-small and CPU-big by bitlen(y)
        uint32_t count_small = 0, count_big = 0;
        for (uint32_t i = 0; i < N; ++i) {
            if (bitlen_u64(ys[i]) <= THRESH) count_small++; else count_big++;
        }
        uint64_t *xs_small = (uint64_t*)malloc((size_t)count_small * sizeof(uint64_t));
        uint64_t *ys_small = (uint64_t*)malloc((size_t)count_small * sizeof(uint64_t));
        uint32_t *idx_small = (uint32_t*)malloc((size_t)count_small * sizeof(uint32_t));
        uint64_t *xs_big   = (uint64_t*)malloc((size_t)count_big * sizeof(uint64_t));
        uint64_t *ys_big   = (uint64_t*)malloc((size_t)count_big * sizeof(uint64_t));
        uint32_t *idx_big  = (uint32_t*)malloc((size_t)count_big * sizeof(uint32_t));
        if (!xs_small || !ys_small || !idx_small || !xs_big || !ys_big || !idx_big) { fprintf(stderr, "alloc2 failed\n"); return 1; }

        uint32_t ps=0, pb=0;
        for (uint32_t i = 0; i < N; ++i) {
            if (bitlen_u64(ys[i]) <= THRESH) {
                xs_small[ps] = xs[i]; ys_small[ps] = ys[i]; idx_small[ps] = i; ps++;
            } else {
                xs_big[pb] = xs[i]; ys_big[pb] = ys[i]; idx_big[pb] = i; pb++;
            }
        }

        // CPU baseline for correctness and reference
        double t0 = now_sec();
        for (uint32_t i = 0; i < N; ++i) {
            q_cpu[i] = xs[i] / ys[i];
            r_cpu[i] = xs[i] % ys[i];
        }
        double t1 = now_sec();
        stats("CPU baseline:", t1 - t0, N, 0, 0);

        // Metal setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device found.\n"); return 1; }
        NSError *err = nil;

        NSString *mslPath = @"gpu_divider_hybrid_v5.metal";
        NSString *src = [NSString stringWithContentsOfFile:mslPath encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Failed to read metal: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLFunction> f_subset = [lib newFunctionWithName:@"div64_radix4_evenfirst_subset"];
        id<MTLComputePipelineState> p_subset = [device newComputePipelineStateWithFunction:f_subset error:&err];
        if (!p_subset) { fprintf(stderr, "Pipeline subset error: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLCommandQueue> q = [device newCommandQueue];

        // GPU buffers for the small subset
        size_t bytes_small = (size_t)count_small * sizeof(uint64_t);
        id<MTLBuffer> bx_s = [device newBufferWithBytes:xs_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_s = [device newBufferWithBytes:ys_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> br_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn_s = [device newBufferWithBytes:&count_small length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // --- Run GPU on small subset ---
        double tg0 = now_sec();
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p_subset];
            [enc setBuffer:bx_s offset:0 atIndex:0];
            [enc setBuffer:by_s offset:0 atIndex:1];
            [enc setBuffer:bq_s offset:0 atIndex:2];
            [enc setBuffer:br_s offset:0 atIndex:3];
            [enc setBuffer:bn_s offset:0 atIndex:4];

            NSUInteger tew = p_subset.threadExecutionWidth;
            NSUInteger mt  = p_subset.maxTotalThreadsPerThreadgroup;
            NSUInteger tgW = (mt/tew)*tew; if (tgW==0) tgW=tew; if (tgW>1024) tgW=1024;
            MTLSize tg = MTLSizeMake(tgW, 1, 1);
            MTLSize grid = MTLSizeMake(count_small, 1, 1);

            // warm-up
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double tg1 = now_sec();
        double g0 = now_sec();
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p_subset];
            [enc setBuffer:bx_s offset:0 atIndex:0];
            [enc setBuffer:by_s offset:0 atIndex:1];
            [enc setBuffer:bq_s offset:0 atIndex:2];
            [enc setBuffer:br_s offset:0 atIndex:3];
            [enc setBuffer:bn_s offset:0 atIndex:4];

            NSUInteger tew = p_subset.threadExecutionWidth;
            NSUInteger mt  = p_subset.maxTotalThreadsPerThreadgroup;
            NSUInteger tgW = (mt/tew)*tew; if (tgW==0) tgW=tew; if (tgW>1024) tgW=1024;
            MTLSize tg = MTLSizeMake(tgW, 1, 1);
            MTLSize grid = MTLSizeMake(count_small, 1, 1);

            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double g1 = now_sec();
        double gpu_secs = g1 - g0;

        // Read GPU results for small subset and scatter into final output
        uint64_t *q_small = (uint64_t*)[bq_s contents];
        uint64_t *r_small = (uint64_t*)[br_s contents];
        uint64_t sinkQ_gpu=0, sinkR_gpu=0;
        for (uint32_t s = 0; s < count_small; ++s) {
            uint32_t i = idx_small[s];
            q_out[i] = q_small[s];
            r_out[i] = r_small[s];
            sinkQ_gpu ^= q_small[s];
            sinkR_gpu ^= r_small[s];
        }

        // --- CPU on big subset ---
        double c0 = now_sec();
        uint64_t sinkQ_cpu=0, sinkR_cpu=0;
        for (uint32_t b = 0; b < count_big; ++b) {
            uint64_t q = xs_big[b] / ys_big[b];
            uint64_t r = xs_big[b] % ys_big[b];
            uint32_t i = idx_big[b];
            q_out[i] = q; r_out[i] = r;
            sinkQ_cpu ^= q; sinkR_cpu ^= r;
        }
        double c1 = now_sec();
        double cpu_secs = c1 - c0;

        // --- Validate whole hybrid vs baseline ---
        size_t mism = 0;
        uint64_t sinkQ_all=0, sinkR_all=0;
        for (uint32_t i = 0; i < N; ++i) {
            sinkQ_all ^= q_out[i];
            sinkR_all ^= r_out[i];
            if (q_out[i] != q_cpu[i] || r_out[i] != r_cpu[i]) mism++;
        }

        // --- Report ---
        printf("Hybrid split: small=%u  big=%u  (threshold=%u bits)\n", count_small, count_big, THRESH);
        stats("GPU small-subset:", gpu_secs, count_small, sinkQ_gpu, sinkR_gpu);
        stats("CPU big-subset:",  cpu_secs, count_big,  sinkQ_cpu, sinkR_cpu);
        stats("TOTAL hybrid:",    (gpu_secs + cpu_secs), N, sinkQ_all, sinkR_all);
        printf("Mismatches vs baseline: %zu (of %u)\n", mism, N);

        // speedups vs full CPU baseline
        double base_ops = (double)N / (t1 - t0);
        double hybrid_ops = (double)N / (gpu_secs + cpu_secs);
        printf("Speedup hybrid / CPU baseline: %.2fx\n", (hybrid_ops / base_ops));

        free(xs); free(ys); free(q_out); free(r_out); free(q_cpu); free(r_cpu);
        free(xs_small); free(ys_small); free(idx_small);
        free(xs_big); free(ys_big); free(idx_big);
    }
    return 0;
}
