#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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
static int active_cpu_count(void){
#ifndef HW_AVAILCPU
#define HW_AVAILCPU 25
#endif
    int ncpu = 1; size_t len = sizeof(ncpu);
    int mib[2] = {CTL_HW, HW_AVAILCPU};
    if (sysctl(mib, 2, &ncpu, &len, NULL, 0) != 0 || ncpu < 1) {
#ifdef HW_NCPU
        mib[1] = HW_NCPU;
        len = sizeof(ncpu);
        if (sysctl(mib, 2, &ncpu, &len, NULL, 0) != 0 || ncpu < 1) ncpu = 1;
#else
        ncpu = 1;
#endif
    }
    return ncpu;
}
static void stats(const char* tag, double secs, uint64_t ops, uint64_t sinkQ, uint64_t sinkR){
    double mps = (secs > 0 ? (double)ops / secs / 1e6 : 0.0);
    printf("%-22s %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n", tag, secs, mps, sinkQ, sinkR);
}

static void gpu_subset_run(id<MTLComputePipelineState> pso, id<MTLCommandQueue> q,
                           id<MTLBuffer> bx, id<MTLBuffer> by, id<MTLBuffer> bq, id<MTLBuffer> br, id<MTLBuffer> bn,
                           uint32_t n_small, uint64_t *sinkQ, uint64_t *sinkR, double *secs){
    double g0 = now_sec();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:by offset:0 atIndex:1];
    [enc setBuffer:bq offset:0 atIndex:2];
    [enc setBuffer:br offset:0 atIndex:3];
    [enc setBuffer:bn offset:0 atIndex:4];

    NSUInteger tew = pso.threadExecutionWidth;
    NSUInteger mt  = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger tgW = (mt/tew)*tew; if (tgW==0) tgW=tew; if (tgW>1024) tgW=1024;
    MTLSize tg = MTLSizeMake(tgW, 1, 1);
    MTLSize grid = MTLSizeMake(n_small, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    *secs = now_sec() - g0;

    uint64_t *q_small = (uint64_t*)[bq contents];
    uint64_t *r_small = (uint64_t*)[br contents];
    uint64_t sq=0,sr=0;
    for (uint32_t i=0;i<n_small;++i){ sq ^= q_small[i]; sr ^= r_small[i]; }
    *sinkQ = sq; *sinkR = sr;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t N = 10000000u;
        uint32_t THRESH = 63u;      // ~50% to GPU for uniform64
        int mixed = 0;              // 0=uniform64, 1=mixed50
        int cpu_threads = 0;        // 0=auto
        int gpu_mode = 2;           // 0=builtin, 1=radix4_ilp, 2=recip_nr_ilp
        if (argc > 1) { unsigned long v = strtoul(argv[1], NULL, 10); if (v>0 && v<=4000000000ul) N=(uint32_t)v; }
        if (argc > 2) { unsigned long t = strtoul(argv[2], NULL, 10); if (t>=8 && t<=63) THRESH=(uint32_t)t; }
        if (argc > 3) { mixed = (int)strtoul(argv[3], NULL, 10); }
        if (argc > 4) { cpu_threads = (int)strtoul(argv[4], NULL, 10); }
        if (argc > 5) { gpu_mode = (int)strtoul(argv[5], NULL, 10); }
        if (cpu_threads <= 0) cpu_threads = active_cpu_count();
        const char* gname = (gpu_mode==0? "builtin" : (gpu_mode==1? "radix4_ilp":"recip_nr_ilp"));
        printf("Hybrid v9: THRESH<=%u -> GPU | dataset=%s | CPU_threads=%d | GPU_mode=%s\n",
               THRESH, mixed ? "mixed50" : "uniform64", cpu_threads, gname);

        size_t bytes = (size_t)N * sizeof(uint64_t);
        uint64_t *xs = (uint64_t*)malloc(bytes);
        uint64_t *ys = (uint64_t*)malloc(bytes);
        uint64_t *q_out = (uint64_t*)malloc(bytes);
        uint64_t *r_out = (uint64_t*)malloc(bytes);
        uint64_t *q_ref = (uint64_t*)malloc(bytes);
        uint64_t *r_ref = (uint64_t*)malloc(bytes);
        if (!xs || !ys || !q_out || !r_out || !q_ref || !r_ref) { fprintf(stderr, "alloc failed\n"); return 1; }

        // Inputs
        for (uint32_t i = 0; i < N; ++i) {
            xs[i] = xrng();
            uint64_t y; do { y = xrng(); } while (y == 0);
            ys[i] = y;
        }
        if (mixed){
            for (uint32_t i=0;i<N;++i){
                if ((i&1u)==0){ ys[i] &= ((1ULL<<32)-1ULL); if (!ys[i]) ys[i]=1; }
                else { ys[i] |= (1ULL<<55); }
            }
        }

        // CPU baseline
        double t0=now_sec();
        for (uint32_t i=0;i<N;++i){ q_ref[i]=xs[i]/ys[i]; r_ref[i]=xs[i]%ys[i]; }
        double t1=now_sec();
        stats("CPU baseline (ref):", t1-t0, N, 0, 0);

        // Partition
        uint32_t count_small=0, count_big=0;
        for (uint32_t i=0;i<N;++i){ if (bitlen_u64(ys[i]) <= THRESH) count_small++; else count_big++; }
        uint64_t *xs_small=(uint64_t*)malloc((size_t)count_small*sizeof(uint64_t));
        uint64_t *ys_small=(uint64_t*)malloc((size_t)count_small*sizeof(uint64_t));
        uint32_t *idx_small=(uint32_t*)malloc((size_t)count_small*sizeof(uint32_t));
        uint64_t *xs_big=(uint64_t*)malloc((size_t)count_big*sizeof(uint64_t));
        uint64_t *ys_big=(uint64_t*)malloc((size_t)count_big*sizeof(uint64_t));
        uint32_t *idx_big=(uint32_t*)malloc((size_t)count_big*sizeof(uint32_t));
        if (!xs_small||!ys_small||!idx_small||!xs_big||!ys_big||!idx_big){ fprintf(stderr,"alloc2 failed\n"); return 1; }
        uint32_t ps=0,pb=0;
        for (uint32_t i=0;i<N;++i){
            if (bitlen_u64(ys[i]) <= THRESH){ xs_small[ps]=xs[i]; ys_small[ps]=ys[i]; idx_small[ps]=i; ps++; }
            else { xs_big[pb]=xs[i]; ys_big[pb]=ys[i]; idx_big[pb]=i; pb++; }
        }

        // Metal init
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr,"No Metal device.\n"); return 1; }
        NSError *err=nil;
        NSString *mslPath = @"gpu_divider_hybrid_v9.metal";
        NSString *src = [NSString stringWithContentsOfFile:mslPath encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr,"Read metal failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr,"Metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> f_builtin = [lib newFunctionWithName:@"div64_builtin_subset"];
        id<MTLFunction> f_radix4  = [lib newFunctionWithName:@"div64_radix4_ilp_subset"];
        id<MTLFunction> f_recip   = [lib newFunctionWithName:@"div64_recip_nr_ilp_subset"];
        id<MTLComputePipelineState> p_builtin = [device newComputePipelineStateWithFunction:f_builtin error:&err];
        id<MTLComputePipelineState> p_radix4  = [device newComputePipelineStateWithFunction:f_radix4 error:&err];
        id<MTLComputePipelineState> p_recip   = [device newComputePipelineStateWithFunction:f_recip error:&err];
        if (!p_builtin || !p_radix4 || !p_recip){ fprintf(stderr,"Pipeline error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> cq = [device newCommandQueue];

        // Buffers for small subset
        size_t bytes_small = (size_t)count_small*sizeof(uint64_t);
        id<MTLBuffer> bx_s = [device newBufferWithBytes:xs_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_s = [device newBufferWithBytes:ys_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> br_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn_s = [device newBufferWithBytes:&count_small length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // --- Benchmark GPU small-subset alone (all modes) ---
        uint64_t sq=0,sr=0; double sg=0;
        gpu_subset_run(p_builtin, cq, bx_s, by_s, bq_s, br_s, bn_s, count_small, &sq, &sr, &sg);
        stats("GPU subset builtin:", sg, count_small, sq, sr);
        gpu_subset_run(p_radix4, cq, bx_s, by_s, bq_s, br_s, bn_s, count_small, &sq, &sr, &sg);
        stats("GPU subset radix4:", sg, count_small, sq, sr);
        gpu_subset_run(p_recip, cq, bx_s, by_s, bq_s, br_s, bn_s, count_small, &sq, &sr, &sg);
        stats("GPU subset recipNR:", sg, count_small, sq, sr);

        // --- Overlapped hybrid using selected GPU mode ---
        id<MTLComputePipelineState> p_use = (gpu_mode==0 ? p_builtin : (gpu_mode==1 ? p_radix4 : p_recip));
        __block double gpu_secs=0.0, cpu_secs=0.0;
        __block uint64_t sinkQ_gpu=0, sinkR_gpu=0, sinkQ_cpu=0, sinkR_cpu=0;
        dispatch_group_t group = dispatch_group_create();
        dispatch_queue_t q_cpu = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // GPU task
        dispatch_group_async(group, q_cpu, ^{
            gpu_subset_run(p_use, cq, bx_s, by_s, bq_s, br_s, bn_s, count_small, &sinkQ_gpu, &sinkR_gpu, &gpu_secs);
            // scatter back
            uint64_t *q_small = (uint64_t*)[bq_s contents];
            uint64_t *r_small = (uint64_t*)[br_s contents];
            for (uint32_t s=0;s<count_small;++s){ uint32_t i=idx_small[s]; q_out[i]=q_small[s]; r_out[i]=r_small[s]; }
        });

        // CPU task
        dispatch_group_async(group, q_cpu, ^{
            double c0=now_sec();
            int T = active_cpu_count();
            uint32_t B = count_big;
            uint32_t chunk = (B + T - 1) / T;
            dispatch_apply(T, q_cpu, ^(size_t t){
                uint32_t start=(uint32_t)t*chunk;
                uint32_t end=start+chunk; if (end>B) end=B;
                uint64_t lq=0, lr=0;
                for (uint32_t b=start;b<end;++b){
                    uint64_t qv = xs_big[b] / ys_big[b];
                    uint64_t rv = xs_big[b] % ys_big[b];
                    uint32_t i = idx_big[b];
                    q_out[i]=qv; r_out[i]=rv;
                    lq ^= qv; lr ^= rv;
                }
                __sync_fetch_and_xor(&sinkQ_cpu, lq);
                __sync_fetch_and_xor(&sinkR_cpu, lr);
            });
            cpu_secs = now_sec() - c0;
        });

        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

        // Validate & report
        size_t mism=0; uint64_t sinkQ_all=0, sinkR_all=0;
        for (uint32_t i=0;i<N;++i){ sinkQ_all ^= q_out[i]; sinkR_all ^= r_out[i]; if (q_out[i]!=q_ref[i]||r_out[i]!=r_ref[i]) mism++; }

        double hybrid_secs = (gpu_secs > cpu_secs ? gpu_secs : cpu_secs);
        stats("TOTAL hybrid (ovlp):", hybrid_secs, N, sinkQ_all, sinkR_all);
        printf("Mismatches vs baseline: %zu (of %u)\n", mism, N);

        double base_ops = (double)N / (t1 - t0);
        double hybrid_ops = (double)N / hybrid_secs;
        printf("Speedup hybrid / CPU baseline: %.2fx\n", (hybrid_ops / base_ops));

        free(xs); free(ys); free(q_out); free(r_out); free(q_ref); free(r_ref);
        free(xs_small); free(ys_small); free(idx_small);
        free(xs_big); free(ys_big); free(idx_big);
    }
    return 0;
}
