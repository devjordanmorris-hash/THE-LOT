#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// --- Config ---
// Set to 1 to enable CPU reciprocal LUT for repeated big-Y denominators
#ifndef USE_LUT
#define USE_LUT 0
#endif
#define LUT_SIZE 64

typedef struct { uint64_t y; uint32_t k; uint32_t S; __uint128_t rQ; int valid; } lut_entry;

// --- Utils ---
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
static inline uint32_t ctz_u64(uint64_t x) {
#if defined(__clang__) || defined(__GNUC__)
    return (uint32_t)__builtin_ctzll(x);
#else
    uint32_t k=0; while (((x>>k)&1ULL)==0ULL) k++; return k;
#endif
}
static void stats(const char* tag, double secs, uint64_t ops, uint64_t sinkQ, uint64_t sinkR){
    double mps = (double)ops / secs / 1e6;
    printf("%-22s %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n", tag, secs, mps, sinkQ, sinkR);
}

// --- CPU big-Y path ---
// A) Plain baseline
static inline void cpu_div_plain(uint64_t x, uint64_t y, uint64_t* q, uint64_t* r){
    *q = x / y; *r = x % y;
}

// B) Optional LUT: cache fixed-point reciprocal of odd part and use NR*2 + correction
#if USE_LUT
static inline __uint128_t mul64x64_128(uint64_t a, uint64_t b){ return ((__uint128_t)a)*b; }

static inline void compute_recip(uint64_t y, uint32_t* outS, __uint128_t* outR){
    uint32_t k = ctz_u64(y);
    uint64_t y1 = y >> k; // odd
    uint32_t ybits = bitlen_u64(y1);
    uint32_t S = ybits + 32; if (S > 56) S = 56;
    __uint128_t oneS = ((__uint128_t)1) << S;
    __uint128_t r = oneS / y1;
    __uint128_t twoS = ((__uint128_t)2) << S;
    __uint128_t yr = (mul64x64_128(y1, (uint64_t)r)) >> S;
    r = (r * (twoS - yr)) >> S;
    yr = (mul64x64_128(y1, (uint64_t)r)) >> S;
    r = (r * (twoS - yr)) >> S;
    *outS = S; *outR = r;
}
static inline lut_entry* lut_find_or_fill(lut_entry* lut, uint64_t y){
    uint64_t key = y | 1ULL; // avoid y=0
    uint32_t idx = (uint32_t)((key ^ (key>>17) ^ (key>>32)) % LUT_SIZE);
    for (uint32_t t=0; t<LUT_SIZE; ++t){
        uint32_t i = (idx + t) % LUT_SIZE;
        if (lut[i].valid && lut[i].y == y) return &lut[i];
        if (!lut[i].valid){
            // fill
            lut[i].y = y;
            lut[i].k = ctz_u64(y);
            compute_recip(y, &lut[i].S, &lut[i].rQ);
            lut[i].valid = 1;
            return &lut[i];
        }
    }
    // fallback overwrite idx
    lut[idx].y = y;
    lut[idx].k = ctz_u64(y);
    compute_recip(y, &lut[idx].S, &lut[idx].rQ);
    lut[idx].valid = 1;
    return &lut[idx];
}
static inline void cpu_div_lut(uint64_t x, uint64_t y, lut_entry* lut, uint64_t* q_out, uint64_t* r_out){
    lut_entry* e = lut_find_or_fill(lut, y);
    uint32_t k = e->k;
    uint64_t y1 = y >> k;
    uint64_t t  = (k ? (x >> k) : x);
    __uint128_t X = ((__uint128_t)t) * (uint64_t)e->rQ;
    uint64_t q0 = (uint64_t)(X >> e->S);

    __int128 diff = (__int128)t - (__int128)((__uint128_t)q0 * y1);
    if (diff >= (__int128)y1) { q0++; diff -= (__int128)y1; if (diff >= (__int128)y1){ q0++; diff -= (__int128)y1; } }
    else if (diff < 0) { q0--; diff += (__int128)y1; }

    uint64_t r0 = (uint64_t)diff;
    uint64_t r = (k ? (r0 << k) + (x & ((1ULL<<k)-1ULL)) : r0);
    *q_out = q0; *r_out = r;
}
#endif

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t N = 10000000u;
        uint32_t THRESH = 63u;    // default split: ~50/50 for uniform 64-bit
        int mixed = 0;            // 0=uniform, 1=mixed 50/50 (small vs big)
        if (argc > 1) {
            unsigned long v = strtoul(argv[1], NULL, 10);
            if (v > 0 && v <= 4000000000ul) N = (uint32_t)v;
        }
        if (argc > 2) {
            unsigned long t = strtoul(argv[2], NULL, 10);
            if (t >= 8 && t <= 63) THRESH = (uint32_t)t;
        }
        if (argc > 3) {
            mixed = (int)strtoul(argv[3], NULL, 10); // 1=force 50/50 distribution
        }
        printf("Hybrid threshold: bitlen(y) <= %u -> GPU (radix-4), else CPU  | dataset=%s | USE_LUT=%d\n",
               THRESH, mixed ? "mixed50" : "uniform64", USE_LUT);

        size_t bytes = (size_t)N * sizeof(uint64_t);
        uint64_t *xs = (uint64_t*)malloc(bytes);
        uint64_t *ys = (uint64_t*)malloc(bytes);
        uint64_t *q_out = (uint64_t*)malloc(bytes);
        uint64_t *r_out = (uint64_t*)malloc(bytes);
        uint64_t *q_cpu_ref = (uint64_t*)malloc(bytes);
        uint64_t *r_cpu_ref = (uint64_t*)malloc(bytes);
        if (!xs || !ys || !q_out || !r_out || !q_cpu_ref || !r_cpu_ref) { fprintf(stderr, "alloc failed\n"); return 1; }

        // Generate inputs
        if (!mixed){
            for (uint32_t i = 0; i < N; ++i) {
                xs[i] = xrng();
                uint64_t y; do { y = xrng(); } while (y == 0);
                ys[i] = y;
            }
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                uint64_t x = xrng();
                uint64_t y;
                if ((i & 1u) == 0) {
                    // Small y: ensure <= 32 bits
                    do { y = xrng() & ((1ULL<<32) - 1ULL); } while (y == 0);
                } else {
                    // Large y: ensure >= 56 bits
                    y = xrng() | (1ULL<<55);
                    if (y == 0) y = 1;
                }
                xs[i] = x; ys[i] = y;
            }
        }

        // CPU baseline for correctness
        double t0 = now_sec();
        for (uint32_t i = 0; i < N; ++i) {
            q_cpu_ref[i] = xs[i] / ys[i];
            r_cpu_ref[i] = xs[i] % ys[i];
        }
        double t1 = now_sec();
        stats("CPU baseline (ref):", t1 - t0, N, 0, 0);

        // Partition
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
            if (bitlen_u64(ys[i]) <= THRESH) { xs_small[ps]=xs[i]; ys_small[ps]=ys[i]; idx_small[ps]=i; ps++; }
            else { xs_big[pb]=xs[i]; ys_big[pb]=ys[i]; idx_big[pb]=i; pb++; }
        }

        // Metal setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device found.\n"); return 1; }
        NSError *err = nil;
        NSString *mslPath = @"gpu_divider_hybrid_v6.metal";
        NSString *src = [NSString stringWithContentsOfFile:mslPath encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Failed to read metal: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> f_subset = [lib newFunctionWithName:@"div64_radix4_evenfirst_subset"];
        id<MTLComputePipelineState> p_subset = [device newComputePipelineStateWithFunction:f_subset error:&err];
        if (!p_subset) { fprintf(stderr, "Pipeline subset error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [device newCommandQueue];

        // GPU run on small subset
        size_t bytes_small = (size_t)count_small * sizeof(uint64_t);
        id<MTLBuffer> bx_s = [device newBufferWithBytes:xs_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_s = [device newBufferWithBytes:ys_small length:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> br_s = [device newBufferWithLength:bytes_small options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn_s = [device newBufferWithBytes:&count_small length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        double g0=now_sec();
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
        double g1=now_sec();
        double gpu_secs = g1 - g0;
        uint64_t *q_small = (uint64_t*)[bq_s contents];
        uint64_t *r_small = (uint64_t*)[br_s contents];
        uint64_t sinkQ_gpu=0, sinkR_gpu=0;
        for (uint32_t s = 0; s < count_small; ++s) {
            uint32_t i = idx_small[s];
            q_out[i] = q_small[s]; r_out[i] = r_small[s];
            sinkQ_gpu ^= q_small[s]; sinkR_gpu ^= r_small[s];
        }

        // CPU big subset
        double c0=now_sec();
        uint64_t sinkQ_cpu=0, sinkR_cpu=0;
    #if USE_LUT
        lut_entry lut[LUT_SIZE]; memset(lut, 0, sizeof(lut));
        for (uint32_t b = 0; b < count_big; ++b) {
            uint64_t x = xs_big[b], y = ys_big[b], qv, rv;
            cpu_div_lut(x, y, lut, &qv, &rv);
            uint32_t i = idx_big[b];
            q_out[i]=qv; r_out[i]=rv;
            sinkQ_cpu ^= qv; sinkR_cpu ^= rv;
        }
    #else
        for (uint32_t b = 0; b < count_big; ++b) {
            uint64_t qv = xs_big[b] / ys_big[b];
            uint64_t rv = xs_big[b] % ys_big[b];
            uint32_t i = idx_big[b];
            q_out[i]=qv; r_out[i]=rv;
            sinkQ_cpu ^= qv; sinkR_cpu ^= rv;
        }
    #endif
        double c1=now_sec();
        double cpu_secs = c1 - c0;

        // Validate
        size_t mism=0; uint64_t sinkQ_all=0, sinkR_all=0;
        for (uint32_t i = 0; i < N; ++i) {
            sinkQ_all ^= q_out[i]; sinkR_all ^= r_out[i];
            if (q_out[i] != q_cpu_ref[i] || r_out[i] != r_cpu_ref[i]) mism++;
        }

        // Report
        printf("Split: small=%u  big=%u  (THRESH=%u  dataset=%s)\n", count_small, count_big, THRESH, mixed ? "mixed50" : "uniform64");
        stats("GPU small-subset:", gpu_secs, count_small, sinkQ_gpu, sinkR_gpu);
        stats("CPU big-subset:",  cpu_secs, count_big,  sinkQ_cpu, sinkR_cpu);
        stats("TOTAL hybrid:",    (gpu_secs + cpu_secs), N, sinkQ_all, sinkR_all);
        printf("Mismatches vs baseline: %zu (of %u)\n", mism, N);

        double base_ops = (double)N / (t1 - t0);
        double hybrid_ops = (double)N / (gpu_secs + cpu_secs);
        printf("Speedup hybrid / CPU baseline: %.2fx\n", (hybrid_ops / base_ops));

        free(xs); free(ys); free(q_out); free(r_out); free(q_cpu_ref); free(r_cpu_ref);
        free(xs_small); free(ys_small); free(idx_small);
        free(xs_big); free(ys_big); free(idx_big);
    }
    return 0;
}
