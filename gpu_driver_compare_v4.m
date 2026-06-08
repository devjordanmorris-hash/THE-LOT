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

static void stats(const char* tag, double secs, uint32_t N, uint64_t sinkQ, uint64_t sinkR){
    double mps = (double)N / secs / 1e6;
    printf("%-18s %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n", tag, secs, mps, sinkQ, sinkR);
}

static void run_kernel_compare(
    id<MTLComputePipelineState> pso,
    id<MTLCommandQueue> q,
    id<MTLBuffer> bx,
    id<MTLBuffer> by,
    id<MTLBuffer> bq,
    id<MTLBuffer> br,
    id<MTLBuffer> bn,
    const uint64_t* q_cpu,
    const uint64_t* r_cpu,
    uint32_t N,
    const char* tag,
    uint64_t *sinkQ, uint64_t *sinkR, double *secs)
{
    @autoreleasepool {
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
        NSUInteger tgW = (mt/tew)*tew; if (tgW==0) tgW=tew; if (tgW>512) tgW=512;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        MTLSize grid = MTLSizeMake(N, 1, 1);

        // warm-up
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // timed
        double t0k = now_sec();
        cb = [q commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBuffer:bq offset:0 atIndex:2];
        [enc setBuffer:br offset:0 atIndex:3];
        [enc setBuffer:bn offset:0 atIndex:4];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        *secs = now_sec() - t0k;
    }

    // sinks + mismatches
    uint64_t *qq = (uint64_t*)[bq contents];
    uint64_t *rr = (uint64_t*)[br contents];
    uint64_t sq=0, sr=0;
    size_t mism = 0;
    for (uint32_t i = 0; i < N; ++i) {
        sq ^= qq[i]; sr ^= rr[i];
        // compare to CPU
        if (qq[i] != q_cpu[i] || rr[i] != r_cpu[i]) mism++;
    }
    *sinkQ = sq; *sinkR = sr;
    printf("Compare vs CPU (%s): mismatches = %zu (of %u)\n", tag, mism, N);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t N = 10000000u;
        if (argc > 1) {
            unsigned long v = strtoul(argv[1], NULL, 10);
            if (v > 0 && v <= 4000000000ul) N = (uint32_t)v;
        }

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
        uint64_t cq=0, cr=0;
        for (uint32_t i = 0; i < N; ++i) { cq ^= q_cpu[i]; cr ^= r_cpu[i]; }
        double cpu_secs = t1 - t0;
        double cpu_mps = (double)N / cpu_secs / 1e6;
        printf("CPU baseline:     %.3f s  (%.2f Mops/s)  sinksums: %" PRIu64 ", %" PRIu64 "\n", cpu_secs, cpu_mps, cq, cr);

        // Metal init
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device found.\n"); return 1; }
        NSError *err = nil;

        NSString *mslPath = @"gpu_divider_compare_v4.metal";
        NSString *src = [NSString stringWithContentsOfFile:mslPath encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Failed to read metal: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLFunction> f_builtin = [lib newFunctionWithName:@"div64_builtin"];
        id<MTLFunction> f_radix4  = [lib newFunctionWithName:@"div64_radix4_evenfirst"];
        id<MTLComputePipelineState> p_builtin = [device newComputePipelineStateWithFunction:f_builtin error:&err];
        if (!p_builtin) { fprintf(stderr, "Pipeline builtin error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> p_radix4 = [device newComputePipelineStateWithFunction:f_radix4 error:&err];
        if (!p_radix4) { fprintf(stderr, "Pipeline radix4 error: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLCommandQueue> q = [device newCommandQueue];
        id<MTLBuffer> bx = [device newBufferWithBytes:xs length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [device newBufferWithBytes:ys length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> br = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn = [device newBufferWithBytes:&N length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        uint64_t sq1=0,sr1=0,sq2=0,sr2=0;
        double s1=0.0,s2=0.0;

        run_kernel_compare(p_builtin, q, bx, by, bq, br, bn, q_cpu, r_cpu, N, "GPU builtin", &sq1, &sr1, &s1);
        stats("GPU builtin:", s1, N, sq1, sr1);

        run_kernel_compare(p_radix4, q, bx, by, bq, br, bn, q_cpu, r_cpu, N, "GPU radix4", &sq2, &sr2, &s2);
        stats("GPU radix4:", s2, N, sq2, sr2);

        printf("Speedup radix4 / CPU: %.2fx\n", (s2 > 0 ? ( (double)N / s2 ) / ( (double)N / cpu_secs ) : 0.0));
        printf("Speedup radix4 / GPU_builtin: %.2fx\n", (s2 > 0 ? s1 / s2 : 0.0));

        free(xs); free(ys); free(q_cpu); free(r_cpu);
    }
    return 0;
}
