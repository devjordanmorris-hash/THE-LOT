#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

static uint64_t xrng(void){
    static uint64_t s = 0x9e3779b97f4a7c15ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device found.\\n"); return 1; }

        // Load Metal source from file gpu_divider.metal
        NSError *err = nil;
        NSString *path = @"gpu_divider.metal";
        NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Failed to read gpu_divider.metal: %s\\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal compile error: %s\\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fun = [lib newFunctionWithName:@"div64_kernel"];
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fun error:&err];
        if (!pipe) { fprintf(stderr, "Pipeline error: %s\\n", err.localizedDescription.UTF8String); return 1; }

        // Problem size
        uint32_t N = 1u<<20; // 1,048,576 divides; change with argv if desired
        if (argc > 1) { N = (uint32_t)strtoul(argv[1], NULL, 10); if (N == 0) N = 1; }

        // Host buffers
        size_t bytes = (size_t)N * sizeof(uint64_t);
        uint64_t *xs = (uint64_t*)malloc(bytes);
        uint64_t *ys = (uint64_t*)malloc(bytes);
        for (uint32_t i = 0; i < N; ++i) {
            xs[i] = xrng();
            uint64_t y;
            do { y = xrng(); } while (y == 0);
            ys[i] = y;
        }

        id<MTLCommandQueue> q = [device newCommandQueue];

        // Device buffers
        id<MTLBuffer> bx = [device newBufferWithBytes:xs length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [device newBufferWithBytes:ys length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> br = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn = [device newBufferWithBytes:&N length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Encode
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBuffer:bq offset:0 atIndex:2];
        [enc setBuffer:br offset:0 atIndex:3];
        [enc setBuffer:bn offset:0 atIndex:4];

        MTLSize grid = MTLSizeMake(N, 1, 1);
        NSUInteger w = pipe.maxTotalThreadsPerThreadgroup;
        if (w > 256) w = 256;
        MTLSize tg = MTLSizeMake(w, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // Sanity-check a few entries on CPU
        int ok = 1;
        uint64_t *qout = (uint64_t*)bq.contents;
        uint64_t *rout = (uint64_t*)br.contents;
        for (int i = 0; i < 5; ++i) {
            uint64_t x = xs[i], y = ys[i], qcpu = x / y, rcpu = x % y;
            if (qout[i] != qcpu || rout[i] != rcpu) { ok = 0; break; }
        }
        printf("GPU run complete: N=%u  sample-correct=%s\\n", N, ok ? "yes" : "no");

        // Print a small checksum to avoid deadcode elim
        uint64_t cq = 0, cr = 0;
        for (uint32_t i = 0; i < N; ++i) { cq ^= qout[i]; cr ^= rout[i]; }
        printf("sinksums: %" PRIu64 ", %" PRIu64 "\\n", cq, cr);

        free(xs); free(ys);
    }
    return 0;
}
