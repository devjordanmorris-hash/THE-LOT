#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

static inline uint64_t bitwise_pow_u64(uint64_t x, uint64_t n) {
    // symbolic shift–accumulate power pattern
    uint64_t p = x;
    for (int k = 0; k < 64 && (1ULL << k) <= n; ++k) {
        uint64_t b = (n >> k) & 1ULL;
        // grow if bit is 1, shrink if 0
        uint64_t left  = p << b;
        uint64_t right = p >> (1 - b);
        p = p + (left - right);
    }
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t N = (argc > 1) ? strtoull(argv[1], NULL, 10) : 5000000;
    uint64_t *a = malloc(N * sizeof(uint64_t));
    uint64_t *b = malloc(N * sizeof(uint64_t));
    if (!a || !b) { fprintf(stderr, "alloc fail\n"); return 1; }

    for (size_t i = 0; i < N; ++i) {
        a[i] = (rand() & 0xFFFF) + 1;
        b[i] = (rand() % 10) + 1;       // small exponents 1–10
    }

    double t0, t1;
    volatile uint64_t sink = 0;

    // baseline pow()
    t0 = now_sec();
    for (size_t i = 0; i < N; ++i)
        sink += (uint64_t)pow((double)a[i], (double)b[i]);
    t1 = now_sec();
    double base_t = t1 - t0;

    // bitwise symbolic power
    t0 = now_sec();
    for (size_t i = 0; i < N; ++i)
        sink += bitwise_pow_u64(a[i], b[i]);
    t1 = now_sec();
    double bit_t = t1 - t0;

    printf("Bitwise Power Benchmark | N=%zu\n", N);
    printf("Baseline pow(): %.3f s  |  Bitwise: %.3f s  |  Speedup: %.2fx\n",
           base_t, bit_t, base_t / bit_t);
    printf("sink=%" PRIu64 "\n", sink);

    free(a); free(b);
    return 0;
}