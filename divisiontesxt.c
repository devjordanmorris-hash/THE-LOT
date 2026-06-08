// main.c — Fast 64-bit integer division (Newton reciprocal + single correction)
// Portable C11; tested with clang on macOS.
// Build (Terminal): clang -O3 -march=native -std=c11 main.c -o fastdiv
// Run:   ./fastdiv            (quick benchmark)
//        ./fastdiv  X Y       (compute X/Y and X%Y once)
//
// This replaces the slow hardware divide with: q ≈ (x * ((1<<S)/y)) >> S
// then a single ±y correction to make (q,r) exact.

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

// -------- timing (monotonic) --------
static inline double now_sec(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// -------- utilities --------
static inline uint32_t bitlen_u64(uint64_t x) {
    if (!x) return 0;
#if defined(__clang__) || defined(__GNUC__)
    return 64u - (uint32_t)__builtin_clzll(x);
#else
    uint32_t n = 0; while (x) { n++; x >>= 1; } return n;
#endif
}

// xorshift64* PRNG (good enough for benchmarks)
static inline uint64_t rng64(void) {
    static uint64_t s = 0x9E3779B97F4A7C15ull;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1Dull;
}

// -------- core: universal divide (no hardware divide in hot path) --------
// Computes q,r such that x = q*y + r, with 0 <= r < y.
static inline void udiv_universal_u64(uint64_t x, uint64_t y,
                                      uint64_t *q_out, uint64_t *r_out) {
    // Precondition
    if (y == 0) { // keep behaviour consistent with / and %
        *q_out = 0; *r_out = 0; return;
    }

    // Choose fractional precision S: y_bits + guard.
    // 16 guard bits are enough for one Newton step + single correction.
    uint32_t ybits = bitlen_u64(y);
    uint32_t S = ybits + 16;
    if (S > 56) S = 56; // keep intermediates comfy in 128-bit

    // r0 ≈ (1<<S)/y  in Q_S
    __uint128_t oneS = ((__uint128_t)1) << S;
    __uint128_t r = oneS / y;

    // One Newton step in Q_S: r <- r * (2 - y*r)
    __uint128_t yr = (((__uint128_t)y) * r) >> S;
    __uint128_t two = ((__uint128_t)2) << S;
    r = (r * (two - yr)) >> S;

    // q0 = (x * r) >> S
    __uint128_t Xr = ((__uint128_t)x) * r;
    uint64_t q0 = (uint64_t)(Xr >> S);

    // Single correction to nail exactness
    __int128 rem = (__int128)( (__uint128_t)x - (__uint128_t)q0 * y );
    if (rem >= ( __int128) y)      { q0++; rem -= ( __int128) y; }
    else if (rem < 0)              { q0--; rem += ( __int128) y; }

    *q_out = q0;
    *r_out = (uint64_t)rem;
}

// -------- quick correctness check --------
static int selftest(size_t cases) {
    size_t ok = 0;
    for (size_t i = 0; i < cases; ++i) {
        uint64_t x = rng64();
        uint64_t y;
        do { y = rng64(); } while (y == 0);

        uint64_t q, r;
        udiv_universal_u64(x, y, &q, &r);

        uint64_t qb = x / y;
        uint64_t rb = x % y;

        if (q == qb && r == rb) ok++;
    }
    printf("Correctness: %zu/%zu (%.2f%%)\n", ok, cases, 100.0*ok/(double)cases);
    return (ok == cases) ? 0 : 1;
}

// -------- micro-benchmark --------
static void benchmark(size_t N) {
    uint64_t sinkQ = 0, sinkR = 0;

    // generate inputs once
    uint64_t *xs = (uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t *ys = (uint64_t*)malloc(N*sizeof(uint64_t));
    for (size_t i = 0; i < N; ++i) {
        xs[i] = rng64();
        do { ys[i] = rng64(); } while (ys[i] == 0);
    }

    // universal
    double t0 = now_sec();
    for (size_t i = 0; i < N; ++i) {
        uint64_t q, r;
        udiv_universal_u64(xs[i], ys[i], &q, &r);
        sinkQ ^= q; sinkR ^= r;
    }
    double t1 = now_sec();

    // baseline
    double t2 = now_sec();
    for (size_t i = 0; i < N; ++i) {
        uint64_t q = xs[i] / ys[i];
        uint64_t r = xs[i] % ys[i];
        sinkQ ^= q; sinkR ^= r;
    }
    double t3 = now_sec();

    double ms_univ = (t1 - t0)*1e3;
    double ms_base = (t3 - t2)*1e3;
    double ns_univ = (t1 - t0)*1e9 / (double)N;
    double ns_base = (t3 - t2)*1e9 / (double)N;

    printf("Cases: %zu\n", N);
    printf("Universal: %.2f ms  (%.2f ns/op)\n", ms_univ, ns_univ);
    printf("Baseline : %.2f ms  (%.2f ns/op)\n", ms_base, ns_base);
    printf("Speed ratio (Universal/Baseline): %.3fx\n", ms_univ / ms_base);
    printf("Sink: %" PRIu64 ", %" PRIu64 "\n", sinkQ, sinkR);

    free(xs); free(ys);
}

// -------- main --------
int main(int argc, char** argv) {
    if (argc == 3) {
        // Single-shot mode: ./fastdiv X Y
        uint64_t x = strtoull(argv[1], NULL, 0);
        uint64_t y = strtoull(argv[2], NULL, 0);
        if (y == 0) { fprintf(stderr, "Error: division by zero\n"); return 1; }
        uint64_t q, r;
        udiv_universal_u64(x, y, &q, &r);
        printf("fast q=%" PRIu64 ", r=%" PRIu64 "\n", q, r);
        printf("base q=%" PRIu64 ", r=%" PRIu64 "\n", x/y, x%y);
        return 0;
    }

    // Default: quick correctness + benchmark
    puts("Fast 64-bit divide (Newton reciprocal + single correction)");
    int fail = selftest(200000);          // 200k pairs
    if (fail) { fprintf(stderr, "Selftest failed.\n"); return 1; }
    benchmark(2000000);                   // 2M pairs
    return 0;
}