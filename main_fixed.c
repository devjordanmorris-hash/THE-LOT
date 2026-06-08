/*
 * Historic Reference Implementation — Universal Divider (64-bit)
 * (Corrected remainder adjustment)
 *
 * Author: Jordan Morris
 * Date: 2025
 * Location: United Kingdom
 *
 * Build: clang -O3 -march=native -std=c11 main_fixed.c -o bench
 * Run:   ./bench
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <stdlib.h>

static inline uint64_t rdtsc_fallback_rand64(void) {
    // quick & decent PRNG (xorshift64*)
    static uint64_t s = 0x9e3779b97f4a7c15ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

static inline uint32_t bitlen_u64(uint64_t x) {
    if (x == 0) return 0;
#if defined(__clang__) || defined(__GNUC__)
    return 64u - (uint32_t)__builtin_clzll(x);
#else
    // portable fallback
    uint32_t n = 0; while (x) { n++; x >>= 1; } return n;
#endif
}

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---- Universal divide (64-bit), fixed-point reciprocal with 1 Newton + correction
// Returns q,r such that x = q*y + r  (0 <= r < y).
static inline void udiv_universal_u64(uint64_t x, uint64_t y, uint64_t *q_out, uint64_t *r_out) {
    // Preconditions: y > 0
    uint32_t ybits = bitlen_u64(y);
    uint32_t S = ybits + 16;
    if (S > 56) S = 56;  // keep intermediates within 128-bit headroom

    // Initial reciprocal estimate in Q_S
    __uint128_t oneS = ((__uint128_t)1) << S;
    __uint128_t r = oneS / y;

    // One Newton step in Q_S: r = r * (2 - y*r)
    __uint128_t yr = (((__uint128_t)y) * r) >> S;
    __uint128_t two = ((__uint128_t)2) << S;
    r = (r * (two - yr)) >> S;

    // Provisional quotient
    __uint128_t X = (((__uint128_t)x) * r);
    uint64_t q0 = (uint64_t)(X >> S);

    // Robust single-step correction using signed 128-bit diff
    __int128 diff = (__int128)( (__int128)x - (__int128)((__uint128_t)q0 * y) );
    if (diff >= (__int128)y) {                // q0 too small by at least 1
        q0++; diff -= (__int128)y;
        if (diff >= (__int128)y) {            // very rare: off by 2
            q0++; diff -= (__int128)y;
        }
    } else if (diff < 0) {                    // q0 too big by 1
        q0--; diff += (__int128)y;
    }

    *q_out = q0;
    *r_out = (uint64_t)diff;
}

int main(void) {
    const size_t N = 2000000;   // 2M cases
    size_t exact = 0;

    // Generate random inputs
    uint64_t *xs = (uint64_t*)malloc(N * sizeof(uint64_t));
    uint64_t *ys = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (size_t i = 0; i < N; ++i) {
        xs[i] = rdtsc_fallback_rand64();
        uint64_t y;
        do { y = rdtsc_fallback_rand64(); } while (y == 0); // avoid y=0
        ys[i] = y;
    }

    // ---- Benchmark universal divide
    double t0 = now_sec();
    uint64_t sumQ = 0, sumR = 0; // prevent dead-code elim
    for (size_t i = 0; i < N; ++i) {
        uint64_t q, r;
        udiv_universal_u64(xs[i], ys[i], &q, &r);
        sumQ ^= q; sumR ^= r;
    }
    double t1 = now_sec();

    // ---- Benchmark baseline (C / and %)
    double t2 = now_sec();
    uint64_t sumQb = 0, sumRb = 0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t q = xs[i] / ys[i];
        uint64_t r = xs[i] % ys[i];
        sumQb ^= q; sumRb ^= r;
    }
    double t3 = now_sec();

    // ---- Correctness check on a subset
    for (size_t i = 0; i < 200000; ++i) {
        uint64_t q, r, qb, rb;
        udiv_universal_u64(xs[i], ys[i], &q, &r);
        qb = xs[i] / ys[i];
        rb = xs[i] % ys[i];
        if (q == qb && r == rb) exact++;
    }

    double universal_ms = (t1 - t0) * 1e3;
    double baseline_ms  = (t3 - t2) * 1e3;
    double upc_ns = (t1 - t0) * 1e9 / (double)N;
    double bpc_ns = (t3 - t2) * 1e9 / (double)N;

    printf("Cases: %zu\\n", N);
    printf("Universal:  %.2f ms  (%.2f ns/op)  sinksum=%" PRIu64 ",%" PRIu64 "\\n",
           universal_ms, upc_ns, sumQ, sumR);
    printf("Baseline:   %.2f ms  (%.2f ns/op)  sinksumb=%" PRIu64 ",%" PRIu64 "\\n",
           baseline_ms, bpc_ns, sumQb, sumRb);
    printf("Correctness (subset 200k): %.2f%%\\n", 100.0 * (double)exact / 200000.0);
    printf("Speed ratio Universal/Baseline: %.2fx\\n", (universal_ms / baseline_ms));

    free(xs); free(ys);
    return 0;
}
