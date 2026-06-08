// main_v2.c — Universal divide v2 (even-first + 2x Newton for odd divisors)
// Build: clang -O3 -march=native -std=c11 main_v2.c -o bench_v2
// Run:   ./bench_v2
//
// Highlights:
// - Factor out powers of two from y (y = 2^k * y1, y1 odd) -> divide only by odd part
// - Two Newton steps in fixed-point (Q_S) for the reciprocal of odd y
// - Robust signed correction to guarantee exact q,r
// - Power-of-two fastpath
// - Benchmark vs baseline and v1

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <stdlib.h>

static inline uint64_t xrng(void) {
    // xorshift64*
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

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// -------- v1 core from earlier (for comparison) --------
static inline void udiv_universal_u64_v1(uint64_t x, uint64_t y, uint64_t *q_out, uint64_t *r_out) {
    // Preconditions: y > 0
    uint32_t ybits = bitlen_u64(y);
    uint32_t S = ybits + 16;
    if (S > 56) S = 56;            // keep within 128-bit headroom

    __uint128_t oneS = ((__uint128_t)1) << S;
    __uint128_t r = oneS / y;

    // One Newton step in Q_S
    __uint128_t yr = (((__uint128_t)y) * r) >> S;
    __uint128_t two = ((__uint128_t)2) << S;
    r = (r * (two - yr)) >> S;

    __uint128_t X = (((__uint128_t)x) * r);
    uint64_t q0 = (uint64_t)(X >> S);

    __int128 diff = (__int128)x - (__int128)((__uint128_t)q0 * y);
    if (diff >= (__int128)y) { q0++; diff -= (__int128)y; }
    else if (diff < 0) { q0--; diff += (__int128)y; }

    *q_out = q0;
    *r_out = (uint64_t)diff;
}

// -------- v2: two-Newton odd-core + even-first wrapper --------

// Exact core for odd y (>0). Two Newton refinements in Q_S.
static inline void udiv_core_odd_u64(uint64_t x, uint64_t y_odd, uint64_t *q_out, uint64_t *r_out) {
    // Preconditions: y_odd > 0 and odd
    // Choose S with extra headroom so 2 Newton steps converge tightly
    uint32_t ybits = bitlen_u64(y_odd);
    uint32_t S = ybits + 32;
    if (S > 56) S = 56; // stay comfy in 128-bit

    // Initial reciprocal r ≈ (1<<S)/y in Q_S
    __uint128_t oneS = ((__uint128_t)1) << S;
    __uint128_t r = oneS / y_odd;

    // Two Newton steps: r = r * (2 - y*r) in Q_S
    __uint128_t twoS = ((__uint128_t)2) << S;

    __uint128_t yr = (((__uint128_t)y_odd) * r) >> S;
    r = (r * (twoS - yr)) >> S;

    yr = (((__uint128_t)y_odd) * r) >> S;
    r = (r * (twoS - yr)) >> S;

    // Provisional quotient
    __uint128_t X = (((__uint128_t)x) * r);
    uint64_t q0 = (uint64_t)(X >> S);

    // Robust signed correction to exact (handles off-by-1/2)
    __int128 diff = (__int128)x - (__int128)((__uint128_t)q0 * y_odd);
    if (diff >= (__int128)y_odd) { q0++; diff -= (__int128)y_odd;
        if (diff >= (__int128)y_odd) { q0++; diff -= (__int128)y_odd; } // rare
    } else if (diff < 0) { q0--; diff += (__int128)y_odd; }

    *q_out = q0;
    *r_out = (uint64_t)diff;
}

// Even-first wrapper: factor y = 2^k * y1, divide by odd y1, rebuild remainder.
static inline void udiv_universal_u64_v2(uint64_t x, uint64_t y, uint64_t *q_out, uint64_t *r_out) {
    // Preconditions: y > 0
#if defined(__clang__) || defined(__GNUC__)
    uint32_t k = (uint32_t)__builtin_ctzll(y);   // count factors of two
#else
    uint32_t k = 0; while (((y >> k) & 1ULL) == 0ULL) k++;
#endif
    uint64_t y1 = y >> k;

    // Power-of-two fastpath
    if (y1 == 1) {
        *q_out = x >> k;
        *r_out = (k ? (x & ((1ULL << k) - 1ULL)) : 0ULL);
        return;
    }

    // Split x = 2^k * t + w
    uint64_t t = (k ? (x >> k) : x);
    uint64_t w = (k ? (x & ((1ULL << k) - 1ULL)) : 0ULL);

    // Divide only by the odd core
    uint64_t q, r0;
    udiv_core_odd_u64(t, y1, &q, &r0);

    // Rebuild original remainder: r = (r0 << k) + w   (guaranteed < y)
    uint64_t r = (k ? ((r0 << k) + w) : r0);

    *q_out = q;
    *r_out = r;
}

int main(void) {
    const size_t N = 2000000; // 2M
    uint64_t *xs = (uint64_t*)malloc(N * sizeof(uint64_t));
    uint64_t *ys = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (size_t i = 0; i < N; ++i) {
        xs[i] = xrng();
        uint64_t y; do { y = xrng(); } while (y == 0);
        ys[i] = y;
    }

    // v1
    double t0 = now_sec();
    uint64_t sQ1=0, sR1=0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t q, r;
        udiv_universal_u64_v1(xs[i], ys[i], &q, &r);
        sQ1 ^= q; sR1 ^= r;
    }
    double t1 = now_sec();

    // v2
    double t2 = now_sec();
    uint64_t sQ2=0, sR2=0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t q, r;
        udiv_universal_u64_v2(xs[i], ys[i], &q, &r);
        sQ2 ^= q; sR2 ^= r;
    }
    double t3 = now_sec();

    // Baseline
    double t4 = now_sec();
    uint64_t sQb=0, sRb=0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t q = xs[i] / ys[i];
        uint64_t r = xs[i] % ys[i];
        sQb ^= q; sRb ^= r;
    }
    double t5 = now_sec();

    // Correctness sample
    size_t exact1=0, exact2=0; 
    for (size_t i = 0; i < 500000; ++i) {
        uint64_t q1,r1,q2,r2,qb,rb;
        udiv_universal_u64_v1(xs[i], ys[i], &q1, &r1);
        udiv_universal_u64_v2(xs[i], ys[i], &q2, &r2);
        qb = xs[i] / ys[i]; rb = xs[i] % ys[i];
        if (q1==qb && r1==rb) exact1++;
        if (q2==qb && r2==rb) exact2++;
    }

    double v1_ms = (t1-t0)*1e3, v2_ms=(t3-t2)*1e3, bl_ms=(t5-t4)*1e3;
    double v1_ns = (t1-t0)*1e9/(double)N;
    double v2_ns = (t3-t2)*1e9/(double)N;
    double bl_ns = (t5-t4)*1e9/(double)N;

    printf("Cases: %zu\n", N);
    printf("v1 (1x Newton):   %.2f ms (%.2f ns/op)  sink=%" PRIu64 ",%" PRIu64 "  acc(500k)=%.2f%%\n",
           v1_ms, v1_ns, sQ1, sR1, 100.0*(double)exact1/500000.0);
    printf("v2 (even+2x NR):  %.2f ms (%.2f ns/op)  sink=%" PRIu64 ",%" PRIu64 "  acc(500k)=%.2f%%\n",
           v2_ms, v2_ns, sQ2, sR2, 100.0*(double)exact2/500000.0);
    printf("Baseline / %%:     %.2f ms (%.2f ns/op)  sink=%" PRIu64 ",%" PRIu64 "\n",
           bl_ms, bl_ns, sQb, sRb);
    printf("Speed ratio v2/Baseline: %.2fx\n", (v2_ms / bl_ms));

    free(xs); free(ys);
    return 0;
}
