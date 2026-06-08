{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\info
{\author Jordan Morris}}\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // main.c \'97 64-bit universal divider vs baseline\
// Build: clang -O3 -march=native -std=c11 main.c -o bench\
// Run:   ./bench\
\
#include <stdio.h>\
#include <stdint.h>\
#include <inttypes.h>\
#include <time.h>\
#include <stdlib.h>\
\
static inline uint64_t rdtsc_fallback_rand64(void) \{\
    // quick & decent PRNG (xorshift64*)\
    static uint64_t s = 0x9e3779b97f4a7c15ULL;\
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;\
    return s * 0x2545F4914F6CDD1DULL;\
\}\
\
static inline uint32_t bitlen_u64(uint64_t x) \{\
    if (x == 0) return 0;\
#if defined(__clang__) || defined(__GNUC__)\
    return 64u - (uint32_t)__builtin_clzll(x);\
#else\
    // portable fallback\
    uint32_t n = 0; while (x) \{ n++; x >>= 1; \} return n;\
#endif\
\}\
\
static inline double now_sec(void) \{\
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);\
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;\
\}\
\
// ---- Universal divide (64-bit), fixed-point reciprocal with 1 Newton + correction\
// Returns q,r such that x = q*y + r  (0 <= r < y), no hardware divide in the math path.\
static inline void udiv_universal_u64(uint64_t x, uint64_t y, uint64_t *q_out, uint64_t *r_out) \{\
    // Preconditions: y > 0\
    // Choose S with headroom (y_bits + 16 is plenty for 1 Newton + single correction)\
    uint32_t ybits = bitlen_u64(y);\
    uint32_t S = ybits + 16;\
    if (S > 56) S = 56;            // keep intermediates comfy inside 128 bits\
\
    // r \uc0\u8776  (1<<S)/y in Q_S\
    __uint128_t oneS = ((__uint128_t)1) << S;\
    __uint128_t r = oneS / y;\
\
    // One Newton step: r = r * (2 - y*r)    (Q_S arithmetic)\
    __uint128_t yr = (((__uint128_t)y) * r) >> S;\
    __uint128_t two = ((__uint128_t)2) << S;\
    r = (r * (two - yr)) >> S;\
\
    // q0 = (x * r) >> S\
    __uint128_t X = (((__uint128_t)x) * r);\
    uint64_t q0 = (uint64_t)(X >> S);\
\
    // single correction to nail exact (q,r)\
    __uint128_t rem = (__uint128_t)x - (__uint128_t)q0 * y;\
    if (rem >= y) \{ q0++; rem -= y; \}\
    else if ((int64_t)rem < 0) \{ q0--; rem += y; \}\
\
    *q_out = q0;\
    *r_out = (uint64_t)rem;\
\}\
\
int main(void) \{\
    const size_t N = 2000000;   // 2M cases \'97 adjust if you want shorter runs\
    size_t exact = 0;\
\
    // Generate random inputs\
    uint64_t *xs = (uint64_t*)malloc(N * sizeof(uint64_t));\
    uint64_t *ys = (uint64_t*)malloc(N * sizeof(uint64_t));\
    for (size_t i = 0; i < N; ++i) \{\
        xs[i] = rdtsc_fallback_rand64();\
        uint64_t y;\
        do \{ y = rdtsc_fallback_rand64(); \} while (y == 0); // avoid y=0\
        ys[i] = y;\
    \}\
\
    // ---- Benchmark universal divide\
    double t0 = now_sec();\
    uint64_t sumQ = 0, sumR = 0; // prevent dead-code elim\
    for (size_t i = 0; i < N; ++i) \{\
        uint64_t q, r;\
        udiv_universal_u64(xs[i], ys[i], &q, &r);\
        sumQ ^= q; sumR ^= r;\
    \}\
    double t1 = now_sec();\
\
    // ---- Benchmark baseline (C / and %)\
    double t2 = now_sec();\
    uint64_t sumQb = 0, sumRb = 0;\
    for (size_t i = 0; i < N; ++i) \{\
        uint64_t q = xs[i] / ys[i];\
        uint64_t r = xs[i] % ys[i];\
        sumQb ^= q; sumRb ^= r;\
    \}\
    double t3 = now_sec();\
\
    // ---- Correctness check on a subset (or all if you want)\
    for (size_t i = 0; i < 200000; ++i) \{\
        uint64_t q, r, qb, rb;\
        udiv_universal_u64(xs[i], ys[i], &q, &r);\
        qb = xs[i] / ys[i];\
        rb = xs[i] % ys[i];\
        if (q == qb && r == rb) exact++;\
    \}\
\
    double universal_ms = (t1 - t0) * 1e3;\
    double baseline_ms  = (t3 - t2) * 1e3;\
    double upc_ns = (t1 - t0) * 1e9 / (double)N;\
    double bpc_ns = (t3 - t2) * 1e9 / (double)N;\
\
    printf("Cases: %zu\\n", N);\
    printf("Universal:  %.2f ms  (%.2f ns/op)  sinksum=%" PRIu64 ",%" PRIu64 "\\n",\
           universal_ms, upc_ns, sumQ, sumR);\
    printf("Baseline:   %.2f ms  (%.2f ns/op)  sinksumb=%" PRIu64 ",%" PRIu64 "\\n",\
           baseline_ms, bpc_ns, sumQb, sumRb);\
    printf("Correctness (subset 200k): %.2f%%\\n", 100.0 * (double)exact / 200000.0);\
    printf("Speed ratio Universal/Baseline: %.2fx\\n", (universal_ms / baseline_ms));\
\
    free(xs); free(ys);\
    return 0;\
\}}