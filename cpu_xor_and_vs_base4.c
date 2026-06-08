/*
    cpu_xor_and_vs_base4.c

    CPU benchmark:
      1. Native uint32_t addition
      2. Literal XOR/AND/shift addition
      3. Base-4 carry preconditioned addition

    Build:
      clang -O3 -mcpu=apple-m1 cpu_xor_and_vs_base4.c -o cpu_xor_and_vs_base4

    Run:
      ./cpu_xor_and_vs_base4
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000u

static uint64_t rng_state = 0x123456789abcdefULL;

static inline uint32_t xorshift32(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return (uint32_t)(x >> 32) ^ (uint32_t)x;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint32_t xor_and_add(uint32_t a, uint32_t b, uint32_t *iters_out, uint32_t *carry_field_out) {
    uint32_t carry_field = 0;
    uint32_t iters = 0;

    while (b != 0) {
        uint32_t carry = (a & b) << 1;
        uint32_t sum = a ^ b;

        carry_field |= carry;
        iters++;

        a = sum;
        b = carry;
    }

    *iters_out = iters;
    *carry_field_out = carry_field;

    return a;
}

static inline uint32_t base4_transform_add(uint32_t a, uint32_t b, uint32_t *mask_out) {
    uint32_t mask = 0;
    uint32_t residual_b = b;
    uint32_t carry = 0;

    for (uint32_t shift = 0; shift < 32; shift += 2) {
        uint32_t ad = (a >> shift) & 3u;
        uint32_t bd = (residual_b >> shift) & 3u;

        uint32_t sum = ad + bd + carry;

        if (sum >= 4u && bd > 0u) {
            uint32_t amount = 1u << shift;
            mask |= amount;
            residual_b -= amount;
            bd -= 1u;
            sum = ad + bd + carry;
        }

        carry = (sum >= 4u) ? 1u : 0u;
    }

    *mask_out = mask;
    return a + residual_b + mask;
}

int main(void) {
    printf("CPU XOR/AND vs Base-4 Transform Add\n");
    printf("N=%u\n", N);

    uint32_t *A = malloc(N * sizeof(uint32_t));
    uint32_t *B = malloc(N * sizeof(uint32_t));

    if (!A || !B) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < N; i++) {
        A[i] = xorshift32();
        B[i] = xorshift32();
    }

    volatile uint32_t native_sink = 0;
    volatile uint32_t xor_sink = 0;
    volatile uint32_t base4_sink = 0;

    double n0 = now_sec();
    for (uint32_t i = 0; i < N; i++) {
        native_sink ^= A[i] + B[i];
    }
    double n1 = now_sec();

    uint64_t total_xor_iters = 0;
    uint64_t total_xor_carry_pop = 0;

    double x0 = now_sec();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t iters = 0;
        uint32_t carry_field = 0;
        uint32_t r = xor_and_add(A[i], B[i], &iters, &carry_field);

        total_xor_iters += iters;
        total_xor_carry_pop += __builtin_popcount(carry_field);

        xor_sink ^= r;
    }
    double x1 = now_sec();

    uint64_t total_mask_pop = 0;
    uint64_t overlap = 0;
    uint64_t union_pop = 0;

    double b0 = now_sec();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t mask = 0;
        uint32_t r = base4_transform_add(A[i], B[i], &mask);

        /*
            Recompute carry field for comparison only.
            This is included in base4 timing below, so don't treat base4 timing as pure.
            We'll run a separate pure base4 timing after this.
        */
        uint32_t iters_tmp = 0;
        uint32_t carry_field = 0;
        (void)xor_and_add(A[i], B[i], &iters_tmp, &carry_field);

        total_mask_pop += __builtin_popcount(mask);
        overlap += __builtin_popcount(mask & carry_field);
        union_pop += __builtin_popcount(mask | carry_field);

        base4_sink ^= r;
    }
    double b1 = now_sec();

    volatile uint32_t base4_pure_sink = 0;
    double bp0 = now_sec();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t mask = 0;
        base4_pure_sink ^= base4_transform_add(A[i], B[i], &mask);
    }
    double bp1 = now_sec();

    uint64_t errors = 0;
    for (uint32_t i = 0; i < 10000; i++) {
        uint32_t iters = 0;
        uint32_t cf = 0;
        uint32_t m = 0;

        uint32_t native = A[i] + B[i];
        uint32_t xr = xor_and_add(A[i], B[i], &iters, &cf);
        uint32_t br = base4_transform_add(A[i], B[i], &m);

        if (native != xr || native != br) errors++;
    }

    printf("\n--- Speed ---\n");
    printf("Native CPU add:      %.3f ms | %.2f M ops/sec\n",
           (n1 - n0) * 1000.0,
           (double)N / (n1 - n0) / 1e6);

    printf("XOR/AND CPU add:     %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
           (x1 - x0) * 1000.0,
           (double)N / (x1 - x0) / 1e6,
           (x1 - x0) / (n1 - n0));

    printf("Base4 CPU add pure:  %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
           (bp1 - bp0) * 1000.0,
           (double)N / (bp1 - bp0) / 1e6,
           (bp1 - bp0) / (n1 - n0));

    printf("\n--- Diagnostics ---\n");
    printf("Correctness errors sample: %llu\n", (unsigned long long)errors);
    printf("Avg XOR iterations:        %.4f\n", (double)total_xor_iters / (double)N);
    printf("Avg XOR carry pop:         %.4f / 32 bits\n", (double)total_xor_carry_pop / (double)N);
    printf("Avg base4 mask pop:        %.4f / 16 base4 digits\n", (double)total_mask_pop / (double)N);
    printf("Overlap Jaccard:           %.4f\n", union_pop ? (double)overlap / (double)union_pop : 0.0);

    printf("\n--- Sinks ---\n");
    printf("native=%u xor=%u base4=%u base4pure=%u\n",
           native_sink, xor_sink, base4_sink, base4_pure_sink);

    free(A);
    free(B);

    return 0;
}
