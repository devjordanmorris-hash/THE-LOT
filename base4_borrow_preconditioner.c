/*
    base4_borrow_preconditioner.c

    Experimental arithmetic decomposition / borrow-field preconditioner.

    Idea:
        Work in base-4 digits (2-bit chunks).
        For subtraction A - B, build a 0/1 base-4 mask over B by scanning
        right-to-left. If a digit would borrow, reduce that B digit by 1
        and store a 1 in the mask.

        Then:
            B_residual = B - mask

            A - B = (A - B_residual) - mask

        More generally:
            A - B = (A - mA) - (B - mB) + (mA - mB)

        In this experimental version mA = 0 and mB is the adaptive borrow mask.

    What this tests:
        1. Correctness of decomposition.
        2. Whether A - (B - mask) has fewer borrow events than A - B.
        3. Whether max borrow-chain depth is reduced.
        4. Whether the mask has useful structure.

    Build:
        clang -O3 base4_borrow_preconditioner.c -o base4_borrow_preconditioner

    Run:
        ./base4_borrow_preconditioner

    Notes:
        This is NOT claiming to beat hardware subtraction.
        It is an experiment in moving borrow complexity into a structured
        correction/mask field.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 16                 // 16 base-4 digits = 32 bits
#define TRIALS 1000000u

typedef struct {
    uint32_t borrow_count;
    uint32_t max_chain;
} BorrowStats;

typedef struct {
    uint32_t mask;
    uint32_t residual_b;
} MaskResult;

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

/*
    Get base-4 digit at position pos.

    pos = 0 means most significant base-4 digit.
    pos = WIDTH-1 means least significant base-4 digit.

    Since base-4 digit = 2 bits:
        digit = (x >> shift) & 3
*/
static inline uint32_t get_b4_digit(uint32_t x, int pos) {
    int shift = (WIDTH - 1 - pos) * 2;
    return (x >> shift) & 3u;
}

static inline uint32_t set_b4_digit(uint32_t x, int pos, uint32_t digit) {
    int shift = (WIDTH - 1 - pos) * 2;
    uint32_t mask = 3u << shift;
    return (x & ~mask) | ((digit & 3u) << shift);
}

static void print_base4(uint32_t x) {
    for (int i = 0; i < WIDTH; i++) {
        putchar((char)('0' + get_b4_digit(x, i)));
    }
}

/*
    Count borrow events and max contiguous borrow-chain length in base-4 subtraction.
*/
static BorrowStats borrow_stats_base4(uint32_t a, uint32_t b) {
    uint32_t borrow = 0;
    uint32_t count = 0;
    uint32_t chain = 0;
    uint32_t max_chain = 0;

    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        uint32_t ad = get_b4_digit(a, pos);
        uint32_t bd = get_b4_digit(b, pos);

        int32_t effective_a = (int32_t)ad - (int32_t)borrow;

        if (effective_a < (int32_t)bd) {
            count++;
            chain++;
            if (chain > max_chain) max_chain = chain;
            borrow = 1;
        } else {
            chain = 0;
            borrow = 0;
        }
    }

    BorrowStats s;
    s.borrow_count = count;
    s.max_chain = max_chain;
    return s;
}

/*
    Adaptive scanned borrow mask.

    Scan right-to-left.
    If current base-4 digit would borrow, reduce B_digit by 1 and record
    a 1 in mask at that digit.

    This deliberately makes the residual subtraction A - (B-mask)
    easier, while pushing correction into mask.
*/
static MaskResult adaptive_mask_base4(uint32_t a, uint32_t b) {
    uint32_t mask = 0;
    uint32_t residual_b = b;
    uint32_t borrow = 0;

    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        uint32_t ad = get_b4_digit(a, pos);
        uint32_t bd = get_b4_digit(residual_b, pos);

        int32_t effective_a = (int32_t)ad - (int32_t)borrow;
        uint32_t new_bd = bd;

        if (effective_a < (int32_t)bd && bd > 0) {
            mask = set_b4_digit(mask, pos, 1);
            new_bd = bd - 1;
            residual_b = set_b4_digit(residual_b, pos, new_bd);
        }

        if (effective_a < (int32_t)new_bd) {
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    MaskResult r;
    r.mask = mask;
    r.residual_b = residual_b;
    return r;
}

static uint32_t popcount_base4_ones(uint32_t mask) {
    uint32_t c = 0;
    for (int i = 0; i < WIDTH; i++) {
        if (get_b4_digit(mask, i) != 0) c++;
    }
    return c;
}

static uint32_t count_one_runs_base4(uint32_t mask) {
    uint32_t runs = 0;
    int in_run = 0;

    for (int i = 0; i < WIDTH; i++) {
        uint32_t d = get_b4_digit(mask, i);
        if (d != 0) {
            if (!in_run) {
                runs++;
                in_run = 1;
            }
        } else {
            in_run = 0;
        }
    }
    return runs;
}

static void demo_case(uint32_t a, uint32_t b) {
    if (a < b) {
        uint32_t t = a;
        a = b;
        b = t;
    }

    MaskResult mr = adaptive_mask_base4(a, b);

    uint32_t true_ans = a - b;

    /*
        Because mA = 0:
            A - B = A - (B - mask) - mask
    */
    uint32_t decomposed = (a - mr.residual_b) - mr.mask;

    BorrowStats orig = borrow_stats_base4(a, b);
    BorrowStats resid = borrow_stats_base4(a, mr.residual_b);

    printf("\n--- Demo case ---\n");
    printf("A decimal: %u\n", a);
    printf("B decimal: %u\n", b);

    printf("A base4:       "); print_base4(a); puts("");
    printf("B base4:       "); print_base4(b); puts("");
    printf("mask base4:    "); print_base4(mr.mask); puts("");
    printf("B-mask base4:  "); print_base4(mr.residual_b); puts("");

    printf("true A-B:      %u\n", true_ans);
    printf("decomposed:    %u\n", decomposed);
    printf("correct:       %s\n", true_ans == decomposed ? "yes" : "NO");

    printf("borrows:       %u -> %u\n", orig.borrow_count, resid.borrow_count);
    printf("max chain:     %u -> %u\n", orig.max_chain, resid.max_chain);
    printf("mask popcount: %u / %u\n", popcount_base4_ones(mr.mask), WIDTH);
    printf("mask runs:     %u\n", count_one_runs_base4(mr.mask));
}

int main(void) {
    printf("Base-4 Adaptive Borrow Preconditioner\n");
    printf("WIDTH=%d base-4 digits (%d bits)\n", WIDTH, WIDTH * 2);
    printf("TRIALS=%u\n", TRIALS);

    demo_case(10000u, 7371u);
    demo_case(1749090055u, 224899942u);

    uint64_t correct = 0;
    uint64_t fewer = 0;
    uint64_t same = 0;
    uint64_t more = 0;

    uint64_t total_orig_borrows = 0;
    uint64_t total_resid_borrows = 0;
    uint64_t total_orig_chain = 0;
    uint64_t total_resid_chain = 0;
    uint64_t total_mask_pop = 0;
    uint64_t total_mask_runs = 0;

    volatile uint32_t sink = 0;

    double t0 = now_sec();

    for (uint32_t i = 0; i < TRIALS; i++) {
        uint32_t a = xorshift32();
        uint32_t b = xorshift32();

        if (a < b) {
            uint32_t t = a;
            a = b;
            b = t;
        }

        MaskResult mr = adaptive_mask_base4(a, b);

        uint32_t true_ans = a - b;
        uint32_t decomposed = (a - mr.residual_b) - mr.mask;

        if (true_ans == decomposed) correct++;

        BorrowStats orig = borrow_stats_base4(a, b);
        BorrowStats resid = borrow_stats_base4(a, mr.residual_b);

        total_orig_borrows += orig.borrow_count;
        total_resid_borrows += resid.borrow_count;
        total_orig_chain += orig.max_chain;
        total_resid_chain += resid.max_chain;

        total_mask_pop += popcount_base4_ones(mr.mask);
        total_mask_runs += count_one_runs_base4(mr.mask);

        if (resid.borrow_count < orig.borrow_count) fewer++;
        else if (resid.borrow_count == orig.borrow_count) same++;
        else more++;

        sink ^= decomposed;
    }

    double t1 = now_sec();

    printf("\n--- Random test summary ---\n");
    printf("correct:              %.2f%%\n", 100.0 * (double)correct / (double)TRIALS);
    printf("fewer residual borrows %.2f%%\n", 100.0 * (double)fewer / (double)TRIALS);
    printf("same residual borrows  %.2f%%\n", 100.0 * (double)same / (double)TRIALS);
    printf("more residual borrows  %.2f%%\n", 100.0 * (double)more / (double)TRIALS);

    printf("\n--- Averages ---\n");
    printf("original borrows:     %.4f\n", (double)total_orig_borrows / (double)TRIALS);
    printf("residual borrows:     %.4f\n", (double)total_resid_borrows / (double)TRIALS);
    printf("original max chain:   %.4f\n", (double)total_orig_chain / (double)TRIALS);
    printf("residual max chain:   %.4f\n", (double)total_resid_chain / (double)TRIALS);
    printf("mask popcount:        %.4f / %d\n", (double)total_mask_pop / (double)TRIALS, WIDTH);
    printf("mask runs:            %.4f\n", (double)total_mask_runs / (double)TRIALS);

    printf("\n--- Timing ---\n");
    printf("elapsed:              %.3f ms\n", (t1 - t0) * 1000.0);
    printf("ops/sec:              %.2f M trials/sec\n", (double)TRIALS / (t1 - t0) / 1e6);
    printf("sink:                 %u\n", sink);

    printf("\nInterpretation:\n");
    printf("  This is not expected to beat hardware subtraction.\n");
    printf("  It tests whether borrow complexity can be moved into a structured\n");
    printf("  0/1 base-4 correction mask. If that mask can be processed cheaply,\n");
    printf("  there may be a niche arithmetic-preconditioning angle.\n");

    return 0;
}
