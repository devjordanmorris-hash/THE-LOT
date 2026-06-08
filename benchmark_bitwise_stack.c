
// benchmark_bitwise_stack.c
// Compare bitwise-only integer division & symbolic power vs Apple libm baselines
// Platform: macOS (Apple Silicon compatible), but portable POSIX
// Build: see build.sh
// Author: J. Morris (2025)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#if defined(__APPLE__)
  #include <mach/mach_time.h>
#endif

// ---------------------- Timing ----------------------
static inline double now_seconds() {
#if defined(__APPLE__)
    static mach_timebase_info_data_t info;
    static uint64_t start = 0;
    if (start == 0) {
        mach_timebase_info(&info);
        start = mach_absolute_time();
        return 0.0;
    }
    uint64_t t = mach_absolute_time();
    double ns = (double)(t - start) * (double)info.numer / (double)info.denom;
    return ns / 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

// ---------------------- RNG (deterministic) ----------------------
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97f4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------- Bitwise Integer Division ----------------------
// Restoring long division: pure shifts & subtracts. Not fastest, but a clean bitwise baseline.
// Returns q = num / den and optionally remainder via *rem.
static inline uint64_t bitwise_div_u64(uint64_t num, uint64_t den, uint64_t *rem) {
    if (den == 0) { if (rem) *rem = 0; return UINT64_MAX; }
    uint64_t q = 0, r = 0;
    for (int i = 63; i >= 0; --i) {
        r = (r << 1) | ((num >> i) & 1ULL);
        if (r >= den) {
            r -= den;
            q |= (1ULL << i);
        }
    }
    if (rem) *rem = r;
    return q;
}

// ---------------------- Bitwise Symbolic Power (shift-accumulate) ----------------------
// This is the reduced-steps pattern discussed: log2(n) passes using shifts/adds only.
static inline uint64_t bitwise_pow_u64(uint64_t x, uint64_t n) {
    if (n == 0) return 1ULL;
    uint64_t p = x;
    int bl = 0;
    for (uint64_t tmp = n; tmp; tmp >>= 1) ++bl;
    for (int k = 0; k < bl; ++k) {
        uint64_t b = (n >> k) & 1ULL;
        // Composite shift-add step; stays in 64-bit with natural wrap (mod 2^64)
        uint64_t left  = (b ? (p << 1) : p);          // growth when bit = 1
        uint64_t right = (!b ? (p >> 1) : (p >> 0));  // attenuation when bit = 0
        p = p + left - right;
    }
    return p;
}

// ---------------------- Baselines ----------------------
static inline uint64_t baseline_div_u64(uint64_t a, uint64_t b, uint64_t *rem) {
    if (b == 0) { if (rem) *rem = 0; return UINT64_MAX; }
    uint64_t q = a / b;
    if (rem) *rem = a % b;
    return q;
}

static inline uint64_t baseline_pow_u64(uint64_t x, uint64_t n) {
    // For comparison; uses libm pow on doubles then clamps to uint64_t
    // NOTE: This is not bit-accurate integer exponentiation; it's a speed baseline.
    double dx = (double)(x & 0x1fffffffffffffULL); // reduce overflow risk
    double dn = (double)n;
    double r = pow(dx, dn);
    if (r < 0) r = 0;
    if (r > (double)UINT64_MAX) r = (double)UINT64_MAX;
    return (uint64_t)r;
}

// ---------------------- Harness ----------------------
typedef struct {
    size_t N;
    int repeats;
    uint64_t seed;
} config_t;

static void parse_args(int argc, char** argv, config_t* cfg) {
    cfg->N = 1000000;
    cfg->repeats = 3;
    cfg->seed = 0xC0FFEEULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) cfg->N = (size_t) strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) cfg->repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) cfg->seed = strtoull(argv[++i], NULL, 16);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--N <count>] [--repeats <r>] [--seed <hex>]\n", argv[0]);
            exit(0);
        }
    }
}

static void bench_division(const config_t* cfg) {
    uint64_t state = cfg->seed;
    uint64_t *a = (uint64_t*) malloc(cfg->N * sizeof(uint64_t));
    uint64_t *b = (uint64_t*) malloc(cfg->N * sizeof(uint64_t));
    for (size_t i = 0; i < cfg->N; ++i) {
        a[i] = splitmix64(&state);
        b[i] = (splitmix64(&state) | 1ULL); // avoid zero
    }

    // Baseline time
    double best_base = 1e9, best_bit = 1e9;
    uint64_t sink_base = 0, sink_bit = 0;
    for (int r = 0; r < cfg->repeats; ++r) {
        double t0 = now_seconds();
        uint64_t s = 0;
        for (size_t i = 0; i < cfg->N; ++i) {
            uint64_t rem;
            s += baseline_div_u64(a[i], b[i], &rem);
            s ^= rem;
        }
        double t1 = now_seconds();
        double dt = t1 - t0;
        if (dt < best_base) { best_base = dt; sink_base = s; }
    }

    for (int r = 0; r < cfg->repeats; ++r) {
        double t0 = now_seconds();
        uint64_t s = 0;
        for (size_t i = 0; i < cfg->N; ++i) {
            uint64_t rem;
            s += bitwise_div_u64(a[i], b[i], &rem);
            s ^= rem;
        }
        double t1 = now_seconds();
        double dt = t1 - t0;
        if (dt < best_bit) { best_bit = dt; sink_bit = s; }
    }

    double base_mops = (double)cfg->N / best_base / 1e6;
    double bit_mops  = (double)cfg->N / best_bit  / 1e6;
    printf("INT DIV: baseline=%.2f Mops/s (sink=%llu), bitwise=%.2f Mops/s (sink=%llu), speedup=%.2fx\n",
           base_mops, (unsigned long long)sink_base, bit_mops, (unsigned long long)sink_bit, bit_mops / base_mops);

    free(a); free(b);
}

static void bench_power(const config_t* cfg) {
    uint64_t state = cfg->seed ^ 0xDEADBEEFCAFEBABEULL;
    uint64_t *x = (uint64_t*) malloc(cfg->N * sizeof(uint64_t));
    uint64_t *n = (uint64_t*) malloc(cfg->N * sizeof(uint64_t));
    for (size_t i = 0; i < cfg->N; ++i) {
        x[i] = splitmix64(&state) & 0x00000000FFFFFFFFULL; // keep smaller to reduce overflow
        n[i] = (splitmix64(&state) & 31ULL) + 1ULL;        // exponents 1..32
    }

    double best_base = 1e9, best_bit = 1e9;
    uint64_t sink_base = 0, sink_bit = 0;

    for (int r = 0; r < cfg->repeats; ++r) {
        double t0 = now_seconds();
        uint64_t s = 0;
        for (size_t i = 0; i < cfg->N; ++i) {
            s ^= baseline_pow_u64(x[i], n[i]);
        }
        double t1 = now_seconds();
        if (t1 - t0 < best_base) { best_base = t1 - t0; sink_base = s; }
    }

    for (int r = 0; r < cfg->repeats; ++r) {
        double t0 = now_seconds();
        uint64_t s = 0;
        for (size_t i = 0; i < cfg->N; ++i) {
            s ^= bitwise_pow_u64(x[i], n[i]);
        }
        double t1 = now_seconds();
        if (t1 - t0 < best_bit) { best_bit = t1 - t0; sink_bit = s; }
    }

    double base_mops = (double)cfg->N / best_base / 1e6;
    double bit_mops  = (double)cfg->N / best_bit  / 1e6;
    printf("POW(shift-accum): baseline=%.2f Mops/s (sink=%llu), bitwise=%.2f Mops/s (sink=%llu), speedup=%.2fx\n",
           base_mops, (unsigned long long)sink_base, bit_mops, (unsigned long long)sink_bit, bit_mops / base_mops);

    free(x); free(n);
}

int main(int argc, char** argv) {
    config_t cfg;
    parse_args(argc, argv, &cfg);

    // Prime the timer
    (void)now_seconds();

    printf("Bitwise Stack Benchmark | N=%zu, repeats=%d, seed=0x%llx\n",
           cfg.N, cfg.repeats, (unsigned long long)cfg.seed);

    bench_division(&cfg);
    bench_power(&cfg);
    return 0;
}
