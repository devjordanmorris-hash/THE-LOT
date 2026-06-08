/*
    base4_parallel_carry_wave.c

    Base-4 parallel carry-wave adder experiment.

    Idea:
      1. Add all base-4 digits WITHOUT incoming carries.
      2. Detect all carry positions in parallel.
      3. Convert carries into a shifted carry stream.
      4. Repeat until no carries remain.

    This is similar in spirit to XOR/AND carry iteration,
    but operates on base-4 digits instead of bits.

    We measure:
      - correctness
      - average carry-wave iterations
      - carry density per wave
      - performance vs native add

    Build:
      clang -O3 base4_parallel_carry_wave.c -o base4_parallel_carry_wave

    Run:
      ./base4_parallel_carry_wave
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define TRIALS 1000000u
#define WIDTH 16u

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

typedef struct {
    uint32_t result;
    uint32_t waves;
    uint32_t totalCarryDigits;
    uint32_t maxCarryDigitsInWave;
} WaveResult;

static inline uint32_t digit4(uint32_t x, uint32_t pos) {
    return (x >> (pos * 2u)) & 3u;
}

static inline void set_digit4(uint32_t *x, uint32_t pos, uint32_t digit) {
    uint32_t shift = pos * 2u;
    *x &= ~(3u << shift);
    *x |= (digit & 3u) << shift;
}

/*
    One carry-wave iteration.

    partial:
      current digit values

    carryStream:
      carries shifted into next base-4 digit positions

    We compute:
      partial + carryStream
    without propagating incoming carries serially.
*/
static WaveResult base4_parallel_wave_add(uint32_t a, uint32_t b) {
    uint32_t partial = 0u;

    // Initial digitwise add without incoming carry.
    for (uint32_t pos = 0; pos < WIDTH; pos++) {
        uint32_t local = digit4(a, pos) + digit4(b, pos);
        set_digit4(&partial, pos, local & 3u);
    }

    // Initial carry stream.
    uint32_t carryStream = 0u;

    for (uint32_t pos = 0; pos < WIDTH - 1u; pos++) {
        uint32_t local = digit4(a, pos) + digit4(b, pos);

        if (local >= 4u) {
            carryStream |= 1u << ((pos + 1u) * 2u);
        }
    }

    uint32_t waves = 0u;
    uint32_t totalCarryDigits = 0u;
    uint32_t maxCarryDigitsInWave = 0u;

    while (carryStream != 0u) {
        waves++;

        uint32_t nextPartial = 0u;
        uint32_t nextCarryStream = 0u;
        uint32_t carryDigitsThisWave = 0u;

        for (uint32_t pos = 0; pos < WIDTH; pos++) {
            uint32_t local =
                digit4(partial, pos) +
                digit4(carryStream, pos);

            uint32_t outDigit = local & 3u;
            set_digit4(&nextPartial, pos, outDigit);

            if (local >= 4u && pos < WIDTH - 1u) {
                nextCarryStream |= 1u << ((pos + 1u) * 2u);
                carryDigitsThisWave++;
            }
        }

        totalCarryDigits += carryDigitsThisWave;

        if (carryDigitsThisWave > maxCarryDigitsInWave) {
            maxCarryDigitsInWave = carryDigitsThisWave;
        }

        partial = nextPartial;
        carryStream = nextCarryStream;
    }

    WaveResult r;
    r.result = partial;
    r.waves = waves;
    r.totalCarryDigits = totalCarryDigits;
    r.maxCarryDigitsInWave = maxCarryDigitsInWave;
    return r;
}

static void print_b4(uint32_t x) {
    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        putchar('0' + (int)digit4(x, (uint32_t)pos));
    }
}

static void run_range(const char *name, uint32_t maxv) {
    uint64_t correct = 0;
    uint64_t totalWaves = 0;
    uint64_t totalCarryDigits = 0;
    uint64_t totalMaxCarryWave = 0;

    uint64_t waveHistogram[32] = {0};

    volatile uint32_t nativeSink = 0;
    volatile uint32_t waveSink = 0;

    uint32_t *A = malloc(TRIALS * sizeof(uint32_t));
    uint32_t *B = malloc(TRIALS * sizeof(uint32_t));

    if (!A || !B) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    for (uint32_t i = 0; i < TRIALS; i++) {
        if (maxv == 0xffffffffu) {
            A[i] = xorshift32();
            B[i] = xorshift32();
        } else {
            A[i] = xorshift32() % (maxv + 1u);
            B[i] = xorshift32() % (maxv + 1u);
        }
    }

    double n0 = now_sec();
    for (uint32_t i = 0; i < TRIALS; i++) {
        nativeSink ^= A[i] + B[i];
    }
    double n1 = now_sec();

    double w0 = now_sec();
    for (uint32_t i = 0; i < TRIALS; i++) {
        WaveResult r = base4_parallel_wave_add(A[i], B[i]);
        waveSink ^= r.result;
    }
    double w1 = now_sec();

    uint32_t exampleA = 0;
    uint32_t exampleB = 0;
    WaveResult example = {0};

    for (uint32_t i = 0; i < TRIALS; i++) {
        uint32_t native = A[i] + B[i];
        WaveResult r = base4_parallel_wave_add(A[i], B[i]);

        if (native == r.result) correct++;

        totalWaves += r.waves;
        totalCarryDigits += r.totalCarryDigits;
        totalMaxCarryWave += r.maxCarryDigitsInWave;

        if (r.waves < 32u) {
            waveHistogram[r.waves]++;
        }

        if (i == 0 || r.waves >= 4u) {
            exampleA = A[i];
            exampleB = B[i];
            example = r;
        }
    }

    printf("\n============================================================\n");
    printf("Range: %s [0, %u]\n", name, maxv);
    printf("============================================================\n");

    printf("\n--- Correctness ---\n");
    printf("correct: %.2f%%\n",
           100.0 * (double)correct / (double)TRIALS);

    printf("\n--- Carry-wave structure ---\n");
    printf("avg waves:                 %.4f\n",
           (double)totalWaves / (double)TRIALS);

    printf("avg carry digits total:    %.4f\n",
           (double)totalCarryDigits / (double)TRIALS);

    printf("avg max carry digits/wave: %.4f\n",
           (double)totalMaxCarryWave / (double)TRIALS);

    printf("\nWave histogram:\n");

    for (uint32_t i = 0; i < 12u; i++) {
        if (waveHistogram[i] == 0) continue;

        printf("waves=%2u : %8.4f%%\n",
               i,
               100.0 * (double)waveHistogram[i] / (double)TRIALS);
    }

    printf("\n--- Timing ---\n");
    printf("native add: %.3f ms | %.2f M ops/sec\n",
           (n1 - n0) * 1000.0,
           (double)TRIALS / (n1 - n0) / 1e6);

    printf("wave add:   %.3f ms | %.2f M ops/sec | slowdown %.2fx\n",
           (w1 - w0) * 1000.0,
           (double)TRIALS / (w1 - w0) / 1e6,
           (w1 - w0) / (n1 - n0));

    printf("\n--- Example ---\n");
    printf("A:      "); print_b4(exampleA); printf("  %u\n", exampleA);
    printf("B:      "); print_b4(exampleB); printf("  %u\n", exampleB);
    printf("native: "); print_b4(exampleA + exampleB); printf("  %u\n", exampleA + exampleB);
    printf("wave:   "); print_b4(example.result); printf("  %u\n", example.result);

    printf("waves=%u totalCarryDigits=%u maxCarryDigitsInWave=%u\n",
           example.waves,
           example.totalCarryDigits,
           example.maxCarryDigitsInWave);

    printf("\nsinks: native=%u wave=%u\n", nativeSink, waveSink);

    free(A);
    free(B);
}

int main(void) {
    printf("Base-4 Parallel Carry-Wave Add Experiment\n");
    printf("TRIALS=%u WIDTH=%u\n", TRIALS, WIDTH);

    run_range("LOW", 4095u);
    run_range("MEDIUM", 16777215u);
    run_range("LARGE", 0xffffffffu);

    return 0;
}
