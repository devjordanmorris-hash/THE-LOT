// waveform_benchmark_m1.c
// Benchmark fast LUT sine vs standard math sine on M1 Mac
// Jordan Morris © 2026

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define NUM_SAMPLES 1000000
#define NUM_THREADS 7
#define LUT_SIZE 65536

float lut[LUT_SIZE];

void initLUT() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        lut[i] = sinf((2.0f * M_PI * i) / LUT_SIZE);
    }
}

inline float fast_sin(float x) {
    int idx = (int)(x * (LUT_SIZE / (2.0f * M_PI))) % LUT_SIZE;
    if (idx < 0) idx += LUT_SIZE;
    return lut[idx];
}

float std_result[NUM_SAMPLES];
float fast_result[NUM_SAMPLES];

void* compute_standard_sine(void* arg) {
    long i = (long)arg;
    for (long k = i; k < NUM_SAMPLES; k += NUM_THREADS) {
        std_result[k] = sinf((float)k * 0.001f);
    }
    return NULL;
}

void* compute_fast_sine(void* arg) {
    long i = (long)arg;
    for (long k = i; k < NUM_SAMPLES; k += NUM_THREADS) {
        fast_result[k] = fast_sin((float)k * 0.001f);
    }
    return NULL;
}

void benchmark(const char* label, void* (*func)(void*)) {
    pthread_t threads[NUM_THREADS];
    clock_t start = clock();

    for (long i = 0; i < NUM_THREADS; ++i)
        pthread_create(&threads[i], NULL, func, (void*)i);
    for (int i = 0; i < NUM_THREADS; ++i)
        pthread_join(threads[i], NULL);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s: %.6f sec\n", label, elapsed);
}

void error_stats() {
    double rms = 0;
    float max_err = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float err = fabsf(std_result[i] - fast_result[i]);
        rms += err * err;
        if (err > max_err) max_err = err;
    }
    rms = sqrt(rms / NUM_SAMPLES);
    printf("RMS Error: %.8f\n", (float)rms);
    printf("Max Error: %.8f\n", max_err);
}

int main() {
    initLUT();

    benchmark("Standard sinf()", compute_standard_sine);
    benchmark("Fast LUT sin()", compute_fast_sine);
    error_stats();

    return 0;
}