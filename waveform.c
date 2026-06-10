#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265
#define SAMPLES 100000
#define LUT_SIZE 360

// Fast sine LUT (in degrees)
float sineLUT[LUT_SIZE];

// Setup LUT
void initLUT() {
    for (int i = 0; i < LUT_SIZE; i++) {
        sineLUT[i] = sin(i * PI / 180.0);
    }
}

// Get sine using LUT
float fastSine(float degrees) {
    int idx = ((int)degrees) % 360;
    return sineLUT[idx < 0 ? idx + 360 : idx];
}

// Generate waveform using standard sine
void generateWaveStandard(float* out, int n) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / n;
        out[i] = sin(2 * PI * t);
    }
}

// Generate waveform using fast sine and triangle-based method
void generateWaveFast(float* out, int n) {
    for (int i = 0; i < n; i++) {
        float angle = (360.0f * i) / n;
        out[i] = fastSine(angle);
    }
}

// Compare waveforms
void compare(float* a, float* b, int n) {
    float maxErr = 0, sumErr = 0;
    for (int i = 0; i < n; i++) {
        float err = fabs(a[i] - b[i]);
        sumErr += err * err;
        if (err > maxErr) maxErr = err;
    }
    printf("RMS Error: %.8f\n", sqrt(sumErr / n));
    printf("Max Error: %.8f\n", maxErr);
}

int main() {
    float* waveStandard = malloc(sizeof(float) * SAMPLES);
    float* waveFast = malloc(sizeof(float) * SAMPLES);

    initLUT();

    clock_t start, end;

    start = clock();
    generateWaveStandard(waveStandard, SAMPLES);
    end = clock();
    printf("Standard Sine Time: %.6f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    generateWaveFast(waveFast, SAMPLES);
    end = clock();
    printf("Fast LUT Sine Time: %.6f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    compare(waveStandard, waveFast, SAMPLES);

    free(waveStandard);
    free(waveFast);
    return 0;
}