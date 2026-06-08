#include <metal_stdlib>
using namespace metal;

// =====================================================
// GPU Fast Sine via Rotor LUT
// =====================================================
inline float fastSin_lut(float x,
                         constant float *lut,
                         uint lutSize,
                         float step)
{
    // normalise to [0, 2π)
    float xx = fmod(x, 2.0f * M_PI_F);
    if (xx < 0) xx += 2.0f * M_PI_F;

    float pi2 = M_PI_F * 0.5f;

    if (xx <= pi2) {
        uint idx = uint(xx * step);
        return lut[min(idx, lutSize - 1)];
    }
    if (xx <= M_PI_F) {
        float xx2 = M_PI_F - xx;
        uint idx = uint(xx2 * step);
        return lut[min(idx, lutSize - 1)];
    }
    if (xx <= 3.0f * pi2) {
        float xx3 = xx - M_PI_F;
        uint idx = uint(xx3 * step);
        return -lut[min(idx, lutSize - 1)];
    }
    float xx4 = (2.0f * M_PI_F) - xx;
    uint idx = uint(xx4 * step);
    return -lut[min(idx, lutSize - 1)];
}


// =====================================================
// Tier-0 ROTOR FFT Kernel (DFT per active bin)
// =====================================================
kernel void tier0_fft_rotor(
    constant float *signal      [[ buffer(0) ]],
    constant uint  &N           [[ buffer(1) ]],
    constant uint  *activeBins  [[ buffer(2) ]],
    constant uint  &activeCount [[ buffer(3) ]],
    device   float2 *output     [[ buffer(4) ]],
    constant float *lut         [[ buffer(5) ]],
    constant uint  &lutSize     [[ buffer(6) ]],
    constant float &step        [[ buffer(7) ]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= activeCount) return;

    uint k = activeBins[tid];
    float w = -2.0f * M_PI_F * float(k) / float(N);

    float re = 0.0f;
    float im = 0.0f;

    for (uint n = 0; n < N; ++n) {
        float a = w * float(n);

        float s = fastSin_lut(a, lut, lutSize, step);
        float c = fastSin_lut(a + (M_PI_F * 0.5f), lut, lutSize, step);

        float v = signal[n];
        re += v * c;
        im += v * s;
    }

    output[k] = float2(re, im);
}
