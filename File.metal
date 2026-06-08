// ============================================================
// Tier0Sine.metal
// Jordan Morris © 2025
// Quarter-wave LUT sine arithmetic — GPU
// Drop into Xcode project alongside Swift harness
// ============================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Config
// ============================================================

constant int   LUT_N   = 16384;
constant float HALF_PI = 1.5707963268f;
constant float PI      = 3.1415926536f;
constant float TWO_PI  = 6.2831853072f;
constant float SQRT2   = 1.4142135624f;

// ============================================================
// Quarter-wave LUT — built at kernel launch
// Stored in device memory, passed as buffer
// ============================================================

// ============================================================
// Fast sine — quarter wave + symmetry
// ============================================================

inline float t0_sin(
    device const float *lut,
    float x)
{
    // Wrap to [0, 2PI)
    x = x - TWO_PI * floor(x / TWO_PI);

    float base;
    float sign = 1.0f;

    if (x < HALF_PI) {
        base = x;
    } else if (x < PI) {
        base = PI - x;
    } else if (x < PI + HALF_PI) {
        base = x - PI;
        sign = -1.0f;
    } else {
        base = TWO_PI - x;
        sign = -1.0f;
    }

    int idx = int(base / HALF_PI * float(LUT_N));
    idx     = clamp(idx, 0, LUT_N - 1);
    return sign * lut[idx];
}

inline float t0_cos(
    device const float *lut,
    float x)
{
    return t0_sin(lut, x + HALF_PI);
}

// ============================================================
// Fast arcsin — binary search on LUT
// ============================================================

inline float t0_arcsin(
    device const float *lut,
    float y)
{
    y = clamp(y, 0.0f, 1.0f);
    int lo = 0;
    int hi = LUT_N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (lut[mid] < y) lo = mid + 1;
        else hi = mid;
    }
    return float(lo) * HALF_PI / float(LUT_N);
}

// ============================================================
// Sine arithmetic
// ============================================================

inline float sine_mul(
    device const float *lut,
    float A, float B, float M)
{
    float a  = clamp(A / M, 0.0f, 1.0f);
    float b  = clamp(B / M, 0.0f, 1.0f);
    float sx = t0_sin(lut, t0_arcsin(lut, sqrt(a)));
    float sy = t0_sin(lut, t0_arcsin(lut, sqrt(b)));
    return (sx * sy) * (sx * sy) * M;
}

inline float sine_div(
    device const float *lut,
    float A, float B, float M)
{
    float a  = clamp(A / M, 0.0f, 1.0f);
    float b  = clamp(abs(B) / M, 1e-9f, 1.0f);
    float sx = t0_sin(lut, t0_arcsin(lut, sqrt(a)));
    float sy = t0_sin(lut, t0_arcsin(lut, sqrt(b)));
    float q  = (sx * sy) / (sy * sy);
    float sign = ((A < 0) == (B < 0)) ? 1.0f : -1.0f;
    return q * q * M * sign;
}

inline float sine_sqrt(
    device const float *lut,
    float A, float M)
{
    float a  = clamp(A / M, 0.0f, 1.0f);
    float sx = t0_sin(lut, t0_arcsin(lut, sqrt(a)));
    return sx * sx * M;
}

inline float sine_add(
    device const float *lut,
    float A, float B)
{
    float K = A + B;
    if (K == 0.0f) return 0.0f;
    float a   = clamp(A / K, 0.0f, 1.0f);
    float x   = t0_arcsin(lut, sqrt(a));
    float s   = t0_sin(lut, x);
    float sp4 = t0_sin(lut, x + PI / 4.0f);
    float b   = (SQRT2 * sp4 - s) * (SQRT2 * sp4 - s);
    return (a + b) * K;
}

// ============================================================
// Kernel 1 — Batch sine wave generation
// ============================================================

kernel void sine_batch(
    device const float *angles  [[buffer(0)]],
    device       float *results [[buffer(1)]],
    device const float *lut     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    results[gid] = t0_sin(lut, angles[gid]);
}

// ============================================================
// Kernel 2 — Batch cosine
// ============================================================

kernel void cosine_batch(
    device const float *angles  [[buffer(0)]],
    device       float *results [[buffer(1)]],
    device const float *lut     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    results[gid] = t0_cos(lut, angles[gid]);
}

// ============================================================
// Kernel 3 — Batch multiply
// ============================================================

kernel void sine_mul_batch(
    device const float *A       [[buffer(0)]],
    device const float *B       [[buffer(1)]],
    device const float *M       [[buffer(2)]],
    device       float *results [[buffer(3)]],
    device const float *lut     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    results[gid] = sine_mul(lut, A[gid], B[gid], M[gid]);
}

// ============================================================
// Kernel 4 — Batch divide
// ============================================================

kernel void sine_div_batch(
    device const float *A       [[buffer(0)]],
    device const float *B       [[buffer(1)]],
    device const float *M       [[buffer(2)]],
    device       float *results [[buffer(3)]],
    device const float *lut     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    results[gid] = sine_div(lut, A[gid], B[gid], M[gid]);
}

// ============================================================
// Kernel 5 — Batch sqrt
// ============================================================

kernel void sine_sqrt_batch(
    device const float *A       [[buffer(0)]],
    device const float *M       [[buffer(1)]],
    device       float *results [[buffer(2)]],
    device const float *lut     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    results[gid] = sine_sqrt(lut, A[gid], M[gid]);
}

// ============================================================
// Kernel 6 — Full arithmetic pipeline
// A, B pairs — runs add, mul, div, sqrt simultaneously
// ============================================================

struct SineResult {
    float add_r;
    float mul_r;
    float div_r;
    float sqrt_r;
};

kernel void sine_pipeline(
    device const float       *A       [[buffer(0)]],
    device const float       *B       [[buffer(1)]],
    device const float       *M       [[buffer(2)]],
    device       SineResult  *results [[buffer(3)]],
    device const float       *lut     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    SineResult r;
    r.add_r  = sine_add (lut, A[gid], B[gid]);
    r.mul_r  = sine_mul (lut, A[gid], B[gid], M[gid]);
    r.div_r  = sine_div (lut, A[gid], B[gid], M[gid]);
    r.sqrt_r = sine_sqrt(lut, A[gid],          M[gid]);
    results[gid] = r;
}
