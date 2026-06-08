// prism_cuda_tiles.cu
// CUDA harness: sine via "prism tiles" (fixed-length cubic per tile).
//
// Build (Windows):
//   nvcc -allow-unsupported-compiler -O3 -use_fast_math prism_cuda_tiles.cu -o prism_cuda_tiles.exe
// Run:
//   prism_cuda_tiles.exe
//
// Notes:
// - Fixed-size tiles (e.g. 256 samples) are GPU-friendly.
// - Validation compares against repeated LUT truth (no phase-drift baseline).

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>

static constexpr double PI  = 3.141592653589793238462643383279502884;
static constexpr double TAU = 2.0 * PI;

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

struct TileCubic {
    float a0, a1, a2, a3; // y(k)=a0+a1*k+a2*k^2+a3*k^3, k in [0..T-1]
};

// Build truth samples for one cycle on a fixed grid
static std::vector<float> build_truth_cycle(uint32_t samples_per_cycle) {
    std::vector<float> y(samples_per_cycle);
    const double dtheta = TAU / (double)samples_per_cycle;
    for (uint32_t i = 0; i < samples_per_cycle; ++i) {
        y[i] = (float)std::sin((double)i * dtheta);
    }
    return y;
}

// Hermite cubic on t in [0,1] with y0,y1 and dy/dt endpoints, then substitute t = k/(T-1)
static inline TileCubic hermite_tile_coeffs(float y0, float y1, float dy_dk0, float dy_dk1, int T) {
    // dy/dt = dy/dk * (T-1)
    float m0 = dy_dk0 * (float)(T - 1);
    float m1 = dy_dk1 * (float)(T - 1);

    // Hermite coefficients in t:
    // y(t)=c0 + c1 t + c2 t^2 + c3 t^3
    float c0 = y0;
    float c1 = m0;
    float c2 = 3.0f*(y1 - y0) - 2.0f*m0 - m1;
    float c3 = 2.0f*(y0 - y1) + m0 + m1;

    // Substitute t = k/(T-1)
    float s  = 1.0f / (float)(T - 1);
    float s2 = s * s;
    float s3 = s2 * s;

    TileCubic out{};
    out.a0 = c0;
    out.a1 = c1 * s;
    out.a2 = c2 * s2;
    out.a3 = c3 * s3;
    return out;
}

static inline float eval_cubic(const TileCubic& t, int k) {
    float x = (float)k;
    return ((t.a3 * x + t.a2) * x + t.a1) * x + t.a0;
}

// Build fixed tiles for one cycle. samples_per_cycle must be divisible by TILE.
static std::vector<TileCubic> build_tiles_from_truth(const std::vector<float>& truth_cycle, int TILE) {
    const uint32_t N = (uint32_t)truth_cycle.size();
    if (N % (uint32_t)TILE != 0) {
        std::cerr << "samples_per_cycle must be divisible by TILE\n";
        std::exit(1);
    }

    std::vector<TileCubic> tiles;
    const uint32_t tile_count = N / (uint32_t)TILE;
    tiles.reserve(tile_count);

    const float dtheta = (float)(TAU / (double)N);

    for (uint32_t ti = 0; ti < tile_count; ++ti) {
        uint32_t start = ti * (uint32_t)TILE;
        uint32_t end   = start + (uint32_t)(TILE - 1);

        float y0 = truth_cycle[start];
        float y1 = truth_cycle[end];

        double th0 = (double)start * (double)dtheta;
        double th1 = (double)end   * (double)dtheta;

        // dy/dtheta = cos(theta), dy/dk = cos(theta)*dtheta
        float dy_dk0 = (float)(std::cos(th0) * (double)dtheta);
        float dy_dk1 = (float)(std::cos(th1) * (double)dtheta);

        tiles.push_back(hermite_tile_coeffs(y0, y1, dy_dk0, dy_dk1, TILE));
    }
    return tiles;
}

// CPU: repeat cycle for validation (aligned, no drift)
static std::vector<float> repeat_cycle(const std::vector<float>& cycle, int repeat_cycles) {
    std::vector<float> out((size_t)cycle.size() * (size_t)repeat_cycles);
    for (int r = 0; r < repeat_cycles; ++r) {
        std::copy(cycle.begin(), cycle.end(), out.begin() + (size_t)r * cycle.size());
    }
    return out;
}

static void error_stats(const std::vector<float>& a, const std::vector<float>& b, double& rms, double& maxe) {
    size_t n = std::min(a.size(), b.size());
    long double acc = 0.0L;
    long double mx  = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        long double e  = (long double)a[i] - (long double)b[i];
        acc += e * e;
        long double ae = fabsl(e);
        if (ae > mx) mx = ae;
    }
    rms  = std::sqrt((double)(acc / (long double)n));
    maxe = (double)mx;
}

// ---------------- CUDA kernel ----------------
// One block per tile instance in the full output.
// total_tiles_out = tiles_per_cycle * repeat_cycles.
// Each block generates TILE samples.
__global__ void decode_tiles_kernel(
    const TileCubic* __restrict__ tiles,   // per-cycle tiles
    int tiles_per_cycle,
    int TILE,
    float* __restrict__ out
) {
    int global_tile  = (int)blockIdx.x;            // 0..total_tiles_out-1
    int tile_in_cycle = global_tile % tiles_per_cycle;

    TileCubic t = tiles[tile_in_cycle];

    int base = global_tile * TILE;
    int tid  = (int)threadIdx.x;

    for (int k = tid; k < TILE; k += (int)blockDim.x) {
        float x = (float)k;
        float y = fmaf(fmaf(fmaf(t.a3, x, t.a2), x, t.a1), x, t.a0);
        out[base + k] = y;
    }
}

static float gpu_decode_tiles(
    const std::vector<TileCubic>& tiles,
    int tiles_per_cycle,
    int repeat_cycles,
    int TILE,
    float* d_out
) {
    TileCubic* d_tiles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tiles, tiles.size() * sizeof(TileCubic)));
    CUDA_CHECK(cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(TileCubic), cudaMemcpyHostToDevice));

    int total_tiles_out = tiles_per_cycle * repeat_cycles;

    dim3 block(256);
    dim3 grid(total_tiles_out);

    // warmup
    decode_tiles_kernel<<<grid, block>>>(d_tiles, tiles_per_cycle, TILE, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    decode_tiles_kernel<<<grid, block>>>(d_tiles, tiles_per_cycle, TILE, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_tiles));
    return ms;
}

int main() {
    // ---- knobs ----
    const uint32_t SAMPLES_PER_CYCLE = 32768;
    const int TILE = 256;
    const int REPEAT_CYCLES = 400;

    if (SAMPLES_PER_CYCLE % (uint32_t)TILE != 0) {
        std::cerr << "SAMPLES_PER_CYCLE must be divisible by TILE.\n";
        return 1;
    }

    const int tiles_per_cycle = (int)(SAMPLES_PER_CYCLE / (uint32_t)TILE);
    const uint64_t TOTAL_SAMPLES = (uint64_t)SAMPLES_PER_CYCLE * (uint64_t)REPEAT_CYCLES;

    std::cout << "SAMPLES_PER_CYCLE=" << SAMPLES_PER_CYCLE
              << " TILE=" << TILE
              << " tiles_per_cycle=" << tiles_per_cycle
              << " REPEAT_CYCLES=" << REPEAT_CYCLES
              << " TOTAL_SAMPLES=" << TOTAL_SAMPLES
              << "\n";

    // CPU build
    auto truth_cycle = build_truth_cycle(SAMPLES_PER_CYCLE);
    auto tiles = build_tiles_from_truth(truth_cycle, TILE);
    auto truth_total = repeat_cycle(truth_cycle, REPEAT_CYCLES);

    // CPU sanity (one cycle)
    double rms1=0, max1=0;
    {
        std::vector<float> approx_cycle(SAMPLES_PER_CYCLE);
        for (int ti = 0; ti < tiles_per_cycle; ++ti) {
            const TileCubic& t = tiles[ti];
            int base = ti * TILE;
            for (int k = 0; k < TILE; ++k) approx_cycle[base + k] = eval_cubic(t, k);
        }
        error_stats(approx_cycle, truth_cycle, rms1, max1);
    }
    std::cout << "CPU tile approx (1 cycle) RMS=" << rms1 << " max=" << max1 << "\n";

    // GPU allocate output
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)TOTAL_SAMPLES * sizeof(float)));

    // GPU decode timing
    float ms_kernel = gpu_decode_tiles(tiles, tiles_per_cycle, REPEAT_CYCLES, TILE, d_out);

    // Copy back for validation (in a real GPU pipeline, avoid this)
    std::vector<float> out_total((size_t)TOTAL_SAMPLES);
    CUDA_CHECK(cudaMemcpy(out_total.data(), d_out, (size_t)TOTAL_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    // Validate vs repeated truth
    double rms=0, maxe=0;
    error_stats(out_total, truth_total, rms, maxe);

    double samples_per_ms = (double)TOTAL_SAMPLES / (double)ms_kernel;
    double gsamples_per_s = (samples_per_ms * 1e3) / 1e9;

    std::cout << "GPU kernel time: " << ms_kernel << " ms\n";
    std::cout << "Throughput: " << (samples_per_ms * 1e3) << " samples/sec ("
              << gsamples_per_s << " Gsamples/s)\n";
    std::cout << "Error vs repeated truth: RMS=" << rms << " max=" << maxe << "\n";

    return 0;
}
