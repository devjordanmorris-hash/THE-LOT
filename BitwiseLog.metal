#include <metal_stdlib>
using namespace metal;

#define USE_U64 1
#if USE_U64
using UX = ulong;
#else
using UX = uint;
#endif

// ---- Paste YOUR log transform here (GPU version) ----
// Expected: return floor(log_base(x)) for integers (or your exact mapping).
inline UX bitwise_log(UX x, UX base) {
    // TODO: REPLACE this placeholder with your transform.
    // Safe default: count highest power-of-base ≤ x.
    UX y = 0, p = 1;
    while (true) {
        UX np = p * base;
        if (np == 0 || np > x) break;
        p = np; y++;
    }
    return y;
}
// -----------------------------------------------------

inline UX ipow(UX base, UX exp) { // tiny helper for reconstruction check
    UX r = 1, b = base, e = exp;
    while (e) { if (e & 1) r *= b; e >>= 1; if (e) b *= b; }
    return r;
}

kernel void bitwiseLogKernel(
    device const UX* xs    [[ buffer(0) ]],
    device const UX* bases [[ buffer(1) ]],
    device       UX* ys    [[ buffer(2) ]],
    uint tid               [[ thread_position_in_grid ]]
){
    ys[tid] = bitwise_log(xs[tid], bases[tid]);
}
