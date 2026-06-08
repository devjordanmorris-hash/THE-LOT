#include <metal_stdlib>
using namespace metal;

#define USE_U64 1
#if USE_U64
using UX = ulong;   // 64-bit
#else
using UX = uint;    // 32-bit
#endif

inline UX bitwise_pow(UX x, UX n) {
    UX p = x;
    const uint BW = sizeof(UX) * 8;
    for (uint k = 0; k < BW && ((UX(1) << k) <= n); ++k) {
        UX b = (n >> k) & UX(1);
        UX left  = p << b;
        UX right = p >> (1 - b);
        p = p + (left - right);
    }
    return p;
}

kernel void bitwisePowKernel(
    device const UX* x   [[ buffer(0) ]],
    device const UX* n   [[ buffer(1) ]],
    device       UX* out [[ buffer(2) ]],
    uint tid             [[ thread_position_in_grid ]]
){
    // We dispatch exactly N threads from the host, so tid is always in-range.
    out[tid] = bitwise_pow(x[tid], n[tid]);
}