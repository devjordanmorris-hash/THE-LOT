#include <metal_stdlib>
using namespace metal;

// Optimized restoring division for 64-bit unsigned integers.
// Uses clz() to skip leading zero bits => fewer than 64 iterations on average.
kernel void div64_kernel(
    device const ulong* xs       [[ buffer(0) ]],
    device const ulong* ys       [[ buffer(1) ]],
    device ulong*       q_out    [[ buffer(2) ]],
    device ulong*       r_out    [[ buffer(3) ]],
    constant uint&      n        [[ buffer(4) ]],
    uint gid                     [[ thread_position_in_grid ]]
){
    if (gid >= n) return;
    ulong x = xs[gid];
    ulong y = ys[gid];
    if (y == 0) { q_out[gid] = 0; r_out[gid] = 0; return; }

    // Determine the top bit we need to process.
    // If x == 0, clz(0) is undefined; handle with msb = 0 in that case.
    uint msb;
    if (x == 0) {
        msb = 0;
    } else {
        uint lz = clz(x);
        msb = 63u - lz;
    }

    ulong q = 0;
    ulong r = 0;
    for (int i = int(msb); i >= 0; --i) {
        r = (r << 1) | ((x >> i) & 1ul);
        if (r >= y) {
            r -= y;
            q |= (1ul << i);
        }
    }
    q_out[gid] = q;
    r_out[gid] = r;
}
