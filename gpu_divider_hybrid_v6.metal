#include <metal_stdlib>
using namespace metal;

// (GPU) Even-first + RADIX-4 restoring (2 bits per iteration), exact
inline void div64_radix4_evenfirst_core(ulong x, ulong y, thread ulong& q, thread ulong& r)
{
    uint k = (uint)ctz(y);        // y = 2^k * y1
    ulong y1 = y >> k;            // odd part

    // Split x = 2^k * t + w
    ulong t = (k ? (x >> k) : x);
    ulong w = (k ? (x & ((1ul << k) - 1ul)) : 0ul);

    if (y1 == 1ul) {              // power-of-two fastpath
        q = t;
        r = (k ? w : 0ul);
        return;
    }

    // Two bits per iteration (branch-light with masked subtracts)
    ulong Q = 0ul;
    ulong R = 0ul;
    ulong yv = y1;

    uint msb = (t == 0ul) ? 0u : (63u - (uint)clz(t));
    int pairs = (int)(msb / 2u);

    for (int j = pairs; j >= 0; --j) {
        uint i = (uint)(j * 2u);
        R = (R << 2) | ((t >> i) & 3ul);

        ulong ge1 = (ulong)(R >= yv);
        ulong R1  = R - (yv & -(long)ge1);

        ulong ge2 = (ulong)(R1 >= yv);
        ulong R2  = R1 - (yv & -(long)ge2);

        ulong ge3 = (ulong)(R2 >= yv);
        ulong R3  = R2 - (yv & -(long)ge3);

        ulong addQ = ge1 + ge2 + ge3; // 0..3
        R = R3;
        Q |= (addQ << i);
    }

    ulong r0 = R;
    ulong r_orig = (k ? ((r0 << k) + w) : r0);
    q = Q; r = r_orig;
}

// Subset kernel: processes compacted small indices
kernel void div64_radix4_evenfirst_subset(
    device const ulong* xs_small [[buffer(0)]],
    device const ulong* ys_small [[buffer(1)]],
    device ulong*       q_small  [[buffer(2)]],
    device ulong*       r_small  [[buffer(3)]],
    constant uint&      n_small  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
){
    if (gid >= n_small) return;
    ulong q, r;
    div64_radix4_evenfirst_core(xs_small[gid], ys_small[gid], q, r);
    q_small[gid] = q;
    r_small[gid] = r;
}
