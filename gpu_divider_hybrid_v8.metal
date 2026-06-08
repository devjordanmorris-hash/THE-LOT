#include <metal_stdlib>
using namespace metal;

// =====================
// Helper: bit length
// =====================
inline uint bitlen64(ulong x){
    return x ? (64u - (uint)clz(x)) : 0u;
}

// =====================
// GPU builtin subset
// =====================
kernel void div64_builtin_subset(
    device const ulong* xs_small [[buffer(0)]],
    device const ulong* ys_small [[buffer(1)]],
    device ulong*       q_small  [[buffer(2)]],
    device ulong*       r_small  [[buffer(3)]],
    constant uint&      n_small  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
){
    if (gid >= n_small) return;
    ulong y = ys_small[gid];
    if (y == 0ul) { q_small[gid]=0ul; r_small[gid]=0ul; return; }
    ulong x = xs_small[gid];
    q_small[gid] = x / y;
    r_small[gid] = x % y;
}

// =====================
// Our radix-4 even-first core (exact), branch-light
// =====================
inline void div64_radix4_evenfirst_core(ulong x, ulong y, thread ulong& q, thread ulong& r){
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;                // odd
    ulong t  = (k ? (x >> k) : x);
    ulong w  = (k ? (x & ((1ul<<k)-1ul)) : 0ul);

    if (y1 == 1ul){ q = t; r = (k? w:0ul); return; }

    ulong Q=0ul, R=0ul, yv=y1;
    uint msb = (t==0ul)? 0u : (63u - (uint)clz(t));
    int pairs = (int)(msb/2u);

    for (int j=pairs; j>=0; --j){
        uint i = (uint)(j*2u);
        R = (R<<2) | ((t>>i) & 3ul);
        ulong ge1 = (ulong)(R >= yv);
        ulong R1  = R - (yv & -(long)ge1);
        ulong ge2 = (ulong)(R1 >= yv);
        ulong R2  = R1 - (yv & -(long)ge2);
        ulong ge3 = (ulong)(R2 >= yv);
        ulong R3  = R2 - (yv & -(long)ge3);
        ulong addQ = ge1 + ge2 + ge3;          // 0..3
        R = R3;
        Q |= (addQ << i);
    }
    ulong r0 = R;
    r = (k ? ((r0<<k) + w) : r0);
    q = Q;
}

// =====================
// Radix-4 ILP subset: each thread processes 2 items in a stride
// =====================
kernel void div64_radix4_ilp_subset(
    device const ulong* xs_small [[buffer(0)]],
    device const ulong* ys_small [[buffer(1)]],
    device ulong*       q_small  [[buffer(2)]],
    device ulong*       r_small  [[buffer(3)]],
    constant uint&      n_small  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]],
    uint gridW                    [[threads_per_grid]]
){
    uint stride = gridW;
    for (uint idx = gid; idx < n_small; idx += stride){
        ulong q,r;
        div64_radix4_evenfirst_core(xs_small[idx], ys_small[idx], q, r);
        q_small[idx] = q; r_small[idx] = r;
    }
}
