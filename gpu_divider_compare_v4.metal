#include <metal_stdlib>
using namespace metal;

// (A) Built-in GPU path: rely on Metal's / and % for ulong
kernel void div64_builtin(
    device const ulong* xs [[buffer(0)]],
    device const ulong* ys [[buffer(1)]],
    device ulong* q_out    [[buffer(2)]],
    device ulong* r_out    [[buffer(3)]],
    constant uint& n       [[buffer(4)]],
    uint gid               [[thread_position_in_grid]]
){
    if (gid >= n) return;
    ulong y = ys[gid];
    if (y == 0ul) { q_out[gid]=0ul; r_out[gid]=0ul; return; }
    ulong x = xs[gid];
    q_out[gid] = x / y;
    r_out[gid] = x % y;
}

// (B) Our path: Even-first + RADIX-4 restoring (2 bits per iteration), exact
inline void div64_radix4_evenfirst_core(ulong x, ulong y, thread ulong& q, thread ulong& r)
{
    // Preconditions: y>0
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

    // Precompute multiples (avoid overflow by repeated subtract later if needed)
    ulong m1 = y1;
    ulong m2 = y1 << 1;           // OK in unsigned, we guard usage below
    ulong m3 = m1 + m2;           // may wrap if y1 is huge; we will gate it

    // Find highest bit in t and compute number of 2-bit pairs
    uint msb = (t == 0ul) ? 0u : (63u - (uint)clz(t));
    int pairs = (int)(msb / 2u);  // number of 2-bit groups [pairs..0]

    ulong Q = 0ul;
    ulong R = 0ul;

    // Process two bits per iteration
    for (int j = pairs; j >= 0; --j) {
        uint i = (uint)(j * 2u);              // bit index of low bit in the pair
        // bring next two bits of t
        R = (R << 2) | ((t >> i) & 3ul);

        // Choose how many y1 to subtract: try up to 3 times with guards
        // Use repeated guarded subtract to avoid overflow issues
        ulong Rtmp = R;
        ulong addQ = 0ul;

        if (Rtmp >= y1) { Rtmp -= y1; addQ += 1ul; }
        if (Rtmp >= y1) { Rtmp -= y1; addQ += 1ul; }
        if (Rtmp >= y1) { Rtmp -= y1; addQ += 1ul; }

        R = Rtmp;
        // addQ in [0..3] represents two quotient bits; place at (i) and (i+1)
        Q |= (addQ << i);
    }

    // Rebuild original remainder: r = (R << k) + w (guaranteed < y)
    ulong r0 = R;
    ulong r_orig = (k ? ((r0 << k) + w) : r0);

    q = Q;
    r = r_orig;
}

kernel void div64_radix4_evenfirst(
    device const ulong* xs [[buffer(0)]],
    device const ulong* ys [[buffer(1)]],
    device ulong* q_out    [[buffer(2)]],
    device ulong* r_out    [[buffer(3)]],
    constant uint& n       [[buffer(4)]],
    uint gid               [[thread_position_in_grid]]
){
    if (gid >= n) return;
    ulong q, r;
    div64_radix4_evenfirst_core(xs[gid], ys[gid], q, r);
    q_out[gid] = q;
    r_out[gid] = r;
}
