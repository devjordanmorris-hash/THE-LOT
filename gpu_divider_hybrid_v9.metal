#include <metal_stdlib>
using namespace metal;

// --------------------
// Helpers
// --------------------
inline uint bitlen64(ulong x){ return x ? (64u - (uint)clz(x)) : 0u; }

// --------------------
// Kernel A: GPU builtin (reference fast path)
// --------------------
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

// --------------------
// Our core #1: radix-4 even-first (restoring), exact
// --------------------
inline void div64_radix4_evenfirst_core(ulong x, ulong y, thread ulong& q, thread ulong& r){
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;                     // odd
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

// --------------------
// Kernel B: radix-4 ILP subset (each thread strides the grid)
// --------------------
kernel void div64_radix4_ilp_subset(
    device const ulong* xs_small [[buffer(0)]],
    device const ulong* ys_small [[buffer(1)]],
    device ulong*       q_small  [[buffer(2)]],
    device ulong*       r_small  [[buffer(3)]],
    constant uint&      n_small  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]],
    uint gridW                    [[threads_per_grid]]
){
    for (uint idx = gid; idx < n_small; idx += gridW){
        ulong q,r;
        div64_radix4_evenfirst_core(xs_small[idx], ys_small[idx], q, r);
        q_small[idx] = q; r_small[idx] = r;
    }
}

// --------------------
// Our core #2: fixed-point reciprocal + 1 Newton + exact correction (float-free)
// Steps (y = 2^k * y1, y1 odd):
//   r0 = floor((1<<S)/y1) in Q_S
//   r1 = r0 * (2 - y1*r0) >> S    (one Newton)
//   q0 = floor(t * r1 >> S)
//   corr: adjust q0 by {-1,0,+1} using y1 to make exact remainder in odd domain
//   rebuild remainder with k and low bits w
// --------------------
inline void div64_recip_nr_core(ulong x, ulong y, thread ulong& q, thread ulong& r){
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;                     // odd
    ulong t  = (k ? (x >> k) : x);
    ulong w  = (k ? (x & ((1ul<<k)-1ul)) : 0ul);

    if (y1 == 1ul){ q = t; r = (k? w:0ul); return; }

    // Choose S with headroom;  y1 up to 63 bits -> S <= 56 keeps 128-bit intermediates comfy
    uint ybits = bitlen64(y1);
    uint S = ybits + 16u; if (S > 56u) S = 56u;

    // r0 = floor((1<<S)/y1)
    ulong r0 = (ulong)((((ulong)1) << S) / y1);

    // Newton step in Q_S: r1 = r0 * (2 - y1*r0) >> S
    ulong twoS_lo = (ulong)(2ull << S); // fits in 64+ since S<=56
    ulong yr_lo = (ulong)((((ulong)y1) * r0) >> S);
    // Use 128-bit intermediates via mul_hi/lo by widening to 128 with builtin.
    // In Metal, 128-bit isn't native; emulate with builtin mulhi for 64x64→128 pieces.
    // However, for our restricted S<=56, we can safely compute with 128 by casting to ulong2 and manual steps.
    // Simpler: perform with 128 via builtin functions: use mulhi for high part and reconstruct.
    // We'll do (r0 * (two - yr)) >> S using 128-bit emulation.
    ulong two_minus_yr = (twoS_lo > (yr_lo<<S) ? (twoS_lo - (yr_lo<<S)) : 0ull); // safe fallback

    // Fallback to a simpler refinement: do one Newton using wider math via explicit 128 pieces
    // Compute X = r0 * ( (2<<S) - (y1*r0) ) >> S, but approximate (y1*r0) in Q_S using 64-bit path above.
    ulong r1 = (ulong)(((ulong)(((__uint128_t)r0 * ( (__uint128_t)( (2ull<<S) - (((__uint128_t)y1 * r0)>>S) ) )) >> S)));

    // q0 = floor(t * r1 >> S)
    ulong q0 = (ulong)(((__uint128_t)t * r1) >> S);

    // exact correction in odd domain
    __int128 diff = (__int128)t - (__int128)((__uint128_t)q0 * y1);
    if (diff >= (__int128)y1) { q0++; diff -= (__int128)y1; if (diff >= (__int128)y1){ q0++; diff -= (__int128)y1; } }
    else if (diff < 0) { q0--; diff += (__int128)y1; }

    ulong r0odd = (ulong)diff;
    r = (k ? ((r0odd<<k) + w) : r0odd);
    q = q0;
}

// --------------------
// Kernel C: reciprocal+NR ILP subset (stride loop)
// --------------------
kernel void div64_recip_nr_ilp_subset(
    device const ulong* xs_small [[buffer(0)]],
    device const ulong* ys_small [[buffer(1)]],
    device ulong*       q_small  [[buffer(2)]],
    device ulong*       r_small  [[buffer(3)]],
    constant uint&      n_small  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]],
    uint gridW                    [[threads_per_grid]]
){
    for (uint idx = gid; idx < n_small; idx += gridW){
        ulong q,r;
        div64_recip_nr_core(xs_small[idx], ys_small[idx], q, r);
        q_small[idx] = q; r_small[idx] = r;
    }
}
