#include <metal_stdlib>
using namespace metal;

// -------- Helpers --------
inline uint bitlen64(ulong x){ return x ? (64u - (uint)clz(x)) : 0u; }

// 64x64 -> 128 (hi,lo) using 32-bit limbs (no 128-bit types)
inline void mul64wide(ulong a, ulong b, thread ulong &hi, thread ulong &lo){
    ulong a0 = (ulong)(a & 0xfffffffful);
    ulong a1 = (ulong)(a >> 32);
    ulong b0 = (ulong)(b & 0xfffffffful);
    ulong b1 = (ulong)(b >> 32);

    ulong p00 = a0 * b0;          // 64-bit
    ulong p01 = a0 * b1;          // 64-bit
    ulong p10 = a1 * b0;          // 64-bit
    ulong p11 = a1 * b1;          // 64-bit

    ulong mid = (p00 >> 32) + (p01 & 0xfffffffful) + (p10 & 0xfffffffful);
    lo = (p00 & 0xfffffffful) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Right shift a 128-bit composed from (hi:lo) by s (0..127), return low 64 bits
inline ulong shr128(ulong hi, ulong lo, uint s){
    if (s == 0u) return lo;
    if (s < 64u){
        return (lo >> s) | (hi << (64u - s));
    } else if (s < 128u){
        return (hi >> (s - 64u));
    } else {
        return 0ul;
    }
}

// -------- Kernel A: builtin --------
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

// -------- Core #1: radix-4 even-first (restoring) --------
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

// -------- Kernel B: radix-4 ILP --------
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

// -------- Core #2: reciprocal + 1 Newton (exact) without 128-bit --------
inline void div64_recip_nr_core(ulong x, ulong y, thread ulong& q, thread ulong& r){
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;                     // odd
    ulong t  = (k ? (x >> k) : x);
    ulong w  = (k ? (x & ((1ul<<k)-1ul)) : 0ul);

    if (y1 == 1ul){ q = t; r = (k? w:0ul); return; }

    uint ybits = bitlen64(y1);
    uint S = ybits + 16u; if (S > 56u) S = 56u;

    // r0 = floor((1<<S)/y1)
    ulong oneS = (1ul << S);
    ulong r0 = oneS / y1;

    // yr = (y1 * r0) >> S  using wide multiply + shift
    ulong hi, lo;
    mul64wide(y1, r0, hi, lo);
    ulong yr = shr128(hi, lo, S);

    // tmp = (2<<S) - yr
    ulong twoS = (2ul << S);
    ulong tmp = (twoS > yr ? (twoS - yr) : 0ul);

    // r1 = (r0 * tmp) >> S  (wide mul + shift)
    mul64wide(r0, tmp, hi, lo);
    ulong r1 = shr128(hi, lo, S);

    // q0 = (t * r1) >> S
    mul64wide(t, r1, hi, lo);
    ulong q0 = shr128(hi, lo, S);

    // exact correction in odd domain
    // diff = t - q0*y1  (compute q0*y1 via wide mult then subtract)
    mul64wide(q0, y1, hi, lo);
    // Compose (hi:lo) and subtract from (0:t) -> we only need sign and remainder within 0..y1-1
    // diff128 = t - (hi:lo). Since t fits 64, diff can be negative; do manual compare
    // We'll compute r_odd = t - q0*y1 with 128 support via comparisons:
    // If hi==0 and lo<=t -> diff >= 0, r_odd = t - lo
    // If hi>0 or lo>t -> diff < 0, r_odd = ( (1<<64) + t ) - lo  and we need to decrement q0
    ulong r_odd;
    if (hi == 0ul && lo <= t){
        r_odd = t - lo;
        if (r_odd >= y1){ q0++; r_odd -= y1; if (r_odd >= y1){ q0++; r_odd -= y1; } }
    } else {
        // negative: borrow from high
        r_odd = (~0ul - (lo - t) + 1ul); // (2^64 + t - lo)
        if (r_odd >= y1){ q0--; r_odd += y1; } // move toward exact; conservative single step
        // Final guard: ensure r_odd < y1 adjusting q0 accordingly
        if (r_odd >= y1){ q0--; r_odd -= y1; }
    }

    r = (k ? ((r_odd<<k) + w) : r_odd);
    q = q0;
}

// -------- Kernel C: recip+NR ILP --------
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
