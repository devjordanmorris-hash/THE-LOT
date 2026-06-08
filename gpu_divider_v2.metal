#include <metal_stdlib>
using namespace metal;

// Even-first + non-restoring 64-bit unsigned division on GPU.
// Each thread computes one (x,y) -> (q,r).
// Steps:
//  1) k = ctz(y), y1 = y >> k (odd). Split x = 2^k * t + w
//  2) Non-restoring division of t by y1 using bit-by-bit loop from msb(t)
//  3) Rebuild r = (r0 << k) + w
//
// Notes:
//  - Handles y==0 by writing zeros (caller should avoid y==0).
//  - Uses clz to skip leading zeros for fewer than 64 iterations on average.
//  - Power-of-two fastpath: if y1==1 then q = t, r0 = 0.

kernel void div64_evenfirst_nonrestoring(
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

    // Factor out powers of two: y = 2^k * y1, y1 odd
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;

    // Split x = 2^k * t + w
    ulong t = (k ? (x >> k) : x);
    ulong w = (k ? (x & ((1ul << k) - 1ul)) : 0ul);

    // If y1 == 1, fastpath
    if (y1 == 1ul) {
        q_out[gid] = t;
        r_out[gid] = (k ? w : 0ul);
        return;
    }

    // Non-restoring division of t by odd y1
    // Initialize quotient and partial remainder
    ulong q = 0ul;
    long  r = 0; // signed remainder accumulator for non-restoring

    // Determine msb index of t
    uint msb;
    if (t == 0ul) {
        msb = 0u;
    } else {
        msb = 63u - (uint)clz(t);
    }

    // Process bits from msb down to 0
    for (int i = (int)msb; i >= 0; --i) {
        // Shift left and bring next bit of t
        r = (r << 1) | ((long)((t >> i) & 1ul));
        // Decide subtract or add back based on sign
        if (r >= 0) {
            r = r - (long)y1;
        } else {
            r = r + (long)y1;
        }
        // Set quotient bit based on sign after operation
        if (r >= 0) {
            q |= (1ul << i);
        }
    }

    // Final correction for non-restoring to get non-negative remainder
    if (r < 0) {
        r = r + (long)y1;
    }
    ulong r0 = (ulong)r;

    // Rebuild original remainder
    ulong r_orig = (k ? ((r0 << k) + w) : r0);

    q_out[gid] = q;
    r_out[gid] = r_orig;
}
