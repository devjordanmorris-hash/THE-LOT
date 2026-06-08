#include <metal_stdlib>
using namespace metal;

// (A) Built-in (Metal's own / and % for ulong)
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

// (B) Our path: Even-first + RESTORING division (exact)
kernel void div64_evenfirst_restoring(
    device const ulong* xs [[buffer(0)]],
    device const ulong* ys [[buffer(1)]],
    device ulong* q_out    [[buffer(2)]],
    device ulong* r_out    [[buffer(3)]],
    constant uint& n       [[buffer(4)]],
    uint gid               [[thread_position_in_grid]]
){
    if (gid >= n) return;
    ulong x = xs[gid];
    ulong y = ys[gid];
    if (y == 0ul) { q_out[gid]=0ul; r_out[gid]=0ul; return; }

    // Factor out powers of two
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;

    // Split x = 2^k * t + w
    ulong t = (k ? (x >> k) : x);
    ulong w = (k ? (x & ((1ul << k) - 1ul)) : 0ul);

    if (y1 == 1ul) {
        q_out[gid] = t;
        r_out[gid] = (k ? w : 0ul);
        return;
    }

    // RESTORING division
    ulong q = 0ul;
    ulong r = 0ul;
    uint msb = (t == 0ul) ? 0u : (63u - (uint)clz(t));
    for (int i = (int)msb; i >= 0; --i) {
        r = (r << 1) | ((t >> i) & 1ul);
        if (r >= y1) {
            r -= y1;
            q |= (1ul << i);
        }
    }

    ulong r0 = r;
    ulong r_orig = (k ? ((r0 << k) + w) : r0);
    q_out[gid] = q;
    r_out[gid] = r_orig;
}
