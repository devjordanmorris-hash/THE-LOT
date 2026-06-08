#include <metal_stdlib>
using namespace metal;

kernel void nativeAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    out[id] = A[id] + B[id];
}

/*
    4-bit nibble LUT.

    LUT index:
        idx = (aNibble << 4) | bNibble

    LUT entry:
        bits 0..3 = result if carry-in 0
        bit  4    = carry-out if carry-in 0
        bits 5..8 = result if carry-in 1
        bit  9    = carry-out if carry-in 1

    LUT size = 16 * 16 = 256 entries.
*/
kernel void nibbleLUTAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device const uint *lut [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint result = 0;
    uint carry = 0;

    for (uint shift = 0; shift < 32; shift += 4) {
        uint an = (a >> shift) & 0xfu;
        uint bn = (b >> shift) & 0xfu;

        uint entry = lut[(an << 4) | bn];

        uint r0 = entry & 0xfu;
        uint c0 = (entry >> 4) & 1u;
        uint r1 = (entry >> 5) & 0xfu;
        uint c1 = (entry >> 9) & 1u;

        uint sel = 0u - carry;

        uint r = (r0 & ~sel) | (r1 & sel);
        carry = (c0 & ~sel) | (c1 & sel);

        result |= r << shift;
    }

    out[id] = result;
}

/*
    Direct 4-bit split add, no LUT.
    Computes both carry futures for each nibble.
*/
inline uint add4_chunk(uint aNib, uint bNib, uint carryIn) {
    uint sum = aNib + bNib + carryIn;
    uint result = sum & 0xfu;
    uint carryOut = (sum >> 4) & 1u;
    return result | (carryOut << 4);
}

kernel void nibbleDirectAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint result = 0;
    uint carry = 0;

    for (uint shift = 0; shift < 32; shift += 4) {
        uint an = (a >> shift) & 0xfu;
        uint bn = (b >> shift) & 0xfu;

        uint e0 = add4_chunk(an, bn, 0);
        uint e1 = add4_chunk(an, bn, 1);

        uint r0 = e0 & 0xfu;
        uint c0 = (e0 >> 4) & 1u;
        uint r1 = e1 & 0xfu;
        uint c1 = (e1 >> 4) & 1u;

        uint sel = 0u - carry;

        uint r = (r0 & ~sel) | (r1 & sel);
        carry = (c0 & ~sel) | (c1 & sel);

        result |= r << shift;
    }

    out[id] = result;
}

/*
    Unrolled direct 8-bit split from previous best-ish method.
*/
inline uint add8_chunk(uint aByte, uint bByte, uint carryIn) {
    uint sum = aByte + bByte + carryIn;
    uint result = sum & 0xffu;
    uint carryOut = (sum >> 8) & 1u;
    return result | (carryOut << 8);
}

kernel void split8ChunkAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint a0 =  a        & 0xffu;
    uint a1 = (a >> 8)  & 0xffu;
    uint a2 = (a >> 16) & 0xffu;
    uint a3 = (a >> 24) & 0xffu;

    uint b0 =  b        & 0xffu;
    uint b1 = (b >> 8)  & 0xffu;
    uint b2 = (b >> 16) & 0xffu;
    uint b3 = (b >> 24) & 0xffu;

    uint c00 = add8_chunk(a0, b0, 0);
    uint c10 = add8_chunk(a1, b1, 0);
    uint c11 = add8_chunk(a1, b1, 1);
    uint c20 = add8_chunk(a2, b2, 0);
    uint c21 = add8_chunk(a2, b2, 1);
    uint c30 = add8_chunk(a3, b3, 0);
    uint c31 = add8_chunk(a3, b3, 1);

    uint r0 = c00 & 0xffu;
    uint carry = (c00 >> 8) & 1u;

    uint sel = 0u - carry;
    uint r1 = ((c10 & 0xffu) & ~sel) | ((c11 & 0xffu) & sel);
    carry = (((c10 >> 8) & 1u) & ~sel) | (((c11 >> 8) & 1u) & sel);

    sel = 0u - carry;
    uint r2 = ((c20 & 0xffu) & ~sel) | ((c21 & 0xffu) & sel);
    carry = (((c20 >> 8) & 1u) & ~sel) | (((c21 >> 8) & 1u) & sel);

    sel = 0u - carry;
    uint r3 = ((c30 & 0xffu) & ~sel) | ((c31 & 0xffu) & sel);

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}
