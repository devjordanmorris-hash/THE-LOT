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
    Direct split-chunk add, for comparison.
*/
inline uint add8_chunk(uint aByte, uint bByte, uint carryIn) {
    uint sum = aByte + bByte + carryIn;
    uint result = sum & 0xffu;
    uint carryOut = (sum >> 8) & 1u;
    return result | (carryOut << 8);
}

kernel void splitChunkAddKernel(
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

    uint c1 = (carry == 0u) ? c10 : c11;
    uint r1 = c1 & 0xffu;
    carry = (c1 >> 8) & 1u;

    uint c2 = (carry == 0u) ? c20 : c21;
    uint r2 = c2 & 0xffu;
    carry = (c2 >> 8) & 1u;

    uint c3 = (carry == 0u) ? c30 : c31;
    uint r3 = c3 & 0xffu;

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

/*
    Packed dual-carry LUT.

    LUT index:
        idx = (aByte << 8) | bByte

    LUT entry layout:
        bits  0..7   = result if carry-in 0
        bit   8      = carry-out if carry-in 0
        bits  9..16  = result if carry-in 1
        bit   17     = carry-out if carry-in 1

    This gives both possible carry futures from one lookup.
*/
kernel void packedDualCarryLUTAddKernel(
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

    uint a0 =  a        & 0xffu;
    uint a1 = (a >> 8)  & 0xffu;
    uint a2 = (a >> 16) & 0xffu;
    uint a3 = (a >> 24) & 0xffu;

    uint b0 =  b        & 0xffu;
    uint b1 = (b >> 8)  & 0xffu;
    uint b2 = (b >> 16) & 0xffu;
    uint b3 = (b >> 24) & 0xffu;

    uint e0 = lut[(a0 << 8) | b0];
    uint e1 = lut[(a1 << 8) | b1];
    uint e2 = lut[(a2 << 8) | b2];
    uint e3 = lut[(a3 << 8) | b3];

    uint r0 = e0 & 0xffu;
    uint carry = (e0 >> 8) & 1u;

    uint r1_0 = e1 & 0xffu;
    uint c1_0 = (e1 >> 8) & 1u;
    uint r1_1 = (e1 >> 9) & 0xffu;
    uint c1_1 = (e1 >> 17) & 1u;

    uint r1 = (carry == 0u) ? r1_0 : r1_1;
    carry = (carry == 0u) ? c1_0 : c1_1;

    uint r2_0 = e2 & 0xffu;
    uint c2_0 = (e2 >> 8) & 1u;
    uint r2_1 = (e2 >> 9) & 0xffu;
    uint c2_1 = (e2 >> 17) & 1u;

    uint r2 = (carry == 0u) ? r2_0 : r2_1;
    carry = (carry == 0u) ? c2_0 : c2_1;

    uint r3_0 = e3 & 0xffu;
    uint r3_1 = (e3 >> 9) & 0xffu;

    uint r3 = (carry == 0u) ? r3_0 : r3_1;

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

/*
    Branchless select version of packed LUT.
*/
kernel void packedDualCarryLUTAddBranchlessKernel(
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

    uint a0 =  a        & 0xffu;
    uint a1 = (a >> 8)  & 0xffu;
    uint a2 = (a >> 16) & 0xffu;
    uint a3 = (a >> 24) & 0xffu;

    uint b0 =  b        & 0xffu;
    uint b1 = (b >> 8)  & 0xffu;
    uint b2 = (b >> 16) & 0xffu;
    uint b3 = (b >> 24) & 0xffu;

    uint e0 = lut[(a0 << 8) | b0];
    uint e1 = lut[(a1 << 8) | b1];
    uint e2 = lut[(a2 << 8) | b2];
    uint e3 = lut[(a3 << 8) | b3];

    uint r0 = e0 & 0xffu;
    uint carry = (e0 >> 8) & 1u;

    uint r1_0 = e1 & 0xffu;
    uint c1_0 = (e1 >> 8) & 1u;
    uint r1_1 = (e1 >> 9) & 0xffu;
    uint c1_1 = (e1 >> 17) & 1u;
    uint sel = 0u - carry;
    uint r1 = (r1_0 & ~sel) | (r1_1 & sel);
    carry = (c1_0 & ~sel) | (c1_1 & sel);

    uint r2_0 = e2 & 0xffu;
    uint c2_0 = (e2 >> 8) & 1u;
    uint r2_1 = (e2 >> 9) & 0xffu;
    uint c2_1 = (e2 >> 17) & 1u;
    sel = 0u - carry;
    uint r2 = (r2_0 & ~sel) | (r2_1 & sel);
    carry = (c2_0 & ~sel) | (c2_1 & sel);

    uint r3_0 = e3 & 0xffu;
    uint r3_1 = (e3 >> 9) & 0xffu;
    sel = 0u - carry;
    uint r3 = (r3_0 & ~sel) | (r3_1 & sel);

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}
