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
    Add one 8-bit chunk with carry-in.
    Returns:
      low  8 bits: chunk result
      bit  8: carry out
*/
inline uint add8_chunk(uint aByte, uint bByte, uint carryIn) {
    uint sum = aByte + bByte + carryIn;
    uint result = sum & 0xffu;
    uint carryOut = (sum >> 8) & 1u;
    return result | (carryOut << 8);
}

/*
    Split 32-bit add into four 8-bit chunks.

    For each chunk, compute both possible states:
      carry-in 0
      carry-in 1

    Then resolve the 4 chunk-boundary carries.

    This is not yet fully parallel-prefix, but it proves the split/chunk idea
    and creates a path to make the carry reconciliation smaller.
*/
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
    Shifted split add:
      result = ((A << SHIFT) + (B << SHIFT)) >> SHIFT

    SHIFT must be small enough that overflow is controlled by host input range.
*/
kernel void shiftedSplitChunkAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    constant uint &SHIFT [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id] << SHIFT;
    uint b = B[id] << SHIFT;

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

    uint shiftedResult = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
    out[id] = shiftedResult >> SHIFT;
}
