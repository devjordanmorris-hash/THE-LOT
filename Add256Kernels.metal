#include <metal_stdlib>
using namespace metal;

/*
    256-bit integers represented as 8 x uint32 words.
    word 0 = least significant 32 bits
    word 7 = most significant 32 bits

    Native-style 256-bit add:
      sequential carry across 8 words.

    Split method:
      inside each 32-bit word, split into 4 x 8-bit chunks.
      compute carry futures locally.
      still reconcile word carry boundaries.
*/

inline uint add8_chunk(uint aByte, uint bByte, uint carryIn) {
    uint sum = aByte + bByte + carryIn;
    uint result = sum & 0xffu;
    uint carryOut = (sum >> 8) & 1u;
    return result | (carryOut << 8);
}

inline uint add32_split8(uint a, uint b, uint carryIn, thread uint &carryOut) {
    uint a0 =  a        & 0xffu;
    uint a1 = (a >> 8)  & 0xffu;
    uint a2 = (a >> 16) & 0xffu;
    uint a3 = (a >> 24) & 0xffu;

    uint b0 =  b        & 0xffu;
    uint b1 = (b >> 8)  & 0xffu;
    uint b2 = (b >> 16) & 0xffu;
    uint b3 = (b >> 24) & 0xffu;

    uint c0in = add8_chunk(a0, b0, carryIn);

    uint c10 = add8_chunk(a1, b1, 0);
    uint c11 = add8_chunk(a1, b1, 1);

    uint c20 = add8_chunk(a2, b2, 0);
    uint c21 = add8_chunk(a2, b2, 1);

    uint c30 = add8_chunk(a3, b3, 0);
    uint c31 = add8_chunk(a3, b3, 1);

    uint r0 = c0in & 0xffu;
    uint carry = (c0in >> 8) & 1u;

    uint sel = 0u - carry;
    uint r1 = ((c10 & 0xffu) & ~sel) | ((c11 & 0xffu) & sel);
    carry = (((c10 >> 8) & 1u) & ~sel) | (((c11 >> 8) & 1u) & sel);

    sel = 0u - carry;
    uint r2 = ((c20 & 0xffu) & ~sel) | ((c21 & 0xffu) & sel);
    carry = (((c20 >> 8) & 1u) & ~sel) | (((c21 >> 8) & 1u) & sel);

    sel = 0u - carry;
    uint r3 = ((c30 & 0xffu) & ~sel) | ((c31 & 0xffu) & sel);
    carryOut = (((c30 >> 8) & 1u) & ~sel) | (((c31 >> 8) & 1u) & sel);

    return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

/*
    Baseline 256-bit add using 64-bit word sums.
    This is the obvious sequential word-carry version.
*/
kernel void add256NativeWordKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * 8u;
    ulong carry = 0;

    for (uint i = 0; i < 8; i++) {
        ulong sum = (ulong)A[base + i] + (ulong)B[base + i] + carry;
        OUT[base + i] = (uint)sum;
        carry = sum >> 32;
    }
}

/*
    Split 256-bit add:
      each 32-bit word is internally split into 8-bit carry futures.
      carry still flows between the 8 words.
*/
kernel void add256Split8Kernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * 8u;
    uint carry = 0;

    for (uint i = 0; i < 8; i++) {
        uint carryOut = 0;
        uint r = add32_split8(A[base + i], B[base + i], carry, carryOut);
        OUT[base + i] = r;
        carry = carryOut;
    }
}

/*
    More future-like version:
      for every 32-bit word, compute both carry-in 0 and carry-in 1 outputs.
      Then reconcile only 8 word-level carries.
*/
kernel void add256DualFutureKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * 8u;

    uint r0[8];
    uint c0[8];
    uint r1[8];
    uint c1[8];

    for (uint i = 0; i < 8; i++) {
        uint carryOut0 = 0;
        uint carryOut1 = 0;

        r0[i] = add32_split8(A[base + i], B[base + i], 0, carryOut0);
        r1[i] = add32_split8(A[base + i], B[base + i], 1, carryOut1);

        c0[i] = carryOut0;
        c1[i] = carryOut1;
    }

    uint carry = 0;

    for (uint i = 0; i < 8; i++) {
        uint sel = 0u - carry;

        OUT[base + i] = (r0[i] & ~sel) | (r1[i] & sel);
        carry = (c0[i] & ~sel) | (c1[i] & sel);
    }
}
