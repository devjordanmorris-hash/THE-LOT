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
    Direct byte predictor.

    This is the clean arithmetic version:
      t = aByte + bByte
      g = t >= 256
      p = low(t) == 255

    Then resolve carries across four bytes.
*/
kernel void bytePredict32AddKernel(
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

    uint t0 = a0 + b0;
    uint t1 = a1 + b1;
    uint t2 = a2 + b2;
    uint t3 = a3 + b3;

    uint s0 = t0 & 0xffu;
    uint s1 = t1 & 0xffu;
    uint s2 = t2 & 0xffu;
    uint s3 = t3 & 0xffu;

    uint g0 = (t0 >> 8) & 1u;
    uint g1 = (t1 >> 8) & 1u;
    uint g2 = (t2 >> 8) & 1u;

    uint p1 = (s1 == 0xffu) ? 1u : 0u;
    uint p2 = (s2 == 0xffu) ? 1u : 0u;

    uint c1 = g0;
    uint c2 = g1 | (p1 & c1);
    uint c3 = g2 | (p2 & c2);

    uint r0 = s0;
    uint r1 = (s1 + c1) & 0xffu;
    uint r2 = (s2 + c2) & 0xffu;
    uint r3 = (s3 + c3) & 0xffu;

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

/*
    XOR-memory predictor.

    For each byte:
      x = aByte ^ bByte      // no-carry sum shape
      y = aByte & bByte      // carry seed shape

    We look up whether the byte propagates an incoming carry from x:
      propagate = x == 0xff

    For generate we still use carry seed:
      generate = carry out from a+b without carry-in.

    A pure lookup of generate would need a,b or a larger table.
    This version tests whether using a tiny xor-shape table helps
    at all versus direct comparisons.

    xorTable[x] has:
      bit 0 = propagate, i.e. x == 0xff
*/
kernel void byteXorTablePredict32AddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device const uint *xorTable [[buffer(3)]],
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

    uint x0 = a0 ^ b0;
    uint x1 = a1 ^ b1;
    uint x2 = a2 ^ b2;
    uint x3 = a3 ^ b3;

    uint y0 = a0 & b0;
    uint y1 = a1 & b1;
    uint y2 = a2 & b2;

    /*
        Compute byte addition using XOR/AND carry inside byte.
        This is one-step carry prediction using normal addition equivalence:
          t = a + b
        but keeps x/y exposed for comparing the table approach.
    */
    uint t0 = a0 + b0;
    uint t1 = a1 + b1;
    uint t2 = a2 + b2;
    uint t3 = a3 + b3;

    uint s0 = t0 & 0xffu;
    uint s1 = t1 & 0xffu;
    uint s2 = t2 & 0xffu;
    uint s3 = t3 & 0xffu;

    uint g0 = (t0 >> 8) & 1u;
    uint g1 = (t1 >> 8) & 1u;
    uint g2 = (t2 >> 8) & 1u;

    uint p1 = xorTable[x1] & 1u;
    uint p2 = xorTable[x2] & 1u;

    uint c1 = g0;
    uint c2 = g1 | (p1 & c1);
    uint c3 = g2 | (p2 & c2);

    uint r0 = s0;
    uint r1 = (s1 + c1) & 0xffu;
    uint r2 = (s2 + c2) & 0xffu;
    uint r3 = (s3 + c3) & 0xffu;

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);

    // Prevent overly aggressive dead-code elimination of exposed xor/and shapes.
    // These do not change output but keep the kernel structurally honest.
    (void)x0; (void)x3; (void)y0; (void)y1; (void)y2;
}

/*
    Full 256-entry byte pair table.

    This is NOT the huge LUT from earlier. It is a tiny 256-entry table
    indexed by:
       x = aByte ^ bByte

    That cannot fully determine generate, only propagate.
    For a fully saved byte-pair table you need 65536 entries.
    So this kernel is the realistic small-XOR-memory variant.
*/

/*
    Bitwise generate/propagate predictor using classic carry logic.

    For each byte:
      x = a ^ b
      y = a & b

    carry out of byte with carry-in 0 can be computed by propagating
    carry inside the byte:
      carry = y
      carry spreads through x.

    This is a SWAR-ish local byte carry closure:
      c = y | (x & (y << 1)) | ...
    The carry out of bit 7 tells byte generate.
      g = bit8 of propagated carry.

    For byte propagate:
      p = x == 0xff
*/
inline uint byte_generate_from_xor_and(uint x, uint y) {
    uint c = y;
    c |= x & (c << 1);
    c |= x & (c << 2);
    c |= x & (c << 4);
    return (c >> 7) & 1u;
}

kernel void byteXorAndPredict32AddKernel(
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

    uint x0 = a0 ^ b0;
    uint x1 = a1 ^ b1;
    uint x2 = a2 ^ b2;
    uint x3 = a3 ^ b3;

    uint y0 = a0 & b0;
    uint y1 = a1 & b1;
    uint y2 = a2 & b2;

    uint g0 = byte_generate_from_xor_and(x0, y0);
    uint g1 = byte_generate_from_xor_and(x1, y1);
    uint g2 = byte_generate_from_xor_and(x2, y2);

    uint p1 = (x1 == 0xffu) ? 1u : 0u;
    uint p2 = (x2 == 0xffu) ? 1u : 0u;

    uint c1 = g0;
    uint c2 = g1 | (p1 & c1);
    uint c3 = g2 | (p2 & c2);

    uint r0 = x0;
    uint r1 = x1 ^ c1;
    uint r2 = x2 ^ c2;
    uint r3 = x3 ^ c3;

    /*
        Wait: x ^ carry-in is not enough to reconstruct byte sum after
        internal carries. So use byte add for result bytes while using
        xor/and for carry prediction.
    */
    uint s0 = (a0 + b0) & 0xffu;
    uint s1 = (a1 + b1 + c1) & 0xffu;
    uint s2 = (a2 + b2 + c2) & 0xffu;
    uint s3 = (a3 + b3 + c3) & 0xffu;

    out[id] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

    (void)r0; (void)r1; (void)r2; (void)r3;
}
