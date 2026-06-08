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
    32-bit byte-level carry predictor.

    Split into 4 x 8-bit chunks.

    For each byte:
      s = aByte + bByte
      g = carry generated if carry-in = 0
      p = carry propagates if carry-in = 1

    For byte addition:
      g = s >= 256
      p = (s & 0xff) == 0xff

    Then:
      carry into byte1 = g0
      carry into byte2 = g1 | (p1 & carry1)
      carry into byte3 = g2 | (p2 & carry2)

    Finally:
      result byte i = low_sum_i + carry_in_i
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

    uint p0 = (s0 == 0xffu) ? 1u : 0u;
    uint p1 = (s1 == 0xffu) ? 1u : 0u;
    uint p2 = (s2 == 0xffu) ? 1u : 0u;

    // carry into byte0 = 0
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
    Prefix-ish version for 4 bytes.
    It predicts carry boundaries with a tiny prefix network.
*/
kernel void bytePrefix32AddKernel(
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

    uint p0 = (s0 == 0xffu) ? 1u : 0u;
    uint p1 = (s1 == 0xffu) ? 1u : 0u;
    uint p2 = (s2 == 0xffu) ? 1u : 0u;

    // prefix compose for carries:
    // G10 = carry out of byte1 considering byte0
    uint G1 = g1 | (p1 & g0);
    uint P1 = p1 & p0;

    // G2 after byte2 considering bytes 0..2
    uint G2_pair = g2 | (p2 & g1);
    uint P2_pair = p2 & p1;
    uint G2 = G2_pair | (P2_pair & g0);

    uint c1 = g0;
    uint c2 = G1;
    uint c3 = G2;

    uint r0 = s0;
    uint r1 = (s1 + c1) & 0xffu;
    uint r2 = (s2 + c2) & 0xffu;
    uint r3 = (s3 + c3) & 0xffu;

    out[id] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}
