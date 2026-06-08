#include <metal_stdlib>
using namespace metal;

/*
    256-bit add with carry prediction.

    Each 256-bit value = 8 x uint32 words, least-significant word first.

    Predictor idea:
      For each 32-bit word:
        generate  g = carry-out if carry-in=0
        propagate p = carry-out if carry-in=1 but not if carry-in=0

      Then:
        carry into word0 = 0
        carry into word1 = g0
        carry into word2 = g1 | (p1 & carry1)
        ...

    For normal unsigned addition:
        g = overflow(a + b)
        p = (a + b == 0xffffffff)
      because if sum is all 1s, an incoming carry will propagate out.

    This predicts the word-level carry chain cheaply, then stitches results.
*/

inline uint add32_with_carry(uint a, uint b, uint carryIn, thread uint &carryOut) {
    ulong sum = (ulong)a + (ulong)b + (ulong)carryIn;
    carryOut = (uint)(sum >> 32);
    return (uint)sum;
}

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
    Dual future baseline:
      for each word compute result/carry for carry-in 0 and 1,
      then resolve carry serially.
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
        uint co0 = 0;
        uint co1 = 0;

        r0[i] = add32_with_carry(A[base + i], B[base + i], 0, co0);
        r1[i] = add32_with_carry(A[base + i], B[base + i], 1, co1);

        c0[i] = co0;
        c1[i] = co1;
    }

    uint carry = 0;

    for (uint i = 0; i < 8; i++) {
        uint sel = 0u - carry;

        OUT[base + i] = (r0[i] & ~sel) | (r1[i] & sel);
        carry = (c0[i] & ~sel) | (c1[i] & sel);
    }
}

/*
    Carry predictor version.

    Compute word-level generate/propagate first.

    g[i] = carry out from word i if carry-in is 0.
    p[i] = whether carry-in propagates through word i.

    carryIn[i+1] = g[i] | (p[i] & carryIn[i])

    Since there are only 8 words, this is still written plainly.
    But it avoids computing full dual futures and makes the prediction explicit.
*/
kernel void add256CarryPredictKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * 8u;

    uint s[8];
    uint g[8];
    uint p[8];

    for (uint i = 0; i < 8; i++) {
        uint a = A[base + i];
        uint b = B[base + i];

        uint sum = a + b;

        s[i] = sum;

        // Carry generated without carry-in.
        g[i] = (sum < a) ? 1u : 0u;

        // Carry propagates if sum is all ones.
        p[i] = (sum == 0xffffffffu) ? 1u : 0u;
    }

    uint c0 = 0u;
    uint c1 = g[0] | (p[0] & c0);
    uint c2 = g[1] | (p[1] & c1);
    uint c3 = g[2] | (p[2] & c2);
    uint c4 = g[3] | (p[3] & c3);
    uint c5 = g[4] | (p[4] & c4);
    uint c6 = g[5] | (p[5] & c5);
    uint c7 = g[6] | (p[6] & c6);

    OUT[base + 0] = s[0];
    OUT[base + 1] = s[1] + c1;
    OUT[base + 2] = s[2] + c2;
    OUT[base + 3] = s[3] + c3;
    OUT[base + 4] = s[4] + c4;
    OUT[base + 5] = s[5] + c5;
    OUT[base + 6] = s[6] + c6;
    OUT[base + 7] = s[7] + c7;
}

/*
    Parallel-prefix-ish carry predictor.

    This composes generate/propagate pairs:
        G = g_hi | (p_hi & g_lo)
        P = p_hi & p_lo

    For 8 words, do a small prefix tree.
*/
kernel void add256PrefixPredictKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * 8u;

    uint s0 = A[base+0] + B[base+0];
    uint s1 = A[base+1] + B[base+1];
    uint s2 = A[base+2] + B[base+2];
    uint s3 = A[base+3] + B[base+3];
    uint s4 = A[base+4] + B[base+4];
    uint s5 = A[base+5] + B[base+5];
    uint s6 = A[base+6] + B[base+6];
    uint s7 = A[base+7] + B[base+7];

    uint g0 = (s0 < A[base+0]) ? 1u : 0u;
    uint g1 = (s1 < A[base+1]) ? 1u : 0u;
    uint g2 = (s2 < A[base+2]) ? 1u : 0u;
    uint g3 = (s3 < A[base+3]) ? 1u : 0u;
    uint g4 = (s4 < A[base+4]) ? 1u : 0u;
    uint g5 = (s5 < A[base+5]) ? 1u : 0u;
    uint g6 = (s6 < A[base+6]) ? 1u : 0u;
    uint g7 = (s7 < A[base+7]) ? 1u : 0u;

    uint p0 = (s0 == 0xffffffffu) ? 1u : 0u;
    uint p1 = (s1 == 0xffffffffu) ? 1u : 0u;
    uint p2 = (s2 == 0xffffffffu) ? 1u : 0u;
    uint p3 = (s3 == 0xffffffffu) ? 1u : 0u;
    uint p4 = (s4 == 0xffffffffu) ? 1u : 0u;
    uint p5 = (s5 == 0xffffffffu) ? 1u : 0u;
    uint p6 = (s6 == 0xffffffffu) ? 1u : 0u;
    uint p7 = (s7 == 0xffffffffu) ? 1u : 0u;

    // Prefix distance 1
    uint G1 = g1 | (p1 & g0); uint P1 = p1 & p0;
    uint G2 = g2 | (p2 & g1); uint P2 = p2 & p1;
    uint G3 = g3 | (p3 & g2); uint P3 = p3 & p2;
    uint G4 = g4 | (p4 & g3); uint P4 = p4 & p3;
    uint G5 = g5 | (p5 & g4); uint P5 = p5 & p4;
    uint G6 = g6 | (p6 & g5); uint P6 = p6 & p5;
    uint G7 = g7 | (p7 & g6); uint P7 = p7 & p6;

    // Prefix distance 2
    G2 = G2 | (P2 & g0); P2 = P2 & p0;
    G3 = G3 | (P3 & G1); P3 = P3 & P1;
    G4 = G4 | (P4 & G2); P4 = P4 & P2;
    G5 = G5 | (P5 & G3); P5 = P5 & P3;
    G6 = G6 | (P6 & G4); P6 = P6 & P4;
    G7 = G7 | (P7 & G5); P7 = P7 & P5;

    // Prefix distance 4
    G4 = G4 | (P4 & g0); P4 = P4 & p0;
    G5 = G5 | (P5 & G1); P5 = P5 & P1;
    G6 = G6 | (P6 & G2); P6 = P6 & P2;
    G7 = G7 | (P7 & G3); P7 = P7 & P3;

    // Carry into word i is carry out of previous prefix.
    OUT[base+0] = s0;
    OUT[base+1] = s1 + g0;
    OUT[base+2] = s2 + G1;
    OUT[base+3] = s3 + G2;
    OUT[base+4] = s4 + G3;
    OUT[base+5] = s5 + G4;
    OUT[base+6] = s6 + G5;
    OUT[base+7] = s7 + G6;
}
