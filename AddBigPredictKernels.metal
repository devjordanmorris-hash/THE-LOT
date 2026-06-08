#include <metal_stdlib>
using namespace metal;

#define WORDS 32u   // 32 x uint32 = 1024-bit integers

/*
    1024-bit add benchmark.

    Number layout:
      word 0 = least significant 32 bits
      word 31 = most significant 32 bits

    Kernels:
      1. native sequential word-carry add
      2. carry-predict serial recurrence
      3. carry-predict parallel-prefix style over 32 words

    This tests whether carry prediction becomes more relevant as width grows.
*/

kernel void addBigNativeWordKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * WORDS;
    ulong carry = 0;

    for (uint i = 0; i < WORDS; i++) {
        ulong sum = (ulong)A[base + i] + (ulong)B[base + i] + carry;
        OUT[base + i] = (uint)sum;
        carry = sum >> 32;
    }
}

/*
    Carry predictor:
      s[i] = A[i] + B[i]
      g[i] = carry generated without carry-in
      p[i] = carry propagates if carry-in arrives

    For uint32 word:
      g = s < A
      p = s == 0xffffffff
*/
kernel void addBigCarryPredictKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * WORDS;

    uint s[WORDS];
    uint g[WORDS];
    uint p[WORDS];

    for (uint i = 0; i < WORDS; i++) {
        uint a = A[base + i];
        uint b = B[base + i];

        uint sum = a + b;

        s[i] = sum;
        g[i] = (sum < a) ? 1u : 0u;
        p[i] = (sum == 0xffffffffu) ? 1u : 0u;
    }

    uint carry = 0;

    for (uint i = 0; i < WORDS; i++) {
        OUT[base + i] = s[i] + carry;
        carry = g[i] | (p[i] & carry);
    }
}

/*
    Prefix predictor:
      Compose generate/propagate pairs.

      Pair composition:
        G = G_hi | (P_hi & G_lo)
        P = P_hi & P_lo

    After prefix:
      G[i] is carry-out of block 0..i assuming external carry-in 0.
      carry into word i = G[i-1].
*/
kernel void addBigPrefixPredictKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint base = id * WORDS;

    uint s[WORDS];
    uint G[WORDS];
    uint P[WORDS];

    for (uint i = 0; i < WORDS; i++) {
        uint a = A[base + i];
        uint b = B[base + i];

        uint sum = a + b;

        s[i] = sum;
        G[i] = (sum < a) ? 1u : 0u;
        P[i] = (sum == 0xffffffffu) ? 1u : 0u;
    }

    /*
        Inclusive prefix over generate/propagate.
        For 32 words: distances 1,2,4,8,16.
    */
    for (uint d = 1; d < WORDS; d <<= 1) {
        uint oldG[WORDS];
        uint oldP[WORDS];

        for (uint i = 0; i < WORDS; i++) {
            oldG[i] = G[i];
            oldP[i] = P[i];
        }

        for (uint i = 0; i < WORDS; i++) {
            if (i >= d) {
                G[i] = oldG[i] | (oldP[i] & oldG[i - d]);
                P[i] = oldP[i] & oldP[i - d];
            }
        }
    }

    OUT[base + 0] = s[0];

    for (uint i = 1; i < WORDS; i++) {
        OUT[base + i] = s[i] + G[i - 1];
    }
}
