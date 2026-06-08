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
    Gate-style 2-bit base-4 carry preconditioner.

    This replaces:
        sum = ad + bd + carry
        if sum >= 4 ...
        bd -= 1

    with:
        2-bit full-adder carry logic
        bitwise nonzero test
        bitwise conditional decrement

    Still serial across base-4 chunks because carry-in depends on previous chunk.
*/
kernel void base4GateTransformAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device uint *maskOut [[buffer(3)]],
    device uint *residualOut [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint residualB = B[id];

    uint mask = 0;
    uint carry = 0;

    for (uint shift = 0; shift < 32; shift += 2) {
        uint a0 = (a >> shift) & 1u;
        uint a1 = (a >> (shift + 1u)) & 1u;

        uint b0 = (residualB >> shift) & 1u;
        uint b1 = (residualB >> (shift + 1u)) & 1u;

        // 2-bit full-adder carry out for a_digit + b_digit + carry.
        uint c1 = (a0 & b0) | (carry & (a0 ^ b0));
        uint c2 = (a1 & b1) | (c1 & (a1 ^ b1));

        uint bNonZero = b0 | b1;
        uint doMask = c2 & bNonZero;

        uint amount = doMask << shift;
        mask |= amount;

        /*
            Conditional 2-bit decrement of b when doMask=1.

            01 -> 00
            10 -> 01
            11 -> 10

            dec_b0 = ~b0
            dec_b1 = b1 ^ b0

            Then select original b or decremented b using doMask.
        */
        uint dec_b0 = b0 ^ 1u;
        uint dec_b1 = b1 ^ b0;

        uint new_b0 = (b0 & (doMask ^ 1u)) | (dec_b0 & doMask);
        uint new_b1 = (b1 & (doMask ^ 1u)) | (dec_b1 & doMask);

        // Clear old 2-bit digit and write new one.
        residualB &= ~(3u << shift);
        residualB |= (new_b0 << shift) | (new_b1 << (shift + 1u));

        /*
            Recompute carry after decrement using gate logic:
            a_digit + new_b_digit + carry
        */
        uint nc1 = (a0 & new_b0) | (carry & (a0 ^ new_b0));
        uint nc2 = (a1 & new_b1) | (nc1 & (a1 ^ new_b1));

        carry = nc2;
    }

    maskOut[id] = mask;
    residualOut[id] = residualB;
    out[id] = a + residualB + mask;
}
