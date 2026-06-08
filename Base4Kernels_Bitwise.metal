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
    Branchless base-4 transform add.

    Replaces:

        if (sum >= 4 && bd > 0) { ... }

    with:

        doMask = (sum >= 4 && bd > 0) ? 1 : 0

    Then uses doMask directly as a 0/1 arithmetic control value.

    This is usually more GPU-friendly than branchy code because it avoids
    per-lane divergence.
*/
kernel void base4TransformAddBitwiseKernel(
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
        uint ad = (a >> shift) & 3u;
        uint bd = (residualB >> shift) & 3u;

        uint sum = ad + bd + carry;

        uint carryWouldHappen = (sum >= 4u) ? 1u : 0u;
        uint bNonZero = (bd != 0u) ? 1u : 0u;

        uint doMask = carryWouldHappen & bNonZero;
        uint amount = doMask << shift;

        mask |= amount;
        residualB -= amount;

        uint newBd = bd - doMask;
        uint newSum = ad + newBd + carry;

        carry = (newSum >= 4u) ? 1u : 0u;
    }

    maskOut[id] = mask;
    residualOut[id] = residualB;
    out[id] = a + residualB + mask;
}

/*
    Even more compact variant.

    It computes final output the same way but does not write mask/residual.
    This tests whether output bandwidth was part of the slowdown.
*/
kernel void base4TransformAddBitwiseNoDebugKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint residualB = B[id];

    uint mask = 0;
    uint carry = 0;

    for (uint shift = 0; shift < 32; shift += 2) {
        uint ad = (a >> shift) & 3u;
        uint bd = (residualB >> shift) & 3u;

        uint sum = ad + bd + carry;

        uint doMask = ((sum >= 4u) ? 1u : 0u) & ((bd != 0u) ? 1u : 0u);
        uint amount = doMask << shift;

        mask |= amount;
        residualB -= amount;

        uint newBd = bd - doMask;
        carry = ((ad + newBd + carry) >= 4u) ? 1u : 0u;
    }

    out[id] = a + residualB + mask;
}
