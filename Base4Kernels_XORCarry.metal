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
    Classic bitwise addition:
      sum   = a ^ b
      carry = (a & b) << 1
      repeat until carry == 0

    Also emits:
      carryField = OR of every carry value generated during propagation.
      iterations = number of propagation rounds.
*/
kernel void xorAndCarryKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device uint *carryFieldOut [[buffer(3)]],
    device uint *itersOut [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint carryField = 0;
    uint iters = 0;

    // Fixed upper bound for uint32 carry propagation.
    for (uint i = 0; i < 32; i++) {
        uint carry = (a & b) << 1;
        uint sum = a ^ b;

        carryField |= carry;
        iters += (carry != 0u) ? 1u : 0u;

        a = sum;
        b = carry;

        if (carry == 0u) {
            break;
        }
    }

    out[id] = a;
    carryFieldOut[id] = carryField;
    itersOut[id] = iters;
}

/*
    Base-4 carry preconditioner from earlier.
    Emits:
      transformed result
      base4 mask
*/
kernel void base4TransformAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device uint *maskOut [[buffer(3)]],
    constant uint &N [[buffer(4)]],
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
    maskOut[id] = mask;
}
