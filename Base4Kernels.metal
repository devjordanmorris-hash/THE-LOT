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

kernel void base4TransformAddKernel(
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

        if (sum >= 4u && bd > 0u) {
            uint amount = 1u << shift;

            mask |= amount;
            residualB -= amount;

            bd -= 1u;
            sum = ad + bd + carry;
        }

        carry = (sum >= 4u) ? 1u : 0u;
    }

    maskOut[id] = mask;
    residualOut[id] = residualB;
    out[id] = a + residualB + mask;
}
