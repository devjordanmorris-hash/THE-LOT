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

kernel void base4TransformAddLUTKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *out [[buffer(2)]],
    device uint *maskOut [[buffer(3)]],
    device uint *residualOut [[buffer(4)]],
    device const uint *lut [[buffer(5)]],
    constant uint &N [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint mask = 0;
    uint residual = 0;
    uint carry = 0;

    // 4 chunks: each byte contains 4 base-4 digits.
    for (uint shift = 0; shift < 32; shift += 8) {
        uint aByte = (a >> shift) & 0xffu;
        uint bByte = (b >> shift) & 0xffu;

        uint idx = (carry << 16) | (aByte << 8) | bByte;
        uint entry = lut[idx];

        uint m = entry & 0xffu;
        uint r = (entry >> 8) & 0xffu;

        carry = (entry >> 16) & 1u;

        mask |= m << shift;
        residual |= r << shift;
    }

    maskOut[id] = mask;
    residualOut[id] = residual;
    out[id] = a + residual + mask;
}
