//
//  bitwiseDerivativeKernel.metal
//  symbolic algebra
//
//  Created by Jordan Morris  on 01/11/2025.
//

#include <metal_stdlib>
using namespace metal;

// bitwiseDerivativeKernel:
// Reversible bitwise derivative operation.
// For each 64-bit input, computes the XOR of the word with a rotated version,
// detecting bit transitions. The operation is reversible.

kernel void bitwiseDerivativeKernel(
    const device uint64_t* inA [[buffer(0)]],
    device uint64_t* out [[buffer(1)]],
    constant uint &shift [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    uint64_t x = inA[id];
    uint64_t rot = (x << shift) | (x >> (64 - shift));
    uint64_t deriv = x ^ rot;
    out[id] = deriv;
}
