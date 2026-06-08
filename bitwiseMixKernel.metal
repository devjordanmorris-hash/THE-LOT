//
//  bitwiseMixKernel.metal
//  symbolic algebra
//
//  Created by Jordan Morris on 01/11/2025.
//
//  This kernel performs a rotate+XOR operation for each thread. The rotation offset is configurable
//  via a constant buffer from the CPU, enabling benchmarking and algorithm tuning.
//

#include <metal_stdlib>
using namespace metal;

// Each GPU thread performs one rotate+XOR operation
kernel void bitwiseMixKernel(const device uint64_t* inA        [[buffer(0)]],
                             const device uint64_t* inB        [[buffer(1)]],
                             device uint64_t* out              [[buffer(2)]],
                             constant uint &rotation           [[buffer(3)]],
                             uint id                           [[thread_position_in_grid]])
{
    uint64_t x = inA[id];
    uint64_t y = inB[id];

    // Rotate left by a dynamic offset from CPU
    uint64_t rot = (x << rotation) | (x >> (64 - rotation));

    // Perform XOR mix
    uint64_t mixed = rot ^ y;

    // Write result
    out[id] = mixed;
}
