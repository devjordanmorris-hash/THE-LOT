#include <metal_stdlib>
using namespace metal;

#define WIDTH 16u

/*
    Base-4 parallel carry-wave adder using one threadgroup per 32-bit add.

    Each 32-bit number has 16 base-4 digits.
    Each digit gets one thread/lane inside the threadgroup.

    Threadgroup layout:
      threadgroup_position_in_grid.x = which add
      thread_index_in_threadgroup.x  = base-4 digit lane 0..15

    This is the hardware-shaped version of the CPU carry-wave test:
      - digit sums happen in parallel
      - carries are shifted to the next digit in parallel
      - repeat carry waves until no new carry is produced
      - lane 0 recombines final digits
*/

kernel void nativeAddKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    OUT[id] = A[id] + B[id];
}

/*
    One-thread-per-number serial base-4 wave GPU baseline.
    Useful to compare against the cooperative 16-lane version.
*/
kernel void base4WaveSerialKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    uint a = A[id];
    uint b = B[id];

    uint partial = 0u;
    uint carryStream = 0u;

    for (uint pos = 0u; pos < WIDTH; pos++) {
        uint shift = pos * 2u;
        uint local = ((a >> shift) & 3u) + ((b >> shift) & 3u);

        partial |= (local & 3u) << shift;

        if (local >= 4u && pos < WIDTH - 1u) {
            carryStream |= 1u << ((pos + 1u) * 2u);
        }
    }

    for (uint wave = 0u; wave < WIDTH && carryStream != 0u; wave++) {
        uint nextPartial = 0u;
        uint nextCarry = 0u;

        for (uint pos = 0u; pos < WIDTH; pos++) {
            uint shift = pos * 2u;
            uint local = ((partial >> shift) & 3u) + ((carryStream >> shift) & 3u);

            nextPartial |= (local & 3u) << shift;

            if (local >= 4u && pos < WIDTH - 1u) {
                nextCarry |= 1u << ((pos + 1u) * 2u);
            }
        }

        partial = nextPartial;
        carryStream = nextCarry;
    }

    OUT[id] = partial;
}

/*
    Cooperative lane version.
*/
kernel void base4WaveLaneKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint groupID [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (groupID >= N || lane >= WIDTH) return;

    threadgroup uint partial[WIDTH];
    threadgroup uint carry[WIDTH];
    threadgroup uint nextCarry[WIDTH];
    threadgroup uint activeFlags[WIDTH];
    threadgroup uint parts[WIDTH];
    threadgroup uint active;

    uint a = A[groupID];
    uint b = B[groupID];

    uint shift = lane * 2u;
    uint local = ((a >> shift) & 3u) + ((b >> shift) & 3u);

    partial[lane] = local & 3u;
    carry[lane] = 0u;
    nextCarry[lane] = 0u;
    activeFlags[lane] = 0u;
    parts[lane] = 0u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local >= 4u && lane < WIDTH - 1u) {
        carry[lane + 1u] = 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wave = 0u; wave < WIDTH; wave++) {
        uint c = carry[lane];

        local = partial[lane] + c;
        partial[lane] = local & 3u;

        nextCarry[lane] = 0u;
        activeFlags[lane] = 0u;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local >= 4u && lane < WIDTH - 1u) {
            nextCarry[lane + 1u] = 1u;
            activeFlags[lane] = 1u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            uint any = 0u;
            for (uint i = 0u; i < WIDTH; i++) {
                any |= activeFlags[i];
            }
            active = any;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        carry[lane] = nextCarry[lane];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active == 0u) {
            break;
        }
    }

    parts[lane] = partial[lane] << shift;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint result = 0u;

        for (uint i = 0u; i < WIDTH; i++) {
            result |= parts[i];
        }

        OUT[groupID] = result;
    }
}

/*
    Fixed-wave version:
      no early break / no active reduction.
      This is sometimes faster on GPU because all lanes execute uniform control flow.
*/
kernel void base4WaveLaneFixedKernel(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint groupID [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (groupID >= N || lane >= WIDTH) return;

    threadgroup uint partial[WIDTH];
    threadgroup uint carry[WIDTH];
    threadgroup uint nextCarry[WIDTH];
    threadgroup uint parts[WIDTH];

    uint a = A[groupID];
    uint b = B[groupID];

    uint shift = lane * 2u;
    uint local = ((a >> shift) & 3u) + ((b >> shift) & 3u);

    partial[lane] = local & 3u;
    carry[lane] = 0u;
    nextCarry[lane] = 0u;
    parts[lane] = 0u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local >= 4u && lane < WIDTH - 1u) {
        carry[lane + 1u] = 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wave = 0u; wave < WIDTH; wave++) {
        uint c = carry[lane];

        local = partial[lane] + c;
        partial[lane] = local & 3u;

        nextCarry[lane] = 0u;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local >= 4u && lane < WIDTH - 1u) {
            nextCarry[lane + 1u] = 1u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        carry[lane] = nextCarry[lane];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    parts[lane] = partial[lane] << shift;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint result = 0u;

        for (uint i = 0u; i < WIDTH; i++) {
            result |= parts[i];
        }

        OUT[groupID] = result;
    }
}
