#include <metal_stdlib>
using namespace metal;

struct Params {
    uint count;
};

// Product square:
// index = aByte * 256 + bByte
// value = 16-bit product
inline ushort mul_lut_lookup(const device ushort *lut, ushort aByte, ushort bByte) {
    uint idx = uint(aByte) * 256u + uint(bByte);
    return lut[idx];
}

// Reduction cube:
// index = ((currentByte * 256 + incomingByte) * 4 + carryIn)
// packed:
//   low 8 bits = output byte
//   bits 8..9  = carry out (0...2)
inline ushort reduce_cube_lookup(const device ushort *cube,
                                 ushort currentByte,
                                 ushort incomingByte,
                                 ushort carryIn) {
    uint idx = ((uint(currentByte) * 256u + uint(incomingByte)) * 4u + uint(carryIn));
    return cube[idx];
}

inline void add_byte_via_cube(thread uchar accum[16],
                              ushort startSlot,
                              ushort incomingByte,
                              const device ushort *cube) {
    ushort slot = startSlot;
    ushort value = incomingByte;
    ushort carry = 0;

    while (slot < 16u && (value != 0u || carry != 0u)) {
        ushort packed = reduce_cube_lookup(cube, ushort(accum[slot]), value, carry);
        accum[slot] = uchar(packed & 0xFFu);

        // After the first step, only carry continues upward.
        value = 0u;
        carry = (packed >> 8u) & 0x3u;
        slot += 1u;
    }
}

kernel void byteLUTCubeMultiplyKernel(
    const device ulong *inA        [[buffer(0)]],
    const device ulong *inB        [[buffer(1)]],
    device ulong *outC             [[buffer(2)]],
    const device ushort *mulLUT    [[buffer(3)]],
    const device ushort *cubeLUT   [[buffer(4)]],
    constant Params &params        [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;

    ulong a = inA[gid];
    ulong b = inB[gid];

    uchar accum[16];
    for (uint i = 0u; i < 16u; ++i) {
        accum[i] = 0u;
    }

    // 8x8 byte partial products
    for (uint i = 0u; i < 8u; ++i) {
        ushort aByte = ushort((a >> (i * 8u)) & 0xFFul);
        if (aByte == 0u) continue;

        for (uint j = 0u; j < 8u; ++j) {
            ushort bByte = ushort((b >> (j * 8u)) & 0xFFul);
            if (bByte == 0u) continue;

            ushort prod = mul_lut_lookup(mulLUT, aByte, bByte);
            ushort lo = prod & 0xFFu;
            ushort hi = (prod >> 8u) & 0xFFu;

            add_byte_via_cube(accum, ushort(i + j), lo, cubeLUT);
            if (i + j + 1u < 16u) {
                add_byte_via_cube(accum, ushort(i + j + 1u), hi, cubeLUT);
            }
        }
    }

    // Pack low 64 bits only
    ulong result = 0ul;
    for (uint i = 0u; i < 8u; ++i) {
        result |= (ulong(accum[i]) << (i * 8u));
    }

    outC[gid] = result;
}
