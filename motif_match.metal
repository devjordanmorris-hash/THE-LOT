#include <metal_stdlib>
using namespace metal;

struct Motif {
    uint offset;   // offset into flat bit-patterns
    ushort len;    // motif length in bits
    ushort nrot;   // number of rotations (contiguous in patterns)
};

struct Params {
    uint nBits;        // total bit length of input
    uint nMotifs;      // number of base motifs
};

kernel void motifLongestMatch(
    device const uchar*         inBytes      [[buffer(0)]],   // input bytes
    device const Motif*         motifs       [[buffer(1)]],   // base motifs table
    device const uchar*         patBits      [[buffer(2)]],   // concatenated rotation bit-patterns (packed as bytes, MSB-first within each byte)
    device const ushort*        patOffsets   [[buffer(3)]],   // per rotation: offset (in bits) into patBits
    device const ushort*        patLengths   [[buffer(4)]],   // per rotation: length in bits
    constant Params&            P            [[buffer(5)]],
    device ushort*              outMotifId   [[buffer(6)]],   // best motif id per bit
    device ushort*              outRotIdx    [[buffer(7)]],   // best rotation index per bit (global index in pat arrays)
    device ushort*              outBestLen   [[buffer(8)]],   // best matched length per bit
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= P.nBits) return;

    // helper: get bit at absolute bit index (0..P.nBits-1)
    auto getbit = [&](uint bitIdx) -> uchar {
        if (bitIdx >= P.nBits) return 255; // "out of range" sentinel
        uint byteIdx = bitIdx >> 3;
        uint bitIn   = bitIdx & 7;
        uchar b = inBytes[byteIdx];
        // Use MSB-first for comparison consistency
        return (b >> (7-bitIn)) & 1;
    };

    ushort bestLen = 0;
    ushort bestM   = 0xFFFF;
    ushort bestR   = 0;

    // Iterate all rotations of all motifs (kept small — fine on GPU)
    uint totalRots = 0;
    for (uint m = 0; m < P.nMotifs; ++m) totalRots += motifs[m].nrot;

    for (uint r = 0; r < totalRots; ++r) {
        ushort L = patLengths[r];
        if (L == 0) continue;
        // quick bound
        if (gid + L > P.nBits) continue;

        // Compare bit-by-bit (motifs are small; branch-free compare is ok)
        ushort matched = 0;
        uint patStart = patOffsets[r]; // in bits into patBits

        for (ushort k = 0; k < L; ++k) {
            // motif bit:
            uint pb = patStart + k;
            uchar motifByte = patBits[pb >> 3];
            uchar motifBit = (motifByte >> (7 - (pb & 7))) & 1;

            if (motifBit != getbit(gid + k)) {
                break;
            }
            matched++;
        }

        if (matched > bestLen) {
            bestLen = matched;
            bestR   = r;
            // map rotation r to its base motif id:
            // Cheap approach: store a parallel array rotToMotifId on device. (see host code)
            // For now assume patLengths[r] = 0 means unused; we fill rotToMotifId in buffer(9).
        }
    }

    // Write result
    outBestLen[gid] = bestLen;
    // We'll need rotToMotifId; here we assume it's packed after patLengths in buffer(4),
    // but to keep interfaces clean, pass it separately:
    // For clarity in this skeleton, leave motif id = 0; host can infer using a rot->motif map.
    outRotIdx[gid]  = bestR;
    outMotifId[gid] = 0; // to be filled on CPU by mapping rotIdx -> motifId (or pass rotToMotifId as buffer(9))
}