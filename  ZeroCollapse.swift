//
//  ZeroCollapse.swift
//  JLO Prism Sine Compression
//
//  Collapses zeros while recording their original positions
//

import Foundation

// MARK: - Collapse Result

struct ZeroCollapseResult {
    let collapsed: UInt64      // all 1s packed
    let zeroMask: UInt64       // 1 = was zero in original
    let bitCount: Int          // number of valid bits
    let onesCount: Int
}

// MARK: - Collapse

/// Collapse zeros out of a bitstream while recording positions
func collapseZeros(
    value: UInt64,
    bits: Int = 64
) -> ZeroCollapseResult {

    var collapsed: UInt64 = 0
    var zeroMask: UInt64 = 0

    var writeIndex = 0
    var onesCount = 0

    for readIndex in 0..<bits {
        let bit = (value >> readIndex) & 1

        if bit == 1 {
            // pack ones left-to-right
            collapsed |= (1 << writeIndex)
            writeIndex += 1
            onesCount += 1
        } else {
            // record zero position
            zeroMask |= (1 << readIndex)
        }
    }

    return ZeroCollapseResult(
        collapsed: collapsed,
        zeroMask: zeroMask,
        bitCount: bits,
        onesCount: onesCount
    )
}
