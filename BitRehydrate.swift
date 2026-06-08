//
//  BitRehydrate.swift
//  JLO Prism Sine Compression
//
//  v1.0 – Minimal rehydration
//  Goal: symbol stream → samples (loss-aware, deterministic)
//

import Foundation

// MARK: - Rehydration

struct BitRehydrate {

    /// Rebuilds a waveform from symbolic angles + run lengths
    /// No optimisation, no noise, no tricks
    static func rebuildWave(
        from symbols: [WaveSymbol],
        amplitude: Float = 1.0
    ) -> [Float] {

        var output: [Float] = []
        output.reserveCapacity(
            symbols.reduce(0) { $0 + $1.runLength }
        )

        var phase: Float = 0.0

        for symbol in symbols {

            // angle stored as degrees / 10
            let degrees = Float(symbol.angle)
            let radians = degrees * (.pi / 180.0)

            for _ in 0..<symbol.runLength {
                let value = amplitude * sin(phase)
                output.append(value)
                phase += radians
            }
        }

        return output
    }

    /// Byte-safe comparison helper (debug only)
    static func maxError(
        original: [Float],
        reconstructed: [Float]
    ) -> Float {

        guard original.count == reconstructed.count else {
            return .infinity
        }

        var maxErr: Float = 0

        for i in 0..<original.count {
            maxErr = max(
                maxErr,
                abs(original[i] - reconstructed[i])
            )
        }

        return maxErr
    }
}
