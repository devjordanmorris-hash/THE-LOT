import Foundation

// MARK: - Wave symbols

public typealias AngleID = Int

public struct WaveSymbol: Hashable {
    public let angle: AngleID
    public let runLength: Int
}

// MARK: - Bit runs

public struct BitRun {
    public let bit: UInt8   // 0 or 1
    public let length: Int
}

// MARK: - Geometry

public struct Triangle {
    public let runs: [BitRun]   // exactly 3
}

public struct Prism {
    public let id: Int          // 0–7
    public let runLength: Int
}
