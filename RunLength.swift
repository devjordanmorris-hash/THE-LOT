import Foundation

enum RunLength {

    static func encode(_ bits: [UInt8]) -> [BitRun] {
        guard !bits.isEmpty else { return [] }

        var runs: [BitRun] = []
        var current = bits[0]
        var length = 1

        for b in bits.dropFirst() {
            if b == current {
                length += 1
            } else {
                runs.append(BitRun(bit: current, length: length))
                current = b
                length = 1
            }
        }

        runs.append(BitRun(bit: current, length: length))
        return runs
    }

    static func decode(_ runs: [BitRun]) -> [UInt8] {
        runs.flatMap { repeatElement($0.bit, count: $0.length) }
    }
}
