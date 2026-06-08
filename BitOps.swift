import Foundation

enum BitOps {

    static func bytesToBits(_ data: Data) -> [UInt8] {
        var bits: [UInt8] = []
        for byte in data {
            for i in (0..<8).reversed() {
                bits.append((byte >> i) & 1)
            }
        }
        return bits
    }

    static func bitsToBytes(_ bits: [UInt8]) -> Data {
        var bytes: [UInt8] = []
        var current: UInt8 = 0
        var count = 0

        for bit in bits {
            current = (current << 1) | bit
            count += 1
            if count == 8 {
                bytes.append(current)
                current = 0
                count = 0
            }
        }
        return Data(bytes)
    }
}
