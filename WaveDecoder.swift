import Foundation

enum WaveDecoder {

    static func decode(_ wave: [WaveSymbol]) -> [BitRun] {
        wave.flatMap {
            [
                BitRun(bit: 1, length: $0.runLength),
                BitRun(bit: 0, length: $0.runLength)
            ]
        }
    }
}
