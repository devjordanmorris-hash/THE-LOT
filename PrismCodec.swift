import Foundation

enum PrismCodec {

    static func compress(_ data: Data) -> [WaveSymbol] {

        let bits = BitOps.bytesToBits(data)
        let aligned = Rotation.align(bits)
        let runs = RunLength.encode(aligned)
        let triangles = TriangleDetect.detect(from: runs)
        let prisms = PrismDetect.detect(from: triangles)

        return WaveAssembler.buildFullWave(from: prisms)
    }

    static func decompress(_ wave: [WaveSymbol]) -> Data {

        let runs = WaveDecoder.decode(wave)
        let bits = RunLength.decode(runs)
        let restored = Rotation.restore(bits)

        return BitOps.bitsToBytes(restored)
    }
}
