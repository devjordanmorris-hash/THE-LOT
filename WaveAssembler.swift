import Foundation

let prismToAngle: [Int: AngleID] = [
    0: 10, 1: 20, 2: 30, 3: 40,
    4: 50, 5: 60, 6: 70, 7: 80
]

enum WaveAssembler {

    static func assembleHalf(from prisms: [Prism]) -> [WaveSymbol] {
        var symbols: [WaveSymbol] = []
        var currentAngle: AngleID?
        var run = 0

        for p in prisms {
            guard let angle = prismToAngle[p.id] else { continue }

            if angle == currentAngle {
                run += p.runLength
            } else {
                if let a = currentAngle {
                    symbols.append(WaveSymbol(angle: a, runLength: run))
                }
                currentAngle = angle
                run = p.runLength
            }
        }

        if let a = currentAngle {
            symbols.append(WaveSymbol(angle: a, runLength: run))
        }

        return symbols
    }

    static func buildFullWave(from prisms: [Prism]) -> [WaveSymbol] {
        let half = assembleHalf(from: prisms)
        let mirror = half.dropLast().reversed()
        return half + mirror
    }
}
