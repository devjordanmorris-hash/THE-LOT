import Foundation

enum PrismDetect {

    static func detect(from triangles: [Triangle]) -> [Prism] {
        triangles.enumerated().map { index, tri in
            Prism(
                id: index % 8,
                runLength: tri.runs.map(\.length).min() ?? 0
            )
        }
    }
}
