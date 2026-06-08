import Foundation

enum TriangleDetect {

    static func detect(from runs: [BitRun]) -> [Triangle] {
        var triangles: [Triangle] = []
        var i = 0

        while i + 2 < runs.count {
            triangles.append(
                Triangle(runs: [
                    runs[i],
                    runs[i + 1],
                    runs[i + 2]
                ])
            )
            i += 3
        }

        return triangles
    }
}
