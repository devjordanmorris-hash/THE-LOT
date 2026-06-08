import SwiftUI

struct ContentView: View {

    @State private var inputData: Data = Data()
    @State private var compressedSymbols: [WaveSymbol] = []
    @State private var outputData: Data = Data()

    @State private var status: String = "Idle"
    @State private var debugText: String = ""

    var body: some View {
        VStack(spacing: 14) {

            Text("JLO Prism Sine Compression")
                .font(.title2)
                .bold()

            Text(status)
                .font(.caption)
                .foregroundColor(status.contains("OK") ? .green : .secondary)

            VStack(alignment: .leading, spacing: 4) {
                Text("Input bytes: \(inputData.count)")
                Text("Symbols: \(compressedSymbols.count)")
                Text("Output bytes: \(outputData.count)")
            }
            .font(.caption)

            HStack(spacing: 10) {
                Button("Load Test Data") { loadTestData() }
                Button("Compress") { compress() }
                    .disabled(inputData.isEmpty)
                Button("Decompress") { decompress() }
                    .disabled(compressedSymbols.isEmpty)
            }

            Divider()

            ScrollView {
                Text(debugText.isEmpty ? "Debug output will appear here…" : debugText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .background(Color.black.opacity(0.05))
            .cornerRadius(6)

            Spacer()
        }
        .padding()
        .frame(minWidth: 520, minHeight: 420)
    }

    // MARK: - Actions

    func loadTestData() {
        var bytes: [UInt8] = []
        for i in 0..<4096 {
            bytes.append(UInt8((i ^ (i >> 3)) & 0xff))
        }
        inputData = Data(bytes)
        compressedSymbols = []
        outputData = Data()
        debugText = ""
        status = "Loaded test data"
    }

    func compress() {
        status = "Compressing…"
        compressedSymbols = PrismCodec.compress(inputData)
        debugText = "Compressed into \(compressedSymbols.count) wave symbols\n"

        for (i, s) in compressedSymbols.prefix(16).enumerated() {
            debugText += "[\(i)] angle=\(s.angle) run=\(s.runLength)\n"
        }

        status = "Compressed"
    }

    func decompress() {
        status = "Decompressing…"
        outputData = PrismCodec.decompress(compressedSymbols)

        if outputData == inputData {
            status = "Decompression OK ✅"
            debugText += "\nRound-trip verified ✓"
        } else {
            status = "Mismatch ❌"
            debugText += "\n❌ MISMATCH DETECTED\n"
            debugText += diffReport(input: inputData, output: outputData)
            
        }
    }

    // MARK: - Debug helpers

    func diffReport(input: Data, output: Data) -> String {
        let a = [UInt8](input)
        let b = [UInt8](output)

        let count = min(a.count, b.count)

        for i in 0..<count {
            if a[i] != b[i] {
                return """
                First mismatch at byte \(i)
                input : \(hexDump(a, from: i))
                output: \(hexDump(b, from: i))
                """
            }
        }

        if a.count != b.count {
            return "Length mismatch: input=\(a.count) output=\(b.count)"
        }

        return "Unknown mismatch (no byte diff found)"
    }
    

    func hexDump(_ bytes: [UInt8], from index: Int) -> String {
        let start = max(0, index - 8)
        let end = min(bytes.count, index + 8)

        return bytes[start..<end]
            .map { String(format: "%02X", $0) }
            .joined(separator: " ")
    }
    
}

#Preview {
    ContentView()
}
