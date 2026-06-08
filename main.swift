import Foundation
import Metal

func makeDevice() -> MTLDevice {
    guard let d = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
    return d
}
func makeLibrary(device: MTLDevice, source: String) -> MTLLibrary { try! device.makeLibrary(source: source, options: nil) }
func readFile(_ path: String) -> String { String(data: try! Data(contentsOf: URL(fileURLWithPath: path)), encoding: .utf8)! }

let N = 1_000_000
let useU64 = true

// ---- Paste YOUR log transform here (CPU version) ----
@inline(__always)
func bitwise_log_u64(_ x: UInt64, _ base: UInt64) -> UInt64 {
    // TODO: REPLACE with your transform identical to GPU logic for apples-to-apples.
    var y: UInt64 = 0, p: UInt64 = 1
    while true {
        let np = p &* base
        if np == 0 || np > x { break }
        p = np; y &+= 1
    }
    return y
}
// -----------------------------------------------------

@inline(__always)
func ipow_u64(_ base: UInt64, _ exp: UInt64) -> UInt64 {
    var r: UInt64 = 1, b = base, e = exp
    while e != 0 {
        if (e & 1) != 0 { r = r &* b }
        e >>= 1
        if e != 0 { b = b &* b }
    }
    return r
}

// Inputs
var xs = (0..<N).map { _ in UInt64.random(in: 2...UInt64(1<<32)) }
var bs = (0..<N).map { _ in UInt64([2,3,5,7,10].randomElement()!) }
var ys = [UInt64](repeating: 0, count: N)

// CPU baseline
let t0c = CFAbsoluteTimeGetCurrent()
var sinkCPU: UInt64 = 0
for i in 0..<N { sinkCPU &+= bitwise_log_u64(xs[i], bs[i]) }
let cpuTime = CFAbsoluteTimeGetCurrent() - t0c

// GPU setup
let device = makeDevice()
let metalSrc = readFile("BitwiseLog.metal")
let lib = makeLibrary(device: device, source: metalSrc)
let fn = lib.makeFunction(name: "bitwiseLogKernel")!
let pipe = try! device.makeComputePipelineState(function: fn)
let q = device.makeCommandQueue()!

let bx = device.makeBuffer(bytes: &xs, length: MemoryLayout<UInt64>.stride * N, options: .storageModeShared)!
let bb = device.makeBuffer(bytes: &bs, length: MemoryLayout<UInt64>.stride * N, options: .storageModeShared)!
let by = device.makeBuffer(bytes: &ys, length: MemoryLayout<UInt64>.stride * N, options: .storageModeShared)!

let grid = MTLSize(width: N, height: 1, depth: 1)
let tg = MTLSize(width: min(pipe.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)

// Warm-up
do {
    let cmd = q.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(bx, offset: 0, index: 0)
    enc.setBuffer(bb, offset: 0, index: 1)
    enc.setBuffer(by, offset: 0, index: 2)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
}

// Timed run
let cmd = q.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipe)
enc.setBuffer(bx, offset: 0, index: 0)
enc.setBuffer(bb, offset: 0, index: 1)
enc.setBuffer(by, offset: 0, index: 2)
enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
enc.endEncoding()
cmd.commit(); cmd.waitUntilCompleted()

var gpuTime: Double = 0
if #available(macOS 10.15, *) {
    let s = cmd.gpuStartTime, e = cmd.gpuEndTime
    if s != 0 && e != 0 { gpuTime = e - s }
}

memcpy(&ys, by.contents(), MemoryLayout<UInt64>.stride * N)

// quick correctness sample (20 random indices)
var ok = true
for _ in 0..<20 {
    let i = Int.random(in: 0..<N)
    let y = ys[i], b = bs[i], x = xs[i]
    let lo = ipow_u64(b, y)
    let hi = ipow_u64(b, y &+ 1)
    if !(lo <= x && x < hi) { ok = false; break }
}

let speedup = cpuTime / max(gpuTime, 1e-12)
var sinkGPU: UInt64 = 0; for v in ys { sinkGPU &+= v }
print(String(format: "N=%d | CPU=%.3f s | GPU=%.6f s | speedup=%.2fx | ok=%@ | sinkCPU=%llu | sinkGPU=%llu",
             N, cpuTime, gpuTime, speedup, ok ? "true" : "false", sinkCPU, sinkGPU))
