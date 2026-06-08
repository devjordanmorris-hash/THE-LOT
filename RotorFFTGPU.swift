import Foundation
import Metal
import MetalKit

final class RotorFFTGPU {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    let lut: [Float]
    let lutSize: Int
    let step: Float

    init(device: MTLDevice, lutSize: Int = 65536) {
        self.device = device
        self.queue = device.makeCommandQueue()!

        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "tier0_fft_rotor")!
        self.pipeline = try! device.makeComputePipelineState(function: kernel)

        // ------------------------------
        // Build LUT (CPU rotor)
        // ------------------------------
        self.lutSize = lutSize
        self.lut = RotorFFTGPU.buildRotorLUT(lutSize)
        self.step = Float(lutSize) / (.pi / 2)
    }

    // =====================================================
    // Build LUT using rotor integer decay
    // =====================================================
    static func buildRotorLUT(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)

        var dx: Float = 1.0
        var dy: Float = 0.0

        let k: Float = 0.02
        let decay: Float = 0.015

        for i in 0..<n {
            let xr = dx - dy * k
            let yr = dy + dx * k
            dx = xr * (1 - decay)
            dy = yr * (1 - decay)
            out[i] = dy
        }
        return out
    }

    // =====================================================
    // Curvature Boundary Scan
    // =====================================================
    func computeActiveBins(_ signal: [Float], threshold: Float = 0.001) -> [UInt32] {
        let N = signal.count
        var out: [UInt32] = []
        out.reserveCapacity(N)

        for i in 1..<N-1 {
            let c = abs(signal[i+1] - 2*signal[i] + signal[i-1])
            if c > threshold {
                out.append(UInt32(i))
            }
        }
        return out
    }

    // =====================================================
    // GPU FFT (Tier-0 rotor)
    // =====================================================
    func fftGPU(_ signal: [Float]) -> [(Float, Float)] {

        let N = signal.count
        let active = computeActiveBins(signal)
        let activeCount = active.count

        var output = Array(repeating: (Float(0), Float(0)), count: N)

        let sigBuf = device.makeBuffer(bytes: signal,
                                       length: MemoryLayout<Float>.stride * N,
                                       options: .storageModeShared)!

        let Nbuf = device.makeBuffer(bytes: [UInt32(N)],
                                     length: 4,
                                     options: .storageModeShared)!

        let actBuf = device.makeBuffer(bytes: active,
                                       length: MemoryLayout<UInt32>.stride * activeCount,
                                       options: .storageModeShared)!

        let actCountBuf = device.makeBuffer(bytes: [UInt32(activeCount)],
                                            length: 4,
                                            options: .storageModeShared)!

        let outBuf = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * N,
                                       options: .storageModeShared)!

        let lutBuf = device.makeBuffer(bytes: lut,
                                       length: MemoryLayout<Float>.stride * lutSize,
                                       options: .storageModeShared)!

        let lutSizeBuf = device.makeBuffer(bytes: [UInt32(lutSize)],
                                           length: 4,
                                           options: .storageModeShared)!

        var stepCopy = step
        let stepBuf = device.makeBuffer(bytes: &stepCopy,
                                        length: 4,
                                        options: .storageModeShared)!

        let command = queue.makeCommandBuffer()!
        let enc = command.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)

        enc.setBuffer(sigBuf, offset: 0, index: 0)
        enc.setBuffer(Nbuf, offset: 0, index: 1)
        enc.setBuffer(actBuf, offset: 0, index: 2)
        enc.setBuffer(actCountBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        enc.setBuffer(lutBuf, offset: 0, index: 5)
        enc.setBuffer(lutSizeBuf, offset: 0, index: 6)
        enc.setBuffer(stepBuf, offset: 0, index: 7)

        // One thread per active bin
        let threads = MTLSize(width: activeCount, height: 1, depth: 1)
        let tg = MTLSize(width: max(1, min(32, activeCount)), height: 1, depth: 1)

        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        for i in 0..<N {
            output[i] = (ptr[i].x, ptr[i].y)
        }

        return output
    }
}
