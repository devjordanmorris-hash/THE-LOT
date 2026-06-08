import Foundation
import Accelerate
import Metal

let device = MTLCreateSystemDefaultDevice()!
let fftGPU = RotorFFTGPU(device: device, lutSize: 65536)

let N = 4096
var signal = [Float](repeating: 0, count: N)
for i in 0..<N {
    signal[i] =
        sin(Float(i) * 0.01) +
        0.33 * sin(Float(i) * 0.07) +
        0.12 * sin(Float(i) * 0.37)
}

// GPU FFT ---------------------------------
let t0 = CFAbsoluteTimeGetCurrent()
let outGPU = fftGPU.fftGPU(signal)
let t1 = CFAbsoluteTimeGetCurrent()

print("Tier-0 GPU FFT:", (t1 - t0) * 1000, "ms")

// Accelerate FFT ---------------------------
var realPart = signal.map(Double.init)
var imagPart = [Double](repeating: 0, count: N)
var split = DSPDoubleSplitComplex(realp: &realPart, imagp: &imagPart)

let logN = vDSP_Length(log2(Float(N)))
let setup = vDSP_create_fftsetupD(logN, FFTRadix(kFFTRadix2))!

let t2 = CFAbsoluteTimeGetCurrent()
vDSP_fft_zipD(setup, &split, 1, logN, FFTDirection(FFT_FORWARD))
let t3 = CFAbsoluteTimeGetCurrent()
print("Accelerate FFT:", (t3 - t2) * 1000, "ms")

vDSP_destroy_fftsetupD(setup)
