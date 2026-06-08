//
//  AudioLoader.swift
//  wavefrom to triangle to prism compression
//
//  Created by Jordan Morris  on 20/11/2025.
//

import Foundation
import AVFoundation

final class AudioLoader {

    enum AudioLoaderError: Error {
        case bufferAllocationFailed
        case channelDataMissing
    }

    /// Loads a WAV file into an array of Float samples and returns sample rate.
    func loadSamples(from url: URL) throws -> ([[Float]], Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = UInt32(file.length)

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw AudioLoaderError.bufferAllocationFailed
        }

        try file.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw AudioLoaderError.channelDataMissing
        }

        let channelCount = Int(format.channelCount)
        var channels: [[Float]] = []

        for c in 0..<channelCount {
            let samples = Array(
                UnsafeBufferPointer(
                    start: channelData[c],
                    count: Int(buffer.frameLength)
                )
            )
            channels.append(samples)
        }

        let sampleRate = format.sampleRate
        return (channels, sampleRate)
    }
}
