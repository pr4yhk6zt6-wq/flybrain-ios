//
//  CameraFeed.swift
//  The rear camera, as the fly's eyes.
//
//  Frames are converted to a single-channel luminance texture on the GPU via
//  CVMetalTextureCache — the Y plane of the camera's native 420f output is
//  already exactly what the photoreceptor kernel wants, so there is no colour
//  conversion and no CPU touch of the pixel data.
//

import Foundation
import AVFoundation
import Metal
import CoreVideo
import Combine

final class CameraFeed: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published private(set) var isRunning = false
    @Published private(set) var framesDelivered: Int = 0
    /// Mean luminance of the last frame, 0...1 — proof the pixels are real.
    @Published private(set) var meanLuminance: Float = 0

    /// Exposed so SwiftUI can show an AVCaptureVideoPreviewLayer of the same
    /// session the simulation is reading. One capture session, two consumers.
    let session = AVCaptureSession()

    private let output = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "FlyBrain.camera")

    private var textureCache: CVMetalTextureCache?
    private let lock = NSLock()

    // The MTLTexture returned by CVMetalTextureGetTexture does NOT own its
    // backing store — the CVMetalTexture does. Releasing the CVMetalTexture
    // while still holding the MTLTexture hands the GPU a recycled IOSurface,
    // which is why the camera appeared to do nothing: the kernel was sampling
    // whatever happened to be in a reused buffer. Both must be retained, and
    // the pair must stay alive until the next frame replaces it.
    private var currentCVTexture: CVMetalTexture?
    private var currentTexture: MTLTexture?

    private var configured = false

    init(device: MTLDevice) {
        super.init()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    func requestAccessAndStart(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart(); completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.configureAndStart() }
                    completion(granted)
                }
            }
        default:
            completion(false)
        }
    }

    private func configureAndStart() {
        guard !isRunning else { return }

        if !configured {
            session.beginConfiguration()
            session.sessionPreset = .vga640x480   // each retina cell samples one texel;
                                              // VGA is 200x more than the 2,784 need

            if let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                    for: .video, position: .back),
               let input = try? AVCaptureDeviceInput(device: device),
               session.canAddInput(input) {
                session.addInput(input)
            }

            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String:
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            ]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: sampleQueue)
            if session.canAddOutput(output) { session.addOutput(output) }

            session.commitConfiguration()
            configured = true
        }

        isRunning = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.stopRunning()
        }
        lock.lock()
        currentTexture = nil
        currentCVTexture = nil
        lock.unlock()
    }

    /// Latest luminance texture, or nil if no frame has arrived yet.
    func latestTexture() -> MTLTexture? {
        lock.lock(); defer { lock.unlock() }
        return currentTexture
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let cache = textureCache,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)

        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .r8Unorm, width, height, 0, &cvTexture)

        guard status == kCVReturnSuccess,
              let cvTex = cvTexture,
              let texture = CVMetalTextureGetTexture(cvTex) else { return }

        // Cheap mean luminance from the Y plane, subsampled 16x in both axes,
        // purely so the UI can prove frames are arriving and being read.
        var mean: Float = 0
        if CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess {
            if let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) {
                let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
                let p = base.assumingMemoryBound(to: UInt8.self)
                var total = 0, n = 0
                for y in stride(from: 0, to: height, by: 16) {
                    for x in stride(from: 0, to: width, by: 16) {
                        total += Int(p[y * rowBytes + x]); n += 1
                    }
                }
                if n > 0 { mean = Float(total) / Float(n) / 255.0 }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        lock.lock()
        currentCVTexture = cvTex      // keeps the IOSurface alive
        currentTexture = texture
        lock.unlock()

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.framesDelivered &+= 1
            self.meanLuminance = mean
        }
    }
}
