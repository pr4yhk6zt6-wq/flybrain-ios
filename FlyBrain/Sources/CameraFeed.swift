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

final class CameraFeed: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private(set) var isRunning = false
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "FlyBrain.camera")

    private var textureCache: CVMetalTextureCache?
    private var currentTexture: MTLTexture?
    private let lock = NSLock()

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
        session.beginConfiguration()
        session.sessionPreset = .vga640x480   // the retina is 10,647 cells; VGA is plenty

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
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        session.stopRunning()
        isRunning = false
        lock.lock(); currentTexture = nil; lock.unlock()
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

        lock.lock()
        currentTexture = texture
        lock.unlock()
    }
}
