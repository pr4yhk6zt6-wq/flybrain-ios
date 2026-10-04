//
//  Renderer.swift
//  One draw call for 139,255 neurons, plus a bounded set of spike arcs.
//

import Foundation
import Metal
import MetalKit
import simd

/// Must match `struct RenderParams` in Render.metal.
struct RenderParams {
    var viewProjection: float4x4 = matrix_identity_float4x4
    var model: float4x4 = matrix_identity_float4x4

    var pointScale: Float = 2.5
    var voltageFloor: Float = -0.8
    var voltageCeiling: Float = 1.0
    var glowStrength: Float = 1.0

    var neuronCount: UInt32 = 0
    var systemMask: UInt32 = 0xFFFF_FFFF
    var renderMode: UInt32 = 0
    var alphaScale: Float = 0.85

    var lodThreshold: Float = 1.0
    var time: Float = 0
    var highlightIndex: UInt32 = 0xFFFF_FFFF
    var pad: UInt32 = 0
}

enum RenderMode: UInt32, CaseIterable, Identifiable {
    case voltage = 0, spikingOnly = 1, bySystem = 2
    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .voltage:     return "Voltage"
        case .spikingOnly: return "Spiking"
        case .bySystem:    return "Regions"
        }
    }
}

/// Orbit camera driven by gestures.
struct OrbitCamera {
    var azimuth: Float = 0.6
    var elevation: Float = 0.25
    var distance: Float = 2.2
    var target: SIMD3<Float> = .zero
    var fieldOfView: Float = 55 * .pi / 180

    mutating func rotate(dx: Float, dy: Float) {
        azimuth += dx
        elevation = max(-1.5, min(1.5, elevation + dy))
    }
    mutating func zoom(_ factor: Float) {
        distance = max(0.25, min(8.0, distance / factor))
    }
    mutating func pan(dx: Float, dy: Float) {
        let right = SIMD3<Float>(cos(azimuth), 0, -sin(azimuth))
        let up = SIMD3<Float>(0, 1, 0)
        target += (right * dx + up * dy) * distance * 0.5
    }

    var eye: SIMD3<Float> {
        let ce = cos(elevation), se = sin(elevation)
        return target + SIMD3<Float>(sin(azimuth) * ce, se, cos(azimuth) * ce) * distance
    }

    func viewProjection(aspect: Float) -> float4x4 {
        let view = float4x4(lookAt: eye, target: target, up: SIMD3<Float>(0, 1, 0))
        let proj = float4x4(perspectiveFOV: fieldOfView, aspect: aspect,
                            near: 0.02, far: 40.0)
        return proj * view
    }

    /// Screen point -> world ray, for tap-to-inspect.
    func ray(atNDC ndc: SIMD2<Float>, aspect: Float) -> (SIMD3<Float>, SIMD3<Float>) {
        let inv = viewProjection(aspect: aspect).inverse
        let near = inv * SIMD4<Float>(ndc.x, ndc.y, 0, 1)
        let far  = inv * SIMD4<Float>(ndc.x, ndc.y, 1, 1)
        let p0 = SIMD3<Float>(near.x, near.y, near.z) / near.w
        let p1 = SIMD3<Float>(far.x, far.y, far.z) / far.w
        return (p0, normalize(p1 - p0))
    }
}

final class Renderer: NSObject, MTKViewDelegate {

    var camera = OrbitCamera()
    var renderMode: RenderMode = .voltage
    var systemMask: UInt32 = 0xFFFF_FFFF
    var pointScale: Float = 2.5
    var highlightIndex: Int? = nil
    var showSpikeArcs = true

    private(set) var fps: Double = 0
    private(set) var frameMilliseconds: Double = 0

    /// How many 1 ms brain steps to run per displayed frame. Clamped so that a
    /// slow device degrades by simulating slower, not by dropping frames.
    var stepsPerFrame: Int = 4

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let connectome: Connectome
    private let simulation: SimulationEngine

    private let neuronPipeline: MTLRenderPipelineState
    private var arcPipeline: MTLRenderPipelineState?
    private let depthState: MTLDepthStencilState?

    private var params = RenderParams()
    private var lastFrame = CFAbsoluteTimeGetCurrent()
    private var frameTimes: [Double] = []

    var cameraTextureProvider: (() -> MTLTexture?)?

    init(device: MTLDevice,
         connectome: Connectome,
         simulation: SimulationEngine,
         library: MTLLibrary,
         pixelFormat: MTLPixelFormat,
         depthFormat: MTLPixelFormat) throws {

        self.device = device
        self.connectome = connectome
        self.simulation = simulation
        guard let q = device.makeCommandQueue() else { throw SimulationError.noCommandQueue }
        self.queue = q
        q.label = "FlyBrain.render"

        // --- neuron point cloud ------------------------------------------
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "neurons"
        desc.vertexFunction = library.makeFunction(name: "neuronVertex")
        desc.fragmentFunction = library.makeFunction(name: "neuronFragment")
        desc.colorAttachments[0].pixelFormat = pixelFormat
        // Additive blending: overlapping neurons accumulate brightness, which is
        // what makes a dense firing region read as a glowing mass.
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .one
        desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationAlphaBlendFactor = .one
        desc.depthAttachmentPixelFormat = depthFormat
        neuronPipeline = try device.makeRenderPipelineState(descriptor: desc)

        // --- spike arcs ----------------------------------------------------
        if let vfn = library.makeFunction(name: "spikeArcVertex"),
           let ffn = library.makeFunction(name: "spikeArcFragment") {
            let a = MTLRenderPipelineDescriptor()
            a.label = "spikeArcs"
            a.vertexFunction = vfn
            a.fragmentFunction = ffn
            a.colorAttachments[0].pixelFormat = pixelFormat
            a.colorAttachments[0].isBlendingEnabled = true
            a.colorAttachments[0].rgbBlendOperation = .add
            a.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            a.colorAttachments[0].destinationRGBBlendFactor = .one
            a.depthAttachmentPixelFormat = depthFormat
            arcPipeline = try? device.makeRenderPipelineState(descriptor: a)
        }

        // Depth test off, depth write off: additive point clouds look better
        // without occlusion, and it saves the depth resolve on tiled GPUs.
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .always
        dd.isDepthWriteEnabled = false
        depthState = device.makeDepthStencilState(descriptor: dd)

        params.neuronCount = UInt32(connectome.neuronCount)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - lastFrame
        lastFrame = now
        frameTimes.append(dt)
        if frameTimes.count > 30 { frameTimes.removeFirst() }
        let mean = frameTimes.reduce(0, +) / Double(max(frameTimes.count, 1))
        fps = mean > 0 ? 1.0 / mean : 0
        frameMilliseconds = mean * 1000.0

        // Simulate first, then draw the state it produced.
        simulation.step(count: stepsPerFrame, cameraTexture: cameraTextureProvider?())

        guard let drawable = view.currentDrawable,
              let pass = view.currentRenderPassDescriptor,
              let cb = queue.makeCommandBuffer() else { return }
        cb.label = "FlyBrain.frame"

        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.015, green: 0.02,
                                                            blue: 0.045, alpha: 1)

        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "neurons"

        let size = view.drawableSize
        let aspect = Float(max(size.width, 1) / max(size.height, 1))

        params.viewProjection = camera.viewProjection(aspect: aspect)
        params.model = matrix_identity_float4x4
        params.renderMode = renderMode.rawValue
        params.systemMask = systemMask
        params.pointScale = pointScale * Float(view.contentScaleFactor)
        params.time = Float(now.truncatingRemainder(dividingBy: 1000))
        params.highlightIndex = UInt32(highlightIndex ?? Int(UInt32.max))
        // Level of detail: as the camera pulls back, points shrink below a pixel
        // and the cloud turns to mush. Fading instead of shrinking keeps the
        // density readable and cuts overdraw at the same time.
        params.lodThreshold = camera.distance > 3.0 ? 1.4 : 1.0

        enc.setRenderPipelineState(neuronPipeline)
        if let d = depthState { enc.setDepthStencilState(d) }
        enc.setVertexBuffer(connectome.positions, offset: 0, index: 0)
        enc.setVertexBuffer(simulation.voltageBuffer, offset: 0, index: 1)
        enc.setVertexBuffer(simulation.activityBuffer, offset: 0, index: 2)
        enc.setVertexBuffer(connectome.neuronMeta, offset: 0, index: 3)
        enc.setVertexBytes(&params, length: MemoryLayout<RenderParams>.stride, index: 4)
        enc.drawPrimitives(type: .point, vertexStart: 0,
                           vertexCount: connectome.neuronCount)

        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }

    /// Tap -> nearest neuron.
    func pick(at point: CGPoint, in size: CGSize) -> Int? {
        let ndc = SIMD2<Float>(Float(point.x / size.width) * 2 - 1,
                               1 - Float(point.y / size.height) * 2)
        let aspect = Float(max(size.width, 1) / max(size.height, 1))
        let (origin, direction) = camera.ray(atNDC: ndc, aspect: aspect)
        return connectome.pick(rayOrigin: origin, rayDirection: direction)
    }
}

// MARK: - Matrix helpers

extension float4x4 {
    init(perspectiveFOV fov: Float, aspect: Float, near: Float, far: Float) {
        let y = 1 / tan(fov * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        self.init(SIMD4<Float>( x, 0, 0,  0),
                  SIMD4<Float>( 0, y, 0,  0),
                  SIMD4<Float>( 0, 0, z, -1),
                  SIMD4<Float>( 0, 0, z * near, 0))
    }

    init(lookAt eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) {
        let f = normalize(target - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)
        self.init(SIMD4<Float>(s.x, u.x, -f.x, 0),
                  SIMD4<Float>(s.y, u.y, -f.y, 0),
                  SIMD4<Float>(s.z, u.z, -f.z, 0),
                  SIMD4<Float>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1))
    }
}
