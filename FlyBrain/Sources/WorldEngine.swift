//
//  WorldEngine.swift
//  Owns the closed loop in world mode.
//
//  Per frame, in order:
//      1. render the world from the fly's eyes into a 128x128 texture
//      2. blur it to ommatidial acuity
//      3. step the brain with that texture as photoreceptor input
//      4. read the motor group firing rates back off the GPU
//      5. integrate the body, move it through the world
//      6. write the world's smells, contacts and pain into sensory groups
//      7. render the third-person view
//
//  Step 4 is the only CPU/GPU sync point, and the group reduction kernel keeps
//  it down to ~35 uints per frame.
//

import Foundation
import Metal
import MetalKit
import simd

/// How the chase camera sits behind the fly.
struct ChaseCamera {
    /// World units are centimetres and the fly is 0.25 cm long, so 0.9 cm back
    /// frames it at roughly three body lengths — close enough to read the
    /// wingbeat, far enough to see where it is going.
    var distance: Float = 0.9
    var height: Float = 0.3
    var yawOffset: Float = 0
    var smoothed = SIMD3<Float>(0, 0.5, 2)
    var lookAt = SIMD3<Float>.zero

    mutating func update(target: SIMD3<Float>, heading: Float, dt: Float) {
        let yaw = heading + yawOffset
        let back = SIMD3<Float>(-sin(yaw), 0, cos(yaw))
        let want = target + back * distance + SIMD3<Float>(0, height, 0)

        // A fly crosses the room in well under a second. A fixed exponential
        // follow simply cannot keep up with that, which is why it ended up as
        // a speck in the distance. Snap whenever we have fallen more than a
        // few camera-distances behind, and smooth only the small corrections.
        if length(want - smoothed) > distance * 3 {
            smoothed = want
            lookAt = target
        } else {
            smoothed += (want - smoothed) * min(1, dt * 12)
            lookAt += (target - lookAt) * min(1, dt * 18)
        }
    }

    func matrix(aspect: Float) -> (float4x4, SIMD3<Float>) {
        // Near plane has to be tight: at 0.9 cm from a 0.25 cm animal, a 2 cm
        // near plane would clip the subject away entirely.
        let proj = float4x4(perspectiveFOV: 55 * .pi / 180, aspect: aspect,
                            near: 0.01, far: 80)
        let view = float4x4(lookAt: smoothed, target: lookAt, up: SIMD3<Float>(0, 1, 0))
        return (proj * view, smoothed)
    }
}

@MainActor
final class WorldEngine: NSObject, ObservableObject, MTKViewDelegate {

    @Published private(set) var environment: Environment = .kitchen

    /// Switching worlds rebuilds the scene and drops the fly back in the
    /// middle of it. Called straight from the buttons — no two-way binding to
    /// get out of step with the model.
    func setEnvironment(_ e: Environment) {
        guard e != environment else { return }
        environment = e
        world.environment = e
        body.reset()
        chase.smoothed = SIMD3<Float>(0, 0.6, 2)
    }
    @Published private(set) var fps: Double = 0
    @Published private(set) var pose = FlyPose()
    @Published private(set) var drives = FlyDrives()
    @Published private(set) var energy: Float = 1
    @Published private(set) var odourStrength: Float = 0
    @Published private(set) var speed: Float = 0
    /// Real aerodynamic numbers, in micronewtons.
    @Published private(set) var liftUN: Float = 0
    @Published private(set) var weightUN: Float = 0
    @Published private(set) var strokeAmplitudeDeg: Float = 0
    @Published private(set) var yawRateDegPerSec: Float = 0
    @Published var showBrainPiP = true
    @Published var showEyePiP = true

    /// 1 ms brain steps per displayed frame.
    var stepsPerFrame: Int = 4

    let world = World()
    let body = FlyBody()
    var chase = ChaseCamera()

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var renderer: WorldRenderer?
    private weak var simulation: SimulationEngine?

    private var lastFrame = CFAbsoluteTimeGetCurrent()
    private var frameTimes: [Double] = []
    private var statCounter = 0

    init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
        super.init()
    }

    func attach(simulation: SimulationEngine, library: MTLLibrary, view: MTKView) throws {
        self.simulation = simulation
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearDepth = 1.0
        view.preferredFramesPerSecond = 60
        if renderer == nil {
            renderer = try WorldRenderer(device: device, queue: queue,
                                         library: library,
                                         colorFormat: view.colorPixelFormat)
        }
        if let m = renderer?.flyModel { body.attach(model: m) }
        view.delegate = self
        body.reset()
        world.environment = environment
    }

    /// The texture the photoreceptors read, also shown in the eye PiP.
    var eyeTexture: MTLTexture? { renderer?.eyeBlurred }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CFAbsoluteTimeGetCurrent()
        var dt = Float(now - lastFrame)
        lastFrame = now
        frameTimes.append(Double(dt))
        if frameTimes.count > 30 { frameTimes.removeFirst() }
        let mean = frameTimes.reduce(0, +) / Double(max(frameTimes.count, 1))
        fps = mean > 0 ? 1 / mean : 0
        dt = min(max(dt, 1.0 / 240.0), 1.0 / 20.0)   // never let a stall teleport the fly

        guard let renderer = renderer, let sim = simulation,
              let cb = queue.makeCommandBuffer() else { return }
        cb.label = "FlyWorld.frame"

        // 1+2. What the fly sees, at the acuity it sees it.
        renderer.buildInstances(world: world, body: body, showFly: true)
        renderer.drawEye(commandBuffer: cb, body: body, world: world, time: Float(now))

        // 3. One frame of brain, driven by that image.
        if !sim.isPaused {
            sim.step(count: stepsPerFrame, cameraTexture: renderer.eyeBlurred)
        }

        // 7. The view the user watches, in the same command buffer.
        let aspect = Float(max(view.drawableSize.width, 1) / max(view.drawableSize.height, 1))
        chase.update(target: body.pose.position, heading: body.pose.heading, dt: dt)
        let (viewProj, camPos) = chase.matrix(aspect: aspect)
        renderer.drawMain(in: view, commandBuffer: cb, viewProj: viewProj,
                          cameraPos: camPos, world: world, time: Float(now))

        if let drawable = view.currentDrawable { cb.present(drawable) }
        cb.commit()
        renderer.advanceFrame()

        // 4. Motor rates off the GPU, 5. body, 6. senses back in.
        body.readMotorDrives(from: sim)
        body.update(dt: dt, world: world)
        body.writeSensoryDrives(to: sim, world: world, visionActive: true)
        world.update(dt: dt)

        // Publishing every frame would thrash SwiftUI; 10 Hz is plenty.
        statCounter += 1
        if statCounter % 6 == 0 {
            pose = body.pose
            drives = body.drives
            energy = body.energy
            speed = length(body.pose.velocity)
            odourStrength = world.odour(at: body.pose.position,
                                        heading: body.pose.heading).strength
            liftUN = body.liftMicroNewtons
            weightUN = body.weightMicroNewtons
            strokeAmplitudeDeg = (body.pose.strokeAmplitudeL
                                + body.pose.strokeAmplitudeR) * 0.5 * 180 / .pi
            yawRateDegPerSec = body.pose.yawRate * 180 / .pi
        }
    }

    // MARK: - User actions

    func orbit(dx: Float, dy: Float) {
        chase.yawOffset -= dx * 0.01
        chase.height = max(-0.4, min(2.5, chase.height - dy * 0.004))
    }

    func zoom(_ scale: Float) {
        chase.distance = max(0.25, min(8, chase.distance / max(scale, 0.01)))
    }

    func recentre() {
        chase.yawOffset = 0
        chase.height = 0.3
        chase.distance = 0.9
    }

    /// Drop a crumb. The fly has to find it by smell — nothing teleports it.
    func dropFood() {
        let p = body.pose.position
              + SIMD3<Float>(Float.random(in: -2.5...2.5), 0, Float.random(in: -2.5...2.5))
        world.spawn(.food, at: SIMD3<Float>(p.x, 0.6, p.z), size: 0.14)
    }

    func spawnObject() {
        let p = body.pose.position
              + SIMD3<Float>(Float.random(in: -2...2), 2.0, Float.random(in: -2...2))
        world.spawn(.cube, at: p, size: Float.random(in: 0.15...0.4))
    }

    /// Swat at the fly. The giant fibre pathway gets a very short window.
    func swat() {
        world.swat(at: body.pose.position
                   + SIMD3<Float>(Float.random(in: -0.15...0.15), 0,
                                  Float.random(in: -0.15...0.15)))
    }

    func resetFly() {
        body.reset()
        chase.smoothed = SIMD3<Float>(0, 0.6, 2)
        simulation?.reset()
    }
}
