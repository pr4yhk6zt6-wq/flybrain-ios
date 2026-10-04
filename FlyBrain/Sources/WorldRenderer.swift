//
//  WorldRenderer.swift
//  Draws the world twice per frame: once from a third-person chase camera for
//  the main view, once from between the fly's eyes at 128x128 into an offscreen
//  texture. That second render is the input to the photoreceptor kernel, so the
//  fly genuinely sees the place it is flying through.
//

import Foundation
import Metal
import MetalKit
import simd

struct WorldParams {
    var viewProj = matrix_identity_float4x4
    var cameraPos = SIMD3<Float>(0, 0, 0)
    var time: Float = 0
    var sunDir = normalize(SIMD3<Float>(0.4, 0.85, 0.3))
    var checker: Float = 3
    var groundColour = SIMD3<Float>(0.5, 0.4, 0.3)
    var fogDensity: Float = 0.02
    var skyColour = SIMD3<Float>(0.2, 0.3, 0.4)
    var pad: Float = 0
}

struct WorldInstance {
    var model = matrix_identity_float4x4
    var colour = SIMD4<Float>(1, 1, 1, 0)
    var params = SIMD4<Float>(0, 0, 0, 0)
}

/// CPU-side primitive meshes: a cube, a subdivided icosphere and a quad.
private struct Primitive {
    var vertexBuffer: MTLBuffer
    var indexBuffer: MTLBuffer
    var indexCount: Int
}

final class WorldRenderer {

    enum WorldError: Error { case allocation(String) }

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    private var pipeline: MTLRenderPipelineState!
    private var skyPipeline: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!
    private var skyDepthState: MTLDepthStencilState!
    private var blurPipeline: MTLComputePipelineState?

    private var cube: Primitive!
    private var sphere: Primitive!
    private var quad: Primitive!

    /// Triple-buffered, so the CPU never writes what the GPU is reading.
    private var instanceBuffers: [MTLBuffer] = []
    private var frameIndex = 0
    private let maxInstances = 640

    /// One render target per compound eye, plus the ommatidially-blurred
    /// versions the photoreceptor kernel actually samples. The left texture
    /// feeds the left optic lobe, the right the right — the brain finally
    /// gets a real left/right difference to steer with.
    private(set) var eyeTextureL: MTLTexture!
    private(set) var eyeTextureR: MTLTexture!
    private(set) var eyeBlurredL: MTLTexture!
    private(set) var eyeBlurredR: MTLTexture!
    private var eyeDepthL: MTLTexture!
    private var eyeDepthR: MTLTexture!
    let eyeSize = 128
    /// Each ommatidium covers this many pixels of the 128x128 render.
    var facetSize: UInt32 = 6

    /// Why the anatomical mesh is, or is not, on screen.
    ///
    /// This exists because the mesh used to be loaded with `try?`. When
    /// flymodel.bin was not in the bundle the whole animal silently stopped
    /// being drawn: no crash, no log, no empty silhouette — a room with
    /// nothing in it. The state is published up to the HUD instead, so a
    /// missing asset says so on screen.
    enum FlyMeshState {
        case loaded(bodyMillimetres: Float, wingspanMillimetres: Float)
        case missing
        case failed(String)

        var isLoaded: Bool { if case .loaded = self { return true } else { return false } }

        /// Short text for the HUD. Nil means "nothing to report".
        var note: String? {
            switch self {
            case .loaded: return nil
            case .missing: return "NO FLY MESH (fallback body)"
            case .failed(let why): return "FLY MESH: \(why)"
            }
        }
    }
    private(set) var flyMeshState: FlyMeshState = .missing

    /// The real Drosophila mesh. Optional so the app still runs if the asset
    /// is missing; the primitive fallback in `appendPrimitiveFly` covers it.
    private(set) var flyModel: FlyModel?
    private var partMatrices: [float4x4] = []
    private var flyInstanceStart = 0

    private var instances: [WorldInstance] = []
    private var cubeRange: Range<Int> = 0..<0
    private var sphereRange: Range<Int> = 0..<0
    private var quadRange: Range<Int> = 0..<0

    var params = WorldParams()

    init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary,
         colorFormat: MTLPixelFormat) throws {
        self.device = device
        self.queue = queue

        cube = try makePrimitive(cubeMesh())
        sphere = try makePrimitive(icosphereMesh(subdivisions: 2))
        quad = try makePrimitive(quadMesh())

        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float3
        vd.attributes[1].offset = MemoryLayout<Float>.stride * 3
        vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = MemoryLayout<Float>.stride * 6

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "world"
        desc.vertexFunction = library.makeFunction(name: "worldVertex")
        desc.fragmentFunction = library.makeFunction(name: "worldFragment")
        desc.vertexDescriptor = vd
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = .depth32Float
        pipeline = try device.makeRenderPipelineState(descriptor: desc)

        let sky = MTLRenderPipelineDescriptor()
        sky.label = "sky"
        sky.vertexFunction = library.makeFunction(name: "skyVertex")
        sky.fragmentFunction = library.makeFunction(name: "skyFragment")
        sky.colorAttachments[0].pixelFormat = colorFormat
        sky.depthAttachmentPixelFormat = .depth32Float
        skyPipeline = try device.makeRenderPipelineState(descriptor: sky)

        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: ds)

        let sds = MTLDepthStencilDescriptor()
        sds.depthCompareFunction = .always
        sds.isDepthWriteEnabled = false
        skyDepthState = device.makeDepthStencilState(descriptor: sds)

        if let fn = library.makeFunction(name: "ommatidiaBlur") {
            blurPipeline = try device.makeComputePipelineState(function: fn)
        }

        for i in 0..<3 {
            guard let b = device.makeBuffer(
                length: MemoryLayout<WorldInstance>.stride * maxInstances,
                options: .storageModeShared) else {
                throw WorldError.allocation("instances")
            }
            b.label = "worldInstances\(i)"
            instanceBuffers.append(b)
        }

        // The anatomical fly. 41 parts, 76 joints, from the Janelia/DeepMind
        // CT reconstruction. If it will not load we say so rather than
        // quietly rendering an empty room.
        do {
            let m = try FlyModel(device: device)
            flyModel = m
            flyMeshState = .loaded(
                bodyMillimetres: m.measuredBodyLength * m.normalisationScale * 10,
                wingspanMillimetres: m.measuredWingspan * m.normalisationScale * 10)
            partMatrices = [float4x4](repeating: matrix_identity_float4x4,
                                      count: m.parts.count)
        } catch FlyModelError.missing {
            flyModel = nil
            flyMeshState = .missing
        } catch {
            flyModel = nil
            flyMeshState = .failed(error.localizedDescription)
        }

        try makeEyeTargets(colorFormat: colorFormat)
    }

    private func makeEyeTargets(colorFormat: MTLPixelFormat) throws {
        func make(_ label: String, _ format: MTLPixelFormat,
                  _ usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: eyeSize, height: eyeSize, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else {
                throw WorldError.allocation(label)
            }
            t.label = label
            return t
        }
        eyeTextureL = try make("flyEyeL", colorFormat, [.renderTarget, .shaderRead])
        eyeTextureR = try make("flyEyeR", colorFormat, [.renderTarget, .shaderRead])
        eyeBlurredL = try make("flyEyeOmmatidiaL", .rgba16Float, [.shaderRead, .shaderWrite])
        eyeBlurredR = try make("flyEyeOmmatidiaR", .rgba16Float, [.shaderRead, .shaderWrite])
        eyeDepthL = try make("flyEyeDepthL", .depth32Float, [.renderTarget])
        eyeDepthR = try make("flyEyeDepthR", .depth32Float, [.renderTarget])
    }

    // MARK: - Building the frame's instance list

    func buildInstances(world: World, body: FlyBody, showFly: Bool) {
        var cubes: [WorldInstance] = []
        var spheres: [WorldInstance] = []
        var quads: [WorldInstance] = []

        // Ground plane.
        var ground = WorldInstance()
        ground.model = float4x4(translation: SIMD3<Float>(0, 0, 0))
            * float4x4(scale: SIMD3<Float>(world.bounds * 2.2, 1, world.bounds * 2.2))
        ground.colour = SIMD4<Float>(world.environment.groundColour, 0)
        ground.params.x = world.environment.groundChecker
        quads.append(ground)

        for o in world.objects {
            var inst = WorldInstance()
            var size = o.size
            if o.kind.isEdible { size *= o.amount * 0.6 + 0.4 }
            inst.model = float4x4(translation: o.position)
                * float4x4(rotationY: o.spin)
                * float4x4(scale: size)
            inst.colour = SIMD4<Float>(o.kind.colour, o.kind.isEdible ? 0.12 : 0)
            switch o.kind.mesh {
            case .cube:   cubes.append(inst)
            case .sphere: spheres.append(inst)
            case .quad:   quads.append(inst)
            }
        }

        if let s = world.swatter {
            var inst = WorldInstance()
            inst.model = float4x4(translation: s.position) * float4x4(scale: s.size)
            inst.colour = SIMD4<Float>(s.kind.colour, 0)
            cubes.append(inst)
        }

        // The fallback fly is built from the same three primitives as the
        // scenery, so it has to join the batches before their ranges are
        // frozen. The real mesh is appended after, because each of its parts
        // is drawn separately against its own slice of the mesh buffers.
        if showFly, flyModel == nil {
            appendPrimitiveFly(body: body, cubes: &cubes, spheres: &spheres, quads: &quads)
        }

        cubeRange = 0..<cubes.count
        sphereRange = cubes.count..<(cubes.count + spheres.count)
        quadRange = (cubes.count + spheres.count)..<(cubes.count + spheres.count + quads.count)
        instances = cubes + spheres + quads

        // The fly's parts come last, each with its own slice of the shared
        // mesh buffers, so they are drawn one per part rather than instanced.
        flyInstanceStart = instances.count
        if showFly { appendFly(body: body, into: &instances) }
        if instances.count > maxInstances {
            instances.removeLast(instances.count - maxInstances)
        }

        let buf = instanceBuffers[frameIndex]
        instances.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress, src.count > 0 else { return }
            buf.contents().copyMemory(from: base,
                                      byteCount: MemoryLayout<WorldInstance>.stride * src.count)
        }
    }

    /// A stand-in fly, assembled from the same three primitives as the
    /// scenery.
    ///
    /// It is not decoration. `FlyModel` is optional because the app should
    /// still run if flymodel.bin is not in the bundle — but "still runs" used
    /// to mean "draw the room and skip the animal", and a missing asset is
    /// exactly the kind of failure that only appears on a device. So the
    /// fallback draws a fly: the right size (2.5 mm body, 4.6 mm span), the
    /// right colour, wings beating at the commanded stroke amplitude. It is
    /// worse than the CT mesh, and it is much better than nothing.
    private func appendPrimitiveFly(body: FlyBody,
                                    cubes: inout [WorldInstance],
                                    spheres: inout [WorldInstance],
                                    quads: inout [WorldInstance]) {
        let p = body.pose

        // Body frame: +Y up, -Z forward, so the fly looks where it is going.
        let frame = float4x4(translation: p.position)
            * float4x4(rotationY: p.heading)
            * float4x4(rotationX: p.pitch)
            * float4x4(rotationZ: p.roll)

        // Plain builders, not captures: a nested function that closes over an
        // `inout` array is the sort of thing the compiler argues about.
        func blob(_ t: SIMD3<Float>, _ s: SIMD3<Float>, _ c: SIMD3<Float>,
                  _ alpha: Float = 0) -> WorldInstance {
            var inst = WorldInstance()
            inst.model = frame * float4x4(translation: t) * float4x4(scale: s)
            inst.colour = SIMD4<Float>(c, alpha)
            return inst
        }
        func box(_ t: SIMD3<Float>, _ s: SIMD3<Float>, _ c: SIMD3<Float>,
                 _ roll: Float = 0) -> WorldInstance {
            var inst = WorldInstance()
            inst.model = frame * float4x4(translation: t)
                * float4x4(rotationZ: roll) * float4x4(scale: s)
            inst.colour = SIMD4<Float>(c, 0)
            return inst
        }
        func plate(_ t: SIMD3<Float>, _ s: SIMD3<Float>, _ c: SIMD3<Float>,
                   _ alpha: Float, _ roll: Float) -> WorldInstance {
            var inst = WorldInstance()
            inst.model = frame * float4x4(translation: t)
                * float4x4(rotationZ: roll) * float4x4(scale: s)
            inst.colour = SIMD4<Float>(c, alpha)
            return inst
        }

        let cuticle = SIMD3<Float>(0.674, 0.35, 0.143)      // MJCF "body"
        let eye     = SIMD3<Float>(0.62, 0.06, 0.02)
        let leg     = SIMD3<Float>(0.20, 0.10, 0.05)
        let membrane = SIMD3<Float>(0.539, 0.686, 0.800)    // MJCF "membrane"

        // 2.5 mm nose to tail, so the fallback sits inside the same collision
        // radius the physics uses.
        spheres.append(blob(SIMD3<Float>(0, 0, -0.010),
                            SIMD3<Float>(0.070, 0.070, 0.090), cuticle))
        spheres.append(blob(SIMD3<Float>(0, -0.005, 0.085),
                            SIMD3<Float>(0.055, 0.055, 0.100), cuticle))
        spheres.append(blob(SIMD3<Float>(0, 0.005, -0.075),
                            SIMD3<Float>(0.045, 0.045, 0.045), cuticle))
        spheres.append(blob(SIMD3<Float>( 0.033, 0.015, -0.095),
                            SIMD3<Float>(0.028, 0.028, 0.028), eye))
        spheres.append(blob(SIMD3<Float>(-0.033, 0.015, -0.095),
                            SIMD3<Float>(0.028, 0.028, 0.028), eye))

        // Wings flap about the body's long axis at the commanded amplitude.
        let sweepL = sin(p.wingPhase) * p.strokeAmplitudeL * 0.5
        let sweepR = sin(p.wingPhase + 0.02) * p.strokeAmplitudeR * 0.5
        quads.append(plate(SIMD3<Float>(-0.130, 0.050, 0.010),
                           SIMD3<Float>(0.220, 1, 0.085), membrane, 0.22, sweepL))
        quads.append(plate(SIMD3<Float>( 0.130, 0.050, 0.010),
                           SIMD3<Float>(0.220, 1, 0.085), membrane, 0.22, -sweepR))

        // Six legs: tucked up in flight, dropped when it lands.
        let tuck = 1 - p.airborne
        for side in [Float(-1), Float(1)] {
            for (i, z) in [Float(-0.045), Float(0.010), Float(0.060)].enumerated() {
                let drop = -0.055 - tuck * 0.045
                let splay = side * (0.045 + tuck * 0.020) * (i == 1 ? 1.3 : 1.0)
                cubes.append(box(SIMD3<Float>(splay, drop, z),
                                 SIMD3<Float>(0.010, 0.090 + tuck * 0.030, 0.010), leg))
            }
        }
    }

    /// Pose the real mesh and append one instance per body part.
    private func appendFly(body: FlyBody, into list: inout [WorldInstance]) {
        guard let model = flyModel else { return }
        let p = body.pose

        // MJCF has +Z up and the fly facing +X; our world is +Y up with the
        // fly facing -Z, so the root carries that change of basis.
        let root = float4x4(translation: p.position)
            * float4x4(rotationY: p.heading)
            * float4x4(rotationX: p.pitch)
            * float4x4(rotationZ: p.roll)
            * float4x4(rotationX: -Float.pi / 2)
            * float4x4(rotationZ: Float.pi / 2)

        model.solve(angles: body.jointAngles, root: root, into: &partMatrices)

        flyInstanceStart = list.count
        for (i, part) in model.parts.enumerated() {
            guard part.indexCount > 0 else { continue }
            var inst = WorldInstance()
            inst.model = partMatrices[i]
            // Wings are translucent membrane; everything else is cuticle.
            let isWing = model.partNames[i].hasPrefix("wing")
            inst.colour = SIMD4<Float>(part.colour.x, part.colour.y, part.colour.z,
                                       isWing ? 0.22 : 0.0)
            inst.params = SIMD4<Float>(0, 1, 0, 0)   // y = 1 marks a mesh part
            list.append(inst)
        }
    }

    // MARK: - Drawing

    private func encode(_ encoder: MTLRenderCommandEncoder, params p: WorldParams) {
        var pp = p
        let buf = instanceBuffers[frameIndex]

        encoder.setRenderPipelineState(skyPipeline)
        encoder.setDepthStencilState(skyDepthState)
        encoder.setFragmentBytes(&pp, length: MemoryLayout<WorldParams>.stride, index: 2)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)
        encoder.setVertexBytes(&pp, length: MemoryLayout<WorldParams>.stride, index: 2)
        encoder.setFragmentBytes(&pp, length: MemoryLayout<WorldParams>.stride, index: 2)

        func draw(_ prim: Primitive, _ range: Range<Int>) {
            guard !range.isEmpty else { return }
            encoder.setVertexBuffer(prim.vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(buf,
                offset: MemoryLayout<WorldInstance>.stride * range.lowerBound, index: 1)
            encoder.drawIndexedPrimitives(type: .triangle,
                                          indexCount: prim.indexCount,
                                          indexType: .uint16,
                                          indexBuffer: prim.indexBuffer,
                                          indexBufferOffset: 0,
                                          instanceCount: range.count)
        }
        draw(cube, cubeRange)
        draw(sphere, sphereRange)
        draw(quad, quadRange)

        // --- the fly -------------------------------------------------------
        // One draw per body part: each has its own index range and its own
        // matrix, so they cannot be batched, but 41 calls is nothing.
        if let model = flyModel, flyInstanceStart < instances.count {
            encoder.setVertexBuffer(model.vertexBuffer, offset: 0, index: 0)
            var slot = flyInstanceStart
            for part in model.parts where part.indexCount > 0 {
                encoder.setVertexBuffer(buf,
                    offset: MemoryLayout<WorldInstance>.stride * slot, index: 1)
                encoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: Int(part.indexCount),
                    indexType: .uint32,
                    indexBuffer: model.indexBuffer,
                    indexBufferOffset: Int(part.indexStart) * MemoryLayout<UInt32>.stride,
                    instanceCount: 1,
                    baseVertex: Int(part.vertexStart),
                    baseInstance: 0)
                slot += 1
                if slot >= instances.count { break }
            }
        }
    }

    /// Main third-person pass into the drawable.
    func drawMain(in view: MTKView, commandBuffer cb: MTLCommandBuffer,
                  viewProj: float4x4, cameraPos: SIMD3<Float>, world: World, time: Float) {
        guard let rpd = view.currentRenderPassDescriptor else { return }
        var p = params
        p.viewProj = viewProj
        p.cameraPos = cameraPos
        p.time = time
        p.checker = world.environment.groundChecker
        p.groundColour = world.environment.groundColour
        p.skyColour = world.environment.skyColour
        // The world is metres deep now; fog is what makes the floor read as
        // an endless horizon instead of a quad edge. ~95% fade by 15-20 m.
        p.fogDensity = world.environment == .garden ? 0.0012 : 0.0025
        guard let e = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        e.label = "worldMain"
        encode(e, params: p)
        e.endEncoding()
    }

    /// Fly's-eye pass: one 128x128 render per compound eye, then the
    /// ommatidial blur of each. The two optical axes sit 67 degrees off the
    /// body axis with 140-degree fields (the flybody eye cameras), covering
    /// ~274 degrees together — the panoramic view a single cyclopean frustum
    /// could only fake.
    func drawEye(commandBuffer cb: MTLCommandBuffer, body: FlyBody, world: World, time: Float) {
        let eyes = body.eyeTransforms
        renderOneEye(cb, target: eyeTextureL, depth: eyeDepthL, eye: eyes.left,
                     world: world, time: time)
        renderOneEye(cb, target: eyeTextureR, depth: eyeDepthR, eye: eyes.right,
                     world: world, time: time)
        blurOneEye(cb, src: eyeTextureL, dst: eyeBlurredL)
        blurOneEye(cb, src: eyeTextureR, dst: eyeBlurredR)
    }

    private func renderOneEye(_ cb: MTLCommandBuffer, target: MTLTexture, depth: MTLTexture,
                              eye: (position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>),
                              world: World, time: Float) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.depthAttachment.texture = depth
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        let proj = float4x4(perspectiveFOV: FlyMorphology.eyeFieldOfView, aspect: 1,
                            near: 0.02, far: 6000)
        let viewM = float4x4(lookAt: eye.position,
                             target: eye.position + eye.forward,
                             up: eye.up)

        var p = params
        p.viewProj = proj * viewM
        p.cameraPos = eye.position
        p.time = time
        p.checker = world.environment.groundChecker
        p.groundColour = world.environment.groundColour
        p.skyColour = world.environment.skyColour
        p.fogDensity = 0.0025

        if let e = cb.makeRenderCommandEncoder(descriptor: rpd) {
            e.label = "flyEye"
            encode(e, params: p)
            e.endEncoding()
        }
    }

    private func blurOneEye(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture) {
        guard let blur = blurPipeline, let e = cb.makeComputeCommandEncoder() else { return }
        e.label = "ommatidia"
        e.setComputePipelineState(blur)
        e.setTexture(src, index: 0)
        e.setTexture(dst, index: 1)
        var f = facetSize
        e.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 0)
        let w = min(blur.threadExecutionWidth, 16)
        let h = max(1, min(blur.maxTotalThreadsPerThreadgroup / w, 16))
        e.dispatchThreadgroups(
            MTLSize(width: (eyeSize + w - 1) / w, height: (eyeSize + h - 1) / h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        e.endEncoding()
    }

    func advanceFrame() { frameIndex = (frameIndex + 1) % instanceBuffers.count }

    // MARK: - Primitive construction

    private func makePrimitive(_ mesh: (verts: [Float], idx: [UInt16])) throws -> Primitive {
        guard let vb = device.makeBuffer(bytes: mesh.verts,
                                         length: MemoryLayout<Float>.stride * mesh.verts.count,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: mesh.idx,
                                         length: MemoryLayout<UInt16>.stride * mesh.idx.count,
                                         options: .storageModeShared) else {
            throw WorldError.allocation("primitive")
        }
        return Primitive(vertexBuffer: vb, indexBuffer: ib, indexCount: mesh.idx.count)
    }

    private func cubeMesh() -> (verts: [Float], idx: [UInt16]) {
        let faces: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = [
            (SIMD3<Float>( 0, 0, 1), SIMD3<Float>( 1, 0, 0), SIMD3<Float>(0, 1, 0)),
            (SIMD3<Float>( 0, 0,-1), SIMD3<Float>(-1, 0, 0), SIMD3<Float>(0, 1, 0)),
            (SIMD3<Float>( 1, 0, 0), SIMD3<Float>( 0, 0,-1), SIMD3<Float>(0, 1, 0)),
            (SIMD3<Float>(-1, 0, 0), SIMD3<Float>( 0, 0, 1), SIMD3<Float>(0, 1, 0)),
            (SIMD3<Float>( 0, 1, 0), SIMD3<Float>( 1, 0, 0), SIMD3<Float>(0, 0,-1)),
            (SIMD3<Float>( 0,-1, 0), SIMD3<Float>( 1, 0, 0), SIMD3<Float>(0, 0, 1)),
        ]
        var verts: [Float] = []
        var idx: [UInt16] = []
        for (n, u, v) in faces {
            let base = UInt16(verts.count / 6)
            for (su, sv) in [(Float(-1), Float(-1)), (Float(1), Float(-1)),
                             (Float(1), Float(1)), (Float(-1), Float(1))] {
                let p = (n + u * su + v * sv) * 0.5
                verts += [p.x, p.y, p.z, n.x, n.y, n.z]
            }
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
        return (verts, idx)
    }

    private func quadMesh() -> (verts: [Float], idx: [UInt16]) {
        var verts: [Float] = []
        for (x, z) in [(Float(-0.5), Float(-0.5)), (Float(0.5), Float(-0.5)),
                       (Float(0.5), Float(0.5)), (Float(-0.5), Float(0.5))] {
            verts += [x, 0, z, 0, 1, 0]
        }
        return (verts, [0, 1, 2, 0, 2, 3])
    }

    private func icosphereMesh(subdivisions: Int) -> (verts: [Float], idx: [UInt16]) {
        let t = (1 + sqrt(Float(5))) / 2
        var points: [SIMD3<Float>] = [
            SIMD3<Float>(-1, t, 0), SIMD3<Float>(1, t, 0),
            SIMD3<Float>(-1, -t, 0), SIMD3<Float>(1, -t, 0),
            SIMD3<Float>(0, -1, t), SIMD3<Float>(0, 1, t),
            SIMD3<Float>(0, -1, -t), SIMD3<Float>(0, 1, -t),
            SIMD3<Float>(t, 0, -1), SIMD3<Float>(t, 0, 1),
            SIMD3<Float>(-t, 0, -1), SIMD3<Float>(-t, 0, 1),
        ].map { normalize($0) }

        var tris: [(Int, Int, Int)] = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        ]

        var cache: [Int64: Int] = [:]
        func midpoint(_ a: Int, _ b: Int) -> Int {
            let key = Int64(min(a, b)) << 32 | Int64(max(a, b))
            if let c = cache[key] { return c }
            points.append(normalize((points[a] + points[b]) * 0.5))
            cache[key] = points.count - 1
            return points.count - 1
        }

        for _ in 0..<subdivisions {
            var next: [(Int, Int, Int)] = []
            next.reserveCapacity(tris.count * 4)
            for (a, b, c) in tris {
                let ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a)
                next += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            }
            tris = next
        }

        var verts: [Float] = []
        verts.reserveCapacity(points.count * 6)
        for p in points {
            let h = p * 0.5
            verts += [h.x, h.y, h.z, p.x, p.y, p.z]
        }
        var idx: [UInt16] = []
        idx.reserveCapacity(tris.count * 3)
        for (a, b, c) in tris { idx += [UInt16(a), UInt16(b), UInt16(c)] }
        return (verts, idx)
    }
}

// MARK: - Small matrix helpers (lookAt / perspective live in Renderer.swift)

extension float4x4 {
    init(translation t: SIMD3<Float>) {
        self.init(SIMD4<Float>(1, 0, 0, 0),
                  SIMD4<Float>(0, 1, 0, 0),
                  SIMD4<Float>(0, 0, 1, 0),
                  SIMD4<Float>(t.x, t.y, t.z, 1))
    }
    init(scale s: SIMD3<Float>) {
        self.init(SIMD4<Float>(s.x, 0, 0, 0),
                  SIMD4<Float>(0, s.y, 0, 0),
                  SIMD4<Float>(0, 0, s.z, 0),
                  SIMD4<Float>(0, 0, 0, 1))
    }
    init(rotationX a: Float) {
        let c = cos(a), s = sin(a)
        self.init(SIMD4<Float>(1, 0, 0, 0),
                  SIMD4<Float>(0, c, s, 0),
                  SIMD4<Float>(0, -s, c, 0),
                  SIMD4<Float>(0, 0, 0, 1))
    }
    init(rotationY a: Float) {
        let c = cos(a), s = sin(a)
        self.init(SIMD4<Float>(c, 0, -s, 0),
                  SIMD4<Float>(0, 1, 0, 0),
                  SIMD4<Float>(s, 0, c, 0),
                  SIMD4<Float>(0, 0, 0, 1))
    }
    init(rotationZ a: Float) {
        let c = cos(a), s = sin(a)
        self.init(SIMD4<Float>(c, s, 0, 0),
                  SIMD4<Float>(-s, c, 0, 0),
                  SIMD4<Float>(0, 0, 1, 0),
                  SIMD4<Float>(0, 0, 0, 1))
    }
}
