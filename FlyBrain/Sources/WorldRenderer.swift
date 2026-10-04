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
    private let maxInstances = 512

    /// The fly's-eye render target, and the ommatidially-blurred version the
    /// photoreceptor kernel actually samples.
    private(set) var eyeTexture: MTLTexture!
    private(set) var eyeBlurred: MTLTexture!
    private var eyeDepth: MTLTexture!
    let eyeSize = 128
    /// Each ommatidium covers this many pixels of the 128x128 render.
    var facetSize: UInt32 = 6

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

        try makeEyeTargets(colorFormat: colorFormat)
    }

    private func makeEyeTargets(colorFormat: MTLPixelFormat) throws {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorFormat, width: eyeSize, height: eyeSize, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else { throw WorldError.allocation("eye") }
        t.label = "flyEye"
        eyeTexture = t

        let b = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: eyeSize, height: eyeSize, mipmapped: false)
        b.usage = [.shaderRead, .shaderWrite]
        b.storageMode = .private
        guard let bt = device.makeTexture(descriptor: b) else { throw WorldError.allocation("eyeBlur") }
        bt.label = "flyEyeOmmatidia"
        eyeBlurred = bt

        let dd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: eyeSize, height: eyeSize, mipmapped: false)
        dd.usage = [.renderTarget]
        dd.storageMode = .private
        guard let dt = device.makeTexture(descriptor: dd) else { throw WorldError.allocation("eyeDepth") }
        eyeDepth = dt
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

        if showFly {
            appendFly(body: body, cubes: &cubes, spheres: &spheres, quads: &quads)
        }

        cubeRange = 0..<cubes.count
        sphereRange = cubes.count..<(cubes.count + spheres.count)
        quadRange = (cubes.count + spheres.count)..<(cubes.count + spheres.count + quads.count)
        instances = cubes + spheres + quads
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

    /// A fly assembled out of primitives: thorax, abdomen, head, two eyes, two
    /// wings, six legs. Crude, but it reads as a fly at chase-camera distance
    /// and costs three draw calls total.
    private func appendFly(body: FlyBody,
                           cubes: inout [WorldInstance],
                           spheres: inout [WorldInstance],
                           quads: inout [WorldInstance]) {
        let p = body.pose
        let root = float4x4(translation: p.position)
            * float4x4(rotationY: p.heading)
            * float4x4(rotationX: p.pitch)
            * float4x4(rotationZ: p.roll)
            * float4x4(scale: SIMD3<Float>(repeating: 0.11))

        func add(_ list: inout [WorldInstance], _ local: float4x4,
                 _ colour: SIMD3<Float>, _ emissive: Float = 0) {
            var inst = WorldInstance()
            inst.model = root * local
            inst.colour = SIMD4<Float>(colour, emissive)
            list.append(inst)
        }

        let bodyDark = SIMD3<Float>(0.18, 0.17, 0.16)
        let eyeRed = SIMD3<Float>(0.72, 0.12, 0.10)

        // thorax and abdomen
        add(&spheres, float4x4(translation: SIMD3<Float>(0, 0, 0.1))
                    * float4x4(scale: SIMD3<Float>(0.5, 0.45, 0.6)), bodyDark)
        add(&spheres, float4x4(translation: SIMD3<Float>(0, -0.02, 0.85))
                    * float4x4(scale: SIMD3<Float>(0.42, 0.38, 0.85)),
            SIMD3<Float>(0.22, 0.20, 0.14))

        // head, turned by the neck motor neurons
        let head = float4x4(translation: SIMD3<Float>(0, 0.05, -0.62))
                 * float4x4(rotationY: p.headYaw)
        add(&spheres, head * float4x4(scale: SIMD3<Float>(0.42, 0.40, 0.38)), bodyDark)
        for s in [Float(-1), Float(1)] {
            add(&spheres, head * float4x4(translation: SIMD3<Float>(0.24 * s, 0.06, -0.06))
                               * float4x4(scale: SIMD3<Float>(0.26, 0.30, 0.28)),
                eyeRed, 0.18)
        }
        if p.proboscisExtension > 0.02 {
            let e = p.proboscisExtension
            add(&cubes, head * float4x4(translation: SIMD3<Float>(0, -0.2, -0.2 - 0.18 * e))
                             * float4x4(scale: SIMD3<Float>(0.09, 0.09, 0.4 * e)),
                SIMD3<Float>(0.35, 0.28, 0.22))
        }

        // wings — amplitude from the wing power groups, tilt from steering
        let beat = sin(p.wingPhase)
        for s in [Float(-1), Float(1)] {
            let asym = (s < 0) ? -p.wingAsymmetry : p.wingAsymmetry
            let amp = max(0.12, p.wingAmplitude) * (1 + asym * 0.35)
            let flap = beat * amp
            let local = float4x4(translation: SIMD3<Float>(0.18 * s, 0.22, 0.2))
                      * float4x4(rotationZ: s * (0.5 + flap * 0.8))
                      * float4x4(rotationY: s * -0.35)
                      * float4x4(translation: SIMD3<Float>(0.6 * s, 0, 0.35))
                      * float4x4(scale: SIMD3<Float>(1.3, 1, 0.5))
            add(&quads, local, SIMD3<Float>(0.75, 0.80, 0.88), 0.05)
        }

        // six legs, tripod gait: the two groups move half a cycle apart
        let legZ: [Float] = [-0.25, 0.15, 0.5]
        for pair in 0..<3 {
            for s in [Float(-1), Float(1)] {
                let tripod = (pair + (s < 0 ? 0 : 1)) % 2
                let phase = p.gaitPhase + (tripod == 0 ? 0 : Float.pi)
                let grounded = 1 - p.airborne
                let swing = sin(phase) * grounded * 0.5
                let lift = max(0, sin(phase)) * grounded * 0.18
                let local = float4x4(translation: SIMD3<Float>(0.28 * s, -0.1, legZ[pair]))
                          * float4x4(rotationZ: s * (0.8 + lift))
                          * float4x4(rotationX: swing)
                          * float4x4(translation: SIMD3<Float>(0.35 * s, -0.3, 0))
                          * float4x4(scale: SIMD3<Float>(0.7, 0.07, 0.07))
                add(&cubes, local, SIMD3<Float>(0.14, 0.13, 0.12))
            }
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
        p.fogDensity = world.environment == .garden ? 0.012 : 0.03
        guard let e = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        e.label = "worldMain"
        encode(e, params: p)
        e.endEncoding()
    }

    /// Fly's-eye pass into the offscreen texture, then the ommatidial blur.
    func drawEye(commandBuffer cb: MTLCommandBuffer, body: FlyBody, world: World, time: Float) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = eyeTexture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.depthAttachment.texture = eyeDepth
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        let eye = body.eyeTransform
        // A fly's field of view is nearly panoramic. A single rectilinear
        // projection cannot do 270 degrees, so we render a wide 140-degree
        // frustum and accept the loss at the periphery.
        let proj = float4x4(perspectiveFOV: 140 * .pi / 180, aspect: 1, near: 0.02, far: 40)
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
        p.fogDensity = 0.02

        if let e = cb.makeRenderCommandEncoder(descriptor: rpd) {
            e.label = "flyEye"
            encode(e, params: p)
            e.endEncoding()
        }

        if let blur = blurPipeline, let e = cb.makeComputeCommandEncoder() {
            e.label = "ommatidia"
            e.setComputePipelineState(blur)
            e.setTexture(eyeTexture, index: 0)
            e.setTexture(eyeBlurred, index: 1)
            var f = facetSize
            e.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 0)
            let w = min(blur.threadExecutionWidth, 16)
            let h = max(1, min(blur.maxTotalThreadsPerThreadgroup / w, 16))
            e.dispatchThreadgroups(
                MTLSize(width: (eyeSize + w - 1) / w, height: (eyeSize + h - 1) / h, depth: 1),
                threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
            e.endEncoding()
        }
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
