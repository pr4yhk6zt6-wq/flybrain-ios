//
//  FlyWorld.swift
//  The animal, and the recording of it, as something a phone can hold.
//
//  The geometry is the real Janelia / Google DeepMind *flybody* model
//  (Vaxenburg et al.): 85 meshes scanned from a real animal.
//
//  The poses are no longer a recording. This class carries the meshes, the
//  floor and the lights; `apply(live:)` takes the state of `FlyLiveBody` —
//  the solver running on this device — and poses every node from it, once per
//  display frame. `apply(frame:)` still exists because the mesh geometry and
//  the recorded manifest come out of the same packer, but nothing drives it:
//  the Map screen is not a playback any more.
//
//  The mesh geometry (world.json + fly.bin) and the body asset
//  (fly_body.json, for the physics) are both generated. Run
//  `tools/step4_world.py`, `tools/build_body.py`, then the two pack steps
//  before building, or the world opens with an explanation instead of a fly.
//

import Foundation
import UIKit
import SceneKit

// MARK: - The manifest

/// What `tools/step4_world.py` writes. Mirrors `world/world.json`.
struct WorldManifest: Decodable {
    let legs: [String]
    let floor_z: Double
    let geoms: [WorldGeom]
    let body_geometry: WorldBodyGeometry
    let frames: WorldFrames
    let behaviour: WorldBehaviour?
    let weight: Double
    /// The camera's limits, measured from the animal itself by
    /// `tools/measure_view.py` and written into `world.json`. Optional
    /// because a world written before that tool existed does not carry it, and
    /// such a world gets the conservative fallbacks in `FlyWorld.viewLimits`.
    let view: WorldView?
    /// The smell in the world. See `OdorField` and docs/ASSUMPTIONS.md #26–#29.
    let odor: OdorSpec?
    /// The haltere pair's geometry and constants, written by
    /// `tools/haltere_gyro.py --write`. Optional for the same reason `view` is:
    /// a world that predates the block still loads, and `HaltereGyro` keeps the
    /// defaults the tool measured with.
    let haltere: HaltereSpec?

    /// The odor field's parameters, written by `tools/odor_field.py --write`.
    /// Every field is optional so that a world which predates the block still
    /// loads: `OdorField` applies the tool's own defaults for anything absent.
    struct OdorSpec: Decodable {
        let source_cm: [Double]?
        let wind_cms: [Double]?
        let c0: Double?
        let decay_cm: Double?
        let sigma0_cm: Double?
        let widening: Double?
        let s0_cm: Double?
        let duty: Double?
        let gust_hz: Double?
        let wavenumber: Double?
        let baseline_drive: Double?
    }

    struct WorldView: Decodable {
        let fov_y: Double
        let pane_fov_y: Double
        let body_radius_cm: Double
        let all_radius_cm: Double
        let min_distance_cm: Double
        let home_distance_cm: Double
        let max_distance_cm: Double
        let pane_distance_cm: Double
    }

    struct WorldGeom: Decodable {
        let name: String
        let rgba: [Double]
        /// The mesh scale, which the model needs: the scanned meshes are
        /// about twenty times life size and MuJoCo's geoms carry the factor.
        let size: [Double]
        /// Absent for the parts that are not meshes — the floor, which this
        /// draws itself rather than reading from the file.
        let mesh: Int?
        /// This part's own row in the recorded pose track, written by
        /// `tools/step4_world.py`. Not the node's position in this array: the
        /// floor is a part with no mesh, so anything that counts for itself
        /// while it skips the floor is one ahead of the recording for the
        /// whole animal. Absent only in a manifest written before the field
        /// existed, in which case the counting fallback is what that manifest
        /// was made for.
        let part: Int?
    }

    struct WorldBodyGeometry: Decodable {
        let meshes: [WorldMesh]
        let n_vertex: Int
        let n_face: Int

        struct WorldMesh: Decodable {
            let name: String
            let vert: [Int]        // [offset, count]
            let face: [Int]        // [offset, count]
        }
    }

    struct WorldFrames: Decodable {
        let n: Int
        let parts: Int
        let stride_ms: Int
    }

    struct WorldBehaviour: Decodable {
        let net_displacement: Double?
        let path_length: Double?
        let body_lengths_per_second: Double?
        let foot_contacts_mean: Double?
        let fell_over: Bool?
    }
}

enum WorldLoadError: LocalizedError {
    case missing(String)
    case corrupt(String)

    var errorDescription: String? {
        switch self {
        case .missing(let f):
            return "\(f) is not in the app bundle. Run tools/step4_world.py "
                 + "and tools/pack_world.py, then build again."
        case .corrupt(let why):
            return "The world recording is unreadable: \(why)"
        }
    }
}

// MARK: - The world

/// The animal's body, and every pose of it that was recorded.
///
/// Built once, on a background thread, then handed to the UI: `@unchecked
/// Sendable` records that the handover is a transfer of ownership, not
/// sharing.
final class FlyWorld: @unchecked Sendable {
    let manifest: WorldManifest

    /// The scene: the animal, a floor, three lights. Nothing else.
    let scene = SCNScene()

    /// Where the camera may sit, and where it starts.
    ///
    /// These are not preferences and they are not constants in this file any
    /// more. IMG_2715 is the screenshot of what the old `minDistance = 0.22`
    /// did on a phone: at 0.22 cm a 38° lens sees a window 0.70 mm wide, the
    /// animal's body is 1.24 mm, and 29.7% of the body's vertices were inside
    /// the frustum — a thorax filling the screen with legs and wings crossing
    /// it at angles, which reads as "the model is in pieces" when the model is
    /// fine. `tools/measure_view.py` measures the animal out of this same
    /// asset and writes the distances into `world.json`; `tools/audit_view.py
    /// --gate` fails the build if any camera the app can reach crops the body.
    /// The fallbacks are the v0.7 numbers, so an old world still loads.
    struct ViewLimits {
        let min: Double
        let home: Double
        let max: Double
        let pane: Double
        let fovY: Double
        let paneFovY: Double
    }

    let viewLimits: ViewLimits

    /// The smell, in the same frame as the meshes. The animal samples it at its
    /// own antennae; `WorldModel` is what actually calls it, once per frame.
    let odorField: OdorField
    /// What the gyro is built from (`WorldView.attachCord`). Kept as the spec
    /// rather than as a `HaltereGyro`, because the cord owns the filter state.
    let haltereSpec: HaltereSpec?

    /// One node per mesh-bearing part, in the order they appear in the
    /// recording, so a frame index can pose them directly.
    private(set) var nodes: [SCNNode] = []
    /// Which row of the recorded pose track each node reads, index-aligned
    /// with `nodes`. The manifest's `part`, or — for a manifest written
    /// before that field existed — the same count this file used to keep.
    private var partOf: [Int] = []
    /// How many vertices each node's mesh carries, index-aligned with `nodes`.
    /// The rig centres itself on where the geometry is: the legs, wings and
    /// bristles are many parts with small meshes, and a mean that counts them
    /// equally sits off the animal by about a third of a body radius.
    private var weightOf: [Float] = []

    private var framesData = Data()
    /// node index -> index into the body asset's `visual` array, built once
    /// by name on the first live pose.
    private var liveIndex: [Int]?
    private let floorZ: Float
    private var frameCount: Int
    private let partCount: Int
    private let frameHz: Int

    init(bundle: Bundle = .main) throws {
        func locate(_ name: String, _ ext: String) -> URL? {
            bundle.url(forResource: name, withExtension: ext)
                ?? bundle.url(forResource: name, withExtension: ext,
                              subdirectory: "World")
                ?? bundle.url(forResource: "World/\(name)", withExtension: ext)
        }

        guard let jsonURL = locate("world", "json") else {
            throw WorldLoadError.missing("world.json")
        }
        guard let bodyURL = locate("fly", "bin") else {
            throw WorldLoadError.missing("fly.bin")
        }
        // The recorded pose track is optional now: the Map screen steps the
        // solver instead of playing it back, and a build without the seven
        // megabyte recording is a smaller build, not a broken one.
        let framesURL = locate("frames", "bin")

        let manifest = try JSONDecoder().decode(
            WorldManifest.self, from: try Data(contentsOf: jsonURL))
        let bodyData = try Data(contentsOf: bodyURL, options: .mappedIfSafe)
        if let framesURL {
            framesData = (try? Data(contentsOf: framesURL, options: .mappedIfSafe)) ?? Data()
        }

        let nV = manifest.body_geometry.n_vertex
        let nF = manifest.body_geometry.n_face
        let expectBody = (nV * 3 + nV * 3 + nF * 3) * 4
        guard bodyData.count >= expectBody else {
            throw WorldLoadError.corrupt("fly.bin is \(bodyData.count) bytes, "
                                         + "expected \(expectBody)")
        }
        let stride = manifest.frames.parts * 7
        let expectFrames = manifest.frames.n * stride * 4
        if !framesData.isEmpty && framesData.count < expectFrames {
            throw WorldLoadError.corrupt("frames.bin is "
                + "\(framesData.count) bytes, expected \(expectFrames)")
        }


        self.manifest = manifest
        if let v = manifest.view {
            self.viewLimits = ViewLimits(min: v.min_distance_cm,
                                         home: v.home_distance_cm,
                                         max: v.max_distance_cm,
                                         pane: v.pane_distance_cm,
                                         fovY: v.fov_y,
                                         paneFovY: v.pane_fov_y)
        } else {
            self.viewLimits = ViewLimits(min: 0.82, home: 1.0, max: 2.5,
                                         pane: 0.306, fovY: 38, paneFovY: 50)
        }
        self.odorField = OdorField(spec: manifest.odor)
        self.haltereSpec = manifest.haltere
        self.floorZ = Float(manifest.floor_z)
        self.frameCount = framesData.isEmpty ? 0 : manifest.frames.n
        self.partCount = manifest.frames.parts
        self.frameHz = max(1, 1000 / max(1, manifest.frames.stride_ms))

        let geometries = FlyWorld.buildGeometries(manifest: manifest,
                                                  data: bodyData)

        let animal = SCNNode()
        animal.name = "animal"
        scene.rootNode.addChildNode(animal)

        for geom in manifest.geoms {
            guard let m = geom.mesh, m < geometries.count else { continue }
            let node = SCNNode(geometry: geometries[m])
            weightOf.append(Float(manifest.body_geometry.meshes[m].vert[1]))
            node.name = geom.name
            // The mesh vertices are in model centimetres; the geom's world
            // transform comes from the pose. Any non-unit scale here would be
            // the geom_size bug in docs/AUDIT.md §0, so pin it to 1.
            node.scale = SCNVector3(1, 1, 1)
            // The colour is on the part's material, not on the geom: flybody
            // names its materials — body, red, ocelli, black, brown,
            // membrane — and the geom's own rgba is left at default grey.
            let material = SCNMaterial()
            material.lightingModel = .lambert
            let c = geom.rgba
            let r = c.count > 0 ? CGFloat(c[0]) : 0.6
            let g = c.count > 1 ? CGFloat(c[1]) : 0.6
            let b = c.count > 2 ? CGFloat(c[2]) : 0.6
            let a = c.count > 3 ? CGFloat(c[3]) : 1.0
            material.diffuse.contents = UIColor(red: r, green: g, blue: b,
                                                alpha: a)
            material.isDoubleSided = true
            if a < 0.99 {                        // the wings are translucent
                material.transparency = a
                material.writesToDepthBuffer = false
            }
            node.geometry?.materials = [material]
            animal.addChildNode(node)
            nodes.append(node)
            partOf.append(geom.part ?? nodes.count - 1)
        }

        FlyWorld.dress(scene: scene, floorZ: floorZ)
        FlyWorld.markOdor(in: scene, at: odorField.source, z: floorZ)
        apply(frame: 0)
    }

    // MARK: - Geometry

    /// One SCNGeometry per mesh, built once from the packed binary.
    private static func buildGeometries(manifest: WorldManifest,
                                        data: Data) -> [SCNGeometry] {
        let nV = manifest.body_geometry.n_vertex
        return data.withUnsafeBytes { raw -> [SCNGeometry] in
            guard let base = raw.baseAddress else { return [] }
            let f = base.assumingMemoryBound(to: Float32.self)
            let idxIn = base.assumingMemoryBound(to: Int32.self)
            return manifest.body_geometry.meshes.map { def -> SCNGeometry in
                let vo = def.vert[0], vn = def.vert[1]
                let fo = def.face[0], fn = def.face[1]

                var verts = [Float32](repeating: 0, count: vn * 3)
                var norms = [Float32](repeating: 0, count: vn * 3)
                for k in 0..<(vn * 3) {
                    verts[k] = f[vo * 3 + k]
                    norms[k] = f[nV * 3 + vo * 3 + k]
                }
                // The file indexes with Int32 into the whole model; SceneKit
                // indexes with UInt32 into this mesh.
                var idx = [UInt32]()
                idx.reserveCapacity(fn * 3)
                for k in 0..<(fn * 3) {
                    let global = Int(idxIn[nV * 6 + fo * 3 + k])
                    idx.append(UInt32(truncatingIfNeeded: global - vo))
                }

                let vs = SCNGeometrySource(
                    data: Data(bytes: &verts, count: verts.count * 4),
                    semantic: .vertex, vectorCount: vn,
                    usesFloatComponents: true, componentsPerVector: 3,
                    bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
                let ns = SCNGeometrySource(
                    data: Data(bytes: &norms, count: norms.count * 4),
                    semantic: .normal, vectorCount: vn,
                    usesFloatComponents: true, componentsPerVector: 3,
                    bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
                let el = SCNGeometryElement(
                    data: Data(bytes: &idx, count: idx.count * 4),
                    primitiveType: .triangles, primitiveCount: fn,
                    bytesPerIndex: 4)
                return SCNGeometry(sources: [vs, ns], elements: [el])
            }
        }
    }

    /// A grid on the ground, so that distance is visible. Drawn as lines
    /// rather than a texture: the spacing is then exactly what it says.
    private static func grid(extent: Float, step: Float,
                             z: Float) -> SCNNode {
        let n = Int(extent / step)
        var pts: [Float] = []
        for i in -n...n {
            let t = Float(i) * step
            pts.append(contentsOf: [-extent, t, z, extent, t, z])
            pts.append(contentsOf: [t, -extent, z, t, extent, z])
        }
        var idx = [UInt32]()
        idx.reserveCapacity(pts.count / 3)
        for k in 0..<(pts.count / 3) { idx.append(UInt32(k)) }

        let source = SCNGeometrySource(
            data: Data(bytes: &pts, count: pts.count * 4),
            semantic: .vertex, vectorCount: pts.count / 3,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(
            data: Data(bytes: &idx, count: idx.count * 4),
            primitiveType: .line, primitiveCount: idx.count / 2,
            bytesPerIndex: 4)

        // Dark enough to read against the 0.82 floor and the 0.965 sky: the
        // grid is how a viewer knows the animal is standing on something.
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(white: 0.60, alpha: 1)

        let node = SCNNode(geometry: SCNGeometry(sources: [source],
                                                 elements: [element]))
        node.name = "grid"
        node.geometry?.materials = [material]
        return node
    }

    /// The floor it stands on, a grid, and three lights. Nothing else is in
    /// the world, on purpose: the animal is the subject.
    ///
    /// The ground is deliberately darker than the sky. It used to be 0.93 — a
    /// white floor with 5% of mirror on it — under a 0.965 background, and the
    /// two are the same colour: as the camera orbits, the floor fills the
    /// screen at some angles and the background fills it at others, so the
    /// screen swaps between grey and white and the ground stops reading as
    /// ground. That is the other thing IMG_2715's screenshot was asked about.
    private static func dress(scene: SCNScene, floorZ: Float) {
        let floor = SCNFloor()
        floor.reflectivity = 0                     // no mirror: a floor, not a lake
        floor.firstMaterial?.diffuse.contents = UIColor(white: 0.82, alpha: 1)
        floor.firstMaterial?.roughness.contents = 1.0
        let floorNode = SCNNode(geometry: floor)
        floorNode.position = SCNVector3(0, 0, floorZ)
        scene.rootNode.addChildNode(floorNode)
        // A grid, so the ground reads as ground and you can see the animal
        // move across it. Squares of a tenth of a body length (0.027 of the
        // 0.27 units the animal is long).
        let grid = FlyWorld.grid(extent: 0.6, step: 0.027, z: floorZ)
        scene.rootNode.addChildNode(grid)

        let key = SCNNode()
        key.light = SCNLight()
        key.light?.type = .directional
        key.light?.intensity = 1300
        key.light?.color = UIColor.white
        key.position = SCNVector3(0.35, -0.6, 0.9)
        scene.rootNode.addChildNode(key)

        let fill = SCNNode()
        fill.light = SCNLight()
        fill.light?.type = .directional
        fill.light?.intensity = 380
        fill.light?.color = UIColor(red: 0.82, green: 0.88, blue: 1.0,
                                    alpha: 1)
        fill.position = SCNVector3(-0.5, 0.5, 0.2)
        scene.rootNode.addChildNode(fill)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 400
        ambient.light?.color = UIColor.white
        scene.rootNode.addChildNode(ambient)

        scene.background.contents = UIColor(red: 0.965, green: 0.957,
                                            blue: 0.949, alpha: 1)
    }

    // MARK: - Posing

    /// Pose the animal at a (possibly fractional) frame index.
    func apply(frame: Double) {
        guard frameCount > 0, !nodes.isEmpty else { return }
        let clamped = min(max(frame, 0), Double(frameCount - 1))
        let i0 = Int(clamped.rounded(.down))
        let i1 = min(i0 + 1, frameCount - 1)
        let t = Float(clamped - Double(i0))
        let stride = partCount * 7

        framesData.withUnsafeBytes { raw in
            let s = raw.bindMemory(to: Float32.self)
            guard s.count >= frameCount * stride else { return }
            let b0 = i0 * stride, b1 = i1 * stride
            for (k, node) in nodes.enumerated() {
                // The part this node follows, not the node's own index: see
                // `partOf`.
                let part = k < partOf.count ? partOf[k] : k
                guard part >= 0, part < partCount else { continue }
                let p = b0 + part * 7, q = b1 + part * 7
                node.position = SCNVector3(
                    x: mix(s[p], s[q], t),
                    y: mix(s[p + 1], s[q + 1], t),
                    z: mix(s[p + 2], s[q + 2], t))
                // stored (w, x, y, z); SceneKit wants (x, y, z, w)
                let orient = slerp(
                    (s[p + 3], s[p + 4], s[p + 5], s[p + 6]),
                    (s[q + 3], s[q + 4], s[q + 5], s[q + 6]), t)
                node.orientation = SCNVector4(orient.1, orient.2, orient.3,
                                              orient.0)
                // No scale here, and that is deliberate: geom_size is the
                // mesh's bounding box half-extent, not a scale factor (the
                // thorax reports [0.0439, 0.0584, 0.0601] against a mesh of
                // 0.088 x 0.105 x 0.116 — the same numbers, halved), and
                // `fly.bin` is packed from the compiled model at life size.
                // Multiplying by it draws the animal at 4–9 % of itself;
                // see docs/AUDIT.md §0 for the projection that proves it.
            }
        }
    }

    /// Pose the animal from the live solver.
    ///
    /// One node per visual geom, keyed by name — the mesh packer and the body
    /// asset both come from the same MJCF geoms, so the names are the same
    /// strings, and a name that is missing is a mesh that has no physics to
    /// follow rather than a silent wrong pose on the wrong body.
    func apply(live: FlyLiveBody) {
        guard !nodes.isEmpty else { return }
        let (positions, rotations) = live.visualWorld()
        if liveIndex == nil {
            var byName: [String: Int] = [:]
            for (i, name) in live.visualName.enumerated() { byName[name] = i }
            liveIndex = nodes.map { node in
                guard let name = node.name else { return -1 }
                return byName[name] ?? -1
            }
        }
        guard let map = liveIndex else { return }
        for (k, node) in nodes.enumerated() {
            let i = map[k]
            guard i >= 0, i < positions.count else { continue }
            let p = positions[i]
            node.position = SCNVector3(Float(p.x), Float(p.y), Float(p.z))
            // `quatFromMat` (FlyDynamics.swift) is the one conversion in the
            // app: it is the inverse of `quatToMat`, it is tested, and the
            // local copy this file used to carry returned the conjugate.
            let q = quatFromMat(rotations[i])
            node.orientation = SCNVector4(Float(q.x), Float(q.y), Float(q.z),
                                          Float(q.w))
            // Scale stays what node creation pinned it to (1): the solver
            // gives a pose, and the vertices are already life size.
        }
    }

    /// Where a named part is, in world coordinates — the same frame the odor
    /// field and the camera limits are written in. `head` is where the antennae
    /// are, and the antennae are what smell.
    /// Where the animal smells from: the antennae, which the body model does not
    /// name — it names the head they sit on. Which of the head's parts exists
    /// depends on the body file (flybody's fused model calls the visual head
    /// mesh `head_body`, the un-fused one `head`), so the names are tried in
    /// order and the one that was used is kept.
    static let odorParts = ["head", "head_body", "head_red"]

    /// The part the smell is sampled at, or nil if this body has no head in it.
    private(set) var odorPart: String?

    /// The point the odor field is read at. `WorldModel.sampleOdor` uses this
    /// rather than a name, so a build whose body file uses a different name for
    /// its head reports that it cannot smell instead of reading a silent zero —
    /// a zero that is exactly what clean air reads as.
    func odorSamplePosition() -> SIMD3<Double>? {
        for name in FlyWorld.odorParts {
            if let p = position(of: name) {
                odorPart = name
                return p
            }
        }
        odorPart = nil
        return nil
    }

    func position(of name: String) -> SIMD3<Double>? {
        guard let node = nodes.first(where: { $0.name == name }) else { return nil }
        let p = node.simdPosition
        return SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z))
    }

    /// A disc on the floor where the odor comes from, so the smell is somewhere
    /// you can see rather than a number in a corner of the screen.
    private static func markOdor(in scene: SCNScene, at source: SIMD3<Double>,
                                z: Float) {
        let disc = SCNCylinder(radius: 0.045, height: 0.0008)
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(red: 0.36, green: 0.62, blue: 0.38,
                                            alpha: 1)
        disc.materials = [material]
        let node = SCNNode(geometry: disc)
        node.name = "odor_source"
        node.position = SCNVector3(Float(source.x), Float(source.y), z + 0.0004)
        scene.rootNode.addChildNode(node)
    }

    /// Where the animal is, so a camera can keep it in frame.
    func centre() -> SCNVector3 {
        guard !nodes.isEmpty else { return SCNVector3(0, 0, floorZ) }
        var x: Float = 0, y: Float = 0, z: Float = 0, w: Float = 0
        for (k, node) in nodes.enumerated() {
            let weight = k < weightOf.count ? weightOf[k] : 1
            x += node.position.x * weight
            y += node.position.y * weight
            z += node.position.z * weight
            w += weight
        }
        guard w > 0 else { return SCNVector3(0, 0, floorZ) }
        return SCNVector3(x / w, y / w, z / w)
    }

    /// Frames in the recording, if one was packed. Zero means the screen is
    /// running the solver, which is the only way it runs now.
    var count: Int { frameCount }
    var hz: Int { frameHz }
    var hasRecording: Bool { frameCount > 0 && !framesData.isEmpty }
    var floor: Float { floorZ }
    /// Meshes actually drawn — the parts that carry geometry.
    var meshCount: Int { nodes.count }
    var faceCount: Int { manifest.body_geometry.n_face }
}

// MARK: - Small maths

/// The quaternion of a rotation matrix lives in FlyDynamics.swift beside
/// `quatToMat` — one implementation, tested where the tests can reach it — and
/// the interpolation below is what the (unused) recording path needs.

@inline(__always) private func mix(_ a: Float32, _ b: Float32,
                                   _ t: Float) -> Float {
    a + (b - a) * t
}

/// Spherical interpolation between two quaternions held as (w, x, y, z).
private func slerp(_ a: (Float, Float, Float, Float),
                   _ b: (Float, Float, Float, Float),
                   _ t: Float) -> (Float, Float, Float, Float) {
    var bx = b.1, by = b.2, bz = b.3, bw = b.0
    var dot = a.0 * bw + a.1 * bx + a.2 * by + a.3 * bz
    if dot < 0 {                       // take the short way round
        bx = -bx; by = -by; bz = -bz; bw = -bw
        dot = -dot
    }
    if dot > 0.9995 {
        let w = a.0 + (bw - a.0) * t, x = a.1 + (bx - a.1) * t
        let y = a.2 + (by - a.2) * t, z = a.3 + (bz - a.3) * t
        let n = sqrtf(w * w + x * x + y * y + z * z)
        return n > 0 ? (w / n, x / n, y / n, z / n) : (1, 0, 0, 0)
    }
    let theta = acos(min(max(dot, -1), 1))
    let sinTheta = sin(theta)
    guard sinTheta > 1e-6 else { return a }
    let wa = sin((1 - t) * theta) / sinTheta
    let wb = sin(t * theta) / sinTheta
    let w = a.0 * wa + bw * wb, x = a.1 * wa + bx * wb
    let y = a.2 * wa + by * wb, z = a.3 * wa + bz * wb
    let n = sqrtf(w * w + x * x + y * y + z * z)
    return n > 0 ? (w / n, x / n, y / n, z / n) : (1, 0, 0, 0)
}
