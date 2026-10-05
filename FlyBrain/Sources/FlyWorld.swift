//
//  FlyWorld.swift
//  The animal, and the recording of it, as something a phone can hold.
//
//  The geometry is the real Janelia / Google DeepMind *flybody* model
//  (Vaxenburg et al.): 85 meshes scanned from a real animal. The poses are
//  not an animation — they are a recording of the simulation in
//  tools/step4_world.py, in which each millisecond every leg's sense organ
//  is told where the joint is and how much load the leg carries, a cord
//  taken out of the BANC connectome runs, and its motor pools command the
//  joints. Nothing is keyframed here; the only interpolation is the
//  smoothing between two recorded frames.
//
//  The three files this reads are generated. Run `tools/step4_world.py` then
//  `tools/pack_world.py` before building, or the world opens with an
//  explanation instead of a fly.
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

    struct WorldGeom: Decodable {
        let name: String
        let rgba: [Double]
        let size: [Double]
        /// Absent for the parts that are not meshes — the floor, which this
        /// draws itself rather than reading from the file.
        let mesh: Int?
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

    /// One node per mesh-bearing part, in the order they appear in the
    /// recording, so a frame index can pose them directly.
    private(set) var nodes: [SCNNode] = []
    /// The same parts' descriptions, index-aligned with `nodes`.
    private var parts: [WorldManifest.WorldGeom] = []

    private var framesData = Data()
    private let floorZ: Float
    private let frameCount: Int
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
        guard let framesURL = locate("frames", "bin") else {
            throw WorldLoadError.missing("frames.bin")
        }

        let manifest = try JSONDecoder().decode(
            WorldManifest.self, from: try Data(contentsOf: jsonURL))
        let bodyData = try Data(contentsOf: bodyURL, options: .mappedIfSafe)
        framesData = try Data(contentsOf: framesURL, options: .mappedIfSafe)

        let nV = manifest.body_geometry.n_vertex
        let nF = manifest.body_geometry.n_face
        let expectBody = (nV * 3 + nV * 3 + nF * 3) * 4
        guard bodyData.count >= expectBody else {
            throw WorldLoadError.corrupt("fly.bin is \(bodyData.count) bytes, "
                                         + "expected \(expectBody)")
        }
        let stride = manifest.frames.parts * 7
        let expectFrames = manifest.frames.n * stride * 4
        guard framesData.count >= expectFrames else {
            throw WorldLoadError.corrupt("frames.bin is "
                + "\(framesData.count) bytes, expected \(expectFrames)")
        }

        self.manifest = manifest
        self.floorZ = Float(manifest.floor_z)
        self.frameCount = manifest.frames.n
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
            node.name = geom.name
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
            parts.append(geom)
        }

        FlyWorld.dress(scene: scene, floorZ: floorZ)
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

        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(white: 0.78, alpha: 1)

        let node = SCNNode(geometry: SCNGeometry(sources: [source],
                                                 elements: [element]))
        node.name = "grid"
        node.geometry?.materials = [material]
        return node
    }

    /// The floor it stands on, a grid, and three lights. Nothing else is in
    /// the world, on purpose: the animal is the subject. Nothing else is in the
    /// world, on purpose: the animal is the subject.
    private static func dress(scene: SCNScene, floorZ: Float) {
        let floor = SCNFloor()
        floor.reflectivity = 0.05
        floor.firstMaterial?.diffuse.contents = UIColor(white: 0.93, alpha: 1)
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
            let b0 = i0 * stride, b1 = i1 * stride
            for (k, node) in nodes.enumerated() {
                let p = b0 + k * 7, q = b1 + k * 7
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
                if k < parts.count {
                    let sz = parts[k].size
                    if sz.count >= 3 {
                        node.scale = SCNVector3(Float(sz[0]), Float(sz[1]),
                                                Float(sz[2]))
                    }
                }
            }
        }
    }

    /// Where the animal is, so a camera can keep it in frame.
    func centre() -> SCNVector3 {
        guard !nodes.isEmpty else { return SCNVector3(0, 0, floorZ) }
        var x: Float = 0, y: Float = 0, z: Float = 0
        for node in nodes {
            x += node.position.x
            y += node.position.y
            z += node.position.z
        }
        let n = Float(nodes.count)
        return SCNVector3(x / n, y / n, z / n)
    }

    /// Frames in the recording.
    var count: Int { frameCount }
    var hz: Int { frameHz }
    var floor: Float { floorZ }
    /// Meshes actually drawn — the parts that carry geometry.
    var meshCount: Int { nodes.count }
    var faceCount: Int { manifest.body_geometry.n_face }
}

// MARK: - Small maths

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
