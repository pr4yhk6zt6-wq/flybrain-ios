//
//  FlyModel.swift
//  Loads flymodel.bin — the Janelia / DeepMind `flybody` Drosophila
//  reconstruction, decimated and packed by tools/build_flymodel.py.
//
//  41 rigid parts in a parent/child tree, 76 real hinge joints with the
//  anatomical axes and limits from the published MuJoCo model. We do not invent
//  the skeleton; we only choose the joint ANGLES, and those come from the
//  connectome's motor output.
//
//  Attribution: TuragaLab/flybody, Apache-2.0.
//  Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
//  Nature (2025).
//

import Foundation
import Metal
import simd

struct FlyPart {
    var parent: Int32
    var restTranslation: SIMD3<Float>
    var restRotation: SIMD4<Float>     // xyzw
    var colour: SIMD4<Float>
    var vertexStart: UInt32
    var vertexCount: UInt32
    var indexStart: UInt32
    var indexCount: UInt32
    var jointStart: UInt32
    var jointCount: UInt32
    var nameOffset: UInt32
    var nameLength: UInt32
}

struct FlyJoint {
    var axis: SIMD3<Float>
    var lower: Float
    var upper: Float
    var nameOffset: UInt32
    var nameLength: UInt32

    func clamp(_ a: Float) -> Float { max(lower, min(upper, a)) }
}

enum FlyModelError: Error, LocalizedError {
    case missing, badMagic(String), unsupported(UInt32), allocation

    var errorDescription: String? {
        switch self {
        case .missing:         return "flymodel.bin is not in the app bundle."
        case .badMagic(let m): return "flymodel.bin has bad magic '\(m)'"
        case .unsupported(let v): return "flymodel.bin version \(v) unsupported"
        case .allocation:      return "Could not allocate the fly model buffers"
        }
    }
}

final class FlyModel {

    private(set) var parts: [FlyPart] = []
    private(set) var joints: [FlyJoint] = []
    private(set) var partNames: [String] = []
    private(set) var jointNames: [String] = []
    private(set) var partIndex: [String: Int] = [:]
    private(set) var jointIndex: [String: Int] = [:]

    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let triangleCount: Int

    /// Uniform scale applied so the model's 2.5 mm body length matches our
    /// world units, where 1.0 = 1 cm. Solved from the loaded bounds.
    private(set) var normalisationScale: Float = 1
    private(set) var normalisationOffset = SIMD3<Float>(0, 0, 0)

    init(device: MTLDevice, url: URL? = nil) throws {
        let fileURL: URL
        if let url { fileURL = url }
        else if let u = Bundle.main.url(forResource: "flymodel", withExtension: "bin") {
            fileURL = u
        } else { throw FlyModelError.missing }

        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)

        func u32(_ o: Int) -> UInt32 {
            data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }
        }
        func u64(_ o: Int) -> UInt64 {
            data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) }
        }

        let magic = String(bytes: data[0..<8], encoding: .ascii) ?? "?"
        guard magic == "FLYMODEL" else { throw FlyModelError.badMagic(magic) }
        let version = u32(8)
        guard version == 1 else { throw FlyModelError.unsupported(version) }

        let partCount = Int(u32(12))
        let vertexCount = Int(u32(16))
        let indexCount = Int(u32(20))
        let jointCount = Int(u32(24))

        var offs: [Int] = []
        var lens: [Int] = []
        for i in 0..<5 {
            offs.append(Int(u64(64 + i * 16)))
            lens.append(Int(u64(64 + i * 16 + 8)))
        }

        // ---- parts -------------------------------------------------------
        let partStride = 4 + 12 + 16 + 16 + 24 + 8   // see the Python packer
        parts.reserveCapacity(partCount)
        for i in 0..<partCount {
            let o = offs[0] + i * partStride
            func f(_ k: Int) -> Float {
                data.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: o + k, as: Float.self)
                }
            }
            func uu(_ k: Int) -> UInt32 {
                data.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: o + k, as: UInt32.self)
                }
            }
            parts.append(FlyPart(
                parent: Int32(bitPattern: uu(0)),
                restTranslation: SIMD3<Float>(f(4), f(8), f(12)),
                restRotation: SIMD4<Float>(f(16), f(20), f(24), f(28)),
                colour: SIMD4<Float>(f(32), f(36), f(40), f(44)),
                vertexStart: uu(48), vertexCount: uu(52),
                indexStart: uu(56), indexCount: uu(60),
                jointStart: uu(64), jointCount: uu(68),
                nameOffset: uu(72), nameLength: uu(76)))
        }

        // ---- joints ------------------------------------------------------
        let jointStride = 12 + 8 + 8
        joints.reserveCapacity(jointCount)
        for i in 0..<jointCount {
            let o = offs[1] + i * jointStride
            func f(_ k: Int) -> Float {
                data.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: o + k, as: Float.self)
                }
            }
            func uu(_ k: Int) -> UInt32 {
                data.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: o + k, as: UInt32.self)
                }
            }
            joints.append(FlyJoint(axis: normalize(SIMD3<Float>(f(0), f(4), f(8))),
                                   lower: f(12), upper: f(16),
                                   nameOffset: uu(20), nameLength: uu(24)))
        }

        // ---- names -------------------------------------------------------
        let nameBlob = data.subdata(in: offs[2]..<(offs[2] + lens[2]))
        func name(_ off: UInt32, _ len: UInt32) -> String {
            guard len > 0, Int(off) + Int(len) <= nameBlob.count else { return "" }
            return String(data: nameBlob.subdata(in: Int(off)..<(Int(off) + Int(len))),
                          encoding: .utf8) ?? ""
        }
        partNames = parts.map { name($0.nameOffset, $0.nameLength) }
        jointNames = joints.map { name($0.nameOffset, $0.nameLength) }
        for (i, n) in partNames.enumerated() where !n.isEmpty { partIndex[n] = i }
        for (i, n) in jointNames.enumerated() where !n.isEmpty { jointIndex[n] = i }

        // ---- geometry ----------------------------------------------------
        var verts = [Float](repeating: 0, count: vertexCount * 6)
        _ = verts.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: offs[3]..<(offs[3] + lens[3]))
        }

        // Normalise: centre the model and scale a 2.5 mm fly into world units
        // where 1.0 == 1 cm. D. melanogaster body length is 2.0-2.5 mm.
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in 0..<vertexCount {
            let p = SIMD3<Float>(verts[i * 6], verts[i * 6 + 1], verts[i * 6 + 2])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        let extent = hi - lo
        let longest = max(extent.x, max(extent.y, extent.z))
        normalisationScale = longest > 0 ? (0.25 / longest) : 1    // 2.5 mm = 0.25 cm
        normalisationOffset = -(lo + hi) * 0.5

        guard let vb = device.makeBuffer(bytes: verts,
                                         length: MemoryLayout<Float>.stride * verts.count,
                                         options: .storageModeShared) else {
            throw FlyModelError.allocation
        }
        vb.label = "flyModelVertices"
        vertexBuffer = vb

        var idx = [UInt32](repeating: 0, count: indexCount)
        _ = idx.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: offs[4]..<(offs[4] + lens[4]))
        }
        guard let ib = device.makeBuffer(bytes: idx,
                                         length: MemoryLayout<UInt32>.stride * max(indexCount, 1),
                                         options: .storageModeShared) else {
            throw FlyModelError.allocation
        }
        ib.label = "flyModelIndices"
        indexBuffer = ib
        triangleCount = indexCount / 3
    }

    // MARK: - Posing

    /// Compose every part's world matrix from the rest pose plus the supplied
    /// joint angles. `angles` is indexed the same as `joints`.
    ///
    /// MuJoCo composes a body as:  parent * translate(pos) * rotate(quat) * Π joint rotations
    func solve(angles: [Float], root: float4x4, into out: inout [float4x4]) {
        if out.count != parts.count {
            out = [float4x4](repeating: matrix_identity_float4x4, count: parts.count)
        }
        // Model normalisation folds into the root so the hierarchy stays in
        // MJCF units and the joint offsets remain correct.
        let norm = root
            * float4x4(scale: SIMD3<Float>(repeating: normalisationScale))
            * float4x4(translation: normalisationOffset)

        for (i, part) in parts.enumerated() {
            let parentM = part.parent >= 0 ? out[Int(part.parent)] : norm
            var local = float4x4(translation: part.restTranslation)
                      * float4x4(quaternion: part.restRotation)
            if part.jointCount > 0 {
                for j in 0..<Int(part.jointCount) {
                    let ji = Int(part.jointStart) + j
                    let a = ji < angles.count ? joints[ji].clamp(angles[ji]) : 0
                    if a != 0 {
                        local = local * float4x4(axis: joints[ji].axis, angle: a)
                    }
                }
            }
            out[i] = parentM * local
        }
    }

    /// Index of the first joint on a part whose name matches, or nil.
    func joint(_ name: String) -> Int? { jointIndex[name] }
}

// MARK: - Matrix helpers used only here

extension float4x4 {
    init(quaternion q: SIMD4<Float>) {
        let x = q.x, y = q.y, z = q.z, w = q.w
        self.init(SIMD4<Float>(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0),
                  SIMD4<Float>(2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0),
                  SIMD4<Float>(2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0),
                  SIMD4<Float>(0, 0, 0, 1))
    }

    init(axis: SIMD3<Float>, angle: Float) {
        let a = normalize(axis)
        let c = cos(angle), s = sin(angle), t = 1 - c
        self.init(SIMD4<Float>(t * a.x * a.x + c,       t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y, 0),
                  SIMD4<Float>(t * a.x * a.y - s * a.z, t * a.y * a.y + c,       t * a.y * a.z + s * a.x, 0),
                  SIMD4<Float>(t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c,       0),
                  SIMD4<Float>(0, 0, 0, 1))
    }
}
