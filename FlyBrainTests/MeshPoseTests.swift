//
//  MeshPoseTests.swift
//  The pose a mesh is drawn at, checked against the model it is drawn from.
//
//  Written after uploads/IMG_2713.png: the phone showed the animal's parts
//  scattered away from the body — a dismembered fly — while the physics read
//  feet 6/6, COM z −0.0228 cm, six contacts, i.e. standing perfectly. The
//  positions were right to 1e-15 cm and every part was being *rotated*
//  backwards about its own origin, because the app's matrix → quaternion
//  conversion returned the conjugate (x, y, z negated in the trace branch,
//  w negated in the other three). `FlyWorld` had its own copy, `WorldRig` had
//  another that was right, and nothing could see the difference without a
//  device. Both now call `quatFromMat` (FlyDynamics.swift), and it is checked
//  here, on the pose the animal actually stands in.
//
//  This file is deliberately about the pose, not about the picture: a test that
//  needs a GPU to notice a fly is in pieces is a test that runs too late.
//

import XCTest
@testable import FlyBrain

final class MeshPoseTests: XCTestCase {

    private func golden() throws -> (data: Data, asset: FlyBodyAsset, golden: Data) {
        let bundle = Bundle(for: MeshPoseTests.self)
        func find(_ name: String) -> URL? {
            bundle.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json",
                                   subdirectory: "World")
        }
        guard let bodyURL = find("fly_body"),
              let goldenURL = find("fly_golden") else {
            throw XCTSkip("fly_body.json / fly_golden.json are not in the bundle — "
                          + "run tools/build_body.py, tools/fly_aba.py and "
                          + "tools/pack_body.py")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return (try Data(contentsOf: bodyURL),
                try decoder.decode(FlyBodyAsset.self,
                                   from: try Data(contentsOf: bodyURL)),
                try Data(contentsOf: goldenURL))
    }

    /// The conversion has to be the inverse of the one next to it, for every
    /// rotation, and it has to survive 180° — which is the branch that is
    /// wrong in the textbook when a sign slips.
    func testAQuaternionFromAMatrixIsTheRotationItself() throws {
        let cases: [(axis: Vec3, angle: Double)] = [
            (Vec3(0, 0, 1), 0), (Vec3(0, 0, 1), 0.3), (Vec3(1, 0, 0), 1.2),
            (Vec3(0, 1, 0), -2.0), (Vec3(0, 0, 1), .pi),      // trace ≤ 0, r22 branch
            (Vec3(1, 0, 0), .pi), (Vec3(0, 1, 0), .pi),
            (Vec3(1, 2, 3), 2.9), (Vec3(-2, 0.5, 1), -1.1),
        ]
        for (axis, angle) in cases {
            let m = quatToMat(quatFromAxisAngle(axis, angle))
            let q = quatFromMat(m)
            // 1. the quaternion turns back into the same matrix
            let back = quatToMat(Quat(q.w, q.x, q.y, q.z))
            for (a, b) in [(back.c0, m.c0), (back.c1, m.c1), (back.c2, m.c2)] {
                XCTAssertEqual(a.x, b.x, accuracy: 1e-12)
                XCTAssertEqual(a.y, b.y, accuracy: 1e-12)
                XCTAssertEqual(a.z, b.z, accuracy: 1e-12)
            }
            // 2. and it is *that* rotation, not its inverse: rotating the axis
            //    by it leaves the axis alone, while the inverse would send the
            //    axis to its negative only for a 180° turn — so test the
            //    rotation of a vector that is not the axis, where a conjugate
            //    differs by 2× the angle.
            let n = axis / axis.length
            var v = Vec3(n.y, -n.x, n.z)
            v = v / max(1e-12, v.length)
            let byQuat = rotate(q, v)
            let byMatrix = m * v
            XCTAssertEqual(byQuat.x, byMatrix.x, accuracy: 1e-12)
            XCTAssertEqual(byQuat.y, byMatrix.y, accuracy: 1e-12)
            XCTAssertEqual(byQuat.z, byMatrix.z, accuracy: 1e-12)
        }
    }

    /// Rotate a vector by (x, y, z, w) the same way SceneKit does.
    func rotate(_ q: (x: Double, y: Double, z: Double, w: Double),
                        _ v: Vec3) -> Vec3 {
        // v' = v + 2w(q × v) + 2 q × (q × v)
        let qv = Vec3(q.x, q.y, q.z)
        let t = cross(qv, v) * 2
        return v + t * q.w + cross(qv, t)
    }

    func cross(_ a: Vec3, _ b: Vec3) -> Vec3 {
        Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    /// At the stance the animal actually stands in, every part of the animal
    /// has to be drawn *on* the body, and — this is the one that was broken —
    /// the quaternion the renderer hands SceneKit has to rotate the vertices
    /// the way the solver's own matrix does.
    ///
    /// The scatter in IMG_2713 was exactly this: the positions were right to
    /// 1e-15 cm, the parts were placed on the right bodies, and every part was
    /// spun the wrong way about its own origin. A conjugate quaternion fails
    /// the rotation check below by twice the angle — tens of degrees for most
    /// of this animal — while passing anything that only looks at positions.
    func testEveryPartIsDrawnWhereTheBodyCarriesIt() throws {
        let (_, asset, _) = try golden()
        let body = FlyDynamics(asset: asset)
        body.floorZ = asset.floorZ
        body.reset()
        body.kinematics()

        let (positions, rotations) = body.visualWorld()
        XCTAssertEqual(positions.count, asset.visual.count)
        XCTAssertEqual(rotations.count, asset.visual.count)

        var bodyIndex: [String: Int] = [:]
        for (i, b) in asset.bodies.enumerated() { bodyIndex[b.name] = i }

        var worstPos = 0.0, worstRing = 0.0, worstName = ""
        for (k, v) in asset.visual.enumerated() {
            guard let bi = bodyIndex[v.body] else {
                XCTFail("\(v.name) names a body (\(v.body)) the asset does not have")
                continue
            }
            // 1. where the part has to sit: the body's world pose applied to
            //    the part's own offset.
            let bq = body.linkWorld(bi)
            let wantPos = bq.pos + bq.rot * Vec3(v.pos)
            worstPos = max(worstPos, (positions[k] - wantPos).length)

            // 2. how the part has to be turned: the quaternion the renderer
            //    builds (`quatFromMat`) must turn a vector exactly as the
            //    solver's matrix does. This is the whole renderer contract —
            //    SceneKit is handed that quaternion and nothing else — and a
            //    conjugate quaternion misses it by 2× the rotation.
            //
            //    Three probes, so the check cannot pass by accident on a
            //    rotation that happens to fix one direction.
            let q = quatFromMat(rotations[k])
            for probe in [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)] {
                let byQuat = rotate(q, probe)
                let byMatrix = rotations[k] * probe
                let miss = (byQuat - byMatrix).length
                if miss > worstRing { worstRing = miss; worstName = v.name }
            }
        }
        XCTAssertLessThan(worstPos, 1e-9, "a part is drawn off its own body")
        XCTAssertLessThan(worstRing, 1e-9,
                          "\(worstName) is drawn rotating the wrong way by "
                          + "\(worstRing) cm per unit vector: the matrix → "
                          + "quaternion conversion is not the rotation itself. "
                          + "Returning its conjugate is what put the animal in "
                          + "pieces on the phone (uploads/IMG_2713.png).")

        // 3. and the round trip through both conversions must be the identity,
        //    for every one of the animal's 85 parts, in every branch of the
        //    conversion.
        for k in 0..<min(rotations.count, asset.visual.count) {
            let q = quatFromMat(rotations[k])
            let back = quatToMat(Quat(q.w, q.x, q.y, q.z))
            for probe in [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1),
                          Vec3(0.3, -0.7, 0.65)] {
                let a = back * probe, b = rotations[k] * probe
                XCTAssertLessThan((a - b).length, 1e-9,
                                  "\(asset.visual[k].name): quatToMat("
                                  + "quatFromMat(R)) is not R")
            }
        }
    }
}
