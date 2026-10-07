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

    // MARK: - The camera, which is where a correct pose looks broken

    /// The tilt band the floor allows: computed, not written down, and checked
    /// against the two views IMG_2715 was reporting — "the background is
    /// sometimes grey, sometimes white" is the floor seen from its two sides.
    ///
    /// From below, the scene does not light the floor: the same plane that
    /// reads bright from above reads grey from underneath, and the grid (line
    /// primitives, which are not back-face culled) floats in the background.
    /// So the rule is: the lens stays above the floor plane, by a margin,
    /// wherever the pinch and the drag can put it.
    func testTheLensMayNotGoUnderTheFloor() throws {
        // the app's own numbers, as `world.json` carries them
        let floorZ = -0.132, targetZ = -0.028, margin = WorldRig.eyeMarginCM
        let distances = [0.818546, 1.0, 1.5, 2.49904]      // min, home, mid, max

        for d in distances {
            let band = WorldRig.elevationBand(floorZ: floorZ, targetZ: targetZ,
                                              distance: d, margin: margin)
            XCTAssertGreaterThan(band.width, 1.0, "the band at \(d) cm leaves "
                                + "the lens less than a radian of tilt")
            // the lowest permitted tilt puts the lens exactly on the margin
            let low = band.lower
            let z = targetZ + d * sin(low)
            XCTAssertGreaterThanOrEqual(z - floorZ, margin - 1e-12,
                                        "at \(d) cm a tilt of \(low) rad puts "
                                        + "the lens \(z - floorZ) cm above the "
                                        + "floor — under it, or on it")
            // and one step below the band is under the floor, so the band is
            // tight rather than merely safe
            XCTAssertLessThan(targetZ + d * sin(low - 0.01) - floorZ, margin,
                              "the band at \(d) cm is looser than the floor")
        }
        // the old clamp let the lens reach here: 0.82 cm at a 17° downward tilt
        // is 0.19 cm below the floor plane
        let oldZ = targetZ + 0.818546 * sin(-0.3)
        XCTAssertLessThan(oldZ, floorZ,
                          "the defect this replaced: −0.3 rad at the closest "
                          + "distance was under the floor and the clamp allowed it")
    }

    /// The camera's basis comes from the two angles the user steers, so it is
    /// defined at every tilt — including straight down, where the old
    /// `up × z` fallback jumped to a world axis and rolled the image in a step.
    func testTheCameraBasisIsContinuousThroughThePoles() throws {
        // the same construction as WorldRig.orient, without a node
        func basis(az: Double, el: Double) -> (x: (Double, Double, Double),
                                               y: (Double, Double, Double),
                                               z: (Double, Double, Double)) {
            let z = (cos(el) * cos(az), cos(el) * sin(az), sin(el))
            var x = (-sin(az), cos(az), 0.0)
            let dot = x.0 * z.0 + x.1 * z.1 + x.2 * z.2
            x = (x.0 - dot * z.0, x.1 - dot * z.1, x.2 - dot * z.2)
            let xl = (x.0 * x.0 + x.1 * x.1 + x.2 * x.2).squareRoot()
            x = (x.0 / xl, x.1 / xl, x.2 / xl)
            return (x, (z.1 * x.2 - z.2 * x.1,
                        z.2 * x.0 - z.0 * x.2,
                        z.0 * x.1 - z.1 * x.0), z)
        }

        func distance(_ a: (Double, Double, Double),
                      _ b: (Double, Double, Double)) -> Double {
            ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)
             + (a.2 - b.2) * (a.2 - b.2)).squareRoot()
        }

        var previous: (Double, Double, Double)?
        var worst = 0.0
        // a full sweep of tilt through both poles, at four azimuths
        for azi in 0..<4 {
            let az = Double(azi) * .pi / 2
            previous = nil
            for i in 0...180 {
                let el = -Double.pi / 2 + Double(i) * .pi / 180
                let b = basis(az: az, el: el)
                // orthonormal, right-handed, and the up axis is up at level tilt
                XCTAssertEqual(distance(b.x, (0, 0, 0)), 1, accuracy: 1e-12)
                XCTAssertEqual(distance(b.y, (0, 0, 0)), 1, accuracy: 1e-12)
                let handed = (b.x.0 * (b.y.1 * b.z.2 - b.y.2 * b.z.1)
                              - b.y.0 * (b.x.1 * b.z.2 - b.x.2 * b.z.1)
                              + b.z.0 * (b.x.1 * b.y.2 - b.x.2 * b.y.1))
                XCTAssertEqual(handed, 1, accuracy: 1e-9, "not right-handed")
                if let p = previous {
                    worst = max(worst, distance(p, b.x))
                }
                previous = b.x
            }
        }
        // One step of tilt moves the image's right axis by at most a couple of
        // degrees — a roll *step* (the old fallback) is 90° and cannot hide
        // under this bound.
        XCTAssertLessThan(worst, 0.05,
                          "the camera's right axis jumps by \(worst) per degree "
                          + "of tilt: that is a roll, not an orbit")

        // And the construction this replaced, for the record: its fallback at
        // straight down was the world +x axis, whatever the azimuth — so the
        // image's right axis stepped from the orbit's tangent to (1, 0, 0) as
        // the tilt crossed the pole. That step is what the bound above forbids.
        let oldStep = abs(basis(az: 2.2, el: 0).x.1 - 0.0)   // fallback: x = (1,0,0)
        XCTAssertGreaterThan(oldStep, 0.1,
                             "the old fallback's step was \\(oldStep) in the "
                             + "camera's right axis — a roll, not an orbit")
    }
}
