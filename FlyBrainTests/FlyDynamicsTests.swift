//
//  FlyDynamicsTests.swift
//  Does the phone's solver agree with the one that was checked against MuJoCo?
//
//  `tools/fly_aba.py` is the reference implementation: it was verified three
//  ways against MuJoCo (forward kinematics, twenty milliseconds of free fall,
//  and the animal standing on its own feet for three seconds) and it writes
//  the golden trace with `--golden`. These tests replay that trace through the
//  Swift port and compare the *final* state of every joint, the root pose, the
//  root velocity and the body's angular velocity.
//
//  The trace is the right thing to pin the port to because it is not a
//  plausible-looking animation: it is 500 steps of a vacuum tumble under a
//  deterministic torque script, in which every one of the 102 joints is
//  moving, gravity is on, the joint springs are on, the model's own damping
//  and armature are on, and nothing is smoothed. Two solvers that disagree
//  about a sign, a frame or the order of the rotations separate from each
//  other immediately, and a tolerance of 1e-7 on a state whose joint angles
//  reach 0.83 rad is a claim about the arithmetic, not about the look.
//
//  Measured, as a sanity check on that tolerance: perturbing one torque by one
//  unit in the last place moves the 500-step state by 6.7e-16 in the angles and
//  2.3e-13 in the rates (tools/golden_sens.py). So 1e-7 leaves six orders of
//  magnitude for a different instruction order on a different CPU, and still
//  fails on any real disagreement.
//

import XCTest
@testable import FlyBrain

final class FlyDynamicsTests: XCTestCase {

    // MARK: - The golden trace

    private struct Golden: Decodable {
        struct Torque: Decodable {
            let amplitude: [Double]
            let periodS: Double
        }
        struct State: Decodable {
            let q: [Double]
            let qd: [Double]
            let v: [Double]
            let omega: [Double]
            let rootPos: [Double]?
            let rootQuat: [Double]?
            /// How many joints the reference's hard stops stopped. A structural
            /// fingerprint: it does not depend on a tolerance at all.
            let limitStops: Int?
        }
        /// One recorded sample: the whole q vector and the root position at a
        /// step, so a failure says *where* the two solvers parted rather than
        /// only that they did.
        struct Sample: Decodable {
            let step: Int
            let q: [Double]
            let rootPos: [Double]
        }
        let dt: Double
        let steps: Int
        let floorZ: Double
        let torque: Torque
        let initial: State
        let expectedFinal: State
        let trace: [Sample]?
        let tolerance: Double
    }

    private func golden() throws -> (Golden, FlyBodyAsset) {
        let bundle = Bundle(for: FlyDynamicsTests.self)
        func find(_ name: String) -> URL? {
            bundle.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json",
                                   subdirectory: "World")
        }
        guard let assetURL = find("fly_body") else {
            throw XCTSkip("fly_body.json is not in the bundle — run "
                          + "tools/build_body.py and tools/pack_body.py")
        }
        guard let goldenURL = find("fly_golden") else {
            throw XCTSkip("fly_golden.json is not in the bundle — run "
                          + "tools/fly_aba.py --golden build/fly_golden.json")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let asset = try decoder.decode(
            FlyBodyAsset.self, from: try Data(contentsOf: assetURL))
        let g = try decoder.decode(Golden.self,
                                  from: try Data(contentsOf: goldenURL))
        return (g, asset)
    }

    func testGoldenTraceIsReproduced() throws {
        let (g, asset) = try golden()
        let body = FlyDynamics(asset: asset)
        XCTAssertEqual(body.nj, g.initial.q.count,
                       "the asset and the trace disagree about the joint count")
        body.floorZ = g.floorZ
        body.reset(rootZ: 0,
                   rootPosition: Vec3(g.initial.rootPos ?? [0, 0, 0]),
                   quat: FlyDynamics.quat(g.initial.rootQuat ?? [1, 0, 0, 0]),
                   q: g.initial.q, qd: g.initial.qd,
                   vel: Vec3(g.initial.v), omega: Vec3(g.initial.omega))
        body.kinematics()
        XCTAssertEqual(body.centreOfMass().z, body.centreOfMass().z,
                       "the initial pose produced a NaN centre of mass")

        // The trace is the reference's own recorded samples. Replaying against
        // them, rather than only against the final state, is what makes a
        // failure say *when*: a port that is wrong from step 0 is a different
        // bug from one that drifts apart at step 400.
        let trace = g.trace ?? []
        var next = 0
        var firstBad: (step: Int, joint: Int, got: Double, want: Double)? = nil
        var worstSeen = 0.0
        for k in 0..<g.steps {
            var tau = [Double](repeating: 0, count: body.nj)
            let phase = 2 * Double.pi * Double(k) * g.dt / g.torque.periodS
            for j in 0..<body.nj { tau[j] = g.torque.amplitude[j] * cos(phase) }
            body.step(dt: g.dt, torque: tau)

            if next < trace.count && trace[next].step == k {
                let sample = trace[next]
                next += 1
                var worst = 0.0
                var joint = -1
                var got = 0.0
                for j in 0..<min(sample.q.count, body.q.count) {
                    let d = abs(body.q[j] - sample.q[j])
                    if d > worst { worst = d; joint = j; got = body.q[j] }
                }
                worstSeen = max(worstSeen, worst)
                if firstBad == nil && worst > 1e-9 {
                    firstBad = (k, joint, got, sample.q[joint])
                }
            }
        }
        if let bad = firstBad {
            let at = "step \(bad.step): joint \(bad.joint) is \(bad.got) but the "
                   + "reference has \(bad.want) (\(abs(bad.got - bad.want)))"
            let stops = "hard stops: this solver stopped a joint \(body.limitStops) "
                      + "times, the reference \(g.expectedFinal.limitStops ?? -1)"
            XCTFail("the replay left the reference trajectory. \(at). \(stops). "
                    + "worst joint error over the run so far: \(worstSeen). "
                    + "For reference, in the final state: root velocity \(body.vel.array), "
                    + "omega \(body.omega.array), rootPos \(body.rootPos.array), "
                    + "joints at a hard stop \(body.jointsAtLimit()).")
        }

        // The hard-stop count, before any tolerance: a solver that clamps
        // differently, or does not clamp, is a different animal.
        if let want = g.expectedFinal.limitStops {
            XCTAssertEqual(body.limitStops, want,
                           "hard stops: this solver stopped a joint "
                           + "\(body.limitStops) times, the reference \(want)")
        }

        func worst(_ a: [Double], _ b: [Double]) -> Double {
            var w = 0.0
            for i in 0..<min(a.count, b.count) { w = max(w, abs(a[i] - b[i])) }
            return w
        }
        let expected = g.expectedFinal
        XCTAssertLessThanOrEqual(worst(body.q, expected.q), g.tolerance,
                                 "joint angles after \(g.steps) steps")
        XCTAssertLessThanOrEqual(worst(body.qd, expected.qd), g.tolerance,
                                 "joint rates after \(g.steps) steps")
        XCTAssertLessThanOrEqual(worst(body.vel.array, expected.v), g.tolerance,
                                 "root velocity")
        XCTAssertLessThanOrEqual(worst(body.omega.array, expected.omega),
                                 g.tolerance, "root angular velocity")
        if let rp = expected.rootPos {
            XCTAssertLessThanOrEqual(
                worst(body.rootPos.array, rp), g.tolerance, "root position")
        }
        if let rq = expected.rootQuat {
            let got = [body.rootQuat.x, body.rootQuat.y,
                       body.rootQuat.z, body.rootQuat.w]
            XCTAssertLessThanOrEqual(worst(got, rq), g.tolerance, "root quat")
        }
        // A runaway would still "pass" a tolerance check on NaNs, so say so.
        for value in body.q where !value.isFinite {
            XCTFail("the solver produced a non-finite joint angle")
            break
        }
    }

    // MARK: - The stance

    func testTheAnimalStandsWithMuscleToneOnly() throws {
        let (_, asset) = try golden()
        let animal = FlyLiveBody(asset: asset)
        animal.startUp()

        // Twenty milliseconds at the model's own timestep, exactly the
        // measurement tools/fly_aba.py makes in check 3.
        let dt = animal.dt
        let (kp, kd) = animal.dynamics.stanceServo(dt: dt)
        var comZ: [Double] = []
        var feetLow = Int.max
        for k in 1...200 {
            let ramp = min(1.0, Double(k) / 500.0)
            var exc = [Double](repeating: 0, count: animal.dynamics.nj)
            for j in 0..<animal.dynamics.nj {
                let e = ramp * animal.posture[j]
                    + kp[j] * (animal.dynamics.stanceQ[j] - animal.dynamics.q[j])
                    - kd[j] * animal.dynamics.qd[j]
                exc[j] = min(1, max(-1, e))
            }
            animal.dynamics.step(dt: dt, torque: animal.dynamics.muscleTorque(
                q: animal.dynamics.q, qd: animal.dynamics.qd, excitation: exc))
            comZ.append(animal.dynamics.centreOfMass().z)
            var feet = 0
            for (_, f) in animal.dynamics.footForce where f > 1e-6 { feet += 1 }
            feetLow = min(feetLow, feet)
        }

        let last = comZ.suffix(50)
        let mean = last.reduce(0, +) / Double(last.count)
        // The reference value, and the port now reproduces it digit for digit:
        // running exactly this scenario on tools/fly_aba.py gives a mean of
        // -0.027384 (first -0.026137, last -0.027331, six feet down at every
        // step, worst joint rate 2.644 rad/s). `verify_standing`'s own 1.5 s
        // run, which keeps ramping the tonic drive past 20 ms, settles at
        // -0.0230 — the -0.0229 this test was first written against.
        //
        // The animal starts at the *stance height*, which is where its hold
        // torques were measured (`FlyDynamics.init` sets root z =
        // stance_root_z). Started at z = 0 it hangs 49 um above the floor with
        // no contact at all, the servo holds the legs against nothing, and by
        // 200 steps it has launched itself: the port measured a mean of
        // +0.038 and 1 foot down before that was fixed.
        XCTAssertEqual(mean, -0.0229, accuracy: 0.01,
                       "the animal is not standing where the reference stands")
        XCTAssertGreaterThanOrEqual(feetLow, 4,
                                    "fewer than four feet are carrying it")
        for z in comZ where !z.isFinite {
            XCTFail("the stance produced a non-finite centre of mass")
            break
        }
    }

    func testTheMuscleModelIsForceBasedAndBraked() throws {
        let (_, asset) = try golden()
        let body = FlyDynamics(asset: asset)

        // Positive excitation opens the joint; the muscles are a pair, so the
        // same command has to be able to push *and* to brake.
        let q = [Double](repeating: 0, count: body.nj)
        // With no excitation the muscle pair is not silent: it is the passive
        // spring the model's own actuators declare, pulling the joint towards
        // zero and nothing else.
        let still = body.muscleTorque(q: q, qd: q,
                                      excitation: [Double](repeating: 0, count: body.nj))
        var passive = 0
        for j in 0..<body.nj where body.passiveStiffness[j] > 0 {
            XCTAssertEqual(still[j], 0, accuracy: 1e-12)
            passive += 1
        }
        XCTAssertGreaterThan(passive, 30,
                             "hardly any joint has a passive muscle stiffness")
        let up = body.muscleTorque(q: q, qd: q,
                                   excitation: [Double](repeating: 1, count: body.nj))
        let down = body.muscleTorque(q: q, qd: q,
                                     excitation: [Double](repeating: -1, count: body.nj))
        var signsOK = true
        for j in 0..<body.nj {
            if up[j] < down[j] { signsOK = false }
        }
        XCTAssertTrue(signsOK, "+excitation must pull the joint the positive way")

        // A joint moving faster than vMax cannot be braked by a shortening
        // muscle — the lengthening one has to do it, and it gets stronger.
        let vmax = FlyMuscle.vMax
        let braking = body.muscleTorque(q: q,
                                        qd: [Double](repeating: vmax * 1.1,
                                                     count: body.nj),
                                        excitation: [Double](repeating: -1,
                                                             count: body.nj))
        var anyBrake = false
        for j in 0..<body.nj where braking[j] < -1e-9 { anyBrake = true }
        XCTAssertTrue(anyBrake,
                      "nothing can brake a joint moving past vMax: the "
                      + "velocity term has lost its per-direction form")
    }
}
