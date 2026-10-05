import XCTest
import simd
import Metal
@testable import FlyBrain

/// These tests exist because of a real failure. Shipped build 0.3 flew the fly
/// through the walls at 3.3 m/s with the yaw rate pinned at the clamp, and the
/// only way anyone found out was a screenshot of three empty-looking worlds.
///
/// Every number asserted below is a published measurement of *Drosophila
/// melanogaster*, not a value read back off our own simulation. If a future
/// change breaks the animal, these fail.
///
/// This suite mirrors tools/simcheck.py check-for-check (round 5). simcheck
/// re-implements FlyBody.update in Python so the physics can also be verified
/// on a Linux CI runner; where the two disagree, that is a bug in one of them.
///
/// ROUND-5 REMOVALS. Four tests that encoded the deleted adaptation layer are
/// gone on purpose, not by accident:
///
///   - testSustainedLegAsymmetryDoesNotCircleForever and
///     testSustainedFlightAsymmetryDoesNotSpinAtTheClamp asserted that a
///     CONSTANT asymmetric drive stops turning the fly after a few seconds.
///     That behaviour came from the adapted yaw/turn baselines (yawBias /
///     turnBias). The baselines were the H2 bang-bang bug's cover-up: they
///     silently cancelled any command the animal held for long enough. A
///     sustained asymmetric drive IS a sustained turn command — the brain
///     gets to decide how long it lasts, not a leaky integrator inside the
///     body. Replaced by the sign tests (a turn must go the RIGHT way) and
///     the noise/step tests (noise must not turn it, a real step must, and
///     the spin must stop when the command does).
///
///   - testWalkBoutsAlternateWithStops,
///     testGroomingOccupiesAboutThirteenPercentOfActiveTime,
///     testArousalDriftsSpontaneouslyAndStaysBounded and
///     testArousalStretchesWalkBouts tested the scripted habit/arousal layer
///     (Poisson-flavoured walk/stop/groom timers plus an OU arousal). The
///     round-5 brief classifies that layer as scripted behaviour: walk, stop
///     and grooming must come out of the connectome's own activity, so the
///     layer and its tests were deleted together. When the brain drives the
///     legs, bout structure is measured in the connectome tests (Phase 3),
///     not asserted on a timer in the body.
final class FlightPhysicsTests: XCTestCase {

    // MARK: - the constant set itself

    /// The hovering stroke amplitude is solved from force balance, not chosen.
    /// It must land inside the 130-160 degrees measured for hovering flies
    /// (Lehmann & Dickinson 1997; Fry et al. 2005). This is the test that
    /// catches someone mixing in another species' wing again.
    func testHoverAmplitudeIsBiologicallyPlausible() {
        let phiDeg = FlyMorphology.strokeAmplitudeHover * 180 / .pi
        XCTAssertGreaterThan(phiDeg, 130, "hover stroke amplitude too small")
        XCTAssertLessThan(phiDeg, 160, "hover stroke amplitude too large")
        // and it must still be the value the morphology implies
        XCTAssertEqual(FlyMorphology.strokeAmplitudeHover,
                       FlyMorphology.solvedHoverAmplitude, accuracy: 0.02,
                       "constant has drifted from the force balance")
    }

    /// At the hovering amplitude, two wings must lift exactly one body weight.
    func testHoverLiftEqualsBodyWeight() {
        let lift = FlyMorphology.flightForce(strokeAmplitude:
                        FlyMorphology.strokeAmplitudeHover)
        let weight = FlyMorphology.mass * FlyMorphology.gravity
        XCTAssertEqual(lift / weight, 1.0, accuracy: 0.05,
                       "hovering does not balance weight")
    }

    /// Flies can carry about 1.5-2x their weight briefly; they cannot carry 5x.
    func testMaximumLiftIsBoundedAndSufficient() {
        let lift = FlyMorphology.flightForce(strokeAmplitude:
                        FlyMorphology.strokeAmplitudeMax)
        let weight = FlyMorphology.mass * FlyMorphology.gravity
        let ratio = lift / weight
        XCTAssertGreaterThan(ratio, 1.2, "cannot climb at full throttle")
        XCTAssertLessThan(ratio, 2.2, "superfly: implausible peak lift")
    }

    /// Flapping counter-torque must predict the saccade torque Fry et al.
    /// (2003) measured, ~1e-9 N m to sustain ~1600 deg/s.
    func testYawDampingReproducesMeasuredSaccadeTorque() {
        let rate: Float = 1600 * .pi / 180
        let torque = FlyMorphology.yawDamping * rate
        XCTAssertGreaterThan(torque, 3e-10)
        XCTAssertLessThan(torque, 3e-9)
    }

    // MARK: - closed-loop behaviour

    /// Full throttle for five seconds. A fruit fly tops out near 1 m/s
    /// (David 1978; Fry et al. 2003). The bug shipped 3.3 m/s.
    func testTerminalSpeedIsAroundOneMetrePerSecond() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 400; d.wingPowerR = 400      // saturating drive
        body.setDrives(d)
        for _ in 0..<(60 * 5) { body.update(dt: 1.0 / 60, world: world) }

        let speedMS = length(body.pose.velocity) * 0.01
        XCTAssertLessThan(speedMS, 1.5, "runaway speed: damping is wrong")
    }

    /// Yaw must settle below the clamp. Sitting exactly at the clamp, as the
    /// shipped build did, means the damping never took hold.
    func testYawRateStaysBelowTheClamp() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 300; d.wingPowerR = 300
        d.wingSteerL = 300; d.wingSteerR = 0        // hard asymmetric turn
        body.setDrives(d)
        for _ in 0..<(60 * 5) { body.update(dt: 1.0 / 60, world: world) }

        let degPerSec = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertLessThan(degPerSec, 2000, "yaw saturated")
    }

    /// A wing is a joint. However hard the power and steering groups fire, the
    /// commanded sweep must stay inside the morphological limit of 178 deg —
    /// the build that shipped 0.3 asked for 248.
    func testStrokeAmplitudeStaysInsideTheMorphologicalLimit() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 500; d.wingPowerR = 500
        d.wingSteerL = 500; d.wingSteerR = 0
        body.setDrives(d)
        for _ in 0..<(60 * 3) { body.update(dt: 1.0 / 60, world: world) }

        XCTAssertLessThanOrEqual(body.pose.strokeAmplitudeL,
                                 FlyMorphology.strokeAmplitudeMax + 1e-4,
                                 "left wing commanded past its limit")
        XCTAssertLessThanOrEqual(body.pose.strokeAmplitudeR,
                                 FlyMorphology.strokeAmplitudeMax + 1e-4,
                                 "right wing commanded past its limit")
        XCTAssertGreaterThanOrEqual(body.pose.strokeAmplitudeR, 0)
    }

    /// b1/b2 shift stroke amplitude by up to ~20 degrees (Lehmann & Dickinson
    /// 1997). The steering bias is a muscle limit, not a gain, so no amount of
    /// asymmetric firing may exceed it.
    func testSteeringBiasIsBoundedToTheMeasuredTwentyDegrees() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 300; d.wingPowerR = 300
        d.wingSteerL = 300; d.wingSteerR = 0
        body.setDrives(d)
        for _ in 0..<(60 * 3) { body.update(dt: 1.0 / 60, world: world) }

        let bias = abs(body.pose.strokeAmplitudeL - body.pose.strokeAmplitudeR) * 0.5
        XCTAssertLessThanOrEqual(bias, FlyMorphology.steeringRange + 1e-3,
                                 "steering asked for more than the muscle can do")
    }

    /// The headline regression: after ten seconds of maximum drive in every
    /// environment, the fly is still in the room.
    func testFlyCannotEscapeAnyEnvironment() {
        for env in Environment.allCases {
            let world = World()
            world.environment = env
            world.rebuild()

            let body = FlyBody()
            body.reset()
            var d = FlyDrives()
            d.wingPowerL = 500; d.wingPowerR = 500
            d.wingSteerL = 200; d.jump = 20
            body.setDrives(d)
            for _ in 0..<(60 * 10) { body.update(dt: 1.0 / 60, world: world) }

            let p = body.pose.position
            XCTAssertLessThanOrEqual(abs(p.x), world.bounds + 0.01,
                                     "escaped on x in \(env.rawValue)")
            XCTAssertLessThanOrEqual(abs(p.z), world.bounds + 0.01,
                                     "escaped on z in \(env.rawValue)")
            XCTAssertGreaterThanOrEqual(p.y, -0.01,
                                        "fell through the floor in \(env.rawValue)")
            XCTAssertLessThanOrEqual(p.y, world.ceilingHeight + 0.01,
                                     "escaped through the ceiling in \(env.rawValue)")
            XCTAssertFalse(p.x.isNaN || p.y.isNaN || p.z.isNaN,
                           "NaN position in \(env.rawValue)")
        }
    }

    /// With no drive at all the fly must fall and then rest on the floor,
    /// not sink through it or jitter.
    func testUndrivenFlySettlesOnTheFloor() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 2.0, 0))
        body.setDrives(FlyDrives())
        for _ in 0..<(60 * 4) { body.update(dt: 1.0 / 60, world: world) }

        XCTAssertLessThan(body.pose.position.y, 0.3, "did not fall")
        XCTAssertGreaterThanOrEqual(body.pose.position.y, -0.01, "fell through")
        XCTAssertLessThan(abs(body.pose.velocity.y), 1.0, "still moving")
    }

    /// Walking is bounded by the measured preferred speed, ~25 mm/s
    /// (Mendes et al. 2013), so our 30 mm/s ceiling must actually bind.
    func testWalkSpeedStaysBelowThirtyMillimetresPerSecond() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.02, 0))
        var d = FlyDrives()
        d.legL = 400; d.legR = 400                  // wings silent: walking
        body.setDrives(d)
        for _ in 0..<(60 * 3) { body.update(dt: 1.0 / 60, world: world) }

        let mmPerSec = length(body.pose.velocity) * 10
        XCTAssertLessThan(mmPerSec, 35, "walking faster than a fly can walk")
    }

    /// Very long frames must not let the integrator explode.
    func testLargeTimestepIsClamped() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 400; d.wingPowerR = 400
        body.setDrives(d)
        for _ in 0..<120 { body.update(dt: 2.0, world: world) }   // 2 s frames

        XCTAssertFalse(length(body.pose.position).isNaN)
        XCTAssertLessThanOrEqual(abs(body.pose.position.x), world.bounds + 0.01)
    }

    // MARK: - round 5: the sign of a turn (T1)

    /// A stronger left wing yaws the fly toward the WEAKER side — a real fly
    /// turns away from the harder-beating wing (Fry et al. 2003; the drag on
    /// the stronger wing pushes that shoulder back). The flight branch used
    /// to have this inverted (H1): left wing stronger turned LEFT.
    func testYawSignStrongerWingTurnsTowardWeakerSide() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.5, 0))
        var d = FlyDrives()
        d.wingPowerL = 126; d.wingPowerR = 126      // +5%: a real hover
        body.setDrives(d)                           // command, reference adapts
        for _ in 0..<(60 * 6) { body.update(dt: 1.0 / 60, world: world) }
        XCTAssertFalse(body.grounded, "setup did not reach a hover")

        d.wingPowerL = 164; d.wingPowerR = 88       // +30%/-30%: left stronger
        body.setDrives(d)
        let h0 = body.pose.heading
        for _ in 0..<30 { body.update(dt: 1.0 / 60, world: world) }
        let dh = body.pose.heading - h0
        XCTAssertGreaterThan(dh, 0,
                             "stronger left wing turned the wrong way (right expected)")
    }

    /// ...and the walking decode must agree with flight: faster RIGHT legs
    /// turn the animal LEFT, the differential-drive sign.
    func testYawSignWalkingAgreesWithFlight() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.0125, 0))
        var d = FlyDrives()
        d.legL = 90; d.legR = 150                   // right legs faster
        body.setDrives(d)
        let h0 = body.pose.heading
        for _ in 0..<60 { body.update(dt: 1.0 / 60, world: world) }
        XCTAssertLessThan(body.pose.heading - h0, 0,
                          "faster right legs turned the wrong way (left expected)")
    }

    // MARK: - round 5: gain, not bang-bang (T2)

    /// iid ±10% left/right wing noise must not spin the animal. One spike in
    /// a 12-neuron wing group reads as 5.2 Hz through the engine EMA; the
    /// opponent decode plus the muscle low-pass keep that counting noise
    /// under the 200 deg/s bar instead of slamming the yaw clamp.
    func testYawNoiseFloorUnderNoisyDrive() {
        let rng = SeededGenerator(seed: 11)
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.5, 0))
        var d = FlyDrives()
        d.wingPowerL = 126; d.wingPowerR = 126
        body.setDrives(d)
        for _ in 0..<(60 * 4) { body.update(dt: 1.0 / 60, world: world) }

        var sq: Float = 0
        var n = 0
        for _ in 0..<(60 * 3) {
            let base = body.referenceRate * 1.05
            d.wingPowerL = base * (1 + rng.gaussian() * 0.10)
            d.wingPowerR = base * (1 + rng.gaussian() * 0.10)
            body.setDrives(d)
            body.update(dt: 1.0 / 60, world: world)
            sq += body.pose.yawRate * body.pose.yawRate
            n += 1
        }
        let rms = sqrt(sq / Float(n)) * 180 / .pi
        XCTAssertLessThan(rms, 200, "spike-counting noise spins the fly")
        XCTAssertFalse(body.grounded, "the noisy hover fell out of the air")
    }

    /// ...while a REAL asymmetric command still yanks: a 60% left/right step
    /// peaks above 800 deg/s (measured body saccades run to ~1600, Fry et al.
    /// 2003), and the spin STOPS when the command goes symmetric again.
    func testSixtyPercentStepYanksAndStops() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.5, 0))
        var d = FlyDrives()
        d.wingPowerL = 126; d.wingPowerR = 126
        body.setDrives(d)
        for _ in 0..<(60 * 4) { body.update(dt: 1.0 / 60, world: world) }

        let base = body.referenceRate * 1.05
        var peak: Float = 0
        for _ in 0..<60 {
            d.wingPowerL = base * 1.3
            d.wingPowerR = base * 0.7
            body.setDrives(d)
            body.update(dt: 1.0 / 60, world: world)
            peak = max(peak, abs(body.pose.yawRate))
        }
        for _ in 0..<60 {
            d.wingPowerL = base; d.wingPowerR = base
            body.setDrives(d)
            body.update(dt: 1.0 / 60, world: world)
        }
        let after = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertGreaterThan(peak * 180 / .pi, 800, "real command barely turns the fly")
        XCTAssertLessThan(after, 100, "spin never stopped after the command ended")
    }

    // MARK: - round 5: one continuous dynamics model (T3)

    /// No mode flicker: a noisy collective drive near hover, above the floor,
    /// must not touch down once in 5 s. The old lift>0.98W mode switch sat on
    /// a knife edge and flipped ~10x/s under 10% rate noise (H3). Contact is
    /// now geometry — the collision solver — so there is nothing to flicker.
    func testNoModeFlickerUnderNoisyHover() {
        let rng = SeededGenerator(seed: 7)
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.5, 0))
        var d = FlyDrives()
        d.wingPowerL = 126; d.wingPowerR = 126
        body.setDrives(d)
        for _ in 0..<(60 * 4) { body.update(dt: 1.0 / 60, world: world) }

        var touchdowns = 0
        for _ in 0..<(60 * 5) {
            let base = body.referenceRate * 1.05
            d.wingPowerL = base * (1 + rng.gaussian() * 0.10)
            d.wingPowerR = base * (1 + rng.gaussian() * 0.10)
            body.setDrives(d)
            body.update(dt: 1.0 / 60, world: world)
            if body.grounded { touchdowns += 1 }
        }
        XCTAssertEqual(touchdowns, 0, "noisy hover touched down")
        XCTAssertGreaterThan(body.pose.position.y, 0.1, "noisy hover sank")
    }

    /// ...and the converse: a wingless fly standing on the floor must never
    /// lift off. The old walking branch had no ground check and applied
    /// 2%-gravity "hovering" forever; contact now comes from the floor.
    func testGroundedFlyWithoutWingDriveNeverLiftsOff() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.0125, 0))
        var d = FlyDrives()
        d.legL = 60; d.legR = 60                    // wings silent
        body.setDrives(d)
        var liftoffs = 0
        for _ in 0..<(60 * 5) {
            body.update(dt: 1.0 / 60, world: world)
            if !body.grounded { liftoffs += 1 }
        }
        XCTAssertEqual(liftoffs, 0, "wingless fly left the floor")
        XCTAssertLessThan(body.pose.airborne, 0.1, "animation says airborne")
    }

    // MARK: - round 5: the stance foot must not skate (T4)

    /// During stance the planted foot's body-frame fore-aft velocity must
    /// equal -v (no skating): the gait amplitude AND its inverse-map knots
    /// are derived from the body's own stride through the CT model's FK
    /// (tools/gait_geometry.py). Needs the Metal-backed FlyModel, so it
    /// skips where there is no GPU; simcheck.py check 19 is the same test
    /// and runs everywhere.
    func testStanceFootDoesNotSkate() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("FlyModel needs a Metal device; simcheck.py owns this check headlessly")
        }
        let model = try FlyModel(device: device)
        let world = World()
        let body = FlyBody()
        body.attach(model: model)
        body.reset(at: SIMD3<Float>(0, 0.0125, 0))
        var d = FlyDrives()
        d.legL = 120; d.legR = 120
        body.setDrives(d)
        for _ in 0..<(60 * 3) { body.update(dt: 1.0 / 60, world: world) }

        guard let pi = model.partIndex["tarsus_T3_left"] else {
            return XCTFail("tarsus_T3_left missing from the model")
        }
        let part = model.parts[pi]
        let verts = model.vertexBuffer.contents()
            .bindMemory(to: Float.self, capacity: Int(part.vertexCount) * 6)

        // Fixed material point: the tarsus tip, defined exactly as
        // tools/gait_geometry.py defines it — the mesh vertex farthest from
        // the thorax origin in the REST pose (part-local coordinates),
        // chosen once, never re-selected per frame.
        var vert = 0
        var best: Float = -1
        for k in 0..<Int(part.vertexCount) {
            let x = verts[k * 6], y = verts[k * 6 + 1], z = verts[k * 6 + 2]
            let n = x * x + y * y + z * z
            if n > best { best = n; vert = k }
        }
        let tip = SIMD3<Float>(verts[vert * 6], verts[vert * 6 + 1], verts[vert * 6 + 2])

        func footX() -> Float {
            model.solve(angles: body.jointAngles, root: matrix_identity_float4x4, into: &mats)
            let w = mats[pi] * SIMD4<Float>(tip, 1)
            return w.x * 10                        // world units (cm) -> mm
        }

        var samples: [Float] = []
        var prevX: Float?
        var prevU: Float?
        let dt: Float = 1.0 / 60
        for _ in 0..<(60 * 2) {
            body.update(dt: dt, world: world)
            // T3-left is tripod group 0: u = gaitPhase / 2 pi
            var u = body.pose.gaitPhase / (2 * .pi)
            u -= floor(u)
            let x = footX()
            // Mid-stance only, and only across frames of the SAME stance
            // phase: at 13 Hz / 60 fps a frame can straddle the swing->
            // stance transition, where the finite difference would mix
            // airborne swing with planted stance.
            if let px = prevX, let pu = prevU, 0.05 < u, u < 0.5, pu < u {
                let vfoot = (x - px) / dt                    // mm/s
                let vbody = length(body.pose.velocity) * 10  // cm/s -> mm/s
                samples.append(vfoot / vbody)
            }
            prevX = x; prevU = u
        }
        XCTAssertGreaterThan(samples.count, 20, "no mid-stance samples")
        let rel = samples.map { abs($0 + 1) }.reduce(0, +) / Float(samples.count)
        XCTAssertLessThan(rel, 0.25, "stance foot skates: mean |v_foot/v_body + 1| = \(rel)")
    }

    // MARK: - round 5: the giant fibre escape

    /// A rising edge of the TT-motoneuron rate through the one-spike-per-
    /// frame level launches the escape ONCE (0.9 m/s, Card & Dickinson
    /// 2008), the 0.1 s refractory swallows the rattle of a continuing
    /// burst, and an event in mid-air is counted (the signal is never
    /// discarded) but has nothing to push against.
    func testEscapeSpikeLaunchesOnceWithRefractory() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.0125, 0))
        body.setDrives(FlyDrives())
        body.update(dt: 1.0 / 60, world: world)
        XCTAssertTrue(body.grounded, "setup: fly is not on the floor")

        var d = FlyDrives()
        d.jump = 31.25                    // one of the two TT neurons spiking
        body.setDrives(d)
        body.update(dt: 1.0 / 60, world: world)
        XCTAssertEqual(body.jumpEvents, 1, "no launch on the spike event")
        XCTAssertGreaterThan(body.pose.velocity.y, 80, "launch impulse missing")
        XCTAssertFalse(body.grounded, "still glued to the floor")

        body.update(dt: 1.0 / 60, world: world)   // still hot: no retrigger
        body.update(dt: 1.0 / 60, world: world)
        XCTAssertEqual(body.jumpEvents, 1, "retriggered while the drive stayed hot")

        body.setDrives(FlyDrives())
        body.update(dt: 1.0 / 60, world: world)
        d.jump = 31.25
        body.setDrives(d)                 // t = 0.067 s: inside the refractory
        body.update(dt: 1.0 / 60, world: world)
        XCTAssertEqual(body.jumpEvents, 1, "refractory did not swallow the rattle")

        body.setDrives(FlyDrives())
        for _ in 0..<2 {                  // cool down, stay airborne: the
            body.update(dt: 1.0 / 60, world: world)   // measured 0.9 m/s
        }                                 // jump lands again at ~0.25 s
        d.jump = 31.25
        body.setDrives(d)                 // t = 0.15 s: past the refractory
        let vyBefore = body.pose.velocity.y
        body.update(dt: 1.0 / 60, world: world)
        XCTAssertEqual(body.jumpEvents, 2, "mid-air event was discarded")
        XCTAssertLessThan(body.pose.velocity.y, vyBefore + 1.0,
                          "mid-air event fired the impulse again")
    }
}

/// Deterministic gaussian noise for the drive-noise tests (Box-Muller over a
/// splitmix64 stream), so a flicker regression fails reproducibly.
final class SeededGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }

    func uniform() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Double(z >> 11) / Double(1 << 53)
    }

    func gaussian() -> Float {
        let u1 = max(uniform(), 1e-12)
        let u2 = uniform()
        return Float(sqrt(-2 * log(u1)) * cos(2 * .pi * u2))
    }
}

/// The world has to stay internally consistent too — the containment backstop
/// is now load-bearing.
final class WorldContainmentTests: XCTestCase {

    func testContainClampsAPointFarOutsideTheRoom() {
        let world = World()
        var p = SIMD3<Float>(5000, 5000, 5000)
        let n = world.contain(&p, radius: 0.1)
        XCTAssertNotEqual(n, .zero)
        XCTAssertLessThanOrEqual(abs(p.x), world.bounds)
        XCTAssertLessThanOrEqual(p.y, world.ceilingHeight)
    }

    func testContainLeavesAnInteriorPointAlone() {
        let world = World()
        var p = SIMD3<Float>(0, 1.0, 0)
        let n = world.contain(&p, radius: 0.1)
        XCTAssertEqual(n, .zero)
        XCTAssertEqual(p, SIMD3<Float>(0, 1.0, 0))
    }

    /// The world is open now: a floor under the sky, no drawn walls or
    /// ceiling in any environment. The old 12 cm room made the fly collide
    /// with something every fraction of a second.
    func testEveryEnvironmentIsOpenWithScenery() {
        for env in Environment.allCases {
            let world = World()
            world.environment = env
            world.rebuild()
            XCTAssertFalse(world.objects.isEmpty, "\(env.rawValue) is empty")
            XCTAssertFalse(world.objects.contains { $0.kind.isStructure },
                           "\(env.rawValue) still builds walls or a ceiling")
            XCTAssertGreaterThan(world.ceilingHeight, 0)
        }
    }

    func testOdourIsStrongerNearFood() {
        let world = World()
        world.environment = .kitchen
        world.rebuild()
        _ = world.spawn(.food, at: SIMD3<Float>(2, 0.2, 0), size: 0.3)

        let near = world.odour(at: SIMD3<Float>(1.6, 0.2, 0), heading: 0).strength
        let far  = world.odour(at: SIMD3<Float>(-5, 0.2, 0), heading: 0).strength
        XCTAssertGreaterThan(near, far, "odour does not fall off with distance")
    }

    // MARK: - the climb bug (user video: wings beat harder, view never rose)

    /// A sustained above-hover wing command must raise the animal within
    /// seconds. The 2 s equilibrium trim used to renormalise the command
    /// before it could lift: the fly flapped harder and harder while the
    /// eye view stayed put. With the 10 s trim a seconds-long climb works.
    func testSustainedClimbCommandRaisesTheFly() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.35, 0))
        var d = FlyDrives()
        d.wingPowerL = 240; d.wingPowerR = 240
        body.setDrives(d)
        for _ in 0..<(60 * 3) { body.update(dt: 1.0 / 60, world: world) }
        let y3 = body.pose.position.y
        for _ in 0..<(60 * 2) { body.update(dt: 1.0 / 60, world: world) }
        XCTAssertGreaterThan(y3, 10, "no climb in the first 3 s of a climb command")
        XCTAssertGreaterThan(body.pose.position.y, y3 + 10,
                             "climb command stalled: the trim ate it")
    }

    /// ...but the equilibrium reflex must still exist: after the trim has
    /// caught up, lift returns to one body weight. Removing the trim
    /// altogether is the other historical bug (178 deg stroke, ceiling glue).
    func testLiftRetrimsAfterASustainedClimb() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.35, 0))
        var d = FlyDrives()
        d.wingPowerL = 240; d.wingPowerR = 240
        body.setDrives(d)
        for _ in 0..<(60 * 50) { body.update(dt: 1.0 / 60, world: world) }
        // flightForce() already sums both wings (see testHoverLiftEqualsBodyWeight).
        let meanPhi = (body.pose.strokeAmplitudeL + body.pose.strokeAmplitudeR) * 0.5
        let lift = FlyMorphology.flightForce(strokeAmplitude: meanPhi)
        let ratio = lift / (FlyMorphology.mass * FlyMorphology.gravity)
        XCTAssertEqual(ratio, 1.0, accuracy: 0.05,
                       "trim never re-caught the climb command")
    }
}
