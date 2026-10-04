import XCTest
import simd
@testable import FlyBrain

/// These tests exist because of a real failure. Shipped build 0.3 flew the fly
/// through the walls at 3.3 m/s with the yaw rate pinned at the clamp, and the
/// only way anyone found out was a screenshot of three empty-looking worlds.
///
/// Every number asserted below is a published measurement of *Drosophila
/// melanogaster*, not a value read back off our own simulation. If a future
/// change breaks the animal, these fail.
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

    /// The regression behind the two-eye / open-world change: a constant
    /// left/right leg bias used to circle the fly forever at a steady yaw
    /// rate (the "walks weird, always turning" complaint). The adapted turn
    /// baseline must straighten it out.
    func testSustainedLegAsymmetryDoesNotCircleForever() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.02, 0))
        var d = FlyDrives()
        d.legL = 30; d.legR = 45          // sustained asymmetric drive
        body.setDrives(d)
        for _ in 0..<(60 * 12) { body.update(dt: 1.0 / 60, world: world) }

        let degPerSec = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertLessThan(degPerSec, 5, "still circling after 12 s of bias")
    }

    /// The flight twin of the same complaint, from the user's video of the
    /// open world: a sustained steering asymmetry pinned the fly at the
    /// 1600 deg/s yaw clamp (the HUD read 1464). The adapted yaw baseline
    /// must unwind the spin.
    func testSustainedFlightAsymmetryDoesNotSpinAtTheClamp() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 300; d.wingPowerR = 300
        d.wingSteerL = 300; d.wingSteerR = 0
        body.setDrives(d)
        for _ in 0..<(60 * 12) { body.update(dt: 1.0 / 60, world: world) }

        let degPerSec = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertLessThan(degPerSec, 300, "still spinning after 12 s of bias")
    }

    /// ...while a fresh steering command still yaws the flying animal.
    func testFlightSteeringStillResponds() {
        let world = World()
        let body = FlyBody()
        body.reset()
        var d = FlyDrives()
        d.wingPowerL = 300; d.wingPowerR = 300
        body.setDrives(d)
        for _ in 0..<(60 * 6) { body.update(dt: 1.0 / 60, world: world) }

        d.wingSteerL = 300; d.wingSteerR = 0
        body.setDrives(d)
        for _ in 0..<30 { body.update(dt: 1.0 / 60, world: world) }

        let degPerSec = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertGreaterThan(degPerSec, 200, "flight steering dead")
    }

    /// ...while a fresh asymmetry must still steer, or the adaptation would
    /// have lobotomised the animal's turning.
    func testTurnsStillRespondToNewAsymmetry() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.02, 0))
        var d = FlyDrives()
        d.legL = 30; d.legR = 30
        body.setDrives(d)
        for _ in 0..<(60 * 6) { body.update(dt: 1.0 / 60, world: world) }

        d.legL = 20; d.legR = 45          // a new, stronger asymmetry
        body.setDrives(d)
        for _ in 0..<30 { body.update(dt: 1.0 / 60, world: world) }

        let degPerSec = abs(body.pose.yawRate) * 180 / .pi
        XCTAssertGreaterThan(degPerSec, 5, "steering no longer responds")
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
        let lift = FlyMorphology.flightForce(strokeAmplitude: body.pose.strokeAmplitudeL)
                 + FlyMorphology.flightForce(strokeAmplitude: body.pose.strokeAmplitudeR)
        let ratio = lift / (FlyMorphology.mass * FlyMorphology.gravity)
        XCTAssertEqual(ratio, 1.0, accuracy: 0.05,
                       "trim never re-caught the climb command")
    }

    // MARK: - ethology: the habits layer

    /// Real walking is bout-structured: walk/stop transitions behave like
    /// Poisson events (Demir et al. 2020, eLife 5:e57524: baseline walk
    /// initiation ~0.29/s, stops at least 300 ms). A fly that walked without
    /// ever stopping would not be a fly.
    func testWalkBoutsAlternateWithStops() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.02, 0))
        var d = FlyDrives()
        d.legL = 60; d.legR = 60
        body.setDrives(d)

        var walkDurations: [Float] = []
        var stopDurations: [Float] = []
        var current = body.habit
        var start: Float = 0
        for i in 0..<(60 * 300) {
            body.update(dt: 1.0 / 60, world: world)
            if body.habit != current {
                let t = Float(i) / 60
                switch current {
                case .walk:  walkDurations.append(t - start)
                case .stop:  stopDurations.append(t - start)
                case .groom: break
                }
                current = body.habit
                start = t
            }
        }
        XCTAssertFalse(walkDurations.isEmpty, "the fly never walked")
        XCTAssertFalse(stopDurations.isEmpty, "the fly never stopped")
        let meanWalk = walkDurations.reduce(0, +) / Float(walkDurations.count)
        XCTAssertGreaterThan(meanWalk, 1.5, "walk bouts shorter than measured")
        XCTAssertLessThan(meanWalk, 5.0, "walk bouts longer than measured")
        XCTAssertGreaterThanOrEqual(stopDurations.min() ?? 0, 0.3,
                                    "stop shorter than the 300 ms floor")
    }

    /// Grooming occupies ~13% of waking time (Lazopulo & Syed 2018, eLife
    /// 7:e34497), in bouts of a few tenths of a second to a couple of
    /// seconds sweeping anterior-to-posterior (Seeds et al. 2014; Ray et al.
    /// 2019). The layer is deterministic, so this share is exact.
    func testGroomingOccupiesAboutThirteenPercentOfActiveTime() {
        let world = World()
        let body = FlyBody()
        body.reset(at: SIMD3<Float>(0, 0.02, 0))
        var d = FlyDrives()
        d.legL = 60; d.legR = 60
        body.setDrives(d)

        var groomFrames = 0
        var totalFrames = 0
        for _ in 0..<(60 * 300) {
            body.update(dt: 1.0 / 60, world: world)
            totalFrames += 1
            if body.habit == .groom { groomFrames += 1 }
        }
        let share = Float(groomFrames) / Float(totalFrames)
        XCTAssertGreaterThan(share, 0.08, "fly almost never grooms")
        XCTAssertLessThan(share, 0.20, "fly grooms far more than measured")
    }
}
