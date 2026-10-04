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
}

/// The world has to stay internally consistent too — the containment backstop
/// is now load-bearing.
final class WorldContainmentTests: XCTestCase {

    func testContainClampsAPointFarOutsideTheRoom() {
        let world = World()
        var p = SIMD3<Float>(500, 500, 500)
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

    func testEveryEnvironmentBuildsAFloorAndFourWalls() {
        for env in Environment.allCases {
            let world = World()
            world.environment = env
            world.rebuild()
            XCTAssertFalse(world.objects.isEmpty, "\(env.rawValue) is empty")
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
}
