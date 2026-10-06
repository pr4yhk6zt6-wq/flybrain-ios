//
//  HaltereGyroTests.swift
//  FlyBrainTests
//
//  The gyroscope has to be the *same* gyroscope on the phone as in the report.
//
//  `tools/haltere_gyro.py` derives the pair's sensitivity axes from the body
//  model's own quaternions, measures what the LIF does with them, and prints
//  these numbers with
//
//      python3 tools/haltere_gyro.py --golden build/haltere_golden.json
//
//  The table is chosen so that a port which gets the *sign structure* wrong
//  fails on it, not just a port that gets the magnitudes wrong: it contains a
//  yaw to each side (the difference must reverse), a pitch (both drives must
//  move together), a roll, a rate below `minDPS` (which must read as standing
//  still, exactly), and a mixed rotation. A CI step re-derives the table from
//  the tool and fails if these numbers have drifted from what that run measures,
//  so the test cannot drift with the tool it is checking.
//

import XCTest
import simd
@testable import FlyBrain

final class HaltereGyroTests: XCTestCase {

    /// `[(roll, pitch, yaw) deg/s] -> (left drive, right drive)`, as
    /// `tools/haltere_gyro.py` computed it with the defaults it measured.
    /// Tolerance 1e-9: both sides are double precision and do the same
    /// operations in the same order, so this is an arithmetic identity, not a
    /// numerical-tolerance question.
    private let golden: [(SIMD3<Double>, Double, Double)] = [
        (SIMD3<Double>(0.000000, 0.000000, 0.000000), 3.000000000, 3.000000000),
        (SIMD3<Double>(0.000000, 0.000000, 100.000000), 4.150731821, 1.840029242),
        (SIMD3<Double>(0.000000, 0.000000, -100.000000), 1.849268179, 4.159970758),
        (SIMD3<Double>(0.000000, 0.000000, 400.000000), 7.602927283, -1.639883033),
        (SIMD3<Double>(100.000000, 0.000000, 0.000000), 1.858301137, 4.134694828),
        (SIMD3<Double>(0.000000, 100.000000, 0.000000), 4.171469156, 4.169160163),
        (SIMD3<Double>(0.000000, 0.000000, 0.500000), 3.000000000, 3.000000000),
        (SIMD3<Double>(60.000000, -30.000000, 180.000000), 4.034857213, 1.242121483),
        (SIMD3<Double>(-400.000000, 0.000000, 0.000000), 7.566795451, -1.538779312),
    ]

    func testTheGyroIsTheOneTheToolMeasures() {
        let gyro = HaltereGyro(spec: nil)
        for (dps, left, right) in golden {
            let got = gyro.drive(dps: dps)
            // 1e-8, not 1e-9: the vectors above are the tool's own, at full
            // precision, so the arithmetic is the same — but `simd_dot` is a
            // dot product and a compiler may reassociate it with an fma, which
            // moves the last two bits. The CI step that re-derives this table
            // from the tool keeps 1e-9 because it compares numbers with
            // themselves; this is a port, so it gets the port's tolerance.
            XCTAssertEqual(got.x, left, accuracy: 1e-8,
                           "the port and the tool disagree at \(dps) deg/s: "
                           + "\(got.x) against \(left)")
            XCTAssertEqual(got.y, right, accuracy: 1e-8,
                           "the port and the tool disagree at \(dps) deg/s: "
                           + "\(got.y) against \(right)")
        }
    }

    /// The two channels the pair is *for*, tested on the Swift side so the app
    /// cannot keep a gyro whose signs have been flipped: yaw and roll live in
    /// the difference of the two drives, pitch lives in their sum, and a body
    /// that is not turning gets exactly nothing.
    func testYawIsTheDifferenceAndPitchTheSum() {
        let gyro = HaltereGyro(spec: nil)

        // standing still: both halteres get the same tonic drive, exactly
        let still = gyro.drive(dps: SIMD3(0, 0, 0))
        XCTAssertEqual(still.x, still.y)
        let small = gyro.drive(dps: SIMD3(0, 0, 0.5))
        XCTAssertEqual(small.x, small.y, accuracy: 1e-12,
                       "a rate below minDPS must read as standing still, not as "
                       + "a small rotation")

        // yaw: the difference moves, the common part stays put
        let yawLeft = gyro.drive(dps: SIMD3(0, 0, 200))
        let yawRight = gyro.drive(dps: SIMD3(0, 0, -200))
        XCTAssertGreaterThan(yawLeft.x - yawLeft.y, 0.5,
                             "a leftward yaw does not move the halteres' difference")
        XCTAssertLessThan(yawRight.x - yawRight.y, -0.5,
                          "a rightward yaw does not reverse it")
        // The common channel does not stay *exactly* put: the two sensitivity
        // vectors are mirrors up to a fraction of a percent (the tool measures
        // 0.4% leakage for yaw), so a 200 deg/s yaw moves the sum by 0.009 of
        // the 3.0 tonic drive. What is asserted is that the leakage is small —
        // 20x smaller than the difference channel's response to the same rate.
        XCTAssertEqual(0.5 * (yawLeft.x + yawLeft.y), still.x, accuracy: 0.02,
                       "yaw moved the common channel, which is the pitch channel")

        // pitch: the common part moves, the difference does not
        let pitchUp = gyro.drive(dps: SIMD3(0, 200, 0))
        XCTAssertGreaterThan(0.5 * (pitchUp.x + pitchUp.y) - still.x, 0.5,
                             "a pitch does not move the halteres' common channel")
        XCTAssertEqual(pitchUp.x - pitchUp.y, 0, accuracy: 0.01,
                       "pitch moved the difference, which is the yaw channel")

        // roll is in the difference as well (the circuit model has it there:
        // Fox, Fairhall & Daniel 2010) — the sign is the geometry's, so only
        // the magnitude is asserted.
        let rollLeft = gyro.drive(dps: SIMD3(200, 0, 0))
        XCTAssertGreaterThan(abs(rollLeft.x - rollLeft.y), 0.5,
                             "roll does not reach the difference channel")
    }

    /// The afferent lag: a step in the body's rate reaches the afferents over
    /// `filterMs`, and not instantly. The haltere campaniforms fire phase-locked
    /// to a 200 Hz stroke, so a population cannot follow a rate that changes
    /// faster than a few tens of hertz — and the reflex's *timing* has to come
    /// from somewhere, or the lag is zero and the sensor is a mathematical
    /// idealisation of an animal that does not exist.
    func testTheAfferentLagIsRealAndBounded() {
        var gyro = HaltereGyro(spec: nil)
        let rate = SIMD3<Double>(0, 0, 100)
        let first = gyro.drive(rate: rate, dtMs: 1)
        XCTAssertLessThan(gyro.sensedDPS.z, 30,
                          "one millisecond of a 100 deg/s rate must not arrive as "
                          + "100 deg/s: the afferents have a pole")
        XCTAssertGreaterThan(gyro.sensedDPS.z, 0, "the rate never arrives at all")
        _ = first
        for _ in 0..<60 { _ = gyro.drive(rate: rate, dtMs: 1) }
        XCTAssertEqual(gyro.sensedDPS.z, 100, accuracy: 0.5,
                       "60 ms of a steady 100 deg/s rate has not reached the "
                       + "afferents: the lag is not a first-order pole")
        // and it is the afferents that are slow, not the sensor: the drive for a
        // rate that is already in `sensedDPS` is instantaneous.
        // A first-order lag never *arrives*, it converges: after 61 ms the
        // sensed rate is 99.9999 of 100, which is the point of a pole rather
        // than a delay. The drive for that rate is what the drive for 100 is,
        // to a part in ten thousand.
        let steady = gyro.drive(dps: gyro.sensedDPS)
        let ideal = HaltereGyro(spec: nil).drive(dps: SIMD3(0, 0, 100))
        XCTAssertEqual(steady.x - steady.y, ideal.x - ideal.y, accuracy: 1e-3,
                       "the drive at the converged rate is not the drive at the "
                       + "rate it converged to")
    }
}
