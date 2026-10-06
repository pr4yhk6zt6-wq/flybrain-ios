//
//  WingAeroTests.swift
//  FlyBrainTests
//
//  Item 6: the wing force model, checked against the tool that made it.
//
//  The golden rows in `WingAero` are written by `tools/wing_aero.py --golden`
//  and pasted by `tools/wing_golden_block.py`; CI re-runs the tool and compares
//  the same rows, so a drift on either side fails. The tests below then check
//  the *physics* the rows cannot state: that the two wings are mirrors, that
//  the sign of the yaw is the one the geometry gives, that a symmetric beat
//  turns nothing, and that the pools are calibrated against themselves.
//

import XCTest
import simd
@testable import FlyBrain

final class WingAeroTests: XCTestCase {

    private let aero = WingAero()

    // -- the golden table ------------------------------------------------------

    private func rows(_ text: String) -> [[Double]] {
        text.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ").compactMap { Double($0) }
            return parts.count > 1 ? parts : nil
        }
    }

    private func close(_ a: Double, _ b: Double, _ tol: Double = 1e-9,
                       _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a, b, accuracy: tol * max(1, abs(b)), what, file: file, line: line)
    }

    /// The wing's measured geometry, as the tool wrote it.
    func testGoldenGeometry() {
        let r = rows(WingAero.goldenWing)
        // the head row, then four vectors each for the two wings
        XCTAssertEqual(r.count, 9)
        let head = r[0]
        XCTAssertEqual(head.count, 12)
        close(aero.p.areaCM2, head[1], 1e-12, "planform area")
        close(aero.p.spanCM, head[2], 1e-12, "span, hinge to tip")
        close(aero.p.chordCM, head[4], 1e-12, "mean chord")
        close(aero.p.secondMomentCM4, head[5], 1e-12, "∫r²dA")
        close(aero.p.r2HatCM, head[6], 1e-12, "r̂₂")
        close(aero.p.comCM, head[8], 1e-12, "hinge to centre of mass")
        close(aero.p.sinTheta, head[9], 1e-12, "sinθ, the strip's radius about the axis")
        close(aero.attitudeDeg, head[10], 1e-12, "the attitude the wing force implies")
        close(aero.weightDyn, head[11], 1e-12, "the animal's weight")
        let keys = ["stroke_L", "span_L", "lift_L", "hinge_L",
                    "stroke_R", "span_R", "lift_R", "hinge_R"]
        for (i, key) in keys.enumerated() {
            XCTAssertEqual(r[i + 1].count, 3, "\(key) is a vector")
            XCTAssertEqual(WingAero.goldenWingVectors[i][0], r[i + 1][0], accuracy: 1e-15,
                           "\(key) is carried by the port's parsed vectors too")
        }
        // the right wing's rows are the right wing's, measured — not the left's
        // mirrored, and not the left's repeated
        XCTAssertNotEqual(r[5], r[1], "the right wing's stroke axis is its own")
        for s in WingAero.Side.allCases {
            let axis = aero.strokeAxis(s), span = aero.spanAxis(s), lift = aero.liftDir(s)
            close(simd_length(axis), 1, 1e-12, "the stroke axis is a unit vector")
            close(simd_length(span), 1, 1e-12, "the span axis is a unit vector")
            close(simd_length(lift), 1, 1e-12, "the lift direction is a unit vector")
        }
        // The geometry the flight force rests on: r̂₂ is 55-65% of the span for a
        // fly wing. A second moment taken over a doubled shell reads out at 98%
        // of the span and inflates every force by 3x — this is the check that
        // caught it in the tool.
        let ratio = aero.p.r2HatCM / aero.p.spanCM
        XCTAssertTrue(ratio > 0.5 && ratio < 0.7, "r̂₂ is \(ratio) of the span")
    }

    /// The strips add up to the wing: an integration whose strips do not is an
    /// integration of some other wing.
    func testStripsCoverTheWing() {
        let area = aero.stripAreasCM2.reduce(0, +)
        close(area, aero.p.areaCM2, 1e-12, "the strips' areas sum to the planform")
        let st = rows(WingAero.goldenStations)
        XCTAssertEqual(st.count, aero.p.stations.count)
        for (i, s) in st.enumerated() {
            close(aero.p.stations[i].r, s[0], 1e-12, "strip \(i) radius")
            close(aero.p.stations[i].chord, s[1], 1e-12, "strip \(i) chord")
            close(aero.p.stations[i].area, s[2], 1e-12, "strip \(i) area")
        }
        // and the right wing is the right wing's own wing: its strips are its
        // own table, they add up to its own area, and they are not the left's —
        // the two differ by 6.6e-05, which is the body model's left-right floor
        // and not something to paper over by sharing one table.
        let str = rows(WingAero.goldenStationsRight)
        XCTAssertEqual(str.count, aero.p.stationsRight.count)
        var rightArea = 0.0
        for (i, s) in str.enumerated() {
            close(aero.p.stationsRight[i].r, s[0], 1e-12, "right strip \(i) radius")
            close(aero.p.stationsRight[i].chord, s[1], 1e-12, "right strip \(i) chord")
            close(aero.p.stationsRight[i].area, s[2], 1e-12, "right strip \(i) area")
            rightArea += aero.p.stationsRight[i].area
        }
        close(rightArea / aero.stripAreasCM2.reduce(0, +), 1, 1e-3,
              "the two wings' areas agree to the model's own mirror error")
        XCTAssertNotEqual(aero.p.stations[3].area, aero.p.stationsRight[3].area,
                          "the right wing's strips are its own")
    }

    /// The force law, at 25 (ω, α) points, against the tool's own numbers.
    func testGoldenStripForces() {
        let table = rows(WingAero.goldenStrip)
        XCTAssertEqual(table.count, 25)
        for row in table {
            let f = aero.stripForce(omegaRadS: row[0], alphaDeg: row[1])
            close(f.cl, row[2], 1e-12, "C_L at α = \(row[1])")
            close(f.cd, row[3], 1e-12, "C_D at α = \(row[1])")
            close(f.lift, row[4], 1e-9, "lift at ω = \(row[0]), α = \(row[1])")
            close(f.drag, row[5], 1e-9, "drag at ω = \(row[0]), α = \(row[1])")
        }
        // A still wing makes no force, and the coefficients have to be the
        // animal's: C_L peaks near 45° and C_D peaks near 90°, which is the
        // shape Dickinson et al. measured, not a stand-in.
        let still = aero.stripForce(omegaRadS: 0, alphaDeg: 45)
        XCTAssertEqual(still.lift, 0, accuracy: 1e-15)
        XCTAssertEqual(still.drag, 0, accuracy: 1e-15)
        var bestCL = 0.0, atCL = 0.0, bestCD = 0.0, atCD = 0.0
        for deg in stride(from: 0.0, through: 90.0, by: 1.0) {
            let c = aero.coefficients(alphaDeg: deg)
            if c.cl > bestCL { bestCL = c.cl; atCL = deg }
            if c.cd > bestCD { bestCD = c.cd; atCD = deg }
        }
        XCTAssertTrue(atCL >= 40 && atCL <= 55, "C_L peaks at \(atCL)°")
        XCTAssertTrue(atCD >= 75 && atCD <= 90, "C_D peaks at \(atCD)°")
    }

    /// Whole beats, each wing driven by its own activation — the shape the app
    /// drives them in.
    func testGoldenBeats() {
        let table = rows(WingAero.goldenBeat)
        XCTAssertEqual(table.count, 9)
        for row in table {
            let b = aero.beat(activationLeft: row[0], activationRight: row[1])
            close(b.amplitudeLeftDeg, row[2], 1e-12, "left amplitude")
            close(b.amplitudeRightDeg, row[3], 1e-12, "right amplitude")
            close(b.flightForceLeftDyn, row[4], 1e-9, "left flight force")
            close(b.flightForceRightDyn, row[5], 1e-9, "right flight force")
            close(b.flightForceDyn, row[6], 1e-9, "flight force")
            close(b.thrustDyn, row[7], 1e-9, "fore-aft force")
            close(b.verticalDyn, row[8], 1e-9, "vertical force")
            close(b.sideDyn, row[9], 1e-9, "sideways force")
            close(b.yawMomentDynCM, row[10], 1e-9, "yaw moment")
            close(b.rollMomentDynCM, row[11], 1e-9, "roll moment")
            close(b.pitchMomentDynCM, row[12], 1e-9, "pitch moment")
            close(b.dragCostDyn, row[13], 1e-9, "the drag the beat costs")
        }
    }

    // -- the physics the table cannot state ------------------------------------

    /// The two wings are mirrors: same force on both, no sideways push, no yaw.
    func testASymmetricBeatTurnsNothing() {
        let b = aero.beat(activationLeft: 1, activationRight: 1)
        // 1e-4, not 1e-9: the two wings in the body model are not identical.
        // The tool measures the difference — 6.6e-05 in area, 7.5e-05 in ∫r²dA
        // — which leaves 1.6e-05 dyn between the two wings' flight force. That
        // is the model's own floor, and it is the floor under every yaw number
        // here; the bound is six times it, so a real asymmetry cannot hide in
        // it, and a model defect that made the wings unequal by more than the
        // meshes do would still fail.
        close(b.flightForceLeftDyn, b.flightForceRightDyn, 1e-4,
              "a symmetric beat drives the two wings equally")
        // The floor is the body model's own: its two wing meshes are mirrors to
        // ~2e-6, so the leak is ~0.3% of the steering signal and not zero. The
        // tool measures the same number; pretending it is zero would hide a
        // model defect if one ever appeared.
        let signal = abs(aero.beat(activationLeft: 1, activationRight: 0.2).yawMomentDynCM)
        XCTAssertTrue(abs(b.yawMomentDynCM) < 0.02 * signal,
                      "a symmetric beat leaks \(b.yawMomentDynCM) of yaw against \(signal)")
        XCTAssertTrue(abs(b.sideDyn) < 1e-3, "a symmetric beat pushes sideways by \(b.sideDyn)")
        XCTAssertGreaterThan(b.dragCostDyn, 0.5, "the beat costs drag")
    }

    /// A left-right asymmetry reverses the yaw, and by the sign the geometry
    /// gives: the harder-driven wing pushes its own side of the body forward, so
    /// the nose goes the other way.
    func testSteeringSign() {
        let left = aero.beat(activationLeft: 1.0, activationRight: 0.5)
        let right = aero.beat(activationLeft: 0.5, activationRight: 1.0)
        XCTAssertLessThan(left.yawMomentDynCM, 0,
                          "driving the left wing harder must yaw the animal right")
        XCTAssertGreaterThan(right.yawMomentDynCM, 0,
                             "driving the right wing harder must yaw the animal left")
        XCTAssertGreaterThan(abs(left.yawMomentDynCM), 1e-3, "and it has to be a real turn")
        XCTAssertGreaterThan(left.rollMomentDynCM, 0, "the harder wing rolls the animal away")
        // The yaw and the roll of an amplitude asymmetry are the same lever with
        // the same force, resolved on two axes: their ratio is the ratio of the
        // stroke plane's fore-aft to vertical components.
        let ratio = abs(left.yawMomentDynCM / left.rollMomentDynCM)
        let expected = abs(aero.p.liftLeft.x / aero.p.liftLeft.z)
        XCTAssertEqual(ratio, expected, accuracy: 0.25 * expected,
                       "yaw and roll come from the same lever")
    }

    /// The force is 47° forward of vertical, so the animal has to hold a nose-up
    /// attitude for its wingbeat to carry it — and at exactly that attitude the
    /// force is vertical *and* equals its weight. The geometry and the flight are
    /// one fact; a model that faked either would not close like this.
    func testTheAttitudeIsTheOneThatCarriesTheAnimal() {
        let b = aero.beat(activationLeft: 1, activationRight: 1)
        let theta = aero.attitudeDeg * .pi / 180
        // rotate the beat's force by a nose-up pitch and it becomes vertical
        let fx = b.thrustDyn * cos(theta) - b.verticalDyn * sin(theta)
        let fz = b.thrustDyn * sin(theta) + b.verticalDyn * cos(theta)
        XCTAssertEqual(fx / max(abs(fz), 1e-12), 0, accuracy: 0.05,
                       "the force is not vertical at the measured attitude")
        close(fz, aero.weightDyn, 0.05, "at that attitude the force is the weight")
        XCTAssertTrue(aero.attitudeDeg > 30 && aero.attitudeDeg < 60,
                      "the attitude is \(aero.attitudeDeg)° nose-up")
    }

    /// The angle of attack that makes the flight force equal the animal's
    /// weight, solved for the way the tool solves it — the animal's own
    /// mid-stroke α is 45-50°, which is the strongest evidence that the force
    /// model is measuring the animal and not a toy.
    func testTheHoveringAngleOfAttack() {
        var lo = 5.0, hi = 85.0
        for _ in 0..<40 {
            let mid = 0.5 * (lo + hi)
            var a = aero
            a.p.alphaMidDeg = mid
            if a.beat(activationLeft: 1, activationRight: 1).flightForceDyn < aero.weightDyn {
                lo = mid
            } else {
                hi = mid
            }
        }
        let solved = 0.5 * (lo + hi)
        XCTAssertTrue(solved > 40 && solved < 52,
                      "the hovering angle of attack solves to \(solved)°")
    }

    /// The beat is a beat: the force pulses, and at the stroke reversal the
    /// strips are momentarily still.
    func testTheBeatPulses() {
        var forces: [Double] = [], atReversal = 0.0
        for i in 0..<200 {
            let phase = Double(i) / 200
            let w = aero.instantaneous(phase: phase, activationLeft: 1, activationRight: 1)
            forces.append(simd_length(w.force))
            if abs(phase - 0.25) < 1e-9 { atReversal = simd_length(w.force) }
        }
        let peak = forces.max() ?? 0
        XCTAssertGreaterThan(peak, 1.5, "the beat's peak force is \(peak) dyn")
        XCTAssertLessThan(atReversal, 0.05 * peak,
                          "the strips are still at the stroke reversal")
        // and the mean of the instantaneous wrench agrees with the beat's mean
        var mean = SIMD3<Double>.zero
        for i in 0..<2000 {
            mean += aero.instantaneous(phase: Double(i) / 2000,
                                       activationLeft: 1, activationRight: 1).force / 2000
        }
        let b = aero.beat(activationLeft: 1, activationRight: 1)
        close(mean.z, b.verticalDyn, 0.02, "the sampled beat is the integrated beat")
    }

    /// The pools are calibrated against themselves, or the animal flies in a
    /// circle: the two power pools do not run at the same rate.
    func testThePowerPoolsAreNotSymmetric() {
        let l = aero.p.rateReference[aero.p.powerLeft] ?? 0
        let r = aero.p.rateReference[aero.p.powerRight] ?? 0
        XCTAssertGreaterThan(l, 0)
        XCTAssertGreaterThan(r, 0)
        XCTAssertGreaterThan(abs(l - r) / max(l, r), 0.05,
                             "the pools differ by more than 5%: one shared "
                             + "reference would hold an asymmetry open")
        // resting below full, both of them, or "at rest" would not be at rest
        for name in [aero.p.powerLeft, aero.p.powerRight] {
            let rest = aero.p.rateTone[name] ?? 0
            XCTAssertLessThan(rest, aero.p.rateReference[name] ?? 0)
            XCTAssertGreaterThan(rest, 0.5 * (aero.p.rateReference[name] ?? 0))
        }
    }

    /// Activation from rates: at rest it is the resting activation, a full-drive
    /// rate is full activation, and the two pools are not the same.
    func testActivationFromRates() {
        var rest = aero.p.rateTone
        let atRest = aero.activation(ratesHz: rest)
        close(atRest.left, (aero.p.rateTone[aero.p.powerLeft] ?? 0) / (aero.p.rateReference[aero.p.powerLeft] ?? 1),
              1e-12, "at rest the left activation is the resting activation")
        rest[aero.p.powerLeft] = aero.p.rateReference[aero.p.powerLeft] ?? 0
        rest[aero.p.steeringLeft] = aero.p.rateTone[aero.p.steeringLeft] ?? 0
        let atFull = aero.activation(ratesHz: rest)
        close(atFull.left, 1.0, 1e-9, "a full-drive power pool is full activation")
        // and the steering pool adds amplitude on top of the power pool
        var steered = aero.p.rateTone
        steered[aero.p.steeringLeft] = (aero.p.rateTone[aero.p.steeringLeft] ?? 0)
            + 0.5 * (aero.p.rateReference[aero.p.steeringLeft] ?? 0)
        XCTAssertGreaterThan(aero.activation(ratesHz: steered).left, atRest.left,
                             "driving the left steering pool harder opens the left stroke")
        let b = aero.beat(activationLeft: aero.activation(ratesHz: steered).left,
                          activationRight: aero.activation(ratesHz: steered).right)
        XCTAssertLessThan(b.yawMomentDynCM, 0, "and the animal yaws right")
    }

    /// The pool names are the ones the packed connectome carries; if they ever
    /// change, this fails rather than silently driving nothing.
    func testPoolNames() {
        for name in [aero.p.powerLeft, aero.p.powerRight,
                     aero.p.steeringLeft, aero.p.steeringRight] {
            XCTAssertTrue(name.hasPrefix("motor_wing_"), name)
            XCTAssertNotNil(aero.p.rateReference[name], "no reference rate for \(name)")
            XCTAssertNotNil(aero.p.rateTone[name], "no resting rate for \(name)")
        }
        let drives = aero.poolDrives(tone: 2.5, steer: 0.9)
        XCTAssertEqual(drives[aero.p.powerLeft], 2.5)
        XCTAssertEqual(drives[aero.p.steeringLeft], 3.4)
        XCTAssertEqual(drives[aero.p.steeringRight], 1.6)
        // a wheel that is asked for more steer than the tone can pay for must
        // not go negative: a negative drive is not a thing a cord can send
        let hard = aero.poolDrives(tone: 2.5, steer: 9.0)
        XCTAssertEqual(hard[aero.p.steeringRight], 0)
    }

    /// The model's own wing mass is a placeholder and the tool says so; the
    /// inertial force is reported with the caveat attached rather than quietly
    /// used as if it were the animal's.
    func testTheWingMassIsFlagged() {
        XCTAssertEqual(aero.p.massUG, 8.0, accuracy: 1e-9)
        XCTAssertGreaterThan(aero.inertialForceDyn(), 3.0)
        // with a real wing mass (~1 µg) the inertial force is comparable with
        // the lift, which is the animal's actual situation
        let real = aero.inertialForceDyn() / 8.0
        let liftPerWing = aero.beat(activationLeft: 1, activationRight: 1).flightForceLeftDyn
        XCTAssertTrue(real > 0.2 * liftPerWing && real < 3 * liftPerWing,
                      "a real wing's inertia (\(real) dyn) is comparable with its "
                      + "lift (\(liftPerWing) dyn)")
    }
}
