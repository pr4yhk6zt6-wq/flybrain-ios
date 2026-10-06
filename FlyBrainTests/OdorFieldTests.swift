//
//  OdorFieldTests.swift
//  FlyBrainTests
//
//  The smell has to be the *same* smell on the phone as in the report.
//
//  `tools/odor_field.py` defines the plume and measures what driving the
//  olfactory receptor neurons from it does to the brain (reports/item11_odor.md).
//  `OdorField.swift` is that field in the language the phone runs, and these are
//  the numbers the two have to agree on, printed by
//
//      python3 tools/odor_field.py --golden build/odor_golden.json
//
//  The samples are chosen, not stepped: they include clean air between filaments
//  (drive exactly 0), the peak of a filament, the crosswind falloff, a point
//  far downwind, and a point *upwind* of the source, where a plume cannot go and
//  the drive must be zero exactly. A port that gets the filaments right but the
//  downwind falloff wrong fails on the same table.
//

import XCTest
import simd
@testable import FlyBrain

final class OdorFieldTests: XCTestCase {

    /// The field as `python3 tools/odor_field.py` computed it. Tolerance 1e-9:
    /// both sides are double precision and do the same operations in the same
    /// order, so this is not a numerical-tolerance question, it is an
    /// arithmetic-identity question.
    private let golden: [(SIMD3<Double>, Double, Double)] = [
        (SIMD3<Double>(0.052000, 0.000000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(0.052000, 0.000000, -0.009000), 364.0000, 13.155658983),
        (SIMD3<Double>(0.052000, 0.000000, -0.009000), 382.0000, 25.924177144),
        (SIMD3<Double>(-0.700000, 0.000000, -0.009000), 16.0000, 7.710491484),
        (SIMD3<Double>(-0.700000, 0.000000, -0.009000), 46.0000, 0.000000000),
        (SIMD3<Double>(-0.700000, 0.000000, -0.009000), 286.0000, 3.741388362),
        (SIMD3<Double>(-1.200000, 0.000000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(-1.200000, 0.000000, -0.009000), 138.0000, 2.023172286),
        (SIMD3<Double>(-1.200000, 0.000000, -0.009000), 152.0000, 3.805948910),
        (SIMD3<Double>(0.700000, 0.000000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(-2.700000, 0.000000, -0.009000), 20.0000, 0.000000000),
        (SIMD3<Double>(-2.700000, 0.000000, -0.009000), 276.0000, 0.597243373),
        (SIMD3<Double>(-2.700000, 0.000000, -0.009000), 290.0000, 0.298744259),
        (SIMD3<Double>(-0.700000, -0.100000, -0.009000), 16.0000, 7.266157278),
        (SIMD3<Double>(-0.700000, -0.100000, -0.009000), 46.0000, 0.000000000),
        (SIMD3<Double>(-0.700000, -0.100000, -0.009000), 286.0000, 3.504287755),
    ]

    func testTheFieldIsTheOneTheToolMeasures() {
        let field = OdorField(spec: nil)
        for (point, tMs, expected) in golden {
            let got = field.drive(at: point, tMs: tMs)
            XCTAssertEqual(got, expected, accuracy: 1e-9,
                           "the port and the tool disagree at \(point) at "
                           + "\(tMs) ms: \(got) against \(expected)")
        }
    }

    /// The three properties the field is *for*, tested on the Swift side so the
    /// app cannot keep a field that stopped being a plume.
    ///
    /// Points here are written as distances downwind of the source (`down(_:)`)
    /// rather than as world coordinates, because the source is a place in the
    /// world and the field's shape is not: `source` is 2.5 mm in front of the
    /// animal and the air blows onto it, so 1 cm downwind is *behind* the
    /// animal, and upwind of the source is in front of it.
    func testItIsAPlumeAndNotACloud() {
        let field = OdorField(spec: nil)
        // the wind is -x, so `s` downwind of a source at x = +0.30 is x = 0.30 - s
        let down = { (s: Double) in
            field.meanConcentration(at: SIMD3(0.30 - s, 0, -0.009))
        }
        XCTAssertGreaterThan(down(0.05), down(0.2))
        XCTAssertGreaterThan(down(0.2), down(0.8))
        XCTAssertEqual(field.meanConcentration(at: SIMD3(0.7, 0, -0.009)), 0,
                       "a plume does not reach upwind of its source")

        // A filament passes: over one gust cycle at a fixed point the
        // concentration must both reach a peak and fall back to nothing.
        var peak = 0.0
        var zero = 0.0
        let period = 1000.0 / field.p.gustHz
        for i in 0..<400 {
            let c = field.concentration(at: SIMD3(0.052, 0, -0.009),
                                        tMs: Double(i) * period / 400)
            peak = max(peak, c)
            zero += c == 0 ? 1 : 0
        }
        XCTAssertGreaterThan(peak, 0, "the field never lights up")
        XCTAssertGreaterThan(zero, 20, "there is no clean air between filaments: "
                             + "\(zero) of 400 samples are exactly zero")
    }
}
