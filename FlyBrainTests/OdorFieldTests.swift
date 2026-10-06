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
        (SIMD3<Double>(0.300000, 0.000000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(0.300000, 0.000000, -0.009000), 128.0000, 0.685601585),
        (SIMD3<Double>(0.300000, 0.000000, -0.009000), 396.0000, 1.367105636),
        (SIMD3<Double>(0.900000, 0.000000, -0.009000), 4.0000, 0.250193741),
        (SIMD3<Double>(0.900000, 0.000000, -0.009000), 18.0000, 0.000000000),
        (SIMD3<Double>(0.900000, 0.000000, -0.009000), 274.0000, 0.517387282),
        (SIMD3<Double>(0.300000, 0.100000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(0.300000, 0.100000, -0.009000), 94.0000, 0.574272943),
        (SIMD3<Double>(0.300000, 0.100000, -0.009000), 396.0000, 1.104873524),
        (SIMD3<Double>(-0.400000, 0.000000, -0.009000), 0.0000, 0.000000000),
        (SIMD3<Double>(3.000000, 0.000000, -0.009000), 20.0000, 0.000000000),
        (SIMD3<Double>(3.000000, 0.000000, -0.009000), 276.0000, 0.034507395),
        (SIMD3<Double>(3.000000, 0.000000, -0.009000), 290.0000, 0.017260779),
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
    func testItIsAPlumeAndNotACloud() {
        let field = OdorField(spec: nil)
        let onAxis = { (s: Double) in
            field.meanConcentration(at: SIMD3(s, 0, -0.009))
        }
        XCTAssertGreaterThan(onAxis(0.05), onAxis(0.2))
        XCTAssertGreaterThan(onAxis(0.2), onAxis(0.8))
        XCTAssertEqual(field.meanConcentration(at: SIMD3(-0.4, 0, -0.009)), 0,
                       "a plume does not reach upwind of its source")

        // A filament passes: over one gust cycle at a fixed point the
        // concentration must both reach a peak and fall back to nothing.
        var peak = 0.0
        var zero = 0.0
        let period = 1000.0 / field.p.gustHz
        for i in 0..<400 {
            let c = field.concentration(at: SIMD3(0.30, 0, -0.009),
                                        tMs: Double(i) * period / 400)
            peak = max(peak, c)
            zero += c == 0 ? 1 : 0
        }
        XCTAssertGreaterThan(peak, 0, "the field never lights up")
        XCTAssertGreaterThan(zero, 20, "there is no clean air between filaments: "
                             + "\(zero) of 400 samples are exactly zero")
    }
}
