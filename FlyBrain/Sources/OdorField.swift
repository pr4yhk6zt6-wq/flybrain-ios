//
//  OdorField.swift
//  FlyBrain
//
//  The smell in the world, and what it does to the antenna.
//
//  Brief item 11: odor as a spatial and temporal field. `tools/odor_field.py`
//  defines the field and measures that driving the olfactory receptor neurons
//  from it reaches the brain; this is the same field in the language the phone
//  runs, so that the animal on the screen is standing in the same smell as the
//  animal in the report.
//
//  The two implementations are pinned to each other by
//  `OdorFieldTests.testTheFieldIsTheOneTheToolMeasures`, whose expected numbers
//  are printed by `python3 tools/odor_field.py --golden`.
//
//  What it is: a kinematic plume with literature-shaped statistics — a mean
//  that decays downwind and spreads crosswind, multiplied by a duty cycle of
//  filaments separated by clean air. What it is not: a solution of the
//  advection–diffusion equation. Every constant is labelled in
//  docs/ASSUMPTIONS.md (#26–#29).
//

import Foundation

struct OdorField {
    /// The field's constants. The defaults are the Python tool's defaults, and
    /// a world written by `tools/odor_field.py --write` carries its own — the
    /// app reads them rather than keeping a second copy, the same way the
    /// camera limits come from the `view` block.
    struct Params {
        var source = SIMD3<Double>(-0.30, 0.0, 0.0)   // 3 mm upwind of the start
        var wind = SIMD3<Double>(0.30, 0.0, 0.0)      // cm/s
        var c0 = 1.0
        var decayCm = 1.20
        var sigma0Cm = 0.045
        var widening = 0.22
        var s0Cm = 0.05
        var duty = 0.20
        var gustHz = 3.5
        var wavenumber = 6.0
        var baselineDrive = 2.6
    }

    let p: Params
    /// Unit wind: the direction the smell travels in.
    let windHat: SIMD3<Double>

    init(spec: WorldManifest.OdorSpec? = nil) {
        var q = Params()
        if let s = spec {
            if let v = s.source_cm, v.count >= 3 {
                q.source = SIMD3(v[0], v[1], v[2])
            }
            if let v = s.wind_cms, v.count >= 3 {
                q.wind = SIMD3(v[0], v[1], v[2])
            }
            q.c0 = s.c0 ?? q.c0
            q.decayCm = s.decay_cm ?? q.decayCm
            q.sigma0Cm = s.sigma0_cm ?? q.sigma0Cm
            q.widening = s.widening ?? q.widening
            q.s0Cm = s.s0_cm ?? q.s0Cm
            q.duty = s.duty ?? q.duty
            q.gustHz = s.gust_hz ?? q.gustHz
            q.wavenumber = s.wavenumber ?? q.wavenumber
            q.baselineDrive = s.baseline_drive ?? q.baselineDrive
        }
        p = q
        let n = (q.wind * q.wind).sum().squareRoot()
        windHat = n > 0 ? q.wind / n : SIMD3(1, 0, 0)
    }

    /// Where the odor comes from, so the world can draw it.
    var source: SIMD3<Double> { p.source }

    /// Downwind distance and crosswind radius of a point relative to the
    /// source, and whether the point is downwind at all. A plume does not
    /// reach upwind: there is no advection there, and a mean that pretends
    /// otherwise makes the animal smell its food from behind a wall.
    private func geometry(_ point: SIMD3<Double>) -> (s: Double, r: Double, downwind: Bool) {
        let d = point - p.source
        let raw = (d * windHat).sum()
        let s = max(raw, 0)
        let across = d - s * windHat
        return (s, (across * across).sum().squareRoot(), raw >= 0)
    }

    /// The time-mean concentration, before the filaments.
    func meanConcentration(at point: SIMD3<Double>) -> Double {
        let g = geometry(point)
        guard g.downwind else { return 0 }
        let sigma = p.sigma0Cm + p.widening * g.s
        return p.c0 * (p.s0Cm / (p.s0Cm + g.s))
            * exp(-(g.r * g.r) / (2 * sigma * sigma))
            * exp(-g.s / p.decayCm)
    }

    /// The concentration a receptor actually sees: filaments passing through
    /// clean air. Inside a filament the concentration is a raised cosine rising
    /// to `2/duty` times the mean; outside it, zero.
    func concentration(at point: SIMD3<Double>, tMs: Double) -> Double {
        let g = geometry(point)
        let mean = meanConcentration(at: point)
        guard mean > 0 else { return 0 }
        let x = p.gustHz * (tMs / 1000.0) - p.wavenumber * g.s / (2 * Double.pi)
        let phase = x - x.rounded(.down)                  // mod 1, always in [0,1)
        guard phase < p.duty else { return 0 }
        let w = 0.5 * (1 - cos(2 * Double.pi * phase / p.duty))
        return mean * (2 / p.duty) * w
    }

    /// The external current the olfactory receptor neurons get, in the app's
    /// tone units: a saturating map of the concentration (assumption #28).
    func drive(at point: SIMD3<Double>, tMs: Double) -> Double {
        let c = concentration(at: point, tMs: tMs)
        return p.baselineDrive * c / (1 + c)
    }
}
