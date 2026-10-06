//
//  HaltereGyro.swift
//  FlyBrain
//
//  The haltere pair, as the animal's gyroscope.
//
//  Brief item 7: the halteres are physically in the model already (two stubs on
//  the metathorax, 0.822 µg each, one hinge joint apiece) and nothing has ever
//  read them. This is the sensor: the body's angular velocity, felt as the
//  Coriolis force on two oscillating masses, turned into a current on the two
//  afferent populations by name.
//
//  Why the pair and not one haltere: the Coriolis force is `2 m (Ω × v)`, i.e.
//  linear in the rate and *sign-flipped* if the stroke reverses. For a yaw the
//  two halteres move in opposite directions along the midline, so the force
//  they feel is antisymmetric; for a pitch it is the same on both. `a_L - a_R`
//  therefore receives yaw and roll, and `a_L + a_R` receives pitch, which is
//  what lets a fly tell a turn from a gust (Fox, Fairhall & Daniel 2010,
//  *Front. Neural Circuits* 4:123). The two vectors here are measured from the
//  body model's own quaternions and written into `world.json` by
//  `tools/haltere_gyro.py --write`, so the app does not carry a second copy.
//
//  Pinned to the tool by `HaltereGyroTests`, whose expected numbers are printed
//  by `python3 tools/haltere_gyro.py --golden`.
//

import Foundation
import simd

/// What `tools/haltere_gyro.py --write` puts in `world.json`.
struct HaltereSpec: Decodable {
    let sensitivity_left: [Double]?
    let sensitivity_right: [Double]?
    let mass_ug: Double?
    let com_mm: Double?
    let stroke_hz: Double?
    let stroke_deg: Double?
    let bias: Double?
    let gain: Double?
    let reference_dps: Double?
    let min_dps: Double?
    let filter_ms: Double?
    let convention: String?
    let group_left: String?
    let group_right: String?
}

struct HaltereGyro {

    struct Params {
        /// The sensitivity axis of each haltere, body frame, measured from the
        /// body model (`a = v̂ × n̂`, the direction of the Coriolis force per
        /// unit rate). The defaults are the tool's, and a world written by
        /// `--write` carries its own.
        ///
        /// Note the y components: `a_left` is (−, +, +) and `a_right` is (+, +, −)
        /// — the two are **not** negatives of each other, because a cross product
        /// is a pseudo-vector and a mirror flips exactly one of its components
        /// (`v̂ × n̂` under y → −y comes back negated, i.e. −S·a). With the
        /// antiphase stroke convention the right-hand axis is minus the mirror of
        /// the left's, which is what these numbers are; the first version of this
        /// file typed the right-hand axis by hand as the plain mirror, and the
        /// pinned golden table failed on it, which is what the table is for.
        var aLeft = SIMD3<Double>(-0.570849431316043, 0.5857345780637347, 0.5753659103793418)
        var aRight = SIMD3<Double>(0.5673474139829302, 0.5845800815918925, -0.5799853791716633)
        /// The tonic current a *stroking* haltere's in-plane forces hold the
        /// afferents at: the fields are tonically active whenever the haltere
        /// moves, and the gyro signal rides on top of that.
        var bias = 3.0
        /// Current per unit of normalised rate (`Ω·a / referenceDPS`).
        var gain = 2.0
        var referenceDPS = 100.0
        /// Below this the body is not turning and the drive is exactly `bias`:
        /// "not moving" has to read as not moving.
        var minDPS = 1.0
        /// The afferents fire phase-locked to a 200 Hz stroke, so the
        /// population cannot follow a rate instantaneously. One first-order lag
        /// stands in for that (assumption #31), and it is short enough not to
        /// be the slowest thing in the reflex.
        var filterMs = 5.0
        var groupLeft = "sensory_haltere_left"
        var groupRight = "sensory_haltere_right"
    }

    var p: Params
    /// The rate the sensor last reported, after the lag — what the HUD shows
    /// and what makes "the gyro is reading something" visible rather than
    /// inferred from a spike count.
    private(set) var sensedDPS = SIMD3<Double>.zero

    init(spec: HaltereSpec? = nil) {
        var q = Params()
        if let s = spec {
            if let v = s.sensitivity_left, v.count >= 3 { q.aLeft = SIMD3(v[0], v[1], v[2]) }
            if let v = s.sensitivity_right, v.count >= 3 { q.aRight = SIMD3(v[0], v[1], v[2]) }
            q.bias = s.bias ?? q.bias
            q.gain = s.gain ?? q.gain
            q.referenceDPS = s.reference_dps ?? q.referenceDPS
            q.minDPS = s.min_dps ?? q.minDPS
            q.filterMs = s.filter_ms ?? q.filterMs
            q.groupLeft = s.group_left ?? q.groupLeft
            q.groupRight = s.group_right ?? q.groupRight
        }
        p = q
    }

    /// The two drives, in the app's tone units, for a body-frame angular
    /// velocity in degrees per second.
    ///
    /// `(roll, pitch, yaw)` — x forward, y left, z up — which is the frame the
    /// body model's own quaternions put the sensitivity axes in.
    func drive(dps: SIMD3<Double>) -> SIMD2<Double> {
        var omega = dps
        if simd_length(omega) < p.minDPS { omega = .zero }
        let left = p.bias + p.gain * simd_dot(omega, p.aLeft) / p.referenceDPS
        let right = p.bias + p.gain * simd_dot(omega, p.aRight) / p.referenceDPS
        return SIMD2(left, right)
    }

    /// The same, with the afferent lag applied to the rate first. `dtMs` is one
    /// cord update; the filter runs because `sensedDPS` is state.
    mutating func drive(rate dps: SIMD3<Double>, dtMs: Double) -> SIMD2<Double> {
        let alpha = p.filterMs > 0 ? min(1, dtMs / p.filterMs) : 1
        sensedDPS += (dps - sensedDPS) * alpha
        return drive(dps: sensedDPS)
    }
}
