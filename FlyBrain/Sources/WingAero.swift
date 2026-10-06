//
//  WingAero.swift
//  FlyBrain
//
//  The wings, as the animal's aerodynamic surfaces.
//
//  Brief item 6: the two wings are in the body model already — 8 µg apiece, a
//  membrane and a hinge plate three joints deep — and nothing has ever pushed
//  air with them. `build_body.py` drives the legs and only the legs. This is the
//  missing half: the force the wings make, computed from the wing's *measured*
//  geometry and the animals' own measured coefficients, handed to the physics as
//  a force and a torque about the thorax. The physics then decides where the
//  animal goes; there is no trajectory anywhere in this file.
//
//  The model is the standard quasi-steady one, and it is not a stand-in for
//  something better — it is the thing Dickinson, Lehmann & Sane measured off
//  tethered *Drosophila* in 1999, with their fits:
//
//      C_L = 0.225 + 1.58 sin(2.13 α − 7.2°)      (α in radians)
//      C_D = 1.92  − 1.55 cos(2.13 α − 9.8°)
//
//  Each strip of the wing (12 of them, by measured chord) sees `v = ω r sinθ`
//  about the stroke axis, gets `½ρv²A C_L` along the stroke plane's normal and
//  `½ρv²A C_D` against its motion. The lift direction is *constant per wing* —
//  a wing pitching about its own span keeps its normal perpendicular to the
//  velocity — so a stroke plane that is inclined 47° forward makes a force that
//  is inclined 47° forward, and the animal has to hold a nose-up attitude for
//  that force to be weight support. `attitude_deg` in `world.json` is that
//  angle, measured from our own body model, and the app is expected to fly at
//  it rather than to be told about it.
//
//  Everything here is pinned to `tools/wing_aero.py` by `WingAeroTests`, whose
//  expected numbers are written by
//  `python3 tools/wing_aero.py --golden build/wing_golden.json`.
//
//  Three conventions that are *physics*, not taste, and that a reader should
//  check rather than trust:
//
//    * the two wings' stroke axes are antiparallel mirrors, so the same joint
//      angle sweeps both wings the same way. The body model's own kinematics
//      are asked to confirm this (the tool's `--gate` measures it), because the
//      alternative — a plain mirror — would drive the wings against each other;
//    * the wing that is driven harder sweeps harder, so it pushes *its* side of
//      the body forward and the animal's nose goes the other way. A flipped
//      sign here is the kind that gets "fixed" downstream and turns the animal
//      into a circle-flyer;
//    * a symmetric beat makes a yaw of zero, exactly, by mirror symmetry. The
//      tool measures its mirror-symmetry floor (0.27% of the steering signal)
//      rather than pretending it is zero: the body model's two wing meshes are
//      mirrors only to ~2e-6.
//

import Foundation
import simd

/// What `tools/wing_aero.py --write` puts in `world.json` under `wings`.
struct WingSpec: Decodable {
    let area_cm2: Double?
    let span_cm: Double?
    let chord_cm: Double?
    let second_moment_cm4: Double?
    let r2_hat_cm: Double?
    let sin_theta: Double?
    let mass_ug: Double?
    let com_cm: Double?
    let hinge_left_cm: [Double]?
    let hinge_right_cm: [Double]?
    let span_axis_left: [Double]?
    let span_axis_right: [Double]?
    let lift_dir_left: [Double]?
    let lift_dir_right: [Double]?
    let stroke_axis_left: [Double]?
    let stroke_axis_right: [Double]?
    let stroke_hz: Double?
    let amplitude_full_deg: Double?
    let alpha_mid_deg: Double?
    let alpha_rot_deg: Double?
    let rho_kg_m3: Double?
    let body_mass_ug: Double?
    let flight_force_dyn: Double?
    let lift_over_weight: Double?
    let attitude_deg: Double?
    let alpha_hover_deg: Double?
    let drag_cost_dyn: Double?
    let yaw_moment_full_dyn_cm: Double?
    let rate_reference_hz: [String: Double]?
    let rate_tone_hz: [String: Double]?
    let power_left: String?
    let power_right: String?
    let steering_left: String?
    let steering_right: String?
    let stations: [Station]?
    let stations_right: [Station]?
    let wing_asymmetry: [String: Double]?
    let mass_ug_source: String?

    struct Station: Decodable {
        let r_cm: Double
        let chord_cm: Double
        let area_cm2: Double?
    }
}

struct WingAero {

    /// The wing's constants. The defaults are the measured values from the
    /// round of `--probe --write` that first pinned this file; a `world.json`
    /// written by `--write` carries its own, and the app prefers those.
    struct Params {
        // -- the wing, measured out of the body model ----------------------
        var areaCM2 = 0.01740760852663423          // planform, both faces halved
        var spanCM = 0.26466228266454234           // hinge to tip
        var chordCM = 0.0662630654372785           // mean chord = A / length
        var secondMomentCM4 = 4.164410152235529e-04   // ∫r² dA about the hinge
        var r2HatCM = 0.15467039425878712          // √(∫r²dA / A)
        /// The strips are binned by their distance from the hinge; their speed
        /// is set by their distance from the stroke *axis*, which is that
        /// distance times this. It is 0.995 for this wing — the wing is very
        /// nearly in the stroke plane — but it is measured, not assumed.
        var sinTheta = 0.9951448997514517
        var massUG = 8.0                           // ENGINEERING PLACEHOLDER (#35)
        var comCM = 0.1530715518964905
        // The four vectors of each wing — its hinge, its span, its stroke
        // axis and its stroke plane's normal — all read out of the golden
        // block below, where the tool writes them. Both wings' are *measured*:
        // the right wing's are not mirrored from the left's, because the two
        // meshes are mirrors only to ~2e-6 and copying that error into the port
        // would put a number the port cannot account for into every left-right
        // comparison. The gate measures the floor instead, where it is visible.
        //
        // The lift direction is each wing's *stroke plane normal*, not the
        // mesh's rest plane: the wings sit out flat in the rest pose, and
        // pitching away from *that* flies them edge-on — measured, and what the
        // tool's `--lift-from rest_plane` rehearsal is for. The pair's normals
        // are mirrors of each other and both point up, which is the one
        // convention the fly forces on us: a positive α has to make lift.
        var hingeLeft = WingAero.goldenWingVectors[3]
        var hingeRight = WingAero.goldenWingVectors[7]
        var spanLeft = WingAero.goldenWingVectors[1]
        var spanRight = WingAero.goldenWingVectors[5]
        var liftLeft = WingAero.goldenWingVectors[2]
        var liftRight = WingAero.goldenWingVectors[6]
        var strokeLeft = WingAero.goldenWingVectors[0]
        var strokeRight = WingAero.goldenWingVectors[4]

        // -- the beat ------------------------------------------------------
        var strokeHz = 200.0
        /// The stroke amplitude at full drive. 77.5° is the animal's measured
        /// amplitude (APPROXIMATION #34); the joint's own limit is 82°, so the
        /// model can drive the wing to its limit but not past it.
        var amplitudeFullDeg = 77.5
        /// The mid-stroke angle of attack, and the sweep about it as the stroke
        /// reverses (APPROXIMATION #36: Dickinson et al. measured 90-180° of
        /// rotation per reversal; this is a smooth stand-in for it).
        var alphaMidDeg = 45.0
        var alphaRotDeg = 30.0
        var rhoGCM3 = 1.225e-3
        var bodyMassUG = 985.0

        /// The strips: distance from the hinge and chord, in cm, as measured.
        /// Measured: the mesh binned into 12 strips. The areas are the bins'
        /// own sums, so a port that gets the binning wrong fails the golden
        /// table on the *areas*, not only on the forces.
        /// The 12 strips the tool measured — the table the port integrates.
        /// Parsed from the golden block below, so there is one copy of the
        /// numbers and no chance of a spec-less `WingAero()` integrating
        /// nothing at all.
        var stations: [(r: Double, chord: Double, area: Double)] = WingAero.defaultStations
        /// and the right wing's own, for the same reason its vectors are its
        /// own: the two wings in the body model differ by 6.6e-05 in area.
        var stationsRight: [(r: Double, chord: Double, area: Double)] = WingAero.defaultStationsRight

        // -- what the pools do ---------------------------------------------
        /// Each wing power pool's own rate at full drive, measured through the
        /// connectome by `--probe`. The two are *not* the same (173.7 vs 212.3
        /// Hz): a single shared reference would hold an amplitude asymmetry
        /// open and the animal would fly in a slow circle with no command. Each
        /// pool is calibrated against itself.
        /// The four pools' resting rates at the app's own tone, also measured
        /// through the connectome (`--probe`). Activation is a *difference*
        /// against these: "the pool is doing what it does when nothing is being
        /// asked of it" has to read as zero, the same way the haltere sensor's
        /// zero rate reads as its tonic bias.
        var rateTone = ["motor_wing_power_left": 158.66666666666666,
                        "motor_wing_power_right": 182.5,
                        "motor_wing_steering_left": 109.5,
                        "motor_wing_steering_right": 130.16666666666666]
        var rateReference = ["motor_wing_power_left": 173.66666666666666,
                             "motor_wing_power_right": 212.33333333333334,
                             "motor_wing_steering_left": 105.83333333333333,
                             "motor_wing_steering_right": 114.83333333333333]
        var powerLeft = "motor_wing_power_left"
        var powerRight = "motor_wing_power_right"
        var steeringLeft = "motor_wing_steering_left"
        var steeringRight = "motor_wing_steering_right"
    }

    /// One wing's force and moment, in the body frame (`+x` forward, `+y` left,
    /// `+z` up), for one instant of one wing's beat. `moment` is about the
    /// thorax origin, which is where the animal's centre of mass is — the hinge
    /// offset being in it is exactly what turns a left-right force asymmetry
    /// into yaw.
    struct Wrench {
        var force = SIMD3<Double>.zero
        var moment = SIMD3<Double>.zero
        /// The drag magnitude on this wing (the cost; the *net* drag over a
        /// symmetric beat is zero, which is why a fly that is not turning does
        /// not accelerate).
        var drag = 0.0
    }

    /// What one beat makes, both wings: the quantities item 6 is judged on.
    struct Beat {
        var amplitudeLeftDeg = 0.0
        var amplitudeRightDeg = 0.0
        /// Each wing's force along its own stroke plane's normal.
        var flightForceLeftDyn = 0.0
        var flightForceRightDyn = 0.0
        var flightForceDyn = 0.0
        /// The same force in the body's frame, cycle-mean.
        var thrustDyn = 0.0
        var verticalDyn = 0.0
        var sideDyn = 0.0
        var yawMomentDynCM = 0.0
        var rollMomentDynCM = 0.0
        var pitchMomentDynCM = 0.0
        var dragCostDyn = 0.0
        /// The largest force and yaw moment reached during the beat. A fly's
        /// thorax feels these, not the means: they are what the resonator and
        /// the steering muscles have to survive.
        var peakForceDyn = 0.0
        var peakYawMomentDynCM = 0.0
    }

    var p: Params

    init(spec: WingSpec? = nil) {
        var q = Params()
        if let s = spec {
            q.areaCM2 = s.area_cm2 ?? q.areaCM2
            q.spanCM = s.span_cm ?? q.spanCM
            q.chordCM = s.chord_cm ?? q.chordCM
            q.secondMomentCM4 = s.second_moment_cm4 ?? q.secondMomentCM4
            q.r2HatCM = s.r2_hat_cm ?? q.r2HatCM
            q.sinTheta = s.sin_theta ?? q.sinTheta
            q.massUG = s.mass_ug ?? q.massUG
            q.comCM = s.com_cm ?? q.comCM
            if let v = s.hinge_left_cm, v.count >= 3 { q.hingeLeft = SIMD3(v[0], v[1], v[2]) }
            if let v = s.hinge_right_cm, v.count >= 3 { q.hingeRight = SIMD3(v[0], v[1], v[2]) }
            if let v = s.span_axis_left, v.count >= 3 { q.spanLeft = SIMD3(v[0], v[1], v[2]) }
            if let v = s.span_axis_right, v.count >= 3 { q.spanRight = SIMD3(v[0], v[1], v[2]) }
            if let v = s.lift_dir_left, v.count >= 3 { q.liftLeft = SIMD3(v[0], v[1], v[2]) }
            if let v = s.lift_dir_right, v.count >= 3 { q.liftRight = SIMD3(v[0], v[1], v[2]) }
            if let v = s.stroke_axis_left, v.count >= 3 { q.strokeLeft = SIMD3(v[0], v[1], v[2]) }
            if let v = s.stroke_axis_right, v.count >= 3 { q.strokeRight = SIMD3(v[0], v[1], v[2]) }
            q.strokeHz = s.stroke_hz ?? q.strokeHz
            q.amplitudeFullDeg = s.amplitude_full_deg ?? q.amplitudeFullDeg
            q.alphaMidDeg = s.alpha_mid_deg ?? q.alphaMidDeg
            q.alphaRotDeg = s.alpha_rot_deg ?? q.alphaRotDeg
            if let rho = s.rho_kg_m3 { q.rhoGCM3 = rho * 1e-3 }
            q.bodyMassUG = s.body_mass_ug ?? q.bodyMassUG
            if let st = s.stations, !st.isEmpty {
                q.stations = st.map { ($0.r_cm, $0.chord_cm, $0.area_cm2 ?? $0.chord_cm * 0.022) }
            }
            if let st = s.stations_right, !st.isEmpty {
                q.stationsRight = st.map { ($0.r_cm, $0.chord_cm, $0.area_cm2 ?? $0.chord_cm * 0.022) }
            }
            if let ref = s.rate_reference_hz { q.rateReference = ref }
            if let tone = s.rate_tone_hz { q.rateTone = tone }
            q.powerLeft = s.power_left ?? q.powerLeft
            q.powerRight = s.power_right ?? q.powerRight
            q.steeringLeft = s.steering_left ?? q.steeringLeft
            q.steeringRight = s.steering_right ?? q.steeringRight
        }
        p = q
    }

    // -- the wing, as geometry ------------------------------------------------

    /// Which wing. Left is `+y` in the body frame, and the two are mirrors.
    enum Side: String, CaseIterable { case left, right }

    func hinge(_ side: Side) -> SIMD3<Double> {
        side == .left ? p.hingeLeft : p.hingeRight
    }

    func spanAxis(_ side: Side) -> SIMD3<Double> { side == .left ? p.spanLeft : p.spanRight }

    /// The strip's radius about the stroke *axis*, as a fraction of its radius
    /// about the hinge: the wing lies very nearly in the stroke plane, so this
    /// is 0.995 — but it is 0.995 of *this wing's* own geometry, and the two
    /// wings' meshes are mirrors only to ~2e-6, so it is recomputed per side
    /// from the two measured vectors rather than shared as one number. (Shared,
    /// it put a 3e-5 relative error into every right-wing force in the port
    /// while the tool, which measures per wing, had none — the golden table
    /// caught it.)
    func sinTheta(_ side: Side) -> Double {
        let a = spanAxis(side), n = strokeAxis(side)
        return simd_length(a - n * simd_dot(a, n))
    }
    func liftDir(_ side: Side) -> SIMD3<Double> { side == .left ? p.liftLeft : p.liftRight }
    func strokeAxis(_ side: Side) -> SIMD3<Double> { side == .left ? p.strokeLeft : p.strokeRight }

    static func rodrigues(_ v: SIMD3<Double>, _ axis: SIMD3<Double>, _ angle: Double) -> SIMD3<Double> {
        let a = simd_normalize(axis)
        let c = cos(angle), s = sin(angle)
        return v * c + simd_cross(a, v) * s + a * (simd_dot(a, v) * (1 - c))
    }

    /// Where one wing's strips are, which way they travel, and which way they
    /// push, at stroke angle `phi` (radians) and angle of attack `alphaDeg`.
    ///
    /// `v` is the direction the strips move *when the stroke angle is
    /// increasing*; the caller supplies the sign of the stroke rate separately,
    /// so that a wing's strips reverse without any of the geometry having to be
    /// rebuilt.
    func frame(_ side: Side, phi: Double, alphaDeg: Double)
        -> (r: SIMD3<Double>, v: SIMD3<Double>, lift: SIMD3<Double>) {
        let r = Self.rodrigues(spanAxis(side), strokeAxis(side), phi)
        let v = simd_normalize(simd_cross(strokeAxis(side), r))
        // the wing's normal, pitched away from the stroke plane's normal, with
        // its component along the velocity removed: the lift direction
        let n = Self.rodrigues(liftDir(side), r, alphaDeg * .pi / 180)
        let lift = simd_normalize(n - v * simd_dot(n, v))
        return (r, v, lift)
    }

    /// The wing's plane geometry: the strip list with its areas, which are
    /// `chord × dr` — the tool bins the measured mesh into 12 strips and the
    /// areas are what its ∫r²dA and its per-strip forces are built on.
    var stripRadiiCM: [Double] { p.stations.map { $0.r } }
    var stripAreasCM2: [Double] { p.stations.map { $0.area } }
    var stripChordsCM: [Double] { p.stations.map { $0.chord } }
    /// Each wing integrates its own strips: two wings whose areas differ by
    /// 6.6e-05 are not one wing used twice.
    func stripRadii(_ side: Side) -> [Double] {
        (side == .left ? p.stations : p.stationsRight).map { $0.r }
    }
    func stripAreas(_ side: Side) -> [Double] {
        (side == .left ? p.stations : p.stationsRight).map { $0.area }
    }

    /// The lift and drag coefficients, Dickinson/Lehmann/Sane 1999, from the
    /// animal's own tethered flight. `alphaDeg` is the angle of attack in
    /// degrees; the fits take radians, and the phase offsets (7.2°, 9.8°) are
    /// the measured ones, not fitted knobs.
    func coefficients(alphaDeg: Double) -> (cl: Double, cd: Double) {
        let a = alphaDeg * .pi / 180
        let cl = 0.225 + 1.58 * sin(2.13 * a - 7.2 * .pi / 180)
        let cd = 1.92 - 1.55 * cos(2.13 * a - 9.8 * .pi / 180)
        return (cl, cd)
    }

    /// The scalar force law, per wing, for the tool's golden table: the lift
    /// along the stroke plane's normal and the drag against the motion, in dyn,
    /// for a wing turning at `omega` rad/s with angle of attack `alphaDeg`.
    func stripForce(omegaRadS: Double, alphaDeg: Double) -> (lift: Double, drag: Double, cl: Double, cd: Double) {
        let (cl, cd) = coefficients(alphaDeg: alphaDeg)
        var lift = 0.0, drag = 0.0
        let areas = stripAreasCM2
        for (i, r) in stripRadiiCM.enumerated() {
            let v = abs(omegaRadS) * r * p.sinTheta               // cm/s
            let q = 0.5 * p.rhoGCM3 * v * v * areas[i]            // dyn
            lift += q * cl
            drag += q * cd
        }
        return (lift, drag, cl, cd)
    }

    /// One wing's wrench at one instant: the strips at radius `r` moving at
    /// `ω r sinθ` along `v̂`, `½ρv²A C_L` along the lift direction and
    /// `½ρv²A C_D` against the motion, each force's moment about the thorax
    /// origin.
    func wrench(_ side: Side, phi: Double, omegaRadS: Double, alphaDeg: Double) -> Wrench {
        let f = frame(side, phi: phi, alphaDeg: alphaDeg)
        let (cl, cd) = coefficients(alphaDeg: alphaDeg)
        let sgn = omegaRadS >= 0 ? 1.0 : -1.0
        let hinge = hinge(side)
        var out = Wrench()
        let areas = stripAreas(side)
        let sinT = sinTheta(side)
        for (i, r) in stripRadii(side).enumerated() {
            let v = abs(omegaRadS) * r * sinT
            let q = 0.5 * p.rhoGCM3 * v * v * areas[i]
            let force = f.lift * (q * cl) - f.v * (q * cd * sgn)
            out.force += force
            out.moment += simd_cross(hinge + f.r * r, force)
            out.drag += q * cd
        }
        return out
    }

    // -- the beat -------------------------------------------------------------

    /// The stroke amplitude (degrees) a wing's activation buys. Activation is
    /// 0...1 and comes from the motor pool's rate against its own reference:
    /// more spikes, bigger stroke. (APPROXIMATION #36: the thorax is a
    /// resonator and a muscle; this is the straight line through it.)
    func amplitudeDeg(activation: Double) -> Double {
        max(0, min(1, activation)) * p.amplitudeFullDeg
    }

    /// The angle of attack through the beat. It sweeps twice per stroke cycle,
    /// which is the wing's flip: at the stroke reversals the wing pitches
    /// through, which is what keeps the lift pointing the same way on both half
    /// strokes instead of cancelling.
    func alphaDeg(tMS: Double) -> Double {
        let w = 2 * Double.pi * p.strokeHz * tMS / 1000
        return p.alphaMidDeg - p.alphaRotDeg * cos(2 * w)
    }

    /// One beat, both wings, each driven by its own activation — which is the
    /// shape the app drives them in: two pools, two muscles, no made-up average.
    func beat(activationLeft: Double, activationRight: Double, samples: Int = 720) -> Beat {
        let ampL = amplitudeDeg(activation: activationLeft)
        let ampR = amplitudeDeg(activation: activationRight)
        let period = 1000.0 / p.strokeHz
        var out = Beat()
        out.amplitudeLeftDeg = ampL
        out.amplitudeRightDeg = ampR
        var total = SIMD3<Double>.zero
        var moment = SIMD3<Double>.zero
        for i in 0..<samples {
            let t = period * Double(i) / Double(samples)
            let w = 2 * Double.pi * p.strokeHz * t / 1000
            let alpha = alphaDeg(tMS: t)
            let phiL = ampL * .pi / 180 * sin(w)
            let phiR = ampR * .pi / 180 * sin(w)
            let omL = 2 * Double.pi * p.strokeHz * ampL * .pi / 180 * cos(w)
            let omR = 2 * Double.pi * p.strokeHz * ampR * .pi / 180 * cos(w)
            let wl = wrench(.left, phi: phiL, omegaRadS: omL, alphaDeg: alpha)
            let wr = wrench(.right, phi: phiR, omegaRadS: omR, alphaDeg: alpha)
            let pair = wl.force + wr.force
            out.flightForceLeftDyn += simd_dot(wl.force, liftDir(.left)) / Double(samples)
            out.flightForceRightDyn += simd_dot(wr.force, liftDir(.right)) / Double(samples)
            total += pair / Double(samples)
            moment += (wl.moment + wr.moment) / Double(samples)
            out.dragCostDyn += (wl.drag + wr.drag) / Double(samples)
            out.peakForceDyn = max(out.peakForceDyn, simd_length(pair))
        }
        out.flightForceDyn = out.flightForceLeftDyn + out.flightForceRightDyn
        out.thrustDyn = total.x
        out.verticalDyn = total.z
        out.sideDyn = total.y
        out.yawMomentDynCM = moment.z
        out.rollMomentDynCM = moment.x
        out.pitchMomentDynCM = moment.y
        return out
    }

    /// The wrench at one instant of the beat, for the physics: cheap enough to
    /// call every substep (24 strips), and *not* the cycle mean — the pulsatile
    /// force is the thing the animal's thorax and its muscles are built around,
    /// and smoothing it away would make the model quieter than the animal.
    ///
    /// `phase` runs 0...1 through the stroke, and is the app's own clock (the
    /// stroke frequency, integrated in the solver), not a lookup into a table.
    func instantaneous(phase: Double, activationLeft: Double, activationRight: Double) -> Wrench {
        let w = 2 * Double.pi * phase
        let alpha = alphaDeg(tMS: phase * 1000 / p.strokeHz)
        let ampL = amplitudeDeg(activation: activationLeft)
        let ampR = amplitudeDeg(activation: activationRight)
        let wl = wrench(.left, phi: ampL * .pi / 180 * sin(w),
                        omegaRadS: 2 * Double.pi * p.strokeHz * ampL * .pi / 180 * cos(w),
                        alphaDeg: alpha)
        let wr = wrench(.right, phi: ampR * .pi / 180 * sin(w),
                        omegaRadS: 2 * Double.pi * p.strokeHz * ampR * .pi / 180 * cos(w),
                        alphaDeg: alpha)
        var out = wl
        out.force += wr.force
        out.moment += wr.moment
        out.drag += wr.drag
        return out
    }

    /// The force the wing's own mass needs at mid-stroke, `m ω² r` — for a fly
    /// this is comparable with the aerodynamic force, which is why the flight
    /// muscles are as big as they are. Reported with the model's own wing mass,
    /// which is an ENGINEERING PLACEHOLDER: a real wing is ~1 µg, eight times
    /// lighter (#35), so this number as it stands is ~8× the animal's.
    func inertialForceDyn() -> Double {
        let w = 2 * Double.pi * p.strokeHz * (p.amplitudeFullDeg * .pi / 180)
        return p.massUG * 1e-6 * w * w * p.comCM
    }

    /// The animal's weight, dyn.
    ///
    /// `g` is carried as the same rounded standard value the tool uses
    /// (981 cm/s², which is 980.665 rounded to the cm/s² the tool works in).
    /// It was 980 here and 981 there, and the golden's weight row caught the
    /// 0.1 %: two implementations of one model cannot each keep their own
    /// copy of a constant.
    var weightDyn: Double { p.bodyMassUG * 1e-6 * WingAero.gravityCM_S2 }
    static let gravityCM_S2 = 981.0

    /// How far the wings' force is from vertical while the body is held level.
    /// The stroke plane belongs to the thorax, so this is a *posture*: the
    /// animal has to hold this much nose-up pitch for its wingbeat to be weight
    /// support, and a hovering fly does hold one (BIOLOGICAL DATA #34).
    var attitudeDeg: Double {
        acos(max(-1, min(1, abs(p.liftLeft.z)))) * 180 / .pi
    }

    // -- the pools ------------------------------------------------------------

    /// The four drives the cord sends to the wing motor pools, in the app's
    /// tone units: the power pools get the flight tone, the steering pools get
    /// it differentially. `steer` is the cord's steering command, whatever
    /// produced it — the halteres, the eyes, or a test.
    func poolDrives(tone: Double, steer: Double) -> [String: Double] {
        [p.powerLeft: tone,
         p.powerRight: tone,
         p.steeringLeft: tone + steer,
         p.steeringRight: max(0, tone - steer)]
    }

    /// The activation of each wing, from the four pools' *measured* rates.
    ///
    /// Each pool is divided by its own full-drive rate, and each is differenced
    /// against its own resting rate — because the two power pools do not run at
    /// the same rate for the same drive (173.7 and 212.3 Hz at full, 158.7 and
    /// 182.5 at the app's tone). Dividing them by one shared number, or against
    /// one shared resting rate, would hold an amplitude asymmetry open and the
    /// animal would fly in a slow circle with no command to. Both tables are
    /// measured through the connectome and written into `world.json` by
    /// `tools/wing_aero.py --probe --write`; the resting one is what the app
    /// itself reads at start-up if it can, and what it falls back to if it
    /// cannot.
    ///
    /// The steering pool's rate buys stroke *amplitude* on top of the power
    /// pool's, at one full steering rate to one full activation unit
    /// (APPROXIMATION #36: the thorax's rate-to-amplitude map, and the muscle
    /// that implements it, are not simulated here — the muscles are, in
    /// `FlyLiveBody`; the *thorax resonating* is not).
    func activation(ratesHz: [String: Double]) -> (left: Double, right: Double) {
        func delta(_ name: String) -> Double {
            guard let rate = ratesHz[name], let ref = p.rateReference[name], ref > 0 else { return 0 }
            return (rate - (p.rateTone[name] ?? 0)) / ref
        }
        func restActivation(_ name: String) -> Double {
            guard let ref = p.rateReference[name], ref > 0 else { return 0 }
            return (p.rateTone[name] ?? 0) / ref
        }
        let left = restActivation(p.powerLeft) + delta(p.powerLeft) + delta(p.steeringLeft)
        let right = restActivation(p.powerRight) + delta(p.powerRight) + delta(p.steeringRight)
        return (max(0, min(1, left)), max(0, min(1, right)))
    
    }

    // MARK: - the golden table
    //
    // Written by `python3 tools/wing_aero.py --golden build/wing_golden.json`
    // and pasted here by `tools/wing_golden_block.py`, so that these numbers are
    // the tool's own and not typed by hand. `WingAeroTests` parses them and
    // checks this port against them; CI compares the same rows against a fresh
    // run of the tool, so a change to either side fails the build.

    /// `area span length chord ∫r²dA r̂₂ mass com sinθ attitude weight`, then the
    /// stroke axis, the span axis, the lift direction and the hinge of the left
    /// wing and then of the right one — each on its own line. The wing's
    /// measured geometry, in the body frame; the right wing's rows are measured,
    /// not mirrored.
    static let goldenWing = """
1 0.0174076085266342 0.264662282664542 0.262704546065884 0.0662630654372785 0.000416441015223553 0.154670394258787 8 0.15307155189649 0.995144899751452 47.2981245714733 0.966285
0.737760020167381 0 0.675063073084749
-0.131805656948767 0.991274036157107 -0.00174758035618449
0.728323703550576 0.0980378394155563 0.678183725025233
-0.00694 0.0432 0.0091
-0.737760020167381 0 -0.675063073084749
-0.131806039773842 -0.99127398042459 -0.00175031777837222
0.728323537728309 -0.0980399767400617 0.678183594133424
-0.00694 -0.0432 0.0091
"""

    /// The two wings' four vectors each, parsed out of the literal above, so
    /// `Params` can default to the tool's own numbers instead of to copies of
    /// them: order is stroke, span, lift, hinge, left wing then right.
    static var goldenWingVectors: [SIMD3<Double>] {
        goldenWing.split(separator: "\n").compactMap { line in
            let v = line.split(separator: " ").compactMap { Double($0) }
            return v.count == 3 ? SIMD3<Double>(v[0], v[1], v[2]) : nil
        }
    }

    /// The strips: `r chord area`, along the span.
    static let goldenStations = """
0.0129037593514033 0.0131182965898211 0.000287186345898882
0.0347958048568936 0.0209962270945286 0.000459650358997031
0.056687850362384 0.0731316688925422 0.00160100182328799
0.0785798958678743 0.100910547556596 0.00220913829909294
0.100471941373365 0.0764987488391895 0.00167471409070061
0.122363986878855 0.066609549049175 0.00145821927888473
0.144256032384345 0.074859426679764 0.00163882597538831
0.166148077889836 0.0691732851285302 0.00151434470579804
0.188040123395326 0.0974091911801054 0.00213248644596788
0.209932168900816 0.151859721287126 0.00332451992886886
0.231824214406307 0.0480648553681592 0.00105223800093455
0.253716259911797 0.00252526758180444 5.52832728144024e-05
"""


    /// The right wing's strips. The two wings in the body model are *not*
    /// identical — their areas differ by 6.6e-05 — so each wing is integrated
    /// with its own table; sharing one put a 3e-05 error into every right-wing
    /// force in the port, which the golden table caught.
    static let goldenStationsRight = """
0.0129039009807094 0.0131184541060768 0.000287188790160075
0.0347958699457906 0.0209965489385332 0.000459655797736179
0.0566878389108718 0.0731331819042935 0.00160102934856643
0.078579807875953 0.100912652393215 0.00220917665437629
0.100471776841034 0.0765004293339416 0.00167474502479404
0.122363745806115 0.0666109614266638 0.00145824510028675
0.144255714771197 0.0747943291743286 0.00163739513304847
0.166147683736278 0.0691748780943414 0.0015143742844046
0.188039652701359 0.0974113817594276 0.00213252694632307
0.20993162166644 0.151863174031832 0.00332458389284359
0.231823590631522 0.0480659636383051 0.0010522585842465
0.253715559596603 0.00252532919541026 5.52844283725348e-05
"""

    /// Both tables as the port integrates them.
    static var defaultStations: [(r: Double, chord: Double, area: Double)] {
        parseStations(goldenStations)
    }
    static var defaultStationsRight: [(r: Double, chord: Double, area: Double)] {
        parseStations(goldenStationsRight)
    }
    private static func parseStations(_ text: String) -> [(r: Double, chord: Double, area: Double)] {
        text.split(separator: "\n").compactMap { line in
            let v = line.split(separator: " ").compactMap { Double($0) }
            return v.count >= 3 ? (v[0], v[1], v[2]) : nil
        }
    }

    /// The force law at a strip: `ω α CL CD lift drag`.
    static let goldenStrip = """
0 0 0.0269734909683993 0.39261775735059 0 0
0 22.5 1.25583804160007 0.700668142728643 0 0
0 45 1.80456143974444 1.81322684192653 0 0
0 67.5 1.31109906796447 2.99623387137238 0 0
0 90 0.101034628750006 3.46914783470972 0 0
50 0 0.0269734909683993 0.39261775735059 1.7018334240926e-05 0.000247713587808974
50 22.5 1.25583804160007 0.700668142728643 0.000792343548317811 0.00044207124168298
50 45 1.80456143974444 1.81322684192653 0.00113854857629799 0.00114401582230035
50 67.5 1.31109906796447 2.99623387137238 0.000827209284394308 0.0018904082362471
50 90 0.101034628750006 3.46914783470972 6.37455894748619e-05 0.00218878295921881
300 0 0.0269734909683993 0.39261775735059 0.000612660032673337 0.00891768916112306
300 22.5 1.25583804160007 0.700668142728643 0.0285243677394412 0.0159145647005873
300 45 1.80456143974444 1.81322684192653 0.0409877487467275 0.0411845696028128
300 67.5 1.31109906796447 2.99623387137238 0.0297795342381951 0.0680546965048957
300 90 0.101034628750006 3.46914783470972 0.00229484122109503 0.0787961865318772
1697 0 0.0269734909683993 0.39261775735059 0.0196038208448108 0.285347127837829
1697 22.5 1.25583804160007 0.700668142728643 0.912719232615026 0.509232296175928
1697 45 1.80456143974444 1.81322684192653 1.31152097478405 1.31781882448118
1697 67.5 1.31109906796447 2.99623387137238 0.952881896832916 2.17760586096741
1697 90 0.101034628750006 3.46914783470972 0.073430048912005 2.52131074600199
-1697 0 0.0269734909683993 0.39261775735059 0.0196038208448108 0.285347127837829
-1697 22.5 1.25583804160007 0.700668142728643 0.912719232615026 0.509232296175928
-1697 45 1.80456143974444 1.81322684192653 1.31152097478405 1.31781882448118
-1697 67.5 1.31109906796447 2.99623387137238 0.952881896832916 2.17760586096741
-1697 90 0.101034628750006 3.46914783470972 0.073430048912005 2.52131074600199
"""

    /// Whole beats, driven by each wing's own activation:
    /// `aL aR ampL ampR liftL liftR flight thrust vertical side yaw roll pitch drag`.
    static let goldenBeat = """
1 1 77.5 77.5 0.484627651115822 0.48459887638067 0.969226527496491 0.711710624831921 0.657390533067092 0.000341167601965384 -0.000124089177466334 -0.000116917645753633 0.0228477813993537 0.808569308216712
1 0.75 77.5 58.125 0.484627651115822 0.272873552878355 0.757501203994177 0.556707529297363 0.512688938525199 0.00746757831440666 -0.0245372342370708 0.0222354558971344 0.0182723473911443 0.63170001320753
1 0.5 77.5 38.75 0.484627651115822 0.121372017953141 0.605999669068963 0.445778198245877 0.40963655532036 0.0158382093301935 -0.0456224382377134 0.0414664173028076 0.0145944362008433 0.505364802486685
1 0 77.5 0 0.484627651115822 0 0.484627651115822 0.357163987323938 0.327286357801506 0.0258713556369007 -0.0648574510939587 0.0589664900323897 0.0113772484229832 0.404296633910009
0.75 0.75 58.125 58.125 0.272889742233981 0.272873552878355 0.545763295112335 0.399702036320654 0.370157065037444 7.97180063182495e-05 -2.94778915115858e-05 -2.65649423192607e-05 0.0137775999871565 0.454820235871901
0.5 1 38.75 77.5 0.121379214574506 0.48459887638067 0.605978090955176 0.443229069183769 0.412390467141609 -0.0154884168229148 0.0454945944956953 -0.0415854901185512 0.0146873429001963 0.505346832784206
0 1 0 77.5 0 0.48459887638067 0.48459887638067 0.354546637507982 0.330104175265587 -0.0255301880349353 0.0647333619164923 -0.0590834076781434 0.0114705329763706 0.404272674306704
0.4 0.4 31 31 0.0777004740314809 0.077695867919488 0.155396341950969 0.113388909764112 0.105390047343714 2.52118825452351e-06 -1.37937919492272e-06 -2.74408467073127e-07 0.00417661708623149 0.129371089314674
0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""

    /// The law the beats were made with, and the two pools' measured rates.
    static let goldenLaw = """
200 77.5 45 30 0.001225 985
"""

}
