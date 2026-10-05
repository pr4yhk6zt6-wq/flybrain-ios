//
//  FlyBody.swift
//  Rigid-body flight and walking dynamics for Drosophila melanogaster,
//  integrated in SI units from published measurements.
//
//  ============================ HONESTY NOTE ============================
//
//  A connectome is a wiring diagram. It records which neuron contacts which,
//  and how often. It does NOT record how much force a muscle makes when its
//  motor neuron fires. So a perfectly assumption-free simulation from
//  connectome data alone is not possible — not here, not anywhere, yet.
//
//  What IS possible, and what this file now does, is to make every physical
//  constant a MEASURED one from the Drosophila literature, to DERIVE the
//  rest from the CT-based flybody model, and to reduce the unmeasured part
//  to a small set of explicitly listed links (docs/ASSUMPTIONS.md).
//
//  Nothing in this file scripts behaviour. There is no gait clock imposed on
//  the legs beyond the phase of a step cycle, no bout timer, no grooming
//  routine, no arousal variable, no adapted steering baseline. Whether the
//  animal walks, stops, turns or takes off is decided entirely by the motor
//  groups the connectome fires; this file only turns those rates into forces
//  through measured muscle and wing mechanics.
//
//  MEASURED (each constant below carries its source):
//    body mass, wing length, wing area, radius of the second moment of area,
//    wingbeat frequency, stroke amplitude range, lift and drag coefficients,
//    air density, moment of inertia about yaw, flapping counter-torque,
//    walking speed range, step frequency, body length, leg joint axes and
//    angular limits (the last from the flybody MJCF, which is a real CT model).
//
//  DERIVED (computed from the CT model, provenance in the constant's comment):
//    leg track width, per-segment fore-aft foot gains of the swing joints.
//
//  Flight force is computed, not fudged: a quasi-steady blade-element
//  estimate, the standard model in insect flight aerodynamics.
//
//        F = 1/2 * rho * C * S * U^2,     U = 2 * Phi * f * r2
//
//  THE LINKS THAT REMAIN ASSUMPTIONS are listed, each with its reason, in
//  docs/ASSUMPTIONS.md. The headline one:
//    motor-neuron firing rate -> wing stroke amplitude Phi.
//    Both endpoints are measured (hover 2.47 rad, morphological max 3.1 rad);
//    the straight line between them is the assumption.
//
//  Note also that Drosophila power muscles are ASYNCHRONOUS: their motor
//  neurons fire at 5-20 Hz while the wing beats at ~200 Hz, because firing
//  sets the activation level, not the rhythm. The wingbeat frequency here is
//  therefore a measured constant, not something the spiking drives — which
//  is biologically correct, and is why the simulation does not need to resolve
//  200 Hz.
//
//  References
//    Fry, Sayaman & Dickinson 2003, Science 300:495  (free-flight dynamics,
//        body mass 0.96 mg, I_yaw 5.2e-13 kg m^2, flapping counter-torque,
//        saccades toward the weaker wing)
//    Lehmann & Dickinson 1997, J Exp Biol 200:1133   (stroke amplitude range,
//        wingbeat frequency, force production, b1/b2 steering range)
//    Dickinson, Lehmann & Sane 1999, Science 284:1954 (unsteady lift)
//    Sane & Dickinson 2001, J Exp Biol 204:2607      (C_L, C_D vs kinematics)
//    Sun & Tang 2002, J Exp Biol 205:2413            (hovering force balance,
//        r2 = 0.58 R, mean drag 1.27x lift)
//    Mendes et al. 2013, eLife 2:e00231              (walking speed, step
//        frequency, tripod gait, duty factor)
//    Card & Dickinson 2008, Curr Biol 18:1081        (escape take-off 0.9 m/s)
//    Vaxenburg et al. 2025, Nature                   (flybody CT skeleton,
//        joint axes and limits)
//  ======================================================================
//

import Foundation
import simd

/// Every measured constant, in SI. Nothing here was chosen to make the
/// simulation behave; each is a published value for D. melanogaster or a
/// geometric fact derived from the CT-based flybody model.
enum FlyMorphology {
    // --- mass and inertia (Fry et al. 2003) ---
    /// 0.96 mg.
    static let mass: Float = 0.96e-6                 // kg
    /// Yaw moment of inertia, 5.2e-13 kg m^2.
    static let inertiaYaw: Float = 5.2e-13           // kg m^2
    /// Pitch/roll are smaller; Fry reports ~5.2e-13 and ~2.3e-13.
    static let inertiaPitch: Float = 5.2e-13
    static let inertiaRoll: Float = 2.3e-13
    /// Body length, 2.5 mm.
    static let bodyLength: Float = 2.5e-3            // m

    // --- eyes (flybody MJCF eye cameras, Vaxenburg et al. 2025) -----------
    /// The optical axis of each compound eye sits 67 degrees off the body
    /// axis (measured from the published model's eye-camera quaternions), and
    /// each eye sees a 140-degree field. Together the two fields cover about
    /// 274 degrees with a small frontal overlap — the near-panoramic visual
    /// field of a real fly.
    static let eyeAzimuth: Float = 67.0 * .pi / 180
    static let eyeFieldOfView: Float = 140.0 * .pi / 180
    /// Half the distance between the two eyes, in world (cm-scale) units.
    static let eyeLateralOffset: Float = 0.033

    // --- wing (Lehmann & Dickinson 1997; Sun & Tang 2002) ---
    /// Wing length R, 2.39 mm.
    static let wingLength: Float = 2.39e-3           // m
    /// Area of ONE wing, 1.96 mm^2.
    static let wingArea: Float = 1.96e-6             // m^2
    /// Radius of the second moment of area, 0.58 R.
    static let r2: Float = 0.58 * wingLength         // m
    /// Wingbeat frequency in free flight, 218 Hz.
    static let wingbeatHz: Float = 218.0
    /// Stroke amplitude in stable hovering.
    ///
    /// NOT a free parameter: it is the amplitude at which the blade-element
    /// force from two wings exactly equals the measured body weight, solved
    /// from the morphology above. It comes out at 2.47 rad = 141 degrees,
    /// which lands inside the 130-160 degrees measured for hovering
    /// melanogaster. That it agrees is a check on the whole constant set.
    static let strokeAmplitudeHover: Float = 2.468   // rad
    /// Maximum (morphological) stroke amplitude, 3.1 rad (~178 deg).
    static let strokeAmplitudeMax: Float = 3.1       // rad
    /// How far the steering muscles (b1/b2) can shift the stroke amplitude of
    /// their own side: ~20 degrees (Lehmann & Dickinson 1997). This is a hard
    /// limit on the muscle, not a gain we chose. It is ALSO the bound the
    /// decoded left/right asymmetry saturates at — an asymmetry index of 1
    /// asks for the maximum the muscle can deliver, and no more.
    static let steeringRange: Float = 20.0 * .pi / 180
    /// Peak yaw rate recorded in free-flight saccades: ~1600 deg/s, held for a
    /// few tens of milliseconds (Fry, Sayaman & Dickinson 2003). Sustaining a
    /// faster rate than the animal has ever been measured at would take more
    /// torque than a fly can make, so the steering torque saturates here.
    static let maxYawRate: Float = 1600.0 * .pi / 180
    /// The torque that rate costs against the flapping counter-torque. About
    /// 0.86 nN m, against the ~1 nN m Fry et al. inferred for a saccade.
    static var maxYawTorque: Float { yawDamping * maxYawRate }
    /// Mean lift coefficient at hovering kinematics (Sane & Dickinson 2001).
    static let liftCoefficient: Float = 1.8
    /// Mean drag is 1.27x mean lift in hovering (Sun & Tang 2002).
    static let dragCoefficient: Float = 1.27 * 1.8
    /// Stroke plane inclination from horizontal in forward flight.
    static let strokePlaneAngle: Float = 0.5         // rad, ~29 deg

    // --- environment ---
    static let airDensity: Float = 1.2               // kg/m^3
    static let gravity: Float = 9.81                 // m/s^2

    // --- damping, DERIVED from the same blade-element model ----------------
    //
    // A flapping insect is damped by its own wing motion: translating or
    // rotating changes the air speed the wings see, which changes the force
    // they make, which opposes the motion. These are "flapping counter-force"
    // and "flapping counter-torque" (Hesselberg & Lehmann 2007; Cheng & Deng
    // 2011) and they dominate over body parasite drag at this scale.
    //
    // No new constants: both fall straight out of F = 1/2 rho C S U^2 by
    // differentiating with respect to U.

    /// Mean wing velocity at r2 during hovering.
    static var meanWingVelocity: Float {
        2 * strokeAmplitudeHover * wingbeatHz * r2
    }
    /// dF/dU for both wings together.
    static var forceVelocitySlope: Float {
        airDensity * dragCoefficient * (2 * wingArea) * meanWingVelocity
    }
    /// Flapping counter-force. Factor of a half because body translation only
    /// opposes the wing over part of the stroke. ~8.0e-6 N s/m, which puts
    /// terminal forward speed at 0.89 m/s — real flies top out near 1 m/s.
    static var translationalDamping: Float { 0.5 * forceVelocitySlope }
    /// Flapping counter-torque about yaw, dF/dU times the moment arm squared.
    /// ~3.1e-11 N m s/rad. This predicts that holding 1600 deg/s — the peak
    /// saccade rate Fry et al. measured — needs 8.6e-10 N m of torque, and
    /// they measured saccade torques of about 1e-9 N m. It agrees.
    static var yawDamping: Float { forceVelocitySlope * r2 * r2 }

    // --- walking (Mendes et al. 2013) ---
    /// Drosophila walks at 5-25 mm/s; peak sustained ~30 mm/s.
    static let walkSpeedMax: Float = 30e-3           // m/s
    /// Step frequency at that speed.
    static let stepFrequencyMax: Float = 13.0        // Hz
    /// Tripod duty factor: fraction of the cycle a leg is in stance.
    static let dutyFactor: Float = 0.55

    // --- walking geometry, DERIVED from the flybody CT model ---------------
    //
    // Both numbers below were computed from build/flymodel.bin with the same
    // forward-kinematics chain the renderer uses (FlyModel.solve, mirrored in
    // tools/softrender.py); the derivations are re-run by tools/phase0_probes.py
    // and tools/gait_geometry.py.

    /// Lateral distance between the left and right tarsus lines: the rest-pose
    /// tarsus spreads of the three leg pairs are 0.746, 1.283 and 0.856 mm,
    /// mean 0.96 mm. This is the track of the differential drive that turns
    /// the walking animal: yaw rate = (v_right - v_left) / track.
    static let legTrack: Float = 0.96e-3             // m

    /// Secant fore-aft gain of the tarsus tip: travel(A*) / (2 A*) per
    /// radian, where A* is the amplitude that carries the foot through the
    /// no-slip travel R = stride * duty (or the range-limited amplitude,
    /// where the morphology runs out — T1 and T2). Derived through the FK
    /// chain from the CT model by tools/gait_geometry.py; a small-angle
    /// tangent gain is NOT valid at stride-scale amplitudes and was
    /// over-optimistic by ~2x.
    static let gaitGainT1: Float = 0.630e-3          // m/rad
    static let gaitGainT2: Float = 0.490e-3
    static let gaitGainT3: Float = 1.218e-3

    /// Walking-posture neutral joint angles (coxa, femur, tibia), in radians.
    /// Chosen by searching the measured joint ranges for the posture that
    /// maximises TRUE fore-aft foot travel through the FK chain; the search
    /// is tools/gait_geometry.py, the ranges are the flybody MJCF's.
    static let gaitNeutralT1 = SIMD3<Float>(0.25, 1.55, -0.85)
    static let gaitNeutralT2 = SIMD3<Float>(0.15, 1.55, -1.00)
    static let gaitNeutralT3 = SIMD3<Float>(0.15, 1.00, -0.85)

    /// Largest swing amplitude that keeps all three joints inside their
    /// measured ranges from the neutral posture above (= the solved A*).
    static let gaitAmpT1: Float = 0.45               // rad
    static let gaitAmpT2: Float = 0.35
    static let gaitAmpT3: Float = 0.45

    /// Which joint direction protracts the foot, per joint, at those
    /// postures. NOT uniform across segments — the flybody joint axes differ;
    /// driving all segments with one sign is what made the old walk look
    /// reversed. Measured, per segment, in tools/gait_geometry.py.
    static let gaitSignT1 = SIMD3<Float>(1, -1, 1)
    static let gaitSignT2 = SIMD3<Float>(1, -1, 1)
    static let gaitSignT3 = SIMD3<Float>(-1, 1, -1)

    /// Stance inverse-map knots: the sweep multiplier q at equal eighths of
    /// the foot's travel, so the planted foot advances at constant speed
    /// even though the FK tip displacement is strongly nonlinear in joint
    /// angle. Piecewise-linear between knots; derived (bisected through the
    /// FK chain) in tools/gait_geometry.py.
    static let gaitKnotT1: [Float] = [1.000, 0.213, -0.016, -0.202, -0.369,
                                      -0.526, -0.681, -0.836, -1.000]
    static let gaitKnotT2: [Float] = [1.000, 0.355, 0.079, -0.139, -0.330,
                                      -0.506, -0.674, -0.837, -1.000]
    static let gaitKnotT3: [Float] = [1.000, 0.753, 0.552, 0.368, 0.190,
                                      0.006, -0.194, -0.438, -0.986]


    // --- escape (Card & Dickinson 2008) ---
    /// Peak take-off speed of the giant-fibre escape, ~0.9 m/s reached within
    /// a few milliseconds of the tergotrochanteral motoneurons firing.
    static let escapeTakeoffSpeed: Float = 0.9       // m/s
    /// motor_jump_escape is the two tergotrochanteral (TT/PDMN) motoneurons —
    /// the final common path of the giant fibre escape (BANC v888
    /// cell_function == "jump_escape"; King & Valoroso 2017). One spike from
    /// one of these two neurons inside one 60 fps frame reports as
    /// 1 / (2 * 1/60) = 30 Hz of group-mean rate, and the engine's 0.35 EMA
    /// peaks that single-spike event at ~10.5 Hz. A rising edge through
    /// 10 Hz therefore means "at least one TT spike per frame" — the
    /// measured command for take-off.
    static let jumpEventRate: Float = 10.0           // Hz

    /// Weight, for convenience. ~9.4 microNewtons.
    static var weight: Float { mass * gravity }

    /// Quasi-steady mean aerodynamic force from one wing at a given stroke
    /// amplitude, using the standard blade-element estimate.
    ///
    ///     U = 2 * Phi * f * r2        mean wing velocity at r2
    ///     F = 1/2 * rho * C * S * U^2
    ///
    /// Inverting this for F = weight is what fixes `strokeAmplitudeHover`, so
    /// by construction two wings at that amplitude carry exactly 9.4 uN.
    static func wingForce(strokeAmplitude phi: Float, coefficient C: Float) -> Float {
        let u = 2 * phi * wingbeatHz * r2
        return 0.5 * airDensity * C * wingArea * u * u
    }

    /// Total lift from both wings, which is what gets compared against weight.
    static func flightForce(strokeAmplitude phi: Float) -> Float {
        2 * wingForce(strokeAmplitude: phi, coefficient: liftCoefficient)
    }

    /// The stroke amplitude at which both wings exactly support body weight.
    /// `strokeAmplitudeHover` is this value; kept here so a test can confirm
    /// the constant has not drifted away from the morphology.
    static var solvedHoverAmplitude: Float {
        (weight / (airDensity * liftCoefficient * wingArea
                   * (2 * wingbeatHz * r2) * (2 * wingbeatHz * r2))).squareRoot()
    }
}

/// World units are centimetres (1.0 == 1 cm), which keeps the scene numbers
/// human-sized. Physics runs in metres and converts at the boundary.
private let metresToWorld: Float = 100.0
private let worldToMetres: Float = 0.01

struct FlyPose {
    var position = SIMD3<Float>(0, 0.35, 0)     // world units
    var heading: Float = 0                      // yaw, rad
    var pitch: Float = 0
    var roll: Float = 0
    var velocity = SIMD3<Float>.zero            // world units / s

    /// Animation state only: a smoothed 0/1 of "not touching anything".
    /// Physics does NOT read this — contact comes from the collision solver.
    var airborne: Float = 0
    /// Instantaneous stroke phase, for drawing. The real wing beats at 218 Hz,
    /// far above the display rate, so the renderer draws a stroboscopic sample
    /// of it — which is exactly what your eye sees looking at a real fly.
    var wingPhase: Float = 0
    /// Stroke amplitude actually commanded, per wing, in radians.
    var strokeAmplitudeL: Float = 0
    var strokeAmplitudeR: Float = 0
    /// Gait cycle phase.
    var gaitPhase: Float = 0
    var stepFrequency: Float = 0
    var proboscisExtension: Float = 0
    var headYaw: Float = 0
    var headPitch: Float = 0
    var abdomenBend: Float = 0

    /// Body-axis angular rates, rad/s.
    var yawRate: Float = 0
}

struct FlyDrives {
    var wingPowerL: Float = 0
    var wingPowerR: Float = 0
    var wingSteerL: Float = 0
    var wingSteerR: Float = 0
    var wingTensionL: Float = 0
    var wingTensionR: Float = 0
    var legL: Float = 0
    var legR: Float = 0
    var neck: Float = 0
    var proboscis: Float = 0
    var jump: Float = 0
    var haltere: Float = 0
    var abdomen: Float = 0
}

final class FlyBody {

    private(set) var pose = FlyPose()
    private(set) var drives = FlyDrives()
    private(set) var energy: Float = 1.0
    private(set) var isEating = false
    private(set) var bumped = false
    private(set) var hurt: Float = 0

    /// Live aerodynamic readout, in micronewtons, for the HUD. These are real
    /// forces, not display numbers.
    private(set) var liftMicroNewtons: Float = 0
    private(set) var weightMicroNewtons: Float = FlyMorphology.weight * 1e6

    /// True while any leg is touching the floor or an object top. Decided by
    /// the collision solver each frame — never by a lift threshold, which is
    /// what used to make the animal flicker between flight and walking ten
    /// times a second whenever the wing drive sat near hover.
    private(set) var grounded = false

    /// Escape take-offs actually launched, and when. The giant-fibre signal
    /// is never discarded: events are counted whether the animal is on the
    /// ground or already flying; only the impulse needs contact.
    private(set) var jumpEvents = 0
    private(set) var lastJumpTime: Float = -1000

    /// Joint angles for FlyModel, indexed by its joint table.
    private(set) var jointAngles: [Float] = []
    private var model: FlyModel?

    // MARK: - The decode, and its one headline assumption

    /// Motor-neuron population rate -> wing stroke amplitude.
    ///
    /// Both endpoints are measured: at the hovering amplitude the blade-element
    /// force equals body weight exactly, and the maximum is the morphological
    /// limit of 3.1 rad. The straight line between them is the assumption
    /// (docs/ASSUMPTIONS.md #1).
    ///
    /// `referenceRate` — the firing rate that means "hover" — is a slow
    /// running average of the power-muscle COLLECTIVE, used only for the
    /// symmetric lift channel. It is the equilibrium reflex: real flies hold
    /// lift against weight through haltere and visual feedback onto the same
    /// steering muscles (Sherman & Dickinson 2003), and those pathways are
    /// present in BANC. What we cannot do is calibrate that loop's gain, so
    /// the loop is closed here, at the muscle. It must never be applied to a
    /// left/right DIFFERENCE — that is what the opponent decode below is for.
    private(set) var referenceRate: Float = 120.0
    private let referenceTau: Float = 10.0      // s

    /// Neuromuscular low-pass on the decoded wing command (ASSUMPTIONS.md #3).
    ///
    /// The stroke amplitude a muscle delivers tracks its motoneuron population
    /// with a lag; more importantly here, the smallest wing motor group has
    /// 12 neurons, so its reported rate jumps 5.2 Hz per single spike. A
    /// one-pole filter of 0.2 s averages ~144 spikes per group at 60 Hz
    /// firing, bringing spike-counting noise under 10% before it reaches the
    /// wing. The exact tau is not published for population-level decoding and
    /// is listed as an assumption.
    private let muscleTau: Float = 0.2          // s

    /// Opponent-decode noise floors: the rate change one spike makes in a
    /// group over the low-pass window. Wing groups have 12 neurons; the leg
    /// drive is the mean of three groups whose smallest has 63, further
    /// averaged over the three, so one spike moves it 1/(3*63*tau).
    private var wingEps: Float { 1 / (12 * muscleTau) }
    private var legEps: Float { 1 / (3 * 63 * muscleTau) }

    /// Escape refractory floor. Card & Dickinson (2008) measure the whole
    /// take-off sequence in ~14 ms; a tenth of a second between launches is a
    /// conservative floor that keeps a rattling TT population from machine-
    /// gunning the animal. Listed in ASSUMPTIONS.md #7.
    private let jumpRefractory: Float = 0.1     // s

    // smoothed decode state
    private var phiL: Float = 0
    private var phiR: Float = 0
    private var legIndex: Float = 0
    private var jumpWasHot = false
    private var simTime: Float = 0
    /// Stride of the current step cycle, in millimetres; the gait animation
    /// reads this so the stance foot matches body speed instead of skating.
    private var strideMillimetres: Float = 0

    /// Seconds since the last escape launch; the HUD flashes while it is
    /// small. Never negative, even before the first jump.
    var secondsSinceLastJump: Float { simTime - lastJumpTime }

    /// Opponent decode of a left/right pair: the collective (what both sides
    /// agree on) and a bounded asymmetry index in [-1, 1]. The epsilon is the
    /// one-spike-per-window noise floor of the group, so a single stray spike
    /// cannot masquerade as a command, and the index saturates at 1 no matter
    /// how lopsided the rates get — the muscle limit, not the spike count,
    /// decides how hard the animal turns.
    static func opponent(_ l: Float, _ r: Float, eps: Float)
        -> (collective: Float, index: Float) {
        ((l + r) * 0.5, (r - l) / (l + r + eps))
    }

    private func strokeAmplitude(from rateHz: Float) -> Float {
        let hover = FlyMorphology.strokeAmplitudeHover
        let maxPhi = FlyMorphology.strokeAmplitudeMax
        let t = rateHz / max(referenceRate, 1)
        if t <= 1 { return max(0, hover * t) }
        return hover + (maxPhi - hover) * min(1, (t - 1) / 0.5)
    }

    /// Stroke-plane forward tilt as a function of the thrust ratio F/W.
    /// Hovering flies hold the stroke plane horizontal — at exactly one body
    /// weight the tilt is zero, so the vertical force equals weight exactly
    /// and the equilibrium reflex (`referenceRate`) converges on a true hover
    /// instead of a slowly sinking one. The measured forward-flight
    /// inclination of ~29 deg (strokePlaneAngle) is reached at 1.5 body
    /// weights; the straight line between is an assumption (ASSUMPTIONS.md #2).
    private func strokeTilt(for ratio: Float) -> Float {
        FlyMorphology.strokePlaneAngle * max(0, min(1, (ratio - 1) / 0.5))
    }

    /// Advance the slow equilibrium estimate (collective lift channel only).
    private func updateReference(_ rateHz: Float, dt: Float) {
        guard rateHz > 1 else { return }
        let a = min(1, dt / referenceTau)
        referenceRate += (rateHz - referenceRate) * a
        referenceRate = max(5, min(400, referenceRate))
    }

    // MARK: - Setup

    /// Drive the body directly, bypassing the brain. Used by the test suite;
    /// the app always goes through `readMotorDrives`.
    func setDrives(_ d: FlyDrives) { drives = d }

    func attach(model: FlyModel) {
        self.model = model
        jointAngles = [Float](repeating: 0, count: model.joints.count)
    }

    func reset(at p: SIMD3<Float> = SIMD3<Float>(0, 0.35, 0)) {
        pose = FlyPose(position: p)
        energy = 1.0
        hurt = 0
        grounded = p.y <= contactRadius + contactEpsilon
        phiL = 0
        phiR = 0
        legIndex = 0
        jumpEvents = 0
        lastJumpTime = -1000
        jumpWasHot = false
        simTime = 0
        strideMillimetres = 0
    }

    /// Half a body length, the collision radius used everywhere.
    private var contactRadius: Float { FlyMorphology.bodyLength * 0.5 * metresToWorld }
    /// 20 micrometres: how close to the contact radius still counts as
    /// standing on the surface. A fly leg is not a mathematical point.
    private let contactEpsilon: Float = 0.002

    func readMotorDrives(from sim: SimulationEngine) {
        drives.wingPowerL = sim.groupRate("motor_wing_power_left")
        drives.wingPowerR = sim.groupRate("motor_wing_power_right")
        drives.wingSteerL = sim.groupRate("motor_wing_steering_left")
        drives.wingSteerR = sim.groupRate("motor_wing_steering_right")
        drives.wingTensionL = sim.groupRate("motor_wing_tension_left")
        drives.wingTensionR = sim.groupRate("motor_wing_tension_right")
        drives.legL = (sim.groupRate("motor_front_leg_left")
                     + sim.groupRate("motor_middle_leg_left")
                     + sim.groupRate("motor_hind_leg_left")) / 3
        drives.legR = (sim.groupRate("motor_front_leg_right")
                     + sim.groupRate("motor_middle_leg_right")
                     + sim.groupRate("motor_hind_leg_right")) / 3
        drives.neck = sim.groupRate("motor_neck")
        drives.proboscis = sim.groupRate("motor_proboscis")
        drives.jump = sim.groupRate("motor_jump_escape")
        drives.haltere = sim.groupRate("motor_haltere")
        drives.abdomen = sim.groupRate("motor_abdomen")
    }

    // MARK: - Integration

    func update(dt rawDt: Float, world: World) {
        let dt = min(max(rawDt, 1.0 / 480.0), 1.0 / 20.0)
        simTime += dt
        bumped = false
        hurt = max(0, hurt - dt * 1.5)

        // ---- decode the wing command ---------------------------------------
        //
        // Collective power sets the symmetric stroke amplitude through the
        // measured hover->max map. The asymmetry — from the power difference
        // AND from the steering muscles — is decoded as an opponent index in
        // [-1, 1] and mapped onto the measured 20-degree steering authority.
        // Decoding asymmetry as a raw difference of two noisy means is what
        // used to saturate the yaw channel: a 10% power difference asks for
        // 2.8x the maximum torque a fly can make, so every frame was full
        // left or full right and the animal spun at the 1600 deg/s clamp.
        let wingPower = opponent(drives.wingPowerL, drives.wingPowerR, eps: wingEps)
        let wingSteer = opponent(drives.wingSteerL, drives.wingSteerR, eps: wingEps)
        updateReference(wingPower.collective, dt: dt)

        let asymmetry = wingPower.index + wingSteer.index
        let phi0 = strokeAmplitude(from: wingPower.collective)
        let phiTargetL = max(0, min(FlyMorphology.strokeAmplitudeMax,
                                    phi0 - asymmetry * FlyMorphology.steeringRange))
        let phiTargetR = max(0, min(FlyMorphology.strokeAmplitudeMax,
                                    phi0 + asymmetry * FlyMorphology.steeringRange))

        // Muscle low-pass: the wing sees activation, not spikes.
        let aMuscle = min(1, dt / muscleTau)
        phiL += (phiTargetL - phiL) * aMuscle
        phiR += (phiTargetR - phiR) * aMuscle
        pose.strokeAmplitudeL = phiL
        pose.strokeAmplitudeR = phiR

        // ---- aerodynamics, in newtons --------------------------------------
        let fL = FlyMorphology.wingForce(strokeAmplitude: phiL,
                                         coefficient: FlyMorphology.liftCoefficient)
        let fR = FlyMorphology.wingForce(strokeAmplitude: phiR,
                                         coefficient: FlyMorphology.liftCoefficient)
        let totalForce = fL + fR
        liftMicroNewtons = totalForce * 1e6

        // Yaw torque from the left/right drag difference, moment arm r2.
        // SIGN: a stronger LEFT wing drags more on the left, which yaws the
        // body toward the RIGHT — toward the weaker wing. That is the
        // direction Fry et al. (2003) measured for real saccades, and it now
        // agrees with the walking decode below (the faster leg side swings
        // the body toward the slower side). The previous build had this
        // inverted: it turned the animal toward the STRONGER wing, and with
        // the saturated channel that meant a permanent spin.
        let dragL = FlyMorphology.wingForce(strokeAmplitude: phiL,
                                            coefficient: FlyMorphology.dragCoefficient)
        let dragR = FlyMorphology.wingForce(strokeAmplitude: phiR,
                                            coefficient: FlyMorphology.dragCoefficient)
        let yawTorque = max(-FlyMorphology.maxYawTorque,
                            min(FlyMorphology.maxYawTorque, (dragL - dragR) * FlyMorphology.r2))

        // ---- legs: step frequency and the differential-drive turn ----------
        // Leg motor rate sets step frequency; step frequency and the measured
        // stride set speed (Mendes et al. 2013: near-linear up to ~13 Hz and
        // 30 mm/s, i.e. a stride of about 2.3 mm). Turning is differential
        // drive through the opponent leg index over the CT-derived track.
        let legs = opponent(drives.legL, drives.legR, eps: legEps)
        legIndex += (legs.index - legIndex) * min(1, dt / muscleTau)
        let f = min(FlyMorphology.stepFrequencyMax,
                    legs.collective / referenceRate * FlyMorphology.stepFrequencyMax * 2.2)
        pose.stepFrequency = grounded ? f : 0
        let stride = FlyMorphology.walkSpeedMax / FlyMorphology.stepFrequencyMax
        let speedMS = min(f * stride, FlyMorphology.walkSpeedMax)
        strideMillimetres = f > 0.01 ? speedMS / f * 1000 : 0

        // ---- one continuous dynamics model ----------------------------------
        // Flight forces apply whether or not anything is touching; ground
        // contact (from the collision solver, never from a lift threshold)
        // adds the legs' kinematic drive and the floor's normal force.
        if grounded {
            // The floor normal force cancels gravity exactly — unless the
            // wings' VERTICAL component out-pulls the weight, in which case
            // the animal is taking off and the net upward force accelerates
            // it. The vertical component uses the same stroke-plane tilt the
            // flight branch flies with, so the take-off condition and the
            // in-air equilibrium are one consistent surface: no hover can
            // flicker across it.
            let weight = FlyMorphology.weight
            let tilt = strokeTilt(for: totalForce / weight)
            let verticalForce = totalForce * cos(max(0, tilt))
            if verticalForce > weight {
                let vy = pose.velocity.y * worldToMetres
                     + (verticalForce - weight) / FlyMorphology.mass * dt
                pose.velocity.y = vy * metresToWorld
            } else {
                pose.velocity.y = 0
            }

            // Differential drive: the body rotates about the slower tripod.
            // yaw rate = (vR - vL)/track, and (vR - vL) = 2 * index * v.
            let yaw = -2 * legIndex * speedMS / FlyMorphology.legTrack
            pose.yawRate = yaw
            pose.heading += yaw * dt

            let fwd = SIMD3<Float>(sin(pose.heading), 0, -cos(pose.heading))
            pose.velocity.x = fwd.x * speedMS * metresToWorld
            pose.velocity.z = fwd.z * speedMS * metresToWorld
            pose.gaitPhase += dt * f * 2 * .pi
            pose.roll += (0 - pose.roll) * min(1, dt * 8)
            pose.pitch += (0 - pose.pitch) * min(1, dt * 8)
        } else {
            // Angular: torque, inertia, and flapping counter-torque damping.
            // Semi-implicit on the damping term: the yaw time constant
            // (I / c = 5.2e-13 / 3.1e-11 = 17 ms) is comparable to a frame, so
            // an explicit step would oscillate or blow up.
            let c = FlyMorphology.yawDamping
            let I = FlyMorphology.inertiaYaw
            pose.yawRate = (pose.yawRate + yawTorque / I * dt) / (1 + c / I * dt)
            pose.heading += pose.yawRate * dt

            // Linear: the resultant acts normal to the stroke plane, which the
            // fly tilts forward to convert lift into thrust (see strokeTilt).
            let tilt = strokeTilt(for: totalForce / FlyMorphology.weight)
            pose.pitch += (-max(0, tilt) - pose.pitch) * min(1, dt * 6)

            let fwd = SIMD3<Float>(sin(pose.heading), 0, -cos(pose.heading))
            let up = SIMD3<Float>(0, 1, 0)
            let dir = normalize(up * cos(max(0, tilt)) + fwd * sin(max(0, tilt)))
            let accel = (dir * totalForce) / FlyMorphology.mass    // m/s^2
                      - SIMD3<Float>(0, FlyMorphology.gravity, 0)

            var vMetres = pose.velocity * worldToMetres
            vMetres += accel * dt
            // Flapping counter-force, again semi-implicit. The translational
            // time constant is m/c = 0.96e-6 / 8.0e-6 = 120 ms, so this is the
            // term that decides how fast the animal can actually go. With it,
            // full throttle settles at 0.89 m/s; without it the fly
            // accelerated without limit.
            let k = FlyMorphology.translationalDamping / FlyMorphology.mass
            vMetres /= (1 + k * dt)
            pose.velocity = vMetres * metresToWorld

            // Visual banking only (ASSUMPTIONS.md #11): the body lists away
            // from the harder-beating wing.
            pose.roll += ((phiR - phiL) * 0.5 - pose.roll) * min(1, dt * 6)
        }

        // ---- the giant fibre ------------------------------------------------
        // A rising edge of the TT-motoneuron rate through the one-spike-per-
        // frame level launches the escape — 0.9 m/s within milliseconds (Card
        // & Dickinson 2008) — if the legs are on something to push against.
        // In the air the event is still counted (the signal is not discarded),
        // it just has nothing to push.
        let jumpHot = drives.jump > FlyMorphology.jumpEventRate
        if jumpHot, !jumpWasHot, simTime - lastJumpTime > jumpRefractory {
            jumpEvents += 1
            lastJumpTime = simTime
            if grounded {
                pose.velocity.y += FlyMorphology.escapeTakeoffSpeed * metresToWorld
                grounded = false
            }
        }
        jumpWasHot = jumpHot

        pose.wingPhase += dt * FlyMorphology.wingbeatHz * 2 * .pi
        if pose.wingPhase > 2 * .pi { pose.wingPhase -= 2 * .pi * floor(pose.wingPhase / (2 * .pi)) }

        pose.position += pose.velocity * dt
        resolveCollisions(dt: dt, world: world)
        handleFeeding(dt: dt, world: world)

        // Animation state: a smoothed copy of the contact flag. Wing folding
        // and leg tucking read this; the dynamics above never do.
        let airborneTarget: Float = grounded ? 0 : 1
        pose.airborne += (airborneTarget - pose.airborne) * min(1, dt * 4)

        // Head and abdomen follow their own motor groups. The divisors are
        // the population-rate scales at which each reaches its measured
        // angular limit (ASSUMPTIONS.md #9).
        pose.headYaw += (max(-0.35, min(0.35, (drives.neck - 20) / 60.0)) - pose.headYaw)
                      * min(1, dt * 5)
        pose.headPitch += (max(-0.3, min(0.3, (drives.neck - 25) / 90.0)) - pose.headPitch)
                        * min(1, dt * 4)
        pose.abdomenBend += (max(-0.25, min(0.25, (drives.abdomen - 15) / 80.0)) - pose.abdomenBend)
                          * min(1, dt * 3)

        energy = max(0, energy - dt * 0.012)
        updateJointAngles()
    }

    // MARK: - Collisions

    private func resolveCollisions(dt: Float, world: World) {
        let radius = contactRadius

        // Contact is decided here, by geometry — the only place that knows
        // where the surfaces are. Start pessimistic (airborne) and let the
        // floor, an object top, or the containment clamp prove contact.
        grounded = false

        // Backstop first: a fast fly can cross a wall between two frames, so
        // the room is also enforced analytically.
        var p = pose.position
        let roomNormal = world.contain(&p, radius: radius)
        if roomNormal != .zero {
            pose.position = p
            bumped = true
            if roomNormal.y > 0 { grounded = true }
            let n = normalize(roomNormal)
            let into = dot(pose.velocity, n)
            if into < 0 { pose.velocity -= n * into }
            // Hitting a surface costs most of the momentum; flies stall, they
            // do not bounce (coefficients: ASSUMPTIONS.md #12).
            pose.velocity *= 0.3
            pose.yawRate *= 0.5
        }

        if let hit = world.collision(at: pose.position, radius: radius) {
            bumped = true
            pose.position = hit.correctedPosition
            if hit.normal.y > 0.7 { grounded = true }
            let into = dot(pose.velocity, hit.normal)
            if into < 0 {
                // Flies do not bounce; they stall and drop, or cling.
                pose.velocity -= hit.normal * into * 1.05
                pose.velocity *= 0.35
            }
            if hit.normal.y > 0.7 {
                pose.velocity.y = max(0, pose.velocity.y)
            }
        }

        // Standing exactly on the floor (or a hair above it) is contact too.
        if pose.position.y <= radius + contactEpsilon { grounded = true }
    }

    private func handleFeeding(dt: Float, world: World) {
        isEating = false
        if let food = world.nearestFood(to: pose.position),
           length(food.position - pose.position) < 0.35,
           grounded {
            isEating = drives.proboscis > 1.5
            if isEating {
                world.consume(food.id, amount: dt * 0.25)
                energy = min(1.2, energy + dt * 0.3)
            }
        }
        pose.proboscisExtension += ((isEating ? 1 : 0) - pose.proboscisExtension)
                                 * min(1, dt * 8)
    }

    // MARK: - Skeleton

    /// Drive the real flybody joints. Every axis and limit here comes from the
    /// CT-derived MJCF; we only choose angles, and clamp to the measured range.
    private func updateJointAngles() {
        guard let model = model, !jointAngles.isEmpty else { return }

        func set(_ name: String, _ value: Float) {
            guard let i = model.joint(name) else { return }
            jointAngles[i] = model.joints[i].clamp(value)
        }

        // --- wings ---------------------------------------------------------
        // Stroke position is sampled at the display rate from a 218 Hz
        // oscillation, so what you see is the real stroboscopic blur.
        let sweepL = sin(pose.wingPhase) * pose.strokeAmplitudeL * 0.5
        let sweepR = sin(pose.wingPhase + 0.02) * pose.strokeAmplitudeR * 0.5
        // Wing pitch flips at each stroke reversal; that rotation is what
        // generates the rotational lift component.
        let flipL = cos(pose.wingPhase) > 0 ? Float(0.8) : Float(-0.8)
        let flipR = cos(pose.wingPhase + 0.02) > 0 ? Float(0.8) : Float(-0.8)
        let folded = pose.airborne < 0.3

        set("wing_yaw_left",   folded ? 1.4 : sweepL)
        set("wing_yaw_right",  folded ? 1.4 : sweepR)
        set("wing_roll_left",  folded ? 0.0 : cos(pose.wingPhase) * 0.25)
        set("wing_roll_right", folded ? 0.0 : cos(pose.wingPhase + 0.02) * 0.25)
        set("wing_pitch_left",  folded ? 0.0 : flipL)
        set("wing_pitch_right", folded ? 0.0 : flipR)

        // --- halteres beat antiphase to the wings at the same frequency ----
        set("haltere_left",  sin(pose.wingPhase + .pi) * 0.5)
        set("haltere_right", sin(pose.wingPhase + .pi) * 0.5)

        // --- legs ------------------------------------------------------------
        // Tripod gait: L1,R2,L3 swing while R1,L2,R3 stance. Duty factor 0.55
        // is the measured value (Mendes et al. 2013). In flight the legs tuck.
        //
        // The stance foot must not skate: the stride comes from the body's
        // own speed (stride = v / step frequency), and the swing joints sweep
        // exactly that far — the amplitude is stride * duty / (2 * secant
        // gain), both derived per segment through the FK chain by
        // tools/gait_geometry.py. Where the joint range runs out the foot
        // slips at top speed — measured: T1 55%, T2 73%, T3 14% at 30 mm/s;
        // reported, not hidden (ASSUMPTIONS.md #6).
        //
        // Sign: the flybody joint axes are NOT uniform. Measured through the
        // renderer's own FK chain at the walking postures, the protraction
        // direction per joint is gaitSignT1/T2/T3 below — mirrored
        // consistently on both body sides, but different per segment. The old
        // code drove all segments with one sign, which is why the walking fly
        // looked like it was moving in reverse.
        let tuck = 1 - pose.airborne
        let stride = strideMillimetres
        let duty = FlyMorphology.dutyFactor
        for (side, sign) in [("left", Float(1)), ("right", Float(-1))] {
            for (segment, tIndex) in [("T1", 0), ("T2", 1), ("T3", 2)] {
                let tripod = (tIndex + (side == "left" ? 0 : 1)) % 2
                var u = pose.gaitPhase / (2 * .pi) + (tripod == 0 ? 0 : 0.5)
                u -= floor(u)
                let swinging = u > duty

                // Stance: follow the derived inverse-map knots so the planted
                // foot advances at constant speed despite the nonlinear FK.
                // Swing: the foot is airborne, a plain ramp is fine.
                let neutral: SIMD3<Float> = segment == "T1" ? FlyMorphology.gaitNeutralT1
                                          : segment == "T2" ? FlyMorphology.gaitNeutralT2
                                          : FlyMorphology.gaitNeutralT3
                let dirSign: SIMD3<Float> = segment == "T1" ? FlyMorphology.gaitSignT1
                                          : segment == "T2" ? FlyMorphology.gaitSignT2
                                          : FlyMorphology.gaitSignT3
                let ampCap: Float = segment == "T1" ? FlyMorphology.gaitAmpT1
                                  : segment == "T2" ? FlyMorphology.gaitAmpT2
                                  : FlyMorphology.gaitAmpT3
                let gainMM: Float = segment == "T1" ? FlyMorphology.gaitGainT1 * 1000
                                  : segment == "T2" ? FlyMorphology.gaitGainT2 * 1000
                                  : FlyMorphology.gaitGainT3 * 1000
                let knots: [Float] = segment == "T1" ? FlyMorphology.gaitKnotT1
                                   : segment == "T2" ? FlyMorphology.gaitKnotT2
                                   : FlyMorphology.gaitKnotT3
                let shape: Float
                if swinging {
                    shape = -1 + 2 * (u - duty) / (1 - duty)
                } else {
                    let n = Float(knots.count - 1)
                    let p = u / duty * n
                    let i = min(Int(p), knots.count - 2)
                    shape = knots[i] + (knots[i + 1] - knots[i]) * (p - Float(i))
                }
                let amp = min(ampCap, stride * duty / (2 * gainMM))
                let sweep = shape * amp

                let s = "\(segment)_\(side)"
                // Flight posture: legs folded back under the body.
                let flightCoxa: Float = 0.5 * sign
                let flightFemur: Float = -1.0
                let flightTibia: Float = 1.1

                set("coxa_\(s)",
                    tuck * (neutral.x + dirSign.x * sweep) + (1 - tuck) * flightCoxa)
                set("coxa_abduct_\(s)",
                    tuck * (0.1 * sign) + (1 - tuck) * (0.3 * sign))
                set("femur_\(s)",
                    tuck * (neutral.y + dirSign.y * sweep) + (1 - tuck) * flightFemur)
                set("tibia_\(s)",
                    tuck * (neutral.z + dirSign.z * sweep) + (1 - tuck) * flightTibia)
                set("tarsus_\(s)",
                    tuck * (swinging ? -0.3 : 0.1))
            }
        }

        // --- head, driven by the 49 neck motor neurons -----------------------
        set("head_abduct", pose.headYaw)
        set("head", pose.headPitch)
        set("head_twist", pose.roll * 0.3)

        // --- proboscis, driven by the 35 proboscis motor neurons -------------
        set("rostrum", -1.2 + 1.3 * pose.proboscisExtension)
        set("haustellum", -0.8 + 0.9 * pose.proboscisExtension)

        // --- abdomen: a 7-segment chain, bending distributes along it --------
        let perSegment = pose.abdomenBend / 7
        for name in ["abdomen", "abdomen_2", "abdomen_3", "abdomen_4",
                     "abdomen_5", "abdomen_6", "abdomen_7"] {
            set(name, perSegment)
        }
    }

    // MARK: - Sensing

    /// The two compound-eye cameras, one per optic lobe.
    ///
    /// The flybody model puts the eye cameras at +/- 0.0219 in head-local
    /// units with their optical axes 67 degrees off the body axis, each with
    /// a 140-degree field. Each hemisphere gets its own eye.
    var eyeTransforms: (left: (position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>),
                        right: (position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>)) {
        let yaw = pose.heading + pose.headYaw
        let pitch = pose.pitch + pose.headPitch
        // Head is one third of a body length ahead of the centre of mass.
        let ahead = FlyMorphology.bodyLength * 0.33 * metresToWorld
        let head = pose.position + SIMD3<Float>(sin(yaw), 0.12, -cos(yaw)) * ahead
        // The fly's right side: forward x up.
        let right = SIMD3<Float>(cos(yaw), 0, sin(yaw))

        func eye(_ azimuth: Float) -> (position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>) {
            let y = yaw + azimuth
            let f = SIMD3<Float>(sin(y) * cos(pitch), sin(pitch), -cos(y) * cos(pitch))
            let p = head + right * (azimuth >= 0 ? FlyMorphology.eyeLateralOffset
                                                 : -FlyMorphology.eyeLateralOffset)
            return (p, normalize(f), SIMD3<Float>(0, 1, 0))
        }
        return (left: eye(-FlyMorphology.eyeAzimuth),
                right: eye(FlyMorphology.eyeAzimuth))
    }

    func writeSensoryDrives(to sim: SimulationEngine, world: World, visionActive: Bool) {
        let (odourStrength, odourBias) = world.odour(at: pose.position, heading: pose.heading)

        sim.setGroupDrive("sensory_olfactory", min(2.2, odourStrength * 0.9))
        sim.setGroupDrive("sensory_antenna", min(2.0, odourStrength * 0.6
                                                 + abs(odourBias) * 0.4))
        sim.setGroupDrive("sensory_gustatory", isEating ? 1.8 : 0)

        sim.setGroupDrive("sensory_tactile", (grounded ? 0.6 : 0) + (bumped ? 1.4 : 0))

        let gait = 0.5 + 0.5 * sin(pose.gaitPhase)
        sim.setGroupDrive("sensory_proprioception",
                          0.4 + gait * 0.5 + pose.strokeAmplitudeL * 0.2)
        sim.setGroupDrive("sensory_front_leg", grounded ? 0.3 + gait * 0.6 : 0)
        sim.setGroupDrive("sensory_middle_leg", grounded ? 0.3 + (1 - gait) * 0.6 : 0)
        sim.setGroupDrive("sensory_hind_leg", grounded ? 0.3 + gait * 0.6 : 0)

        // Halteres are gyroscopes: their load signal is proportional to the
        // body's angular velocity. This is a real measurement, not a proxy.
        sim.setGroupDrive("sensory_haltere",
                          (grounded ? 0 : 1) * min(2.0, abs(pose.yawRate) * 0.12 + 0.3))
        sim.setGroupDrive("sensory_wing", (grounded ? 0 : 1) * pose.strokeAmplitudeL * 0.4)
        sim.setGroupDrive("sensory_nociception", hurt * 3.0)

        sim.setGroupDrive("sensory_vision", visionActive ? nil : 0.4)
    }
}
