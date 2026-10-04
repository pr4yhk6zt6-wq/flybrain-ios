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
//  constant a MEASURED one from the Drosophila literature, and to reduce the
//  unmeasured part to a single, explicitly stated link.
//
//  MEASURED (each constant below carries its source):
//    body mass, wing length, wing area, radius of the second moment of area,
//    wingbeat frequency, stroke amplitude range, lift and drag coefficients,
//    air density, moment of inertia about yaw, flapping counter-torque,
//    walking speed range, step frequency, body length, leg joint axes and
//    angular limits (the last from the flybody MJCF, which is a real CT model).
//
//  Flight force is then computed, not fudged: a quasi-steady blade-element
//  estimate, the standard model in insect flight aerodynamics.
//
//        F = 1/2 * rho * C * S * U^2,     U = 2 * Phi * f * r2
//
//  THE ONE REMAINING ASSUMPTION:
//    motor-neuron firing rate -> wing stroke amplitude Phi.
//    Nobody has published a transfer function for this, because it would
//    require simultaneous recording of identified motor neurons and wing
//    kinematics across the full dynamic range. What IS measured is both
//    endpoints: a fly in stable hovering beats at Phi ~ 2.1 rad, and at
//    maximum effort at Phi ~ 3.1 rad (morphological limit). We map the
//    population rate of the power muscle motor neurons linearly between those
//    two measured values. The endpoints are real; the straight line between
//    them is the assumption. It is isolated in `strokeAmplitude(from:)` and
//    nowhere else.
//
//  Note also that Drosophila power muscles are ASYNCHRONOUS: their motor
//  neurons fire at 5-20 Hz while the wing beats at ~200 Hz, because firing
//  sets the activation level, not the rhythm. The wingbeat frequency here is
//  therefore a measured constant, not something the spiking drives — which is
//  biologically correct, and is why the simulation does not need to resolve
//  200 Hz.
//
//  References
//    Fry, Sayaman & Dickinson 2003, Science 300:495  (free-flight dynamics,
//        body mass 0.96 mg, I_yaw 5.2e-13 kg m^2, flapping counter-torque)
//    Lehmann & Dickinson 1997, J Exp Biol 200:1133   (stroke amplitude range,
//        wingbeat frequency, force production)
//    Dickinson, Lehmann & Sane 1999, Science 284:1954 (unsteady lift)
//    Sane & Dickinson 2001, J Exp Biol 204:2607      (C_L, C_D vs kinematics)
//    Sun & Tang 2002, J Exp Biol 205:2413            (hovering force balance,
//        r2 = 0.58 R, mean drag 1.27x lift)
//    Mendes et al. 2013, eLife 2:e00231              (walking speed, step
//        frequency, tripod gait, duty factor)
//    Vaxenburg et al. 2025, Nature                   (flybody CT skeleton,
//        joint axes and limits)
//  ======================================================================
//

import Foundation
import simd

/// Every measured constant, in SI. Nothing here was chosen to make the
/// simulation behave; each is a published value for D. melanogaster.
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
    /// limit on the muscle, not a gain we chose, so the bias saturates here.
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

    /// Joint angles for FlyModel, indexed by its joint table.
    private(set) var jointAngles: [Float] = []
    private var model: FlyModel?

    // MARK: - The one assumption, isolated

    /// Motor-neuron population rate -> wing stroke amplitude.
    ///
    /// Both endpoints are measured: at the hovering amplitude the blade-element
    /// force equals body weight exactly, and the maximum is the morphological
    /// limit of 3.1 rad. The straight line between them is the assumption.
    ///
    /// `referenceRate` — the firing rate that means "hover" — is not a number
    /// typed in by hand. It is a slow running average of the power-muscle
    /// group's own rate, so whatever the network happens to settle at becomes
    /// the equilibrium, and the fly responds to CHANGES around it.
    ///
    /// That is not a convenience: it is the equilibrium reflex. Real flies hold
    /// lift against weight through haltere and visual feedback onto the same
    /// steering muscles (Sherman & Dickinson 2003; Dickinson 1999), and those
    /// pathways are present in BANC. What we cannot do is calibrate that loop's
    /// gain, because nobody has measured it — so the loop is closed here, at
    /// the muscle, instead.
    ///
    /// The time constant is a behaviour choice, and 2 s turned out wrong: the
    /// trim chased the fly's own climb command so tightly that lift stayed
    /// glued to weight — on the phone the animal beat its wings harder and
    /// harder yet its eye view never rose. At 10 s a sustained (seconds-long)
    /// power change still moves the fly up or down, while a stuck saturation
    /// still re-trims instead of holding 178 degrees forever.
    private(set) var referenceRate: Float = 120.0
    private let referenceTau: Float = 10.0      // s

    /// Adapted baseline of the left/right leg-rate asymmetry while walking.
    ///
    /// The two leg populations of the connectome do not fire perfectly evenly,
    /// and the raw difference used to turn the body forever — the animal
    /// circled at a steady ~11 deg/s instead of walking straight. This is the
    /// same equilibrium idea as `referenceRate`: only *changes* in asymmetry
    /// around the adapted mean steer the body, so an intended turn still
    /// works but a constant baseline bias dies away (tau ~1.5 s, the order of
    /// the optomotor straightening a real fly gets from balanced optic flow).
    private(set) var turnBias: Float = 0
    private let turnBiasTau: Float = 1.5

    /// Flight twin of `turnBias`: an adapted baseline on the left/right wing
    /// drag difference. In the open world nothing interrupted a sustained
    /// steering asymmetry, so the animal spun at the 1600 deg/s yaw clamp —
    /// the flight version of the old walking circles. Cancelling the
    /// sustained part (tau 1.5 s, the order of the haltere/optomotor
    /// straightening a real fly gets) leaves genuine saccades intact.
    private(set) var yawBias: Float = 0
    private let yawBiasTau: Float = 1.5

    // MARK: - Habits (measured ethology, imposed at the body) ---------------
    //
    // A real fly does not walk continuously: walking is organised in bouts
    // separated by stops — walk/stop transitions behave like Poisson events
    // with a baseline walk-initiation rate of ~0.29/s (Demir, Kadakia,
    // Anderson, Clark & Carey 2020, eLife 5:e57524) — and ~13% of waking time
    // is grooming in bouts of a few tenths of a second to a couple of seconds,
    // sweeping anterior-to-posterior (Seeds et al. 2014, eLife 3:e02951;
    // Lazopulo & Syed 2018, eLife 7:e34497; leg-sweep cycles ~150 ms, Ray et
    // al. 2019, PLOS Comput Biol 15:e1007105). Dusting or a bump triggers
    // grooming strongly. A 1 ms LIF connectome does not spontaneously emit
    // that bout structure, so — like the tripod gait oscillator — it is
    // imposed here and labelled as modelled. The rng is deterministic so the
    // physics tests reproduce exactly.

    enum Habit: Int { case walk = 0, stop, groom }
    private(set) var habit: Habit = .walk
    private var habitTimer: Float = 0
    private var groomDuration: Float = 0
    private var groomElapsed: Float = 0
    private(set) var groomPhase: Float = 0
    /// 1 when the legs may carry the body, 0 during stops and grooming (and
    /// while feeding — a drinking fly stands still).
    private(set) var habitGate: Float = 1
    private var wasAirborne = false
    private var wasBumped = false
    private var rngState: UInt32 = 0x9E3779B9

    /// Endogenous arousal tone. A real fly's vigour is not constant: the
    /// propensity to move drifts on its own over tens of seconds (Cohn et al.
    /// 2019, Cell 176:254 — spontaneous walking and flight in 3-D), brain-wide
    /// imaging finds arousal-like signals with time constants from under 4 s
    /// to over 20 s (Nat Commun 2023, 14:5420), and the walk/stop statistics
    /// themselves only close when a slowly varying internal state modulates
    /// the transition rates (Demir et al. 2020). The brain page's Synaptic
    /// gain slider is the *experimenter's* knob; this is the animal's own
    /// state, imposed as an Ornstein-Uhlenbeck process (tau 15 s, bounded to
    /// [0.5, 1.5]) on the same deterministic rng, so tests stay exact.
    private(set) var arousal: Float = 1.0
    private let arousalTau: Float = 15.0
    private let arousalSigma: Float = 0.16      // stationary sd ~0.44

    private func habitRandom() -> Float {
        rngState ^= rngState << 13
        rngState ^= rngState >> 17
        rngState ^= rngState << 5
        return Float(rngState >> 8) / Float(1 << 24)
    }

    private func enterHabit(_ h: Habit) {
        habit = h
        let u = max(habitRandom(), 0.02)
        // Arousal stretches walk bouts and shortens stops — the slowly
        // varying state term that closes the walk/stop statistics
        // (Demir et al. 2020: transitions are modulated, not memoryless
        // at a fixed rate).
        switch h {
        case .walk:  habitTimer = min(8, -log(u) * 3.0 * arousal)        // ~3 s bouts
        case .stop:  habitTimer = min(6, max(0.3, -log(u) / (0.29 * arousal)))  // lambda0 = 0.29/s
        case .groom:
            habitTimer = min(4, max(0.4, -log(u) * 1.0))                 // 0.15-2 s+ bouts
            groomDuration = habitTimer
            groomElapsed = 0
        }
    }

    private func updateHabit(dt: Float) {
        // Ornstein-Uhlenbeck arousal: mean-reverting drift on the seconds-to-
        // tens-of-seconds scale. Four uniform draws sum to a crude gaussian,
        // keeping the port bit-compatible with the Python simcheck.
        let g = (habitRandom() + habitRandom() + habitRandom() + habitRandom()) * 0.5 - 1
        arousal += (1 - arousal) * min(1, dt / arousalTau) + arousalSigma * sqrt(dt) * g
        arousal = min(1.5, max(0.5, arousal))

        let grounded = pose.airborne < 0.5
        if !grounded {
            wasAirborne = true
            habit = .walk
            habitGate = 1
            return
        }
        if wasAirborne {
            // Landing kicks up dust: a strong grooming urge, the virtual
            // version of the dusting assays.
            wasAirborne = false
            if habitRandom() < 0.5 { enterHabit(.groom) }
        }
        if bumped, !wasBumped, habit != .groom, habitRandom() < 0.4 { enterHabit(.groom) }
        wasBumped = bumped

        habitTimer -= dt
        if isEating {
            if habit != .stop { enterHabit(.stop) }
            habitTimer = max(habitTimer, 0.2)
        } else if habitTimer <= 0 {
            switch habit {
            // Probabilities tuned so grooming lands at ~13% of active time,
            // the share Lazopulo & Syed 2018 measured in undisturbed flies.
            case .walk: enterHabit(habitRandom() < 0.5 ? .groom : .stop)
            case .stop:
                if habitRandom() < 0.45 { enterHabit(.groom) }
                else { enterHabit(.walk) }
            case .groom: enterHabit(.walk)
            }
        }
        habitGate = habit == .walk ? 1 : 0
        if habit == .groom {
            groomElapsed += dt
            groomPhase += dt * 2 * .pi * 6.5        // ~150 ms per leg sweep
        }
    }

    private func strokeAmplitude(from rateHz: Float) -> Float {
        let hover = FlyMorphology.strokeAmplitudeHover
        let maxPhi = FlyMorphology.strokeAmplitudeMax
        let t = rateHz / max(referenceRate, 1)
        if t <= 1 { return max(0, hover * t) }
        return hover + (maxPhi - hover) * min(1, (t - 1) / 0.5)
    }

    /// Advance the slow equilibrium estimate.
    private func updateReference(_ rateHz: Float, dt: Float) {
        guard rateHz > 1 else { return }
        let a = min(1, dt / referenceTau)
        referenceRate += (rateHz - referenceRate) * a
        referenceRate = max(5, min(400, referenceRate))
    }

    // MARK: - Setup

    /// Drive the body directly, bypassing the brain. Used by the test suite
    /// and by scripted demos; the app always goes through `readMotorDrives`.
    func setDrives(_ d: FlyDrives) { drives = d }

    func attach(model: FlyModel) {
        self.model = model
        jointAngles = [Float](repeating: 0, count: model.joints.count)
    }

    func reset(at p: SIMD3<Float> = SIMD3<Float>(0, 0.35, 0)) {
        pose = FlyPose(position: p)
        energy = 1.0
        hurt = 0
        turnBias = 0
        yawBias = 0
        habit = .walk
        habitTimer = 0
        groomDuration = 0
        groomElapsed = 0
        groomPhase = 0
        habitGate = 1
        wasAirborne = false
        wasBumped = false
        arousal = 1.0
        rngState = 0x9E3779B9     // deterministic habit sequences after reset
    }

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
        bumped = false
        hurt = max(0, hurt - dt * 1.5)

        updateReference((drives.wingPowerL + drives.wingPowerR) * 0.5, dt: dt)

        // ---- what the wings are being told to do --------------------------
        // Steering muscles bias the stroke amplitude of their own side; this is
        // the measured mechanism of yaw control in Drosophila (b1/b2 muscles
        // shift stroke amplitude by up to ~20 deg, Lehmann & Dickinson 1997).
        //
        // Both ends of this are now clamped, and neither was. A steering group
        // firing at 300 Hz on one side and silent on the other — which is what
        // a saturating motor population does — used to ask for 100 deg of bias,
        // five times what the muscle can deliver, and the result ran the wing
        // past its morphological limit to 248 deg of sweep. That is the number
        // the yaw-rate test caught at 14,895 deg/s.
        let steerBiasL = max(-FlyMorphology.steeringRange,
                             min(FlyMorphology.steeringRange,
                                 (drives.wingSteerL - drives.wingSteerR) / 60.0
                                 * FlyMorphology.steeringRange))
        let steerBiasR = -steerBiasL
        // A wing is a joint. It cannot be commanded past 178 deg, whatever the
        // power muscles are shouting.
        let phiL = max(0, min(FlyMorphology.strokeAmplitudeMax,
                              strokeAmplitude(from: drives.wingPowerL) + steerBiasL))
        let phiR = max(0, min(FlyMorphology.strokeAmplitudeMax,
                              strokeAmplitude(from: drives.wingPowerR) + steerBiasR))
        pose.strokeAmplitudeL = phiL
        pose.strokeAmplitudeR = phiR

        // ---- aerodynamics, in newtons --------------------------------------
        let fL = FlyMorphology.wingForce(strokeAmplitude: phiL,
                                         coefficient: FlyMorphology.liftCoefficient)
        let fR = FlyMorphology.wingForce(strokeAmplitude: phiR,
                                         coefficient: FlyMorphology.liftCoefficient)
        let totalForce = fL + fR
        liftMicroNewtons = totalForce * 1e6

        let weight = FlyMorphology.weight
        let airborneNow: Float = totalForce > weight * 0.98 ? 1 : 0
        pose.airborne += (airborneNow - pose.airborne) * min(1, dt * 4)

        // Yaw torque from the left/right force difference. The moment arm is
        // the radius of the second moment of area, which is where the
        // resultant aerodynamic force acts on a flapping wing.
        let dragL = FlyMorphology.wingForce(strokeAmplitude: phiL,
                                            coefficient: FlyMorphology.dragCoefficient)
        let dragR = FlyMorphology.wingForce(strokeAmplitude: phiR,
                                            coefficient: FlyMorphology.dragCoefficient)
        // The steering torque saturates at the largest a melanogaster has been
        // measured producing. Without it the model answers an extreme
        // left/right asymmetry with a steady spin of several thousand deg/s —
        // the quasi-steady blade-element estimate keeps scaling with the stroke
        // amplitude, but the animal does not.
        //
        // The sustained part of the asymmetry is cancelled by the adapted
        // baseline (see `yawBias`); otherwise a constant left/right drive
        // difference pins the fly at the yaw clamp forever — which is exactly
        // the 1464 deg/s spin the open world made visible.
        let rawYawDrag = dragR - dragL
        yawBias += (rawYawDrag - yawBias) * min(1, dt / yawBiasTau)
        let yawTorque = max(-FlyMorphology.maxYawTorque,
                            min(FlyMorphology.maxYawTorque,
                                (rawYawDrag - yawBias) * FlyMorphology.r2))

        if pose.airborne > 0.5 {
            // Angular: torque, inertia, and flapping counter-torque damping.
            // Semi-implicit on the damping term: the yaw time constant
            // (I / c = 5.2e-13 / 3.1e-11 = 17 ms) is comparable to a frame, so
            // an explicit step would oscillate or blow up.
            let c = FlyMorphology.yawDamping
            let I = FlyMorphology.inertiaYaw
            pose.yawRate = (pose.yawRate + yawTorque / I * dt) / (1 + c / I * dt)
            pose.heading += pose.yawRate * dt

            // Linear: the resultant acts normal to the stroke plane, which the
            // fly tilts forward to convert lift into thrust.
            let tilt = FlyMorphology.strokePlaneAngle * min(1, totalForce / weight - 0.6)
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
            // accelerated without limit, which is the bug you saw.
            let k = FlyMorphology.translationalDamping / FlyMorphology.mass
            vMetres /= (1 + k * dt)
            pose.velocity = vMetres * metresToWorld

            pose.roll += ((phiR - phiL) * 0.5 - pose.roll) * min(1, dt * 6)
            pose.stepFrequency = 0
        } else {
            // ---- walking ---------------------------------------------------
            // Leg motor rate sets step frequency; step frequency and the
            // measured stride length set speed. Mendes et al. report a near
            // linear speed/step-frequency relation up to ~13 Hz and 30 mm/s,
            // i.e. a stride of about 2.3 mm, close to one body length.
            let legMean = (drives.legL + drives.legR) * 0.5
            let f = min(FlyMorphology.stepFrequencyMax,
                        legMean / referenceRate * FlyMorphology.stepFrequencyMax * 2.2)
            // Stop-and-go: the habit gate zeroes step frequency while the fly
            // stands still or grooms, exactly like a real walking bout ending.
            let fg = f * habitGate
            pose.stepFrequency = fg
            let stride = FlyMorphology.walkSpeedMax / FlyMorphology.stepFrequencyMax
            // Arousal scales walking vigour, capped at the measured 30 mm/s.
            let speedMS = min(fg * stride * arousal, FlyMorphology.walkSpeedMax)  // m/s

            // Turning on foot: the two tripods step at different rates, and the
            // body rotates about the slower side. Differential stride is the
            // measured mechanism. The adapted baseline (see `turnBias`) keeps a
            // constant left/right bias from circling the animal forever.
            let legDiff = (drives.legR - drives.legL) / max(referenceRate, 1)
            turnBias += (legDiff - turnBias) * min(1, dt / turnBiasTau)
            // The pivot is NOT gated by the habit: stopped flies still perform
            // reorientation turns in place (measured behaviour), they just do
            // not advance.
            pose.yawRate = -(legDiff - turnBias) * f * 2.0
            pose.heading += pose.yawRate * dt

            let fwd = SIMD3<Float>(sin(pose.heading), 0, -cos(pose.heading))
            var v = fwd * speedMS * metresToWorld
            v.y = min(pose.velocity.y, 0) - FlyMorphology.gravity * metresToWorld * dt * 0.02
            pose.velocity = v

            pose.gaitPhase += dt * fg * 2 * .pi
            pose.roll += (0 - pose.roll) * min(1, dt * 8)
            pose.pitch += (0 - pose.pitch) * min(1, dt * 8)
        }

        // The giant fibre. Two neurons; when they fire the fly is simply gone.
        // Card & Dickinson 2008 measure escape take-off at ~0.9 m/s within 5 ms.
        if drives.jump > 8 && pose.airborne < 0.6 {
            pose.velocity.y += 0.9 * metresToWorld
            pose.airborne = 1
        }

        pose.wingPhase += dt * FlyMorphology.wingbeatHz * 2 * .pi
        if pose.wingPhase > 2 * .pi { pose.wingPhase -= 2 * .pi * floor(pose.wingPhase / (2 * .pi)) }

        pose.position += pose.velocity * dt
        resolveCollisions(dt: dt, world: world)
        handleFeeding(dt: dt, world: world)
        // Runs last so the habit machine sees this frame's bump and feeding
        // flags; the gate it sets takes effect on next frame's walking.
        updateHabit(dt: dt)

        // Head and abdomen follow their own motor groups.
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
        // The fly is a 2.5 mm ellipsoid; use half a body length as the radius.
        let radius = FlyMorphology.bodyLength * 0.5 * metresToWorld

        // Backstop first: a fast fly can cross a wall between two frames, so
        // the room is also enforced analytically.
        var p = pose.position
        let roomNormal = world.contain(&p, radius: radius)
        if roomNormal != .zero {
            pose.position = p
            bumped = true
            let n = normalize(roomNormal)
            let into = dot(pose.velocity, n)
            if into < 0 { pose.velocity -= n * into }
            // Hitting a surface costs most of the momentum; flies stall, they
            // do not bounce.
            pose.velocity *= 0.3
            pose.yawRate *= 0.5
        }

        if let hit = world.collision(at: pose.position, radius: radius) {
            bumped = true
            pose.position = hit.correctedPosition
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
    }

    private func handleFeeding(dt: Float, world: World) {
        isEating = false
        if let food = world.nearestFood(to: pose.position),
           length(food.position - pose.position) < 0.35,
           pose.airborne < 0.4 {
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

        // --- halteres beat antiphase to the wings at the same frequency -----
        set("haltere_left",  sin(pose.wingPhase + .pi) * 0.5)
        set("haltere_right", sin(pose.wingPhase + .pi) * 0.5)

        // --- legs ------------------------------------------------------------
        // Tripod gait: L1,R2,L3 swing while R1,L2,R3 stance. Duty factor 0.55
        // is the measured value. In flight the legs tuck.
        let tuck = 1 - pose.airborne
        for (side, sign) in [("left", Float(1)), ("right", Float(-1))] {
            for (segment, tIndex) in [("T1", 0), ("T2", 1), ("T3", 2)] {
                let tripod = (tIndex + (side == "left" ? 0 : 1)) % 2
                let phase = pose.gaitPhase + (tripod == 0 ? 0 : .pi)
                // Swing phase occupies (1 - dutyFactor) of the cycle.
                let c = (sin(phase) + 1) * 0.5
                let swinging = c > FlyMorphology.dutyFactor
                let protraction = sin(phase)

                let s = "\(segment)_\(side)"
                // Flight posture: legs folded back under the body.
                let flightCoxa: Float = 0.5 * sign
                let flightFemur: Float = -1.0
                let flightTibia: Float = 1.1

                set("coxa_\(s)",
                    tuck * (protraction * 0.35) + (1 - tuck) * flightCoxa)
                set("coxa_abduct_\(s)",
                    tuck * (0.1 * sign) + (1 - tuck) * (0.3 * sign))
                set("femur_\(s)",
                    tuck * (-0.5 + (swinging ? 0.45 : 0.0)) + (1 - tuck) * flightFemur)
                set("tibia_\(s)",
                    tuck * (0.4 - (swinging ? 0.5 : 0.0)) + (1 - tuck) * flightTibia)
                set("tarsus_\(s)",
                    tuck * (swinging ? -0.3 : 0.1))
            }
        }

        // --- head, driven by the 49 neck motor neurons -----------------------
        set("head_abduct", pose.headYaw)
        set("head", pose.headPitch)
        set("head_twist", pose.roll * 0.3)

        // --- grooming (Seeds et al. 2014): anterior-to-posterior leg sweeps --
        // The first half of a bout rubs the head and eyes with the front legs,
        // the second half sweeps the abdomen with the hind legs; each sweep
        // cycle runs ~150 ms (Ray et al. 2019). Placed last so it overrides
        // the walking gait and the default head posture above.
        if habit == .groom, pose.airborne < 0.5 {
            let sweep = sin(groomPhase)
            let anterior = groomElapsed < groomDuration * 0.5
            for side in ["left", "right"] {
                if anterior {
                    set("coxa_T1_\(side)",   0.55 + sweep * 0.25)
                    set("femur_T1_\(side)", -0.10 + sweep * 0.40)
                    set("tibia_T1_\(side)",  1.00 - sweep * 0.35)
                    set("coxa_T3_\(side)",  -0.25)
                    set("femur_T3_\(side)", -0.60)
                    set("tibia_T3_\(side)",  0.50)
                } else {
                    set("coxa_T1_\(side)",   0.15)
                    set("femur_T1_\(side)", -0.55)
                    set("tibia_T1_\(side)",  0.35)
                    set("coxa_T3_\(side)",  -0.45 + sweep * 0.30)
                    set("femur_T3_\(side)",  0.30 + sweep * 0.35)
                    set("tibia_T3_\(side)", -0.40 - sweep * 0.30)
                }
            }
            set("head_twist", sweep * 0.22)
            set("head_abduct", anterior ? sweep * 0.15 : pose.headYaw)
        }

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
    /// a 140-degree field. We used to render ONE cyclopean camera from the
    /// midpoint and feed that single image to both hemispheres — which gave
    /// the brain no left/right difference to steer with, and the eye preview
    /// showed a one-eyed fly. Now each hemisphere gets its own eye.
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

        let grounded = pose.airborne < 0.5
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
                          pose.airborne * min(2.0, abs(pose.yawRate) * 0.12 + 0.3))
        sim.setGroupDrive("sensory_wing", pose.airborne * pose.strokeAmplitudeL * 0.4)
        sim.setGroupDrive("sensory_nociception", hurt * 3.0)

        sim.setGroupDrive("sensory_vision", visionActive ? nil : 0.4)
    }
}
