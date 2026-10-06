//
//  FlyLiveBody.swift
//  The animal, running, at whatever rate the phone can afford.
//
//  This is the class that replaced `frames.bin`. Where the Map screen used to
//  step a pre-recorded pose track at 25 Hz and interpolate between two
//  frames, it now advances *this* — a solver that is told a torque per joint
//  and works out the rest, with a floor under it and gravity through it — and
//  the meshes follow the state, never the other way round.
//
//  The loop, per substep:
//
//      excitation ──> antagonist muscle pair ──> torque ──> articulated-body
//      solve ──> contact wrench on 74 geoms ──> integrate ──> pose
//
//  `drive` is where the nerve cord goes. It is handed the current state and
//  returns an excitation offset per joint; whatever it returns is added to the
//  animal's resting muscle tone and clipped to the muscle's range. With no
//  drive attached the animal holds the posture its own actuators are
//  calibrated to hold — that is muscle tone, not behaviour, and it is what
//  makes the floor contact, the mass distribution and the joint springs all
//  have to be right at the same time.
//

import Foundation
import QuartzCore

/// What the cord is allowed to see. Every field is something a fly has a
/// sense organ for (docs/STEP3.md): joint angles and rates come from the
/// chordotonal organs, the load per leg from the campaniform sensilla on the
/// tarsi. Nothing here is ground truth handed to a controller — the cord
/// receives it through the same pools a real one does.
struct FlyProprioception {
    /// Joint angle and rate, per hinge, in the model's own order.
    var angle: [Double]
    var rate: [Double]
    /// Force on the floor under each leg, keyed "T1_left" ...
    var legLoad: [String: Double]
    /// Body height above the floor, cm, and the speed of the centre of mass.
    var height: Double
    var speed: Double
    /// How many legs currently have any load on them.
    var feetDown: Int
    /// The body's angular velocity, **body frame**, in degrees per second.
    ///
    /// This is what the halteres measure (item 7): `omega` in the solver is the
    /// root body's rate in the world frame, and the haltere sensitivity axes
    /// are in the body frame, so the body takes the world-frame rate into its
    /// own frame here — one rotation, once per cord update. Nothing else about
    /// it is invented: the group drives are computed from it by `HaltereGyro`,
    /// and what crosses the synapse is a current like any other.
    var angularRate: SIMD3<Double>
}

/// The animal, live.
final class FlyLiveBody {

    /// The asset it was built from, so the cord can be built from the same one
    /// (FlyCord.swift).
    let asset: FlyBodyAsset
    let dynamics: FlyDynamics
    let dt: Double

    /// Resting muscle tone: the excitation that holds the stance the animal's
    /// own actuators were measured holding. Not behaviour.
    private(set) var posture: [Double]
    /// What the muscles are actually being told, per joint: `posture + drive`.
    private(set) var excitation: [Double]

    /// The nerve cord. Give it the state, get back an excitation offset per
    /// joint. Set from the world screen once the connectome is up.
    var drive: ((FlyProprioception) -> [Double])?

    /// How often the cord is asked, in simulated milliseconds.
    ///
    /// The physics runs at 100 µs, which is the body's own scale; a ventral
    /// nerve cord is a 1 kHz machine and does not re-decide at 10 kHz. Asking
    /// it every substep would also rebuild the proprioception dictionary ten
    /// thousand times a simulated second for nothing. Between updates the
    /// excitation it returned is held, which is what a muscle sees anyway: it
    /// fuses twitches over tens of milliseconds (docs/ASSUMPTIONS.md #7).
    var cordIntervalMs: Double = 1.0
    /// Cord updates run so far, for the readout.
    private(set) var cordUpdates = 0
    private var nextCordAtMS: Double = 0
    private var cordOffset: [Double] = []

    // -- the wings (brief item 6) -------------------------------------------

    /// The wing force model, attached once the world's `wings` block is known.
    var wingAero: WingAero?
    /// The four wing pools' rates, asked of the cord once per cord update.
    var wingRates: (() -> [String: Double])?
    /// Where the wing beat has got to, 0...1. The app's own clock, integrated
    /// at the stroke frequency — not a lookup into a recording.
    private(set) var strokePhase = 0.0
    /// Each wing's stroke amplitude activation, from the pools' rates.
    private(set) var wingActivation = SIMD2<Double>(0, 0)
    /// The wrench the wings are putting on the thorax right now.
    private(set) var wingWrench = WingAero.Wrench()
    /// The beat's cycle mean, for the readout (recomputed a few times a second,
    /// not every substep: it is 720 samples and the HUD is 10 Hz).
    private(set) var wingBeat = WingAero.Beat()
    private(set) var liftOverWeight = 0.0
    private var lastWingReadoutMS: Double = -1e9
    private var thoraxIndex = -1
    private var wingYawJoints: (Int, Int) = (-1, -1)
    private var wingWrenchArray: [SVec6] = []

    /// Attach the wings. The spec is the `wings` block of `world.json`.
    func attachWings(spec: WingSpec?) {
        let aero = WingAero(spec: spec)
        wingAero = aero
        thoraxIndex = dynamics.bodyIndex(named: "thorax") ?? -1
        wingYawJoints = (dynamics.hingeIndex(named: "wing_yaw_left") ?? -1,
                         dynamics.hingeIndex(named: "wing_yaw_right") ?? -1)
        if thoraxIndex >= 0 {
            wingWrenchArray = [SVec6](repeating: .zero, count: dynamics.asset.bodies.count)
            dynamics.externalWrench = wingWrenchArray
        }
        wingBeat = aero.beat(activationLeft: 0, activationRight: 0, samples: 8)
    }

    /// One substep of wings: advance the beat, hold the two stroke joints at the
    /// measured kinematics with the joint's *own spring*, and hand the air's
    /// answer back to the solver as a force and a torque on the thorax.
    ///
    /// The boundary this draws is the honest one for item 6: the *beat* is the
    /// animal's measured stroke — the thorax resonator and the stretch-activated
    /// flight muscles that would generate it are not modelled (ASSUMPTIONS #37)
    /// — while everything the beat then does to the animal is the solver's,
    /// because what leaves this function is a wrench and nothing else.
    private func stepWings() {
        guard let aero = wingAero else { return }
        strokePhase += dt * aero.p.strokeHz
        strokePhase -= floor(strokePhase)
        let w = aero.instantaneous(phase: strokePhase,
                                  activationLeft: wingActivation.x,
                                  activationRight: wingActivation.y)
        wingWrench = w
        if thoraxIndex >= 0 {
            wingWrenchArray[thoraxIndex] = SVec6(
                Vec3(w.moment.x, w.moment.y, w.moment.z),
                Vec3(w.force.x, w.force.y, w.force.z))
            dynamics.externalWrench = wingWrenchArray
        }
        let s = sin(2 * Double.pi * strokePhase)
        for (j, activation) in [(wingYawJoints.0, wingActivation.x),
                                (wingYawJoints.1, wingActivation.y)] where j >= 0 {
            dynamics.hingeTargets[j] = aero.amplitudeDeg(activation: activation)
                * Double.pi / 180 * s
        }
    }

    /// The cord's wing pools, once per cord update: rates in, stroke amplitudes
    /// out. Nothing here decides what the pools do — that is the connectome's,
    /// with the haltere afferents landing on the wing steering motor neurons
    /// among its own synapses.
    private func updateWingActivation() {
        guard let aero = wingAero, let rates = wingRates?() else { return }
        let a = aero.activation(ratesHz: rates)
        wingActivation = SIMD2(a.left, a.right)
        if simulatedMS - lastWingReadoutMS >= 100 {
            lastWingReadoutMS = simulatedMS
            wingBeat = aero.beat(activationLeft: a.left, activationRight: a.right,
                                 samples: 180)
            liftOverWeight = wingBeat.flightForceDyn / max(aero.weightDyn, 1e-12)
        }
    }

    // -- what the HUD reads ------------------------------------------------

    /// Simulated time since start-up, ms.
    private(set) var simulatedMS: Double = 0
    private(set) var contacts = 0
    private(set) var feetDown = 0
    private(set) var centreHeight = 0.0
    private(set) var centreSpeed = 0.0
    private(set) var jointRate = 0.0
    /// Simulated seconds per wall second, measured over the last window.
    private(set) var realtime = 0.0

    private var steps = 0
    private var windowSteps = 0
    private var windowStart: CFTimeInterval = 0
    /// A running estimate of how many substeps this device manages per second,
    /// so a slow phone runs the same physics at the same step and simply
    /// takes longer than real time rather than taking bigger steps. The
    /// alternative — stretching dt to keep up — would change the contact
    /// stiffness the animal stands on, which is the one thing in this file
    /// that must not vary with the hardware.
    private var throughput = 4000.0

    init(asset: FlyBodyAsset) {
        self.asset = asset
        dynamics = FlyDynamics(asset: asset)
        dt = asset.timestepS
        posture = dynamics.holdExcitation()
        excitation = posture
    }

    /// Stand the animal up. The stance is a stiff mode, so the first twenty
    /// milliseconds are run at a tenth of the step; after that the model's own
    /// timestep holds it.
    func startUp() {
        dynamics.settle(ms: 20, dt: dt / 10, posture: posture)
        simulatedMS = 0
        publish()
    }

    /// Advance by up to `wallSeconds` of wall clock, never by more than the
    /// device can actually run.
    func advance(wallSeconds: Double) {
        guard wallSeconds > 0 else { return }
        let now = CACurrentMediaTime()
        if windowStart == 0 { windowStart = now }
        let affordable = throughput * wallSeconds * 0.9
        let want = wallSeconds / dt
        let count = max(1, Int(min(want, affordable).rounded(.down)))

        for _ in 0..<count {
            // --- the cord, once per millisecond of simulated time ---------
            if let drive, simulatedMS >= nextCordAtMS {
                nextCordAtMS = simulatedMS + cordIntervalMs
                cordOffset = drive(proprioception())
                cordUpdates += 1
                updateWingActivation()
            }
            var exc = posture
            for j in 0..<exc.count {
                if j < cordOffset.count {
                    exc[j] = min(1, max(-1, exc[j] + cordOffset[j]))
                }
            }
            excitation = exc
            stepWings()
            dynamics.step(dt: dt, torque: dynamics.muscleTorque(
                q: dynamics.q, qd: dynamics.qd, excitation: exc))
        }
        steps += count
        windowSteps += count
        simulatedMS += Double(count) * dt * 1000

        let elapsed = now - windowStart
        if elapsed > 0.5 {
            let rate = Double(windowSteps) / elapsed
            throughput = max(200.0, 0.5 * throughput + 0.5 * rate)
            realtime = rate * dt
            windowSteps = 0
            windowStart = now
        }
        publish()
    }

    /// The state, as the cord's sense organs would report it.
    func proprioception() -> FlyProprioception {
        var feet = 0
        var load: [String: Double] = [:]
        for (key, f) in dynamics.footForce {
            load[key] = f
            if f > 1e-6 { feet += 1 }
        }
        var rate = 0.0
        for r in dynamics.qd { rate = max(rate, abs(r)) }
        // rad/s world frame -> deg/s body frame: `quatToMat(rootQuat)` takes a
        // body-frame vector into the world, so its transpose takes the rate
        // back into the body.
        let R = quatToMat(dynamics.rootQuat)
        let w = dynamics.omega
        let body = R.transpose * w
        let deg = 180.0 / Double.pi
        return FlyProprioception(angle: dynamics.q, rate: dynamics.qd,
                                 legLoad: load,
                                 height: dynamics.centreOfMass().z,
                                 speed: dynamics.vel.length,
                                 feetDown: feet,
                                 angularRate: SIMD3(body.x, body.y, body.z) * deg)
    }

    /// World position and orientation of every mesh, in the asset's `visual`
    /// order — what the renderer poses.
    func visualWorld() -> (pos: [Vec3], rot: [Mat3]) {
        dynamics.visualWorld()
    }

    var visualName: [String] { dynamics.visualName }

    private func publish() {
        contacts = dynamics.contacts
        var feet = 0
        for (_, f) in dynamics.footForce where f > 1e-6 { feet += 1 }
        feetDown = feet
        centreHeight = dynamics.centreOfMass().z
        centreSpeed = dynamics.vel.length
        var rate = 0.0
        for r in dynamics.qd { rate = max(rate, abs(r)) }
        jointRate = rate
    }

    /// Substeps run so far, for the readout.
    var stepCount: Int { steps }
}
