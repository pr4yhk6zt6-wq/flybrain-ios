//
//  FlyCord.swift
//  The closed loop, on the phone.
//
//      sense organ ──> sensory neurons ──> the connectome ──> motor pools
//            ^                                                    │
//            │                                                    v
//        the joint <── muscles <── excitation <── the balance of the pools
//
//  Every arrow of that is a thing this file does, and none of them is a
//  behaviour. There is no gait here, no state machine, no "walk" command and no
//  schedule: the cord is handed what the body's organs would report, it asks the
//  running connectome how fast each motor pool is firing, and it turns those
//  two facts into the excitation of the muscles that move the joints. If the
//  animal does nothing, that is the result — docs/banc_simulation.json measures
//  that 79.7% of these cells never fire at the operating point, and a cord that
//  is mostly silent is what a connectome looks like when nobody is telling it
//  what to do.
//
//  It is the port of `tools/step3_closedloop.py` and `tools/step4_world.py`,
//  which is where the law comes from. Every constant below is either read out of
//  the body asset (the joint's own rest angle and range, the pool's own sign, as
//  measured on the assembled animal) or is one of the declared assumptions in
//  docs/ASSUMPTIONS.md, named in the comment next to it.
//
//  Nothing here talks to Metal. `FlyCordRateSource` is the whole of what it
//  needs from the connectome, which is what lets the loop be tested on its own —
//  see FlyCordTests.
//

import Foundation

// MARK: - What the cord needs from the connectome

/// The connectome, as far as the loop is concerned: how fast a named population
/// is firing, and a way to drive another one. `SimulationEngine` satisfies this
/// directly; a stub satisfies it in the tests.
protocol FlyCordRateSource: AnyObject {
    /// Is this group in the connectome the app loaded?
    ///
    /// The loop asks before it uses a name, because a group that is missing
    /// reads as 0 Hz, and 0 Hz is indistinguishable from a pool that is simply
    /// quiet — the one failure mode this project refuses to hide.
    func hasGroup(_ name: String) -> Bool
    /// Smoothed firing rate of a named population, Hz.
    func poolRate(_ name: String) -> Double
    /// Drive a named population: a current, in threshold units. Called once per
    /// cord update with what the organ is reporting.
    func setOrganDrive(_ name: String, _ value: Double)
}

// MARK: - The constants the loop has to choose

/// Everything here is a modelling choice, and every one of them is an entry in
/// docs/ASSUMPTIONS.md with the test that would catch it being wrong. The
/// bodies of the entries name the same numbers.
struct FlyCordSettings {
    /// The descending tone a standing fly's brain holds its cord at, in
    /// threshold units. Assumption #5: below 2.0 the motor pools are silent,
    /// above 3.0 they saturate, and 2.5 is where both antagonists of a joint are
    /// active together, as they are in a standing leg.
    var tone: Double = 2.5
    /// A real femoral chordotonal organ is range-fractionated: it spans its
    /// working range over part of the joint's anatomical range, not all of it.
    /// κ = 1 is the whole range. Assumption #11, swept at 1/4/16.
    var kappa: Double = 1.0
    /// +1: flexion stretches the organ, which silences it (published polarity,
    /// FeCO). Reversing it reverses the reflex — that is the control experiment
    /// in step 3, and it is why the sign is a setting rather than a comment.
    var polarity: Double = 1.0
    /// Fusion time of an insect skeletal muscle: a joint cannot follow
    /// individual spikes, so a pool's rate is filtered before it becomes an
    /// excitation. Assumption #7.
    var tauMuscleMs: Double = 60.0
    /// Excitation per unit of balance, in the app's [-1, 1] muscle units.
    ///
    /// This is the one number the Python experiments did not have to choose,
    /// because there the command was a joint *angle* and the span of the joint
    /// did the scaling. Here the actuator is a force pair, so the balance
    /// (which is dimensionless and in [-1, 1]) has to be turned into an
    /// excitation. Assumption #24: 0.5 means a pool pair that swings from all
    /// one side to all the other moves the excitation by its full range, and
    /// half of that at the midpoint. It is swept on the phone by the same
    /// argument the sweep would make: the closed-loop tests assert the command
    /// stays inside the muscle's range and that silence gives the stance back.
    var gain: Double = 0.5
    /// How long the balance is measured for before the loop is allowed to use
    /// it, in simulated milliseconds. Step 3 measured `b_ref` by running the
    /// cord at the stance with the organs clamped (assumption #10's method);
    /// this is the same measurement, done on the device, so the reference is the
    /// one *this* connectome and *this* body agree on.
    var calibrateMs: Double = 300.0
    /// Which organs are wired up. Both by default; a single-channel run is how
    /// step 3 showed which one does the work.
    var channels: Set<String> = ["chordotonal", "campaniform"]
}

// MARK: - The loop

final class FlyCord {

    /// Which phase the loop is in. `calibrating` is not a behaviour — it is the
    /// measurement of `b_ref`, and the organs are held at their standing value
    /// while it runs, exactly as step 3's `mode="clamped"` does.
    enum Phase: Equatable {
        case calibrating
        case running
    }

    /// A muscle pool on one leg: where its spikes are read, which joint it
    /// moves, and which way.
    struct Pool {
        let name: String        // "tibia flexor" — the animal's own vocabulary
        let group: String       // "pool:T1_left:tibia_flexor"
        let hinge: Int          // index into the body asset's hinge list
        let sign: Double        // +1 opens the joint's qpos, -1 closes it
        let neurons: Int        // how many motor neurons the pool has
    }

    /// One leg's two sense organs, as the body can report them.
    struct Organ {
        let kind: String        // "chordotonal" | "campaniform"
        let group: String       // "organ:T1_left:chordotonal"
        let leg: String         // "T1_left"
        let hinge: Int          // the knee the chordotonal organ spans
        let rest: Double        // its angle while standing
        let halfSpan: Double    // half its anatomical range
    }

    let settings: FlyCordSettings

    private(set) var pools: [Pool] = []
    private(set) var organs: [Organ] = []
    /// Hinges the pools land on — the only joints this loop commands. Every
    /// other joint keeps the muscle tone the animal was measured holding.
    private(set) var drivenHinges: [Int] = []
    /// The group names the connectome does not have, if any. Non-empty means
    /// part of the loop is not connected, and the HUD says so.
    private(set) var missingGroups: [String] = []

    private weak var source: FlyCordRateSource?

    // -- state ---------------------------------------------------------------

    /// Muscle activation per pool, Hz, filtered with τ — the muscle, not the
    /// spike train.
    private(set) var activation: [Double] = []
    /// The balance of each driven hinge: up / (up + down) over its pools.
    private(set) var balance: [Double] = []
    /// The balance the animal holds its stance at, measured in phase
    /// `calibrating`. Step 3's `b_ref`.
    private(set) var reference: [Double] = []
    private(set) var phase: Phase = .calibrating
    /// The excitation offsets handed out last, in hinge order — what the muscles
    /// are being told beyond the tone.
    private(set) var offset: [Double] = []
    /// What each organ is reporting, for the HUD and the tests.
    private(set) var organDrive: [String: Double] = [:]

    private var calibratingMs: Double = 0
    private var calibrationSamples: [Double] = []
    /// The load each leg carries while standing — the campaniform organ's own
    /// normalisation, measured on the body, not chosen.
    private var referenceLoad: [String: Double] = [:]

    /// Hinge count, from the asset: the offset array the body expects.
    let hingeCount: Int

    // MARK: Build

    init(asset: FlyBodyAsset, source: FlyCordRateSource,
         settings: FlyCordSettings = FlyCordSettings()) {
        self.settings = settings
        self.source = source

        let hinges = asset.joints.filter { $0.kind == "hinge" }
        self.hingeCount = hinges.count
        var hingeOf: [String: Int] = [:]
        for (i, j) in hinges.enumerated() { hingeOf[j.name] = i }

        // The pools, exactly as tools/build_body.py placed them with step 3's own
        // mapper: a pool the body could not place is not in the asset's pool
        // table at all, and is listed there under `pools_not_placed`.
        for (legKey, leg) in asset.legs {
            for (name, p) in leg.pools {
                // The asset names a pool's joint by the short key step 3's mapper
                // uses ("tibia") and the hinge by its full name
                // ("tibia_T1_left"): the pool table is read *through* the leg's
                // joint table, which is also what tools/verify_loop.py checks, so
                // a pool that cannot be resolved is one this build did not place.
                guard let full = leg.joints[p.joint]?.name,
                      let hinge = hingeOf[full] else { continue }
                pools.append(Pool(name: name, group: p.group, hinge: hinge,
                                  sign: p.qposSign >= 0 ? 1 : -1,
                                  neurons: 0))
            }
            for (kind, o) in leg.organs {
                guard let group = o.group else { continue }
                // The chordotonal organ spans the knee; it is the joint step 3
                // read the reflex out at, and the joint its drive is scaled by.
                guard let joint = o.joint ?? leg.joints["tibia"]?.name,
                      let hinge = hingeOf[joint],
                      let j = leg.joints["tibia"] else { continue }
                let span = j.range.count > 1 ? abs(j.range[1] - j.range[0]) : 0
                organs.append(Organ(kind: kind, group: group, leg: legKey,
                                    hinge: hinge, rest: j.rest,
                                    halfSpan: max(1e-6, span / 2)))
            }
        }
        pools.sort { $0.group < $1.group }
        organs.sort { $0.group < $1.group }

        drivenHinges = Array(Set(pools.map { $0.hinge })).sorted()
        activation = [Double](repeating: 0, count: pools.count)
        balance = [Double](repeating: 0, count: drivenHinges.count)
        reference = [Double](repeating: 0, count: drivenHinges.count)
        calibrationSamples = [Double](repeating: 0, count: drivenHinges.count)
        offset = [Double](repeating: 0, count: hingeCount)

        // Which of the names this loop depends on the connectome actually has.
        for name in Set(pools.map { $0.group } + organs.map { $0.group }).sorted() {
            if !source.hasGroup(name) { missingGroups.append(name) }
        }
    }

    /// How many pools the connectome answered for, for the HUD.
    var poolsPresent: Int {
        pools.filter { !missingGroups.contains($0.group) }.count
    }

    var isCalibrated: Bool { phase == .running }

    // MARK: Update — one millisecond of cord

    /// One millisecond: what the organs say, what the pools are doing, and the
    /// excitation that follows. Returns the offset per hinge, added by the body
    /// to its resting tone and clipped there.
    ///
    /// `dtMs` is the cord's own step. The body runs at 100 µs and asks this once
    /// per millisecond of simulated time, which is the connectome's own timestep.
    @discardableResult
    func update(dtMs: Double, proprio: FlyProprioception) -> [Double] {
        // --- the load a leg carries standing: the campaniform organ's scale ---
        if referenceLoad.isEmpty {
            for (leg, force) in proprio.legLoad where force > 0 {
                referenceLoad[leg] = force
            }
        }

        // --- body -> organ ---------------------------------------------------
        if phase == .running {
            for organ in organs where settings.channels.contains(organ.kind) {
                let value: Double
                switch organ.kind {
                case "chordotonal":
                    // x: how far the knee is from standing, over half its range.
                    // Published polarity: flexion stretches the organ, and a
                    // stretched chordotonal organ is silenced. Assumption #11.
                    let x = organ.halfSpan > 0
                        ? min(1, max(-1, (angle(proprio, organ.hinge) - organ.rest)
                                          / organ.halfSpan))
                        : 0
                    value = settings.tone
                          * (1 - settings.polarity * settings.kappa * x)
                case "campaniform":
                    // A strain gauge: this leg's share of the floor force, in
                    // units of the force it carries standing. There is no free
                    // gain in it. Assumption #12.
                    let load = proprio.legLoad[organ.leg] ?? 0
                    let ref = referenceLoad[organ.leg] ?? 0
                    value = ref > 1e-9 ? settings.tone * (load / ref) : settings.tone
                default:
                    value = settings.tone
                }
                let driven = max(0, value)
                organDrive[organ.group] = driven
                source?.setOrganDrive(organ.group, driven)
            }
        } else {
            // Calibrating: the organs are held at the value they have while
            // standing, so the balance that comes back is the stance's and
            // nothing else. Step 3 calls this "clamped".
            for organ in organs {
                organDrive[organ.group] = settings.tone
                source?.setOrganDrive(organ.group, settings.tone)
            }
        }

        // --- cord -> muscle --------------------------------------------------
        // A pool's rate, filtered as a muscle fuses it.
        let alpha = settings.tauMuscleMs > 0
            ? min(1, dtMs / settings.tauMuscleMs) : 1
        for (i, pool) in pools.enumerated() {
            let rate = source?.poolRate(pool.group) ?? 0
            activation[i] += (rate - activation[i]) * alpha
        }

        // The balance of each driven hinge: the ratio of the two antagonists'
        // activity, so how *large* the rates are drops out of the command
        // (assumption #8 — the synaptic scale N* is not in this loop).
        for (k, hinge) in drivenHinges.enumerated() {
            var up = 0.0, down = 0.0
            for (i, pool) in pools.enumerated() where pool.hinge == hinge {
                if pool.sign > 0 { up += activation[i] } else { down += activation[i] }
            }
            let total = up + down
            balance[k] = total > 1e-9 ? up / total : 0
        }

        // --- the reference, measured once ------------------------------------
        if phase == .calibrating {
            for k in 0..<drivenHinges.count { calibrationSamples[k] += balance[k] }
            calibratingMs += dtMs
            if calibratingMs >= settings.calibrateMs {
                for k in 0..<drivenHinges.count {
                    reference[k] = calibrationSamples[k] / max(1, calibratingMs / dtMs)
                }
                phase = .running
            }
        }

        // --- muscle excitation ------------------------------------------------
        for o in 0..<offset.count { offset[o] = 0 }
        for (k, hinge) in drivenHinges.enumerated() where hinge < offset.count {
            let command = settings.gain * (balance[k] - reference[k])
            offset[hinge] = min(1, max(-1, command))
        }
        return offset
    }

    private func angle(_ p: FlyProprioception, _ hinge: Int) -> Double {
        hinge < p.angle.count ? p.angle[hinge] : 0
    }

    // MARK: Readouts

    /// Mean activation of the pools, Hz — "how hard the cord is pushing".
    var meanPoolHz: Double {
        guard !activation.isEmpty else { return 0 }
        return activation.reduce(0, +) / Double(activation.count)
    }

    /// The pools that are firing at all.
    var activePools: Int { activation.filter { $0 >= 1 }.count }

    /// One line for the HUD: the state of the loop, in the animal's terms.
    var summary: String {
        switch phase {
        case .calibrating:
            return String(format: "measuring the stance · %d pools", pools.count)
        case .running:
            return String(format: "%d/%d pools firing · %.1f Hz · organs %@",
                          activePools, pools.count, meanPoolHz,
                          settings.channels.sorted().joined(separator: "+"))
        }
    }
}

// MARK: - The cord over the running connectome

/// `FlyCordRateSource` over `SimulationEngine`: the names go straight through,
/// because `tools/motor_pools.py` names them in both the asset and the binary.
final class SimulationRateSource: FlyCordRateSource {
    private let engine: SimulationEngine
    private let groups: Set<String>

    init(engine: SimulationEngine, groups: [String]) {
        self.engine = engine
        self.groups = Set(groups)
    }

    func hasGroup(_ name: String) -> Bool { groups.contains(name) }
    func poolRate(_ name: String) -> Double { Double(engine.groupRate(name)) }
    func setOrganDrive(_ name: String, _ value: Double) {
        engine.setGroupDrive(name, Float(value))
    }
}
