//
//  FlyCordTests.swift
//  The loop, tested for what the loop is supposed to do.
//
//  These tests do not need Metal, a GPU or the connectome: `FlyCord` takes its
//  rates from anything that answers `FlyCordRateSource`, and here that is a stub
//  that can be told that one pool is firing and the other is not. That is the
//  point of the protocol — a closed loop whose only testable path runs through a
//  GPU is a closed loop nobody can check.
//
//  What is asserted here is deliberately not "the fly walks". Nothing in this
//  repository walks yet. What is asserted is that the mapping is the animal's
//  own (every pool placed on the joint the asset says it moves), that the loop is
//  balanced around the stance it measured and reaches both directions from it,
//  that silence gives the stance back rather than a drift, that the organs report
//  the joint and the load they are supposed to, and that the published polarity
//  of the chordotonal organ is load-bearing — reversing it reverses the drive,
//  which is step 3's control experiment, reproduced on the phone's code path.
//

import XCTest
@testable import FlyBrain

/// A connectome that answers whatever the test tells it to, and records every
/// name it was asked for.
final class StubRates: FlyCordRateSource {
    var rates: [String: Double] = [:]
    /// What a name that is not in `rates` answers. 0 by default, so a test
    /// that lists the pools it cares about still sees silence everywhere else.
    var defaultRate: Double = 0
    var absent: Set<String> = []
    private(set) var asked: [String] = []
    private(set) var drives: [String: Double] = [:]

    func hasGroup(_ name: String) -> Bool { !absent.contains(name) }

    func poolRate(_ name: String) -> Double {
        asked.append(name)
        return rates[name] ?? defaultRate
    }

    func setGroupDrive(_ name: String, _ value: Double) {
        drives[name] = value
    }
}

final class FlyCordTests: XCTestCase {

    private func asset() throws -> FlyBodyAsset {
        let bundle = Bundle(for: FlyCordTests.self)
        func find(_ name: String) -> URL? {
            bundle.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json",
                                   subdirectory: "World")
        }
        guard let url = find("fly_body") else {
            throw XCTSkip("fly_body.json is not in the bundle — run "
                          + "tools/build_body.py and tools/pack_body.py")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FlyBodyAsset.self,
                                  from: try Data(contentsOf: url))
    }

    /// The hinge index the cord uses for a joint.
    ///
    /// `asset.joints` starts with the root's **free** joint (`kind == "free"`),
    /// so the index into that array is one more than the index into the hinge
    /// list the solver and `FlyCord` work in. The first version of this helper
    /// indexed `asset.joints` directly and every angle it wrote out landed on
    /// the neighbouring joint — which is exactly what
    /// `testTheChordotonalOrganReportsTheJointItSpans` reported (the organ's
    /// drive never moved). The trap is pinned by
    /// `testTheJointListStartsWithTheFreeJoint` below.
    private func hingeIndex(_ asset: FlyBodyAsset, named name: String) -> Int? {
        var k = 0
        for j in asset.joints {
            if j.kind != "hinge" { continue }
            if j.name == name { return k }
            k += 1
        }
        return nil
    }

    /// The state the animal is in while it stands: every joint at its standing
    /// angle, each leg carrying its own weight.
    private func stance(_ asset: FlyBodyAsset, hingeCount: Int,
                        kneeAngle: Double? = nil,
                        loads: [String: Double]? = nil,
                        rotate: SIMD3<Double> = .zero) -> FlyProprioception {
        var angle = [Double](repeating: 0, count: hingeCount)
        var load: [String: Double] = [:]
        for (key, leg) in asset.legs {
            load[key] = loads?[key] ?? 1.0
            if let t = leg.joints["tibia"],
               let i = hingeIndex(asset, named: t.name) {
                XCTAssertLessThan(i, hingeCount)
                angle[i] = kneeAngle ?? t.rest
            }
        }
        return FlyProprioception(angle: angle,
                                 rate: [Double](repeating: 0, count: hingeCount),
                                 legLoad: load, height: 0, speed: 0,
                                 feetDown: 6,
                                 angularRate: rotate)
    }

    /// The trap this file's helper fell into once: the asset's `joints` array
    /// is not the hinge list.
    func testTheJointListStartsWithTheFreeJoint() throws {
        let asset = try asset()
        XCTAssertEqual(asset.joints.first?.kind, "free",
                       "the asset lists the root's free joint first, so an index "
                       + "into asset.joints is not an index into the hinges")
        XCTAssertEqual(asset.joints.filter { $0.kind == "hinge" }.count, 102,
                       "the flybody model has 102 hinge joints")
        XCTAssertEqual(asset.joints.filter { $0.kind == "free" }.count, 1)
    }

    // MARK: - The mapping is the animal's

    func testEveryPoolIsPlacedOnTheJointTheAssetNames() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)

        // Six legs, seven placeable pools each (tools/build_body.py places them
        // with step 3's own mapper, and reports anything it could not place in
        // `pools_not_placed`).
        var expected = 0
        for (_, leg) in asset.legs { expected += leg.pools.count }
        XCTAssertEqual(cord.pools.count, expected)
        XCTAssertGreaterThan(cord.pools.count, 30,
                             "the asset carries far fewer pools than the six legs "
                             + "have — was it written by an older build_body.py?")
        XCTAssertEqual(cord.organs.count, asset.legs.count * 2,
                       "each leg reports two organs: the chordotonal organ and "
                       + "the campaniform sensilla")

        // Three joints per leg, and the loops must agree with the asset about
        // which hinge each pool pulls.
        XCTAssertEqual(cord.drivenHinges.count, asset.legs.count * 3,
                       "the loop should command exactly the three joints per leg "
                       + "the pulses can be placed on")
        for pool in cord.pools {
            XCTAssertLessThan(pool.hinge, cord.hingeCount)
            XCTAssertTrue(pool.group.hasPrefix("pool:"),
                          "\(pool.group) is not a pool group name")
            XCTAssertTrue(pool.sign == 1 || pool.sign == -1)
        }
        for organ in cord.organs {
            XCTAssertTrue(organ.group.hasPrefix("organ:"),
                          "\(organ.group) is not an organ group name")
            XCTAssertTrue(organ.halfSpan > 0,
                          "\(organ.group) has no range to report its joint over")
        }
    }

    func testTheConnectomeIsAskedForTheNamesTheAssetCarries() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount))

        // Every pool the loop drives was asked for by name, and no name outside
        // the asset's own was invented.
        let wanted = Set(cord.pools.map { $0.group })
        XCTAssertEqual(wanted.subtracting(Set(stub.asked)), [],
                       "the loop drives a pool it never asked the connectome for")
        XCTAssertEqual(cord.missingGroups, [],
                       "a group the connectome has reported as missing")
    }

    func testAMissingGroupIsReportedRatherThanReadAsSilence() throws {
        let asset = try asset()
        let stub = StubRates()
        let all = FlyCord(asset: asset, source: stub)
        stub.absent = [all.pools[0].group, all.organs[0].group]
        let cord = FlyCord(asset: asset, source: stub)

        XCTAssertEqual(cord.missingGroups, stub.absent.sorted(),
                       "a group the connectome does not have must be reported: "
                       + "it reads as 0 Hz, which is what a quiet pool reads as")
        XCTAssertEqual(cord.poolsPresent, cord.pools.count - 1)
    }

    /// The cord's tone is the brain's, and it arrives where the reference sends
    /// it: to the descending neurons. Until this test existed, the app put the
    /// tone on the sense organs and on nothing else, so nothing in the loop
    /// carried the brain's state into the cord at all — the animal stood on six
    /// disconnected legs (AUDIT item 9, delivery order #3).
    func testTheDescendingNeuronsCarryTheBrainsToneIntoTheCord() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        let p = stance(asset, hingeCount: cord.hingeCount)

        XCTAssertTrue(cord.descendingPresent,
                      "the shipped connectome carries the descending "
                      + "population by this name (build/flybanc_meta.json)")

        // The first millisecond, and every millisecond after it: the reference
        // injects the tone into the descending neurons on every step of every
        // mode, so a cord that is still measuring its own stance is not a cord
        // that has been left without a brain.
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(stub.drives[cord.settings.descendingGroup],
                       cord.settings.tone,
                       "the brain's tone is not on the descending neurons")
        XCTAssertEqual(cord.descendingDrive, cord.settings.tone)
        for _ in 0..<400 { _ = cord.update(dtMs: 1, proprio: p) }
        XCTAssertEqual(cord.phase, .running)
        XCTAssertEqual(stub.drives[cord.settings.descendingGroup],
                       cord.settings.tone,
                       "the descending drive was dropped once the loop started")

        // The organs report what they see, and at rest that is the same number:
        // assumption #10 re-uses the brain's tone so that this step adds no
        // magnitude of its own. It is a restatement of the brain's tone, not a
        // substitute for the descending drive above.
        // (`?? .nan` because the stub answers for any name, and a missing name
        // has to fail this assertion rather than not compile.)
        XCTAssertEqual(stub.drives[cord.organs[0].group] ?? .nan,
                       cord.settings.tone, accuracy: 1e-12)

        // The HUD's numbers have to be the numbers the connectome gave, not a
        // second opinion about them: the device's screenshot
        // (`uploads/IMG_2713.png`) reads `0/42 pools firing · 0.0 Hz` beside
        // `desc 2.5`, and telling a silent input from a bored animal needs both
        // halves measured rather than one of them assumed.
        let stub2 = StubRates()
        let cord2 = FlyCord(asset: asset, source: stub2)
        for pool in cord2.pools { stub2.rates[pool.group] = 7.5 }
        stub2.rates[cord2.settings.descendingGroup] = 12.5
        for organ in cord2.organs { stub2.rates[organ.group] = 3.5 }
        _ = cord2.update(dtMs: 1, proprio: p)
        XCTAssertEqual(cord2.descendingRateHz, 12.5,
                       "the descending rate is not read from the connectome")
        XCTAssertEqual(cord2.organRateHz, 3.5, accuracy: 1e-12,
                       "the organ rate is not read from the connectome")
        XCTAssertEqual(cord2.poolRateHz.count, cord2.pools.count)
        for (i, pool) in cord2.pools.enumerated() {
            XCTAssertEqual(cord2.poolRateHz[i], 7.5,
                           "\(pool.group) reports a rate the connectome did "
                           + "not give")
        }
        // ... and one update is one millisecond of a muscle that fuses over 60:
        // the *filtered* activation is a fraction of the rate, which is exactly
        // why a single frame of a quiet-looking HUD proves nothing on its own.
        XCTAssertLessThan(cord2.meanPoolHz, 1.0)
        let loudest = cord2.loudestPools(3).map { $0.1 }
        XCTAssertEqual(loudest, [7.5, 7.5, 7.5])

        // No descending population in the connectome is a fact the HUD has to
        // state, not a silence it reads as 0 Hz.
        let stub3 = StubRates()
        stub3.absent = [cord.settings.descendingGroup]
        let cord3 = FlyCord(asset: asset, source: stub3)
        XCTAssertFalse(cord3.descendingPresent)
        XCTAssertTrue(cord3.missingGroups.contains(cord.settings.descendingGroup))
        XCTAssertEqual(cord3.poolsPresent, cord3.pools.count,
                       "a missing descending group is not a missing pool")
    }

    // MARK: - The stance, and both directions from it

    func testTheLoopIsBalancedAroundTheStanceItMeasured() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        let p = stance(asset, hingeCount: cord.hingeCount)

        // The animal stands: both antagonists of every joint are active, in
        // equal measure. Every pool the loop drives gets the same rate.
        for pool in cord.pools { stub.rates[pool.group] = 20 }
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(cord.phase, .calibrating)
        for _ in 0..<400 { _ = cord.update(dtMs: 1, proprio: p) }   // 300 ms + slack

        XCTAssertEqual(cord.phase, .running,
                       "the loop never finished measuring the stance")
        // With every pool firing at the same rate, a joint's balance is the
        // share of its *pools* that open it — the loop sums the activation of
        // each side and takes the ratio (step 3's law, which is what its
        // measured reflex came through). A joint with two closing pools and one
        // opening pool therefore sits at 1/3, not at 1/2, and that is a
        // property of this animal's anatomy rather than a bug in the arithmetic.
        // The reason it costs nothing is the next assertion: the *stance's* own
        // balance is measured on the device and subtracted, so whatever the
        // asymmetry is, the command at the stance is zero.
        //
        // (Whether a pool should count by its motor neurons rather than as one
        // set is an open question, and it is one line here if it ever should:
        // docs/ASSUMPTIONS.md #21.)
        for (k, hinge) in cord.drivenHinges.enumerated() {
            let opens = cord.pools.filter { $0.hinge == hinge && $0.sign > 0 }.count
            let closes = cord.pools.filter { $0.hinge == hinge && $0.sign < 0 }.count
            let share = Double(opens) / Double(max(1, opens + closes))
            XCTAssertEqual(cord.balance[k], share, accuracy: 1e-9,
                           "joint \(hinge): \(opens) pools open it and "
                           + "\(closes) close it, so an equal drive has to "
                           + "balance at \(share), not at \(cord.balance[k])")
            XCTAssertEqual(cord.reference[k], share, accuracy: 1e-6,
                           "the reference is the stance's own balance")
        }
        // And therefore the command is the stance: nothing moves.
        for (k, hinge) in cord.drivenHinges.enumerated() {
            XCTAssertEqual(cord.offset[hinge], 0, accuracy: 1e-12,
                           "joint \(k) is being commanded away from a stance it "
                           + "is already holding")
        }
    }

    /// While the loop is measuring the stance, it must not be pushing the
    /// animal it is measuring.
    ///
    /// The first version of `FlyCord.update` computed `gain * (balance −
    /// reference)` in every phase, and `reference` is zero until the window
    /// closes — so for the first 300 ms the cord pushed every driven joint with
    /// up to `gain` of excitation, while step 3's `mode="clamped"` (the phase
    /// this one is copied from) commands every joint to the rest pose and
    /// measures the stance an animal is *holding*. The loop run headless with
    /// the body answering it (`tools/walk_loop.py`) is what made the difference
    /// visible: measured this way the animal lifted five of six feet and pressed
    /// itself 2.1× deeper into the floor; held this way it stands on six.
    func testTheCordHoldsTheStanceItIsMeasuring() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        let p = stance(asset, hingeCount: cord.hingeCount)

        // Every pool at 20 Hz: a joint with only an extensor on it balances at
        // 1.0, so `gain * (balance − 0)` is a large positive command if the
        // calibration window is not holding anything.
        for pool in cord.pools { stub.rates[pool.group] = 20 }
        XCTAssertEqual(cord.phase, .calibrating)
        for ms in 0..<299 {
            let offset = cord.update(dtMs: 1, proprio: p)
            XCTAssertEqual(cord.phase, .calibrating, "left the window early at \(ms) ms")
            XCTAssertTrue(offset.allSatisfy { $0 == 0 },
                          "the cord commanded the animal at \(ms) ms of a window that "
                          + "exists to measure the pose it is already holding")
        }
        // ...and one millisecond later it is running, and commanding again.
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(cord.phase, .running)
    }

    /// The campaniform organ's scale is the load a leg carries *standing*, and
    /// a standing leg's load is not one contact sample.
    func testTheStandingLoadIsTheWindowAndNotOneSample() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        for pool in cord.pools { stub.rates[pool.group] = 20 }

        // A leg that is between contacts for the first milliseconds — a spike of
        // 40, then the real standing load — must not end up with a scale of 40.
        var noisy = [String: Double]()
        for (key, _) in asset.legs { noisy[key] = 1.0 }
        for ms in 0..<320 {
            if ms == 0, let first = asset.legs.keys.first { noisy[first] = 40 }
            if ms == 1, let first = asset.legs.keys.first { noisy[first] = 1.0 }
            let offset = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount,
                                                              loads: noisy))
            XCTAssertEqual(offset.count, cord.hingeCount)
        }
        XCTAssertEqual(cord.phase, .running)
        // The organ's normalisation is measured, not chosen: whatever the scale
        // ends up as, the median of a window that was 1.0 for 319 of its 320
        // samples is 1.0.
        XCTAssertEqual(cord.referenceLoad.count, asset.legs.count,
                       "a leg's standing load was never measured")
        for (_, value) in cord.referenceLoad {
            XCTAssertEqual(value, 1.0, accuracy: 1e-9,
                           "the scale picked up a sample instead of the window")
        }
    }

    func testTheLoopReachesBothDirectionsFromTheStance() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        let p = stance(asset, hingeCount: cord.hingeCount)
        guard let tibia = cord.pools.first(where: { $0.name == "tibia extensor" }),
              let flexor = cord.pools.first(where: { $0.name == "tibia flexor" }),
              tibia.hinge == flexor.hinge, tibia.sign == 1, flexor.sign == -1 else {
            throw XCTSkip("the asset has no tibia antagonist pair on one joint")
        }
        for pool in cord.pools { stub.rates[pool.group] = 20 }
        for _ in 0..<500 { _ = cord.update(dtMs: 1, proprio: p) }
        XCTAssertEqual(cord.phase, .running)

        // The extensor alone: the joint's qpos opens, and the offset is positive.
        for pool in cord.pools { stub.rates[pool.group] = 0 }
        stub.rates[tibia.group] = 40
        for _ in 0..<3000 { _ = cord.update(dtMs: 1, proprio: p) }
        let opened = cord.offset[tibia.hinge]
        XCTAssertGreaterThan(opened, 0,
                             "the extensor pool firing must open the joint")
        XCTAssertLessThanOrEqual(opened, cord.settings.gain + 1e-9,
                                 "the command left the muscle's range")

        // The flexor alone: the other way, by the same amount. (The muscle
        // filter is a 60 ms one-pole, so the two directions are compared after
        // long enough for it to have converged, to 1e-3 of the gain.)
        for pool in cord.pools { stub.rates[pool.group] = 0 }
        stub.rates[flexor.group] = 40
        for _ in 0..<3000 { _ = cord.update(dtMs: 1, proprio: p) }
        let closed = cord.offset[tibia.hinge]
        XCTAssertLessThan(closed, 0,
                          "the flexor pool firing must close the joint")
        XCTAssertEqual(opened, -closed, accuracy: 1e-3,
                       "the loop should be symmetric about the stance it measured")
    }

    func testSilenceGivesTheStanceBackRatherThanADrift() throws {
        let asset = try asset()
        let stub = StubRates()                      // every pool silent
        let cord = FlyCord(asset: asset, source: stub)
        let p = stance(asset, hingeCount: cord.hingeCount)

        for _ in 0..<500 { _ = cord.update(dtMs: 1, proprio: p) }
        XCTAssertEqual(cord.phase, .running)
        for (k, hinge) in cord.drivenHinges.enumerated() {
            XCTAssertEqual(cord.reference[k], 0, accuracy: 1e-12)
            XCTAssertEqual(cord.offset[hinge], 0, accuracy: 1e-12,
                           "a silent cord must leave the animal holding its own "
                           + "tone, not command it somewhere")
            _ = k
        }
        for a in cord.activation {
            XCTAssertEqual(a, 0, accuracy: 1e-12)
        }
        XCTAssertEqual(cord.meanPoolHz, 0, accuracy: 1e-12)
        XCTAssertEqual(cord.activePools, 0)
    }

    // MARK: - The organs

    func testTheChordotonalOrganReportsTheJointItSpans() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        guard let organ = cord.organs.first(where: { $0.kind == "chordotonal" }),
              let leg = asset.legs[organ.leg],
              let tibia = leg.joints["tibia"] else {
            throw XCTSkip("the asset has no chordotonal organ on a knee")
        }
        for pool in cord.pools { stub.rates[pool.group] = 20 }
        // The organs are only driven once the loop is running. For the first
        // `calibrateMs` (300 ms, assumption #23) the cord holds every organ at
        // its standing value on purpose — that is the pose `b_ref` is measured
        // from, and step 3 calls it "clamped" — so a test that asks what a
        // joint angle does to the drive has to get past the calibration first.
        // Without this the assertion below reads 2.5, the clamped value, and
        // says nothing about the organ at all. The two sibling tests in this
        // file (polarity, campaniform) both warm the loop up for exactly this
        // reason; this one did not, and run 55 failed on it.
        for _ in 0..<500 {
            _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount))
        }
        XCTAssertEqual(cord.phase, .running)
        // At the standing angle the organ is at its tonic drive.
        _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount))
        XCTAssertEqual(stub.drives[organ.group] ?? -1, cord.settings.tone,
                       accuracy: 1e-9)
        // Flexion stretches it, and a stretched organ is silenced (polarity +1):
        // half its range of flexion is half of `halfSpan`, so at κ = 1 the
        // drive falls by half.
        let flexed = stance(asset, hingeCount: cord.hingeCount,
                            kneeAngle: tibia.rest + organ.halfSpan / 2)
        _ = cord.update(dtMs: 1, proprio: flexed)
        XCTAssertEqual(stub.drives[organ.group] ?? -1,
                       cord.settings.tone * 0.5, accuracy: 1e-6,
                       "the organ does not follow the joint it spans")
    }

    func testReversingThePublishedPolarityReversesTheOrgan() throws {
        let asset = try asset()
        let stub = StubRates()
        var settings = FlyCordSettings()
        settings.polarity = -1               // the control experiment
        let cord = FlyCord(asset: asset, source: stub, settings: settings)
        guard let organ = cord.organs.first(where: { $0.kind == "chordotonal" }),
              let tibia = asset.legs[organ.leg]?.joints["tibia"] else {
            throw XCTSkip("no chordotonal organ on a knee")
        }
        let flexed = stance(asset, hingeCount: cord.hingeCount,
                            kneeAngle: tibia.rest + organ.halfSpan / 2)
        // The organs are only driven once the loop is running.
        for _ in 0..<500 {
            _ = cord.update(dtMs: 1, proprio: stance(asset,
                                                     hingeCount: cord.hingeCount))
        }
        XCTAssertEqual(cord.phase, .running)
        _ = cord.update(dtMs: 1, proprio: flexed)
        XCTAssertEqual(stub.drives[organ.group] ?? -1,
                       cord.settings.tone * 1.5, accuracy: 1e-6,
                       "flipping the published polarity must change what the "
                       + "organ reports, or the sign is not reaching the loop")
    }

    func testTheCampaniformOrganReportsTheLoadTheLegCarries() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        guard let organ = cord.organs.first(where: { $0.kind == "campaniform" }),
              let chord = cord.organs.first(where: {
                  $0.leg == organ.leg && $0.kind == "chordotonal" }) else {
            throw XCTSkip("no campaniform organ in the asset")
        }
        // Standing: the leg carries its own weight, which is the organ's unit.
        var standing = [String: Double]()
        for (key, _) in asset.legs { standing[key] = 1.0 }
        var p = stance(asset, hingeCount: cord.hingeCount, loads: standing)
        for _ in 0..<500 { _ = cord.update(dtMs: 1, proprio: p) }
        XCTAssertEqual(stub.drives[organ.group] ?? -1, cord.settings.tone,
                       accuracy: 1e-6)
        XCTAssertEqual(stub.drives[chord.group] ?? -1, cord.settings.tone,
                       accuracy: 1e-6,
                       "an unloaded-standing knee must report its tonic drive")

        // Twice the weight on that leg: a strain gauge reads twice.
        standing[organ.leg] = 2.0
        p = stance(asset, hingeCount: cord.hingeCount, loads: standing)
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(stub.drives[organ.group] ?? -1, cord.settings.tone * 2,
                       accuracy: 1e-6,
                       "the load the leg carries is not reaching the organ")
        // And no load at all silences it.
        standing[organ.leg] = 0.0
        p = stance(asset, hingeCount: cord.hingeCount, loads: standing)
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(stub.drives[organ.group] ?? -1, 0, accuracy: 1e-9)
    }

    /// The cord has to own the thing it reads its rates from.
    ///
    /// This is the bug that made the phone show `0/42 pools · 0.0 Hz · desc
    /// 0.0 Hz (tone 2.5)` while the GPU's own tally on the same line said the
    /// pools were spiking (`uploads/IMG_2714.png`): `FlyCord.source` was
    /// `weak`, and the Body screen builds the source as a **local**
    ///
    ///     let source = SimulationRateSource(engine: sim, groups: …)
    ///     let c = FlyCord(asset: live.asset, source: source)
    ///
    /// so ARC released it as soon as `WorldView.attachCord` returned and every
    /// `source?…` in the loop became a no-op — no tone out to the descending
    /// cells, no organ drives, and 0 Hz for every rate in. Nothing in the test
    /// suite could see it, because a test holds its stub in a local that lives
    /// for the whole test *function*: the same code, a longer lifetime, the
    /// opposite result.
    ///
    /// So this test ends the lifetime the way the app does — the stub is built
    /// in a scope that closes — and then asks whether the cord is still
    /// connected. `WeakBox` is what does the asking: it holds the stub the way
    /// `FlyCord` used to.
    func testTheCordOwnsTheRateSourceItReadsFrom() throws {
        let asset = try asset()
        // Built in a function of its own, so the source's scope really does
        // end: this test must not hold the stub itself, or it would be testing
        // its own strong reference (the first version of this test did exactly
        // that, and passed against the weak source it was written to catch).
        let (cord, box) = cordBuiltTheWayTheAppBuildsOne(asset)
        let stub = box.value as? StubRates

        XCTAssertNotNil(stub,
                        "the cord let its rate source deallocate: every pool "
                        + "reads 0 Hz and nothing the body drives ever reaches "
                        + "the connectome (uploads/IMG_2714.png)")
        guard let stub else { return }

        // And with the caller's hands off it the loop is still live, in both
        // directions: the tone reaches the connectome, and a rate comes back
        // rather than being flattened to zero.
        let p = stance(asset, hingeCount: cord.hingeCount)
        _ = cord.update(dtMs: 1, proprio: p)
        XCTAssertEqual(stub.drives[cord.settings.descendingGroup],
                       cord.settings.tone,
                       "the brain's tone did not reach the connectome")
        XCTAssertEqual(cord.descendingRateHz, 24.0, accuracy: 1e-9,
                       "a group the connectome reports at 24 Hz read as 0 Hz")
        // Named, not counted: a count accepts a fourth group added without
        // anyone deciding the body should drive it (adding the antenna in item
        // 11 is what made this test fail, at 14 against 13 — which is the test
        // doing its job, so it now says *which* three things are driven instead
        // of how many).
        let driven = Set([cord.settings.descendingGroup, cord.settings.odorGroup,
                          cord.settings.haltereLeftGroup,
                          cord.settings.haltereRightGroup])
            .union(cord.organs.map { $0.group })
            // and the four wing motor pools (item 6): the flight tone, and the
            // differential the steering pools get. Named from the cord's own
            // settings rather than typed here, so renaming a pool in one place
            // fails this test instead of quietly driving nothing.
            .union(cord.settings.wingGroups)
        XCTAssertEqual(Set(stub.drives.keys), driven,
                       "the organs, the descending population, the antennae, the "
                       + "halteres and the four wing pools (tone + steering) are "
                       + "the whole of what this body drives")
    }

    /// The body's rotation has to *reach* the halteres, and the pair has to
    /// carry it as a left-right difference that reverses when the turn does.
    ///
    /// This is the claim item 7 makes at the loop's level — the measurement of
    /// what the LIF does with it is in `tools/haltere_gyro.py`, on the shipped
    /// connectome. Here the cord is driven by a stub, so what is being checked
    /// is the wiring: the rate the body reports is converted through the pair's
    /// own sensitivity axes (assumption #31) onto the two named populations,
    /// every simulated millisecond, with the afferents' lag.
    func testTheBodyRotationReachesTheHalteres() throws {
        let asset = try asset()
        let stub = StubRates()
        let cord = FlyCord(asset: asset, source: stub)
        let hinges = cord.hingeCount
        func drive(leftRight dps: SIMD3<Double>) -> Double {
            // 40 ms: long enough for the 5 ms afferent lag to have arrived, and
            // short enough that this is a measurement of the wiring rather than
            // of the solver
            for _ in 0..<40 {
                _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: hinges,
                                                         rotate: dps))
            }
            return (stub.drives[cord.settings.haltereLeftGroup] ?? .nan)
                 - (stub.drives[cord.settings.haltereRightGroup] ?? .nan)
        }
        let still = drive(leftRight: SIMD3(0, 0, 0))
        XCTAssertEqual(0.5 * (stub.drives[cord.settings.haltereLeftGroup]!
                              + stub.drives[cord.settings.haltereRightGroup]!),
                       3.0, accuracy: 1e-9,
                       "a still animal's halteres are not at their tonic drive")
        let left = drive(leftRight: SIMD3(0, 0, 200))
        let right = drive(leftRight: SIMD3(0, 0, -200))
        XCTAssertGreaterThan(left, 1.0,
                             "a 200 deg/s yaw to the left does not show up as a "
                             + "difference between the two halteres")
        XCTAssertLessThan(right, -1.0, "and the other way round must reverse it")
        XCTAssertEqual(still, 0, accuracy: 1e-12,
                       "a still body must give the two halteres the same drive, "
                       + "exactly: a difference at rest is a gyro reading a "
                       + "rotation that is not there")
    }

    /// Half a gyro is not a gyro. A build whose connectome has only one of the
    /// two haltere populations must say so and drive neither, rather than
    /// driving one and letting the other read as a pool that is simply quiet.
    func testAMissingHaltereIsReportedNotSilentlyHalved() throws {
        let asset = try asset()
        let stub = StubRates()
        stub.absent = ["sensory_haltere_right"]
        let cord = FlyCord(asset: asset, source: stub)
        XCTAssertFalse(cord.hasHaltereGroups)
        XCTAssertTrue(cord.missingGroups.contains("sensory_haltere_right"),
                      "the missing haltere is not in the missing-groups list, so "
                      + "nothing would tell the operator why the fly cannot feel "
                      + "its own rotation")
        _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount,
                                                rotate: SIMD3(0, 0, 300)))
        XCTAssertNil(stub.drives["sensory_haltere_left"],
                     "one haltere was driven while the other was missing")
    }
}

/// Builds a cord the way `WorldView.attachCord` builds one — the rate source is
/// a local, and the only thing that leaves this function is a cord and a
/// **weak** box. If the cord does not own the source, the source is gone by the
/// time this returns, which is what the phone did for the whole of IMG_2714.
private func cordBuiltTheWayTheAppBuildsOne(_ asset: FlyBodyAsset)
    -> (cord: FlyCord, box: WeakBox) {
    let source = StubRates()
    source.defaultRate = 24.0
    return (FlyCord(asset: asset, source: source), WeakBox(source))
}

/// Holds an object the way `FlyCord` used to hold its rate source.
private final class WeakBox {
    weak var value: AnyObject?
    init(_ value: AnyObject) { self.value = value }
}
