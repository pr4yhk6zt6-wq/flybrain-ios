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
    var absent: Set<String> = []
    private(set) var asked: [String] = []
    private(set) var drives: [String: Double] = [:]

    func hasGroup(_ name: String) -> Bool { !absent.contains(name) }

    func poolRate(_ name: String) -> Double {
        asked.append(name)
        return rates[name] ?? 0
    }

    func setOrganDrive(_ name: String, _ value: Double) {
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
                        loads: [String: Double]? = nil) -> FlyProprioception {
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
                                 feetDown: 6)
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
        // At the standing angle the organ is at its tonic drive.
        _ = cord.update(dtMs: 1, proprio: stance(asset, hingeCount: cord.hingeCount))
        XCTAssertEqual(stub.drives[organ.group] ?? -1, cord.settings.tone,
                       accuracy: 1e-9)
        // Flexion stretches it, and a stretched organ is silenced (polarity +1).
        // Half its range of flexion is a quarter of the joint's range, so at
        // κ = 1 the drive falls by half.
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
}
