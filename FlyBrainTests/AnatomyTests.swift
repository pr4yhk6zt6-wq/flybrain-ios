//
//  AnatomyTests.swift
//  FlyBrainTests
//
//  Step 1's premise, asserted on the device rather than in a report: the app
//  carries the real fly — 67 parts, 102 articulated degrees of freedom, an
//  animal of the right mass and the right size — and the numbers come out of
//  the shipped asset, not out of a constant written here.
//

import XCTest
@testable import FlyBrain

final class AnatomyTests: XCTestCase {

    /// The anatomy file ships inside the app bundle; `tools/step1_anatomy.py`
    /// writes it from the MJCF and CI copies it into `FlyBrain/Resources`.
    private func anatomy() throws -> [String: Any] {
        let candidates = [Bundle.main, Bundle(for: type(of: self))]
        for bundle in candidates {
            guard let url = bundle.url(forResource: "fly_anatomy", withExtension: "json"),
                  let data = try? Data(contentsOf: url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            return json
        }
        throw XCTSkip("fly_anatomy.json is not in this bundle — "
                      + "run tools/step1_anatomy.py and rebuild")
    }

    func testTheAnimalHasThePublishedAnatomy() throws {
        let a = try anatomy()
        let counts = try XCTUnwrap(a["counts"] as? [String: Any])

        XCTAssertEqual(counts["parts"] as? Int, 67,
                       "the flybody model has 67 body parts")
        XCTAssertEqual(counts["joints_articulated"] as? Int, 102,
                       "102 joints move, the 103rd is the free root")
        XCTAssertEqual(counts["dof_articulated"] as? Int, 102,
                       "this is the 102-DOF figure the Janelia work quotes")
        XCTAssertEqual(counts["actuators"] as? Int, 78)
    }

    func testTheAdhesionModelIsPresent() throws {
        let a = try anatomy()
        let actuators = try XCTUnwrap(a["actuators"] as? [[String: Any]])
        let adhesion = actuators.filter {
            ($0["name"] as? String)?.hasPrefix("adhere_") == true
        }
        // Two on the labrum, one on each of six claws. Their absence would mean
        // the animal cannot grip anything, which is how the original flight
        // model failed.
        XCTAssertEqual(adhesion.count, 8)
    }

    func testTheAnimalIsTheRightSizeAndMass() throws {
        let a = try anatomy()
        let massMilligrams = try XCTUnwrap(a["mass_total_kg"] as? Double) * 1e6
        XCTAssertTrue((0.8...1.2).contains(massMilligrams),
                      "a fruit fly weighs about 0.96 mg, got \(massMilligrams)")

        let size = try XCTUnwrap(a["size"] as? [String: Any])
        let length = try XCTUnwrap(size["body_length_mm"] as? Double)
        XCTAssertTrue((2.0...3.5).contains(length),
                      "a fruit fly is 2.5–3 mm long, got \(length) mm")
    }

    func testEveryArticulatedDegreeOfFreedomIsAnatomicallyLimitedOrFree() throws {
        let a = try anatomy()
        let joints = try XCTUnwrap(a["joints"] as? [[String: Any]])
        let hinges = joints.filter { ($0["kind"] as? String) == "hinge" }
        XCTAssertEqual(hinges.count, 102)
        for j in hinges {
            // Every hinge in this model is limited; a joint that can swing
            // freely is a modelling bug, not an insect.
            XCTAssertNotNil(j["range"], "joint \(j["name"] ?? "?") has no range")
        }
    }
}
