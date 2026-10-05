//
//  ReadoutTests.swift
//  The readout path, run for real, on a GPU.
//
//  The phone printed `0/42 pools firing · 0.0 Hz` while its cord was driving the
//  descending population every millisecond. `tools/pool_probe.py` shows the LIF
//  and the drive convention fire — with the device's own numbers, 23 of the
//  cord's 42 pools at >= 1 Hz, mean 12.60 Hz — but it is a NumPy reference and
//  it cannot reach the Metal kernels that sit between the spikes and the cord:
//  `reduceGroupSpikes` counts each named group's spikes, `harvestGroupRates`
//  divides those counts by the group's size, and only then does `FlyCord` see a
//  rate. Every one of those three is a place a zero can be born, and none of
//  them had ever been executed anywhere except the phone.
//
//  This test executes them, on the simulator's GPU, with the app's own
//  connectome and the cord's own operating point, and asks the two questions the
//  world HUD's readout line is built from:
//
//    * `net` — spikes per second over the whole connectome. Only the GPU writes
//      this one, so a zero is a GPU that ran nothing.
//    * `pools` — the group tallies restricted to the motor pools: the number the
//      HUD's `n/42` is made of, before the divide and the smoothing.
//
//  It is in `tools/check_tests.py`'s required list by name, so a skip here is a
//  failure in CI: if the simulator cannot give this test a Metal device, that is
//  a fact about the build and it should be said out loud, not stepped over.

import XCTest

#if canImport(Metal)
import Metal

@testable import FlyBrain

final class ReadoutTests: XCTestCase {

    /// Load the app's own binary and kernels, the way `BrainEngine.load` does.
    private func makeEngine(_ device: MTLDevice) throws -> SimulationEngine {
        let connectome: Connectome
        do {
            connectome = try Connectome(device: device)
        } catch {
            throw XCTSkip("flybanc.bin is not reachable from this bundle: \(error)")
        }
        var library: MTLLibrary?
        for bundle in [Bundle.main, Bundle(for: ReadoutTests.self)] {
            library = try? device.makeDefaultLibrary(bundle: bundle)
            if library != nil { break }
        }
        guard let library else {
            throw XCTSkip("no default.metallib in this bundle — there is nothing to run")
        }
        return try SimulationEngine(device: device, connectome: connectome,
                                    library: library)
    }

    /// The cord's operating point: the gain the app ships (`12`, the
    /// configuration `tools/pool_probe.py` calls "reference"), the flat retinal
    /// drive the world view sets while there is no camera, and the tone
    /// `FlyCord` puts on the descending population (assumption #5).
    func testTheGroupTalliesSeeThePoolsFiring() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device in this environment")
        }
        let sim = try makeEngine(device)
        XCTAssertGreaterThan(sim.groupNames.count, 100,
                             "the group names are the bin's own metadata: \(sim.readoutLine)")
        sim.setFlatRetinalInput(1.0)
        sim.setGroupDrive("descending", 2.5)

        var bestNet = 0.0
        var bestGroups = 0
        var bestPools = 0
        for _ in 0..<10 {
            sim.step(count: 8, leftEye: nil, rightEye: nil)
            // The harvest happens in the command buffer's completion handler on
            // another thread, so give the GPU its time instead of guessing a
            // duration. Ten frames at this network size is a few hundred
            // milliseconds on a simulator.
            Thread.sleep(forTimeInterval: 0.15)
            bestNet = max(bestNet, sim.stats.spikesPerSecond)
            bestGroups = max(bestGroups, sim.groupSpikeSum)
            bestPools = max(bestPools, sim.poolSpikeSum)
        }

        XCTAssertGreaterThan(bestNet, 0,
            "the connectome's own spike counter never moved, and only the GPU "
            + "writes it: the kernels did not run. \(sim.readoutLine)")
        XCTAssertGreaterThan(bestGroups, 0,
            "no named group saw a single spike in ten frames. \(sim.readoutLine)")
        XCTAssertGreaterThan(bestPools, 0,
            "the connectome fired but the motor pools are not among the cells "
            + "the group kernels count — which is the phone's `0/42` with the "
            + "silence above it. \(sim.readoutLine)")
    }
}
#else
// No Metal here: the kernels cannot run, so this is a skip rather than a
// quiet absence — `tools/run_tests_linux.sh` reads the test names out of these
// files and would otherwise generate a runner for a class that does not exist.
// CI is what runs the real thing (simulator), and `tools/check_tests.py` counts
// a skip in CI as a failure.
final class ReadoutTests: XCTestCase {
    func testTheGroupTalliesSeeThePoolsFiring() throws {
        throw XCTSkip("this platform has no Metal — the readout path is untested here")
    }
}
#endif
