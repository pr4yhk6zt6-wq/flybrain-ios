//
//  SimulationEngine.swift
//  Drives the LIF kernels. One `step()` is one millisecond of fly brain.
//

import Foundation
import Metal
import MetalKit
import simd

/// Must match `struct SimParams` in LIF.metal, field for field.
struct SimParams {
    var neuronCount: UInt32 = 0
    var edgeCount: UInt32 = 0
    var ringSlots: UInt32 = 0
    var step: UInt32 = 0

    var decayMembrane: Float = 0
    var decaySynaptic: Float = 0
    var vThreshold: Float = 1.0
    var vReset: Float = 0.0

    var vRest: Float = 0.0
    var gain: Float = 6.0
    var noiseSigma: Float = 0.015
    var externalDrive: Float = 1.5

    var refractorySteps: Int32 = 2
    var retinaCount: UInt32 = 0
    var flags: UInt32 = 0
    var dtMillis: Float = 1.0

    var contrastGain: Float = 8.0
    var adaptationRate: Float = 0.0033
    var cameraBias: Float = 0.5
    var pad1: Float = 0
}

/// Live numbers for the HUD.
struct SimulationStats {
    var spikesThisStep: Int = 0
    var spikesPerSecond: Double = 0
    var activeNeurons: Int = 0
    var populationRateHz: Double = 0
    var perSystemSpikes: [Int] = Array(repeating: 0, count: 10)
    var stepsPerFrame: Int = 0
    var simulatedMilliseconds: Int = 0
    /// Very rough, from Apple's published ~0.4 nJ per GPU ALU-op-equivalent.
    var estimatedMilliwatts: Double = 0
}

final class SimulationEngine {

    // Biophysics the user can change at runtime.
    var gain: Float = 6.0 {
        didSet { params.gain = gain }
    }
    var retinalDrive: Float = 1.5 {
        didSet { params.externalDrive = retinalDrive }
    }
    /// Simulation timestep in milliseconds. The brief asks for 0.1–5 ms.
    var timestepMillis: Float = 1.0 {
        didSet { recomputeDecays() }
    }
    var isPaused: Bool = false

    /// How hard camera contrast drives the photoreceptors.
    var contrastGain: Float = 8.0 {
        didSet { params.contrastGain = contrastGain }
    }

    private(set) var stats = SimulationStats()

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let connectome: Connectome

    // Pipelines
    private let psoIntegrate: MTLComputePipelineState
    private let psoScatter: MTLComputePipelineState
    private let psoDispatchArgs: MTLComputePipelineState
    private let psoRetina: MTLComputePipelineState
    private let psoActivity: MTLComputePipelineState
    private let psoStats: MTLComputePipelineState
    private let psoReset: MTLComputePipelineState
    private let psoGroupSpikes: MTLComputePipelineState
    private let psoDriveGroups: MTLComputePipelineState

    // State
    private let vMembrane: MTLBuffer
    private let iSynaptic: MTLBuffer
    private let refractory: MTLBuffer
    private let ring: MTLBuffer
    private let rngState: MTLBuffer
    private let externalInput: MTLBuffer
    private let adaptation: MTLBuffer
    private let spikeList: MTLBuffer
    private let spikeCount: MTLBuffer
    private let spikeFlags: MTLBuffer
    private let activity: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let dispatchArgs: MTLBuffer

    // Named body groups: (start, count) ranges into connectome.groupIndices,
    // a spike counter per group, and a drive value per group.
    private let groupRanges: MTLBuffer
    private let groupCounts: MTLBuffer
    private let groupDrives: MTLBuffer
    private(set) var groupNames: [String] = []
    private(set) var groupSizes: [Int] = []
    /// Smoothed firing rate per group, in Hz. Read by the body every frame.
    private(set) var groupRatesHz: [Float] = []
    private var groupIndexCount = 0

    private var params = SimParams()

    /// Longest synaptic delay in the dataset is 19 ms; the ring needs one more
    /// slot than that so the write for t+19 never lands on the slot being read.
    private let ringSlots = 24

    // Biophysical constants, matched to tools/verify_and_simulate.py
    private let tauMembrane: Float = 20.0   // ms
    private let tauSynaptic: Float = 5.0    // ms
    private let refractoryMillis: Float = 2.0

    private var lastStatsTime = CFAbsoluteTimeGetCurrent()
    private var spikeAccumulator = 0

    // MARK: - Setup

    init(device: MTLDevice, connectome: Connectome, library: MTLLibrary) throws {
        self.device = device
        self.connectome = connectome
        guard let q = device.makeCommandQueue() else {
            throw SimulationError.noCommandQueue
        }
        self.queue = q
        q.label = "FlyBrain.simulation"

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw SimulationError.missingKernel(name)
            }
            return try device.makeComputePipelineState(function: fn)
        }
        psoIntegrate    = try pipeline("lifIntegrate")
        psoScatter      = try pipeline("spikeScatter")
        psoDispatchArgs = try pipeline("buildDispatchArgs")
        psoRetina       = try pipeline("sampleRetina")
        psoActivity     = try pipeline("updateActivityTrace")
        psoStats        = try pipeline("reduceStats")
        psoReset        = try pipeline("resetCounters")
        psoGroupSpikes  = try pipeline("reduceGroupSpikes")
        psoDriveGroups  = try pipeline("driveGroups")

        let n = connectome.neuronCount

        func zeroed(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw SimulationError.allocationFailed(label)
            }
            memset(b.contents(), 0, bytes)
            b.label = label
            return b
        }

        vMembrane     = try zeroed(n * MemoryLayout<Float>.stride, "vMembrane")
        iSynaptic     = try zeroed(n * MemoryLayout<Float>.stride, "iSynaptic")
        refractory    = try zeroed(n * MemoryLayout<Int32>.stride, "refractory")
        ring          = try zeroed(ringSlots * n * MemoryLayout<Int32>.stride, "delayRing")
        rngState      = try zeroed(n * MemoryLayout<UInt32>.stride, "rngState")
        externalInput = try zeroed(n * MemoryLayout<Float>.stride, "externalInput")
        adaptation    = try zeroed(connectome.retinaCount * MemoryLayout<Float>.stride, "adaptation")
        spikeList     = try zeroed(n * MemoryLayout<UInt32>.stride, "spikeList")
        spikeCount    = try zeroed(MemoryLayout<UInt32>.stride, "spikeCount")
        spikeFlags    = try zeroed(n, "spikeFlags")
        activity      = try zeroed(n * MemoryLayout<UInt16>.stride, "activityTrace")
        statsBuffer   = try zeroed(16 * MemoryLayout<UInt32>.stride, "stats")
        dispatchArgs  = try zeroed(3 * MemoryLayout<UInt32>.stride, "dispatchArgs")

        // ---- named groups ---------------------------------------------
        let groups = connectome.metadata.groups
        let groupCount = max(groups.count, 1)
        groupRanges = try zeroed(groupCount * MemoryLayout<UInt32>.stride * 2, "groupRanges")
        groupCounts = try zeroed(groupCount * MemoryLayout<UInt32>.stride, "groupCounts")
        groupDrives = try zeroed(groupCount * MemoryLayout<Float>.stride, "groupDrives")

        groupNames = groups.map { $0.name }
        groupSizes = groups.map { $0.count }
        groupRatesHz = [Float](repeating: 0, count: groups.count)
        groupIndexCount = connectome.groupIndexCount
        do {
            let r = groupRanges.contents().assumingMemoryBound(to: UInt32.self)
            for (i, g) in groups.enumerated() {
                r[i * 2] = UInt32(g.start)
                r[i * 2 + 1] = UInt32(g.count)
            }
            // -1 means "this group is not being driven", which is the default.
            let d = groupDrives.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<groups.count { d[i] = -1 }
        }

        params.neuronCount = UInt32(n)
        params.edgeCount   = UInt32(connectome.edgeCount)
        params.ringSlots   = UInt32(ringSlots)
        params.retinaCount = UInt32(connectome.retinaCount)
        recomputeDecays()
        reset()
    }

    private func recomputeDecays() {
        let dt = max(0.1, min(5.0, timestepMillis))
        params.dtMillis        = dt
        params.decayMembrane   = exp(-dt / tauMembrane)
        params.decaySynaptic   = exp(-dt / tauSynaptic)
        params.refractorySteps = Int32(max(1, (refractoryMillis / dt).rounded()))
    }

    /// Randomise membrane potentials and clear all history. Same initial
    /// distribution as the NumPy reference, so the two can be compared directly.
    func reset() {
        let n = connectome.neuronCount
        let v = vMembrane.contents().assumingMemoryBound(to: Float.self)
        let r = rngState.contents().assumingMemoryBound(to: UInt32.self)
        var seed: UInt32 = 0x9E3779B9
        for i in 0..<n {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5
            v[i] = Float(seed % 1000) / 2000.0          // uniform [0, 0.5)
            r[i] = seed | 1                              // xorshift must not be 0
        }
        memset(iSynaptic.contents(), 0, n * MemoryLayout<Float>.stride)
        memset(refractory.contents(), 0, n * MemoryLayout<Int32>.stride)
        memset(ring.contents(), 0, ringSlots * n * MemoryLayout<Int32>.stride)
        memset(activity.contents(), 0, n * MemoryLayout<UInt16>.stride)
        memset(spikeFlags.contents(), 0, n)
        params.step = 0
        stats = SimulationStats()
    }

    // MARK: - Stepping

    /// Advance the brain by `steps` milliseconds, then update the render trace.
    /// All of it goes in one command buffer: at 1 ms per step and 60 fps we need
    /// ~16 steps per frame, and 16 separate command buffers per frame would spend
    /// more time in the driver than in the kernels.
    func step(count steps: Int, cameraTexture: MTLTexture?) {
        guard !isPaused, steps > 0 else { return }
        guard let cb = queue.makeCommandBuffer() else { return }
        cb.label = "FlyBrain.step x\(steps)"

        let n = connectome.neuronCount

        // --- body -> sensory populations, once per frame -------------------
        // The body writes a scalar drive per named group; this spreads it over
        // the group's members. Vision is excluded (drive -1) because the retina
        // kernel owns those neurons.
        if !groupNames.isEmpty, groupIndexCount > 0,
           let e = cb.makeComputeCommandEncoder() {
            e.label = "driveGroups"
            e.setComputePipelineState(psoDriveGroups)
            e.setBuffer(connectome.groupIndices, offset: 0, index: 0)
            e.setBuffer(groupRanges, offset: 0, index: 1)
            e.setBuffer(groupDrives, offset: 0, index: 2)
            e.setBuffer(externalInput, offset: 0, index: 3)
            var gi = UInt32(groupIndexCount)
            var gc = UInt32(groupNames.count)
            e.setBytes(&gi, length: MemoryLayout<UInt32>.stride, index: 4)
            e.setBytes(&gc, length: MemoryLayout<UInt32>.stride, index: 5)
            dispatch(e, psoDriveGroups, threads: groupIndexCount)
            e.endEncoding()
        }
        memset(groupCounts.contents(), 0,
               max(groupNames.count, 1) * MemoryLayout<UInt32>.stride)

        for _ in 0..<steps {
            // --- reset the per-step counters ------------------------------
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "reset"
                e.setComputePipelineState(psoReset)
                e.setBuffer(spikeCount, offset: 0, index: 0)
                e.setBuffer(statsBuffer, offset: 0, index: 1)
                e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
                e.endEncoding()
            }

            // --- camera -> photoreceptors ---------------------------------
            if let tex = cameraTexture, let e = cb.makeComputeCommandEncoder() {
                e.label = "retina"
                e.setComputePipelineState(psoRetina)
                e.setTexture(tex, index: 0)
                e.setBuffer(connectome.retinaIdx, offset: 0, index: 0)
                e.setBuffer(connectome.retinaUV, offset: 0, index: 1)
                e.setBuffer(externalInput, offset: 0, index: 2)
                e.setBuffer(adaptation, offset: 0, index: 3)
                setParams(e, index: 4)
                dispatch(e, psoRetina, threads: connectome.retinaCount)
                e.endEncoding()
            }

            // --- integrate -------------------------------------------------
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "integrate"
                e.setComputePipelineState(psoIntegrate)
                e.setBuffer(vMembrane, offset: 0, index: 0)
                e.setBuffer(iSynaptic, offset: 0, index: 1)
                e.setBuffer(refractory, offset: 0, index: 2)
                e.setBuffer(ring, offset: 0, index: 3)
                e.setBuffer(rngState, offset: 0, index: 4)
                e.setBuffer(externalInput, offset: 0, index: 5)
                e.setBuffer(spikeList, offset: 0, index: 6)
                e.setBuffer(spikeCount, offset: 0, index: 7)
                e.setBuffer(spikeFlags, offset: 0, index: 8)
                setParams(e, index: 9)
                dispatch(e, psoIntegrate, threads: n)
                e.endEncoding()
            }

            // --- build the indirect dispatch for the scatter ---------------
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "dispatchArgs"
                e.setComputePipelineState(psoDispatchArgs)
                e.setBuffer(spikeCount, offset: 0, index: 0)
                e.setBuffer(dispatchArgs, offset: 0, index: 1)
                e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                e.endEncoding()
            }

            // --- scatter spikes along the CSR ------------------------------
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "scatter"
                e.setComputePipelineState(psoScatter)
                e.setBuffer(spikeList, offset: 0, index: 0)
                e.setBuffer(spikeCount, offset: 0, index: 1)
                e.setBuffer(connectome.csrRowPtr, offset: 0, index: 2)
                e.setBuffer(connectome.csrColIdx, offset: 0, index: 3)
                e.setBuffer(connectome.csrWeight, offset: 0, index: 4)
                e.setBuffer(connectome.csrDelay, offset: 0, index: 5)
                e.setBuffer(ring, offset: 0, index: 6)
                setParams(e, index: 7)
                e.dispatchThreadgroups(indirectBuffer: dispatchArgs,
                                       indirectBufferOffset: 0,
                                       threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                e.endEncoding()
            }

            // --- HUD statistics --------------------------------------------
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "stats"
                e.setComputePipelineState(psoStats)
                e.setBuffer(spikeFlags, offset: 0, index: 0)
                e.setBuffer(connectome.neuronMeta, offset: 0, index: 1)
                e.setBuffer(statsBuffer, offset: 0, index: 2)
                setParams(e, index: 3)
                let tg = 256
                e.setThreadgroupMemoryLength(tg * MemoryLayout<UInt32>.stride, index: 0)
                e.dispatchThreadgroups(MTLSize(width: (n + tg - 1) / tg, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                e.endEncoding()
            }

            // --- motor groups -> firing rates ------------------------------
            // Accumulated across every step of the frame, so the body sees a
            // rate rather than one millisecond of noise.
            if !groupNames.isEmpty, groupIndexCount > 0,
               let e = cb.makeComputeCommandEncoder() {
                e.label = "groupSpikes"
                e.setComputePipelineState(psoGroupSpikes)
                e.setBuffer(connectome.groupIndices, offset: 0, index: 0)
                e.setBuffer(groupRanges, offset: 0, index: 1)
                e.setBuffer(spikeFlags, offset: 0, index: 2)
                e.setBuffer(groupCounts, offset: 0, index: 3)
                var gi = UInt32(groupIndexCount)
                var gc = UInt32(groupNames.count)
                e.setBytes(&gi, length: MemoryLayout<UInt32>.stride, index: 4)
                e.setBytes(&gc, length: MemoryLayout<UInt32>.stride, index: 5)
                dispatch(e, psoGroupSpikes, threads: groupIndexCount)
                e.endEncoding()
            }

            params.step &+= 1
        }

        // --- decay the render trace once per frame, not per step ------------
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "activity"
            e.setComputePipelineState(psoActivity)
            e.setBuffer(spikeFlags, offset: 0, index: 0)
            e.setBuffer(activity, offset: 0, index: 1)
            setParams(e, index: 2)
            dispatch(e, psoActivity, threads: n)
            e.endEncoding()
        }

        cb.addCompletedHandler { [weak self] _ in
            self?.harvestStats(steps: steps)
        }
        cb.commit()
    }

    private func setParams(_ encoder: MTLComputeCommandEncoder, index: Int) {
        var p = params
        encoder.setBytes(&p, length: MemoryLayout<SimParams>.stride, index: index)
    }

    private func dispatch(_ e: MTLComputeCommandEncoder,
                          _ pso: MTLComputePipelineState,
                          threads: Int) {
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        e.dispatchThreadgroups(MTLSize(width: (threads + w - 1) / w, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Set the drive on a named population. Pass nil to hand the group back to
    /// whatever else writes it (the retina kernel, for vision).
    func setGroupDrive(_ name: String, _ value: Float?) {
        guard let i = groupNames.firstIndex(of: name) else { return }
        let d = groupDrives.contents().assumingMemoryBound(to: Float.self)
        d[i] = value ?? -1
    }

    /// Smoothed firing rate of a named population, in Hz.
    func groupRate(_ name: String) -> Float {
        guard let i = groupNames.firstIndex(of: name), i < groupRatesHz.count else { return 0 }
        return groupRatesHz[i]
    }

    private func harvestGroupRates(steps: Int) {
        guard !groupNames.isEmpty else { return }
        let c = groupCounts.contents().assumingMemoryBound(to: UInt32.self)
        let windowMs = Float(steps) * params.dtMillis
        guard windowMs > 0 else { return }
        for i in 0..<groupNames.count {
            let size = Float(max(groupSizes[i], 1))
            let hz = Float(c[i]) / size * 1000.0 / windowMs
            // Light smoothing: the body should respond to a rate, not to the
            // frame-to-frame jitter of a few hundred neurons.
            groupRatesHz[i] += (hz - groupRatesHz[i]) * 0.35
        }
    }

    private func harvestStats(steps: Int) {
        harvestGroupRates(steps: steps)

        let s = statsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let fired = Int(s[0])
        spikeAccumulator += fired

        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - lastStatsTime
        if dt >= 0.25 {
            stats.spikesPerSecond = Double(spikeAccumulator) / dt
            spikeAccumulator = 0
            lastStatsTime = now
        }

        stats.spikesThisStep = fired
        stats.activeNeurons = fired
        stats.stepsPerFrame = steps
        stats.simulatedMilliseconds = Int(params.step)
        let n = Double(connectome.neuronCount)
        stats.populationRateHz = Double(fired) / n * 1000.0 / Double(params.dtMillis)
        for i in 0..<10 { stats.perSystemSpikes[i] = Int(s[1 + i]) }

        // Energy: each spike touches outDegree edges, each edge is one atomic
        // add plus three loads. ~2 nJ per edge-event on an A14-class GPU is the
        // order of magnitude Apple's own counters suggest.
        let edgeEvents = Double(fired) * (Double(connectome.edgeCount) / n)
        stats.estimatedMilliwatts = edgeEvents * 2e-9 * Double(steps) * 60.0 * 1000.0
    }

    // MARK: - Buffers the renderer needs

    var voltageBuffer: MTLBuffer { vMembrane }
    var activityBuffer: MTLBuffer { activity }
    var spikeFlagBuffer: MTLBuffer { spikeFlags }
    var spikeListBuffer: MTLBuffer { spikeList }
    var spikeCountBuffer: MTLBuffer { spikeCount }

    /// Inject a current pulse into one neuron — used by tap-to-stimulate.
    func stimulate(neuron index: Int, amplitude: Float = 3.0) {
        guard index >= 0 && index < connectome.neuronCount else { return }
        let e = externalInput.contents().assumingMemoryBound(to: Float.self)
        e[index] = amplitude
    }

    /// Mean drive currently sitting on the photoreceptors. Reading a handful of
    /// cells from the shared buffer is enough for a UI meter and costs nothing.
    func meanRetinalDrive() -> Float {
        let input = externalInput.contents().assumingMemoryBound(to: Float.self)
        let idx = connectome.retinaIdx.contents().assumingMemoryBound(to: UInt32.self)
        let n = connectome.retinaCount
        guard n > 0 else { return 0 }
        let stride = max(1, n / 256)
        var total: Float = 0
        var count = 0
        var i = 0
        while i < n {
            total += input[Int(idx[i])]
            count += 1
            i += stride
        }
        return count > 0 ? total / Float(count) : 0
    }

    /// Drive the whole retina with a flat value when there is no camera.
    func setFlatRetinalInput(_ value: Float) {
        let input = externalInput.contents().assumingMemoryBound(to: Float.self)
        let idx = connectome.retinaIdx.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<connectome.retinaCount {
            input[Int(idx[i])] = value
        }
    }
}

enum SimulationError: Error, LocalizedError {
    case noCommandQueue
    case missingKernel(String)
    case allocationFailed(String)

    var errorDescription: String? {
        switch self {
        case .noCommandQueue:          return "Could not create a Metal command queue"
        case .missingKernel(let k):    return "Kernel '\(k)' is missing from the Metal library"
        case .allocationFailed(let b): return "Could not allocate the '\(b)' buffer"
        }
    }
}
