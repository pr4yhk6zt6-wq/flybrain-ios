//
//  BrainEngine.swift
//  Owns the Metal device, the connectome, the simulation and the renderer, and
//  publishes the bits SwiftUI needs to observe.
//

import Foundation
import Metal
import MetalKit
import SwiftUI
import Combine

@MainActor
final class BrainEngine: ObservableObject {

    @Published private(set) var isReady = false
    @Published private(set) var loadError: String?
    @Published private(set) var stats = SimulationStats()
    @Published private(set) var fps: Double = 0
    @Published var selected: NeuronInfo?
    @Published private(set) var selectedOutDegree: Int = 0

    @Published var renderMode: RenderMode = .voltage {
        didSet { renderer?.renderMode = renderMode }
    }
    @Published var gain: Float = 12.0 {
        didSet { simulation?.gain = gain }
    }
    @Published var retinalDrive: Float = 1.5 {
        didSet { simulation?.retinalDrive = retinalDrive }
    }
    @Published var pointScale: Float = 2.5 {
        didSet { renderer?.pointScale = pointScale }
    }
    @Published var isPaused: Bool = false {
        didSet { simulation?.isPaused = isPaused }
    }
    @Published var cameraEnabled: Bool = false {
        didSet { cameraEnabled ? startCamera() : stopCamera() }
    }
    @Published var showCameraWindow: Bool = true
    /// Live mean drive sitting on the photoreceptors, for the preview meter.
    @Published private(set) var retinalActivity: Float = 0
    @Published private(set) var cameraFrames: Int = 0
    @Published private(set) var cameraLuminance: Float = 0

    /// Orbit the brain, or fly through it.
    @Published var cameraMode: CameraMode = .orbit {
        didSet { renderer?.camera.setMode(cameraMode) }
    }
    /// Both default to the corrected polarity; these are for the minority who
    /// want the old inverted feel back.
    @Published var invertLookX: Bool = false {
        didSet { renderer?.camera.invertX = invertLookX }
    }
    @Published var invertLookY: Bool = false {
        didSet { renderer?.camera.invertY = invertLookY }
    }

    let device: MTLDevice
    private(set) var renderer: Renderer?
    private var connectome: Connectome?
    private(set) var simulation: SimulationEngine?
    private(set) var library: MTLLibrary?
    private(set) var commandQueue: MTLCommandQueue?
    /// Created eagerly: the preview window needs the AVCaptureSession to exist
    /// before capture starts, otherwise the first frames land nowhere visible.
    private(set) lazy var camera: CameraFeed = CameraFeed(device: device)
    private var cameraCancellables = Set<AnyCancellable>()
    private weak var view: MTKView?

    private var systemMask: UInt32 = 0xFFFF_FFFF
    private var statsTimer: AnyCancellable?
    private var wasPausedByBackground = false

    var metadata: ConnectomeMetadata { connectome?.metadata ?? .fallback }

    /// How many retina cells this connectome actually has, so the eye window
    /// quotes the number in the data rather than a number from an older one.
    /// BANC v888: 1,372 left + 1,412 right.
    var retinaCells: Int { connectome?.retinaCount ?? 0 }

    init() {
        guard let d = MTLCreateSystemDefaultDevice() else {
            // The simulator before iOS 13 / an unsupported device.
            fatalError("This device has no Metal GPU.")
        }
        device = d
    }

    // MARK: - Loading

    func load() async {
        guard !isReady, loadError == nil else { return }
        do {
            let library = try device.makeDefaultLibrary(bundle: .main)
            self.library = library
            self.commandQueue = device.makeCommandQueue()
            let c = try Connectome(device: device)
            let sim = try SimulationEngine(device: device, connectome: c, library: library)
            let r = try Renderer(device: device, connectome: c, simulation: sim,
                                 library: library,
                                 pixelFormat: .bgra8Unorm,
                                 depthFormat: .depth32Float)
            sim.gain = gain
            sim.retinalDrive = retinalDrive
            // With no camera yet, give the retina a steady drive so the brain is
            // alive the moment the view appears.
            sim.setFlatRetinalInput(1.0)
            r.renderMode = renderMode
            r.pointScale = pointScale
            r.cameraTextureProvider = { [weak self] in
                guard let self, self.cameraEnabled else { return nil }
                return self.camera.latestTexture()
            }

            camera.$framesDelivered
                .receive(on: DispatchQueue.main)
                .sink { [weak self] in self?.cameraFrames = $0 }
                .store(in: &cameraCancellables)
            camera.$meanLuminance
                .receive(on: DispatchQueue.main)
                .sink { [weak self] in self?.cameraLuminance = $0 }
                .store(in: &cameraCancellables)

            connectome = c
            simulation = sim
            renderer = r
            if let v = view { v.delegate = r }

            statsTimer = Timer.publish(every: 0.2, on: .main, in: .common)
                .autoconnect()
                .sink { [weak self] _ in self?.pollStats() }

            isReady = true
        } catch {
            loadError = error.localizedDescription
        }
    }

    func attach(view: MTKView) {
        self.view = view
        if let r = renderer { view.delegate = r }
    }

    private func pollStats() {
        guard let sim = simulation, let r = renderer else { return }
        stats = sim.stats
        fps = r.fps
        retinalActivity = sim.meanRetinalDrive()
        // Keep the brain running at wall-clock-ish speed without dropping frames:
        // if we have headroom, simulate more milliseconds per frame.
        if r.fps > 55 && r.stepsPerFrame < 16 {
            r.stepsPerFrame += 1
        } else if r.fps < 40 && r.stepsPerFrame > 1 {
            r.stepsPerFrame -= 1
        }
    }

    // MARK: - Camera
    //
    // Deltas arrive already scaled from the gesture recognisers in
    // MetalViewBridge; the polarity lives in Camera.

    func look(dx: Float, dy: Float) { renderer?.camera.look(dx: dx, dy: dy) }
    func pan(dx: Float, dy: Float)  { renderer?.camera.pan(dx: dx, dy: dy) }
    func dolly(scale: Float)        { renderer?.camera.dolly(scale: scale) }

    func glide(amount: Float) {
        renderer?.camera.move(forwardAmount: amount)
    }

    func resetCamera() {
        renderer?.camera.reset()
        renderer?.camera.setMode(cameraMode)
    }

    func tap(at location: CGPoint) {
        guard let r = renderer, let c = connectome, let v = view else { return }
        let size = v.bounds.size
        if let index = r.pick(at: location, in: size) {
            selected = c.info(at: index)
            selectedOutDegree = c.outDegree(of: index)
            r.highlightIndex = index
        } else {
            selected = nil
            r.highlightIndex = nil
        }
    }

    // MARK: - Actions

    func resetSimulation() { simulation?.reset() }

    func stimulateSelected() {
        guard let i = selected?.index else { return }
        simulation?.stimulate(neuron: i, amplitude: 4.0)
    }

    func isSystemVisible(_ index: Int) -> Bool {
        (systemMask & (1 << UInt32(index))) != 0
    }

    func toggleSystem(_ index: Int) {
        systemMask ^= (1 << UInt32(index))
        renderer?.systemMask = systemMask
        objectWillChange.send()
    }

    // MARK: - Camera

    private func startCamera() {
        showCameraWindow = true
        camera.requestAccessAndStart { [weak self] granted in
            guard let self else { return }
            if !granted {
                self.cameraEnabled = false
                self.loadError = nil
            }
        }
    }

    private func stopCamera() {
        camera.stop()
        // Fall back to a flat drive so the brain does not simply go dark.
        simulation?.setFlatRetinalInput(1.0)
    }

    // MARK: - Lifecycle

    /// The brief asks for the simulation to stop in the background, and it has
    /// to: a GPU command buffer submitted after the app is suspended gets the
    /// process killed for "background GPU use".
    func setBackgrounded(_ backgrounded: Bool) {
        if backgrounded {
            wasPausedByBackground = !isPaused
            isPaused = true
            view?.isPaused = true
            camera.stop()
        } else if wasPausedByBackground {
            wasPausedByBackground = false
            isPaused = false
            view?.isPaused = false
            if cameraEnabled { startCamera() }
        }
    }
}
