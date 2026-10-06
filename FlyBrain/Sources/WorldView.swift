//
//  WorldView.swift
//  The 3-D world: the animal, and two floating views of it you can drag.
//
//  Reachable from the "Map" button at the top right of the brain view.
//
//  The main view is the animal, which you can orbit and pinch. The two
//  floating panes are cameras on the animal's own left and right, looking
//  back at it, in the same scene — so what you see in them is not a second
//  simulation, it is the same animal from somewhere else. Drag either pane
//  anywhere on screen.
//
//  There is no gait here, no schedule and no clock in the controller. The
//  screen advances `FlyLiveBody` — the solver in FlyDynamics, on this device,
//  with the floor under it and gravity through it — and poses the meshes from
//  the state that produces. Nothing here is a recording and nothing here
//  decides a step: the posture comes from the animal's own muscle tone, and
//  the behaviour, when it comes, will come from the nerve cord through
//  `FlyLiveBody.drive`.
//

import SwiftUI
import SceneKit
import QuartzCore

// MARK: - The three cameras

/// A rig that follows the animal, carrying one camera you steer and two
/// that watch its flanks.
final class WorldRig {
    let root = SCNNode()            // moved to the animal's centre
    let focus = SCNNode()           // sits at the rig origin, which
                                    // is where the animal is
    let main = SCNNode()
    let left = SCNNode()
    let right = SCNNode()

    /// Which way is up, in a world where the fly walks on the xy plane and
    /// z is the sky. SceneKit's look-at assumes y is up, so the orientation
    /// is worked out here instead.
    private static let up = (Float(0), Float(0), Float(1))

    init(mainFov: Double = 38, paneFov: Double = 50,
         paneDistance: Double = 0.306) {
        root.name = "rig"
        root.addChildNode(focus)

        // The fly faces +x; its left is +y and its right is −y. The two
        // flanking cameras sit a body length out and a little above, so the
        // whole animal fits the little pane — and how far out that is comes
        // from the asset (`tools/measure_view.py`), because the distance at
        // which the animal fits is a property of the animal and the lens and
        // not of this file. The panes are the one view in IMG_2715 that showed
        // a fly standing correctly assembled: they are the measurement that
        // says the pose and the meshes are fine and the main camera is not.
        let base = (0.03 * 0.03 + 0.30 * 0.30 + 0.05 * 0.05).squareRoot()
        let s = paneDistance / base
        camera(main, CGFloat(mainFov), SCNVector3(0.40, -0.30, 0.25))
        camera(left, CGFloat(paneFov),
               SCNVector3(Float(0.03 * s), Float(0.30 * s), Float(0.05 * s)))
        camera(right, CGFloat(paneFov),
               SCNVector3(Float(0.03 * s), Float(-0.30 * s), Float(0.05 * s)))
    }

    private func camera(_ n: SCNNode, _ fov: CGFloat, _ at: SCNVector3) {
        let c = SCNCamera()
        c.fieldOfView = fov
        c.zNear = 0.002          // the animal is millimetres long; the
        c.zFar = 6.0             // default far plane is a hundred metres
        n.camera = c
        n.position = at
        root.addChildNode(n)
        aim(n)
    }

    /// Point a camera back at the animal. Its −z axis is the way it looks.
    private func aim(_ n: SCNNode) {
        let p = (n.position.x, n.position.y, n.position.z)
        let len = sqrtf(p.0 * p.0 + p.1 * p.1 + p.2 * p.2)
        guard len > 1e-6 else { return }
        let z = (p.0 / len, p.1 / len, p.2 / len)        // away from target
        var x = (WorldRig.up.1 * z.2 - WorldRig.up.2 * z.1,
                 WorldRig.up.2 * z.0 - WorldRig.up.0 * z.2,
                 WorldRig.up.0 * z.1 - WorldRig.up.1 * z.0)
        var xl = sqrtf(x.0 * x.0 + x.1 * x.1 + x.2 * x.2)
        if xl < 1e-5 {                                   // looking down the
            x = (1, 0, 0)                                // up axis: pick one
            xl = 1
        }
        x = (x.0 / xl, x.1 / xl, x.2 / xl)
        let y = (z.1 * x.2 - z.2 * x.1,
                 z.2 * x.0 - z.0 * x.2,
                 z.0 * x.1 - z.1 * x.0)
        n.orientation = WorldRig.quaternion(x: x, y: y, z: z)
    }

    /// The rotation of a node whose local axes are `x`, `y`, `z` — the
    /// columns of its rotation matrix — as SceneKit's orientation.
    ///
    /// The conversion lives in `FlyDynamics.swift` (`quatFromMat`), beside the
    /// `quatToMat` it inverts, because it is testable there and because there
    /// used to be two copies: this one, and one in `FlyWorld.swift` that
    /// returned the conjugate and drew the animal in pieces
    /// (uploads/IMG_2713.png). One implementation, one test.
    private static func quaternion(
        x: (Float, Float, Float),
        y: (Float, Float, Float),
        z: (Float, Float, Float)) -> SCNVector4 {
        let m = Mat3(columns: (Vec3(Double(x.0), Double(x.1), Double(x.2)),
                               Vec3(Double(y.0), Double(y.1), Double(y.2)),
                               Vec3(Double(z.0), Double(z.1), Double(z.2))))
        let q = quatFromMat(m)
        return SCNVector4(Float(q.x), Float(q.y), Float(q.z), Float(q.w))
    }

    /// Where the camera you steer sits, in spherical coordinates about the
    /// animal.
    func set(azimuth: Double, elevation: Double, distance: Double,
             min minDistance: Double, max maxDistance: Double) {
        let d = max(minDistance, min(maxDistance, distance))
        let el = max(-1.45, min(1.45, elevation))
        main.position = SCNVector3(
            Float(d * cos(el) * cos(azimuth)),
            Float(d * cos(el) * sin(azimuth)),
            Float(d * sin(el)))
        aim(main)
    }
}

// MARK: - Playback

/// Owns the world and the clock that advances the animal in it.
///
/// Main-actor, because that is what it is: a `CADisplayLink` ticks it on the
/// main thread, it publishes to SwiftUI, and since step 6 it also talks to
/// `BrainEngine`, which is itself main-actor. The compiler is right to insist;
/// an actor-agnostic `WorldModel` would have been a claim that it is used off
/// the main thread, which it is not.
@MainActor
final class WorldModel: ObservableObject {
    /// Running, or paused mid-stance.
    @Published var running: Bool = true
    /// Bumped at 10 Hz so the HUD re-reads the solver's own counters without
    /// rebuilding itself sixty times a second.
    @Published private(set) var pulses: Int = 0

    @Published var azimuth: Double = WorldModel.homeAzimuth
    @Published var elevation: Double = WorldModel.homeElevation
    /// Where the camera starts, and how far it may go either way. All three
    /// come from the world asset (`FlyWorld.ViewLimits`, measured by
    /// `tools/measure_view.py`) and are assigned in `init`; the value here is
    /// only what the property is worth before that.
    @Published var distance: Double = 1.0
    private let limits: FlyWorld.ViewLimits
    /// Magnification while a pinch is in flight; committed into `distance`
    /// when the pinch ends. Not published: only the renderer reads it.
    var liveZoom: Double = 1.0

    // MARK: - The camera

    /// Which way round the camera starts looking, and how high, and where the
    /// reset button puts it back. The *distance* is not here: it is the
    /// asset's own measurement (`limits.home`), because it is the only one of
    /// the three that depends on how big the animal is.
    static let homeAzimuth   = 2.2
    static let homeElevation = 0.42

    /// Radians of orbit per point of drag, and the same for tilt.
    ///
    /// These used to be applied to `DragGesture.translation` **as a total** on
    /// every `onChanged` event, which is the bug `IMG_2714` reported as "the
    /// camera spins and will not be controlled": a drag delivers many events,
    /// and adding the whole translation again each time makes the rotation grow
    /// as the *square* of the event count — a 200-point drag over 100 events
    /// moved the camera by roughly 0.008 × 200 × 50 = 80 radians. `orbit(dx:dy:)`
    /// takes deltas; `WorldScreen` measures them against the previous event.
    /// At 0.0035 rad/pt a full-width drag on a phone is 1.2 rad — a third of a
    /// turn, which is what a hand expects.
    static let orbitRate = 0.0035
    static let tiltRate  = 0.0025

    // `minDistance`/`maxDistance`/`homeDistance` used to live here as 0.22,
    // 3.0 and 1.0. IMG_2715 is the screenshot of what 0.22 did: a 38° lens at
    // 0.22 cm sees a window 0.70 mm wide, the animal's body is 1.24 mm across,
    // and 29.7% of the body's vertices were inside the frame — so the screen
    // showed a thorax filling it with legs and wings crossing at angles, which
    // reads as "the model is in pieces" when the model is fine. They are
    // measured from the animal now (`tools/measure_view.py`, into
    // `world.json`, checked by `tools/audit_view.py --gate`), and this screen
    // reads them out of the asset rather than carrying its own copy.

    /// One camera event, in points of drag **since the last event**.
    func orbit(dx: Double, dy: Double) {
        azimuth -= dx * WorldModel.orbitRate
        // Keep it in (−π, π] so a long spin cannot lose precision.
        if azimuth > .pi { azimuth -= 2 * .pi }
        if azimuth < -.pi { azimuth += 2 * .pi }
        elevation = min(1.45, max(-1.45, elevation + dy * WorldModel.tiltRate))
    }

    /// Pinch, in flight and committed.
    func zoom(by factor: Double) {
        guard factor > 0.01 else { return }
        distance = min(limits.max, max(limits.min, distance / factor))
    }

    /// Put the camera back where it starts. The reset button on this screen.
    func resetView() {
        azimuth = WorldModel.homeAzimuth
        elevation = WorldModel.homeElevation
        distance = limits.home
        liveZoom = 1.0
    }

    let world: FlyWorld
    /// The animal. Not a recording of one: see FlyLiveBody.
    let live: FlyLiveBody
    let rig: WorldRig

    /// The brain the animal's cord runs on, if the app has loaded one. The Map
    /// screen is not drawing while this screen is up, so this screen has to
    /// advance the connectome itself (BrainEngine.pump).
    private let engine: BrainEngine?
    /// The loop: organs out, pools back in (FlyCord.swift).
    private(set) var cord: FlyCord?
    @Published private(set) var cordLine = "the cord is not attached"
    /// The line under `cordLine`: the loudest pools by the rate the connectome
    /// reported, and how many connectome milliseconds those rates were
    /// averaged over. A rate measured over one millisecond is a different
    /// animal from one measured over a hundred, and the number is on screen so
    /// that a silent cord and a mis-measured one can be told apart.
    @Published private(set) var cordDetail = ""
    /// Simulated milliseconds of cord per wall second, measured — 1.0 is real
    /// time. Reported, never assumed.
    @Published private(set) var cordRealtime: Double = 0

    /// How many 1 ms steps of connectome this screen asks for per frame. Started
    /// at 4 and moved by what the device actually manages, the same way the Map
    /// screen's renderer moves its own.
    private var cordStepsPerFrame = 4
    private var lastCordMS = 0

    private var link: CADisplayLink?
    private var last: CFTimeInterval = 0
    private var lastPublish: CFTimeInterval = 0

    init(world: FlyWorld, live: FlyLiveBody, engine: BrainEngine?) {
        self.world = world
        self.live = live
        self.engine = engine
        let v = world.viewLimits
        self.limits = v
        self.rig = WorldRig(mainFov: v.fovY, paneFov: v.paneFovY,
                            paneDistance: v.pane)
        self.distance = v.home
        world.scene.rootNode.addChildNode(rig.root)
        world.apply(live: live)
    }

    // No `deinit { stop() }`: a CADisplayLink retains its target, so a display
    // link that is still scheduled is a display link that keeps this object
    // alive — the deinit could never run while there was anything to stop. The
    // screen's `.onDisappear` calls `stop()`, and `stop()` invalidates the link,
    // which is the only thing that can actually end the loop.

    func start() {
        guard link == nil else { return }
        attachCord()
        last = 0
        let l = CADisplayLink(target: self, selector: #selector(tick(_:)))
        l.add(to: .main, forMode: .common)
        link = l
    }

    func stop() {
        link?.invalidate()
        link = nil
    }

    /// Build the loop and give it to the animal, once the connectome is up.
    ///
    /// From here on nothing on this screen decides anything: the body advances,
    /// asks the cord what its muscles should be doing, and the cord answers from
    /// the pools of the connectome that is running behind it. If the connectome
    /// is not loaded yet the animal keeps its tone and the HUD says so.
    private func attachCord() {
        guard cord == nil, let engine, engine.isReady,
              let sim = engine.simulation else {
            if engine == nil || engine?.isReady != true {
                cordLine = "no connectome loaded — muscle tone only"
            }
            return
        }
        let source = SimulationRateSource(engine: sim, groups: engine.groupNames)
        let c = FlyCord(asset: live.asset, source: source)
        cord = c
        live.cordIntervalMs = 1.0
        live.drive = { [weak c] p in c?.update(dtMs: 1.0, proprio: p) ?? [] }
        if c.missingGroups.isEmpty {
            cordLine = c.summary
        } else {
            // A name the connectome does not have would read as a silent pool,
            // which looks exactly like an animal doing nothing. Say so instead.
            cordLine = "\(c.missingGroups.count) of "
                     + "\(c.missingGroups.count + c.poolsPresent) groups missing"
        }
    }

    /// Advance the connectome by as much as this frame can afford, and measure
    /// what that turned out to be. The cord runs in the same 1 ms steps the
    /// connectome's own timestep is; there is no interpolation and no skipping.
    private func pumpCord(wallSeconds: Double) {
        guard let engine, engine.isReady else { return }
        // A frame may ask for at most the cord's nominal rate plus a fifth, so a
        // long frame (a rotation, a background app coming back) cannot turn into
        // a hundred-step catch-up that stalls the next one.
        let affordable = Int((wallSeconds * 1000.0 * 1.2).rounded(.up))
        engine.pump(count: min(cordStepsPerFrame, max(1, affordable)))

        // Adapt on what the connectome actually managed over the last window, so
        // a slow phone runs the cord slowly and says so rather than pretending.
        let now = CFAbsoluteTimeGetCurrent()
        if lastCordAdapt == 0 {
            lastCordAdapt = now
            lastCordPulse = engine.stats.simulatedMilliseconds
            return
        }
        let window = now - lastCordAdapt
        if window >= 0.5 {
            let advanced = Double(engine.stats.simulatedMilliseconds - lastCordPulse)
            cordRealtime = advanced / 1000.0 / window
            lastCordAdapt = now
            lastCordPulse = engine.stats.simulatedMilliseconds
            if cordRealtime < 0.85 && cordStepsPerFrame > 1 {
                cordStepsPerFrame -= 1
            } else if cordRealtime > 1.0 && cordStepsPerFrame < 24 {
                cordStepsPerFrame += 1
            }
        }
        if let c = cord {
            cordLine = c.summary
            let loudest = c.loudestPools(3)
                .map { String(format: "%@ %.1f", $0.0, $0.1) }
                .joined(separator: " · ")
            // The readout line carries the three numbers that separate the
            // ways the pools can read zero: the connectome's own spike rate
            // (a GPU-written counter), the size of the group machinery, and the
            // group tallies the per-pool rates are divided out of. A screenshot
            // of this line is comparable, number for number, with
            // `python3 tools/pool_probe.py --only device` on the same binary.
            // `engine` is this screen's BrainEngine, and it is not optional
            // here — the readout line lives on the SimulationEngine it owns.
            cordDetail = String(format: "over %d ms · %@ · loudest %@",
                                engine.stats.simulatedMilliseconds,
                                engine.simulation?.readoutLine ?? "no simulation",
                                loudest.isEmpty ? "—" : loudest)
        }
    }

    private var lastCordAdapt: CFTimeInterval = 0
    private var lastCordPulse = 0

    @objc private func tick(_ l: CADisplayLink) {
        let now = l.timestamp
        let dt = last == 0 ? 0.0 : min(0.1, now - last)
        last = now

        if running {
            // The body first: that writes this millisecond's organ drives, and
            // then the connectome is stepped so those drives are what it fires
            // on. The rates come back on the next frame's completion.
            live.advance(wallSeconds: dt)
            pumpCord(wallSeconds: dt)
        }
        render()

        // 60 Hz publishing would rebuild the whole HUD sixty times a second.
        if now - lastPublish > 0.1 {
            lastPublish = now
            pulses &+= 1
        }
    }

    /// Pose the animal and put the cameras where they belong. Public so the
    /// screen can show the first frame before the display link ticks.
    func render() {
        world.apply(live: live)
        rig.root.position = world.centre()
        rig.set(azimuth: azimuth,
                elevation: elevation,
                distance: distance / max(0.01, liveZoom),
                min: limits.min, max: limits.max)
    }

    /// Simulated seconds since the animal was stood up.
    var simulatedSeconds: Double { live.simulatedMS / 1000 }
}

// MARK: - Loading

/// Reads the three generated files off the main thread, then hands the
/// world to the screen.
struct WorldContainer: View {
    @State private var world: FlyWorld?
    /// The animal. Not called `body`: `View` already has one of those, and a
    /// stored property wins the redeclaration before the computed `body` ever
    /// gets a chance — which is a build failure, not a shadowing warning.
    @State private var animal: FlyLiveBody?
    @State private var message: String?
    @State private var started = false
    @State private var stage = "reading the model"
    let engine: BrainEngine?
    let onClose: () -> Void

    var body: some View {
        Group {
            if let w = world, let b = animal {
                WorldScreen(world: w, live: b, engine: engine, onClose: onClose)
            } else if let m = message {
                WorldUnavailableView(message: m, onClose: onClose)
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                    Text(stage)
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(red: 0.965, green: 0.957, blue: 0.949))
            }
        }
        .preferredColorScheme(.light)
        .task {
            guard !started else { return }
            started = true
            let r = await Task.detached(priority: .userInitiated) {
                () -> Result<(FlyWorld, FlyLiveBody), Error> in
                do {
                    let w = try FlyWorld()
                    await MainActor.run { stage = "standing the animal up" }
                    let asset = try FlyBodyAsset.loadFromBundle()
                    let b = FlyLiveBody(asset: asset)
                    b.startUp()
                    return .success((w, b))
                } catch { return .failure(error) }
            }.value
            switch r {
            case .success(let pair):
                world = pair.0
                animal = pair.1
            case .failure(let e):
                message = e.localizedDescription
            }
        }
    }
}

// MARK: - The screen

/// `PaneBand`, `BarHeight` and the height-reporting preference live in
/// LayoutBand.swift: the Map screen floats a window over the same two bars, and
/// the two screens clamp their floating windows the same way rather than each
/// keeping its own guess.
@MainActor
struct WorldScreen: View {
    @StateObject private var model: WorldModel
    /// The previous orbit event's translation, so the camera gets deltas.
    @State private var lastDrag = CGSize.zero
    @State private var leftOffset = CGSize(width: -104, height: 120)
    @State private var rightOffset = CGSize(width: 104, height: 120)
    @State private var showPanes = true
    /// Measured, not assumed: the bars report their own heights and the panes
    /// are confined to the gap between them.
    @State private var bars = BarHeight()
    let onClose: () -> Void

    /// The floating panes are one size, fixed here and used by everything that
    /// has to reason about where they fit.
    private static let paneSize = CGSize(width: 134, height: 106)
    private static let paneGap: CGFloat = 8

    init(world: FlyWorld, live: FlyLiveBody, engine: BrainEngine?,
         onClose: @escaping () -> Void) {
        _model = StateObject(wrappedValue: WorldModel(world: world, live: live,
                                                      engine: engine))
        self.onClose = onClose
    }

    /// The box a pane may occupy, given the screen and the two bars.
    ///
    /// The panes used to be free to go anywhere: they could be dragged onto
    /// each other, onto the readouts at the top, or under the transport at the
    /// bottom, and the result looked like a broken screen. Now each one is
    /// confined to its own half — the left pane's right edge stops at the
    /// middle and the right pane's left edge at the same line, so they cannot
    /// overlap each other — and both are confined vertically to the gap
    /// between the bars, which is measured from the bars themselves.
    private func band(_ size: CGSize) -> PaneBand {
        PaneBand.between(bars: bars, within: size,
                         pane: WorldScreen.paneSize,
                         gap: WorldScreen.paneGap,
                         split: true)
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // The animal, which you can orbit and pinch.
                WorldMainView(model: model)
                    .ignoresSafeArea()
                    .gesture(orbit)
                    .simultaneousGesture(pinch)

                VStack(spacing: 0) {
                    topBar
                        .padding(.horizontal, 14)
                        .padding(.top, 10)
                        .reportBarHeight(top: true)

                    Spacer(minLength: 8)

                    WorldTransport(model: model)
                        .padding(.horizontal, 14)
                        .padding(.bottom, 20)
                        .reportBarHeight(top: false)
                }

                // The two flanking cameras, floating over the scene — each
                // confined to its own half of the gap between the bars.
                if showPanes {
                    DraggablePane(offset: $leftOffset, title: "left eye",
                                  band: band(geo.size), side: -1) {
                        SideSceneView(scene: model.world.scene,
                                      pointOfView: model.rig.left)
                    }
                    DraggablePane(offset: $rightOffset, title: "right eye",
                                  band: band(geo.size), side: 1) {
                        SideSceneView(scene: model.world.scene,
                                      pointOfView: model.rig.right)
                    }
                }
            }
        }
        .onPreferenceChange(BarHeightKey.self) { bars = $0 }
        .onAppear { model.start() }
        .onDisappear { model.stop() }
    }

    /// The readouts on the left, the two buttons on the right. They are one
    /// row, so they cannot overlap each other; the readouts give way first when
    /// the screen is narrow.
    private var topBar: some View {
        HStack(alignment: .top, spacing: 8) {
            WorldHUD(model: model)
            Spacer(minLength: 8)
            VStack(spacing: 8) {
                roundButton("xmark") { onClose() }
                roundButton(showPanes ? "rectangle.on.rectangle" : "rectangle") {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                        showPanes.toggle()
                    }
                }
                // Back to the view that frames the whole animal. A phone
                // camera you can pinch into a 3 mm fly is a camera you can get
                // lost in — IMG_2714 is a screenshot from inside the thorax —
                // and a screen with no way back is a screen that looks broken.
                roundButton("arrow.counterclockwise") {
                    withAnimation(.easeInOut(duration: 0.25)) { model.resetView() }
                }
            }
            // Close, panes and reset are the only things here: they never
            // shrink and never sit under a readout.
            .fixedSize()
        }
    }

    private func roundButton(_ system: String,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.primary)
                .frame(width: 34, height: 34)
                .background(.ultraThinMaterial, in: Circle())
        }
    }

    /// Drag to orbit. `translation` is the total since the gesture began, so
    /// what gets applied here is the *difference* from the previous event —
    /// see `WorldModel.orbit(dx:dy:)` for what applying the total every time
    /// did to this screen.
    private var orbit: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { v in
                // `v.translation` is a CGSize of CGFloats; the difference and
                // the conversion are done in one place each, so no expression
                // here is mixing Double and CGFloat.
                let dx = Double(v.translation.width - lastDrag.width)
                let dy = Double(v.translation.height - lastDrag.height)
                lastDrag = v.translation
                model.orbit(dx: dx, dy: dy)
            }
            .onEnded { _ in lastDrag = .zero }
    }

    /// Pinch to zoom. The magnification is an absolute multiplier for the
    /// gesture, so it drives `liveZoom` while the fingers are down and is
    /// committed into `distance` once, on the way up.
    private var pinch: some Gesture {
        MagnificationGesture()
            .onChanged { v in model.liveZoom = Double(v) }
            .onEnded { v in
                model.zoom(by: Double(v))
                model.liveZoom = 1.0
            }
    }
}

// MARK: - Scene views

/// An SCNView on the shared scene. It renders on SceneKit's own clock, so it
/// shows whatever pose the model last wrote — there is nothing to keep in
/// step.
struct SideSceneView: UIViewRepresentable {
    let scene: SCNScene
    let pointOfView: SCNNode

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = scene
        v.pointOfView = pointOfView
        v.allowsCameraControl = false
        v.isUserInteractionEnabled = false
        v.preferredFramesPerSecond = 30
        v.antialiasingMode = .multisampling2X
        v.autoenablesDefaultLighting = false
        v.backgroundColor = UIColor(red: 0.965, green: 0.957, blue: 0.949,
                                    alpha: 1)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {
        if v.scene !== scene { v.scene = scene }
        if v.pointOfView !== pointOfView { v.pointOfView = pointOfView }
    }
}

struct WorldMainView: UIViewRepresentable {
    @ObservedObject var model: WorldModel

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = model.world.scene
        v.pointOfView = model.rig.main
        v.allowsCameraControl = false      // the gestures are ours, so the
                                           // camera can keep following the
                                           // animal while you steer it
        v.preferredFramesPerSecond = 60
        v.antialiasingMode = .multisampling4X
        v.autoenablesDefaultLighting = false
        v.backgroundColor = UIColor(red: 0.965, green: 0.957, blue: 0.949,
                                    alpha: 1)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {
        if v.scene !== model.world.scene { v.scene = model.world.scene }
        if v.pointOfView !== model.rig.main { v.pointOfView = model.rig.main }
    }
}

// MARK: - Floating panes

struct DraggablePane<Content: View>: View {
    @Binding var offset: CGSize
    let title: String
    /// Where this pane is allowed to be, and which side of the middle it
    /// belongs to.
    let band: PaneBand
    let side: Double
    let content: Content
    @State private var settled = CGSize.zero

    init(offset: Binding<CGSize>, title: String, band: PaneBand, side: Double,
         @ViewBuilder content: () -> Content) {
        _offset = offset
        self.title = title
        self.band = band
        self.side = side
        self.content = content()
    }

    private func clamp(_ o: CGSize) -> CGSize { band.clamp(o, side: side) }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 4) {
                Image(systemName: "video.fill")
                    .font(.system(size: 8))
                Text(title)
                    .font(.system(size: 9, weight: .semibold,
                                  design: .monospaced))
                Spacer()
                Image(systemName: "hand.draw")
                    .font(.system(size: 8))
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(.ultraThinMaterial)

            content
                .frame(minWidth: 0, maxWidth: .infinity,
                       minHeight: 0, maxHeight: .infinity)
        }
        .frame(width: 134, height: 106)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12)
            .stroke(Color.white.opacity(0.25), lineWidth: 1))
        .shadow(color: .black.opacity(0.3), radius: 10, y: 4)
        .offset(offset)
        .gesture(
            DragGesture()
                .onChanged { g in
                    offset = clamp(CGSize(width: settled.width
                                                   + g.translation.width,
                                          height: settled.height
                                                   + g.translation.height))
                }
                .onEnded { _ in settled = offset }
        )
        .onAppear {
            offset = clamp(offset)
            settled = offset
        }
        // The band changes when the bars do — a rotation, or a larger text
        // size — and a pane sitting where the band used to be would be sitting
        // on a readout. It goes back inside.
        .onChange(of: band) { _ in
            offset = clamp(offset)
            settled = offset
        }
    }
}

// MARK: - HUD and transport

struct WorldHUD: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            // The build number rides on this line because this screen is the
            // one being looked at when the question is "is this the new
            // build?" — see `BuildStamp` in ContentView.swift.
            Text("the fly, running on this phone · \(BuildStamp.short)")
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
            line(String(format: "feet %d/6 · COM z %+.4f cm · %d contacts",
                        model.live.feetDown, model.live.centreHeight,
                        model.live.contacts))
            line("\(model.world.meshCount) meshes · "
                 + "\(model.world.faceCount) triangles · 102 joints")
            // The loop, in one line: what the pools are doing, and how fast the
            // connectome that owns them is actually running on this device.
            Text(model.cordLine)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            // The per-pool readout: the loudest three, and the window the rates
            // came from. Never more than one line, so it cannot collide with
            // anything below it.
            Text(model.cordDetail)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            if model.cord != nil {
                Text(String(format: "cord %d updates · %.2f× real time",
                            model.live.cordUpdates, model.cordRealtime))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                Text("muscle tone only — no cord attached")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        // The readouts give way before they push the buttons, and never
        // overlap them: they are one row, and the text scales down inside it
        // (no layout priority — priority here would let the card push the
        // buttons off the edge on a narrow phone).
        .lineLimit(1)
        .minimumScaleFactor(0.72)
        .foregroundStyle(.primary)
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func line(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9))
            .foregroundStyle(.secondary)
            .lineLimit(1)
    }
}

struct WorldTransport: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(spacing: 7) {
            HStack(spacing: 12) {
                Button { model.running.toggle() } label: {
                    Image(systemName: model.running ? "pause.fill" : "play.fill")
                        .frame(width: 26, height: 26)
                }
                .buttonStyle(.bordered)

                Text(String(format: "%.1f s simulated", model.simulatedSeconds))
                    .font(.system(size: 11, design: .monospaced))
                    .lineLimit(1)

                Spacer(minLength: 4)

                Text(String(format: "%.2f× real time", model.live.realtime))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            HStack(spacing: 8) {
                Text(String(format: "%d steps · %.2f ms each",
                            model.live.stepCount, model.live.dt * 1000))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 4)
                Text("drag to orbit · pinch to zoom")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .minimumScaleFactor(0.8)
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - When the world has not been built yet

struct WorldUnavailableView: View {
    let message: String
    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "globe.desk")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No world yet").font(.headline)
            Text(message)
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 36)
            Button("Close", action: onClose).buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.965, green: 0.957, blue: 0.949))
        .padding()
    }
}
