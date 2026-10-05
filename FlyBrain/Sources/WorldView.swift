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
//  animal fell over and lay on its side for the last 26 ms of the
//  recording, and this plays that back as it happened.
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

    init() {
        root.name = "rig"
        root.addChildNode(focus)

        // The fly faces +x; its left is +y and its right is −y. The two
        // flanking cameras sit a body length out and a little above, so the
        // whole animal fits the little pane.
        camera(main, 38, SCNVector3(0.40, -0.30, 0.25))
        camera(left, 50, SCNVector3(0.03, 0.30, 0.05))
        camera(right, 50, SCNVector3(0.03, -0.30, 0.05))
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

    /// A rotation matrix's columns, as the quaternion that is the same
    /// rotation.
    private static func quaternion(
        x: (Float, Float, Float),
        y: (Float, Float, Float),
        z: (Float, Float, Float)) -> SCNVector4 {
        // columns x, y, z of the rotation; m_ab is row a, column b
        let m00 = x.0, m10 = x.1, m20 = x.2
        let m01 = y.0, m11 = y.1, m21 = y.2
        let m02 = z.0, m12 = z.1, m22 = z.2

        let tr = m00 + m11 + m22
        if tr > 0 {
            let s = 0.5 / sqrtf(tr + 1)
            return SCNVector4((m21 - m12) * s, (m02 - m20) * s,
                              (m10 - m01) * s, 0.25 / s)
        } else if m00 > m11 && m00 > m22 {
            let s = 2 * sqrtf(1 + m00 - m11 - m22)
            return SCNVector4(0.25 * s, (m01 + m10) / s, (m02 + m20) / s,
                              (m21 - m12) / s)
        } else if m11 > m22 {
            let s = 2 * sqrtf(1 + m11 - m00 - m22)
            return SCNVector4((m01 + m10) / s, 0.25 * s, (m12 + m21) / s,
                              (m02 - m20) / s)
        } else {
            let s = 2 * sqrtf(1 + m22 - m00 - m11)
            return SCNVector4((m02 + m20) / s, (m12 + m21) / s, 0.25 * s,
                              (m10 - m01) / s)
        }
    }

    /// Where the camera you steer sits, in spherical coordinates about the
    /// animal.
    func set(azimuth: Double, elevation: Double, distance: Double) {
        let d = max(0.05, min(3.0, distance))
        let el = max(-1.45, min(1.45, elevation))
        main.position = SCNVector3(
            Float(d * cos(el) * cos(azimuth)),
            Float(d * cos(el) * sin(azimuth)),
            Float(d * sin(el)))
        aim(main)
    }
}

// MARK: - Playback

/// Owns the recording and the clock that plays it.
final class WorldModel: ObservableObject {
    /// Running, or paused mid-stance.
    @Published var running: Bool = true
    /// Bumped at 10 Hz so the HUD re-reads the solver's own counters without
    /// rebuilding itself sixty times a second.
    @Published private(set) var pulses: Int = 0

    @Published var azimuth: Double = 2.2
    @Published var elevation: Double = 0.42
    @Published var distance: Double = 0.60
    /// Magnification while a pinch is in flight; committed into `distance`
    /// when the pinch ends. Not published: only the renderer reads it.
    var liveZoom: Double = 1.0

    let world: FlyWorld
    /// The animal. Not a recording of one: see FlyLiveBody.
    let live: FlyLiveBody
    let rig = WorldRig()

    private var link: CADisplayLink?
    private var last: CFTimeInterval = 0
    private var lastPublish: CFTimeInterval = 0

    init(world: FlyWorld, live: FlyLiveBody) {
        self.world = world
        self.live = live
        world.scene.rootNode.addChildNode(rig.root)
        world.apply(live: live)
    }

    deinit { stop() }

    func start() {
        guard link == nil else { return }
        last = 0
        let l = CADisplayLink(target: self, selector: #selector(tick(_:)))
        l.add(to: .main, forMode: .common)
        link = l
    }

    func stop() {
        link?.invalidate()
        link = nil
    }

    @objc private func tick(_ l: CADisplayLink) {
        let now = l.timestamp
        let dt = last == 0 ? 0.0 : min(0.1, now - last)
        last = now

        if running { live.advance(wallSeconds: dt) }
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
                distance: distance / max(0.05, liveZoom))
    }

    /// Simulated seconds since the animal was stood up.
    var simulatedSeconds: Double { live.simulatedMS / 1000 }
}

// MARK: - Loading

/// Reads the three generated files off the main thread, then hands the
/// world to the screen.
struct WorldContainer: View {
    @State private var world: FlyWorld?
    @State private var body: FlyLiveBody?
    @State private var message: String?
    @State private var started = false
    @State private var stage = "reading the model"
    let onClose: () -> Void

    var body: some View {
        Group {
            if let w = world, let b = body {
                WorldScreen(world: w, live: b, onClose: onClose)
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
                body = pair.1
            case .failure(let e):
                message = e.localizedDescription
            }
        }
    }
}

// MARK: - The screen

struct WorldScreen: View {
    @StateObject private var model: WorldModel
    @State private var leftOffset = CGSize(width: -110, height: -150)
    @State private var rightOffset = CGSize(width: 110, height: -150)
    @State private var showPanes = true
    let onClose: () -> Void

    init(world: FlyWorld, live: FlyLiveBody, onClose: @escaping () -> Void) {
        _model = StateObject(wrappedValue: WorldModel(world: world, live: live))
        self.onClose = onClose
    }

    /// What a floating pane is allowed to occupy.
    ///
    /// The panes used to be free: they could be dragged onto each other, onto
    /// the readouts at the top, or under the transport at the bottom, and the
    /// result looked like a broken screen. Now each one is confined to its own
    /// half — the left pane can never cross to the right of the middle, the
    /// right pane can never cross to the left of it — and neither can reach
    /// the 150 points at the top where the readouts and the buttons live, or
    /// the 160 at the bottom where the transport does.
    private func limit(_ size: CGSize, side: Double) -> (CGSize) -> CGSize {
        let halfWidth = max(70.0, size.width / 2 - 78)
        let top = -size.height / 2 + 150
        let bottom = size.height / 2 - 160
        return { o in
            let x = side < 0 ? min(-8.0, max(-halfWidth, o.width))
                             : max(8.0, min(halfWidth, o.width))
            let y = bottom > top ? min(bottom, max(top, o.height)) : o.height
            return CGSize(width: x, height: y)
        }
    }

    var body: some View {
        GeometryReader { geo in
        ZStack {
            // The animal, which you can orbit and pinch.
            WorldMainView(model: model)
                .ignoresSafeArea()
                .gesture(orbit)
                .simultaneousGesture(pinch)

            VStack {
                HStack(alignment: .top) {
                    WorldHUD(model: model)
                    Spacer()
                    VStack(spacing: 10) {
                        roundButton("xmark") { onClose() }
                        roundButton(showPanes ? "rectangle.on.rectangle"
                                              : "rectangle") {
                            withAnimation(.spring(response: 0.35,
                                                  dampingFraction: 0.8)) {
                                showPanes.toggle()
                            }
                        }
                    }
                }
                .padding(.horizontal, 14)
                .padding(.top, 8)

                Spacer(minLength: 8)

                WorldTransport(model: model)
                    .padding(.horizontal, 14)
                    .padding(.bottom, 24)
            }

            // The two flanking cameras, floating over everything — each
            // confined to its own half by `limit`.
            if showPanes {
                DraggablePane(offset: $leftOffset, title: "left eye",
                              limit: limit(geo.size, side: -1)) {
                    SideSceneView(scene: model.world.scene,
                                  pointOfView: model.rig.left)
                }
                DraggablePane(offset: $rightOffset, title: "right eye",
                              limit: limit(geo.size, side: 1)) {
                    SideSceneView(scene: model.world.scene,
                                  pointOfView: model.rig.right)
                }
            }
        }
        }
        .onAppear { model.start() }
        .onDisappear { model.stop() }
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

    private var orbit: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { v in
                model.azimuth -= Double(v.translation.width) * 0.008
                model.elevation = min(1.45, max(-1.45,
                    model.elevation + Double(v.translation.height) * 0.006))
            }
    }

    private var pinch: some Gesture {
        MagnificationGesture()
            .onChanged { v in model.liveZoom = Double(v) }
            .onEnded { v in
                model.distance = max(0.06, min(3.0,
                    model.distance / Double(v)))
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
    /// Where this pane is allowed to be. Nil means anywhere.
    let limit: ((CGSize) -> CGSize)?
    let content: Content
    @State private var settled = CGSize.zero

    init(offset: Binding<CGSize>, title: String,
         limit: ((CGSize) -> CGSize)? = nil,
         @ViewBuilder content: () -> Content) {
        _offset = offset
        self.title = title
        self.limit = limit
        self.content = content()
    }

    private func clamp(_ o: CGSize) -> CGSize { limit?(o) ?? o }

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
    }
}

// MARK: - HUD and transport

struct WorldHUD: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("the fly, running here")
                .font(.system(size: 12, weight: .semibold))
            Text("the real model · \(model.world.meshCount) meshes · "
                 + "\(model.world.faceCount) triangles")
                .font(.system(size: 9))
            Text("simulated on this device — 103 bodies, 102 joints, "
                 + "74 contact geoms")
                .font(.system(size: 9))
            Text(String(format: "feet down %d/6 · contacts %d · COM z %+.4f cm",
                        model.live.feetDown, model.live.contacts,
                        model.live.centreHeight))
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(String(format: "muscle tone: posture only (%d of 102 joints) "
                        + "· the cord is next",
                        model.live.posture.filter { abs($0) > 1e-9 }.count))
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct WorldTransport: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                Button { model.running.toggle() } label: {
                    Image(systemName: model.running ? "pause.fill" : "play.fill")
                        .frame(width: 26, height: 26)
                }
                .buttonStyle(.bordered)

                Text(String(format: "%.1f s simulated", model.simulatedSeconds))
                    .font(.system(size: 11, design: .monospaced))
                    .frame(minWidth: 104, alignment: .leading)

                Spacer(minLength: 4)

                Text(String(format: "%d steps · %.2f ms each",
                            model.live.stepCount, model.live.dt * 1000))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 6) {
                Text(String(format: "%.2f× real time on this device",
                            model.live.realtime))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 4)
                Text("drag to orbit · pinch to zoom")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - When the recording has not been made

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
