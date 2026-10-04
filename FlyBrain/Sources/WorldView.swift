//
//  WorldView.swift
//  World mode: the fly in its environment, third person, with two draggable
//  picture-in-picture windows — the brain, and what the fly's eyes see.
//

import SwiftUI
import MetalKit
import simd

// MARK: - The main world MTKView

struct WorldMetalView: UIViewRepresentable {
    let brain: BrainEngine
    let world: WorldEngine

    func makeCoordinator() -> Coordinator { Coordinator(world: world) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: brain.device)
        view.isOpaque = true
        view.backgroundColor = .black

        if let sim = brain.simulation, let lib = brain.library {
            try? world.attach(simulation: sim, library: lib, view: view)
        }

        let c = context.coordinator
        let orbit = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.handleOrbit(_:)))
        orbit.maximumNumberOfTouches = 1
        view.addGestureRecognizer(orbit)

        let pinch = UIPinchGestureRecognizer(target: c, action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pinch)

        let doubleTap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.handleDoubleTap(_:)))
        doubleTap.numberOfTapsRequired = 2
        view.addGestureRecognizer(doubleTap)

        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject {
        let world: WorldEngine
        private var last: CGPoint = .zero
        init(world: WorldEngine) { self.world = world }

        @objc func handleOrbit(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { last = .zero }
            let dx = Float(t.x - last.x), dy = Float(t.y - last.y)
            last = t
            world.orbit(dx: dx, dy: dy)
        }

        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            guard g.state == .changed else { g.scale = 1; return }
            let s = Float(g.scale)
            g.scale = 1
            world.zoom(s)
        }

        @objc func handleDoubleTap(_ g: UITapGestureRecognizer) {
            world.recentre()
        }
    }
}

// MARK: - The fly's-eye PiP

/// Draws the two ommatidially-filtered eye textures side by side — left eye
/// in the left half, right eye in the right half — with a facet lattice over
/// each, so the preview shows the fly's pair of compound eyes rather than a
/// single cyclopean view.
final class EyePreviewRenderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private var pipeline: MTLRenderPipelineState?
    var sourceLeft: (() -> MTLTexture?)?
    var sourceRight: (() -> MTLTexture?)?
    var facets: Float = 22

    init?(device: MTLDevice, library: MTLLibrary, format: MTLPixelFormat) {
        guard let q = device.makeCommandQueue() else { return nil }
        queue = q
        super.init()
        let d = MTLRenderPipelineDescriptor()
        d.label = "eyePreview"
        d.vertexFunction = library.makeFunction(name: "blitVertex")
        d.fragmentFunction = library.makeFunction(name: "eyePairFragment")
        d.colorAttachments[0].pixelFormat = format
        pipeline = try? device.makeRenderPipelineState(descriptor: d)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pipeline,
              let texL = sourceLeft?(),
              let texR = sourceRight?(),
              let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(),
              let e = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        e.setRenderPipelineState(pipeline)
        e.setFragmentTexture(texL, index: 0)
        e.setFragmentTexture(texR, index: 1)
        var f = facets
        e.setFragmentBytes(&f, length: MemoryLayout<Float>.stride, index: 0)
        e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        e.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

struct EyeMetalView: UIViewRepresentable {
    let brain: BrainEngine
    let world: WorldEngine

    func makeCoordinator() -> EyePreviewRenderer? {
        guard let lib = brain.library else { return nil }
        return EyePreviewRenderer(device: brain.device, library: lib, format: .bgra8Unorm)
    }

    func makeUIView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: brain.device)
        v.colorPixelFormat = .bgra8Unorm
        v.preferredFramesPerSecond = 30
        v.isOpaque = true
        v.backgroundColor = .black
        let w = world
        context.coordinator?.sourceLeft = { [weak w] in w?.eyeTextureL }
        context.coordinator?.sourceRight = { [weak w] in w?.eyeTextureR }
        v.delegate = context.coordinator
        return v
    }

    func updateUIView(_ view: MTKView, context: Context) {}
}

// MARK: - A generic draggable PiP frame

struct DraggablePiP<Content: View>: View {
    let title: String
    let onClose: () -> Void
    @ViewBuilder let content: () -> Content

    @AppStorage private var storedX: Double
    @AppStorage private var storedY: Double
    @State private var dragOffset: CGSize = .zero
    @State private var expanded = true

    /// Size of the open window; the eye pair is 2:1 so it gets a wider frame
    /// than the brain view.
    var expandedSize: CGSize = CGSize(width: 136, height: 152)

    init(title: String, storageKey: String,
         defaultX: Double = 0, defaultY: Double = 0,
         expandedSize: CGSize = CGSize(width: 136, height: 152),
         onClose: @escaping () -> Void,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.onClose = onClose
        self.content = content
        self.expandedSize = expandedSize
        _storedX = AppStorage(wrappedValue: defaultX, "pip_\(storageKey)_x")
        _storedY = AppStorage(wrappedValue: defaultY, "pip_\(storageKey)_y")
    }

    private var size: CGSize {
        expanded ? expandedSize : CGSize(width: 108, height: 20)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 9, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.85))
                Spacer(minLength: 0)
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) { expanded.toggle() }
                } label: {
                    Image(systemName: expanded ? "chevron.down" : "chevron.up")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(.white.opacity(0.7))
                }
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            .padding(.horizontal, 7)
            .frame(height: 20)
            .background(Color.black.opacity(0.65))

            if expanded {
                content()
                    .frame(width: size.width, height: size.height - 20)
                    .clipped()
            }
        }
        .frame(width: size.width)
        .background(Color.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.5), radius: 8, y: 3)
        .offset(x: storedX + dragOffset.width, y: storedY + dragOffset.height)
        .gesture(
            DragGesture()
                .onChanged { dragOffset = $0.translation }
                .onEnded { v in
                    storedX += v.translation.width
                    storedY += v.translation.height
                    dragOffset = .zero
                }
        )
    }
}

// MARK: - World mode

struct WorldView: View {
    @ObservedObject var brain: BrainEngine
    @ObservedObject var world: WorldEngine

    var body: some View {
        ZStack(alignment: .topLeading) {
            WorldMetalView(brain: brain, world: world)
                .ignoresSafeArea()

            worldHUD
                .padding(.horizontal, 14)
                .padding(.top, 100)

            if world.showBrainPiP {
                DraggablePiP(title: "BRAIN", storageKey: "brain",
                             defaultX: 14, defaultY: 320,
                             onClose: { world.showBrainPiP = false }) {
                    BrainMetalView(engine: brain)
                }
            }

            if world.showEyePiP {
                DraggablePiP(title: "FLY EYES", storageKey: "eye",
                             defaultX: 170, defaultY: 320,
                             expandedSize: CGSize(width: 178, height: 110),
                             onClose: { world.showEyePiP = false }) {
                    EyeMetalView(brain: brain, world: world)
                }
            }

            VStack {
                Spacer()
                worldControls
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
            }
        }
    }

    // MARK: HUD

    private var worldHUD: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 11) {
                stat("FPS", String(format: "%.0f", world.fps))
                stat(world.pose.airborne > 0.5 ? "FLIGHT"
                         : (world.habit == 2 ? "GROOM"
                         : world.habit == 1 ? "STOP" : "WALK"),
                     String(format: "%.0f mm/s", world.speed * 10))
                // Measured quasi-steady lift against measured body weight.
                stat("LIFT/W", String(format: "%.2f",
                                      world.weightUN > 0 ? world.liftUN / world.weightUN : 0))
                stat("STROKE", String(format: "%.0f°", world.strokeAmplitudeDeg))
                stat("YAW", String(format: "%.0f°/s", world.yawRateDegPerSec))
                stat("ODOUR", String(format: "%.2f", world.odourStrength))
            }
            // The motor channels actually steering the animal.
            HStack(spacing: 8) {
                bar("WING L", world.drives.wingPowerL, 160, .cyan)
                bar("WING R", world.drives.wingPowerR, 160, .cyan)
                bar("LEG L", world.drives.legL, 60, .orange)
                bar("LEG R", world.drives.legR, 60, .orange)
                bar("JUMP", world.drives.jump, 20, .red)
            }
            // If the anatomical mesh did not load, say so on screen. It used
            // to fail silently, and the symptom was just "no fly".
            if let note = world.flyMeshNote {
                Text(note)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(Color.orange.opacity(0.18),
                                in: RoundedRectangle(cornerRadius: 4))
            }
        }
        .padding(9)
        .background(Color.black.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.system(size: 8, weight: .semibold, design: .rounded))
                .foregroundColor(.white.opacity(0.5))
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
        }
    }

    private func bar(_ label: String, _ value: Float, _ maxValue: Float,
                     _ colour: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 7, weight: .semibold, design: .rounded))
                .foregroundColor(.white.opacity(0.45))
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.12))
                    Capsule().fill(colour)
                        .frame(width: geo.size.width *
                               CGFloat(min(1, max(0, value / maxValue))))
                }
            }
            .frame(width: 44, height: 4)
        }
    }

    // MARK: Controls

    private var worldControls: some View {
        VStack(spacing: 8) {
            // Plain buttons rather than a Picker: the segmented control was
            // being rebuilt ~10x a second by the live telemetry and never
            // committed a selection.
            HStack(spacing: 6) {
                ForEach(Environment.allCases) { e in
                    Button {
                        world.setEnvironment(e)
                    } label: {
                        Text(e.label)
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(world.environment == e
                                        ? Color.accentColor.opacity(0.85)
                                        : Color.white.opacity(0.10),
                                        in: RoundedRectangle(cornerRadius: 8,
                                                             style: .continuous))
                            .foregroundColor(.white)
                    }
                    .buttonStyle(.plain)
                }
            }

            HStack(spacing: 8) {
                actionButton("Food", "circle.hexagongrid.fill") { world.dropFood() }
                actionButton("Object", "cube.fill") { world.spawnObject() }
                actionButton("Swat", "hand.raised.fill", tint: .red) { world.swat() }
                actionButton("Reset", "arrow.counterclockwise") { world.resetFly() }
            }

            HStack(spacing: 10) {
                Toggle("Brain", isOn: $world.showBrainPiP)
                Toggle("Eye", isOn: $world.showEyePiP)
                Spacer()
            }
            .toggleStyle(.button)
            .font(.system(size: 11, weight: .medium, design: .rounded))
        }
        .padding(10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func actionButton(_ title: String, _ icon: String,
                              tint: Color = .accentColor,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 14))
                Text(title).font(.system(size: 9, weight: .medium, design: .rounded))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .background(tint.opacity(0.22))
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundColor(tint)
    }
}
