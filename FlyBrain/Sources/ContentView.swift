//
//  ContentView.swift
//  SwiftUI shell: the 3-D view, the HUD, the controls, the neuron inspector.
//

import SwiftUI
import MetalKit
import Combine

// MARK: - Root

struct ContentView: View {
    @StateObject private var engine = BrainEngine()
    @State private var showControls = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let error = engine.loadError {
                FailureView(message: error)
            } else if engine.isReady {
                BrainMetalView(engine: engine)
                    .ignoresSafeArea()

                VStack {
                    HUDView(stats: engine.stats,
                            fps: engine.fps,
                            simulatedMs: engine.stats.simulatedMilliseconds)
                    Spacer()
                    if let info = engine.selected {
                        NeuronInspector(info: info,
                                        metadata: engine.metadata,
                                        outDegree: engine.selectedOutDegree,
                                        onStimulate: { engine.stimulateSelected() },
                                        onDismiss: { engine.selected = nil })
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                    if showControls {
                        ControlPanel(engine: engine)
                            .transition(.move(edge: .bottom))
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
                .animation(.easeInOut(duration: 0.2), value: engine.selected?.index)
        .animation(.spring(response: 0.35, dampingFraction: 0.8),
                   value: engine.showCameraWindow)

                // Floating camera window, above the HUD but below nothing.
                if engine.cameraEnabled && engine.showCameraWindow {
                    FloatingCameraWindow(
                        session: engine.camera.session,
                        retinalDrive: engine.retinalActivity,
                        framesDelivered: engine.cameraFrames,
                        meanLuminance: engine.cameraLuminance,
                        onClose: { engine.showCameraWindow = false })
                    .transition(.scale.combined(with: .opacity))
                }

                VStack {
                    HStack {
                        Spacer()
                        Button {
                            withAnimation { showControls.toggle() }
                        } label: {
                            Image(systemName: showControls ? "chevron.down.circle.fill"
                                                           : "slider.horizontal.3")
                                .font(.title2)
                                .foregroundStyle(.white.opacity(0.8))
                        }
                        .padding(.trailing, 16)
                        .padding(.top, 54)
                    }
                    Spacer()
                }
            } else {
                LoadingView()
            }
        }
        .preferredColorScheme(.dark)
        .task { await engine.load() }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didEnterBackgroundNotification)) { _ in
            engine.setBackgrounded(true)
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.willEnterForegroundNotification)) { _ in
            engine.setBackgrounded(false)
        }
    }

}

// MARK: - HUD

struct HUDView: View {
    let stats: SimulationStats
    let fps: Double
    let simulatedMs: Int

    var body: some View {
        HStack(spacing: 14) {
            metric("FPS", String(format: "%.0f", fps),
                   colour: fps >= 55 ? .green : fps >= 28 ? .yellow : .orange)
            metric("spikes/s", compact(stats.spikesPerSecond))
            metric("active", compact(Double(stats.activeNeurons)))
            metric("rate", String(format: "%.1f Hz", stats.populationRateHz))
            metric("brain t", String(format: "%.1fs", Double(simulatedMs) / 1000))
            metric("≈power", String(format: "%.0f mW", stats.estimatedMilliwatts))
        }
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.top, 50)
    }

    private func metric(_ label: String, _ value: String,
                        colour: Color = .white) -> some View {
        VStack(spacing: 2) {
            Text(value).foregroundStyle(colour)
            Text(label).foregroundStyle(.secondary).font(.system(size: 9))
        }
    }

    private func compact(_ v: Double) -> String {
        v >= 1_000_000 ? String(format: "%.1fM", v / 1_000_000)
      : v >= 1_000     ? String(format: "%.1fk", v / 1_000)
                       : String(format: "%.0f", v)
    }
}

// MARK: - Controls

struct ControlPanel: View {
    @ObservedObject var engine: BrainEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {

            Picker("Mode", selection: $engine.renderMode) {
                ForEach(RenderMode.allCases) { m in Text(m.label).tag(m) }
            }
            .pickerStyle(.segmented)

            // The gain slider is the headline control: the network has a sharp
            // percolation threshold between 4 and 6, so dragging across it is a
            // visible phase transition rather than a brightness tweak.
            labelledSlider("Synaptic gain", value: $engine.gain,
                           range: 1...20,
                           detail: engine.gain < 5 ? "optic lobe only"
                                 : engine.gain > 12 ? "runaway" : "whole brain")

            labelledSlider("Retinal drive", value: $engine.retinalDrive,
                           range: 0...4, detail: nil)

            labelledSlider("Point size", value: $engine.pointScale,
                           range: 0.5...10, detail: nil)

            // Camera. One finger orbits or looks, two fingers slide, pinch
            // dollies, long press glides forward.
            HStack(spacing: 8) {
                Picker("Camera", selection: $engine.cameraMode) {
                    ForEach(CameraMode.allCases) { m in Text(m.label).tag(m) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 150)

                Button("Recentre") { engine.resetCamera() }
                    .buttonStyle(.bordered)
                    .font(.caption)
            }

            HStack(spacing: 14) {
                Toggle("inv X", isOn: $engine.invertLookX)
                Toggle("inv Y", isOn: $engine.invertLookY)
                Spacer()
            }
            .toggleStyle(.button)
            .font(.system(size: 10, weight: .medium))

            HStack {
                Toggle("Camera", isOn: $engine.cameraEnabled)
                    .toggleStyle(.switch)
                if engine.cameraEnabled && !engine.showCameraWindow {
                    Button {
                        engine.showCameraWindow = true
                    } label: {
                        Image(systemName: "pip.enter").font(.caption)
                    }
                }
                Spacer()
                Button(engine.isPaused ? "Resume" : "Pause") { engine.isPaused.toggle() }
                    .buttonStyle(.bordered)
                Button("Reset") { engine.resetSimulation() }
                    .buttonStyle(.bordered)
            }
            .font(.caption)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(engine.metadata.systems.enumerated()), id: \.offset) { i, name in
                        Button {
                            engine.toggleSystem(i)
                        } label: {
                            Text(prettify(name))
                                .font(.system(size: 10, weight: .medium))
                                .padding(.horizontal, 8).padding(.vertical, 5)
                                .background(engine.isSystemVisible(i)
                                            ? Color.accentColor.opacity(0.75)
                                            : Color.white.opacity(0.1),
                                            in: Capsule())
                                .foregroundStyle(.white)
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func labelledSlider(_ title: String,
                                value: Binding<Float>,
                                range: ClosedRange<Float>,
                                detail: String?) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(title).font(.system(size: 10, weight: .medium))
                Spacer()
                Text(String(format: "%.1f", value.wrappedValue))
                    .font(.system(size: 10, design: .monospaced))
                if let d = detail {
                    Text(d).font(.system(size: 9)).foregroundStyle(.secondary)
                }
            }
            Slider(value: value, in: range)
        }
    }

    private func prettify(_ s: String) -> String {
        s.replacingOccurrences(of: "_", with: " ")
    }
}

// MARK: - Inspector

struct NeuronInspector: View {
    let info: NeuronInfo
    let metadata: ConnectomeMetadata
    let outDegree: Int
    let onStimulate: () -> Void
    let onDismiss: () -> Void

    private static let ntNames = ["ACh", "GABA", "Glut", "DA", "5-HT", "OA", "?"]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(metadata.cellTypeName(info.cellTypeID))
                    .font(.headline)
                Spacer()
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
            Text("FlyWire id \(String(info.rootID))")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            HStack(spacing: 16) {
                field("region", pretty(metadata.systems, info.system))
                field("class", pretty(metadata.superClasses, info.superClass))
                field("NT", Self.ntNames[min(info.neurotransmitter, 6)])
                field("side", info.side)
            }
            HStack(spacing: 16) {
                field("outputs", "\(outDegree)")
                if info.isRetina { tag("photoreceptor", .blue) }
                if info.isMotor { tag("descending", .orange) }
                if info.isAfferent { tag("afferent", .green) }
            }

            Button("Stimulate this neuron", action: onStimulate)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        }
        .font(.caption)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func field(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 11, weight: .medium))
        }
    }

    private func tag(_ text: String, _ colour: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 6).padding(.vertical, 3)
            .background(colour.opacity(0.3), in: Capsule())
    }

    private func pretty(_ list: [String], _ i: Int) -> String {
        guard i >= 0 && i < list.count else { return "?" }
        return list[i].replacingOccurrences(of: "_", with: " ")
    }
}

// MARK: - Loading / failure

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 14) {
            ProgressView().tint(.white)
            Text("Loading 139,255 neurons…")
                .font(.footnote).foregroundStyle(.secondary)
            Text("FlyWire FAFB v783").font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

struct FailureView: View {
    let message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle).foregroundStyle(.orange)
            Text("Could not start the brain").font(.headline)
            Text(message)
                .font(.caption).multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 32)
        }
    }
}
