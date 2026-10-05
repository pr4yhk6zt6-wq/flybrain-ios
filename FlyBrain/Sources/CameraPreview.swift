//
//  CameraPreview.swift
//  A floating, draggable window showing what the fly is looking at.
//
//  Two things are going on here. The obvious one is the video-call-style picture
//  in picture. The less obvious one is that it doubles as a diagnostic: the
//  camera drives only 10,647 of 139,255 neurons, so its effect on the whole
//  point cloud is genuinely subtle. The drive meter under the video makes the
//  coupling visible — cover the lens and the bar collapses within a second.
//

import SwiftUI
import AVFoundation

// MARK: - The AVFoundation layer, wrapped for SwiftUI

struct CameraLayerView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewContainer {
        let v = PreviewContainer()
        v.backgroundColor = .black
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ view: PreviewContainer, context: Context) {
        if view.previewLayer.session !== session {
            view.previewLayer.session = session
        }
    }

    final class PreviewContainer: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}

// MARK: - The floating window

struct FloatingCameraWindow: View {
    let session: AVCaptureSession
    let retinalDrive: Float
    let framesDelivered: Int
    let meanLuminance: Float
    /// Where this window is allowed to be: the gap between the readouts at the
    /// top and the controls at the bottom, measured from the bars themselves by
    /// the screen that owns it (LayoutBand.swift). A window clamped with a
    /// guessed inset can be dropped onto a button, which is what it used to do.
    let band: PaneBand
    let onClose: () -> Void

    /// Persisted across launches so the window reappears where it was left.
    @AppStorage("cameraWindowX") private var storedX: Double = 0
    @AppStorage("cameraWindowY") private var storedY: Double = 0
    @AppStorage("cameraWindowExpanded") private var expanded: Bool = true

    @State private var dragOffset: CGSize = .zero
    @GestureState private var isDragging = false

    private var size: CGSize {
        expanded ? CGSize(width: 132, height: 200) : CGSize(width: 92, height: 70)
    }

    var body: some View {
        VStack(spacing: 0) {

            ZStack(alignment: .topTrailing) {
                CameraLayerView(session: session)
                    .frame(width: size.width, height: size.height * (expanded ? 0.74 : 1.0))
                    .clipped()

                // A faint grid standing in for the ommatidial lattice: a reminder
                // that this frame is being resampled into 10,647 photoreceptors,
                // not shown as a photograph.
                if expanded {
                    OmmatidiaOverlay()
                        .frame(width: size.width, height: size.height * 0.74)
                        .allowsHitTesting(false)
                }

                HStack(spacing: 4) {
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            expanded.toggle()
                        }
                    } label: {
                        Image(systemName: expanded ? "arrow.down.right.and.arrow.up.left"
                                                   : "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(4)
                            .background(.black.opacity(0.45), in: Circle())
                    }
                    Button(action: onClose) {
                        Image(systemName: "xmark")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(4)
                            .background(.black.opacity(0.45), in: Circle())
                    }
                }
                .padding(5)
            }

            if expanded {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(framesDelivered > 0 ? Color.green : Color.orange)
                            .frame(width: 5, height: 5)
                        Text(framesDelivered > 0 ? "FLY EYE" : "waiting…")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.85))
                        Spacer()
                        Text("10,647")
                            .font(.system(size: 7, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.45))
                    }

                    DriveMeter(value: retinalDrive, maximum: 2.0)
                        .frame(height: 4)

                    HStack {
                        Text("photoreceptor drive")
                            .font(.system(size: 7))
                            .foregroundStyle(.white.opacity(0.45))
                        Spacer()
                        Text(String(format: "%.2f", retinalDrive))
                            .font(.system(size: 7, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                .padding(.horizontal, 7)
                .padding(.vertical, 5)
                .frame(width: size.width)
                .background(Color.black.opacity(0.55))
            }
        }
        .frame(width: size.width)
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(isDragging ? 0.55 : 0.18), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.6), radius: isDragging ? 18 : 10, y: 4)
        .scaleEffect(isDragging ? 1.04 : 1.0)
        .offset(x: storedX + dragOffset.width, y: storedY + dragOffset.height)
        .gesture(
            DragGesture()
                .updating($isDragging) { _, state, _ in state = true }
                .onChanged { value in dragOffset = value.translation }
                .onEnded { value in
                    storedX += value.translation.width
                    storedY += value.translation.height
                    dragOffset = .zero
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                        clampToBand()
                    }
                }
        )
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isDragging)
        .onAppear { clampToBand() }
        // The band moves when a bar grows — a rotation, a larger text size, a
        // longer readout — and a window outside its band puts itself back.
        .onChange(of: band) { _ in clampToBand() }
    }

    /// Keep the window in the gap between the two bars, whatever the screen.
    ///
    /// This replaces a clamp against `UIScreen.main.bounds` with a fixed
    /// 100-point allowance for the controls, which was right on one phone and
    /// let the window cover the readouts or the buttons on every other.
    private func clampToBand() {
        let p = band.clamp(CGSize(width: CGFloat(storedX),
                                  height: CGFloat(storedY)))
        storedX = Double(p.width)
        storedY = Double(p.height)
    }
}

// MARK: - Bits and pieces

/// Horizontal bar whose colour tracks how hard the photoreceptors are driven.
struct DriveMeter: View {
    let value: Float
    let maximum: Float

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.12))
                Capsule()
                    .fill(LinearGradient(colors: [.blue, .cyan, .yellow, .white],
                                         startPoint: .leading, endPoint: .trailing))
                    .frame(width: geo.size.width *
                           CGFloat(min(max(value / maximum, 0), 1)))
            }
        }
        .animation(.linear(duration: 0.15), value: value)
    }
}

/// Hex-ish lattice hinting at the compound eye. Purely cosmetic.
struct OmmatidiaOverlay: View {
    var body: some View {
        Canvas { context, size in
            let spacing: CGFloat = 11
            var path = Path()
            var row = 0
            var y: CGFloat = 0
            while y <= size.height {
                let offset: CGFloat = (row % 2 == 0) ? 0 : spacing / 2
                var x: CGFloat = offset
                while x <= size.width {
                    path.addEllipse(in: CGRect(x: x - 3.6, y: y - 3.6,
                                               width: 7.2, height: 7.2))
                    x += spacing
                }
                y += spacing * 0.86
                row += 1
            }
            context.stroke(path, with: .color(.white.opacity(0.10)), lineWidth: 0.5)
        }
    }
}
