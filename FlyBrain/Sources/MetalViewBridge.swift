//
//  MetalViewBridge.swift
//  MTKView plus real UIKit gesture recognizers.
//
//  SwiftUI's DragGesture cannot tell one finger from two, which makes
//  "orbit with one, pan with two" impossible to express. UIPanGestureRecognizer
//  can, so the camera input lives here instead.
//
//    1 finger   orbit / look
//    2 fingers  truck and pedestal (slide the view sideways and up)
//    pinch      dolly — in Fly mode this actually travels forward
//    tap        select a neuron
//    long press glide forward continuously
//

import SwiftUI
import MetalKit

struct BrainMetalView: UIViewRepresentable {
    let engine: BrainEngine

    func makeCoordinator() -> Coordinator { Coordinator(engine: engine) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.device)
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.preferredFramesPerSecond = 60
        view.isOpaque = true
        view.backgroundColor = .black
        view.delegate = engine.renderer
        engine.attach(view: view)

        let c = context.coordinator
        c.view = view

        let orbit = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.handleOrbit(_:)))
        orbit.maximumNumberOfTouches = 1
        view.addGestureRecognizer(orbit)

        let pan = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.handlePan(_:)))
        pan.minimumNumberOfTouches = 2
        pan.maximumNumberOfTouches = 2
        view.addGestureRecognizer(pan)

        let pinch = UIPinchGestureRecognizer(target: c, action: #selector(Coordinator.handlePinch(_:)))
        pinch.delegate = c
        view.addGestureRecognizer(pinch)

        let tap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.handleTap(_:)))
        tap.delegate = c
        view.addGestureRecognizer(tap)

        let press = UILongPressGestureRecognizer(target: c, action: #selector(Coordinator.handlePress(_:)))
        press.minimumPressDuration = 0.35
        press.delegate = c
        view.addGestureRecognizer(press)

        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let engine: BrainEngine
        weak var view: MTKView?
        private var lastOrbit: CGPoint = .zero
        private var lastPan: CGPoint = .zero
        private var glideTimer: Timer?

        init(engine: BrainEngine) { self.engine = engine }

        // ---- one finger: orbit / look --------------------------------------

        @objc func handleOrbit(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { lastOrbit = .zero }
            let dx = Float(t.x - lastOrbit.x) * 0.006
            let dy = Float(t.y - lastOrbit.y) * 0.006
            lastOrbit = t
            engine.look(dx: dx, dy: dy)
        }

        // ---- two fingers: truck and pedestal --------------------------------

        @objc func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { lastPan = .zero }
            let dx = Float(t.x - lastPan.x) * 0.0025
            let dy = Float(t.y - lastPan.y) * 0.0025
            lastPan = t
            engine.pan(dx: dx, dy: dy)
        }

        // ---- pinch: dolly ----------------------------------------------------

        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            guard g.state == .changed else { g.scale = 1; return }
            let s = Float(g.scale)
            g.scale = 1
            engine.dolly(scale: s)
        }

        // ---- tap: inspect ----------------------------------------------------

        @objc func handleTap(_ g: UITapGestureRecognizer) {
            guard let v = view else { return }
            engine.tap(at: g.location(in: v))
        }

        // ---- long press: glide ------------------------------------------------

        @objc func handlePress(_ g: UILongPressGestureRecognizer) {
            switch g.state {
            case .began:
                startGlide()
            case .ended, .cancelled, .failed:
                stopGlide()
            default:
                break
            }
        }

        private func startGlide() {
            stopGlide()
            glideTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0,
                                              repeats: true) { [weak self] _ in
                guard let self else { return }
                Task { @MainActor in self.engine.glide(amount: 0.012) }
            }
        }

        func stopGlide() {
            glideTimer?.invalidate()
            glideTimer = nil
        }

        // Pinch and the single-finger pan must be able to run together,
        // otherwise a two-finger zoom cancels the orbit mid-gesture.
        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            true
        }
    }
}
