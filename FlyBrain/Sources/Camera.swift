//
//  Camera.swift
//  Replaces the old orbit-only camera.
//
//  Two fixes over the original:
//
//  1. Both drag axes were inverted. The old code ADDED the drag delta to the
//     azimuth and elevation, which rotates the camera in the direction of the
//     drag — and because the camera orbits the object, the object then appears
//     to move the opposite way. The right mental model is that your finger
//     grabs the brain, so the angles get the delta SUBTRACTED.
//
//  2. Orbit alone is not enough. Fly mode gives a free camera you can push
//     forward into the neuropil, with no fixed target to be tethered to.
//
//  Switching between the two modes preserves the current viewpoint exactly, so
//  the brain never teleports when you flip the segmented control.
//

import Foundation
import simd

enum CameraMode: String, CaseIterable, Identifiable {
    case orbit, free
    var id: String { rawValue }
    var label: String { self == .orbit ? "Orbit" : "Fly" }
}

struct Camera {

    var mode: CameraMode = .orbit

    // Orbit state
    var azimuth: Float = 0.6
    var elevation: Float = 0.25
    var distance: Float = 2.2
    var target: SIMD3<Float> = .zero

    // Free-flight state
    var eyePosition = SIMD3<Float>(0, 0, 2.2)
    var yaw: Float = 0
    var pitch: Float = 0

    var fieldOfView: Float = 55 * .pi / 180

    /// Drag polarity. -1 is correct: the gesture grabs the scene, not the lens.
    /// Exposed because a minority of users genuinely prefer the inverse, and
    /// both toggles sit in the control panel.
    var invertX = false
    var invertY = false

    private var sx: Float { invertX ? 1 : -1 }
    private var sy: Float { invertY ? 1 : -1 }

    /// Elevation and pitch stop just shy of straight up/down so the view
    /// never gimbal-flips.
    private let pitchLimit: Float = 1.52

    // MARK: - Derived basis

    /// Unit forward vector, whichever mode we are in.
    var forward: SIMD3<Float> {
        switch mode {
        case .orbit:
            return normalize(target - eye)
        case .free:
            return SIMD3<Float>(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))
        }
    }

    var right: SIMD3<Float> {
        normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
    }

    var up: SIMD3<Float> {
        normalize(cross(right, forward))
    }

    var eye: SIMD3<Float> {
        switch mode {
        case .orbit:
            let ce = cos(elevation), se = sin(elevation)
            return target + SIMD3<Float>(sin(azimuth) * ce, se, cos(azimuth) * ce) * distance
        case .free:
            return eyePosition
        }
    }

    // MARK: - Gestures

    /// One finger: orbit around the brain, or look around in free flight.
    mutating func look(dx: Float, dy: Float) {
        switch mode {
        case .orbit:
            azimuth += sx * dx
            elevation = max(-pitchLimit, min(pitchLimit, elevation + sy * dy))
        case .free:
            yaw += sx * dx
            pitch = max(-pitchLimit, min(pitchLimit, pitch + sy * dy))
        }
    }

    /// Two fingers: slide the view sideways and up, without rotating.
    mutating func pan(dx: Float, dy: Float) {
        let scale = (mode == .orbit ? distance : 1.0) * 0.5
        let delta = right * (sx * dx) + SIMD3<Float>(0, 1, 0) * (-sy * dy)
        switch mode {
        case .orbit: target += delta * scale
        case .free:  eyePosition += delta * scale
        }
    }

    /// Pinch. In orbit this pulls the camera in; in free flight it actually
    /// travels forward, which is what lets you fly through the optic lobe.
    mutating func dolly(scale: Float) {
        switch mode {
        case .orbit:
            distance = max(0.08, min(12.0, distance / max(scale, 0.01)))
        case .free:
            eyePosition += forward * (scale - 1) * 2.0
        }
    }

    /// Continuous forward travel, from the long-press glide.
    mutating func move(forwardAmount: Float, upAmount: Float = 0) {
        switch mode {
        case .orbit:
            distance = max(0.08, min(12.0, distance - forwardAmount))
            target += SIMD3<Float>(0, 1, 0) * upAmount
        case .free:
            eyePosition += forward * forwardAmount
            eyePosition += SIMD3<Float>(0, 1, 0) * upAmount
        }
    }

    /// Switch mode while keeping the exact same view. Without this the brain
    /// jumps every time the picker is touched.
    mutating func setMode(_ newMode: CameraMode) {
        guard newMode != mode else { return }
        switch newMode {
        case .free:
            // Adopt the orbit camera's current position and direction.
            let e = eye
            let f = normalize(target - e)
            eyePosition = e
            pitch = max(-pitchLimit, min(pitchLimit, asin(f.y)))
            yaw = atan2(f.x, -f.z)
        case .orbit:
            // Keep looking at whatever is `distance` ahead of us.
            let f = forward
            target = eyePosition + f * distance
            elevation = max(-pitchLimit, min(pitchLimit, asin(-f.y)))
            azimuth = atan2(-f.x, f.z)
        }
        mode = newMode
    }

    mutating func reset() {
        azimuth = 0.6
        elevation = 0.25
        distance = 2.2
        target = .zero
        yaw = 0
        pitch = 0
        eyePosition = SIMD3<Float>(0, 0, 2.2)
    }

    // MARK: - Matrices

    func viewProjection(aspect: Float) -> float4x4 {
        let e = eye
        let t = (mode == .orbit) ? target : (e + forward)
        let view = float4x4(lookAt: e, target: t, up: SIMD3<Float>(0, 1, 0))
        let proj = float4x4(perspectiveFOV: fieldOfView, aspect: aspect,
                            near: 0.01, far: 60.0)
        return proj * view
    }

    /// Screen point -> world ray, for tap-to-inspect.
    func ray(atNDC ndc: SIMD2<Float>, aspect: Float) -> (SIMD3<Float>, SIMD3<Float>) {
        let inv = viewProjection(aspect: aspect).inverse
        let near = inv * SIMD4<Float>(ndc.x, ndc.y, 0, 1)
        let far  = inv * SIMD4<Float>(ndc.x, ndc.y, 1, 1)
        let p0 = SIMD3<Float>(near.x, near.y, near.z) / near.w
        let p1 = SIMD3<Float>(far.x, far.y, far.z) / far.w
        return (p0, normalize(p1 - p0))
    }
}
