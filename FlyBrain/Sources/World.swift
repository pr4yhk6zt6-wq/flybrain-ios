//
//  World.swift
//  A small 3-D world for the fly to live in.
//
//  The loop runs like this, once per frame:
//
//      world  -> offscreen render from the fly's head  -> photoreceptors
//      brain  -> 391 leg + 60 wing motor neurons fire
//      body   -> wing beat and leg stepping integrate into a pose
//      world  -> smells, collisions and pain write back into sensory groups
//

import Foundation
import simd

// MARK: - Scenery

enum Environment: String, CaseIterable, Identifiable {
    case kitchen, garden, lab
    var id: String { rawValue }

    var label: String {
        switch self {
        case .kitchen: return "Kitchen"
        case .garden:  return "Garden"
        case .lab:     return "Lab"
        }
    }

    var skyColour: SIMD3<Float> {
        switch self {
        case .kitchen: return SIMD3<Float>(0.14, 0.12, 0.14)
        case .garden:  return SIMD3<Float>(0.38, 0.62, 0.86)
        case .lab:     return SIMD3<Float>(0.07, 0.08, 0.10)
        }
    }

    var groundColour: SIMD3<Float> {
        switch self {
        case .kitchen: return SIMD3<Float>(0.55, 0.42, 0.28)   // wooden table
        case .garden:  return SIMD3<Float>(0.22, 0.42, 0.16)   // grass
        case .lab:     return SIMD3<Float>(0.85, 0.86, 0.88)   // white bench
        }
    }

    /// Scale of the ground checker pattern — the texture the fly's optic flow
    /// actually keys off. A featureless floor gives the visual system nothing.
    var groundChecker: Float {
        switch self {
        case .kitchen: return 3.0
        case .garden:  return 10.0
        case .lab:     return 1.5
        }
    }
}

enum ObjectKind: String {
    case food, fruit, cube, pillar, swatter, wall, ceiling

    var isEdible: Bool { self == .food || self == .fruit }

    /// How strongly this object smells. Only food does.
    var odour: Float {
        switch self {
        case .food:  return 1.0
        case .fruit: return 0.7
        default:     return 0.0
        }
    }

    var colour: SIMD3<Float> {
        switch self {
        case .food:    return SIMD3<Float>(0.95, 0.75, 0.25)
        case .fruit:   return SIMD3<Float>(0.85, 0.22, 0.30)
        case .cube:    return SIMD3<Float>(0.45, 0.50, 0.62)
        case .pillar:  return SIMD3<Float>(0.60, 0.58, 0.52)
        case .swatter: return SIMD3<Float>(0.12, 0.12, 0.14)
        case .wall:    return SIMD3<Float>(0.52, 0.47, 0.42)
        case .ceiling: return SIMD3<Float>(0.30, 0.29, 0.28)
        }
    }

    var mesh: MeshKind {
        switch self {
        case .food, .fruit: return .sphere
        default:            return .cube
        }
    }

    /// Scenery the fly can see but that is not worth listing as an obstacle
    /// twice; walls get their own collision path.
    var isStructure: Bool { self == .wall || self == .ceiling }
}

enum MeshKind: Int { case cube = 0, sphere = 1, quad = 2 }

struct WorldObject: Identifiable {
    let id = UUID()
    var kind: ObjectKind
    var position: SIMD3<Float>
    var size: SIMD3<Float>
    var velocity: SIMD3<Float> = .zero
    var spin: Float = 0
    /// Food shrinks as the fly eats it.
    var amount: Float = 1.0
}

/// Result of a collision query: where the body should be pushed back to, and
/// the surface normal it hit.
struct Contact {
    var correctedPosition: SIMD3<Float>
    var normal: SIMD3<Float>
}

// MARK: - The world

final class World {
    private(set) var objects: [WorldObject] = []
    var environment: Environment = .kitchen {
        didSet { if environment != oldValue { rebuild() } }
    }

    /// A descending swatter, when the user takes a swipe at the fly.
    private(set) var swatter: WorldObject?
    private var swatProgress: Float = 0
    private(set) var swatImpact = false

    /// The world is OPEN: a floor under an open sky, no walls, no ceiling.
    ///
    /// The old 12 cm box was a matchbox for an animal that flies at 0.9 m/s —
    /// it crossed the whole room in a sixth of a second and spent its life
    /// colliding with walls, which is exactly what the player saw. Real
    /// kitchens and gardens are metres across relative to a 2.5 mm fly.
    ///
    /// `bounds` survives only as an invisible analytical backstop, 20 m out,
    /// that keeps the integrator from wandering to infinity; the fog hides
    /// it completely, so the horizon reads as endless.
    let bounds: Float = 2000.0
    /// Invisible sky cap. Nothing is drawn at this height; it only keeps the
    /// fly low enough that the ground still fills its eyes and the chase
    /// camera never loses it overhead. At 1.5 m the floor is well inside the
    /// fog horizon; at the old 5 m the fly climbed until the world below it
    /// dissolved into haze and the visual system had nothing to hold on to.
    var ceilingHeight: Float { 150.0 }
    /// Scenery clusters near the origin so the fly always has things to
    /// interact with while the space around it stays open.
    let sceneryExtent: Float = 250.0

    init() { rebuild() }

    func rebuild() {
        objects.removeAll()

        // No room any more — see `bounds`. The floor runs to the fog horizon
        // in every environment; only loose scenery and food are placed.
        switch environment {
        case .kitchen:
            addScenery(count: 5, kind: .cube, sizeRange: 0.25...0.8)
            addScenery(count: 2, kind: .pillar, sizeRange: 0.3...0.5, tall: 2.0)
            spawn(.fruit, at: SIMD3<Float>(1.4, 0.18, -1.1), size: 0.18)
        case .garden:
            addScenery(count: 9, kind: .pillar, sizeRange: 0.08...0.18, tall: 3.0)
            addScenery(count: 4, kind: .cube, sizeRange: 0.2...0.45)
            spawn(.fruit, at: SIMD3<Float>(-1.8, 0.2, 1.6), size: 0.22)
        case .lab:
            addScenery(count: 4, kind: .cube, sizeRange: 0.4...0.6)
        }
        swatter = nil
        swatImpact = false
    }

    private func addScenery(count: Int, kind: ObjectKind,
                            sizeRange: ClosedRange<Float>, tall: Float = 1.0) {
        for _ in 0..<count {
            let s = Float.random(in: sizeRange)
            let x = Float.random(in: -sceneryExtent...sceneryExtent)
            let z = Float.random(in: -sceneryExtent...sceneryExtent)
            let h = s * tall
            objects.append(WorldObject(kind: kind,
                                       position: SIMD3<Float>(x, h * 0.5, z),
                                       size: SIMD3<Float>(s, h, s)))
        }
    }

    @discardableResult
    func spawn(_ kind: ObjectKind, at p: SIMD3<Float>, size: Float) -> UUID {
        let o = WorldObject(kind: kind, position: p,
                            size: SIMD3<Float>(repeating: size))
        objects.append(o)
        return o.id
    }

    func removeAll(ofKind kind: ObjectKind) {
        objects.removeAll { $0.kind == kind }
    }

    /// Bring the swatter down. This is the escape-reflex stimulus.
    func swat(at p: SIMD3<Float>) {
        swatter = WorldObject(kind: .swatter,
                              position: SIMD3<Float>(p.x, 3.0, p.z),
                              size: SIMD3<Float>(1.6, 0.08, 1.6))
        swatProgress = 0
        swatImpact = false
    }

    func update(dt: Float) {
        for i in objects.indices {
            // Walls and the ceiling are fixed; only loose objects fall.
            if objects[i].kind.isStructure { continue }
            let restY = objects[i].size.y * 0.5
            if objects[i].position.y > restY {
                objects[i].velocity.y -= 9.8 * dt * 0.2      // gentle, fly-scale
                objects[i].position += objects[i].velocity * dt
                if objects[i].position.y <= restY {
                    objects[i].position.y = restY
                    objects[i].velocity = .zero
                }
            }
        }
        objects.removeAll { $0.kind.isEdible && $0.amount <= 0.02 }

        if swatter != nil {
            swatProgress += dt * 2.6
            let t = swatProgress
            // Down fast, pause, back up.
            let y: Float
            if t < 0.45 { y = 3.0 - (t / 0.45) * 2.85 }
            else if t < 0.75 { y = 0.15 }
            else { y = 0.15 + ((t - 0.75) / 0.6) * 3.0 }
            swatter?.position.y = y
            swatImpact = (t >= 0.35 && t <= 0.8)
            if t > 1.4 { swatter = nil; swatImpact = false }
        }
    }

    // MARK: - What the body senses

    /// Odour concentration and its left/right asymmetry at a point, from an
    /// inverse-square falloff around every edible object. The asymmetry is what
    /// lets the fly actually track a smell rather than just notice one.
    func odour(at p: SIMD3<Float>, heading: Float) -> (strength: Float, bias: Float) {
        var total: Float = 0
        var lateral: Float = 0
        let rightVec = SIMD3<Float>(cos(heading), 0, -sin(heading))
        for o in objects where o.kind.odour > 0 {
            let d = o.position - p
            let dist = max(length(d), 0.15)
            let c = o.kind.odour * o.amount / (dist * dist)
            total += c
            lateral += c * dot(normalize(d), rightVec)
        }
        return (min(total, 4.0), max(-1, min(1, lateral)))
    }

    func nearestFood(to p: SIMD3<Float>) -> WorldObject? {
        objects.filter { $0.kind.isEdible }
               .min { length($0.position - p) < length($1.position - p) }
    }

    /// Anything solid within `radius` of the point.
    ///
    /// Returns a corrected position as well as a normal, so the caller cannot
    /// end up tunnelling or sticking: we push the body exactly to the surface
    /// rather than nudging it by a fraction and hoping.
    func collision(at p: SIMD3<Float>, radius: Float) -> Contact? {
        var best: Contact?
        var deepest: Float = 0

        for o in objects where !o.kind.isEdible {
            let half = o.size * 0.5
            let lo = o.position - half
            let hi = o.position + half
            let closest = simd_clamp(p, lo, hi)
            let d = p - closest
            let dist = length(d)

            if dist > 1e-5 {
                guard dist < radius else { continue }
                let n = d / dist
                let depth = radius - dist
                if depth > deepest {
                    deepest = depth
                    best = Contact(correctedPosition: closest + n * radius, normal: n)
                }
            } else {
                // Centre is inside the box: escape along the shallowest face.
                let toLo = p - lo
                let toHi = hi - p
                var n = SIMD3<Float>(0, 1, 0)
                var push: Float = .greatestFiniteMagnitude
                let faces: [(Float, SIMD3<Float>)] = [
                    (toLo.x, SIMD3<Float>(-1, 0, 0)), (toHi.x, SIMD3<Float>(1, 0, 0)),
                    (toLo.y, SIMD3<Float>(0, -1, 0)), (toHi.y, SIMD3<Float>(0, 1, 0)),
                    (toLo.z, SIMD3<Float>(0, 0, -1)), (toHi.z, SIMD3<Float>(0, 0, 1)),
                ]
                for (dpt, nn) in faces where dpt < push { push = dpt; n = nn }
                let depth = push + radius
                if depth > deepest {
                    deepest = depth
                    best = Contact(correctedPosition: p + n * depth, normal: n)
                }
            }
        }

        // The ground.
        if p.y < radius {
            let depth = radius - p.y
            if depth > deepest {
                best = Contact(correctedPosition: SIMD3<Float>(p.x, radius, p.z),
                               normal: SIMD3<Float>(0, 1, 0))
            }
        }
        return best
    }

    /// Hard containment. Wall boxes alone are not enough: a fly at 0.9 m/s
    /// crosses 1.5 cm in a single 16 ms frame, so a point-in-box test can miss
    /// a 0.25 cm thick wall entirely and the animal tunnels straight out of
    /// the room. This clamps the position to the interior as a backstop, and
    /// reports which axes were hit so the caller can kill that velocity.
    func contain(_ p: inout SIMD3<Float>, radius: Float) -> SIMD3<Float> {
        var hitNormal = SIMD3<Float>.zero
        let limit = bounds - radius
        if p.x < -limit { p.x = -limit; hitNormal.x =  1 }
        if p.x >  limit { p.x =  limit; hitNormal.x = -1 }
        if p.z < -limit { p.z = -limit; hitNormal.z =  1 }
        if p.z >  limit { p.z =  limit; hitNormal.z = -1 }
        if p.y < radius { p.y = radius; hitNormal.y =  1 }
        let top = ceilingHeight - radius
        if p.y > top { p.y = top; hitNormal.y = -1 }
        return hitNormal
    }

    func consume(_ id: UUID, amount: Float) {
        guard let i = objects.firstIndex(where: { $0.id == id }) else { return }
        objects[i].amount -= amount
    }
}
