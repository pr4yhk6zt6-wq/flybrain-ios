//
//  FlyDynamics.swift
//  The animal, as physics, on the phone.
//
//  This is a transliteration of tools/fly_aba.py — the same Featherstone
//  articulated-body algorithm that was checked against MuJoCo three ways
//  before this file existed (kinematics, twenty milliseconds of free fall,
//  and the animal standing on its own six feet for three seconds on a floor
//  it has never seen). The Python is the reference; this file is the port,
//  and `FlyDynamicsTests` replays the golden trace that `fly_aba.py
//  --golden` writes so the two cannot drift apart in silence.
//
//  What is being simulated, in order of who is in charge:
//
//    * 103 bodies and 102 hinge joints, from the real Janelia / Google
//      DeepMind *flybody* model: masses, inertial tensors, joint axes,
//      anatomical ranges and joint springs are the published values.
//    * Torque at every joint, from an antagonist pair of muscles, with a
//      Hill force-length curve and a *per-direction* force-velocity curve.
//      The solver is never told an angle; it is told a torque.
//    * A floor, as an exact penalty contact on each of the 74 collision
//      geoms — a convex shape's lowest point is its centre minus its support
//      function along the plane normal, so no pair-wise geometry is needed
//      and all 74 cost nothing.
//
//  Conventions, fixed once and used everywhere (Featherstone 2008):
//
//    motion  v = [omega; nu]     angular first, linear second
//    force   f = [n; f_lin]      moment first, linear second
//    v_child = X v_parent ,  f_parent = X^T f_child
//    X = [[E, 0], [-(E r)x, E]]  E = R_child^T R_parent,  r = child origin
//                                relative to the parent origin, in parent
//                                coordinates
//
//  Units are the model's own: centimetres, grams, seconds, so one unit of
//  torque is a nano-newton-metre. Gravity is -981 cm/s^2.
//

import Foundation

// MARK: - 3-vectors and 3x3 matrices

/// A plain 3-vector. Doubles, because the phone and the Python must agree to
/// the last bit that matters and Float would not survive the stiff joints.
struct Vec3 {
    var x: Double = 0, y: Double = 0, z: Double = 0

    init() {}
    init(_ x: Double, _ y: Double, _ z: Double) { self.x = x; self.y = y; self.z = z }
    init(_ a: [Double]) {
        x = a.count > 0 ? a[0] : 0
        y = a.count > 1 ? a[1] : 0
        z = a.count > 2 ? a[2] : 0
    }

    subscript(i: Int) -> Double {
        get { i == 0 ? x : (i == 1 ? y : z) }
        set { if i == 0 { x = newValue } else if i == 1 { y = newValue } else { z = newValue } }
    }

    static let zero = Vec3(0, 0, 0)
    static let up = Vec3(0, 0, 1)

    var lengthSquared: Double { x * x + y * y + z * z }
    var length: Double { lengthSquared.squareRoot() }
    var array: [Double] { [x, y, z] }
    var simd: SIMD3<Double> { SIMD3(x, y, z) }

    static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    static prefix func - (a: Vec3) -> Vec3 { Vec3(-a.x, -a.y, -a.z) }
    static func * (a: Vec3, s: Double) -> Vec3 { Vec3(a.x * s, a.y * s, a.z * s) }
    static func * (s: Double, a: Vec3) -> Vec3 { a * s }
    static func / (a: Vec3, s: Double) -> Vec3 { Vec3(a.x / s, a.y / s, a.z / s) }
    static func += (a: inout Vec3, b: Vec3) { a = a + b }
    static func -= (a: inout Vec3, b: Vec3) { a = a - b }
}

@inline(__always) func dot(_ a: Vec3, _ b: Vec3) -> Double {
    a.x * b.x + a.y * b.y + a.z * b.z
}

@inline(__always) func cross(_ a: Vec3, _ b: Vec3) -> Vec3 {
    Vec3(a.y * b.z - a.z * b.y,
         a.z * b.x - a.x * b.z,
         a.x * b.y - a.y * b.x)
}

/// A 3x3 matrix held as three *columns*, which is what every product in this
/// file wants (matrix times vector is a linear combination of the columns).
struct Mat3 {
    var c0 = Vec3(1, 0, 0)
    var c1 = Vec3(0, 1, 0)
    var c2 = Vec3(0, 0, 1)

    init() {}
    init(columns: (Vec3, Vec3, Vec3)) { c0 = columns.0; c1 = columns.1; c2 = columns.2 }
    /// Row-major, the way a matrix is written on paper.
    init(rows a: [Double], _ b: [Double], _ c: [Double]) {
        c0 = Vec3(a[0], b[0], c[0])
        c1 = Vec3(a[1], b[1], c[1])
        c2 = Vec3(a[2], b[2], c[2])
    }

    static let identity = Mat3()

    /// The zero matrix. `Mat3()` is the *identity* — that is the right default
    /// for an orientation and the wrong one for a block that is supposed to
    /// vanish, and confusing the two is how the spatial transforms below once
    /// grew an identity in their upper-right block. Spell it out.
    static let zero = Mat3(columns: (Vec3.zero, Vec3.zero, Vec3.zero))

    var transpose: Mat3 { Mat3(columns: (Vec3(c0.x, c1.x, c2.x),
                                        Vec3(c0.y, c1.y, c2.y),
                                        Vec3(c0.z, c1.z, c2.z))) }

    /// Row `i` of the matrix, as a vector — used by the plane-support code.
    func row(_ i: Int) -> Vec3 {
        Vec3(c0[i], c1[i], c2[i])
    }

    static func * (m: Mat3, v: Vec3) -> Vec3 {
        m.c0 * v.x + m.c1 * v.y + m.c2 * v.z
    }

    static func * (a: Mat3, b: Mat3) -> Mat3 {
        Mat3(columns: (a * b.c0, a * b.c1, a * b.c2))
    }

    static func += (a: inout Mat3, b: Mat3) { a = a + b }
    static func + (a: Mat3, b: Mat3) -> Mat3 {
        Mat3(columns: (a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2))
    }
    static func * (m: Mat3, s: Double) -> Mat3 {
        Mat3(columns: (m.c0 * s, m.c1 * s, m.c2 * s))
    }
}

/// The cross-product matrix: `crossMat(a) * b == cross(a, b)`.
@inline(__always) func crossMat(_ a: Vec3) -> Mat3 {
    Mat3(rows: [0, -a.z, a.y],
         [a.z, 0, -a.x],
         [-a.y, a.x, 0])
}

// MARK: - Spatial (6D) vectors and matrices

/// A spatial motion or force vector: angular part first, then linear.
struct SVec6 {
    var w = Vec3.zero
    var v = Vec3.zero

    init() {}
    init(_ w: Vec3, _ v: Vec3) { self.w = w; self.v = v }

    static let zero = SVec6(Vec3.zero, Vec3.zero)
    static func + (a: SVec6, b: SVec6) -> SVec6 { SVec6(a.w + b.w, a.v + b.v) }
    static func - (a: SVec6, b: SVec6) -> SVec6 { SVec6(a.w - b.w, a.v - b.v) }
    static prefix func - (a: SVec6) -> SVec6 { SVec6(-a.w, -a.v) }
    static func * (a: SVec6, s: Double) -> SVec6 { SVec6(a.w * s, a.v * s) }
    static func += (a: inout SVec6, b: SVec6) { a = a + b }
    static func -= (a: inout SVec6, b: SVec6) { a = a - b }
}

@inline(__always) func dot(_ a: SVec6, _ b: SVec6) -> Double {
    dot(a.w, b.w) + dot(a.v, b.v)
}

/// A 6x6 matrix as four 3x3 blocks: [[a, b], [c, d]].
struct SMat6 {
    var a = Mat3.identity
    var b = Mat3()
    var c = Mat3()
    var d = Mat3.identity

    init() {}
    init(a: Mat3, b: Mat3, c: Mat3, d: Mat3) {
        self.a = a; self.b = b; self.c = c; self.d = d
    }

    static let identity = SMat6()

    var transpose: SMat6 {
        SMat6(a: a.transpose, b: c.transpose, c: b.transpose, d: d.transpose)
    }

    static func * (m: SMat6, v: SVec6) -> SVec6 {
        SVec6(m.a * v.w + m.b * v.v, m.c * v.w + m.d * v.v)
    }

    static func * (m: SMat6, n: SMat6) -> SMat6 {
        SMat6(a: m.a * n.a + m.b * n.c,
              b: m.a * n.b + m.b * n.d,
              c: m.c * n.a + m.d * n.c,
              d: m.c * n.b + m.d * n.d)
    }

    static func + (m: SMat6, n: SMat6) -> SMat6 {
        SMat6(a: m.a + n.a, b: m.b + n.b, c: m.c + n.c, d: m.d + n.d)
    }

    static func += (m: inout SMat6, n: SMat6) { m = m + n }
    static func * (m: SMat6, s: Double) -> SMat6 {
        SMat6(a: m.a * s, b: m.b * s, c: m.c * s, d: m.d * s)
    }
    static func - (m: SMat6, n: SMat6) -> SMat6 {
        SMat6(a: m.a + (n.a * -1), b: m.b + (n.b * -1),
              c: m.c + (n.c * -1), d: m.d + (n.d * -1))
    }
}

/// `outer(u, u)` scaled: the rank-one update that carries the joint's torque
/// down the tree.
@inline(__always) func outer(_ u: SVec6) -> SMat6 {
    SMat6(a: Mat3(columns: (u.w * u.w.x, u.w * u.w.y, u.w * u.w.z)),
          b: Mat3(columns: (u.w * u.v.x, u.w * u.v.y, u.w * u.v.z)),
          c: Mat3(columns: (u.v * u.w.x, u.v * u.w.y, u.v * u.w.z)),
          d: Mat3(columns: (u.v * u.v.x, u.v * u.v.y, u.v * u.v.z)))
}

/// The motion cross-product matrix: `crm(v) * u == v x_m u`.
@inline(__always) func crm(_ v: SVec6) -> SMat6 {
    SMat6(a: crossMat(v.w), b: Mat3.zero,
          c: crossMat(v.v), d: crossMat(v.w))
}

/// The force cross-product matrix, `v x_f* f`, which is `-crm(v)^T`
/// (Featherstone eq. 2.34).
@inline(__always) func crf(_ v: SVec6) -> SMat6 {
    crm(v).transpose * -1
}

/// Solve `m x = b` for a 6x6 system, by Gaussian elimination with partial
/// pivoting. Only the free root ever needs it, once per substep.
func solve(_ m: SMat6, _ b: SVec6) -> SVec6 {
    var a = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 6)
    for i in 0..<3 {
        a[i][0] = m.a.c0[i]; a[i][1] = m.a.c1[i]; a[i][2] = m.a.c2[i]
        a[i][3] = m.b.c0[i]; a[i][4] = m.b.c1[i]; a[i][5] = m.b.c2[i]
        a[i + 3][0] = m.c.c0[i]; a[i + 3][1] = m.c.c1[i]; a[i + 3][2] = m.c.c2[i]
        a[i + 3][3] = m.d.c0[i]; a[i + 3][4] = m.d.c1[i]; a[i + 3][5] = m.d.c2[i]
    }
    var rhs = [b.w.x, b.w.y, b.w.z, b.v.x, b.v.y, b.v.z]

    for col in 0..<6 {
        var pivot = col
        var best = abs(a[col][col])
        for r in (col + 1)..<6 where abs(a[r][col]) > best {
            best = abs(a[r][col]); pivot = r
        }
        if pivot != col {
            a.swapAt(pivot, col)
            rhs.swapAt(pivot, col)
        }
        let p = a[col][col]
        if p == 0 { continue }
        for r in (col + 1)..<6 {
            let f = a[r][col] / p
            if f == 0 { continue }
            for c2 in col..<6 { a[r][c2] -= f * a[col][c2] }
            rhs[r] -= f * rhs[col]
        }
    }
    var x = [Double](repeating: 0, count: 6)
    for col in stride(from: 5, through: 0, by: -1) {
        var s = rhs[col]
        for c2 in (col + 1)..<6 { s -= a[col][c2] * x[c2] }
        x[col] = a[col][col] == 0 ? 0 : s / a[col][col]
    }
    return SVec6(Vec3(x[0], x[1], x[2]), Vec3(x[3], x[4], x[5]))
}

// MARK: - Quaternions

/// (w, x, y, z), the same order the asset stores them in.
typealias Quat = SIMD4<Double>

@inline(__always) func quatToMat(_ q: Quat) -> Mat3 {
    let n = (q * q).sum().squareRoot()
    if n < 1e-12 { return .identity }
    let w = q.x / n, x = q.y / n, y = q.z / n, z = q.w / n
    return Mat3(rows: [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])
}

/// A rotation matrix as the quaternion that is *the same rotation*, in
/// SceneKit's (x, y, z, w) order — the inverse of `quatToMat`.
///
/// Shepperd's method: four branches, the same formula read from whichever of
/// the four components is largest, which is what keeps it stable near 180°.
/// The signs are the published ones and they are load-bearing. The version of
/// this that lived in `FlyWorld.swift` returned the **conjugate** — x, y, z
/// negated in the trace branch and w negated in the other three, which is the
/// same rotation as the inverse — so every part of the animal was drawn spun
/// the wrong way about its own origin: the scatter of fragments in
/// `uploads/IMG_2713.png`, while the positions, the asset and the solver were
/// all right to 1e-15 cm (`tools/audit_meshes.py`).
/// `FlyDynamicsTests.testAQuaternionFromAMatrixIsTheRotationItself` is the test.
@inline(__always) func quatFromMat(_ m: Mat3)
    -> (x: Double, y: Double, z: Double, w: Double) {
    // `c0`, `c1`, `c2` are the *columns* of the operator, so row I of it is
    // (c0[I], c1[I], c2[I]) — hence the naming: rIJ is row I, column J.
    let r00 = m.c0.x, r10 = m.c0.y, r20 = m.c0.z
    let r01 = m.c1.x, r11 = m.c1.y, r21 = m.c1.z
    let r02 = m.c2.x, r12 = m.c2.y, r22 = m.c2.z
    let trace = r00 + r11 + r22
    if trace > 0 {
        let s = (trace + 1).squareRoot() * 2
        return ((r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, s / 4)
    } else if r00 > r11 && r00 > r22 {
        let s = (1 + r00 - r11 - r22).squareRoot() * 2
        return (s / 4, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s)
    } else if r11 > r22 {
        let s = (1 + r11 - r00 - r22).squareRoot() * 2
        return ((r01 + r10) / s, s / 4, (r12 + r21) / s, (r02 - r20) / s)
    } else {
        let s = (1 + r22 - r00 - r11).squareRoot() * 2
        return ((r02 + r20) / s, (r12 + r21) / s, s / 4, (r10 - r01) / s)
    }
}

@inline(__always) func quatMul(_ a: Quat, _ b: Quat) -> Quat {
    Quat(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
         a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
         a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
         a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x)
}

@inline(__always) func quatFromAxisAngle(_ axis: Vec3, _ angle: Double) -> Quat {
    let n = axis.length
    if n < 1e-12 { return Quat(1, 0, 0, 0) }
    let a = axis / n
    let s = sin(angle / 2)
    return Quat(cos(angle / 2), a.x * s, a.y * s, a.z * s)
}

/// Exponential map: a rotation vector (in whatever frame it is given in) as a
/// quaternion.
@inline(__always) func quatFromVec(_ v: Vec3) -> Quat {
    let angle = v.length
    if angle < 1e-12 { return Quat(1, 0, 0, 0) }
    return quatFromAxisAngle(v / angle, angle)
}

/// The Plücker transform from a parent frame to a child frame.
///
/// The linear block is the part that catches people out: the cross term is
/// `(E r) x (E w)` — the *child*-frame angular velocity — so the lower-left
/// block is `-(E r)x E` and not `-(E r)x`. Get it wrong and everything still
/// runs, and the answer is quietly wrong by a factor that depends on the
/// joint's orientation.
@inline(__always) func xform(_ E: Mat3, _ r: Vec3) -> SMat6 {
    SMat6(a: E, b: Mat3.zero,
          c: crossMat(E * r) * (E * -1), d: E)
}

/// The 6x6 spatial inertia about the body-frame origin.
///
/// `inertiaDiag` is the diagonal in the principal frame that `iquat` rotates
/// into the body frame, so the rotation is applied here rather than assumed
/// to be the identity.
func spatialInertia(mass: Double, com: Vec3, inertiaDiag: Vec3, iquat: Quat) -> SMat6 {
    let R = quatToMat(iquat)
    let Ic = R * Mat3(columns: (Vec3(inertiaDiag.x, 0, 0),
                               Vec3(0, inertiaDiag.y, 0),
                               Vec3(0, 0, inertiaDiag.z))) * R.transpose
    let cx = crossMat(com)
    let mcx = cx * mass
    return SMat6(a: Ic + (cx * cx.transpose) * mass,
                 b: mcx,
                 c: mcx.transpose,
                 d: Mat3(columns: (Vec3(mass, 0, 0), Vec3(0, mass, 0), Vec3(0, 0, mass))))
}

// MARK: - The asset

/// `build/fly_body.json`, as written by `tools/build_body.py`. Field names
/// follow the file: the decoder converts from the file's snake_case.
struct FlyBodyAsset: Decodable {
    struct Units: Decodable {
        let length: String?
        let mass: String?
        let time: String?
        let angle: String?
        let torque: String?
        let note: String?
    }

    struct Body: Decodable {
        let index: Int
        let name: String
        let parentIndex: Int
        let mass: Double
        let com: [Double]
        let inertia: [Double]
        let iquat: [Double]
        let pos: [Double]
        let quat: [Double]
    }

    struct Joint: Decodable {
        let index: Int
        let name: String
        let kind: String
        let body: String
        let axis: [Double]
        let range: [Double]
        let limited: Bool
        let stiffness: Double
        let springRef: Double?
        let damping: Double
        let armature: Double
    }

    struct Muscle: Decodable {
        let joint: String
        let holdTorque: Double?
        let maxTorque: Double
        let maxTorqueSource: String?
        let passiveStiffness: Double?
        let optimalAngle: Double?
        let span: Double?
        let range: [Double]
        let springRef: Double?
        let damping: Double?
        let armature: Double?
    }

    struct CollisionGeom: Decodable {
        let name: String
        let body: String
        let type: String
        let size: [Double]
        let pos: [Double]
        let quat: [Double]
    }

    struct VisualGeom: Decodable {
        let name: String
        let body: String
        let pos: [Double]
        let quat: [Double]
        let mesh: Int?
        let rgba: [Double]?
    }

    struct Organ: Decodable {
        let joint: String?
        let group: String?
    }

    struct Leg: Decodable {
        /// One of the three joints per leg the cord can address, with the
        /// standing angle and the anatomical range the organ's drive is scaled
        /// over.
        struct Joint: Decodable {
            let name: String
            let rest: Double
            let range: [Double]
        }

        /// A muscle pool as the connectome carries it: which joint it moves,
        /// which way, and the name `SimulationEngine.groupRate()` answers to.
        /// The group name is written by `tools/motor_pools.py` — the same
        /// function that names the group inside `flybanc.bin` — so the two
        /// cannot disagree about what a pool is called.
        struct Pool: Decodable {
            let joint: String
            let action: String
            /// +1: a bigger pool rate opens this joint's qpos.
            let qposSign: Int
            let group: String
        }

        let part: String
        let segment: String
        let side: String
        let feetBodies: [String]
        let joints: [String: Joint]
        let pools: [String: Pool]
        let organs: [String: Organ]
    }

    struct Meshes: Decodable {
        let nVertex: Int
        let nFace: Int
        let bytes: Int
    }

    let units: Units
    let gravity: [Double]
    let timestepS: Double
    let floorZ: Double
    let stanceRootZ: Double
    let bodies: [Body]
    let joints: [Joint]
    let muscles: [String: Muscle]
    let collision: [CollisionGeom]
    let visual: [VisualGeom]
    let legs: [String: Leg]
    let meshes: Meshes
    let stanceQ: [String: Double]
    /// The excitation that holds the stance, measured by `tools/stand_body.py`
    /// *by standing the animal up in this solver* — which is the only way to
    /// get it right for this floor. Absent in an asset that has not been
    /// through that tool, in which case `holdExcitation()` inverts the muscle
    /// curve at the MJCF stance instead.
    let stanceExcitation: [String: Double]?
}

extension FlyBodyAsset {
    /// The body asset out of the app bundle, where `tools/pack_body.py` puts
    /// it next to the meshes.
    static func loadFromBundle(_ bundle: Bundle = .main) throws -> FlyBodyAsset {
        let url = bundle.url(forResource: "fly_body", withExtension: "json")
            ?? bundle.url(forResource: "fly_body", withExtension: "json",
                          subdirectory: "World")
            ?? bundle.url(forResource: "World/fly_body", withExtension: "json")
        guard let url else { throw FlyBodyLoadError.missing("fly_body.json") }
        do {
            let decoder = JSONDecoder()
            // The asset is written in the file's own snake_case, which is the
            // same spelling the Python reads. One convention, one place it is
            // translated.
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(
                FlyBodyAsset.self, from: try Data(contentsOf: url))
        } catch {
            throw FlyBodyLoadError.decode(error.localizedDescription)
        }
    }
}

/// The asset from the bundle, or a readable explanation of why not.
enum FlyBodyLoadError: Error, LocalizedError {
    case missing(String)
    case decode(String)

    var errorDescription: String? {
        switch self {
        case .missing(let what):
            return "\(what) is not in the app bundle — run tools/build_body.py "
                 + "and tools/pack_body.py before building"
        case .decode(let why):
            return "fly_body.json could not be read: \(why)"
        }
    }
}

// MARK: - The floor's parameters

/// The contact model, fitted to the thing it must reproduce and then tested
/// by whether the animal stands (docs/ASSUMPTIONS.md #21). The fit: the fly
/// weighs m*g = 0.966 g cm/s^2, and at its stance its own contacts carry that
/// weight at a total penetration of 6.36e-03 cm, so a penalty spring that
/// carries the same weight at the same depth is k = 150.
///
/// The damper is small *because it is explicit*: at dt = 1e-4 s a coefficient
/// b damps a mass m through the factor b*dt/m, and the lightest thing that
/// touches the floor is a 1e-6 g cm^2 leg joint, so b*dt/m must stay under 1.
/// Measured (tools/fly_aba.py): 0.5 is a runaway that throws the animal clear
/// of the floor; 0.02 and below stand. The limit worth remembering is
/// b < m*2/dt.
enum FlyContact {
    static let stiffness = 150.0     // force units per cm of penetration
    static let damping = 0.005       // force units per cm/s
    static let friction = 0.5        // fly tarsi on clean glass
    static let slop = 3.3e-5         // cm: contact begins this far above the
                                     // floor, so the resting penetration lands
                                     // where MuJoCo's soft contact puts it
}

/// Muscle shortening speed: fly leg sarcomeres shorten at ~10-20 rad/s at the
/// joint. The exact value changes the velocity term by a few percent, so it
/// is declared rather than tuned (docs/ASSUMPTIONS.md #22).
enum FlyMuscle {
    static let vMax = 20.0           // rad/s
}

// MARK: - The solver

/// The animal, a hundred microseconds at a time.
///
/// `torque` is per hinge joint, in the model's own units, and it comes from
/// the muscles; this class is never told what angle a joint should be at.
final class FlyDynamics {

    let asset: FlyBodyAsset
    let nBody: Int
    let nj: Int
    let root: Int
    let hasFreeRoot: Bool
    let dtDefault: Double
    /// The floor plane. Settable because the tests drop the animal in vacuum
    /// with the floor far below it, the way the Python reference does.
    var floorZ: Double

    // -- fixed model data ---------------------------------------------------

    private let parent: [Int]
    private let bodyPos: [Vec3]
    private let bodyQuat: [Quat]
    private let bodyMass: [Double]
    private let com: [Vec3]
    private let inertia: [SMat6]

    let jointBody: [Int]
    let jointName: [String]
    private let axis: [Vec3]
    private let lo: [Double]
    private let hi: [Double]
    private let limited: [Bool]
    private let damping: [Double]
    private let armature: [Double]
    private let springRef: [Double]
    private let stiffness: [Double]

    /// Muscles, one per joint: what the actuator table says this joint can do
    /// and what it does on its own.
    let maxTorque: [Double]
    let passiveStiffness: [Double]
    let optimal: [Double]
    let span: [Double]
    let holdTorque: [Double]
    /// `-biasprm[1]` per joint: the stiffness of the muscle pair at rest,
    /// straight out of the model's actuators (0 where the file has none).
    /// Against a 1e-6 g cm^2 joint inertia these are 0.4-0.8, and without
    /// them a leg is a free hinge on a 1e-6 pivot, which is not a leg.
    let stanceQ: [Double]
    /// The measured stance excitation, or nil if the asset predates it.
    let measuredStanceExcitation: [Double]?

    /// Which body carries which hinge (-1 for a welded link).
    private let jointOfBody: [Int]
    private let hingesOfBody: [[Int]]
    /// The joint axis as a spatial vector, per body.
    private let S: [SVec6]

    // -- collision ----------------------------------------------------------

    private let geomType: [String]
    private let geomSize: [Vec3]
    private let geomPos: [Vec3]
    private let geomQuat: [Quat]
    private let geomBody: [Int]
    /// leg key -> the collision geoms that count as its feet.
    private let footGeoms: [String: Set<Int>]

    /// Visual (mesh) geoms, in the asset's `visual` order: what the renderer
    /// poses. Kept here so a caller never has to re-derive the offsets.
    let visualBody: [Int]
    let visualPos: [Vec3]
    let visualQuat: [Quat]
    let visualName: [String]

    // -- state --------------------------------------------------------------

    var rootPos = Vec3.zero
    var rootQuat = Quat(1, 0, 0, 0)
    var q: [Double]
    var qd: [Double]
    var vel = Vec3.zero
    var omega = Vec3.zero
    var torque: [Double]

    private(set) var pos: [Vec3]
    private(set) var rot: [Mat3]
    private(set) var gpos: [Vec3]
    private(set) var grot: [Mat3]
    /// Newton of floor force on each leg, in this model's units (g cm/s^2).
    private(set) var footForce: [String: Double]
    private(set) var contacts = 0
    private(set) var bodiesInContact = Set<Int>()

    // -- scratch (allocated once; the app calls step() ten thousand times a
    //    simulated second and nothing here may allocate) --------------------

    private var v: [SVec6]
    private var c: [SVec6]
    private var impact: [SVec6]          // wrench the floor applies, per body
    private var IA: [SMat6]
    private var pA: [SVec6]
    private var U: [SVec6]
    private var D: [Double]
    private var u: [Double]
    private var X: [SMat6]
    private var ci: [SVec6]
    private var a: [SVec6]
    private var welded: [Bool]
    private var tau: [Double]
    private var qdd: [Double]

    // MARK: Loading

    init(asset: FlyBodyAsset) {
        self.asset = asset
        nBody = asset.bodies.count
        root = asset.joints.first(where: { $0.kind == "free" })
            .flatMap { j in asset.bodies.firstIndex(where: { $0.name == j.body }) } ?? 1
        hasFreeRoot = asset.joints.contains { $0.kind == "free" }
        dtDefault = asset.timestepS
        floorZ = asset.floorZ

        parent = asset.bodies.map { $0.parentIndex }
        bodyPos = asset.bodies.map { Vec3($0.pos) }
        bodyQuat = asset.bodies.map { FlyDynamics.quat($0.quat) }
        bodyMass = asset.bodies.map { $0.mass }
        com = asset.bodies.map { Vec3($0.com) }
        inertia = asset.bodies.map {
            spatialInertia(mass: $0.mass, com: Vec3($0.com),
                           inertiaDiag: Vec3($0.inertia),
                           iquat: FlyDynamics.quat($0.iquat))
        }

        let hinges = asset.joints.filter { $0.kind == "hinge" }
        nj = hinges.count
        jointName = hinges.map { $0.name }
        let bodyIndex = Dictionary(uniqueKeysWithValues:
            asset.bodies.enumerated().map { ($0.element.name, $0.offset) })
        jointBody = hinges.map { bodyIndex[$0.body] ?? 0 }
        axis = hinges.map { v in
            let a = Vec3(v.axis)
            let n = a.length
            return n > 0 ? a / n : Vec3(0, 0, 1)
        }
        lo = hinges.map { $0.range.count > 0 ? $0.range[0] : 0 }
        hi = hinges.map { $0.range.count > 1 ? $0.range[1] : 0 }
        limited = hinges.map { $0.limited }
        damping = hinges.map { $0.damping }
        armature = hinges.map { $0.armature }
        springRef = hinges.map { $0.springRef ?? 0 }
        stiffness = hinges.map { $0.stiffness }

        maxTorque = hinges.map { asset.muscles[$0.name]?.maxTorque ?? 0 }
        passiveStiffness = hinges.map { asset.muscles[$0.name]?.passiveStiffness ?? 0 }
        holdTorque = hinges.map { asset.muscles[$0.name]?.holdTorque ?? 0 }

        // Muscle geometry: the asset records the *joint's* range next to each
        // muscle, so the force-length width is the width of that range, and
        // the angle at which the muscle is strongest is the joint's spring
        // reference if it has one and the middle of its range otherwise.
        // flybody publishes no muscle optimal angles — nothing in the MJCF
        // says where any muscle is longest — so this is an APPROXIMATION,
        // declared in docs/ASSUMPTIONS.md.
        // `springRef` through a local: Swift will not let a closure read a
        // property of `self` while the rest of `self` is still being built
        // ("'self' captured by a closure before all members were
        // initialized"), and this pass is the first place the compiler gets
        // to say so — it was hidden behind the type-checker error in `step()`.
        let references = springRef
        optimal = hinges.enumerated().map { k, j in
            if let o = asset.muscles[j.name]?.optimalAngle { return o }
            let ref = references[k]
            if ref != 0 { return ref }
            let r = j.range
            return r.count > 1 ? 0.5 * (r[0] + r[1]) : 0
        }
        span = hinges.enumerated().map { k, j in
            if let s = asset.muscles[j.name]?.span { return s }
            let r = j.range
            let width = r.count > 1 ? abs(r[1] - r[0]) : 0
            return max(1e-3, width)
        }

        stanceQ = hinges.map { asset.stanceQ[$0.name] ?? 0 }
        measuredStanceExcitation = asset.stanceExcitation.map { table in
            hinges.map { table[$0.name] ?? 0 }
        }

        var job = [Int](repeating: -1, count: nBody)
        var hob = [[Int]](repeating: [], count: nBody)
        for (j, b) in jointBody.enumerated() {
            job[b] = j
            hob[b].append(j)
        }
        jointOfBody = job
        hingesOfBody = hob

        var svec = [SVec6](repeating: SVec6.zero, count: nBody)
        for (j, b) in jointBody.enumerated() {
            svec[b].w = axis[j]
        }
        S = svec

        geomType = asset.collision.map { $0.type }
        geomSize = asset.collision.map { Vec3($0.size) }
        geomPos = asset.collision.map { Vec3($0.pos) }
        geomQuat = asset.collision.map { FlyDynamics.quat($0.quat) }
        geomBody = asset.collision.map { bodyIndex[$0.body] ?? 0 }

        var feet: [String: Set<Int>] = [:]
        for (key, leg) in asset.legs {
            var idx = Set<Int>()
            for (g, body) in geomBody.enumerated()
            where leg.feetBodies.contains(asset.bodies[body].name) {
                idx.insert(g)
            }
            feet[key] = idx
        }
        footGeoms = feet

        visualBody = asset.visual.map { bodyIndex[$0.body] ?? 0 }
        visualPos = asset.visual.map { Vec3($0.pos) }
        visualQuat = asset.visual.map { FlyDynamics.quat($0.quat) }
        visualName = asset.visual.map { $0.name }

        pos = [Vec3](repeating: .zero, count: nBody)
        rot = [Mat3](repeating: .identity, count: nBody)
        gpos = [Vec3](repeating: .zero, count: asset.collision.count)
        grot = [Mat3](repeating: .identity, count: asset.collision.count)
        // The stance, at the height the stance was measured at. `stance_q` is
        // the compressed pose the animal's own actuators hold its joints in,
        // and `stance_root_z` is where its root sits while they do it
        // (tools/stand_body.py measures the hold torques there). Starting at
        // z = 0 instead puts the feet 49 um clear of the floor: the contact
        // that is supposed to carry the animal's weight never forms, and the
        // stance servo holds the legs against nothing — the reference solver
        // starts at `stance_root_z` (`FlySim.reset`), and the port now does
        // too. Found by comparing the standing run step by step with the
        // reference: at z = 0 both have **zero** contacts and the same
        // 0.148 rad/s rate, at `stance_root_z` both have **four** and the
        // same 0.227 rad/s — the port was standing in the air.
        rootPos = Vec3(0, 0, asset.stanceRootZ)
        q = stanceQ
        qd = [Double](repeating: 0, count: nj)
        torque = [Double](repeating: 0, count: nj)
        footForce = Dictionary(uniqueKeysWithValues: asset.legs.keys.map { ($0, 0.0) })

        v = [SVec6](repeating: .zero, count: nBody)
        c = [SVec6](repeating: .zero, count: nBody)
        impact = [SVec6](repeating: .zero, count: nBody)
        IA = [SMat6](repeating: .identity, count: nBody)
        pA = [SVec6](repeating: .zero, count: nBody)
        U = [SVec6](repeating: .zero, count: nBody)
        D = [Double](repeating: 1, count: nBody)
        u = [Double](repeating: 0, count: nBody)
        X = [SMat6](repeating: .identity, count: nBody)
        ci = [SVec6](repeating: .zero, count: nBody)
        a = [SVec6](repeating: .zero, count: nBody)
        welded = [Bool](repeating: false, count: nBody)
        tau = [Double](repeating: 0, count: nj)
        qdd = [Double](repeating: 0, count: nj)

        kinematics()
    }

    static func quat(_ a: [Double]) -> Quat {
        a.count >= 4 ? Quat(a[0], a[1], a[2], a[3]) : Quat(1, 0, 0, 0)
    }

    // MARK: State

    /// The pose the animal settles at with its own actuators holding it up,
    /// legs compressed by its weight — and the pose the hold torques were
    /// measured at. `q == nil` is that stance, not zero: starting from the
    /// all-zero keyframe starts the animal at a pose no muscle is balanced
    /// for, and the first thing it does is snap.
    func reset(rootZ: Double? = nil, rootPosition: Vec3? = nil, quat: Quat? = nil,
               q newQ: [Double]? = nil, qd newQd: [Double]? = nil,
               vel newVel: Vec3? = nil, omega newOmega: Vec3? = nil) {
        rootPos = rootPosition ?? Vec3(0, 0, rootZ ?? asset.stanceRootZ)
        rootQuat = quat ?? Quat(1, 0, 0, 0)
        q = newQ ?? stanceQ
        qd = newQd ?? [Double](repeating: 0, count: nj)
        vel = newVel ?? .zero
        omega = newOmega ?? .zero
        torque = [Double](repeating: 0, count: nj)
        for key in footForce.keys { footForce[key] = 0 }
        contacts = 0
        bodiesInContact.removeAll()
        kinematics()
    }

    func kinematics() {
        rot[0] = .identity
        pos[0] = .zero
        for i in 1..<nBody {
            let p = parent[i]
            if i == root && hasFreeRoot {
                pos[i] = rootPos
                rot[i] = quatToMat(rootQuat)
            } else {
                rot[i] = rot[p] * quatToMat(bodyQuat[i])
                pos[i] = pos[p] + rot[p] * bodyPos[i]
            }
            for j in hingesOfBody[i] {
                rot[i] = rot[i] * quatToMat(quatFromAxisAngle(axis[j], q[j]))
            }
        }
        // collision geoms, and the mesh geoms the renderer poses
        for g in 0..<gpos.count {
            let b = geomBody[g]
            grot[g] = rot[b] * quatToMat(geomQuat[g])
            gpos[g] = pos[b] + rot[b] * geomPos[g]
        }
    }

    /// World position and orientation of the mesh geoms, in `visual` order.
    func visualWorld() -> (pos: [Vec3], rot: [Mat3]) {
        var vp = [Vec3](repeating: .zero, count: visualBody.count)
        var vr = [Mat3](repeating: .identity, count: visualBody.count)
        for k in 0..<visualBody.count {
            let b = visualBody[k]
            vr[k] = rot[b] * quatToMat(visualQuat[k])
            vp[k] = pos[b] + rot[b] * visualPos[k]
        }
        return (vp, vr)
    }

    /// The centre of mass, in world coordinates — what gravity acts on.
    func centreOfMass() -> Vec3 {
        var total = 0.0
        var acc = Vec3.zero
        for i in 1..<nBody {
            total += bodyMass[i]
            acc += (pos[i] + rot[i] * com[i]) * bodyMass[i]
        }
        return total > 0 ? acc / total : .zero
    }

    /// The lowest point of any collision geom: how far the animal's feet are
    /// above the floor.
    func lowestPoint() -> Double {
        var low = Double.greatestFiniteMagnitude
        for g in 0..<gpos.count {
            low = min(low, gpos[g].z - support(g))
        }
        return low
    }

    /// Half-height of geom `g` along the world vertical — the support
    /// function of the shape, which is exact for every shape in the asset:
    /// a convex body's lowest point is its centre minus this.
    private func support(_ g: Int) -> Double {
        let sz = geomSize[g]
        let R = grot[g]
        switch geomType[g] {
        case "sphere":
            return sz.x
        case "capsule":
            return sz.x + sz.y * abs(R.c2.z)
        case "ellipsoid":
            // The support along world +z is |semi-axes * (world z in the
            // geom's own frame)|. Written as norm(R * sz) this is the norm of
            // a rotated vector — constant, orientation-free and wrong by most
            // of a leg segment: a 150 um phantom penetration on the tarsus,
            // which at 150 units of stiffness is twice the animal's weight
            // pushing on a 1e-6 g cm^2 joint. Measured against MuJoCo's own
            // plane distance, the form below agrees to 0.2 um on all 74.
            return (Vec3(sz.x * R.row(2).x, sz.y * R.row(2).y, sz.z * R.row(2).z)).length
        case "cylinder":
            let a = abs(R.c2.z)
            return sz.x * (max(0, 1 - a * a)).squareRoot() + sz.y * a
        case "box":
            let r = R.transpose * Vec3.up
            return abs(r.x) * sz.x + abs(r.y) * sz.y + abs(r.z) * sz.z
        default:
            return -Double.greatestFiniteMagnitude
        }
    }

    // MARK: The recursion

    private func velocities() {
        let R = rot[root]
        v[root] = SVec6(R.transpose * omega, R.transpose * vel)
        for j in hingesOfBody[root] { v[root].w += axis[j] * qd[j] }
        if root + 1 < nBody {
            for i in (root + 1)..<nBody {
                let par = parent[i]
                let E = rot[i].transpose * rot[par]
                let Xi = xform(E, bodyPos[i])
                v[i] = Xi * v[par]
                // every hinge of this link, not just the first
                for j in hingesOfBody[i] { v[i].w += axis[j] * qd[j] }
            }
        }
    }

    /// `v x* I v`, minus gravity, minus the floor.
    ///
    /// Gravity is a force on every body here — `[c x f, f]` about the body
    /// origin, in the body's own frame — and *not* an acceleration handed to
    /// the root of the tree. The two are the same for a fixed base and every
    /// textbook writes it the other way round, but they are not the same for
    /// a free root: with gravity as a force, a hand's hinge in free fall needs
    /// no torque and gets none, while the root-acceleration trick quietly asks
    /// the hinges for the torque that holds the fall rigid.
    private func bias() {
        let g = Vec3(asset.gravity)
        for i in 0..<nBody { c[i] = SVec6.zero }
        if nBody > 1 {
            for i in 1..<nBody {
                c[i] = crf(v[i]) * (inertia[i] * v[i])
                let f = (rot[i].transpose * g) * bodyMass[i]
                c[i] -= SVec6(cross(com[i], f), f)
            }
        }
        let wrench = floorContact()
        for i in 0..<nBody { c[i] -= wrench[i] }
    }

    /// The floor, as a penalty contact on every collision geom.
    ///
    /// Returns the wrench each body receives, in that body's own frame — both
    /// halves of it. A world-frame force crossed with a body-frame lever arm
    /// is neither: on a fly standing level it looks harmless, on a leg rotated
    /// 45 degrees the floor pushes the leg sideways, and the animal comes
    /// apart in 2 ms.
    private func floorContact() -> [SVec6] {
        for i in 0..<nBody { impact[i] = SVec6.zero }
        for key in footForce.keys { footForce[key] = 0 }
        contacts = 0
        bodiesInContact.removeAll()

        for g in 0..<gpos.count {
            let h = support(g)
            if h == -Double.greatestFiniteMagnitude { continue }
            let pen = (floorZ + FlyContact.slop) - (gpos[g].z - h)
            if pen <= 0 { continue }

            let body = geomBody[g]
            let Rb = rot[body]
            let origin = pos[body]
            let point = gpos[g]
            let vb = v[body]
            let wWorld = Rb * vb.w
            let vOrigin = Rb * vb.v + cross(wWorld, origin)
            let vPoint = vOrigin + cross(wWorld, point - origin)

            var normal = FlyContact.stiffness * pen - FlyContact.damping * vPoint.z
            if normal < 0 { normal = 0 }
            let tang = Vec3(vPoint.x, vPoint.y, 0)
            let speed = tang.length
            var F = Vec3(0, 0, normal)
            if speed > 1e-12 {
                let mu = FlyContact.friction * min(1.0, speed / 1e-3)
                F += (tang / speed) * (-mu * normal)
            }
            let local = Rb.transpose * (point - origin)
            let Fb = Rb.transpose * F
            impact[body] += SVec6(cross(local, Fb), Fb)

            contacts += 1
            bodiesInContact.insert(body)
            for (key, geoms) in footGeoms where geoms.contains(g) {
                footForce[key, default: 0] += normal
            }
        }
        return impact
    }

    // MARK: One substep

    /// A joint's own constants, and the body it hangs on.
    ///
    /// Inspection, not simulation: the tests and the Linux driver in `local/`
    /// compare these against `tools/fly_aba.py`'s arrays field by field, which
    /// is the only way to tell a modelling difference from a decoding one.
    struct JointFacts {
        let name: String
        let body: Int
        let axis: Vec3
        let stiffness: Double
        let springRef: Double
        let damping: Double
        let armature: Double
        let lo: Double
        let hi: Double
        let limited: Bool
        let mass: Double
        let com: Vec3
        let inertia: SMat6
    }

    func facts(_ j: Int) -> JointFacts {
        let b = jointBody[j]
        return JointFacts(name: jointName[j], body: b, axis: axis[j],
                          stiffness: stiffness[j], springRef: springRef[j],
                          damping: damping[j], armature: armature[j],
                          lo: lo[j], hi: hi[j], limited: limited[j],
                          mass: bodyMass[b], com: com[b], inertia: inertia[b])
    }

    /// The backward pass's working values, after one `step`, for the
    /// cross-language bisection driver in `local/`. A body's IA and pA only
    /// ever grow *before* it is processed, so what is stored at the end of the
    /// pass is what that body's own step used.
    var nb: Int { nBody }
    func passX(_ i: Int) -> SMat6 { X[i] }
    func passIA(_ i: Int) -> SMat6 { IA[i] }
    func passPA(_ i: Int) -> SVec6 { pA[i] }
    func passCi(_ i: Int) -> SVec6 { ci[i] }
    func passU(_ i: Int) -> SVec6 { U[i] }
    func passD(_ i: Int) -> Double { D[i] }
    func passUu(_ i: Int) -> Double { u[i] }
    func passTau() -> [Double] { tau }
    func passV(_ i: Int) -> SVec6 { v[i] }
    var ng: Int { gpos.count }
    func geomWorld(_ g: Int) -> (pos: Vec3, rot: Mat3, body: Int, type: String, size: Vec3, h: Double) {
        (gpos[g], grot[g], geomBody[g], geomType[g], geomSize[g], support(g))
    }
    func passC(_ i: Int) -> SVec6 { c[i] }

    /// A link's world pose, as `kinematics()` computed it.
    func linkWorld(_ i: Int) -> (pos: Vec3, rot: Mat3) { (pos[i], rot[i]) }

    /// The parent link of a body, and the fixed pose it hangs at.
    func linkFacts(_ i: Int) -> (parent: Int, pos: Vec3, quat: Quat, mass: Double) {
        (parent[i], bodyPos[i], bodyQuat[i], bodyMass[i])
    }

    /// How many times a hard stop has killed a joint's rate in this run.
    ///
    /// The golden trace is a violent vacuum tumble whose joints reach hundreds
    /// of radians per second, and 141 of those rate-kills happen inside it
    /// (measured in the Python reference). If the phone's solver and the
    /// reference ever disagree, this count is the first thing to compare: a
    /// solver that is not clamping at all, or clamping the wrong joints, shows
    /// up here long before it shows up in a tolerance.
    private(set) var limitStops = 0

    /// How many joints are sitting on a hard stop right now.
    func jointsAtLimit() -> Int {
        var n = 0
        for j in 0..<nj where limited[j] {
            if q[j] <= lo[j] + 1e-12 || q[j] >= hi[j] - 1e-12 { n += 1 }
        }
        return n
    }

    /// One substep: semi-implicit Euler with the joint dampers folded into the
    /// articulated-body inertia, which is the same thing MuJoCo does and is
    /// what makes a damper on a joint this light integrable at all.
    ///
    /// Returns the joint accelerations, which the app shows and the tests
    /// check; nobody should ever integrate them by hand.
    @discardableResult
    func step(dt: Double, torque commands: [Double]) -> [Double] {
        torque = commands
        kinematics()
        velocities()
        bias()

        for j in 0..<nj {
            tau[j] = torque[j] - stiffness[j] * (q[j] - springRef[j])
        }

        for i in 0..<nBody {
            IA[i] = inertia[i]
            pA[i] = c[i]
            welded[i] = false
            D[i] = 1
            u[i] = 0
            U[i] = .zero
            ci[i] = .zero
            a[i] = .zero
        }

        // Backward pass. The velocity-product acceleration of each joint,
        // `ci = v x (S qd)`, belongs in *both* passes: the articulated-body
        // inertia cancels part of it going down the tree and the rest comes
        // back in the acceleration.
        if nBody - 1 >= root {
            for i in stride(from: nBody - 1, through: root, by: -1) {
                let par = parent[i]
                let E = rot[i].transpose * rot[par]
                let Xi = xform(E, bodyPos[i])
                X[i] = Xi
                if i == root && hasFreeRoot { continue }
                let j = jointOfBody[i]
                if j < 0 {
                    // a link with no joint of its own is welded to its parent:
                    // it contributes its inertia and its bias, and nothing else
                    D[i] = 1
                    u[i] = 0
                    U[i] = .zero
                    welded[i] = true
                    IA[par] += Xi.transpose * IA[i] * Xi
                    pA[par] += Xi.transpose * pA[i]
                    continue
                }
                let Si = S[i]
                let Ui = IA[i] * Si
                // `v x (S qd)`, with the joint's own contribution worked out
                // first. As one expression — `crm(v[i]) * (Si * qd[j])` — the
                // type checker has to solve a `SMat6 * (SVec6 * Double)`
                // against every `*` declared in this file and gives up on it
                // ("binary operator '*' cannot be applied to two 'SMat6'
                // operands"), which is a parsing complaint and not a maths
                // one. The product is identical; only the nesting is gone.
                let Siqd = Si * qd[j]
                let cii = crm(v[i]) * Siqd
                ci[i] = cii
                let Di = dot(Si, Ui) + armature[j] + damping[j] * dt
                // Featherstone 7.42: the articulated-body inertia *loses* the
                // rank-one term the joint cannot see, I^a = I^A - U U^T / D.
                // This read `- (outer(Ui) * (-1 / Di))`, which is I^A + U U^T / D
                // — the wrong sign on the only term that carries a child's
                // inertia into its parent. The reference solver, judged against
                // MuJoCo's own step on the same state, reproduces MuJoCo's rates
                // after one step to 1.7e-04 rad/s out of 68; this line made the
                // phone solver 1620x worse than that and put a 4e-04 rad error
                // into the first step of the golden trace.
                let Ia = IA[i] - outer(Ui) * (1 / Di)
                let pAe = pA[i] + Ia * cii
                let ui = tau[j] - dot(Si, pAe) - damping[j] * qd[j]
                U[i] = Ui
                D[i] = Di
                u[i] = ui
                IA[par] += Xi.transpose * Ia * Xi
                // same reason as `Siqd` above: the scaled joint vector comes
                // out of the nested expression before it is added in
                let correction = U[i] * (ui / Di)
                pA[par] += Xi.transpose * (pAe + correction)
            }
        }

        // The root: a free root is a six-DOF joint whose parent is the world,
        // so its acceleration is whatever leaves the accumulated wrench at
        // zero, -IA^-1 pA. A welded root does not accelerate at all.
        if hasFreeRoot {
            a[root] = solve(IA[root], pA[root]) * -1
        }

        for j in 0..<nj { qdd[j] = 0 }
        let first = hasFreeRoot ? root + 1 : root
        if first < nBody {
            for i in first..<nBody {
                let par = parent[i]
                let acc = par >= 1 ? X[i] * a[par] : SVec6.zero
                let j = jointOfBody[i]
                if j < 0 {
                    a[i] = acc
                    continue
                }
                qdd[j] = (u[i] - dot(U[i], acc + ci[i])) / D[i]
                a[i] = acc + ci[i] + S[i] * qdd[j]
            }
        }

        // ---- integrate -----------------------------------------------------
        // `a` is the spatial acceleration; the acceleration of the body-fixed
        // origin, which is the one that integrates, differs by omega x v.
        let Rb = rot[root]
        let alphaWorld = Rb * a[root].w
        let accWorld = Rb * a[root].v + cross(omega, vel)
        omega += alphaWorld * dt
        vel += accWorld * dt
        rootPos += vel * dt
        rootQuat = quatMul(rootQuat, quatFromVec(Rb.transpose * omega * dt))
        let qn = (rootQuat * rootQuat).sum().squareRoot()
        if qn > 0 { rootQuat /= qn }
        for j in 0..<nj {
            qd[j] += dt * qdd[j]
            q[j] += dt * qd[j]
        }

        // Joint limits here are hard stops — clamp, kill the rate. MuJoCo's
        // are soft constraints solved implicitly, so the two are only
        // comparable with them off (docs/ASSUMPTIONS.md #23).
        for j in 0..<nj where limited[j] {
            if q[j] < lo[j] {
                if qd[j] < 0 { qd[j] = 0; limitStops += 1 }
                q[j] = lo[j]
            } else if q[j] > hi[j] {
                if qd[j] > 0 { qd[j] = 0; limitStops += 1 }
                q[j] = hi[j]
            }
        }
        return qdd
    }

    // MARK: Muscles

    /// Excitation in [-1, 1] per joint (positive opens qpos) to joint torque.
    ///
    /// Force-based, not position-based: an activation becomes a force, the
    /// force becomes a torque, and the joint does whatever the physics says.
    ///
    /// The velocity term is applied *per direction*, because a joint has an
    /// antagonist pair and they do not do the same thing at the same time.
    /// The muscle that is shortening (pulling the joint the way it is already
    /// going) weakens with speed and reaches zero at vMax; the one that is
    /// lengthening (pulling against the motion) is loaded and gets *stronger*,
    /// up to the 1.8 of a stretched fibre. Applying one shortening curve to
    /// both removes the braking torque: past vMax the factor is zero for
    /// either sign of command, so a joint that is already moving fast cannot
    /// be stopped by anything but its own damper, and the animal flies apart
    /// in 200 microseconds.
    ///
    /// The per-direction split was right and the *curve* was not: the
    /// lengthening branch was `(1 + s)/(1 - 2s)`, which has a pole at half of
    /// vMax and is negative for every faster stretch, so the clamp turned it
    /// into zero exactly where the brake was needed. `hillVelocityFactor` is
    /// the same idea without the pole, and
    /// `FlyDynamicsTests.testTheMuscleModelIsForceBasedAndBraked` is the test
    /// that found it — a test that had never run in CI until the assets that
    /// skipped it were packed into the bundle (reports/step6_loop.md).
    func muscleTorque(q qq: [Double], qd qqd: [Double], excitation: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: nj)
        for j in 0..<nj {
            let fl = exp(-pow((qq[j] - optimal[j]) / (0.5 * span[j]), 2))
            let ratio = qqd[j] / FlyMuscle.vMax
            let e = excitation.indices.contains(j) ? excitation[j] : 0
            let pullUp = min(1, max(0, e)) * hillVelocityFactor(ratio)
            let pullDown = min(1, max(0, -e)) * hillVelocityFactor(-ratio)
            let active = maxTorque[j] * fl * (pullUp - pullDown)
            out[j] = active - passiveStiffness[j] * qq[j]
        }
        return out
    }

    /// Hill's force-velocity, in the sense the joint needs it.
    ///
    /// `s` is the speed at which *this* muscle is shortening, in units of
    /// vMax: positive while it shortens, negative while it is stretched.
    /// Isometric (s = 0) is 1; s = 1 is the speed at which it can no longer
    /// pull at all, and past it the pulling muscle is exhausted (0) while the
    /// stretched one keeps getting stronger, saturating at the 1.8 of a
    /// stretched fibre.
    @inline(__always) func hillVelocityFactor(_ s: Double) -> Double {
        if s <= 0 { return min(1.8, 1 - s) }
        return min(1.8, max(0, (1 - s) / (1 + 2 * s)))
    }

    /// The excitation that holds the animal up: `hold_torque` is the torque
    /// the animal's own actuators needed to stand, measured by inverse
    /// dynamics at the stance pose, and the muscle's force-length curve is
    /// what turns excitation into torque — at a wing joint whose rest angle
    /// is 1.5 rad the head of the curve is far from the stance, so `fl` at
    /// the stance is a fraction of one. Inverting the curve here keeps the
    /// number honest.
    ///
    /// This is *muscle tone*, not behaviour: it is the posture the animal is
    /// calibrated to hold, and everything it does on top of that comes out of
    /// the nerve cord.
    func holdExcitation() -> [Double] {
        if let measured = measuredStanceExcitation { return measured }
        var out = [Double](repeating: 0, count: nj)
        for j in 0..<nj {
            let fl = exp(-pow((stanceQ[j] - optimal[j]) / (0.5 * span[j]), 2))
            var denom = maxTorque[j] * fl
            if denom < 1e-12 { denom = 1e-12 }
            let wanted = holdTorque[j] + passiveStiffness[j] * stanceQ[j]
            out[j] = min(1, max(-1, wanted / denom))
        }
        return out
    }

    /// Excitation gains for holding a pose, sized by the step and the joint.
    ///
    /// This is *test harness and start-up*, not behaviour: nothing in the app
    /// holds a pose, the app's excitation comes out of the connectome. It says
    /// what the discretisation allows: a joint whose inertia is the armature
    /// (1e-6 g cm^2) and whose muscle makes O(1) torque cannot be pushed
    /// faster than omegaDt/dt or damped harder than kdDt*D/dt without the
    /// servo itself becoming the instability.
    func stanceServo(dt: Double, omegaDt: Double = 0.1,
                     kdDt: Double = 0.5) -> (kp: [Double], kd: [Double]) {
        var kp = [Double](repeating: 0, count: nj)
        var kd = [Double](repeating: 0, count: nj)
        for j in 0..<nj {
            let i = armature[j] > 0 ? armature[j] : 1e-6
            let cap = maxTorque[j] > 0 ? maxTorque[j] : 1.0
            kp[j] = omegaDt * omegaDt * i / (dt * dt * cap)
            kd[j] = kdDt * i / (dt * cap)
        }
        return (kp, kd)
    }

    /// Stand the animal up before measuring it.
    ///
    /// The stance is a stiff mode — a 1e-6 g cm^2 leg against a contact that
    /// carries its weight — and at dt = 1e-4 s the initial penetration error
    /// of a few tens of microns is resolved as a 250 rad/s kick that throws
    /// the animal clear of the floor. At a tenth of that step the same pose
    /// settles. The app pays this once, at start-up, and then runs at the
    /// model's own timestep.
    func settle(ms: Double = 20.0, dt: Double = 1e-5, rampMS: Double = 50.0,
                posture: [Double]? = nil) {
        let base = posture ?? holdExcitation()
        let steps = max(1, Int((ms / 1000.0 / dt).rounded()))
        let rampSteps = max(1, Int((rampMS / 1000.0 / dt).rounded()))
        let (kp, kd) = stanceServo(dt: dt)
        for k in 1...steps {
            let ramp = min(1.0, Double(k) / Double(rampSteps))
            var exc = [Double](repeating: 0, count: nj)
            for j in 0..<nj {
                let e = ramp * base[j] + kp[j] * (stanceQ[j] - q[j]) - kd[j] * qd[j]
                exc[j] = min(1, max(-1, e))
            }
            step(dt: dt, torque: muscleTorque(q: q, qd: qd, excitation: exc))
        }
    }
}
