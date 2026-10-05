// A local driver for the golden-trace test on Linux.
//
// The same replay FlyBrainTests runs, with the numbers printed instead of
// asserted, so the divergence can be localised here and not in a 45-minute CI
// cycle. Compile:
//
//   swiftc -O FlyBrain/Sources/FlyDynamics.swift local/GoldenMain.swift -o /tmp/golden
//   /tmp/golden build/fly_body.json build/fly_golden.json
//
import Foundation

struct Golden: Decodable {
    struct Torque: Decodable { let amplitude: [Double]; let periodS: Double }
    struct State: Decodable {
        let q: [Double]; let qd: [Double]; let v: [Double]; let omega: [Double]
        let rootPos: [Double]?; let rootQuat: [Double]?; let limitStops: Int?
    }
    struct Sample: Decodable { let step: Int; let q: [Double]; let rootPos: [Double] }
    let dt: Double
    let steps: Int
    let floorZ: Double
    let torque: Torque
    let initial: State
    let expectedFinal: State
    let trace: [Sample]?
    let tolerance: Double
}

let args = CommandLine.arguments
let bodyPath = args.count > 1 ? args[1] : "build/fly_body.json"
let goldenPath = args.count > 2 ? args[2] : "build/fly_golden.json"

let dec = JSONDecoder()
dec.keyDecodingStrategy = .convertFromSnakeCase
let asset = try dec.decode(FlyBodyAsset.self, from: Data(contentsOf: URL(fileURLWithPath: bodyPath)))
let g = try dec.decode(Golden.self, from: Data(contentsOf: URL(fileURLWithPath: goldenPath)))

let body = FlyDynamics(asset: asset)
body.floorZ = g.floorZ
print("nj=\(body.nj) steps=\(g.steps) dt=\(g.dt) tol=\(g.tolerance)")
body.reset(rootZ: 0,
           rootPosition: Vec3(g.initial.rootPos ?? [0, 0, 0]),
           quat: FlyDynamics.quat(g.initial.rootQuat ?? [1, 0, 0, 0]),
           q: g.initial.q, qd: g.initial.qd,
           vel: Vec3(g.initial.v), omega: Vec3(g.initial.omega))
body.kinematics()

// A one-step mode: print q and qd after a single step, so the two solvers can
// be compared joint by joint instead of in a table of tolerances.
// `facts`: every constant the two solvers must agree on, in a fixed order, so
// a disagreement can be diffed instead of guessed. One line per joint:
//
//   F j name body parent axis.xyz k ref c arm mass com.xyz pos.xyz quat.wxyz I(36)
//
if args.count > 3 && args[3] == "facts" {
    func nums(_ v: [Double]) -> String {
        v.map { String(format: "%.17e", $0) }.joined(separator: " ")
    }
    for j in 0..<body.nj {
        let f = body.facts(j)
        let l = body.linkFacts(f.body)
        let m = f.inertia
        // Row-major over the whole 6x6, the same order numpy's reshape gives
        // the Python side — the first version of this printed the blocks
        // column-major and every comparison after column 3 was a transpose.
        var inertia: [Double] = []
        for r in 0..<3 {
            inertia += [m.a.row(r)[0], m.a.row(r)[1], m.a.row(r)[2],
                        m.b.row(r)[0], m.b.row(r)[1], m.b.row(r)[2]]
        }
        for r in 0..<3 {
            inertia += [m.c.row(r)[0], m.c.row(r)[1], m.c.row(r)[2],
                        m.d.row(r)[0], m.d.row(r)[1], m.d.row(r)[2]]
        }
        print("F \(j) \(f.name) \(f.body) \(l.parent) "
              + nums([f.axis.x, f.axis.y, f.axis.z])
              + " " + nums([f.stiffness, f.springRef, f.damping, f.armature])
              + " " + nums([f.mass, f.com.x, f.com.y, f.com.z])
              + " " + nums([l.pos.x, l.pos.y, l.pos.z])
              + " " + nums([l.quat.x, l.quat.y, l.quat.z, l.quat.w])
              + " " + nums(inertia))
    }
    exit(0)
}

// `kin`: every link's world position and rotation, to compare the forward
// kinematics itself before blaming the dynamics.
if args.count > 3 && args[3] == "kin" {
    body.kinematics()
    func nums(_ v: [Double]) -> String { v.map { String(format: "%.17e", $0) }.joined(separator: " ") }
    for (i, b) in asset.bodies.enumerated() {
        let w = body.linkWorld(i)
        let r = w.rot
        print("K \(i) \(b.name) " + nums([w.pos.x, w.pos.y, w.pos.z]) + " "
              + nums([r.c0.x, r.c0.y, r.c0.z, r.c1.x, r.c1.y, r.c1.z, r.c2.x, r.c2.y, r.c2.z]))
    }
    exit(0)
}

// `pass`: the backward pass's own numbers, body by body, so a disagreement in
// the recursion can be read directly instead of inferred from a joint angle.
func dumpPass() {
    let nums: ([Double]) -> String = { $0.map { String(format: "%.17e", $0) }.joined(separator: " ") }
    for i in 0..<body.nb {
        let x = body.passX(i), ia = body.passIA(i), pa = body.passPA(i)
        let ci = body.passCi(i), u6 = body.passU(i)
        var xf: [Double] = []
        for r in 0..<3 { xf += [x.a.row(r)[0], x.a.row(r)[1], x.a.row(r)[2], x.b.row(r)[0], x.b.row(r)[1], x.b.row(r)[2]] }
        for r in 0..<3 { xf += [x.c.row(r)[0], x.c.row(r)[1], x.c.row(r)[2], x.d.row(r)[0], x.d.row(r)[1], x.d.row(r)[2]] }
        print("P \(i) " + nums(xf) + " " + nums([ci.w.x, ci.w.y, ci.w.z, ci.v.x, ci.v.y, ci.v.z])
              + " " + nums([body.passD(i), body.passUu(i)])
              + " " + nums([u6.w.x, u6.w.y, u6.w.z, u6.v.x, u6.v.y, u6.v.z])
              + " " + nums({ var v: [Double] = []
                    for r in 0..<3 {
                        v += [ia.a.row(r)[0], ia.a.row(r)[1], ia.a.row(r)[2],
                              ia.b.row(r)[0], ia.b.row(r)[1], ia.b.row(r)[2]]
                    }
                    for r in 0..<3 {
                        v += [ia.c.row(r)[0], ia.c.row(r)[1], ia.c.row(r)[2],
                              ia.d.row(r)[0], ia.d.row(r)[1], ia.d.row(r)[2]]
                    }
                    return v }())
              + " " + nums([pa.w.x, pa.w.y, pa.w.z, pa.v.x, pa.v.y, pa.v.z])
              + " " + nums(body.passTau().map { Double($0) }))
    }
}

if args.count > 3 && args[3] == "pass" {
    var passTau = [Double](repeating: 0, count: body.nj)
    for j in 0..<body.nj { passTau[j] = g.torque.amplitude[j] }
    body.step(dt: g.dt, torque: passTau)
    dumpPass()
    exit(0)
}

// `settle`: the standing scenario's first twenty milliseconds, printed every
// hundred steps, to find where the port and the reference part company.
if args.count > 3 && args[3] == "settle" {
    body.floorZ = asset.floorZ
    // the driver above reset the body to the golden's test state; this mode
    // wants the animal's own stance, which is what `reset()` gives
    body.reset()
    // `reset()` puts the animal's root at the stance height and the
    // initialiser does not; try both here, because that difference is what
    // the standing numbers are sensitive to.
    if args.count > 4 && args[4] == "stance" {
        body.rootPos = Vec3(0, 0, asset.stanceRootZ)
    }
    let posture = body.holdExcitation()
    let dt = asset.timestepS
    let h = dt / 10
    let (kp, kd) = body.stanceServo(dt: h)
    print("k comZ contacts maxQd maxExc")
    do {
        let posture0 = posture
        print(String(format: "posture: maxAbs %.17e  first5 %.6e %.6e %.6e %.6e %.6e", posture0.map { abs($0) }.max() ?? 0,
                     posture0[0], posture0[1], posture0[2], posture0[3], posture0[4]))
        print(String(format: "kp: maxAbs %.17e  first5 %.6e %.6e %.6e %.6e %.6e", kp.map { abs($0) }.max() ?? 0,
                     kp[0], kp[1], kp[2], kp[3], kp[4]))
        print(String(format: "kd: maxAbs %.17e  first5 %.6e %.6e %.6e %.6e %.6e", kd.map { abs($0) }.max() ?? 0,
                     kd[0], kd[1], kd[2], kd[3], kd[4]))
        print(String(format: "stanceQ: first5 %.17e %.17e %.17e %.17e %.17e",
                     body.stanceQ[0], body.stanceQ[1], body.stanceQ[2], body.stanceQ[3], body.stanceQ[4]))
        print(String(format: "root: rootPos %.17e %.17e %.17e  floorZ %.17e  stanceRootZ %.17e  timestep %.17e",
                     body.rootPos.x, body.rootPos.y, body.rootPos.z,
                     body.floorZ, asset.stanceRootZ, asset.timestepS))
    }
    for k in 1...2000 {
        let ramp = min(1.0, Double(k) / 5000.0)
        var exc = [Double](repeating: 0, count: body.nj)
        for j in 0..<body.nj {
            let e = ramp * posture[j] + kp[j] * (body.stanceQ[j] - body.q[j]) - kd[j] * body.qd[j]
            exc[j] = min(1, max(-1, e))
        }
        body.step(dt: h, torque: body.muscleTorque(q: body.q, qd: body.qd, excitation: exc))
        if [1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 800, 1600, 2000].contains(k) {
            var mq = 0.0, me = 0.0
            for j in 0..<body.nj { mq = max(mq, abs(body.qd[j])); me = max(me, abs(exc[j])) }
            print(String(format: "%d %.9f %d %.6e %.6e", k, body.centreOfMass().z, body.contacts, mq, me))
        }
    }
    exit(0)
}

// `stand`: the standing test's own scenario, run here so a solver change can
// be checked against its assertions without a 45-minute CI cycle. The numbers
// printed are the ones FlyDynamicsTests asserts on.
if args.count > 3 && args[3] == "stand" {
    // the floor the asset came with, not the golden's "no floor" (the replay
    // above sets floorZ from the golden, and a fly with no floor falls)
    body.floorZ = asset.floorZ
    // the driver reset the body to the golden's test state; the standing test
    // builds a fresh animal, which is this
    body.reset()
    let posture = body.holdExcitation()
    let dt = asset.timestepS
    body.settle(ms: 20, dt: dt / 10, posture: posture)
    let (kp, kd) = body.stanceServo(dt: dt)
    var comZ: [Double] = []
    var feetLow = Int.max
    var worstRate = 0.0
    for k in 1...200 {
        let ramp = min(1.0, Double(k) / 500.0)
        var exc = [Double](repeating: 0, count: body.nj)
        for j in 0..<body.nj {
            let e = ramp * posture[j] + kp[j] * (body.stanceQ[j] - body.q[j]) - kd[j] * body.qd[j]
            exc[j] = min(1, max(-1, e))
        }
        body.step(dt: dt, torque: body.muscleTorque(q: body.q, qd: body.qd, excitation: exc))
        comZ.append(body.centreOfMass().z)
        var feet = 0
        for (_, f) in body.footForce where f > 1e-6 { feet += 1 }
        feetLow = min(feetLow, feet)
        for r in body.qd { worstRate = max(worstRate, abs(r)) }
    }
    let last = comZ.suffix(50)
    let mean = last.reduce(0, +) / Double(last.count)
    print(String(format: "stand: COM z mean %.6f (test wants -0.0229 +/- 0.01), first %.6f, last %.6f",
                 mean, comZ.first ?? 0, comZ.last ?? 0))
    print(String(format: "stand: feet down minimum %d of 6 (test wants >= 4), worst joint rate %.3e, contacts %d",
                 feetLow, worstRate, body.contacts))
    print("stand: " + (mean.isFinite ? "finite" : "NOT FINITE"))
    exit(0)
}

// `judge <state.json>`: one step from a state given on the command line, so
// the same state can be handed to MuJoCo and to `tools/fly_aba.py` and the
// three accelerations compared. This is how the sign of the articulated-body
// inertia's rank-one term was settled: whoever matches MuJoCo is right.
struct JudgeState: Decodable {
    let q: [Double]; let qd: [Double]; let v: [Double]; let omega: [Double]
    let rootPos: [Double]?; let rootQuat: [Double]?
    let tau: [Double]; let dt: Double; let limits: Bool?
    let passDump: Bool?
    let floorZ: Double?
    let excitation: [Double]?
}
if args.count > 3 && args[3] == "judge" {
    let st = try dec.decode(JudgeState.self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))
    let rq = st.rootQuat ?? [1, 0, 0, 0]
    body.reset(rootPosition: Vec3(st.rootPos ?? [0, 0, 0]),
               quat: Quat(rq[0], rq[1], rq[2], rq[3]),
               q: st.q, qd: st.qd, vel: Vec3(st.v), omega: Vec3(st.omega))
    if let fz = st.floorZ { body.floorZ = fz }
    if let exc = st.excitation {
        // the muscle model's own torque, so the muscle curve can be judged on
        // its own: the rates below are then the solver's, not the muscles'
        let mt = body.muscleTorque(q: body.q, qd: body.qd, excitation: exc)
        for j in 0..<body.nj { print(String(format: "M %d %.17e", j, mt[j])) }
    }
    let qdd = body.step(dt: st.dt, torque: st.tau)
    for j in 0..<body.nj { print(String(format: "A %d %.17e", j, qdd[j])) }
    if st.passDump == true { dumpPass() }
    // The rates after the step, which is what MuJoCo can actually be compared
    // against: MuJoCo handles joint damping implicitly inside its integrator,
    // so its own `qacc` is not the acceleration it integrates and disagreeing
    // with `qacc` means nothing.
    var footSum = 0.0
    for (_, f) in body.footForce { footSum += f }
    print(String(format: "S contacts %d feet %d footForceSum %.17e", body.contacts, body.footForce.count, footSum))
    for j in 0..<body.nj { print(String(format: "V %d %.17e", j, body.qd[j])) }
    for g in 0..<body.ng {
        let w = body.geomWorld(g)
        let r = w.rot
        print("G \(g) \(w.type) \(w.body) " + String(format: "%.17e %.17e %.17e %.17e",
              w.size.x, w.size.y, w.size.z, w.h) + " "
              + (0..<9).map { String(format: "%.17e", [$0].isEmpty ? 0 : r.row($0 / 3)[$0 % 3]) }.joined(separator: " ")
              + " " + String(format: "%.17e %.17e %.17e", w.pos.x, w.pos.y, w.pos.z))
    }
    // the velocities the recursion saw, and the bias it used, body by body
    for i in 0..<body.nb {
        let vv = body.passV(i), cc = body.passC(i)
        print(String(format: "W %d %.17e %.17e %.17e %.17e %.17e %.17e", i,
                     vv.w.x, vv.w.y, vv.w.z, vv.v.x, vv.v.y, vv.v.z))
        print(String(format: "C %d %.17e %.17e %.17e %.17e %.17e %.17e", i,
                     cc.w.x, cc.w.y, cc.w.z, cc.v.x, cc.v.y, cc.v.z))
    }
    exit(0)
}

if args.count > 3 && args[3] == "onestep" {
    var tau = [Double](repeating: 0, count: body.nj)
    for j in 0..<body.nj { tau[j] = g.torque.amplitude[j] }
    let qdd = body.step(dt: g.dt, torque: tau)
    print("after one step:")
    for j in 0..<body.nj {
        print(String(format: "J %3d q %+.15e qd %+.15e qdd %+.15e", j, body.q[j], body.qd[j], qdd[j]))
    }
    exit(0)
}

let trace = g.trace ?? []
var next = 0
var firstBad = -1
var firstBadInfo = (joint: -1, got: 0.0, want: 0.0)
for k in 0..<g.steps {
    var tau = [Double](repeating: 0, count: body.nj)
    let phase = 2 * Double.pi * Double(k) * g.dt / g.torque.periodS
    for j in 0..<body.nj { tau[j] = g.torque.amplitude[j] * cos(phase) }
    body.step(dt: g.dt, torque: tau)
    if next < trace.count && trace[next].step == k {
        let s = trace[next]
        next += 1
        var worst = 0.0, joint = -1, got = 0.0
        for j in 0..<min(s.q.count, body.q.count) {
            let d = abs(body.q[j] - s.q[j])
            if d > worst { worst = d; joint = j; got = body.q[j] }
        }
        let dr = zip(body.rootPos.array, s.rootPos).map { abs($0 - $1) }.max() ?? 0
        print(String(format: "  step %3d  worst|dq| %10.3e (joint %3d)   |drootPos| %10.3e", k, worst, joint, dr))
        if firstBad < 0 && worst > 1e-9 {
            firstBad = k; firstBadInfo = (joint, got, s.q[joint])
        }
    }
}

func worst(_ a: [Double], _ b: [Double]) -> Double {
    var w = 0.0
    for i in 0..<min(a.count, b.count) { w = max(w, abs(a[i] - b[i])) }
    return w
}
let e = g.expectedFinal
print("final, this solver vs the reference:")
print(String(format: "  joint angles     %12.6f   (tol %.0e)", worst(body.q, e.q), g.tolerance))
print(String(format: "  joint rates      %12.6f", worst(body.qd, e.qd)))
print(String(format: "  root velocity    %12.6f", worst(body.vel.array, e.v)))
print(String(format: "  root omega       %12.6f", worst(body.omega.array, e.omega)))
if let rp = e.rootPos { print(String(format: "  root position    %12.6f", worst(body.rootPos.array, rp))) }
if let rq = e.rootQuat {
    let got = [body.rootQuat.x, body.rootQuat.y, body.rootQuat.z, body.rootQuat.w]
    print(String(format: "  root quaternion  %12.6f", worst(got, rq)))
}
print("  hard stops: this solver \(body.limitStops), reference \(e.limitStops ?? -1)")
print("  joints at a stop at the end: \(body.jointsAtLimit())")
if firstBad >= 0 {
    print(String(format: "FIRST DIVERGENCE at step %d: joint %d is %.12f, reference %.12f",
                 firstBad, firstBadInfo.joint, firstBadInfo.got, firstBadInfo.want))
} else {
    print("no divergence above 1e-9 at any traced step")
}
