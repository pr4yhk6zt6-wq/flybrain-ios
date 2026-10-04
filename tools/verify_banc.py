#!/usr/bin/env python3
"""
Validate flybanc.bin and run a reference LIF simulation across the whole central
nervous system — brain plus ventral nerve cord.

The thing worth measuring here, which the FAFB build could not measure at all,
is the sensorimotor loop: drive the photoreceptors, and see whether activity
reaches the leg and wing motor neurons through real wiring.
"""
from __future__ import annotations
import argparse, json, struct, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="build/flybanc.bin")
ap.add_argument("--meta", default="build/flybanc_meta.json")
ap.add_argument("--ms", type=int, default=400)
ap.add_argument("--gain", type=float, default=6.0)
ap.add_argument("--drive", type=float, default=1.5)
ap.add_argument("--out", default="reports/banc_simulation.json")
args = ap.parse_args()

buf = np.memmap(args.bin, dtype=np.uint8, mode="r")
magic, version, N, E, nGroupIdx, nVision, nSections = struct.unpack_from("<8sIIIIII", buf, 0)
assert magic == b"FLYBANC_", magic
log(f"magic OK v{version}  N={N:,}  E={E:,}  groupIdx={nGroupIdx:,}  vision={nVision:,}")

sec = {}
for i, name in enumerate(SECTIONS):
    off, length = struct.unpack_from("<QQ", buf, 64 + i * 16)
    sec[name] = (off, length)
    assert off % 256 == 0, f"{name} misaligned"

def view(name, dtype, cols=1):
    off, length = sec[name]
    a = np.frombuffer(buf, dtype=dtype,
                      count=length // np.dtype(dtype).itemsize, offset=off)
    return a.reshape(-1, cols) if cols > 1 else a

positions = view("positions", np.float16, cols=3)
meta = view("neuronMeta", np.uint8, cols=8)
rowPtr = view("csrRowPtr", np.uint32)
colIdx = view("csrColIdx", np.uint32)
weight = view("csrWeight", np.float16)
delay = view("csrDelay", np.uint8)
groupIdx = view("groupIndices", np.uint32)

M = json.load(open(args.meta))
groups = {g["name"]: g for g in M["groups"]}
def group(name):
    g = groups[name]
    return groupIdx[g["start"]:g["start"] + g["count"]]

# ================================================================== validate
log("validating ...")
checks, W = {}, weight.astype(np.float32)
def check(name, cond, detail=""):
    checks[name] = bool(cond)
    print(f"   {'PASS' if cond else 'FAIL'}  {name}  {detail}")

check("positions finite", np.isfinite(positions.astype(np.float32)).all())
check("positions in unit cube", np.abs(positions.astype(np.float32)).max() <= 0.5001)
check("rowPtr monotonic", bool((np.diff(rowPtr.astype(np.int64)) >= 0).all()))
check("rowPtr terminates at E", int(rowPtr[-1]) == E)
check("colIdx in range", int(colIdx.max()) < N)
check("weight nonzero finite", bool(np.isfinite(W).all() and (W != 0).all()))
check("delay >= 1", int(delay.min()) >= 1, f"1..{int(delay.max())} ms")
check("systems contiguous", bool((np.diff(meta[:, 0].astype(np.int16)) >= 0).all()))
check("groupIndices in range", int(groupIdx.max()) < N)
exc = float((W > 0).mean())
check("E/I balance", 0.4 < exc < 0.9, f"{exc*100:.1f}% excitatory")

leg_motor = np.concatenate([group(f"motor_{p}_{s}")
                            for p in ("front_leg", "middle_leg", "hind_leg")
                            for s in ("left", "right")])
wing_motor = np.concatenate([group(f"motor_{f}_{s}")
                             for f in ("wing_power", "wing_steering", "wing_tension")
                             for s in ("left", "right")])
vision = group("sensory_vision")
check("leg motor neurons exist", len(leg_motor) > 300, f"{len(leg_motor)}")
check("wing motor neurons exist", len(wing_motor) > 40, f"{len(wing_motor)}")
check("vision input exists", len(vision) > 5000, f"{len(vision)}")
check("motor neurons are reachable",
      bool((np.isin(colIdx, leg_motor)).any()), "leg motor appears as a postsynaptic target")

log(f"{sum(checks.values())}/{len(checks)} checks passed")
assert all(checks.values()), "validation failed"

# =============================================================== simulation
log(f"running CNS-wide LIF for {args.ms} ms ...")
DT, TAU_M, TAU_SYN = 1.0, 20.0, 5.0
V_TH, V_RESET, T_REF, NOISE = 1.0, 0.0, 2, 0.015
rng = np.random.default_rng(42)
V = rng.uniform(0, 0.5, N).astype(np.float32)
I_syn = np.zeros(N, np.float32)
refrac = np.zeros(N, np.int16)
MAX_D = int(delay.max()) + 1
ring = np.zeros((MAX_D, N), np.float32)

I_ext = np.zeros(N, np.float32)
I_ext[vision] = args.drive

dm, ds = np.exp(-DT / TAU_M), np.exp(-DT / TAU_SYN)
spikes = np.zeros(N, np.int64)
trace, leg_trace, wing_trace, desc_trace = [], [], [], []
descending = group("descending")
ascending = group("ascending")
prop = group("sensory_proprioception")

for t in range(args.ms):
    slot = t % MAX_D
    I_syn = I_syn * ds + ring[slot]; ring[slot] = 0
    I = I_syn * args.gain + I_ext + rng.normal(0, NOISE, N).astype(np.float32)
    free = refrac <= 0
    V[free] = V[free] * dm + I[free] * (1 - dm)
    refrac[~free] -= 1
    fired = np.flatnonzero((V >= V_TH) & free)
    V[fired] = V_RESET; refrac[fired] = T_REF; spikes[fired] += 1
    if len(fired):
        s0, s1 = rowPtr[fired].astype(np.int64), rowPtr[fired + 1].astype(np.int64)
        if (s1 - s0).sum():
            e = np.concatenate([np.arange(a, b) for a, b in zip(s0, s1)])
            np.add.at(ring, ((t + delay[e].astype(np.int64)) % MAX_D, colIdx[e]), W[e])
    trace.append(len(fired) / N * 1000.0)
    f = np.zeros(N, bool); f[fired] = True
    leg_trace.append(f[leg_motor].sum() / max(len(leg_motor), 1) * 1000.0)
    wing_trace.append(f[wing_motor].sum() / max(len(wing_motor), 1) * 1000.0)
    desc_trace.append(f[descending].sum() / max(len(descending), 1) * 1000.0)
    if t % 100 == 0:
        log(f"  t={t:4d}  fired={len(fired):6,}  pop={trace[-1]:6.2f} Hz  "
            f"desc={desc_trace[-1]:6.1f}  leg={leg_trace[-1]:6.1f}  wing={wing_trace[-1]:6.1f}")

def hz(idx): return float(spikes[idx].sum() / max(len(idx), 1) / (args.ms / 1000.0))

summary = {
    "population_hz": float(np.mean(trace[50:])),
    "never_fired_pct": float((spikes == 0).mean() * 100),
    "vision_hz": hz(vision),
    "descending_hz": hz(descending),
    "ascending_hz": hz(ascending),
    "proprioception_hz": hz(prop),
    "leg_motor_hz": hz(leg_motor),
    "wing_motor_hz": hz(wing_motor),
    "wing_power_hz": hz(np.concatenate([group("motor_wing_power_left"),
                                        group("motor_wing_power_right")])),
    "wing_steering_hz": hz(np.concatenate([group("motor_wing_steering_left"),
                                           group("motor_wing_steering_right")])),
    "neck_motor_hz": hz(group("motor_neck")),
    "front_leg_L_hz": hz(group("motor_front_leg_left")),
    "front_leg_R_hz": hz(group("motor_front_leg_right")),
}
log("")
log("=== SENSORIMOTOR LOOP ===")
for k, v in summary.items():
    log(f"  {k:<22} {v:8.2f}")

loop_closed = summary["leg_motor_hz"] > 0.5 and summary["wing_motor_hz"] > 0.5
log(f"  loop closed (vision -> leg/wing motor): {loop_closed}")

json.dump({"checks": checks, "summary": summary,
           "params": {"gain": args.gain, "drive": args.drive, "ms": args.ms},
           "population_trace_hz": trace,
           "leg_motor_trace_hz": leg_trace,
           "wing_motor_trace_hz": wing_trace},
          open(args.out, "w"), indent=2)
log(f"wrote {args.out}")
