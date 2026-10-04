#!/usr/bin/env python3
"""
Step 2b — read `flybrain.bin` back exactly the way the Swift loader will, verify
every section, then run a reference LIF simulation in NumPy.

The point of the NumPy run is to pin down the parameters BEFORE writing Metal:
a GPU kernel that produces silence or a seizure is very hard to debug, so we fix
the gain / threshold / leak here and the shader just has to reproduce these numbers.

Usage: python3 tools/verify_and_simulate.py --bin build/flybrain.bin --ms 500
"""
from __future__ import annotations
import argparse, json, struct, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="build/flybrain.bin")
ap.add_argument("--meta", default="build/flybrain_meta.json")
ap.add_argument("--ms", type=int, default=500)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--drive", type=float, default=1.2, help="input current to retina")
ap.add_argument("--out", default="reports/step2_simulation.json")
args = ap.parse_args()

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "retinaIdx", "retinaUV", "motorIdx"]

# ============================================================ read the header
buf = np.memmap(args.bin, dtype=np.uint8, mode="r")
magic, version, N, E, nRetina, nMotor, nSections = struct.unpack_from("<8sIIIIII", buf, 0)
assert magic == b"FLYBRAIN", f"bad magic {magic!r}"
log(f"magic OK  version={version}  N={N:,}  E={E:,}  retina={nRetina:,}  motor={nMotor:,}")
assert nSections == len(SECTIONS)

sec = {}
for i, name in enumerate(SECTIONS):
    off, length = struct.unpack_from("<QQ", buf, 64 + i * 16)
    sec[name] = (off, length)
    assert off % 256 == 0, f"{name} not 256-aligned"

def view(name, dtype, count=None, cols=1):
    off, length = sec[name]
    a = np.frombuffer(buf, dtype=dtype, count=(length // np.dtype(dtype).itemsize), offset=off)
    return a.reshape(-1, cols) if cols > 1 else a

rootIDs = view("rootIDs", np.uint64)
positions = view("positions", np.float16, cols=3)
meta = view("neuronMeta", np.uint8, cols=8)
rowPtr = view("csrRowPtr", np.uint32)
colIdx = view("csrColIdx", np.uint32)
weight = view("csrWeight", np.float16)
delay = view("csrDelay", np.uint8)
retinaIdx = view("retinaIdx", np.uint32)
retinaUV = view("retinaUV", np.float16, cols=2)
motorIdx = view("motorIdx", np.uint32)

# ================================================================= validate
log("validating sections ...")
checks = {}
def check(name, cond, detail=""):
    checks[name] = bool(cond)
    print(f"   {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    return cond

check("rootIDs length", len(rootIDs) == N)
check("rootIDs unique", len(np.unique(rootIDs)) == N)
check("positions length", len(positions) == N)
check("positions finite", np.isfinite(positions.astype(np.float32)).all())
P = positions.astype(np.float32)
check("positions in unit cube", np.abs(P).max() <= 0.5001, f"max |p| = {np.abs(P).max():.4f}")
check("meta length", len(meta) == N)
check("rowPtr length", len(rowPtr) == N + 1)
check("rowPtr monotonic", bool((np.diff(rowPtr.astype(np.int64)) >= 0).all()))
check("rowPtr terminates at E", int(rowPtr[-1]) == E, f"{int(rowPtr[-1])} vs {E}")
check("colIdx length", len(colIdx) == E)
check("colIdx in range", int(colIdx.max()) < N, f"max = {int(colIdx.max())}")
check("weight length", len(weight) == E)
W = weight.astype(np.float32)
check("weight finite", np.isfinite(W).all())
check("weight nonzero", (W != 0).all())
check("delay length", len(delay) == E)
check("delay >= 1", int(delay.min()) >= 1, f"min = {int(delay.min())} max = {int(delay.max())}")
check("retinaIdx length", len(retinaIdx) == nRetina)
check("retinaIdx in range", int(retinaIdx.max()) < N)
check("retinaUV in [0,1]", 0 <= retinaUV.astype(np.float32).min() and retinaUV.astype(np.float32).max() <= 1.0001)
check("motorIdx in range", int(motorIdx.max()) < N)

systems = meta[:, 0]
check("system ids contiguous", bool((np.diff(systems.astype(np.int16)) >= 0).all()),
      "view-mode ranges are valid")

exc_frac = float((W > 0).mean())
check("E/I balance plausible", 0.5 < exc_frac < 0.9, f"{exc_frac*100:.1f}% excitatory")

log(f"{sum(checks.values())}/{len(checks)} checks passed")
assert all(checks.values()), "validation failed"

# ====================================================== reference LIF simulation
# Classic current-based LIF, exactly what the Metal kernel will implement:
#   tau_m dV/dt = -(V - V_rest) + R*I
#   spike when V >= V_th  ->  V := V_reset, refractory for t_ref
# Synaptic input arrives `delay[e]` milliseconds after the presynaptic spike, so
# we keep a ring buffer of future currents.
log(f"running reference LIF for {args.ms} ms (dt = 1 ms) ...")
DT        = 1.0      # ms
TAU_M     = 20.0     # ms  membrane time constant
V_REST    = 0.0
V_TH      = 1.0
V_RESET   = 0.0
T_REF     = 2        # ms
TAU_SYN   = 5.0      # ms  synaptic current decay
GAIN      = args.gain
NOISE     = 0.015    # per-step gaussian, keeps the net from being perfectly silent

rng = np.random.default_rng(42)
V      = rng.uniform(0.0, 0.5, N).astype(np.float32)
I_syn  = np.zeros(N, np.float32)
refrac = np.zeros(N, np.int16)

MAX_DELAY = int(delay.max()) + 1
ring = np.zeros((MAX_DELAY, N), np.float32)      # future synaptic input

# precompute the presynaptic index of every edge (CSR row -> edge)
edge_pre = np.repeat(np.arange(N, dtype=np.uint32), np.diff(rowPtr.astype(np.int64)))

I_ext = np.zeros(N, np.float32)
I_ext[retinaIdx] = args.drive        # steady "the lights are on" retinal drive

decay_m = float(np.exp(-DT / TAU_M))
decay_s = float(np.exp(-DT / TAU_SYN))

spike_counts = np.zeros(N, np.int64)
rate_trace, active_trace = [], []
sys_names = json.load(open(args.meta))["systems"]
sys_trace = {s: [] for s in sys_names}
sys_of = meta[:, 0]
sys_size = np.bincount(sys_of, minlength=len(sys_names)).astype(np.float64)

for t in range(args.ms):
    slot = t % MAX_DELAY
    I_syn = I_syn * decay_s + ring[slot]
    ring[slot] = 0.0

    I = I_syn * GAIN + I_ext + rng.normal(0, NOISE, N).astype(np.float32)
    free = refrac <= 0
    V[free] = V_REST + (V[free] - V_REST) * decay_m + I[free] * (1.0 - decay_m)
    refrac[~free] -= 1

    fired = np.flatnonzero((V >= V_TH) & free)
    V[fired] = V_RESET
    refrac[fired] = T_REF
    spike_counts[fired] += 1

    if len(fired):
        # scatter each firing neuron's outgoing edges into the delay ring
        starts = rowPtr[fired].astype(np.int64)
        ends   = rowPtr[fired + 1].astype(np.int64)
        n_e = ends - starts
        if n_e.sum():
            eidx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
            tgt = colIdx[eidx]
            slots = (t + delay[eidx].astype(np.int64)) % MAX_DELAY
            np.add.at(ring, (slots, tgt), W[eidx])

    rate = len(fired) / N * 1000.0 / DT            # Hz, population mean
    rate_trace.append(rate)
    active_trace.append(int(len(fired)))
    if t % 10 == 0:
        c = np.bincount(sys_of[fired], minlength=len(sys_names))
        for i, s in enumerate(sys_names):
            sys_trace[s].append(float(c[i] / max(sys_size[i], 1) * 1000.0))

    if t % 100 == 0:
        log(f"  t={t:4d} ms  fired={len(fired):6,}  mean rate={rate:7.2f} Hz  "
            f"V mean={V.mean():.3f}")

mean_rate = float(np.mean(rate_trace[50:]))       # discard the transient
frac_silent = float((spike_counts == 0).mean())
per_neuron_hz = spike_counts / (args.ms / 1000.0)

log("")
log("=== SIMULATION SUMMARY ===")
log(f"  population mean rate    {mean_rate:.2f} Hz")
log(f"  neurons that never fired {frac_silent*100:.1f}%")
log(f"  median rate of active    {np.median(per_neuron_hz[spike_counts>0]):.2f} Hz")
log(f"  99th pct rate            {np.percentile(per_neuron_hz, 99):.2f} Hz")
log(f"  max rate                 {per_neuron_hz.max():.2f} Hz")
log("  per-system mean rate (Hz):")
for s in sys_names:
    if sys_trace[s]:
        log(f"    {s:<26} {np.mean(sys_trace[s][5:]):7.2f}")

verdict = ("HEALTHY" if 0.5 < mean_rate < 40 and frac_silent < 0.95
           else "SEIZURE" if mean_rate >= 40 else "TOO QUIET")
log(f"  verdict: {verdict}")

json.dump({
    "checks": checks,
    "params": {"dt_ms": DT, "tau_m": TAU_M, "v_th": V_TH, "t_ref": T_REF,
               "tau_syn": TAU_SYN, "gain": GAIN, "retina_drive": args.drive,
               "noise": NOISE},
    "results": {"ms": args.ms, "mean_rate_hz": mean_rate,
                "frac_never_fired": frac_silent,
                "median_active_hz": float(np.median(per_neuron_hz[spike_counts > 0]))
                                     if (spike_counts > 0).any() else 0.0,
                "p99_hz": float(np.percentile(per_neuron_hz, 99)),
                "max_hz": float(per_neuron_hz.max()),
                "verdict": verdict},
    "rate_trace_hz": rate_trace,
    "per_system_hz": {s: (float(np.mean(v[5:])) if len(v) > 5 else 0.0)
                      for s, v in sys_trace.items()},
}, open(args.out, "w"), indent=2)
log(f"wrote {args.out}")
