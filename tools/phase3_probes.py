#!/usr/bin/env python3
"""
phase3_probes.py — the round-5 connectome-level checks (T5-T7), run headless
against flybanc.bin with the same LIF constants as the Metal engine and
verify_banc.py (dt 1 ms, tau_m 20 ms, tau_syn 5 ms, v_th 1.0, refractory
2 ms, noise sigma 0.015).

The retina is driven SYNTHETICALLY: each sensory_vision neuron carries a
retinotopic UV (the per-hemisphere SVD in build_banc.py), so a visual scene
is just a luminance function lum(u, v, t) sampled per neuron per frame and
pushed through the same high-pass the sampleRetina kernel uses (tau 0.1 s,
gain 8, bias 0.5, clamp [0, 4]).

  T5  no stimulus                -> wing-motor left/right asymmetry stays small
  T6  looming vs receding/static -> LC4 / LPLC2 / DNp01 must fire more for
                                    the approaching dark disc (looming). If
                                    they do not, that is REPORTED, not tuned.
  T7  optomotor                  -> a world-wide image sweep must bias wing
                                    power asymmetrically, i.e. the connectome
                                    produces a steering command; the SIGN is
                                    reported against the slip-reducing
                                    direction, with the SVD axis caveat
                                    (the u-axis orientation of each
                                    hemisphere's projection is only defined up
                                    to a sign, so the absolute sign of this
                                    probe is not calibrated — see report).

Run:  python3 tools/phase3_probes.py [--ms 400] [--gain 12]
"""
from __future__ import annotations
import argparse, json, math, struct, sys, time

import numpy as np
import pyarrow.feather as feather

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="build/flybanc.bin")
ap.add_argument("--meta", default="build/flybanc_meta.json")
ap.add_argument("--raw", default="data/banc")
ap.add_argument("--ms", type=int, default=400, help="ms per condition")
ap.add_argument("--gain", type=float, default=12.0)
ap.add_argument("--out", default="reports/phase3_probes.json")
args = ap.parse_args()

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]
buf = np.memmap(args.bin, dtype=np.uint8, mode="r")
magic, version, N, E, nG, nV, nS = struct.unpack_from("<8sIIIIII", buf, 0)
assert magic == b"FLYBANC_", magic
sec = {}
for i, name in enumerate(SECTIONS):
    off, length = struct.unpack_from("<QQ", buf, 64 + i * 16)
    sec[name] = (off, length)

def view(name, dtype, cols=1):
    off, length = sec[name]
    a = np.frombuffer(buf, dtype=dtype,
                      count=length // np.dtype(dtype).itemsize, offset=off)
    return a.reshape(-1, cols) if cols > 1 else a

roots = view("rootIDs", np.uint64)
rowPtr = view("csrRowPtr", np.uint32)
colIdx = view("csrColIdx", np.uint32)
W = view("csrWeight", np.float16).astype(np.float32)
delay = view("csrDelay", np.uint8)
groupIdx = view("groupIndices", np.uint32)
retinaUV = view("retinaUV", np.float16, cols=2).astype(np.float32)

M = json.load(open(args.meta))
groups = {g["name"]: g for g in M["groups"]}
def group(name):
    g = groups[name]
    return groupIdx[g["start"]:g["start"] + g["count"]]

meta = feather.read_table(
    f"{args.raw}/banc_888_meta.feather",
    columns=["root_id", "cell_type", "side"]).to_pandas()
cell_type = dict(zip(meta["root_id"].astype(np.uint64), meta["cell_type"]))
side_of = dict(zip(meta["root_id"].astype(np.uint64), meta["side"]))

vision = group("sensory_vision")
vside = np.array([side_of.get(r, None) for r in roots[vision]], dtype=object)
vis_L = vision[vside == "left"]
vis_R = vision[vside == "right"]
uv_of = {int(i): retinaUV[k] for k, i in enumerate(vision)}

def celltype_group(prefix):
    hits = [i for i, r in enumerate(roots)
            if str(cell_type.get(r, "")).startswith(prefix)]
    return np.array(hits, np.uint32)

LC4 = celltype_group("LC4")
LPLC2 = celltype_group("LPLC2")
DNp01 = celltype_group("DNp01")
log(f"LC4 {len(LC4)}  LPLC2 {len(LPLC2)}  DNp01 {len(DNp01)}  "
    f"vision {len(vision)} (L {len(vis_L)} / R {len(vis_R)})")

wingL = group("motor_wing_power_left")
wingR = group("motor_wing_power_right")
steerL = group("motor_wing_steering_left")
steerR = group("motor_wing_steering_right")

# --------------------------------------------------------------- LIF engine
DT, TAU_M, TAU_SYN = 1.0, 20.0, 5.0
V_TH, V_RESET, T_REF, NOISE = 1.0, 0.0, 2, 0.015
ADAPT_TAU_MS, CONTRAST_GAIN, CAMERA_BIAS = 100.0, 8.0, 0.5   # ASSUMPTIONS #15
MAX_D = int(delay.max()) + 1


def run(ms, lum_fn, seed=42):
    """lum_fn(uv[L], uv[R], t_ms) -> (lumL, lumR) arrays or scalars."""
    rng = np.random.default_rng(seed)
    V = rng.uniform(0, 0.5, N).astype(np.float32)
    I_syn = np.zeros(N, np.float32)
    refrac = np.zeros(N, np.int16)
    ring = np.zeros((MAX_D, N), np.float32)
    dm, ds = math.exp(-DT / TAU_M), math.exp(-DT / TAU_SYN)
    spikes = np.zeros(N, np.int64)
    spike_times = []

    I_ext = np.zeros(N, np.float32)
    adaptL = np.full(len(vis_L), CAMERA_BIAS, np.float32)
    adaptR = np.full(len(vis_R), CAMERA_BIAS, np.float32)
    uvL = np.array([uv_of[int(i)] for i in vis_L])
    uvR = np.array([uv_of[int(i)] for i in vis_R])
    alpha = 1.0 - math.exp(-DT / ADAPT_TAU_MS)

    wing_trace = []
    for t in range(ms):
        lumL, lumR = lum_fn(uvL, uvR, t)
        adaptL += (lumL - adaptL) * alpha
        adaptR += (lumR - adaptR) * alpha
        I_ext[vis_L] = np.clip(CAMERA_BIAS + (lumL - adaptL) * CONTRAST_GAIN,
                               0.0, 4.0)
        I_ext[vis_R] = np.clip(CAMERA_BIAS + (lumR - adaptR) * CONTRAST_GAIN,
                               0.0, 4.0)

        slot = t % MAX_D
        I_syn = I_syn * ds + ring[slot]; ring[slot] = 0
        I = I_syn * args.gain + I_ext + rng.normal(0, NOISE, N).astype(np.float32)
        free = refrac <= 0
        V[free] = V[free] * dm + I[free] * (1 - dm)
        refrac[~free] -= 1
        fired = np.flatnonzero((V >= V_TH) & free)
        V[fired] = V_RESET; refrac[fired] = T_REF; spikes[fired] += 1
        spike_times.append(fired)
        if len(fired):
            s0, s1 = rowPtr[fired].astype(np.int64), rowPtr[fired + 1].astype(np.int64)
            if (s1 - s0).sum():
                e = np.concatenate([np.arange(a, b) for a, b in zip(s0, s1)])
                np.add.at(ring, ((t + delay[e].astype(np.int64)) % MAX_D,
                                 colIdx[e]), W[e])
        f = np.zeros(N, bool); f[fired] = True
        wing_trace.append((f[wingL].sum() / max(len(wingL), 1) * 1000.0,
                           f[wingR].sum() / max(len(wingR), 1) * 1000.0))
    return spikes, wing_trace, spike_times


def hz(spikes, idx):
    return float(spikes[idx].sum() / max(len(idx), 1) / (args.ms / 1000.0))


report = {"params": {"ms": args.ms, "gain": args.gain}}
fails = []

# ---- T5: no stimulus -> small wing asymmetry --------------------------------
log("T5: no stimulus, uniform field ...")
spikes, wt, _ = run(args.ms, lambda uL, uR, t: (0.5, 0.5))
a = np.array(wt[100:])                                   # skip the onset
rms_asym = float(np.sqrt(((a[:, 0] - a[:, 1]) ** 2).mean()))
base = float((a[:, 0] + a[:, 1]).mean() / 2)
t5_ok = rms_asym < max(5.0, 0.25 * base)
log(f"  wing L/R rates {a[:,0].mean():.1f}/{a[:,1].mean():.1f} Hz, "
    f"RMS asymmetry {rms_asym:.1f} Hz, bar {max(5.0, 0.25*base):.1f}")
report["T5_no_stimulus"] = {"wingL_hz": float(a[:, 0].mean()),
                            "wingR_hz": float(a[:, 1].mean()),
                            "rms_asym_hz": rms_asym, "pass": bool(t5_ok)}
if not t5_ok: fails.append("T5_no_stimulus")

# ---- T6: looming vs receding vs static --------------------------------------
def disc(side, r_of_t, centre=(0.5, 0.5)):
    """Dark expanding/contracting disc on one eye, bright on the other."""
    def fn(uvL, uvR, t):
        r = r_of_t(t)
        def lum(uv):
            d = np.linalg.norm(uv - np.array(centre), axis=1)
            return np.where(d < r, 0.0, 1.0).astype(np.float32)
        if side == "left":
            return lum(uvL), np.full(len(uvR), 1.0, np.float32)
        return np.full(len(uvL), 1.0, np.float32), lum(uvR)
    return fn

r_loom = lambda t: 0.05 + 0.40 * min(1.0, t / 300.0)
r_recd = lambda t: 0.45 - 0.40 * min(1.0, t / 300.0)
r_stat = lambda t: 0.25

t6 = {}
for label, fn in (("looming", disc("left", r_loom)),
                  ("receding", disc("left", r_recd)),
                  ("static", disc("left", r_stat))):
    log(f"T6: {label} disc, left eye ...")
    spikes, _, stimes = run(args.ms, fn)
    # Score the stimulus window only (t >= 150 ms): the disc ONSET transient
    # is identical in all three conditions and would dilute any difference.
    late = np.concatenate(stimes[150:]) if len(stimes) > 150 else np.array([], np.uint32)
    window = (args.ms - 150) / 1000.0
    def whz(idx):
        return float(np.isin(late, idx).sum() / max(len(idx), 1) / window)
    t6[label] = {"LC4_hz": whz(LC4), "LPLC2_hz": whz(LPLC2),
                 "DNp01_hz": whz(DNp01)}
    log(f"    LC4 {t6[label]['LC4_hz']:.1f}  LPLC2 {t6[label]['LPLC2_hz']:.1f}  "
        f"DNp01 {t6[label]['DNp01_hz']:.1f} Hz")

def prefers_looming(key):
    return (t6["looming"][key] > t6["receding"][key]
            and t6["looming"][key] > t6["static"][key])

t6_pass = {k: bool(prefers_looming(k)) for k in ("LC4_hz", "LPLC2_hz", "DNp01_hz")}
for k, ok in t6_pass.items():
    log(f"  {k}: looming > receding and static: {ok}")
    if not ok:
        fails.append(f"T6_{k}")     # reported, NOT tuned (round-5 rule 3)
report["T6_looming"] = {"rates": t6, "pass": t6_pass}

# ---- T7: optomotor ----------------------------------------------------------
log("T7: full-field image sweep ...")
SWEEP = 0.8          # uv-units per second
def sweep_fn(uvL, uvR, t):
    ph = SWEEP * t / 1000.0
    return (0.5 + 0.45 * np.sin(2 * np.pi * (4 * uvL[:, 0] - ph))).astype(np.float32), \
           (0.5 + 0.45 * np.sin(2 * np.pi * (4 * uvR[:, 0] - ph))).astype(np.float32)

spikes, wt, _ = run(args.ms, sweep_fn)
spL, spR = hz(spikes, wingL), hz(spikes, wingR)
stL, stR = hz(spikes, steerL), hz(spikes, steerR)
t7_responds = abs(spL - spR) > 1.0 or abs(stL - stR) > 1.0
log(f"  wing power L/R {spL:.1f}/{spR:.1f} Hz, steering L/R {stL:.1f}/{stR:.1f} Hz")
log(f"  responds asymmetrically: {t7_responds}")
log("  NOTE: the per-hemisphere SVD defines the u-axis only up to a sign, so")
log("  the absolute direction of this bias is NOT calibrated headlessly; the")
log("  in-app closed-loop check (world visibly slips -> fly must reduce slip)")
log("  is the calibrated test. Reported, not flipped.")
report["T7_optomotor"] = {"wingL_hz": spL, "wingR_hz": spR,
                          "steerL_hz": stL, "steerR_hz": stR,
                          "responds": bool(t7_responds)}
if not t7_responds:
    fails.append("T7_responds")

report["fails"] = fails
json.dump(report, open(args.out, "w"), indent=2)
log(f"wrote {args.out}")
log("")
if fails:
    log("REPORTED (per round-5 rule: report failures, do not tune): "
        + ", ".join(fails))
else:
    log("all probes met their bars")
