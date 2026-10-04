#!/usr/bin/env python3
"""
Step 1 — FlyWire FAFB v783 dataset profiling for the iOS brain-simulation app.

Reads the public Codex CSV dumps (no auth required) and reports:
  * neuron census (count, coverage of coordinates / cell types / classification)
  * connectivity census (pairs, synapses, thresholds, degree distribution)
  * neuropil inventory grouped into brain systems (optic lobe / central brain / ...)
  * candidate subsets for on-device simulation, with byte-exact Metal buffer budgets

Usage:
  python3 tools/step1_profile_flywire.py --raw data/raw --out reports
"""
from __future__ import annotations
import argparse, gzip, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

t0 = time.time()

def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---------------------------------------------------------------- neuropil map
# FlyWire FAFB v783 primary neuropils, grouped into systems.
# (names come straight out of connections.csv.gz; _L/_R = hemisphere)
OPTIC_LOBE = {
    "LA", "LO", "LOP", "ME", "ME_L", "ME_R", "LA_L", "LA_R", "LO_L", "LO_R",
    "LOP_L", "LOP_R", "ME_CA", "AMMC",  # AMMC listed separately below
}
OPTIC_SET = {"LA", "LO", "LOP", "ME", "MED", "AOTU", "AME", "LOPLA", "LOPM",
             "LA_L","LA_R","LO_L","LO_R","LOP_L","LOP_R","ME_L","ME_R","ME_CA",
             "AMMC_L","AMMC_R"}
CENTRAL_COMPLEX = {"AL","EB","FB","PB","NO","PON","GA","CRE","IDF","ICL","RUB",
                   "SAD","SLP","SMP","VES","SPS","IPS","SOG","GNG","PRW","CAN",
                   "FLT","OCI","PED","CA","MB","LH","AVLP","PVLP","PLP","GOR",
                   "WED","EPA","EPP","SIP","SFS","SMP_L","SMP_R","SLP_L","SLP_R"}

SYSTEM_OF = {
    "LA":"optic_lobe","LO":"optic_lobe","LOP":"optic_lobe","ME":"optic_lobe",
    "MED":"optic_lobe","AMMC":"optic_lobe","AOTU":"optic_lobe","AME":"optic_lobe",
    "LOPLA":"optic_lobe","LOPM":"optic_lobe",
    "EB":"central_complex","FB":"central_complex","PB":"central_complex","NO":"central_complex",
    "PON":"central_complex","SAD":"central_complex","CAN":"central_complex","FLT":"central_complex",
    "GA":"central_complex","ICL":"central_complex","IDF":"central_complex","RUB":"central_complex",
    "AL":"olfactory","LH":"olfactory","MB":"learning_memory","CA":"learning_memory","PED":"learning_memory",
    "SOG":"gnathic","GNG":"gnathic","PRW":"gnathic","SPS":"gnathic","IPS":"gnathic","WED":"gnathic",
    "AMMC_L":"optic_lobe","AMMC_R":"optic_lobe",
}
def system_of(neuropil: str) -> str:
    base = neuropil.rsplit("_", 1)[0] if neuropil.endswith(("_L","_R")) else neuropil
    if base in ("AMMC",):
        return "antennal_mechanosensory"
    if base in SYSTEM_OF:
        return SYSTEM_OF[base]
    if base.startswith(("ME","LA","LO","LOP","AM","AOTU")):
        return "optic_lobe"
    if base.startswith(("EB","FB","PB","NO","PON","SAD","CAN","FLT","GA")):
        return "central_complex"
    if base.startswith(("AL","LH","DA","VA","DL","VM","D","V","DC","VC","M")):
        return "olfactory/lateral"
    if base.startswith(("SLP","AVLP","PVLP","PLP","SMP","SIP","SFS","GOR","CRE","VES","EPA","EPP","OCI","WED","SPS","IPS","SOG","GNG","PRW","MB","CA","PED")):
        return "central_brain_other"
    return "other"

# ------------------------------------------------------------------- argparse
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="reports")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = args.raw

# ------------------------------------------------------------------- neurons
log("reading neurons.csv.gz ...")
neu = pd.read_csv(f"{R}/neurons.csv.gz")
log("reading classification.csv.gz ...")
cls = pd.read_csv(f"{R}/classification.csv.gz")
log("reading consolidated_cell_types.csv.gz ...")
cct = pd.read_csv(f"{R}/consolidated_cell_types.csv.gz")
log("reading coordinates.csv.gz ...")
coo = pd.read_csv(f"{R}/coordinates.csv.gz")
log("reading names.csv.gz ...")
nam = pd.read_csv(f"{R}/names.csv.gz")

# coordinates.position is a stringified numpy array "[x y z]"
def parse_pos(s):
    return np.array([int(v) for v in str(s).strip()[1:-1].split()], dtype=np.float64)

log("parsing coordinates ...")
pos = np.stack([parse_pos(s) for s in coo["position"].to_numpy()])
coo["x"], coo["y"], coo["z"] = pos[:,0], pos[:,1], pos[:,2]

# ---------------------------------------------------------------- connections
log("reading connections.csv.gz (this is the big one) ...")
conn = pd.read_csv(
    f"{R}/connections.csv.gz",
    dtype={"pre_root_id":"int64","post_root_id":"int64","neuropil":"category",
           "syn_count":"int32","nt_type":"category"},
)
log(f"connections rows = {len(conn):,}  mem = {conn.memory_usage(deep=True).sum()/2**20:.0f} MiB")

# ------------------------------------------------------------------- census
report = {}
report["source"] = {
    "dataset": "FlyWire FAFB v783 (Dorkenwald et al. 2024; Schlegel et al. 2024)",
    "base_url": "https://storage.googleapis.com/flywire-data/codex/data/fafb/783",
    "license": "CC BY-NC 4.0 (non-commercial)",
}
report["census"] = {
    "neurons_table_rows": int(len(neu)),
    "classification_rows": int(len(cls)),
    "cell_type_rows": int(len(cct)),
    "coordinates_rows": int(len(coo)),
    "names_rows": int(len(nam)),
    "connection_rows_pair_x_neuropil": int(len(conn)),
    "total_synapses_in_connection_table": int(conn["syn_count"].sum()),
}
uniq_pre = set(conn["pre_root_id"].unique()); uniq_post = set(conn["post_root_id"].unique())
nodes = uniq_pre | uniq_post
report["census"]["distinct_neurons_in_connectivity"] = len(nodes)
report["census"]["neurons_with_3d_position"] = int(len(coo))
report["census"]["coverage_position_pct"] = round(100*len(coo)/max(1,len(neu)),2)

# pair-level aggregation (collapse neuropil)
log("aggregating to pair level ...")
pair = (conn.groupby(["pre_root_id","post_root_id"], observed=True)["syn_count"]
            .sum().rename("syn").reset_index())
log(f"unique pairs = {len(pair):,}")
report["census"]["unique_directed_pairs"] = int(len(pair))
for thr in (1,2,5,10):
    sub = pair[pair["syn"]>=thr]
    report["census"][f"pairs_ge{thr}_synapses"] = int(len(sub))
    report["census"][f"synapses_ge{thr}"] = int(sub["syn"].sum())

# degree stats
outd = pair.groupby("pre_root_id")["syn"].sum()
ind  = pair.groupby("post_root_id")["syn"].sum()
report["degree"] = {
    "out_synapses_mean": round(float(outd.mean()),1), "out_synapses_median": float(outd.median()),
    "out_synapses_max": int(outd.max()),
    "in_synapses_mean": round(float(ind.mean()),1),   "in_synapses_median": float(ind.median()),
    "in_synapses_max": int(ind.max()),
    "pairs_per_pre_median": float(pair.groupby("pre_root_id").size().median()),
    "pairs_per_post_median": float(pair.groupby("post_root_id").size().median()),
}

# neurotransmitters
nt = neu["nt_type"].value_counts()
report["neurotransmitters"] = {str(k): int(v) for k,v in nt.items()}
report["flow"] = {str(k): int(v) for k,v in cls["flow"].value_counts().items()}
report["super_class_top20"] = {str(k): int(v) for k,v in cls["super_class"].value_counts().head(20).items()}

# ------------------------------------------------------------------ neuropils
log("neuropil breakdown ...")
npre = conn.assign(np2=conn["neuropil"].astype(str).map(lambda s: s.rsplit("_",1)[0] if s.endswith(("_L","_R")) else s))
np_tab = conn.groupby("neuropil", observed=True).agg(rows=("syn_count","size"), synapses=("syn_count","sum"))
np_tab["system"] = [system_of(i) for i in np_tab.index]
np_tab = np_tab.sort_values("synapses", ascending=False)
report["neuropil_count"] = int(len(np_tab))
report["systems"] = {str(k): {"synapses": int(v["synapses"]), "neuropils": int(v["neuropil_count"])}
                     for k,v in np_tab.groupby("system").agg(synapses=("synapses","sum"),
                                                             neuropil_count=("rows","size")).iterrows()}
np_tab.to_csv(f"{args.out}/step1_neuropils.csv")
log("top neuropils:\n" + np_tab.head(25).to_string())

# ------------------------------------------ neurons per neuropil (dominant np)
log("assigning each neuron a dominant neuropil ...")
by_post = conn.groupby(["post_root_id","neuropil"], observed=True)["syn_count"].sum().reset_index()
idx = by_post.groupby("post_root_id")["syn_count"].idxmax()
dom = by_post.loc[idx].set_index("post_root_id")["neuropil"].astype(str)
neu_dom = neu.set_index("root_id").join(dom.rename("dominant_neuropil"))
neu_dom["system"] = [system_of(x) if isinstance(x,str) else "unassigned" for x in neu_dom["dominant_neuropil"]]
report["neurons_per_system"] = {str(k): int(v) for k,v in neu_dom["system"].value_counts().items()}
log(json.dumps(report["neurons_per_system"], indent=1))

# --------------------------------------------------------------- coordinate space
report["coordinate_space"] = {
    "note": "FAFB14.1 raw voxel space, 4x4x40 nm voxels",
    "x_range": [float(coo["x"].min()), float(coo["x"].max())],
    "y_range": [float(coo["y"].min()), float(coo["y"].max())],
    "z_range": [float(coo["z"].min()), float(coo["z"].max())],
}
log(json.dumps(report["coordinate_space"]))

# --------------------------------------------------------- subset candidates
# A neuron's "system" = system of its dominant neuropil. Build candidate subsets
# and price them out in bytes for the Metal CSR buffers.
coo_idx = coo.set_index("root_id")
have_pos = set(coo_idx.index)

def price(n_neurons:int, n_edges:int, n_syn:int) -> dict:
    """Byte budget for the on-device Metal buffers (CSR, half weights)."""
    return {
        "neurons": n_neurons, "edges": n_edges, "synapses": n_syn,
        "positions_f16x3_bytes": n_neurons*6,
        "state_bytes_v_i_refrac_i32f32": n_neurons*12,
        "csr_row_ptr_uint32_bytes": (n_neurons+1)*4,
        "csr_col_idx_uint32_bytes": n_edges*4,
        "csr_weight_f16_bytes": n_edges*2,
        "csr_delay_u8_bytes": n_edges*1,
        "total_bytes": (n_neurons*6 + n_neurons*12 + (n_neurons+1)*4 + n_edges*4 + n_edges*2 + n_edges),
    }

subsets = {}
def build_subset(name, mask_neu, min_syn=1):
    ids = set(neu_dom.index[mask_neu]) & have_pos
    if not ids:
        return
    sub = pair[pair["pre_root_id"].isin(ids) & pair["post_root_id"].isin(ids) & (pair["syn"]>=min_syn)]
    n_ids = set(sub["pre_root_id"]) | set(sub["post_root_id"])
    # keep isolated neurons too so the render matches the region
    keep = ids | n_ids
    subsets[name] = {
        "definition": mask_neu.sum() and "",
        "neurons_with_position": len(ids),
        "neurons_simulated": len(keep),
        "edges": int(len(sub)), "synapses": int(sub["syn"].sum()),
        "budget": price(len(keep), len(sub), int(sub["syn"].sum())),
    }

build_subset("full_brain_all",       np.ones(len(neu_dom), bool), 1)
build_subset("full_brain_thr5",      np.ones(len(neu_dom), bool), 5)
build_subset("optic_lobe",           neu_dom["system"]=="optic_lobe", 1)
build_subset("optic_lobe_thr5",      neu_dom["system"]=="optic_lobe", 5)
build_subset("central_complex",      neu_dom["system"]=="central_complex", 1)
build_subset("olfactory",            neu_dom["system"]=="olfactory", 1)
build_subset("gnathic",              neu_dom["system"]=="gnathic", 1)
build_subset("learning_memory",      neu_dom["system"]=="learning_memory", 1)
build_subset("central_brain_other",  neu_dom["system"]=="central_brain_other", 1)

report["subsets"] = subsets
log("subset budgets:")
for k,v in subsets.items():
    b=v["budget"]
    log(f"  {k:<22} neurons={v['neurons_simulated']:>7,} edges={v['edges']:>9,} "
        f"syn={v['synapses']:>10,} gpu_bytes={b['total_bytes']/2**20:8.1f} MiB")

# save neuron table for step 2 (positions + type + system)
log("writing intermediate tables ...")
out = neu_dom.join(coo_idx[["x","y","z"]], how="left")
out = out.join(cls.set_index("root_id")[["flow","super_class","class","sub_class","side"]], how="left")
out = out.join(cct.set_index("root_id")[["primary_type"]], how="left")
out["has_position"] = out["x"].notna()
out.to_parquet(f"{args.out}/step1_neurons.parquet")
pair.to_parquet(f"{args.out}/step1_pairs.parquet")

with open(f"{args.out}/step1_report.json","w") as fh:
    json.dump(report, fh, indent=2, default=str)

log("=== REPORT ===")
print(json.dumps(report, indent=2, default=str))
log("done")
