"""P7 full-paper data dump — recompute every number the paper will cite from raw JSONL."""
import json
import numpy as np
from collections import defaultdict

def load(st):
    rows = []
    for l in open(f"results/sweep_stage{st}.jsonl", encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            if "error" not in r:
                rows.append(r)
    return rows

def m(row, key="ridge", sub="mean"):
    mm = row.get("metrics", {})
    if key in mm and isinstance(mm[key], dict) and sub in mm[key]:
        return mm[key][sub]
    if key in mm and isinstance(mm[key], (int, float)):
        return mm[key]
    return mm.get(key + "_" + sub if False else f"{key}_{sub}", mm.get("linear_probe_acc_mean"))

s1b = load("1b")
s2, s3, s4, s5, s6, s7 = [load(st) for st in (2, 3, 4, 5, 6, 7)]
print(f"rows: stage1b={len(s1b)} stage2={len(s2)} stage3={len(s3)} stage4={len(s4)} stage5={len(s5)} stage6={len(s6)} stage7={len(s7)}")

def key(r):
    return (r["model"], r.get("S"), r.get("dim"), r.get("alpha"), r.get("K"), r.get("seed"))

# --- L0 gate (stage 1b): ridge mean per (model, alpha), mean over seeds ---
print("\n=== L0 gate stage1b: ridge R^2 by alpha (mean over seeds) ===")
acc = defaultdict(list)
for r in s1b:
    if r["world"] == "L0":
        acc[(r["model"], r.get("alpha"))].append(m(r))
for (model, alpha), v in sorted(acc.items()):
    if model in ("jepa", "contrastive"):
        print(f"  {model:12s} alpha={alpha:7}: {np.mean(v):.3f} (n={len(v)})")

# --- L1 bridge (stage 2): zK acc per K ---
print("\n=== L1 bridge stage2: discrete acc by K (jepa) ===")
acc = defaultdict(list)
for r in s2:
    if r["world"] == "L1":
        acc[(r["model"], r.get("K"))].append(r["metrics"].get("linear_probe_acc_mean"))
for (model, K), v in sorted(acc.items()):
    if model == "jepa":
        print(f"  K={K:3}: {np.mean(v):.3f} (n={len(v)})")

# --- L2 same-metric table rows (stage 2 L2 vs stage 5 L2t zK) for the paper ---
print("\n=== H1a same-metric (paper table source): mean acc per (model, S, dim) ===")
l2 = defaultdict(list); l2t = defaultdict(list)
for r in s2 + s5:
    if "error" in r:
        continue
    v = r["metrics"].get("linear_probe_acc_mean")
    if r["world"] == "L2" and r.get("arm") != "l2m" and r.get("S") in (3, 5, 10) and r.get("dim") in (8, 32):
        l2[(r["model"], r.get("S"), r.get("dim"))].append(v)
    elif r["world"] == "L2t" and r.get("S") in (3, 5, 10) and r.get("dim") in (8, 32):
        l2t[(r["model"], r.get("S"), r.get("dim"))].append(v)
for k in sorted(set(l2) & set(l2t)):
    model, S, dim = k
    a2, a5 = np.mean(l2[k]), np.mean(l2t[k])
    print(f"  {model:12s} S={S:2} d={dim:2}: L2={a2:.3f} L2t-zK={a5:.3f} gap={a5-a2:+.3f}")

# --- L4 (stage 3) ---
print("\n=== L4 v2 stage3: acc by model ===")
acc = defaultdict(list)
for r in s3:
    if r["world"] == "L4":
        acc[r["model"]].append(r["metrics"].get("linear_probe_acc_mean"))
for model, v in sorted(acc.items()):
    print(f"  {model:12s}: {np.mean(v):.3f} (n={len(v)})")

# --- VICReg L0 (stage 6): alpha curve ---
print("\n=== VICReg L0 stage6: ridge by alpha ===")
acc = defaultdict(list)
for r in s6:
    if r["world"] == "L0":
        acc[r.get("alpha")].append(m(r))
for alpha, v in sorted(acc.items()):
    print(f"  alpha={alpha:7}: {np.mean(v):.3f} (n={len(v)})")

# --- L3 per-dim (stage 4, jepa vs recon) ---
print("\n=== L3 v2 per-dim acc (stage4) ===")
for r in s4:
    if r["world"] == "L3" and r["model"] in ("jepa", "recon") and r["seed"] == 0:
        print(f"  {r['model']}: per_dim={np.round(r['metrics']['discrete']['per_dim_acc'],3)}")

# --- decoupling per model per clean single-world stage ---
# Uses the committed instrument jepa_id.analyze.decoupling_index (A5/R4:
# within-stage z over ALL models pooled, pred sign negated so higher = better
# prediction). An earlier inline copy here recomputed it with an inverted pred
# sign and a literal 'ridge.mean' nested-key lookup, contradicting the paper;
# removed 2026-09-09 so paper_numbers has one source of truth.
print("\n=== decoupling index (A5/R4, stage-pooled z) ===")
from jepa_id.analyze import decoupling_index
def decoup(rows, label, stage):
    d = decoupling_index(rows, stage)
    if not d:
        print(f"  {label}: missing"); return
    for model, info in sorted(d.items()):
        print(f"  {label} {model:12s}: d={info['mean_d']:+.2f} "
              f"CI=[{info['ci'][0]:+.2f},{info['ci'][1]:+.2f}] (n={info['n']})")
decoup(s1b, "stage1b(L0+L1)", "1b")
decoup(s3, "stage3(L4)", "3")
decoup(s5, "stage5(L2t)", "5")
