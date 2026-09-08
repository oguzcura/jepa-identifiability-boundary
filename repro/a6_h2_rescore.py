"""A6 H2 re-scoring: AMI-based (primary per A6), purity as labeled secondary.
H2: SIGReg's Gaussian-forcing blurs discrete clusters -> recon AMI > jepa AMI
by >0.1 (bar UNCHANGED), CI excluding 0. Discrete worlds L2/L3/L4.
Audit (be11f8a) found purity (KMeans on full h, k=S) sits near floor for both
models on d>2 worlds -> gap ~0 by construction; AMI is per-dim probe-based.
"""
import json
import numpy as np
from collections import defaultdict
from scipy import stats

def load(stage):
    out = []
    for line in open(f"results/sweep_stage{stage}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line and "error" not in line:
            out.append(json.loads(line))
    return out

stages = {2: load(2), 3: load(3), 4: load(4), 5: load(5)}
# stage2 has L2 (S-dial) + L1; stage3 L4; stage4 L3 + L2m; stage5 L2t twins

def get(rows, model, seed, world, S=None, arm=None, dim=None):
    for r in rows:
        if (r["model"] == model and r["seed"] == seed and r["world"] == world
                and (S is None or r.get("S") == S) and (arm is None or r.get("arm") == arm)
                and (dim is None or r.get("dim") == dim)):
            return r
    return None

print("=== H2 re-score: recon AMI - jepa AMI (paired per seed, >0.1 bar) ===")
cells = []
# L2 S-dial (stage 2)
for S in (3, 5, 10):
    for dim in (8, 32):
        cells.append((f"L2 S={S} d={dim}", 2, "L2", S, None, dim))
# L4 (stage 3)
cells.append(("L4", 3, "L4", None, None, None))
# L3 v2 (stage 4)
cells.append(("L3 v2", 4, "L3", None, None, None))
# L2m (stage 4 control)
cells.append(("L2m S=3", 4, "L2", 3, "l2m", 3))

for label, st, world, S, arm, dim in cells:
    gaps_ami, gaps_pur = [], []
    for seed in range(5):
        rj = get(stages[st], "jepa", seed, world, S, arm, dim)
        rr = get(stages[st], "recon", seed, world, S, arm, dim)
        if rj is None or rr is None:
            print(f"  MISSING {label} seed={seed}")
            continue
        mj = rj["metrics"]
        mr = rr["metrics"]
        gaps_ami.append(mr["ami_mean"] - mj["ami_mean"])
        gaps_pur.append(mr["purity_mean"] - mj["purity_mean"])
    ga = np.array(gaps_ami); gp = np.array(gaps_pur)
    if len(ga) < 5:
        continue
    t, p = stats.ttest_1samp(ga, 0)
    ci = stats.t.interval(0.95, len(ga) - 1, loc=ga.mean(),
                          scale=ga.std(ddof=1) / np.sqrt(len(ga)))
    bar = ">0.1 PASS" if ga.mean() > 0.1 and ci[0] > 0 else "null"
    print(f"  {label:14s}: AMI gap={ga.mean():+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}] "
          f"t={t:.2f} p={p:.4f} | purity gap={gp.mean():+.3f} [{bar}]")
