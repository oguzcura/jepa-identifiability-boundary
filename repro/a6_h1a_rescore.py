"""A6 H1a re-scoring: same-metric comparison (L2 acc_mean vs L2t zK acc_mean),
paired by (S, dim, model, seed). The earlier 'partial pass at S=10' compared
acc (chance 1/S) against ridge R^2 (chance 0) — metric mixing (audit be11f8a).

Primary readout per A6: linear_probe_acc_mean on both arms (L2's own discrete
z; L2t's zK = S equiprobable bins of its continuous z — same information
content, same S-way classification task). Bar UNCHANGED: >0.3, CI excl bound.
Ridge R^2 reported as labeled secondary (continuous-tracking sanity only).
"""
import json
import numpy as np
from scipy import stats

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["jepa", "recon", "contrastive"]
S_GRID = [3, 5, 10]
DIMS = [8, 32]

def load(stage):
    out = []
    for line in open(f"results/sweep_stage{stage}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line and "error" not in line:
            out.append(json.loads(line))
    return out

s2 = load(2)  # L2 discrete (S-dial)
s5 = load(5)  # L2t twins (same S, dim, seed)

def find(rows, world, model, seed, S, dim, arm=None):
    for r in rows:
        if (r["world"] == world and r["model"] == model and r["seed"] == seed
                and r.get("S") == S and r["dim"] == dim
                and (arm is None or r.get("arm") == arm)):
            return r
    return None

print("=== H1a SAME-METRIC re-score (L2 acc vs L2t zK acc) ===")
print(f"{'cell':26s} {'L2 acc':>7s} {'zK acc':>7s} {'gap':>7s}  verdict")
worst = []
for model in MODELS:
    for S in S_GRID:
        for dim in DIMS:
            gaps = []
            for seed in SEEDS:
                r2 = find(s2, "L2", model, seed, S, dim)
                r5 = find(s5, "L2t", model, seed, S, dim)
                if r2 is None or r5 is None:
                    print(f"  MISSING {model} S={S} d={dim} seed={seed}")
                    continue
                a2 = r2["metrics"]["linear_probe_acc_mean"]
                a5 = r5["metrics"]["linear_probe_acc_mean"]
                gaps.append(a5 - a2)   # H1a direction: continuous − discrete
            gaps = np.array(gaps)
            if len(gaps) < 5:
                continue
            t, p = stats.ttest_1samp(gaps, 0)
            ci = stats.t.interval(0.95, len(gaps) - 1, loc=gaps.mean(),
                                  scale=gaps.std(ddof=1) / np.sqrt(len(gaps)))
            worst.append((abs(gaps.mean()), model, S, dim))
            bar = ">0.3 PASS" if gaps.mean() > 0.3 and ci[0] > 0 else "null"
            print(f"{model} S={S:2d} d={dim:2d}          {gaps.mean():+.3f} "
                  f"{ci[1]:+.3f} ci-lo={ci[0]:+.3f} p={p:.4f} t={t:.2f} [{bar}]")
worst.sort(reverse=True)
print(f"\nmax |same-metric gap| across all cells: {worst[0][0]:.3f} "
      f"({worst[0][1]} S={worst[0][2]} d={worst[0][3]})")

# Secondary: ridge R^2 (the metric A5 used) — shown to demonstrate the scale gap
# NOTE: the paper quotes seed-MEANS here (L2 acc 0.456 vs L2t ridge 0.943);
# an earlier version printed seed-0 only, which read as a mismatch. Fixed
# 2026-09-09 to report the 5-seed mean, matching the paper.
print("\n=== secondary: L2t ridge R^2 at same cells (A5's old metric, for contrast) ===")
for model in ["jepa"]:
    for S in [10]:
        for dim in [8, 32]:
            r5s = [find(s5, "L2t", model, s, S, dim) for s in range(5)]
            r2s = [find(s2, "L2", model, s, S, dim) for s in range(5)]
            r5s = [r for r in r5s if r]
            r2s = [r for r in r2s if r]
            if r5s and r2s:
                ridge = float(np.mean([r['metrics']['ridge']['mean'] for r in r5s]))
                acc = float(np.mean([r['metrics']['linear_probe_acc_mean'] for r in r2s]))
                zK = float(np.mean([r['metrics']['linear_probe_acc_mean'] for r in r5s]))
                print(f"  {model} S={S} d={dim}: L2 acc={acc:.3f} "
                      f"L2t ridge={ridge:.3f} L2t zK acc={zK:.3f}")
