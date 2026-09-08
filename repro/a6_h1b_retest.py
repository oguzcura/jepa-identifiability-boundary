"""A6 confirmatory H1b re-test: R_L3 vs corrected controls (paired by seed).

Design (paired, same seed/model/hyperparams):
  treatment = stage-4 L3 rows (jepa/recon/contrastive, 5 seeds)
  control-A = stage-7 L2mH (2,4,4) — marginal-matched, independent dims
  control-B = stage-7 L2mS2 (S=2) — joint-entropy-matched (8 states/3 bits)

Readout: acc_mean (both discrete S-way per-dim; L3 and L2mH share per-dim
chance floors 0.5/0.25/0.25; L2mS2 all 2-way chance 0.5 — note in report).

Decision rule (frozen, A6 restated): R_L3 - R_control > 0.2 with 95% CI
excluding 0 (one-sided bar on the point estimate; two-sided CI reported).

Per statistical-analysis skill: paired t + bootstrap CI + Cohen's dz effect
size + assumption note (n=5, normality not testable — report deltas).
"""
import json
import numpy as np
from scipy import stats

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["jepa", "recon", "contrastive"]

def load(stage, path=None):
    p = path or f"results/sweep_stage{stage}.jsonl"
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        out.append(r)
    return out

def row_key(r, model):
    return (r["model"], r["seed"])

def get_rec(rows, model, seed, world="L3", arm=None):
    for r in rows:
        if r["model"] == model and r["seed"] == seed:
            if world == "L3":
                if r["world"] == "L3":
                    return r
            else:
                if r["world"] == world and r.get("arm") == arm:
                    return r
    raise KeyError(f"missing {world}/{arm}/{model}/{seed}")

s4 = load(4)   # stage 4: L3 v2 (15 rows) + L2m S=3 (15 rows, old control)
s7 = load(7)   # stage 7: L2mH + L2mS2 (30 rows)

def paired_table(ctl_rows, ctl_arm, ctl_label):
    print(f"\n=== H1b: L3 vs {ctl_label} (paired by seed) ===")
    for model in MODELS:
        d = []
        for seed in SEEDS:
            r3 = get_rec(s4, model, seed, "L3")
            rc = get_rec(ctl_rows, model, seed, "L2", ctl_arm)
            a3 = r3["metrics"]["linear_probe_acc_mean"]
            ac = rc["metrics"]["linear_probe_acc_mean"]
            d.append(a3 - ac)
        d = np.array(d)
        t, p = stats.ttest_1samp(d, 0)
        # two-sided 95% CI on the mean delta
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=d.std(ddof=1) / np.sqrt(len(d)))
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        mean_l3 = np.mean([get_rec(s4, model, s, "L3")["metrics"]["linear_probe_acc_mean"] for s in SEEDS])
        mean_ctl = np.mean([get_rec(ctl_rows, model, s, "L2", ctl_arm)["metrics"]["linear_probe_acc_mean"] for s in SEEDS])
        bar = "PASS (>0.2)" if (d.mean() > 0.2 and ci[0] > 0) else "fail bar"
        print(f"  {model:12s}: L3={mean_l3:.3f} ctl={mean_ctl:.3f} "
              f"d={d.mean():+.3f} CI95=[{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"t={t:.2f} p={p:.4f} dz={dz:.2f} [{bar}]")
        print(f"             per-seed deltas: {np.round(d, 3)}")

paired_table(s7, "l2mh", "L2mH (2,4,4) marginal-matched")
paired_table(s7, "l2ms2", "L2mS2 (S=2) joint-matched")
