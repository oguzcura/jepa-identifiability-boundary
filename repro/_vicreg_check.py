"""Verify VICReg discrete-ladder numbers + stage-7 per-seed deltas for the paper."""
import json
from collections import defaultdict
import numpy as np

rows = []
for l in open("results/sweep_stage6.jsonl", encoding="utf-8"):
    l = l.strip()
    if l:
        r = json.loads(l)
        if "error" not in r:
            rows.append(r)

# VICReg discrete arms: L3, L2/l2m, L4
print("=== VICReg discrete (stage 6) ===")
acc = defaultdict(list)
for r in rows:
    if r["world"] == "L3":
        acc[("L3", "")].append(r["metrics"].get("linear_probe_acc_mean"))
    elif r["world"] == "L2" and r.get("arm") == "l2m":
        acc[("L2", "l2m")].append(r["metrics"].get("linear_probe_acc_mean"))
    elif r["world"] == "L4":
        acc[("L4", "")].append(r["metrics"].get("linear_probe_acc_mean"))
    elif r["world"] == "L2" and r.get("arm") == "s2vicreg":
        acc[("L2", f"s={r.get('S')}")].append(r["metrics"].get("linear_probe_acc_mean"))
for k in sorted(acc, key=str):
    v = [x for x in acc[k] if x is not None]
    print(f"  {k}: mean={np.mean(v):.3f} (n={len(v)})")

# stage-7 jepa deltas per seed for L3 (stage4) vs L2mH (stage7)
def load(st):
    out = []
    for l in open(f"results/sweep_stage{st}.jsonl", encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            if "error" not in r:
                out.append(r)
    return out

s4, s7 = load(4), load(7)
print("\n=== stage-7 per-seed (jepa L3 vs L2mH) ===")
l3 = {r["seed"]: r["metrics"]["linear_probe_acc_mean"] for r in s4 if r["world"] == "L3" and r["model"] == "jepa"}
lm = {r["seed"]: r["metrics"]["linear_probe_acc_mean"] for r in s7 if r.get("arm") == "l2mh" and r["model"] == "jepa"}
d = np.array([l3[s] - lm[s] for s in sorted(l3)])
print("  L3:", [round(l3[s], 3) for s in sorted(l3)])
print("  L2mH:", [round(lm[s], 3) for s in sorted(lm)])
print("  deltas:", [round(x, 3) for x in d], "mean=", round(d.mean(), 3))
