"""Verify L1 bridge continuous R2 flatness (stage-2 rows) for paper prose."""
import json
from collections import defaultdict
import numpy as np

rows = []
for l in open("results/sweep_stage2.jsonl", encoding="utf-8"):
    l = l.strip()
    if l:
        r = json.loads(l)
        if "error" not in r and r.get("world") == "L1" and r.get("model") == "jepa":
            rows.append(r)
acc = defaultdict(list)
for r in rows:
    K = r.get("K")
    mm = r["metrics"]
    r2 = mm.get("ridge", {}).get("mean")
    accr = mm.get("linear_probe_acc_mean")
    acc[K].append((r2, accr, r.get("dim")))
for K in sorted(acc):
    vals = acc[K]
    r2s = [v[0] for v in vals if v[0] is not None]
    a = [v[1] for v in vals]
    dims = sorted(set(v[2] for v in vals))
    print(f"K={K:2} dims={dims}: ridgeR2 mean={np.mean(r2s):.3f} (n={len(r2s)})  acc_mean={np.mean(a):.3f} (n={len(a)})")
