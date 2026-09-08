"""Paper decoupling numbers via the repo's committed analyze.decoupling_index."""
import sys
sys.path.insert(0, "src")
from jepa_id.analyze import decoupling_index, load_rows, recovery_metric

import json

def load(st):
    return load_rows([f"results/sweep_stage{st}.jsonl"])

for st, label in [("1b", "L0 gate"), ("3", "L4"), ("5", "L2t")]:
    rows = load(st)
    d = decoupling_index(rows, st)
    print(f"stage {st} ({label}): n={len(rows)}")
    for model, info in sorted(d.items()):
        print(f"   {model:12s}: d={info['mean_d']:+.2f} CI=[{info['ci'][0]:+.2f},{info['ci'][1]:+.2f}] n={info['n']}")
