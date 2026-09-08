"""P6 final analysis runner — pre-registered confirmatory report (sections 3, 9).

Usage:  PYTHONPATH=src uv run python repro/analyze_report.py [--out notes/analysis_YYYYMMDD.md]

Loads every sweep_stage*.jsonl and computes the pre-registered contrasts:
  * H1a: L2 (stage 2, discrete acc) vs L2t (stage 5, continuous ridge R^2) —
        moment-matched twin; gap > 0.3 with CI excluding it (frozen rule)
  * H1b: L3 v2 (stage 4) vs L2m (stage 4) — structure at matched entropy
  * decoupling index (A5) per stage/model
Honest-null discipline: decision rules are the frozen pre-registered ones.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from jepa_id.analyze import (
    load_rows, h1a_test, h1b_test, decoupling_index, paired_delta_ci,
    paired_t_pval,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_all() -> dict[str, list[dict]]:
    stages = {}
    for p in sorted(RESULTS.glob("sweep_stage*.jsonl")):
        stages[p.stem] = load_rows([str(p)])
    return stages


def _row_groups(rows: list[dict]) -> dict:
    """rows keyed by (world, model, S, dim, arm)."""
    g = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("world"), r.get("model"), r.get("S"), r.get("dim"),
               r.get("arm", ""))
        g.setdefault(key, []).append(r)
    return g


def fmt_ci(a, b, seed=3):
    d = np.array(a, float) - np.array(b, float)
    lo, hi = paired_delta_ci(np.array(b, float), np.array(a, float), seed=seed)
    p = paired_t_pval(np.array(b, float), np.array(a, float))
    return d.mean(), lo, hi, p


def report(stages: dict[str, list[dict]]) -> str:
    L = []
    L.append(f"# P6 confirmatory analysis — {date.today().isoformat()}")
    L.append("Pre-registered decision rules in force (frozen + A1-A5 append-only).")
    L.append("")

    # ---- H1a: stage-2 L2 vs stage-5 L2t ----------------------------------
    s2 = stages.get("sweep_stage2", [])
    s5 = stages.get("sweep_stage5", [])
    l2_all = [r for r in s2 if r.get("world") == "L2"]
    l2t_all = [r for r in s5 if r.get("world") == "L2t"]
    if l2_all and l2t_all:
        L.append("## H1a: discrete L2 vs moment-matched continuous twin L2t")
        L.append("(frozen rule: R_L2 below the continuous prediction by >0.3 abs,")
        L.append(" 95% CI excluding that bound; paired by seed; cont = L2t ridge R2)")
        seen = set()
        for r in l2t_all:
            key = (r["S"], r["dim"])
            if key in seen:
                continue
            seen.add(key)
            S, dim = key
            for model in ("jepa", "recon", "contrastive"):
                res = h1a_test(l2_all, l2t_all, S=S, dim=dim, model=model)
                if res.get("n_seeds", 0) < 2:
                    continue
                L.append(
                    f"- S={S} dim={dim} {model:12s}: disc={res['R_L2_discrete']:.3f} "
                    f"twinR2={res['R_L2t_continuous']:.3f} "
                    f"gap={res['gap_cont_minus_disc']:+.3f} "
                    f"(CI {res['ci_95'][0]:.3f},{res['ci_95'][1]:.3f}) "
                    f"p={res['p_paired_t']:.4f} -> "
                    f"**{'PASS' if res['passed_03_bar'] else 'FAIL'}** "
                    f"(n={res['n_seeds']})")
        L.append("")

    # ---- H1b: stage-4 L3 v2 vs L2m ----------------------------------------
    s4 = stages.get("sweep_stage4", [])
    if s4:
        L.append("## H1b: L3 v2 (rule grammar) vs L2m (entropy-matched iid)")
        L.append("(frozen rule: R_L3 - R_L2m > 0.2, 95% CI excluding 0)")
        for model in ("jepa", "recon", "contrastive"):
            res = h1b_test(s4, model)
            if res.get("n_seeds", 0) < 2:
                continue
            L.append(
                f"- {model:12s}: R_L3={res['R_L3']:.3f} R_L2m={res['R_L2m']:.3f} "
                f"delta={res['delta']:+.3f} (CI {res['ci_95'][0]:.3f},"
                f"{res['ci_95'][1]:.3f}) p={res['p_paired_t']:.4f} -> "
                f"**{'PASS' if res['passed_02_bar'] else 'FAIL'}** "
                f"(n={res['n_seeds']})")
        L.append("")

    # ---- Decoupling index per stage ---------------------------------------
    L.append("## Decoupling index (A5: z-scored pred-loss minus z-scored recovery)")
    for st, rows in stages.items():
        di = decoupling_index(rows, st)
        if not di:
            continue
        parts = [f"{m}: d={v['mean_d']:+.3f} (CI {v['ci'][0]:.3f},{v['ci'][1]:.3f})"
                 for m, v in di.items()]
        L.append(f"- {st}: " + "; ".join(parts))
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stages = load_all()
    txt = report(stages)
    out = args.out or (ROOT / "notes" / f"analysis_{date.today().isoformat()}.md")
    Path(out).write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
