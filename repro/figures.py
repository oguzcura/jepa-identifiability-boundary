"""P6 figure generation — pre-planned figures F1-F5 (pre-registration section 9).

Reads results/sweep_stage*.jsonl, writes PNGs to figures/. Runs on whatever
data exists; missing stages are skipped with a note. Deterministic seeds.

Figures:
  F1: L0 Fig-4b reproduction (ridge R^2 vs alpha, per model + CI) — sanity gate.
  F2: boundary map — recovery vs discreteness (L1 K-dial, L2 S-dial) 3 model curves.
  F3: decoupling scatter — prediction loss (x) vs recovery (y) per world/model.
  F4: compositional-vs-unstructured (L3 vs L2m) contrast for H1b.
  F5: SIGReg cluster-blur portrait (collapse effective rank vs alpha) for H2.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jepa_id.analyze import load_rows, recovery_metric, paired_delta_ci

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# consistent style
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
MODEL_STYLE = {
    "jepa": {"color": "#1f77b4", "marker": "o", "label": "LeJEPA (SIGReg)"},
    "recon": {"color": "#2ca02c", "marker": "s", "label": "Recon (input-space)"},
    "contrastive": {"color": "#d62728", "marker": "^", "label": "InfoNCE"},
    "vicreg": {"color": "#9467bd", "marker": "D", "label": "VICReg"},
}


def _ci_band(x: np.ndarray, n_boot: int = 500, seed: int = 0):
    """Per-x mean + percentile bootstrap CI over the x samples (rows)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        means[i] = x[idx].mean()
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _group_cells(rows: list[dict], world: str | None = None,
                 arm: str | None = None) -> dict[tuple, list[dict]]:
    """Group rows into cells by (world, model, dial, seed...) — return per-cell seed lists."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        if world is not None and r.get("world") != world:
            continue
        if arm is not None and r.get("arm") != arm:
            continue
        key = (r.get("world"), r.get("model"),
               r.get("alpha"), r.get("K"), r.get("S"), r.get("dim"))
        cells.setdefault(key, []).append(r)
    return cells


# --------------------------------------------------------------------------- #
# F1: L0 alpha-sweep (Fig-4b reproduction sanity)
# --------------------------------------------------------------------------- #
def f1(rows: list[dict], out: Path = FIG / "F1_l0_fig4b.png") -> Path:
    sub = [r for r in rows if r.get("world") == "L0" and r.get("dim") in (2, 8)]
    if not sub:
        print("F1: no L0 rows"); return out
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ("jepa", "recon", "contrastive", "vicreg"):
        xs = sorted({r["alpha"] for r in sub if r.get("model") == model})
        if not xs:
            continue
        means, los, his = [], [], []
        for a in xs:
            vals = [recovery_metric(r, "1") for r in sub
                    if r.get("model") == model and r["alpha"] == a]
            if vals:
                m, lo, hi = _ci_band(vals)
                means.append(m); los.append(lo); his.append(hi)
        style = MODEL_STYLE[model]
        ax.plot(xs, means, style["marker"] + "-", color=style["color"],
                label=style["label"])
        ax.fill_between(xs, los, his, color=style["color"], alpha=0.15)
    ax.axvline(2.0, color="gray", ls=":", lw=1, label=r"$\alpha=2$ (Gaussian)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2 ** i for i in range(-3, 6)])
    ax.set_xticklabels([f"2^{i}" for i in range(-3, 6)])
    ax.set_xlabel(r"latent shape $\alpha$ (generalized normal)")
    ax.set_ylabel("recovery (ridge R²)")
    ax.set_title("F1: L0 α-sweep — Fig-4b reproduction (sanity gate)")
    ax.legend(frameon=False)
    fig.savefig(out); plt.close(fig)
    print(f"F1 written: {out}")
    return out


# --------------------------------------------------------------------------- #
# F2: boundary map (recovery vs discreteness)
# --------------------------------------------------------------------------- #
def f2(rows: list[dict], out: Path = FIG / "F2_boundary_map.png") -> Path:
    """Two panels: L1 K-dial (discrete acc vs K) and L2 S-dial (acc vs S)."""
    l1 = [r for r in rows if r.get("world") == "L1" and r.get("dim") == 8]
    l2 = [r for r in rows if r.get("world") == "L2" and r.get("dim") == 8 and not r.get("arm")]
    if not l1 and not l2:
        print("F2: no L1/L2 rows"); return out
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sub, dial, xlabel, title in (
        (axes[0], l1, "K", "discretization K (bins)", "F2a: L1 continuous→discrete bridge"),
        (axes[1], l2, "S", "Markov states S", "F2b: L2 categorical dial"),
    ):
        for model in ("jepa", "recon", "contrastive", "vicreg"):
            xs = sorted({r[dial] for r in sub if r.get("model") == model})
            means, los, his = [], [], []
            for x in xs:
                vals = [r["metrics"].get("linear_probe_acc_mean") for r in sub
                        if r.get("model") == model and r[dial] == x]
                vals = [v for v in vals if v is not None]
                if vals:
                    m, lo, hi = _ci_band(vals)
                    means.append(m); los.append(lo); his.append(hi)
            if not means:
                continue
            style = MODEL_STYLE[model]
            ax.plot(xs, means, style["marker"] + "-", color=style["color"], label=style["label"])
            ax.fill_between(xs, los, his, color=style["color"], alpha=0.15)
        ax.set_xlabel(xlabel); ax.set_ylabel("recovery (discrete acc_mean)")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print(f"F2 written: {out}")
    return out


# --------------------------------------------------------------------------- #
# F3: decoupling scatter
# --------------------------------------------------------------------------- #
def f3(rows: list[dict], out: Path = FIG / "F3_decoupling.png") -> Path:
    """pred_loss (x) vs recovery (y) per world; per-model color; annotate worlds."""
    if not rows:
        print("F3: no rows"); return out
    fig, ax = plt.subplots(figsize=(7.5, 6))
    world_color = {"L0": "tab:blue", "L1": "tab:orange", "L2": "tab:green",
                   "L2t": "tab:cyan", "L3": "tab:red", "L4": "tab:purple"}
    for r in rows:
        m = r["metrics"]
        rec = recovery_metric(r, "x")
        pred = m.get("pred_loss")
        if rec is None or pred is None or np.isnan(pred):
            continue
        w = r.get("world")
        ax.scatter(pred, rec, color=world_color.get(w, "gray"), alpha=0.45, s=14,
                   edgecolors="none")
    # per-world centroids
    for w, col in world_color.items():
        ws = [r for r in rows if r.get("world") == w]
        if not ws:
            continue
        preds = [r["metrics"].get("pred_loss") for r in ws]
        recs = [recovery_metric(r, "x") for r in ws]
        preds = [p for p in preds if p is not None]
        if preds and recs:
            ax.scatter(np.mean(preds), np.mean(recs), s=90, color=col, marker="X",
                       edgecolor="k", linewidth=0.8, zorder=5)
            ax.annotate(w, (np.mean(preds), np.mean(recs)),
                        textcoords="offset points", xytext=(7, 7), fontsize=9)
    ax.set_xlabel("prediction loss (lower = better prediction)")
    ax.set_ylabel("state recovery")
    ax.set_title("F3: decoupling — predicting well vs understanding")
    fig.savefig(out); plt.close(fig)
    print(f"F3 written: {out}")
    return out


# --------------------------------------------------------------------------- #
# F4: compositional vs unstructured (H1b)
# --------------------------------------------------------------------------- #
def f4(rows: list[dict], out: Path = FIG / "F4_l3_vs_l2m.png") -> Path:
    l3 = [r for r in rows if r.get("world") == "L3"]
    # A6 correction: H1b control is the marginal-matched L2mH (2,4,4) from
    # stage 7, NOT the old homogeneous L2m S=3 (state-count confounded).
    l2m = [r for r in rows if r.get("world") == "L2" and r.get("arm") == "l2mh"]
    if not l3 or not l2m:
        print("F4: no L3/L2mH rows"); return out
    fig, ax = plt.subplots(figsize=(6.5, 5))
    models = [mo for mo in ("jepa", "recon", "contrastive", "vicreg")
              if any(r.get("model") == mo for r in l3)
              and any(r.get("model") == mo for r in l2m)]
    xpos = np.arange(len(models))
    width = 0.36
    for offset, (arm, col, lab) in enumerate(((-width / 2, "#4c72b0", "L3 (rule grammar)"),
                                              (width / 2, "#dd8452", "L2mH (2,4,4) marginal-matched iid"))):
        means, errs = [], []
        for model in models:
            vals_a = np.array([r["metrics"]["linear_probe_acc_mean"] for r in l3
                               if r.get("model") == model])
            vals_b = np.array([r["metrics"]["linear_probe_acc_mean"] for r in l2m
                               if r.get("model") == model])
            vals = vals_a if arm == "L3 (rule grammar)" else vals_b
            means.append(vals.mean())
            errs.append(vals.std() / np.sqrt(len(vals)))
        ax.bar(xpos + offset, means, width, yerr=errs, color=col, label=lab, alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels([MODEL_STYLE[m]["label"] for m in models], rotation=12)
    ax.set_ylabel("recovery (discrete acc_mean, mean ± se over seeds)")
    ax.set_title("F4: H1b — compositional vs marginal-matched unstructured (A6 control)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print(f"F4 written: {out}")
    return out


# --------------------------------------------------------------------------- #
# F5: SIGReg cluster-blur portrait (H2)
# --------------------------------------------------------------------------- #
def f5(rows: list[dict], out: Path = FIG / "F5_sigreg_blur.png") -> Path:
    """Effective rank + max_frac (collapse) vs alpha for jepa — Gaussianizing
    should make embeddings full-rank-ish (blur) while discrete worlds collapse."""
    sub = [r for r in rows if r.get("world") == "L0" and r.get("model") == "jepa"]
    if not sub:
        print("F5: no L0 jepa rows"); return out
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = sorted({r["alpha"] for r in sub})
    effs = [np.mean([r["metrics"]["collapse"]["effective_rank"] for r in sub
                     if r["alpha"] == a]) for a in xs]
    ax.plot(xs, effs, "o-", color="#1f77b4", label="effective rank (h)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2 ** i for i in range(-3, 6)])
    ax.set_xticklabels([f"2^{i}" for i in range(-3, 6)])
    ax.set_xlabel(r"latent shape $\alpha$")
    ax.set_ylabel("embedding effective rank")
    ax.set_title("F5: H2 portrait — does SIGReg keep embeddings full-rank?")
    ax.legend(frameon=False)
    fig.savefig(out); plt.close(fig)
    print(f"F5 written: {out}")
    return out


def make_all(rows_by_stage: dict[str, list[dict]] | None = None) -> list[Path]:
    if rows_by_stage is None:
        rows_by_stage = {}
        for p in sorted(RESULTS.glob("sweep_stage*.jsonl")):
            rows_by_stage[p.stem] = load_rows([str(p)])
    all_rows = [r for rs in rows_by_stage.values() for r in rs]
    outs = [f1(all_rows), f2(all_rows), f3(all_rows), f4(all_rows), f5(all_rows)]
    return outs


if __name__ == "__main__":
    paths = sorted(RESULTS.glob("sweep_stage*.jsonl"))
    print(f"{len(paths)} result files found")
    make_all()
