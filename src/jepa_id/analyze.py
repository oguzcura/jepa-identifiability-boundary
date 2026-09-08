"""Confirmatory analysis machinery for the identifiability ladder (P6).

Implements the pre-registered statistics (pre_registration_2026-09-06.md §9):
- bootstrap CIs over seeds (n_boot resamples; replication unit = training run)
- Holm-corrected hypothesis family (H0/H1a/H1b/H2)
- Cohen's d effect sizes
- R4 decoupling index (A5): within-stage z-scored (pred_loss - recovery) gap

Pure-numpy so it runs anywhere and is unit-testable without GPU.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def holm_correct(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni: sort p ascending, reject while p_k <= alpha/(m-k+1).

    Returns boolean array aligned with the INPUT order.
    """
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    rej_sorted = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            rej_sorted[rank] = True
        else:
            break  # Holm is step-down: once one fails, all later fail
    rej = np.zeros(m, dtype=bool)
    rej[order] = rej_sorted
    return rej


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d (positive = a > b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI over the sample mean (resample WITH replacement).

    Replication unit note: x should be per-seed values (n=5 typically) — each
    seed is an independent training run; resampling over runs is the
    pre-registered procedure.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        means[i] = x[idx].mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def paired_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
                    seed: int = 0, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI of the paired difference mean(a-b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired: same length required"
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        means[i] = d[idx].mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def paired_t_pval(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test p-value (scipy-free fallback via t CDF)."""
    from scipy.stats import t as tdist
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() != 0 else 1.0
    tstat = d.mean() / (sd / np.sqrt(n))
    return float(2 * tdist.sf(abs(tstat), df=n - 1))


# --------------------------------------------------------------------------- #
# R4 decoupling index (A5 §3)
# --------------------------------------------------------------------------- #
def decoupling_index(rows: list[dict], stage: str) -> dict:
    """Within-stage z-scored prediction-vs-recovery gap per model.

    For each model: d_i = z(pred_loss_i) - z(recovery_i) over the stage's cells
    (prediction = pred_loss, recovery = stage-appropriate primary metric).
    Positive d = predicts better than stage-average while recovering worse.
    Returns per-model mean d + bootstrap CI + the per-cell values.
    """
    recovery_metric = "ridge_r2_mean" if stage in ("1", "1b", "pilot") else "linear_probe_acc_mean"
    out = {}
    for model in ("jepa", "recon", "contrastive", "vicreg"):
        cells = [r for r in rows if r.get("model") == model]
        if not cells:
            continue
        rec = np.array([r["metrics"].get(recovery_metric)
                        or r["metrics"].get("linear_probe_acc_mean", np.nan)
                        for r in cells], dtype=float)
        pred = np.array([r["metrics"].get("pred_loss", np.nan) for r in cells], dtype=float)
        ok = ~np.isnan(rec) & ~np.isnan(pred)
        if ok.sum() < 2:
            continue
        rec, pred = rec[ok], pred[ok]
        # z-scored (higher pred_loss = worse prediction; negate so "better
        # prediction" is positive like recovery)
        zp = -(pred - pred.mean()) / (pred.std() + 1e-12)
        zr = (rec - rec.mean()) / (rec.std() + 1e-12)
        d = zp - zr
        lo, hi = bootstrap_ci(d, seed=model.__hash__() % 2**32)
        out[model] = {"mean_d": float(d.mean()), "ci": (lo, hi),
                      "per_cell": d.tolist()}
    return out
