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
def recovery_metric(row: dict, stage: str) -> float:
    """Schema-aware recovery score for a results row.

    Continuous worlds (L0/L1/L2t) nest recovery under ridge.mean (ridge R^2);
    pure-discrete worlds (L2/L3/L4) emit flat linear_probe_acc_mean. Both are
    0-1-ish bounded scores of the same role (state recovery).
    """
    m = row["metrics"]
    if "ridge" in m and isinstance(m["ridge"], dict) and "mean" in m["ridge"]:
        return float(m["ridge"]["mean"])
    if "linear_probe_acc_mean" in m:
        return float(m["linear_probe_acc_mean"])
    if "mlp_control" in m and isinstance(m["mlp_control"], dict):
        return float(m["mlp_control"].get("mean", np.nan))
    return float(np.nan)


def decoupling_index(rows: list[dict], stage: str) -> dict:
    """Within-stage z-scored prediction-vs-recovery gap per model.

    A5 rule: z(pred_loss) and z(recovery) are computed over the STAGE's cells
    (all models pooled — otherwise each model's mean d is ~0 by construction),
    then per-model mean of d_i = z_pred_quality_i - z_recovery_i where
    prediction quality = -z(pred_loss) (higher = better prediction). Positive
    d = predicts better than stage-average while recovering worse.
    """
    cells = [r for r in rows if "error" not in r]
    rec_all = np.array([recovery_metric(r, stage) for r in cells], dtype=float)
    pred_all = np.array([r["metrics"].get("pred_loss", np.nan) for r in cells],
                        dtype=float)
    ok = ~np.isnan(rec_all) & ~np.isnan(pred_all)
    rec_all, pred_all = rec_all[ok], pred_all[ok]
    if len(rec_all) < 4:
        return {}
    # stage-wide location/scale
    mu_r, sd_r = rec_all.mean(), rec_all.std() + 1e-12
    mu_p, sd_p = pred_all.mean(), pred_all.std() + 1e-12
    out = {}
    for model in ("jepa", "recon", "contrastive", "vicreg"):
        idx = [i for i, r in enumerate(cells) if r.get("model") == model and ok[i]]
        if len(idx) < 2:
            continue
        i = np.array(idx)
        zr = (rec_all[i] - mu_r) / sd_r
        zp = -(pred_all[i] - mu_p) / sd_p   # higher = better prediction
        d = zp - zr
        # Deterministic per-model bootstrap seed: str.__hash__() is
        # process-randomized (PYTHONHASHSEED), which made CI endpoints drift
        # between runs. crc32 is stable across processes and platforms.
        import zlib
        lo, hi = bootstrap_ci(d, seed=zlib.crc32(model.encode("utf-8")))
        out[model] = {"mean_d": float(d.mean()), "ci": (lo, hi),
                      "per_cell": d.tolist(), "n": int(len(d))}
    return out

# --------------------------------------------------------------------------- #
# Row loading + pairing helpers
# --------------------------------------------------------------------------- #
import json as _json


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
    return rows


def _arm_values(rows: list[dict], model: str, stage: str,
                world: str | None = None, S: int | None = None,
                dim: int | None = None, arm: str | None = None) -> dict[int, float]:
    """Per-seed recovery values for one (model, world/S/dim/arm) cell."""
    out = {}
    for r in rows:
        if r.get("model") != model:
            continue
        if world is not None and r.get("world") != world:
            continue
        if S is not None and r.get("S") != S:
            continue
        if dim is not None and r.get("dim") != dim:
            continue
        if arm is not None and r.get("arm") != arm:
            continue
        out[r["seed"]] = recovery_metric(r, stage)
    return out


# --------------------------------------------------------------------------- #
# H-family decision rules (pre-registration section 3, amendments A4-A5)
# --------------------------------------------------------------------------- #
def h1b_test(rows: list[dict], model: str = "jepa") -> dict:
    """H1b (A4): R_L3v2 - R_L2m > 0.2, 95% CI excluding 0. Paired by seed."""
    l3 = _arm_values(rows, model, "4", world="L3")
    l2m = _arm_values(rows, model, "4", world="L2", arm="l2m")
    seeds = sorted(set(l3) & set(l2m))
    a = np.array([l3[s] for s in seeds])
    b = np.array([l2m[s] for s in seeds])
    d = a - b
    lo, hi = paired_delta_ci(a, b, seed=7)
    p = paired_t_pval(a, b)
    return {
        "model": model, "n_seeds": len(seeds),
        "R_L3": float(a.mean()), "R_L2m": float(b.mean()),
        "delta": float(d.mean()), "ci_95": (lo, hi),
        "d_cohens": float(cohens_d(a, b)),
        "passed_02_bar": bool(d.mean() > 0.2 and lo > 0),
        "passed_ci_only": bool(lo > 0),
        "p_paired_t": float(p),
        "per_seed_delta": d.tolist(),
    }


def h1a_test(rows_l2: list[dict], rows_l2t: list[dict], S: int, dim: int,
             model: str = "jepa") -> dict:
    """H1a: R_L2 below the moment-matched continuous prediction by > 0.3 abs,
    95% CI excluding that bound. L2t continuous ridge R^2 = the continuous
    prediction; L2 discrete acc_mean = the discrete reality. Paired by seed."""
    disc = _arm_values(rows_l2, model, "2", world="L2", S=S, dim=dim)
    cont = _arm_values(rows_l2t, model, "5", world="L2t", S=S, dim=dim)
    seeds = sorted(set(disc) & set(cont))
    a = np.array([disc[s] for s in seeds])   # discrete
    b = np.array([cont[s] for s in seeds])   # continuous twin
    gap = b - a  # continuous prediction MINUS discrete recovery (want > 0.3)
    lo, hi = paired_delta_ci(b, a, seed=11)
    p = paired_t_pval(b, a)
    return {
        "S": S, "dim": dim, "model": model, "n_seeds": len(seeds),
        "R_L2_discrete": float(a.mean()), "R_L2t_continuous": float(b.mean()),
        "gap_cont_minus_disc": float(gap.mean()), "ci_95": (lo, hi),
        "d_cohens": float(cohens_d(b, a)),
        "passed_03_bar": bool(gap.mean() > 0.3 and lo > 0.3),
        "p_paired_t": float(p),
        "per_seed_gap": gap.tolist(),
    }
