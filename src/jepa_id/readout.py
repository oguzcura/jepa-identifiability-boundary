"""Identifiability readouts — mechanically measure recovery of known z from h.

Continuous targets  -> ridge linear R^2 (per-dim and mean), CCA top-k
Discrete targets    -> linear-probe top-1 accuracy, adjusted mutual
                       information (AMI), cluster purity
Collapse detection  -> effective rank of h covariance, % variance explained,
                       per-dim variance distribution
Control             -> nonlinear (2-layer MLP) probe R^2: distinguishes
                       "linearly identifiable" from "nonlinearly recoverable"
                       (the theorem is about LINEAR identifiability)

All readouts are model-agnostic: they consume embeddings h plus ground truth z.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neural_network import MLPRegressor

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Continuous readouts
# --------------------------------------------------------------------------- #
def ridge_r2(h: np.ndarray, z: np.ndarray, alpha: float = 1.0,
             train_frac: float = 0.5, seed: int = 0) -> dict:
    """Linear recovery R^2 of z from h, per-dim and mean, on held-out rows."""
    rng = np.random.default_rng(seed)
    n = len(h)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    tr, te = idx[:cut], idx[cut:]
    reg = Ridge(alpha=alpha).fit(h[tr], z[tr])
    pred = np.atleast_2d(reg.predict(h[te]))
    if pred.shape[0] == 1 and pred.shape[1] == z.shape[1] and z.shape[0] != 1:
        pred = pred.T  # sklearn returns (n,) for single-column y -> fix orientation
    elif pred.ndim == 2 and pred.shape[1] != z.shape[1]:
        pred = pred.T
    per_dim = np.array([r2_score(z[te][:, j], pred[:, j]) for j in range(z.shape[1])])
    # pooled R^2 (Flatten both) — robust when z has 1 col
    pooled = r2_score(z[te].ravel(), pred.ravel())
    return {"per_dim": per_dim, "mean": float(per_dim.mean()), "pooled": float(pooled)}


def cca_recovery(h: np.ndarray, z: np.ndarray, n_components: int | None = None) -> dict:
    """Top canonical correlations between h and z (geometry-agnostic cross-check)."""
    n_comp = min(h.shape[1], z.shape[1]) if n_components is None else n_components
    cca = CCA(n_components=n_comp)
    cca.fit(h, z)
    h_c, z_c = cca.transform(h, z)
    corrs = []
    for j in range(n_comp):
        c = np.corrcoef(h_c[:, j], z_c[:, j])[0, 1]
        corrs.append(c if c == c else 0.0)
    corrs = np.array(corrs)
    return {"canonical_corrs": corrs, "mean": float(np.abs(corrs).mean())}


# --------------------------------------------------------------------------- #
# Discrete readouts
# --------------------------------------------------------------------------- #
def linear_probe_acc(h: np.ndarray, zc: np.ndarray, train_frac: float = 0.5,
                     seed: int = 0) -> float:
    """Linear-probe top-1 accuracy recovering categorical z from h."""
    rng = np.random.default_rng(seed)
    n = len(h)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression(max_iter=2000).fit(h[tr], zc[tr])
    return float(clf.score(h[te], zc[te]))


def ami(zc_pred_h: np.ndarray | None, h: np.ndarray, zc: np.ndarray,
        n_clusters: int | None = None, seed: int = 0) -> float:
    """Adjusted mutual information between (clusters of h) and true zc.

    Geometry-agnostic cross-check: if AMI is high, discrete structure is
    recoverable from h even if a linear probe struggles.
    """
    from sklearn.cluster import KMeans
    k = n_clusters if n_clusters is not None else len(np.unique(zc))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(h)
    return float(normalized_mutual_info_score(zc, km.labels_))


def cluster_purity(h: np.ndarray, zc: np.ndarray, n_clusters: int | None = None,
                   seed: int = 0) -> float:
    """Purity of KMeans-on-h clusters against true zc (0..1)."""
    from sklearn.cluster import KMeans
    k = n_clusters if n_clusters is not None else len(np.unique(zc))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(h)
    labels = km.labels_
    # purity = sum over clusters of max class count / n
    total = 0.0
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(zc[mask], minlength=len(np.unique(zc)))
        total += counts.max()
    return float(total / len(h))


def adjusted_rand(h: np.ndarray, zc: np.ndarray, n_clusters: int | None = None,
                  seed: int = 0) -> float:
    """Adjusted Rand index between KMeans-on-h clusters and true zc."""
    from sklearn.cluster import KMeans
    k = n_clusters if n_clusters is not None else len(np.unique(zc))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(h)
    return float(adjusted_rand_score(zc, km.labels_))


# --------------------------------------------------------------------------- #
# Collapse detection
# --------------------------------------------------------------------------- #
def collapse_metrics(h: np.ndarray) -> dict:
    """Detect latent collapse: low effective rank / concentrated variance."""
    hc = h - h.mean(axis=0)
    cov = np.cov(hc, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum() + EPS
    frac = evals / total
    # effective rank = exp(entropy of fraction distribution)
    eff_rank = float(np.exp(-(frac * np.log(frac + EPS)).sum()))
    # how many dims explain 95% of variance
    cum = np.cumsum(np.sort(frac)[::-1])
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    return {"effective_rank": eff_rank,
            "dims_for_95pct": n_95,
            "max_frac": float(frac.max()),
            "n_dims": int(len(frac))}


# --------------------------------------------------------------------------- #
# Nonlinear control (distinguishes linear-identifiable from recoverable)
# --------------------------------------------------------------------------- #
def mlp_probe_r2(h: np.ndarray, z: np.ndarray, train_frac: float = 0.5,
                 seed: int = 0) -> float:
    """Nonlinear (2-layer MLP) recovery R^2 of z from h — the control probe."""
    rng = np.random.default_rng(seed)
    n = len(h)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    tr, te = idx[:cut], idx[cut:]
    reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000, random_state=seed)
    reg.fit(h[tr], z[tr])
    return float(r2_score(z[te].ravel(), reg.predict(h[te]).ravel()))


# --------------------------------------------------------------------------- #
# Unified per-world readout dispatcher
# --------------------------------------------------------------------------- #
def evaluate_identifiability(h: np.ndarray, z: np.ndarray, zc: np.ndarray | None = None,
                             seed: int = 0) -> dict:
    """Run the full readout battery on embeddings h with ground truth z.

    z  = continuous latents (may be all-zero for pure-discrete worlds).
    zc = categorical ground truth (optional; if None, R^2-based only).
    """
    out = {"n": len(h)}
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # continuous part (R^2 + CCA) — only meaningful if z has continuous signal
    out["ridge"] = ridge_r2(h, z, seed=seed)
    out["cca"] = cca_recovery(h, z)
    out["mlp_control"] = mlp_probe_r2(h, z, seed=seed)
    out["collapse"] = collapse_metrics(h)

    if zc is not None:
        zc = np.asarray(zc)
        if zc.ndim > 1:  # take first column as the categorical target
            zc = zc[:, 0]
        out["linear_probe_acc"] = linear_probe_acc(h, zc, seed=seed)
        out["ami"] = ami(None, h, zc, seed=seed)
        out["purity"] = cluster_purity(h, zc, seed=seed)
        out["adj_rand"] = adjusted_rand(h, zc, seed=seed)
    return out