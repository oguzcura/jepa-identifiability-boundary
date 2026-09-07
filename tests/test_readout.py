"""Tests for the identifiability readout suite (known-synthetic cases)."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from jepa_id.readout import (
    ridge_r2, cca_recovery, linear_probe_acc, ami, cluster_purity,
    adjusted_rand, collapse_metrics, mlp_probe_r2, evaluate_identifiability,
)


def make_linear(h_dim=8, n=400, seed=0):
    """h and z linearly related: z = W h (perfect linear recovery)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, h_dim))
    W = rng.normal(size=(h_dim, 3))
    z = h @ W
    return h, z


def make_discrete(h_dim=8, n=400, k=4, seed=0):
    """h contains perfect linearly-separable discrete structure: z = cluster id."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, h_dim)) * 3.0
    labels = rng.integers(0, k, size=n)
    h = centers[labels] + rng.normal(scale=0.05, size=(n, h_dim))
    return h, labels


class TestRidgeR2:
    def test_perfect_linear_recovery(self):
        h, z = make_linear()
        out = ridge_r2(h, z)
        assert out["mean"] > 0.99

    def test_random_h_gives_low_r2(self):
        rng = np.random.default_rng(1)
        h = rng.normal(size=(400, 8))
        z = rng.normal(size=(400, 3))
        out = ridge_r2(h, z)
        assert out["mean"] < 0.15


class TestCCA:
    def test_linear_recovery_high_corr(self):
        h, z = make_linear()
        out = cca_recovery(h, z)
        assert out["mean"] > 0.95

    def test_random_low_corr(self):
        rng = np.random.default_rng(2)
        h = rng.normal(size=(400, 8))
        z = rng.normal(size=(400, 3))
        out = cca_recovery(h, z)
        assert out["mean"] < 0.3


class TestDiscrete:
    def test_perfect_linear_probe(self):
        h, labels = make_discrete()
        acc = linear_probe_acc(h, labels)
        assert acc > 0.98

    def test_ami_high_for_clustered(self):
        h, labels = make_discrete()
        val = ami(None, h, labels)
        assert val > 0.9

    def test_purity_high_for_clustered(self):
        h, labels = make_discrete()
        assert cluster_purity(h, labels) > 0.98

    def test_adj_rand_high_for_clustered(self):
        h, labels = make_discrete()
        assert adjusted_rand(h, labels) > 0.9

    def test_random_h_chance_acc(self):
        rng = np.random.default_rng(3)
        h = rng.normal(size=(400, 8))
        labels = rng.integers(0, 4, size=400)
        acc = linear_probe_acc(h, labels)
        assert acc < 0.5  # 4 classes -> chance ~0.25, allow slack


class TestCollapse:
    def test_full_rank_h(self):
        rng = np.random.default_rng(4)
        h = rng.normal(size=(1000, 8))
        out = collapse_metrics(h)
        assert out["effective_rank"] > 5.0  # ~8 for iid normal, allow slack

    def test_collapsed_h(self):
        rng = np.random.default_rng(5)
        h = rng.normal(size=(1000, 1)).repeat(8, axis=1)  # rank 1
        out = collapse_metrics(h)
        assert out["effective_rank"] < 2.0
        assert out["dims_for_95pct"] <= 1


class TestMLPControl:
    def test_nonlinear_recovery_when_linear_fails(self):
        # z = f(h) with strong nonlinearity: linear probe poor, MLP good
        rng = np.random.default_rng(6)
        h = rng.normal(size=(600, 2))
        z = np.sign(h[:, 0]) * (h[:, 1] ** 2)  # nonlinear target (600,)
        z = z.reshape(-1, 1)
        lin = ridge_r2(h, z)
        mlp = mlp_probe_r2(h, z)
        assert mlp > lin["mean"] + 0.3


class TestDispatcher:
    def test_full_battery_continuous(self):
        h, z = make_linear()
        out = evaluate_identifiability(h, z)
        assert "ridge" in out and "cca" in out and "collapse" in out
        assert out["ridge"]["mean"] > 0.99

    def test_full_battery_discrete(self):
        h, labels = make_discrete()
        out = evaluate_identifiability(h, np.zeros((len(h), 1)), zc=labels)
        assert out["linear_probe_acc"] > 0.98
        assert out["ami"] > 0.9