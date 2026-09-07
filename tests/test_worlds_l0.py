"""Tests for L0 world generator + generalized-normal sampler."""

import numpy as np
import pytest

from jepa_id.worlds import L0World, gennorm_sample, alpha_grid


class TestGenNormDistribution:
    """Distribution sanity: alpha=2 Gaussian, alpha=1 Laplace, alpha->inf uniform."""

    def test_alpha2_is_unit_gaussian(self):
        rng = np.random.default_rng(0)
        x = gennorm_sample(2.0, 200_000, rng)
        assert abs(x.mean()) < 0.02
        assert abs(x.std() - 1.0) < 0.02

    def test_alpha1_is_laplace_unit_var(self):
        # Laplace with unit variance
        rng = np.random.default_rng(1)
        x = gennorm_sample(1.0, 200_000, rng)
        assert abs(x.std() - 1.0) < 0.03

    def test_large_alpha_is_uniform(self):
        rng = np.random.default_rng(2)
        x = gennorm_sample(100.0, 200_000, rng)
        assert np.all(np.abs(x) <= np.sqrt(3.0) + 1e-9)
        assert abs(x.mean()) < 0.02
        assert abs(x.std() - 1.0) < 0.03  # uniform(-sqrt3,sqrt3) has unit var

    def test_shape_changes_kurtosis(self):
        # heavy-tailed (alpha small) should have heavier tails than Gaussian
        rng = np.random.default_rng(3)
        heavy = gennorm_sample(0.5, 200_000, rng)
        rng2 = np.random.default_rng(4)
        gauss = gennorm_sample(2.0, 200_000, rng2)
        assert np.percentile(np.abs(heavy), 99.9) > np.percentile(np.abs(gauss), 99.9)


class TestL0World:
    def test_returns_known_ground_truth(self):
        w = L0World(latent_dim=4, seed=0)
        d = w.generate(1000)
        assert d["z"].shape == (1000, 4)
        assert d["z_prime"].shape == (1000, 4)
        assert d["x"].shape == (1000, 4)
        assert d["x_prime"].shape == (1000, 4)

    def test_ou_transition_correlation(self):
        # For Gaussian alpha=2, corr(z, z') should be approx rho
        w = L0World(latent_dim=1, rho=0.7, alpha=2.0, seed=5)
        d = w.generate(50_000)
        z = d["z"][:, 0]
        zp = d["z_prime"][:, 0]
        corr = np.corrcoef(z, zp)[0, 1]
        assert abs(corr - 0.7) < 0.03

    def test_gaussian_marginal_is_stationary(self):
        # For alpha=2, both z and z' ~ N(0,1) (OU preserves Gaussian)
        w = L0World(latent_dim=1, rho=0.9, alpha=2.0, seed=6)
        d = w.generate(100_000)
        assert abs(d["z_prime"][:, 0].std() - 1.0) < 0.03

    def test_nonlinear_mixing_shape(self):
        w = L0World(latent_dim=4, g="nonlinear", seed=7)
        d = w.generate(500)
        assert d["x"].shape == (500, 4)
        assert np.all(np.isfinite(d["x"]))

    def test_deterministic_given_seed(self):
        a = L0World(latent_dim=4, seed=42).generate(100)
        b = L0World(latent_dim=4, seed=42).generate(100)
        assert np.array_equal(a["z"], b["z"])


class TestAlphaGrid:
    def test_grid_covers_gaussian(self):
        g = alpha_grid()
        assert 2.0 in g  # alpha=2 is the Gaussian point
        assert g.min() < 1.0 and g.max() > 2.0
