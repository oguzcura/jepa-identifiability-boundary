"""Tests for L1-L4 world generators (discrete ladder)."""

import numpy as np
import pytest

from jepa_id.worlds import (
    L1World, L2World, L3World, L4World, WORLD_REGISTRY,
)


class TestL1DiscretizedGaussian:
    def test_returns_continuous_and_discrete_ground_truth(self):
        w = L1World(latent_dim=4, K=8, seed=0)
        d = w.generate(2000)
        assert d["z"].shape == (2000, 4)          # continuous
        assert d["zK"].shape == (2000, 4)         # binned int categories
        assert d["zK"].dtype.kind == "i"
        assert d["zK"].min() >= 0 and d["zK"].max() <= 7

    def test_large_K_approximates_continuous(self):
        # As K grows, binned z should be monotone-preserving of continuous z
        w = L1World(latent_dim=4, K=64, seed=1)
        d = w.generate(5000)
        # per-dim Spearman-ish: sorting consistency between z and zK
        for dim in range(4):
            z = d["z"][:, dim]
            zk = d["zK"][:, dim]
            # categories must be non-decreasing function of continuous value
            order = np.argsort(z)
            assert np.all(np.diff(zk[order]) >= 0)

    def test_small_K_is_truly_discrete(self):
        w = L1World(latent_dim=4, K=2, seed=2)   # binary categories
        d = w.generate(1000)
        assert set(np.unique(d["zK"])) <= {0, 1}

    def test_deterministic_given_seed(self):
        a = L1World(K=8, seed=9).generate(100)
        b = L1World(K=8, seed=9).generate(100)
        assert np.array_equal(a["z"], b["z"]) and np.array_equal(a["zK"], b["zK"])


class TestL2MarkovChain:
    def test_categorical_ground_truth(self):
        w = L2World(latent_dim=4, S=5, seed=0)
        d = w.generate(2000)
        assert d["z"].shape == (2000, 4)
        assert d["z"].dtype.kind == "i"
        assert set(np.unique(d["z"])) <= set(range(5))

    def test_transition_follows_matrix(self):
        # z_prime given z should follow the row of P
        w = L2World(latent_dim=1, S=4, seed=3)
        d = w.generate(50_000)
        z, zp = d["z"][:, 0], d["z_prime"][:, 0]
        # empirical transition from state 0
        rows = zp[z == 0]
        counts = np.bincount(rows, minlength=4).astype(float)
        empirical = counts / counts.sum()
        assert np.allclose(empirical, w.P[0], atol=0.03)

    def test_deterministic_given_seed(self):
        a = L2World(seed=5).generate(50)
        b = L2World(seed=5).generate(50)
        assert np.array_equal(a["z"], b["z"])


class TestL3RuleGrammar:
    def test_harmony_rule_holds_in_ground_truth(self):
        w = L3World(seed=0)
        d = w.generate(2000)
        # z = [stem_class, affix]; rule: affix == stem class
        assert np.array_equal(d["z"][:, 0], d["z"][:, 1])

    def test_observation_encodes_composition(self):
        w = L3World(seed=1)
        d = w.generate(1000)
        assert d["x"].shape == (1000, 6)
        # harmony consistency feature x[:,4] = (stem+affix)%2 = 0 + noise(sd~0.05)
        assert np.all(np.abs(d["x"][:, 4]) < 0.3)

    def test_rule_governed_not_iid(self):
        # affix is fully determined by stem -> perfect correlation in z
        w = L3World(seed=2)
        d = w.generate(5000)
        corr = np.corrcoef(d["z"][:, 0], d["z"][:, 1])[0, 1]
        assert corr > 0.999


class TestL4TurkishRules:
    def test_caret_required_always_true(self):
        w = L4World(seed=0)
        d = w.generate(200)
        assert np.all(d["z"][:, 1] == 1)   # all words need caret

    def test_stem_class_matches_vowel_harmony(self):
        w = L4World(seed=1)
        d = w.generate(200)
        for i, wd in enumerate(w.vocab):
            pass
        # verify stem class correctness for a few known words
        assert L4World.stem_class_of("adet") == 0   # front (e)
        assert L4World.stem_class_of("kar") == 1    # back (a)

    def test_observation_featurization(self):
        w = L4World(seed=2)
        d = w.generate(100)
        assert d["x"].shape == (100, 8)
        assert d["z"].shape == (100, 3)

    def test_registry_has_all_levels(self):
        assert set(WORLD_REGISTRY) == {"L0", "L1", "L2", "L3", "L4"}
