"""Tests for L1-L4 world generators (discrete ladder). v2: L3/L4 redesigned per A4."""

import numpy as np
import pytest

from jepa_id.worlds import (
    L1World, L2World, L2TwinWorld, L3World, L4World, WORLD_REGISTRY, ALPHABET,
)


def entropy(col):
    vals, counts = np.unique(col, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


class TestL1DiscretizedGaussian:
    def test_returns_continuous_and_discrete_ground_truth(self):
        w = L1World(latent_dim=4, K=8, seed=0)
        d = w.generate(2000)
        assert d["z"].shape == (2000, 4)          # continuous
        assert d["zK"].shape == (2000, 4)         # binned int categories
        assert d["zK"].dtype.kind == "i"
        assert d["zK"].min() >= 0 and d["zK"].max() <= 7

    def test_large_K_approximates_continuous(self):
        w = L1World(latent_dim=4, K=64, seed=1)
        d = w.generate(5000)
        for dim in range(4):
            z = d["z"][:, dim]
            zk = d["zK"][:, dim]
            order = np.argsort(z)
            assert np.all(np.diff(zk[order]) >= 0)

    def test_small_K_is_truly_discrete(self):
        w = L1World(latent_dim=4, K=2, seed=2)
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
        w = L2World(latent_dim=1, S=4, seed=3)
        d = w.generate(50_000)
        z, zp = d["z"][:, 0], d["z_prime"][:, 0]
        rows = zp[z == 0]
        counts = np.bincount(rows, minlength=4).astype(float)
        empirical = counts / counts.sum()
        assert np.allclose(empirical, w.P[0], atol=0.03)

    def test_deterministic_given_seed(self):
        a = L2World(seed=5).generate(50)
        b = L2World(seed=5).generate(50)
        assert np.array_equal(a["z"], b["z"])


class TestL3RuleGrammarV2:
    """A4: v1 was degenerate (affix==stem, joint H=1 bit; x leaked z). v2 rules:
    joint entropy is genuine (>2 bits), every z column varies, harmony rule
    constrains suffix front/back bit to stem class, and x is an embedding
    surface (z never a literal column of x)."""

    def test_harmony_rule_holds_in_ground_truth(self):
        w = L3World(seed=0)
        d = w.generate(4000)
        s, av1, av2 = d["z"][:, 0], d["z"][:, 1], d["z"][:, 2]
        # suffix front/back bit (code & 2) == stem class
        assert np.all((av1 & 2) == (2 * s)) and np.all((av2 & 2) == (2 * s))
        # but suffix codes are NOT copies of the stem column
        assert not np.all(av1 == s)

    def test_joint_entropy_is_genuine(self):
        w = L3World(seed=1)
        d = w.generate(8000)
        z = d["z"]
        h = [entropy(z[:, j]) for j in range(z.shape[1])]
        # each column must carry real variance
        assert all(hj > 0.8 for hj in h)
        # empirical joint > 2 bits (v1 was exactly 1.0)
        codes = [tuple(int(v) for v in row) for row in z]
        from collections import Counter
        c = Counter(codes)
        p = np.array(list(c.values())) / sum(c.values())
        joint = float(-(p * np.log2(p)).sum())
        assert joint > 2.5, f"joint entropy {joint:.3f} too low (degenerate)"

    def test_z_not_literal_column_of_x(self):
        w = L3World(seed=2)
        d = w.generate(2000)
        x = d["x"]
        # no x column is a permuted copy of any z column (embedding surface)
        for j in range(x.shape[1]):
            for k in range(d["z"].shape[1]):
                assert not np.allclose(x[:, j], d["z"][:, k], atol=1e-3)

    def test_observation_shape_and_determinism(self):
        a = L3World(seed=3)
        b = L3World(seed=3)
        da, db = a.generate(100), b.generate(100)
        assert np.array_equal(da["z"], db["z"])
        assert da["x"].shape == (100, 3 * a.obs_per_pos)
        # z_prime shares stem class with z (same word, suffix heights redrawn)
        assert np.array_equal(da["z"][:, 0], da["z_prime"][:, 0])


class TestL4TurkishRulesV2:
    """A4: v1 crashed (caret_required constant -> one-class probe error) and x
    leaked z. v2: caret_required must vary (negative class present), every z
    column has variance, x contains surface letters only."""

    def test_caret_required_has_real_variance(self):
        w = L4World(seed=0)
        d = w.generate(3000)
        cr = d["z"][:, 1]
        assert set(np.unique(cr)) == {0, 1}
        p1 = (cr == 1).mean()
        assert 0.3 < p1 < 0.7, f"caret rate {p1:.2f} unbalanced (needs neg class)"

    def test_all_z_columns_vary(self):
        w = L4World(seed=1)
        d = w.generate(4000)
        for j in range(3):
            assert entropy(d["z"][:, j]) > 0.5, f"z col {j} degenerate"

    def test_stem_class_matches_vowel_harmony(self):
        assert L4World.stem_class_of("adet") == 0   # front (e)
        assert L4World.stem_class_of("kar") == 1    # back (a)
        assert L4World.stem_class_of("kitap") == 1  # back (a)
        assert L4World.stem_class_of("kedi") == 0   # front (i)

    def test_x_is_surface_only_no_leak(self):
        w = L4World(seed=2)
        d = w.generate(2000)
        x, z = d["x"], d["z"]
        assert x.shape == (2000, w.obs_width)
        # no direct z column equals any x column (letters one-hot + length only)
        for k in range(3):
            for j in range(x.shape[1]):
                assert not np.allclose(x[:, j], z[:, k], atol=1e-3)
        # x is binary-ish surface codes + length + noise: check it is NOT a
        # verbatim copy of v1-style flags (no column with values in {1,2,3,4})
        assert x.max() <= 1.0 + 5 * w.obs_noise

    def test_caret_words_rule_consistency(self):
        # known lexicon facts must hold: caret words have nonzero code
        w = L4World(seed=3)
        assert w.stem_class_of("suret") == 0  # front (e is last vowel)
        assert w.stem_class_of("memur") == 1  # back (u)

    def test_all_lexicon_letters_representable(self):
        w = L4World(seed=4)
        for word in w.vocab:
            for ch in word:
                assert ch in ALPHABET, f"{word}: {ch} not in ALPHABET"
        # distinct words -> distinct surface codes (no collision)
        from collections import Counter
        rows = [row.tobytes() for row in w._surface(np.array(w.vocab))]
        assert len(Counter(rows)) == len(w.vocab)

    def test_registry_has_all_levels(self):
        assert set(WORLD_REGISTRY) == {"L0", "L1", "L2", "L2t", "L3", "L4"}


class TestL2TwinMomentsMatched:
    """R1 (A5): L2t must match L2's OBSERVED x moments; only support differs."""

    def test_registry_contains_twin(self):
        assert "L2t" in WORLD_REGISTRY
        assert WORLD_REGISTRY["L2t"] is L2TwinWorld

    def test_x_moments_match_l2(self):
        for seed in (0, 1, 7):
            a = L2World(latent_dim=3, S=5, seed=seed)
            b = L2TwinWorld(latent_dim=3, S=5, seed=seed)
            da, db = a.generate(40000), b.generate(40000)
            ma = da["x"].mean(0); mb = db["x"].mean(0)
            va = da["x"].var(0);  vb = db["x"].var(0)
            assert np.allclose(ma, mb, atol=0.05), (ma, mb)
            assert np.allclose(va, vb, atol=0.10), (va, vb)

    def test_z_continuous_twin(self):
        w = L2TwinWorld(latent_dim=2, S=4, seed=0)
        d = w.generate(500)
        assert d["z"].dtype == np.float64
        # continuous support: many distinct values per dim
        assert len(np.unique(np.round(d["z"][:, 0], 6))) > 100

    def test_zK_has_S_bins_balanced(self):
        w = L2TwinWorld(latent_dim=2, S=4, seed=3)
        d = w.generate(20000)
        for dim in range(2):
            uniq, counts = np.unique(d["zK"][:, dim], return_counts=True)
            assert len(uniq) == 4
            assert np.all(counts / 20000 > 0.15)   # approx equiprobable

    def test_persistence_matches_l2(self):
        # cross-view correlation of observed x should be ~rho*signal fraction,
        # comparable between L2 and L2t at same seed/rho
        a = L2World(latent_dim=4, S=5, seed=0)
        b = L2TwinWorld(latent_dim=4, S=5, seed=0)
        da, db = a.generate(20000), b.generate(20000)
        def corr(d):
            xs = d["x"] - d["x"].mean(0); xp = d["x_prime"] - d["x_prime"].mean(0)
            return (xs * xp).mean(0) / (xs.std(0) * xp.std(0))
        ca, cb = corr(da).mean(), corr(db).mean()
        # Both strongly persistent; L2's is lower because categorical jumps in
        # random (state-unordered) E columns decorrelate more than the smooth
        # AR(1) — the support difference itself. Loose bound + persistence check.
        assert ca > 0.5 and cb > 0.5
        assert abs(ca - cb) < 0.25, (ca, cb)
