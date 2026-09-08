"""Tests for the confirmatory analysis module (P6): Holm, Cohen's d, bootstrap CI."""

import numpy as np
import pytest

from jepa_id.analyze import holm_correct, cohens_d, bootstrap_ci, paired_delta_ci


class TestHolm:
    def test_holm_rejects_ordered(self):
        # classic example: p = [0.01, 0.04, 0.05], alpha=0.05
        # Holm: sort asc, threshold alpha/(m-k+1). 0.01 <= 0.05/3 -> reject
        # 0.04 <= 0.025? no -> stop. So only the first rejects.
        pvals = np.array([0.01, 0.04, 0.05])
        rej = holm_correct(pvals, alpha=0.05)
        assert rej.tolist() == [True, False, False]

    def test_holm_identity_when_single(self):
        assert holm_correct(np.array([0.03]))[0]

    def test_holm_all_strong(self):
        pvals = np.array([0.001, 0.002, 0.003])
        assert holm_correct(pvals).all()

    def test_holm_none_weak(self):
        pvals = np.array([0.2, 0.3, 0.4])
        assert not holm_correct(pvals).any()

    def test_holm_exact_boundary(self):
        # p1 = 0.05/4 exactly -> reject; second p must beat 0.05/3
        pvals = np.array([0.0125, 0.02, 0.05, 0.2])
        rej = holm_correct(pvals, alpha=0.05)
        assert rej[0] and not rej[1]


class TestCohensD:
    def test_equal_means_zero(self):
        # same underlying samples -> exactly 0
        a = np.random.default_rng(0).normal(0, 1, 100)
        b = a.copy()
        assert abs(cohens_d(a, b)) < 1e-12
        # independent equal-mean draws: |d| within ~2/sqrt(n) sampling error
        b2 = np.random.default_rng(1).normal(0, 1, 100)
        assert abs(cohens_d(a, b2)) < 0.3

    def test_separated_means_large(self):
        a = np.random.default_rng(0).normal(0, 1, 200)
        b = np.random.default_rng(1).normal(5, 1, 200)
        # b has the higher mean -> d(a,b) strongly negative
        assert cohens_d(a, b) < -3.0
        assert cohens_d(b, a) > 3.0

    def test_sign_direction(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert cohens_d(a, b) < 0  # b bigger -> negative
        assert cohens_d(b, a) > 0


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(1.0, 0.5, 50)
        lo, hi = bootstrap_ci(x, n_boot=500, seed=0)
        assert lo <= x.mean() <= hi

    def test_ci_narrow_with_tight_data(self):
        x = np.array([1.0, 1.01, 0.99, 1.0, 1.005])
        lo, hi = bootstrap_ci(x, n_boot=1000, seed=1)
        assert (hi - lo) < 0.05

    def test_ci_excludes_zero_when_effect_clear(self):
        x = np.random.default_rng(2).normal(2.0, 0.1, 30)
        lo, hi = bootstrap_ci(x, n_boot=1000, seed=3)
        assert lo > 0

    def test_paired_delta_ci(self):
        a = np.random.default_rng(4).normal(0.8, 0.05, 5)
        b = np.random.default_rng(5).normal(0.6, 0.05, 5)
        lo, hi = paired_delta_ci(a, b, n_boot=1000, seed=6)
        assert lo > 0 and hi > 0  # a > b consistently
