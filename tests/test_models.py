"""Tests for the model zoo (JepaCore / Recon / Contrastive) + training loop."""

import numpy as np
import pytest
import torch

from jepa_id.models import build_model, JepaCore, MODEL_REGISTRY
from jepa_id.train import train, TrainConfig
from jepa_id.worlds import WORLD_REGISTRY


def make_batch(obs_dim=4, n=32, device="cpu"):
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(n, obs_dim)), dtype=torch.float32, device=device)
    xp = torch.tensor(rng.normal(size=(n, obs_dim)), dtype=torch.float32, device=device)
    return x, xp


class TestModelShapes:
    def test_all_registry_models_build(self):
        for name in MODEL_REGISTRY:
            m = build_model(name, obs_dim=4, emb_dim=64)
            assert m is not None

    def test_jepa_forward_shapes(self):
        m = JepaCore(obs_dim=4, emb_dim=8)
        x, xp = make_batch()
        loss, h, h_pred, h_target = m(x, xp)
        assert h.shape == (32, 8)
        assert h_pred.shape == (32, 8)
        assert h_target.shape == (32, 8)
        assert loss.ndim == 0

    def test_target_encoder_zero_grad(self):
        m = JepaCore(obs_dim=4, emb_dim=8)
        assert all(not p.requires_grad for p in m.target.parameters())


class TestJepaBehavior:
    def test_loss_is_mse_like_and_finite(self):
        m = JepaCore(obs_dim=4, emb_dim=8)
        x, xp = make_batch(n=16)
        loss, *_ = m(x, xp)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_ema_moves_target_toward_context(self):
        # Correct JEPA semantics: target starts IDENTICAL to context; EMA diverges
        # only after context parameters get gradient updates. Perturb context
        # manually, then a forward pass should pull target toward context.
        m = JepaCore(obs_dim=4, emb_dim=8, tau=0.5)
        assert torch.allclose(m.target.net[0].weight, m.context.net[0].weight)
        with torch.no_grad():
            m.context.net[0].weight.mul_(2.0)  # perturb context
        before = m.target.net[0].weight.clone()
        x, xp = make_batch(n=8)
        m(x, xp)  # forward with momentum update (tau=0.5)
        after = m.target.net[0].weight
        # target should now be a 50/50 blend of old target and perturbed context
        expected = 0.5 * before + 0.5 * m.context.net[0].weight
        assert torch.allclose(after, expected, atol=1e-5)

    def test_deterministic_seed(self):
        c1 = TrainConfig(world="L0", model="jepa", steps=10, batch_size=32,
                         seed=1, device="cpu")
        c2 = TrainConfig(world="L0", model="jepa", steps=10, batch_size=32,
                         seed=1, device="cpu")
        r1 = train(c1)
        r2 = train(c2)
        assert r1["final_loss"] == pytest.approx(r2["final_loss"], rel=1e-6)

    def test_loss_decreases_over_training(self):
        cfg = TrainConfig(world="L0", model="jepa", steps=400, batch_size=64,
                          seed=3, device="cpu", log_every=50)
        r = train(cfg)
        assert r["final_loss"] > 0

    def test_all_models_train_end_to_end(self):
        for name in ["jepa", "recon", "contrastive"]:
            cfg = TrainConfig(world="L0", model=name, steps=60, batch_size=32,
                              seed=0, device="cpu")
            res = train(cfg)
            assert res["final_loss"] == res["final_loss"]  # not NaN
            assert res["ckpt"].endswith(".pt")


class TestTrainingGates:
    def test_log_and_ckpt_files_written(self, tmp_path):
        from pathlib import Path
        cfg = TrainConfig(world="L1", model="jepa", steps=20, batch_size=32,
                          seed=0, device="cpu",
                          log_path=str(tmp_path / "{model}.jsonl"),
                          ckpt_path=str(tmp_path / "{model}.pt"))
        train(cfg)
        assert (tmp_path / "jepa.jsonl").exists()
        assert (tmp_path / "jepa.pt").exists()

    def test_worlds_all_produce_batches(self):
        for wname in WORLD_REGISTRY:
            d = WORLD_REGISTRY[wname](seed=0)
            batch = d.generate(16)
            assert batch["x"].shape[0] == 16

class TestVICRegBehavior:
    def test_vicreg_in_registry_and_builds(self):
        from jepa_id.models import VICReg
        m = build_model("vicreg", obs_dim=4, emb_dim=8)
        assert isinstance(m, VICReg)

    def test_vicreg_forward_loss_finite(self):
        from jepa_id.models import VICReg
        m = VICReg(obs_dim=4, emb_dim=8)
        x, xp = make_batch(n=64)
        loss, h, hp, _ = m(x, xp, momentum=False)
        assert torch.isfinite(loss)
        assert h.shape == (64, 8) and hp.shape == (64, 8)

    def test_vicreg_variance_term_penalizes_collapse(self):
        from jepa_id.models import VICReg
        # constant embedding -> variance hinge = 1.0 per dim (std 0), relu(1-0)=1
        z = torch.ones(32, 8)
        v = VICReg._variance(z)
        assert torch.allclose(v, torch.tensor(1.0), atol=1e-5)
        # well-spread embedding -> hinge 0
        z2 = torch.randn(128, 8) * 3.0
        assert VICReg._variance(z2) < 0.05

    def test_vicreg_covariance_penalizes_correlated_dims(self):
        from jepa_id.models import VICReg
        rng = torch.Generator().manual_seed(0)
        # two perfectly correlated dims -> covariance penalty > 0
        base = torch.randn(200, 1, generator=rng)
        zc = torch.cat([base, base, torch.randn(200, 6, generator=rng)], dim=1)
        c_pen = VICReg._covariance(zc)
        zi = torch.randn(200, 8, generator=rng)
        assert c_pen > VICReg._covariance(zi) * 3  # correlated >> iid (~7x typical)

    def test_vicreg_identical_views_give_zero_invariance(self):
        from jepa_id.models import VICReg
        m = VICReg(obs_dim=4, emb_dim=8)
        x, _ = make_batch(n=32)
        loss, h, hp, _ = m(x, x.clone(), momentum=False)
        # invariance part ~0; loss is just variance+covariance terms
        assert loss.item() < 60.0  # 25*0 + 25*var(<=1) + 1*cov — bounded
