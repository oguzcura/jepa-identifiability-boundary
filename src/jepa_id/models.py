"""Model zoo for the identifiability ladder.

Three objectives, all on the SAME world data (x, x_prime pairs with known z):

  1. JepaCore    — JEPA-style latent prediction. Context encoder E_theta maps
                   x -> h; target encoder E_psi (EMA of E_theta, stop-grad)
                   maps x_prime -> h_prime; predictor P_phi predicts h_prime
                   from h. Loss = || P_phi(h) - h_prime ||^2 (target detached).
  2. Recon       — input-space prediction (MAE-style). Encoder E maps x -> h,
                   decoder D maps h -> x_prime. Loss = || D(h) - x_prime ||^2.
  3. Contrastive — InfoNCE between E(x) and E(x_prime) within a batch
                   (positive pairs = same index, negatives = other batch rows).

All use a shared MLP backbone so differences are attributable to the objective,
not the architecture. Everything deterministic-seeded and GPU-aware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared encoder backbone
# --------------------------------------------------------------------------- #
class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, emb_dim: int = 64, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor(nn.Module):
    def __init__(self, emb_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class MLPDecoder(nn.Module):
    def __init__(self, emb_dim: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class JepaCore(nn.Module):
    """LeJEPA-style latent prediction: alignment + SIGReg, EMA target + stop-grad.

    SIGReg (Sketched Isotropic Gaussian Regularization) forces the context
    embedding distribution toward an isotropic Gaussian — the condition that
    makes linear identifiability hold, and whose violation is what breaks it
    for non-Gaussian worlds (the theorem's mechanism).
    """
    obs_dim: int
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    tau: float = 0.99          # EMA decay for target encoder
    sigreg_lambda: float = 1.0  # SIGReg penalty weight
    device: str = "cpu"

    def __post_init__(self):
        super().__init__()
        self.context = MLPEncoder(self.obs_dim, self.emb_dim, self.hidden, self.depth)
        self.target = MLPEncoder(self.obs_dim, self.emb_dim, self.hidden, self.depth)
        self.target.load_state_dict(self.context.state_dict())  # init same
        self.predictor = MLPPredictor(self.emb_dim, self.hidden)
        for p in self.target.parameters():
            p.requires_grad_(False)      # target is EMA, not gradient-trained
        self.to(self.device)

    def _sigreg(self, h: torch.Tensor) -> torch.Tensor:
            """Sketched Gaussian-moment regularization (LeJEPA-style).

            Projects h onto k fixed random directions drawn at init, then penalizes
            deviation of each 1-D marginal from N(0,1): mean->0, var->1, skew->0,
            kurt->3. The random sketch MIXES all dimensions, so the model cannot
            dodge the penalty by hiding non-Gaussian signal in a few dims and
            filling the rest with Gaussian noise (which would dilute per-dim
            moment penalties to ~0). This is the mechanism that forces h(z) to be
            a genuinely nonlinear Gaussian-izing map for non-Gaussian latents,
            breaking linear identifiability exactly as the theorem predicts.
            """
            if not hasattr(self, "_sketch"):
                            g = torch.Generator(device=h.device).manual_seed(1234)
                            self._sketch = torch.randn(h.size(1), 32, generator=g, device=h.device)
                            self._sketch = self._sketch / self._sketch.norm(dim=0, keepdim=True)
            proj = h @ self._sketch                      # (B, k) mixtures of all dims
            eps = 1e-6
            mu = proj.mean(dim=0)
            pc = proj - mu
            var = pc.pow(2).mean(dim=0) + eps
            std = var.sqrt()
            pn = pc / std
            skew = pn.pow(3).mean(dim=0)
            kurt = pn.pow(4).mean(dim=0)
            cross = (pc.T @ pc) / pc.size(0) - torch.eye(32, device=h.device)
            pen = (
                (var - 1.0).pow(2).mean()
                + skew.pow(2).mean()
                + (kurt - 3.0).pow(2).mean()
                + cross.pow(2).mean()
                + mu.pow(2).mean()
            )
            return pen

    @torch.no_grad()
    def _ema_update(self):
        with torch.no_grad():
            for pc, pt in zip(self.context.parameters(), self.target.parameters()):
                pt.mul_(self.tau).add_(pc, alpha=1.0 - self.tau)
            for bc, bt in zip(self.context.buffers(), self.target.buffers()):
                bt.copy_(bc)

    def forward(self, x, x_prime, momentum: bool = True):
        """Return (loss, h, h_pred, h_target). loss = alignment + lambda*SIGReg."""
        h = self.context(x)
        with torch.no_grad():                       # stop-grad on target
            h_target = self.target(x_prime)
        h_pred = self.predictor(h)
        align = F.mse_loss(h_pred, h_target)
        reg = self._sigreg(h)
        loss = align + self.sigreg_lambda * reg
        if momentum:
            self._ema_update()
        return loss, h, h_pred, h_target


@dataclass(eq=False)
class Recon(nn.Module):
    """Input-space reconstruction (MAE-style)."""
    obs_dim: int
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    device: str = "cpu"

    def __post_init__(self):
        super().__init__()
        self.encoder = MLPEncoder(self.obs_dim, self.emb_dim, self.hidden, self.depth)
        self.decoder = MLPDecoder(self.emb_dim, self.obs_dim, self.hidden)
        self.to(self.device)

    def forward(self, x, x_prime, momentum: bool = True):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        loss = F.mse_loss(x_rec, x_prime)
        return loss, h, x_rec, x_prime


@dataclass(eq=False)
class Contrastive(nn.Module):
    """InfoNCE between paired views (positive = same row)."""
    obs_dim: int
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    temperature: float = 0.1
    device: str = "cpu"

    def __post_init__(self):
        super().__init__()
        self.encoder = MLPEncoder(self.obs_dim, self.emb_dim, self.hidden, self.depth)
        self.to(self.device)

    def forward(self, x, x_prime, momentum: bool = True):
        h = self.encoder(x)
        h_p = self.encoder(x_prime)
        h = F.normalize(h, dim=-1)
        h_p = F.normalize(h_p, dim=-1)
        logits = h @ h_p.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss, h, h_p, h_p


@dataclass(eq=False)
class VICReg(nn.Module):
    """Variance-invariance-covariance (Bardes et al. 2021) — the theorem paper's
    third Fig-4b objective (their trio: LeJEPA / InfoNCE / VICReg).

    Views = the same paired world observations (x, x_prime) used by every
    objective. Loss = lambda_inv * invariance (MSE between view embeddings)
    + lambda_var * variance (hinge keeping per-dim std >= 1)
    + lambda_cov * covariance (off-diagonal -> 0). No predictor, no EMA:
    VICReg prevents collapse by its variance term rather than by stop-grad.
    """
    obs_dim: int
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    lambda_inv: float = 25.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        super().__init__()
        self.encoder = MLPEncoder(self.obs_dim, self.emb_dim, self.hidden, self.depth)
        self.to(self.device)

    @staticmethod
    def _variance(z: torch.Tensor) -> torch.Tensor:
        std = z.std(dim=0)
        return F.relu(1.0 - std).mean()

    @staticmethod
    def _covariance(z: torch.Tensor) -> torch.Tensor:
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.size(0) - 1)
        off = cov - torch.diag(cov.diag())
        return off.pow(2).sum() / z.size(1)

    def forward(self, x, x_prime, momentum: bool = True):
        h = self.encoder(x)
        h_p = self.encoder(x_prime)
        loss = (
            self.lambda_inv * F.mse_loss(h, h_p)
            + self.lambda_var * (self._variance(h) + self._variance(h_p))
            + self.lambda_cov * (self._covariance(h) + self._covariance(h_p))
        )
        return loss, h, h_p, h_p


MODEL_REGISTRY = {
    "jepa": JepaCore,
    "recon": Recon,
    "contrastive": Contrastive,
    "vicreg": VICReg,
}


def build_model(name: str, obs_dim: int, **kwargs) -> nn.Module:
    cls = MODEL_REGISTRY[name]
    # Accept only kwargs the target class accepts (e.g. tau only for JepaCore)
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k for k in sig.parameters if k not in ("self", "obs_dim")}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(obs_dim=obs_dim, **filtered)
