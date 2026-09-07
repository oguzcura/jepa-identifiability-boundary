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
    """JEPA-style latent prediction with EMA target + stop-grad."""
    obs_dim: int
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    tau: float = 0.99          # EMA decay for target encoder
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

    @torch.no_grad()
    def _ema_update(self):
        with torch.no_grad():
            for pc, pt in zip(self.context.parameters(), self.target.parameters()):
                pt.mul_(self.tau).add_(pc, alpha=1.0 - self.tau)
            for bc, bt in zip(self.context.buffers(), self.target.buffers()):
                bt.copy_(bc)

    def forward(self, x, x_prime, momentum: bool = True):
        """Return (loss, h, h_pred, h_target). h = context embedding (used for
        identifiability readout); loss is the JEPA latent-prediction objective."""
        h = self.context(x)
        with torch.no_grad():                       # stop-grad on target
            h_target = self.target(x_prime)
        h_pred = self.predictor(h)
        loss = F.mse_loss(h_pred, h_target)
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


MODEL_REGISTRY = {
    "jepa": JepaCore,
    "recon": Recon,
    "contrastive": Contrastive,
}


def build_model(name: str, obs_dim: int, **kwargs) -> nn.Module:
    cls = MODEL_REGISTRY[name]
    # Accept only kwargs the target class accepts (e.g. tau only for JepaCore)
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k for k in sig.parameters if k not in ("self", "obs_dim")}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(obs_dim=obs_dim, **filtered)
