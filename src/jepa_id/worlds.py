"""Procedural world generators with KNOWN ground-truth latents.

Implements the "world ladder" from the pre-registration (L0-L4). Every world
samples a true latent z, applies a stationary transition to a second view z',
and renders observations x = g(z) + eps. Ground truth z is always returned so
identifiability can be measured mechanically.

L0 = Gaussian OU world + generalized-normal sweep (reproduces the theorem
paper's Sec 6.2 / Fig 4b regime: alpha=2 Gaussian, alpha=1 Laplace,
alpha->inf uniform, alpha->0 heavy-tailed).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Generalized normal (exponential power) sampling
# --------------------------------------------------------------------------- #
def gennorm_sample(alpha: float, size, rng: np.random.Generator) -> np.ndarray:
    """Sample from the generalized normal (exponential power) distribution.

    p(x) ∝ exp(-|x|^alpha), normalized to UNIT VARIANCE (so the sweep holds
    variance fixed and varies only shape):
      alpha = 2 -> N(0,1)
      alpha = 1 -> Laplace(0, b=1/sqrt(2))  [unit variance]
      alpha -> inf -> uniform(-sqrt(3), sqrt(3))  [unit variance]
      alpha -> 0  -> heavy-tailed

    Raw draw: X = sign(U) * V^(1/alpha), V ~ Gamma(shape=1/alpha, scale=1).
    E[X^2] = Gamma(3/alpha)/Gamma(1/alpha); we divide by sqrt(E[X^2]) to get
    unit variance. For alpha -> inf we special-case to uniform(-sqrt(3),sqrt(3)).
    """
    from scipy.special import gamma as gammafn

    if alpha >= 40.0:
        s = np.sqrt(3.0)  # uniform(-1,1) has var 1/3 -> scale to var 1
        return rng.uniform(-1.0, 1.0, size=size) * s
    sign = rng.choice([-1.0, 1.0], size=size)
    v = _gamma_loop(1.0 / alpha, size, rng)
    raw = sign * np.power(v, 1.0 / alpha)
    # unit-variance normalization
    ex2 = gammafn(3.0 / alpha) / gammafn(1.0 / alpha)
    return raw / np.sqrt(ex2)


def _gamma_loop(shape: float, size, rng: np.random.Generator) -> np.ndarray:
    """Gamma(shape, scale=1) via Marsaglia-Tsang rejection.

    `size` may be an int or a shape tuple; output matches `size`.
    """
    size = np.asarray(size, dtype=int) if isinstance(size, tuple) else int(size)
    out = np.empty(int(np.prod(size)), dtype=float)
    if shape >= 1.0:
        d = shape - 1.0 / 3.0
        c = 1.0 / np.sqrt(9.0 * d)
        for i in range(out.size):
            while True:
                x = rng.normal()
                v = (1.0 + c * x) ** 3
                if v <= 0.0:
                    continue
                u = rng.random()
                if u < 1.0 - 0.0331 * x ** 4:
                    break
                if np.log(u) < 0.5 * x * x + d * (1.0 - v + np.log(v)):
                    break
            out[i] = d * v
    else:
        # shape < 1: sample Gamma(shape+1) then multiply by U^(1/shape)
        base = _gamma_loop(shape + 1.0, out.size, rng)
        out = base * np.power(rng.random(out.size), 1.0 / shape)
    return out.reshape(size)


# --------------------------------------------------------------------------- #
# L0: Gaussian OU world (with gennorm sweep)
# --------------------------------------------------------------------------- #
@dataclass
class L0World:
    """Gaussian world with Ornstein-Uhlenbeck transition (Eq 1 of theorem).

    z ~ gennorm(alpha) ; z' = rho z + sqrt(1-rho^2) eta, eta ~ gennorm(alpha).
    For alpha=2 this is exactly the theorem's Gaussian world. Sweeping alpha
    reproduces their Sec 6.2 "latent-distribution sweep".

    Observation: x = g(z) + eps.  g linear (A z) or nonlinear (spiral mix).
    """
    latent_dim: int = 4
    rho: float = 0.9
    alpha: float = 2.0
    g: str = "linear"          # "linear" | "nonlinear"
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # fixed random mixing matrix A (seeded once, deterministic across calls)
        self.A = np.linalg.qr(self.rng.normal(size=(self.latent_dim, self.latent_dim)))[0]

    def _sample_z(self, n: int) -> np.ndarray:
        return gennorm_sample(self.alpha, (n, self.latent_dim), self.rng)

    def _transition(self, z: np.ndarray) -> np.ndarray:
        eta = gennorm_sample(self.alpha, z.shape, self.rng)
        return self.rho * z + np.sqrt(1.0 - self.rho ** 2) * eta

    def _mix(self, z: np.ndarray) -> np.ndarray:
        if self.g == "linear":
            x = z @ self.A.T
        else:  # nonlinear: spiral-like per-dimension nonlinearity
            r = np.linalg.norm(z, axis=1, keepdims=True)
            th = np.arctan2(z[:, 1], z[:, 0])[:, None]
            x = np.hstack([r * np.cos(2 * th), r * np.sin(2 * th), z[:, 2:4]])
        return x

    def generate(self, n_pairs: int) -> dict:
        """Return a positive pair (z, z') with observations (x, x')."""
        z = self._sample_z(n_pairs)
        z_prime = self._transition(z)
        x = self._mix(z) + self.obs_noise * self.rng.normal(size=(n_pairs, self.latent_dim))
        x_prime = self._mix(z_prime) + self.obs_noise * self.rng.normal(size=(n_pairs, self.latent_dim))
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


def alpha_grid() -> np.ndarray:
    """Default alpha sweep matching the theorem's App H.7 grid (2^-3 ... 2^5)."""
    return np.array([2.0 ** p for p in range(-3, 6)], dtype=float)
