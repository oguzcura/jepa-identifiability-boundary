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
    amp: float = 0.5            # nonlinear mixing amplitude (0 = linear)
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
        else:
            # Genuinely nonlinear but globally invertible mixing (unit Jacobian
            # determinant: x_j = z_j + 0.5 z_{j+1}^2, triangular with unit
            # diagonal). Strong nonlinearity + no information loss — the
            # regime where the theorem predicts linear identifiability fails
            # for non-Gaussian latents (SIGReg Gaussian-izes the embedding,
            # producing a nonlinear distortion h(z) that linear probes can't
            # invert).
            x = np.empty_like(z)
            n = z.shape[1]
            am = self.amp
            x[:, 0] = z[:, 0] + am * z[:, 1] ** 2
            for j in range(1, n - 1):
                x[:, j] = z[:, j] + am * z[:, j + 1] ** 2
            x[:, n - 1] = z[:, n - 1]
            x = x @ self.A.T
        return x

    def generate(self, n_pairs: int) -> dict:
        """Return a positive pair (z, z') with observations (x, x')."""
        z = self._sample_z(n_pairs)
        z_prime = self._transition(z)
        x = self._mix(z) + self.obs_noise * self.rng.normal(size=(n_pairs, self.latent_dim))
        x_prime = self._mix(z_prime) + self.obs_noise * self.rng.normal(size=(n_pairs, self.latent_dim))
        # winsorize extreme observations (heavy-tail worlds) — info-preserving
        # for the bulk, prevents training divergence from rare outliers
        x = np.clip(x, -10.0, 10.0)
        x_prime = np.clip(x_prime, -10.0, 10.0)
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


def alpha_grid() -> np.ndarray:
    """Default alpha sweep matching the theorem's App H.7 grid (2^-3 ... 2^5)."""
    return np.array([2.0 ** p for p in range(-3, 6)], dtype=float)


# --------------------------------------------------------------------------- #
# L1: Discretized Gaussian (K bins) — bridge from continuous to discrete
# --------------------------------------------------------------------------- #
@dataclass
class L1World:
    """Gaussian latents binned into K equal-probability categories per dim.

    Draw z_cont ~ N(0,1), then map each coordinate to a category index via
    quantile bins (equal probability). K is the "discreteness dial": large K
    approximates the continuous case, small K is fully discrete. Both the
    continuous z (for R^2 bridge) and the binned z^K (for accuracy/AMI) are
    returned as ground truth.

    Transition: z'_cont = rho z_cont + sqrt(1-rho^2) eta (Gaussian OU), then
    the same binned projection applies to z'.
    """
    latent_dim: int = 4
    rho: float = 0.9
    K: int = 8                     # bins per dimension
    g: str = "linear"
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # quantile bin edges for N(0,1), K equal-probability bins
        self.edges = np.linspace(0.0, 1.0, self.K + 1)
        self.edges = np.array([self._inv_phi(p) for p in self.edges[1:-1]])
        self.A = np.linalg.qr(self.rng.normal(size=(self.latent_dim, self.latent_dim)))[0]

    @staticmethod
    def _inv_phi(p: float) -> float:
        # inverse standard normal CDF via scipy
        from scipy.stats import norm
        return float(norm.ppf(p))

    def _bin(self, z_cont: np.ndarray) -> np.ndarray:
        # equal-probability binning per dimension -> int categories 0..K-1
        idx = np.zeros(z_cont.shape, dtype=int)
        for k, e in enumerate(self.edges):
            idx = idx + (z_cont > e).astype(int)
        return np.clip(idx, 0, self.K - 1)

    def _mix(self, z_cont: np.ndarray) -> np.ndarray:
        if self.g == "linear":
            return z_cont @ self.A.T
        # nonlinear: mixing on continuous z before binning
        return np.tanh(z_cont) @ self.A.T

    def generate(self, n_pairs: int) -> dict:
        zc = self.rng.normal(size=(n_pairs, self.latent_dim))
        zc_p = self.rho * zc + np.sqrt(1.0 - self.rho ** 2) * self.rng.normal(size=zc.shape)
        x = self._mix(zc) + self.obs_noise * self.rng.normal(size=zc.shape)
        x_p = self._mix(zc_p) + self.obs_noise * self.rng.normal(size=zc_p.shape)
        return {
            "z": zc, "z_prime": zc_p,          # continuous ground truth (R^2 bridge)
            "zK": self._bin(zc), "zK_prime": self._bin(zc_p),  # discrete (accuracy/AMI)
            "x": x, "x_prime": x_p,
        }


# --------------------------------------------------------------------------- #
# L2: Markov-chain categorical latents
# --------------------------------------------------------------------------- #
@dataclass
class L2World:
    """Genuinely categorical latent states over a stationary Markov chain.

    z_t in {0..S-1}^d evolves via a transition matrix P (row-stochastic),
    drawn once (seeded) and stationary (satisfies detailed balance / uniform
    invariant distribution for symmetric P). Observations x = one-hot-style
    embedding + noise, so the encoder must recover discrete structure.

    Ground truth: the discrete state z (category index per dim).
    """
    latent_dim: int = 4
    S: int = 5                       # number of categories per dimension
    rho: float = 0.85               # stay-probability (diagonal strength)
    g: str = "linear"
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # symmetric transition matrix with strong diagonal (persistence)
        base = self.rng.random((self.S, self.S))
        P = (base + base.T) / 2.0
        P = (1.0 - self.rho) * P + self.rho * np.eye(self.S)
        P /= P.sum(axis=1, keepdims=True)
        self.P = P
        # mixing matrix maps category to continuous obs embedding
        self.E = self.rng.normal(size=(self.S, self.latent_dim))

    def _mix(self, z: np.ndarray) -> np.ndarray:
        # z: (n, d) int categories -> (n, d) continuous via per-dim embeddings
        n, d = z.shape
        out = np.zeros((n, d))
        for dim in range(d):
            out[:, dim] = self.E[z[:, dim], dim]
        return out

    def generate(self, n_pairs: int) -> dict:
        n, d = self.latent_dim, self.latent_dim
        # sample two views from the Markov chain
        z = np.zeros((n_pairs, d), dtype=int)
        for dim in range(d):
            z[:, dim] = self.rng.choice(self.S, size=n_pairs)
        z_prime = np.zeros_like(z)
        for dim in range(d):
            for i in range(n_pairs):
                z_prime[i, dim] = self.rng.choice(
                    self.S, p=self.P[z[i, dim]])
        x = self._mix(z) + self.obs_noise * self.rng.normal(size=(n_pairs, d))
        x_prime = self._mix(z_prime) + self.obs_noise * self.rng.normal(size=(n_pairs, d))
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


# --------------------------------------------------------------------------- #
# L3: Toy rule-governed grammar (vowel harmony / agreement)
# --------------------------------------------------------------------------- #
@dataclass
class L3World:
    """Rule-governed discrete latents: a stem+affix grammar with harmony.

    Each latent sample is a (stem_vowel_class, affix_vowel) pair generated by an
    explicit harmony rule, so the true latent structure is compositional and
    rule-constrained (not just i.i.d. categories). Observation = one-hot of the
    vowel classes + stem identity + noise.

    Rule (Turkish-like): affix vowel agrees with stem vowel class
    (front/back harmony). Ground truth z encodes (stem_class, affix_should).
    """
    latent_dim: int = 6
    stem_classes: int = 2            # front/back
    affix_choices: int = 2           # e.g. -e/-a
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def generate(self, n_pairs: int) -> dict:
        # stem vowel class (front=0/back=1), uniformly
        stem = self.rng.integers(0, self.stem_classes, size=n_pairs)
        # HARMONY RULE: affix vowel = stem class (compositional constraint)
        affix = stem  # affix agrees with stem (the rule)
        # second view: same stem, affix re-sampled but must still satisfy rule
        affix_p = affix.copy()
        # ground truth latents: stack stem + affix
        z = np.stack([stem, affix], axis=1)          # (n,2)
        z_prime = np.stack([stem, affix_p], axis=1)  # (n,2)
        # observations: encode stem and affix + filler dims
        x = np.zeros((n_pairs, self.latent_dim))
        x[:, 0] = stem
        x[:, 1] = affix
        x[:, 2] = stem  # redundant stem (compositional info)
        x[:, 3] = affix
        x[:, 4] = (stem + affix) % 2   # harmony consistency feature
        x[:, 5] = self.rng.normal(size=n_pairs)  # irrelevant filler
        x_prime = x.copy()
        x_prime[:, 1] = affix_p
        x_prime[:, 4] = (stem + affix_p) % 2
        x += self.obs_noise * self.rng.normal(size=x.shape)
        x_prime += self.obs_noise * self.rng.normal(size=x_prime.shape)
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


# --------------------------------------------------------------------------- #
# L4: Turkish orthography/morphology — Imla rule set as ground truth
# --------------------------------------------------------------------------- #
# Vowel harmony classes (Turkish). Front vowels: e,i,ö,ü ; back: a,ı,o,u.
# Caret rule (düzeltme işareti): a long /a/ or /u/ in certain loan words is
# written with ^ when it would otherwise be misread (e.g. â, î, û, âdet, kâr).
# The Imla benchmark's mechanical rule set is the ground truth here.
FRONT_VOWELS = {"e", "i", "ö", "ü"}
BACK_VOWELS = {"a", "ı", "o", "u"}
CARET_WORDS = {  # word -> (vowel_index, vowel_char, front_back)
    "adet": (1, "a", "b"),      # âdet (long a)
    "kar": (1, "a", "b"),       # kâr (long a)
    "hala": (1, "a", "b"),      # hâlâ
    "alem": (1, "a", "b"),      # âlem
    "sair": (1, "a", "b"),      # şâir
    "dahi": (1, "a", "b"),      # dâhi
    "memur": (2, "u", "b"),     # memûr
    "suret": (1, "u", "b"),     # sûret
}
STEM_CLASSES = ["front", "back"]


@dataclass
class L4World:
    """Turkish orthography rule generator (Imla rule set as ground truth).

    Each sample is a word drawn from a lexicon; the true latent structure is
    governed by explicit TDK rules: (1) vowel-harmony class of the stem,
    (2) whether a caret is REQUIRED (düzeltme işareti), and (3) the caret
    position. The model must recover these rule-governed discrete latents.

    Ground truth z: [stem_class(0/1), caret_required(0/1), caret_char_idx].

    Observation x: a featurized surface form (one-hot of stem identity, length,
    vowel positions) + noise — the "text-like" observation the encoder sees.
    """
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    vocab: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.vocab = tuple(CARET_WORDS.keys())

    @staticmethod
    def stem_class_of(word: str) -> int:
        """0=front, 1=back, using last vowel (Turkish vowel harmony rule)."""
        vowels = [c for c in word if c in FRONT_VOWELS or c in BACK_VOWELS]
        last = vowels[-1] if vowels else "a"
        return 0 if last in FRONT_VOWELS else 1

    def generate(self, n_pairs: int) -> dict:
        words = self.rng.choice(self.vocab, size=n_pairs)
        z = np.zeros((n_pairs, 3), dtype=int)
        x = np.zeros((n_pairs, 8), dtype=float)  # fixed featurized width
        for i, w in enumerate(words):
            cls = self.stem_class_of(w)
            caret_required = 1  # all CARET_WORDS require caret by construction
            caret_idx, caret_char, _ = CARET_WORDS[w]
            z[i] = [cls, caret_required, caret_idx]
            # featurized observation: vocab id, length, caret pos, stem class flag
            x[i, 0] = self.vocab.index(w)
            x[i, 1] = len(w)
            x[i, 2] = caret_idx
            x[i, 3] = cls
            x[i, 4] = 1.0 if caret_required else 0.0
            x[i, 5] = 1.0 if caret_char == "a" else 0.0
            x[i, 6] = 1.0 if caret_char == "u" else 0.0
            x[i, 7] = self.rng.normal()  # irrelevant filler
        # second view: same word (stationary), caret position preserved
        z_prime = z.copy()
        x_prime = x.copy()
        x += self.obs_noise * self.rng.normal(size=x.shape)
        x_prime += self.obs_noise * self.rng.normal(size=x_prime.shape)
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


WORLD_REGISTRY = {
    "L0": L0World,
    "L1": L1World,
    "L2": L2World,
    "L3": L3World,
    "L4": L4World,
}
