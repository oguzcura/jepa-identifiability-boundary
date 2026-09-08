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
        elif self.g == "spiral":
            # Mild, bounded, globally invertible spiral: x_j = z_j + amp*sin(z_{j+1}).
            # Triangular with unit diagonal -> invertible everywhere; |sin|<=1 keeps
            # the map scale-bounded (no tail explosion), so a modest MLP encoder can
            # actually learn the inverse at alpha=2 (Gaussian) — restoring the
            # theorem's positive regime — while SIGReg still forces non-Gaussian
            # latents through a nonlinear Gaussian-izing map (breaking linear
            # recovery exactly at non-Gaussian alpha).
            x = np.empty_like(z)
            n = z.shape[1]
            am = self.amp
            for j in range(n - 1):
                x[:, j] = z[:, j] + am * np.sin(z[:, j + 1])
            x[:, n - 1] = z[:, n - 1]
            x = x @ self.A.T
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

    S may be an int (homogeneous: every dim has S states — A4 L2m control)
    or a tuple/list of per-dim state counts (A6 L2mH control, e.g. (2,4,4)
    matching L3 v2's heterogeneous per-dim cardinalities with NO rule).

    Ground truth: the discrete state z (category index per dim).
    """
    latent_dim: int = 4
    S: int | tuple[int, ...] | list[int] = 5  # per-dim categories (int = homogeneous)
    rho: float = 0.85               # stay-probability (diagonal strength)
    g: str = "linear"
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)    # homogeneous only
    E: np.ndarray = field(init=False, repr=False)    # homogeneous only
    P_list: list = field(init=False, repr=False)     # heterogeneous per-dim
    E_list: list = field(init=False, repr=False)
    per_dim: list[int] = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if isinstance(self.S, (tuple, list)):
            # A6 L2mH: per-dim state counts; independent dims, per-dim P/E
            self.per_dim = [int(s) for s in self.S]
            assert len(self.per_dim) == self.latent_dim, \
                f"per-dim S {self.per_dim} must match latent_dim {self.latent_dim}"
            self.P_list, self.E_list = [], []
            for Sj in self.per_dim:
                base = self.rng.random((Sj, Sj))
                Pj = (base + base.T) / 2.0
                Pj = (1.0 - self.rho) * Pj + self.rho * np.eye(Sj)
                Pj /= Pj.sum(axis=1, keepdims=True)
                self.P_list.append(Pj)
                self.E_list.append(self.rng.normal(size=Sj))
            self.P, self.E = None, None
        else:
            self.per_dim = [int(self.S)] * self.latent_dim
            base = self.rng.random((self.S, self.S))
            P = (base + base.T) / 2.0
            P = (1.0 - self.rho) * P + self.rho * np.eye(self.S)
            P /= P.sum(axis=1, keepdims=True)
            self.P = P
            self.E = self.rng.normal(size=(self.S, self.latent_dim))
            self.P_list, self.E_list = None, None

    def _mix(self, z: np.ndarray) -> np.ndarray:
        # z: (n, d) int categories -> (n, d) continuous via per-dim embeddings
        n, d = z.shape
        out = np.zeros((n, d))
        if self.E_list is None:      # homogeneous: shared S, per-dim columns
            for dim in range(d):
                out[:, dim] = self.E[z[:, dim], dim]
        else:                        # heterogeneous: per-dim embedding vector
            for dim in range(d):
                out[:, dim] = self.E_list[dim][z[:, dim]]
        return out

    def generate(self, n_pairs: int) -> dict:
        n, d = self.latent_dim, self.latent_dim
        # sample two views from the Markov chain
        z = np.zeros((n_pairs, d), dtype=int)
        for dim in range(d):
            z[:, dim] = self.rng.choice(self.per_dim[dim], size=n_pairs)
        z_prime = np.zeros_like(z)
        for dim in range(d):
            Pj = self.P if self.P_list is None else self.P_list[dim]
            for i in range(n_pairs):
                z_prime[i, dim] = self.rng.choice(
                    self.per_dim[dim], p=Pj[z[i, dim]])
        x = self._mix(z) + self.obs_noise * self.rng.normal(size=(n_pairs, d))
        x_prime = self._mix(z_prime) + self.obs_noise * self.rng.normal(size=(n_pairs, d))
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


# --------------------------------------------------------------------------- #
# L3 v2: Rule-governed grammar (vowel harmony) — non-degenerate, no z leak
# (A4: replaced the degenerate affix==stem v1; x = surface embeddings, z not literal)
# --------------------------------------------------------------------------- #
VOWEL_CODES = {"i": 0, "e": 1, "ı": 2, "a": 3}


@dataclass
class L3World:
    """Rule-governed discrete latents: stem + two suffixes, vowel-harmony rule.

    z = [stem_class s (0/1), suffix1_vowel_code, suffix2_vowel_code] where each
    suffix code = s*2 + height_bit — harmony forces the front/back bit of every
    suffix vowel to equal s (a cross-position rule constraint), while height is
    free. Joint entropy H(z) = 4 bits (s, h0, h1, h2 free); NOT degenerate.

    Two views = same word (s, h0 fixed), suffix heights re-drawn under the rule
    (like L2's Markov re-draw), so the alignment task mirrors L2's.

    Observation x = E[stem_vowel] ++ E[suffix1_vowel] ++ E[suffix2_vowel] per-dim
    random embeddings + noise; E is a shared (4, obs_per_pos) random map. z is
    never a literal column of x.
    """
    latent_dim: int = 6            # obs width = 3 * obs_per_pos
    obs_per_pos: int = 2           # embedding dim per surface vowel position
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    E: np.ndarray = field(init=False, repr=False)   # (4 vowels, obs_per_pos)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.E = self.rng.normal(size=(4, self.obs_per_pos))
        self.latent_dim = 3 * self.obs_per_pos  # obs width is what train.py reads

    @property
    def n_states_per_dim(self) -> int:
        """Per-dim state count for entropy-matched L2m (S = round(2^H̄))."""
        return 4  # s has 2; suffix codes have 4 -> conservative bound

    def _embed(self, codes: np.ndarray) -> np.ndarray:
        # codes: (n, 3) int in 0..3 -> (n, 3*obs_per_pos) via shared E
        n = codes.shape[0]
        out = np.zeros((n, 3 * self.obs_per_pos))
        for pos in range(3):
            out[:, pos * self.obs_per_pos:(pos + 1) * self.obs_per_pos] = \
                self.E[codes[:, pos]]
        return out

    def generate(self, n_pairs: int) -> dict:
        # free bits
        s = self.rng.integers(0, 2, size=n_pairs)          # stem class
        h0 = self.rng.integers(0, 2, size=n_pairs)         # stem vowel height
        h1 = self.rng.integers(0, 2, size=n_pairs)         # suffix1 height
        h2 = self.rng.integers(0, 2, size=n_pairs)         # suffix2 height
        sv = 2 * s + h0                                    # stem vowel code
        av1 = 2 * s + h1                                   # suffix1 (rule)
        av2 = 2 * s + h2                                   # suffix2 (rule)
        # second view: same word, suffix heights re-drawn (rule still holds)
        h1p = self.rng.integers(0, 2, size=n_pairs)
        h2p = self.rng.integers(0, 2, size=n_pairs)
        av1p = 2 * s + h1p
        av2p = 2 * s + h2p
        # ground truth: [stem_class, suffix1_code, suffix2_code]
        z = np.stack([s, av1, av2], axis=1)
        z_prime = np.stack([s, av1p, av2p], axis=1)
        # surface observations (embeddings; z not literal in x)
        surf = np.stack([sv, av1, av2], axis=1)
        surf_p = np.stack([sv, av1p, av2p], axis=1)
        x = self._embed(surf)
        x_prime = self._embed(surf_p)
        x += self.obs_noise * self.rng.normal(size=x.shape)
        x_prime += self.obs_noise * self.rng.normal(size=x_prime.shape)
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime,
                "surf": surf}



# --------------------------------------------------------------------------- #
# L4 v2: Turkish orthography rules — caret (düzeltme) with negative class
# (A4: v1 had caret_required=1 for every word (zero variance) and x leaked
# caret_idx/class/flag as literal columns. v2 adds a matched non-caret lexicon
# and exposes only SURFACE letters.)
# --------------------------------------------------------------------------- #
# Turkish vowel harmony: front {e,i,ö,ü}, back {a,ı,o,u}. The caret rule
# (düzeltme işareti) marks a long /a/ or /u/ in loan words (â, û) when the
# plain spelling would mislead. Here z = (stem_class, caret_required,
# caret_vowel_code) with caret_vowel_code: 0=none, 1=â(a), 2=û(u).
# Observation x = per-position one-hot of the surface letters (NO caret marks,
# NO class/flag columns) + length + noise: the model must infer the rule-
# governed latents from the letter pattern alone; nothing in x equals z.
FRONT_VOWELS = {"e", "i", "ö", "ü"}
BACK_VOWELS = {"a", "ı", "o", "u"}
ALPHABET = tuple("adehiklmoprstuy")  # all letters used in ORTHO_LEXICON

# word -> (caret_vowel_code, caret_char); None in value => no caret required
ORTHO_LEXICON = {
    # caret REQUIRED (loan words with long vowel): vowel_code 1=â, 2=û
    "adet":  (1, "a"),   # âdet
    "kar":   (1, "a"),   # kâr
    "alem":  (1, "a"),   # âlem
    "dahi":  (1, "a"),   # dâhi
    "suret": (2, "u"),   # sûret
    "memur": (2, "u"),   # memûr
    # caret FORBIDDEN (native words, matched length/vowel structure)
    "kitap": (0, ""),    # back vowels, no caret
    "masa":  (0, ""),
    "okul":  (0, ""),
    "kedi":  (0, ""),    # front vowels, no caret
    "sepet": (0, ""),
    "yol":   (0, ""),
}


@dataclass
class L4World:
    """Turkish caret-rule generator with a real negative class (A4 v2).

    z columns: [stem_class 0/1 (front/back by last vowel), caret_required 0/1,
    caret_vowel_code 0/1/2 (none/â/û)] — every column has genuine variance.
    x columns: surface letters one-hot over ALPHABET for up to MAXLEN positions
    + length + noise. caret marks are NOT in x; the rule latents must be
    inferred from letter patterns. Two views = the same word re-noised
    (stationary recognition task, as in v1).
    """
    obs_noise: float = 0.05
    seed: int = 0
    maxlen: int = 5
    rng: np.random.Generator = field(init=False, repr=False)
    vocab: tuple = field(init=False, repr=False)
    obs_width: int = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.vocab = tuple(sorted(ORTHO_LEXICON))
        self.obs_width = self.maxlen * len(ALPHABET) + 1

    @staticmethod
    def stem_class_of(word: str) -> int:
        vowels = [c for c in word if c in FRONT_VOWELS or c in BACK_VOWELS]
        last = vowels[-1] if vowels else "a"
        return 0 if last in FRONT_VOWELS else 1

    def _surface(self, words: np.ndarray) -> np.ndarray:
        n = len(words)
        x = np.zeros((n, self.maxlen * len(ALPHABET) + 1))
        for i, w in enumerate(words):
            for p, ch in enumerate(w):
                if ch in ALPHABET and p < self.maxlen:
                    x[i, p * len(ALPHABET) + ALPHABET.index(ch)] = 1.0
            x[i, -1] = len(w) / self.maxlen  # normalized length feature
        return x

    def generate(self, n_pairs: int) -> dict:
        words = self.rng.choice(self.vocab, size=n_pairs)
        z = np.zeros((n_pairs, 3), dtype=int)
        for i, w in enumerate(words):
            vcode, vchar = ORTHO_LEXICON[w]
            z[i] = [self.stem_class_of(w), 1 if vcode else 0, vcode]
        z_prime = z.copy()  # same word, stationary
        x = self._surface(words)
        x_prime = x.copy()
        x += self.obs_noise * self.rng.normal(size=x.shape)
        x_prime += self.obs_noise * self.rng.normal(size=x_prime.shape)
        return {"z": z, "z_prime": z_prime, "x": x, "x_prime": x_prime}


# --------------------------------------------------------------------------- #
# L2t: Continuous moment-matched twin of L2 (R1 / A5)
# --------------------------------------------------------------------------- #
@dataclass
class L2TwinWorld:
    """Continuous twin of L2: identical observable moments, continuous support.

    R1 control (A5): L2's categorical latents (S point masses per dim, uniform)
    vs this world's Gaussian AR(1) latents — same per-dim mean/variance of the
    OBSERVED x (matched by construction via the SAME embedding matrix E as
    L2World with the same seed: x_marginal moments are fixed by E's column
    statistics), same persistence rho, same obs noise, same diagonal per-dim
    observation structure. The ONLY difference is the latent support: S point
    masses vs a continuum with identical 1st/2nd moments.

    z_cont ~ N(0,1) AR(1):  z_cont' = rho*z_cont + sqrt(1-rho^2)*eps
    x[:,d] = m_d + s_d * z_cont[:,d] + obs_noise*eps    (m_d,s_d = E[:,d] col stats)

    zK = z_cont quantized into S equiprobable bins (theoretical N(0,1) quantiles)
    -> discrete readout comparable to L2's acc_mean: same #classes (S), same
    marginal entropy (log2 S per dim), matched moments — only support differs.
    """
    latent_dim: int = 4
    S: int = 5
    rho: float = 0.85
    g: str = "linear"
    obs_noise: float = 0.05
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        from scipy.stats import norm
        self.rng = np.random.default_rng(self.seed)
        # identical E/P construction to L2World (same draw order) so that
        # per-dim observable moments match L2(S, seed) exactly
        base = self.rng.random((self.S, self.S))
        P = (base + base.T) / 2.0
        P = (1.0 - self.rho) * P + self.rho * np.eye(self.S)
        P /= P.sum(axis=1, keepdims=True)
        self.P = P
        self.E = self.rng.normal(size=(self.S, self.latent_dim))
        # per-dim column statistics of E -> observable moments of the twin
        self.m = self.E.mean(axis=0, keepdims=True)        # (1, d)
        self.s = self.E.std(axis=0, keepdims=True) + 1e-6  # (1, d)
        # theoretical N(0,1) quantile edges for S equiprobable bins
        self.edges = norm.ppf(np.arange(1, self.S) / self.S)

    def _bin(self, z_cont: np.ndarray) -> np.ndarray:
        idx = np.zeros(z_cont.shape, dtype=int)
        for e in self.edges:
            idx = idx + (z_cont > e).astype(int)
        return np.clip(idx, 0, self.S - 1)

    def generate(self, n_pairs: int) -> dict:
        z = self.rng.normal(size=(n_pairs, self.latent_dim))
        z_prime = self.rho * z + np.sqrt(1.0 - self.rho ** 2) * self.rng.normal(size=z.shape)
        x = self.m + self.s * z + self.obs_noise * self.rng.normal(size=z.shape)
        x_prime = self.m + self.s * z_prime + self.obs_noise * self.rng.normal(size=z_prime.shape)
        return {
            "z": z, "z_prime": z_prime,
            "zK": self._bin(z), "zK_prime": self._bin(z_prime),
            "x": x, "x_prime": x_prime,
        }


WORLD_REGISTRY = {
    "L0": L0World,
    "L1": L1World,
    "L2": L2World,
    "L2t": L2TwinWorld,
    "L3": L3World,
    "L4": L4World,
}
