# Theorem Notes — LeJEPA Identifiability (arXiv:2605.26379)

**Source:** D. Klindt, Y. LeCun, R. Balestriero, *"When Does LeJEPA Learn a World Model?"*, arXiv:2605.26379 (May 25, 2026). All five results formally verified in Lean 4 + Mathlib, zero `sorry`; a few background lemmas (Hermite properties, Mazur–Ulam) taken as axioms.
**Extracted:** 2026-09-06 from full text (ar5iv). **Verification:** id_list ✓, full-text grep ✓.

## The setup (World Model)

- True latent variables $z \in \mathbb{R}^n$ (a.k.a. sources / factors of variation). Never observed.
- Unknown nonlinear mixing $g$: $x = g(z)$ renders the world into observations (Plato's cave framing).
- Learned encoder $f$: $y = f(x)$.
- **Identifiability is always a joint statement about the World (DGP) and the Learner (objective).**

### Assumptions 3.1 (World)
1. **Independence.** $p(z_i) \perp p(z_j)$ and $p(z'_i \mid z_i) \perp p(z'_j \mid z_j)$ for all $i \neq j$.
2. **Stationarity.** Both views share the same marginal: $p(z) = p(z')$.
3. **Additive noise.** $z'_i = m_i(z_i) + \eta_i$, $\eta_i \perp z_i$.

### 3.1.1 The Gaussian World
- $z \sim \mathcal{N}(0, I_n)$.
- Gaussian latents + Assumptions 3.1 ⇒ Gaussian transitions. The *only* additive-noise perturbation of a Gaussian that preserves the distribution is the **Ornstein–Uhlenbeck (OU) transition**:

$$z' = \rho\, z + \sqrt{1-\rho^2}\,\eta, \qquad \eta \sim \mathcal{N}(0, I_n), \ \eta \perp z, \quad \rho \in (0,1).$$

(Equation 1.) $\rho$ controls the correlation between the two views.

## The three objectives (hierarchy of Gaussianity constraints)

| Objective | Gaussianity constraint | Nature |
|---|---|---|
| InfoNCE | implicit | contrastive, negatives |
| VICReg | second-moment | covariance regularization |
| LeJEPA (alignment + **SIGReg**) | full | Sketched Isotropic Gaussian Regularization — pushes embedding distribution toward $\mathcal{N}(0,I)$ |

## Theorems

### Theorem 5.1 (LeJEPA Linear Identifiability) — forward
> Consider the Gaussian world (3.1.1). Let $h: \mathbb{R}^n \to \mathbb{R}^n$ be any measurable map with $h(z) \sim \mathcal{N}(0, I_n)$. Then $\mathcal{L}(h) \geq 2(1-\rho)n$, with equality **iff** $h$ recovers the latents and transition dynamics up to a global rotation. (Proof App. A; Sturm–Liouville spectral theory.)

Interpretation: any representation satisfying both LeJEPA objectives must recover a rotation/reflection of the true latents *and* the true transition dynamics. Only remaining ambiguity is a global rotation (inherent to isotropic Gaussian).

### Theorem 5.2 (Gaussian Uniqueness) — converse
> Consider any world satisfying Assumptions 3.1. Suppose every minimizer of (3) with $\mathrm{Cov}(h(z)) = I_n$ is linear, $h(z) = Qz$. Then $z$ is **Gaussian**.

**This is the load-bearing converse for our project:** it rules out every non-Gaussian alternative — including discrete/categorical/symbolic latents. **Non-Gaussian latents break linear identifiability.**

### Theorem 5.3 (Approximate Identifiability)
> In the Gaussian world, if $h$ has $\mathbb{E}[h(z)]=0$ and satisfies *approximate* alignment ($\mathcal{L}(h) \leq 2(1-\rho)\,\mathrm{tr}(\mathrm{Cov}(h(z))) + \varepsilon$, roughly), the guarantee degrades **gracefully**.

(Roughly: a bounded violation of alignment ⇒ bounded deviation from identifiability — the "graceful degradation" result.)

## Section 6.2 — Latent-Distribution Sweep (what they ALREADY did)

- Sweep latent through **generalized normal family**, shape $\alpha$: $\alpha \to 0$ heavy-tailed, $\alpha=1$ Laplace, **$\alpha=2$ Gaussian**, $\alpha \to \infty$ uniform.
- App H.7: $\alpha \in \{2^{-3}, \dots, 2^5\}$, across multiple mixings.
- **Linear recovery peaks sharply at $\alpha = 2$ across all three objectives (Fig 4b).** SIGReg and InfoNCE retain a wider plateau than VICReg for heavy-tailed latents.
- **Pixel-Based RL (DMC Reacher):** 2D latent $z=(\theta_0,\theta_1)$. OU pairs reach $R^2 = 0.95$ at $\rho = 0.99$; real trajectories (non-Gaussian, anisotropic $\rho_0\neq\rho_1$, joint-limit wrapping) → total $R^2$ **never exceeds 0.5**.

## What this means for OUR project (the gap)

- The theorem proves **iff Gaussian**. Their experiments sweep the *continuous* generalized-normal family and real *continuous* (RL) trajectories.
- **Discrete latents are categorically outside Assumptions 3.1's density machinery** (no smooth density, no score function, no Sturm–Liouville spectrum). The converse (Thm 5.2) says they *cannot* be linearly identifiable, but **nobody has empirically mapped *how* they fail** — the failure mode, whether it degrades "gracefully" (Thm 5.3-style) or collapses qualitatively, and whether compositional/rule structure behaves differently from unstructured categorical noise.
- **Our ladder (L0–L4) fills exactly this:** L0 reproduces their Fig 4b (sanity), L1 bridges continuous→discrete via binning, L2 Markov chains, L3 rule grammars, L4 Turkish morphology (real case).
- **Metric note:** R² is the right tool on L1's continuous z-bridge, but for genuinely categorical targets we use linear-probe top-1 accuracy + AMI + purity. See pre-registration.

## Direct quotations (for the paper's related-work section)

> "We prove that LeJEPA (alignment plus Gaussian regularization) linearly recovers the world's latent variables from nonlinear observations, a property known as *linear identifiability*... our main result is that among all such worlds, the Gaussian is the unique latent distribution for which this guarantee holds."

> (Sec 6.2) "We sweep the latent variable through the generalized normal family with shape parameter α (α→0 heavy-tailed, α=1 Laplace, α=2 Gaussian, α→∞ uniform). Linear recovery peaks sharply at α=2 across all three objectives (Fig. 4b)."
