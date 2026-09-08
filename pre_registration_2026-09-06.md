# Pre-registration — The Discrete Boundary: Identifiability of Symbolic and Rule-Governed Worlds in Joint-Embedding Predictive Models

**Date:** 2026-09-06
**Status:** FROZEN — do not modify after the first grid run is launched. Amendments only as append-only sections at the bottom, dated and signed.
**Commit at freezing:** (FROZEN on commit — see git log)
**Git hash of this file (self-hash, SHA-256):** `cdd123d0317caa5eb24cf4437d63996717eb7a7729a48675772d637a8787572d`

This contract is registered BEFORE any experimental data is collected. All hypotheses,
decision rules, seeds, grid cells, and analysis procedures are fixed here. The intent is
confirmatory analysis: results are reported against these pre-specified rules, and any
post-hoc analysis is clearly labeled exploratory.

---

## 1. Research question (one sentence)

Does the LeJEPA linear-identifiability guarantee (and its failure mode) extend to
**discrete, categorical, and rule-governed generative latent worlds** — the regime
language lives in — and if it fails, **how exactly does it fail** (graceful degradation
vs. qualitative collapse), in a way an Imla-style rule-verifiable probe can detect and
characterize?

## 2. Theoretical anchor

Klindt, LeCun & Balestriero (arXiv:2605.26379, Lean-4-verified):
- **Thm 5.1 (forward):** in the Gaussian world (OU transition, $z\sim\mathcal{N}(0,I_n)$), the alignment + SIGReg objective recovers latents up to global rotation.
- **Thm 5.2 (converse):** under Assumptions 3.1 (independence, stationarity, additive noise), linear identifiability implies $z$ is Gaussian. **⇒ non-Gaussian latents break linear identifiability.**
- **Thm 5.3 (approximate):** the guarantee degrades *gracefully* with bounded alignment violation (in the Gaussian world).

Their empirical validation (Sec 6.2) sweeps the **continuous** generalized-normal family
(α: heavy-tailed→Laplace→Gaussian→uniform; recovery peaks at α=2, Fig 4b) and real
continuous RL trajectories. **Discrete/categorical latents are not tested anywhere in the
prior literature** (verified 2026-09-06; see `notes/research_direction_v2_discrete_ladder.md`).

## 3. Hypotheses (primary, pre-specified)

Let $R$ be our discrete-recovery score (see §5) on a held-out split, per (world, model, seed).

- **H0 (continuity holds):** discretizing a Gaussian latent degrades recovery *exactly as
  predicted by matching continuous moments*. I.e., a K-binned Gaussian (L1) behaves like the
  continuous α-sweep at the equivalent moment-matching point — discreteness is not a new
  failure mode. **Decision rule:** $R_{L1,K}$ matches the continuous $\alpha$-sweep $R(\alpha_{eq})$ within CI at every K.
- **H1a (qualitative collapse):** discrete latents break recovery **worse** than any
  moment-matched continuous analog — recovery drops discontinuously (not gracefully). 
  **Decision rule:** $R_{L2}$ is below the moment-matched continuous prediction by > 0.3
  (absolute) with the 95% CI excluding that bound.
- **H1b (composition buys identifiability):** rule-governed/compositional discrete worlds
  (L3) recover **better** than unstructured categorical worlds (L2) at matched marginal
  statistics — compositional structure itself is a resource independent of the Gaussian
  mechanism. **Decision rule:** $R_{L3} - R_{L2} > 0.2$ with 95% CI excluding 0.
- **H2 (insurance; reportable regardless of H0/H1):** SIGReg's Gaussian-forcing measurably
  blurs/merges genuinely discrete clusters relative to a non-Gaussian-pressured objective.
  **Decision rule:** cluster-purity gap between JEPA-core (SIGReg) and reconstruction
  baseline > 0.1 with CI excluding 0, in L2/L3/L4.

## 4. World ladder (the factor: discreteness)

| Lvl | World generator | Discreteness control | Ground truth |
|---|---|---|---|
| **L0** | Gaussian OU + gennorm α-sweep | α ∈ {2⁻³,…,2⁵} (reproduce their Fig 4b) | $z$ known, Gaussian |
| **L1** | Discretized Gaussian (K bins per dim) | **K ∈ {2, 4, 8, 16}** (dial: small=discrete, large≈continuous) | $z$ known + binned $z^K$ |
| **L2** | Markov-chain latents (categorical) | transition matrix sparsity / state count | $z \in \{1..S\}^d$, known P |
| **L3** | Toy rule-governed grammar (vowel harmony / agreement) | rule set size, composition depth | rule-generated $z$, fully known |
| **L4** | Turkish orthography/morphology rules | — | Imla rule set = ground truth |

Each world: stationary, with a controllable observation map $x = g(z) + \epsilon$ (both
linear and nonlinear $g$ variants). Data is **procedurally generated** → contamination-proof
by construction.

## 5. Models (the factor: objective)

Three objectives, matched capacity, all trained from scratch (no pretrained weights):

1. **JEPA-core** — context encoder + predictor + EMA target encoder, alignment loss in latent space, **SIGReg** Gaussian regularizer (this is our primary object of study; mirrors LeWorldModel's anti-collapse mechanism).
2. **Reconstruction baseline** — predicts $x$ in input space (MAE-style) — no Gaussian pressure on the latent.
3. **Contrastive baseline** — InfoNCE (explicit negatives).

Matched: encoder architecture (MLP for L0-L3; small ViT/MLP for L4), optimizer (Adam),
lr, batch size, steps, latent dim, seeds.

## 6. Recovery metrics (the response; adapted because R² doesn't transfer to categoricals)

Per (world, model, seed), on a **held-out split**:
- **Linear-probe top-1 accuracy** (ridge/logistic probe $z \leftarrow h$): primary for discrete $z$.
- **Adjusted Mutual Information (AMI)** between true $z$ and a $k$-means clustering of $h$: geometry-agnostic cross-check.
- **Cluster purity** (fraction of majority-class in each $k$-means cluster): for H2.
- **Linear-probe $R^2$** (ridge): **only** on L1's continuous $z$ bridge (to link to the literature) and L0 (Fig 4b reproduction).
- **Collapse diagnostics:** effective rank / variance-explained spectrum of $\mathrm{Cov}(h)$.

Metric design is itself a small methodological contribution (no prior discrete-recovery
metric for predictive SSL exists; we define and validate these).

## 7. Design & factorial structure

Full factorial, completely randomized (seeded), units = independent training runs:

```
world ∈ {L0,L1,K,L2,L3,L4}        (L1 has K∈{2,4,8,16})
g     ∈ {linear, nonlinear}
model ∈ {jepa, recon, contrastive}
seed  ∈ {0,1,2,3,4}               (5 replicates)
latent_dim ∈ {8, 32}              (L0-L3); fixed 128 (L4)
```

**Replication unit = the training run** (independent seed → independent optimizer
trajectory). Repeated probes within a run are NOT independent replicates
(pseudoreplication guard per experimental-design skill). $n=5$ seeds per cell.

Estimated cell count (full): roughly (L0: 9 α) + (L1: 4 K) + (L2, L3) + (L4) ×
2 g × 3 model × 5 seed × 2 dim ≈ **~2,400–3,600 runs** at toy scale (minutes each on
RTX 4070). Stage 1 pilot (below) calibrates the true per-run cost before the full grid.

## 8. Grid launch plan (staged, fast-pass first)

- **Stage 1 (pilot, FAST-PASS — must complete first, days 1–3):** L0 (all α) + L1 (all K),
  1 seed, linear g. **Gate:** if L0 does NOT reproduce their Fig 4b peak at α=2
  (within CI), the instrument is broken — STOP, fix pipeline, re-run before anything else.
- **Stage 2 (L2 + L3):** the discrete core. Full factors, 5 seeds.
- **Stage 3 (L4):** Turkish morphology case study.
- Each stage is an independent overnight batch; checkpointable and resumable.

## 9. Analysis plan (confirmatory)

1. For each cell: point estimates + **95% bootstrap CIs** (resample over seeds, n=1000).
2. **H0/H1a/H1b/H2 tests:** per-hypothesis contrast defined in §3, evaluated with the
   pre-specified decision rule and **Holm-corrected** across the family of tests.
3. **Effect sizes:** Cohen's $d$ for recovery differences between regimes.
4. **Figures (pre-planned):**
   - F1: L0 Fig 4b reproduction (sanity).
   - F2 (**the boundary map**): recovery score vs. discreteness (K on L1 / state count on L2),
     three model curves + CI bands.
   - F3 (**the decoupling figure**): prediction-MSE (flat?) vs. recovery (collapsing?) —
     "predicting well without understanding."
   - F4: compositional-vs-unstructured (L3 vs L2) contrast for H1b.
   - F5: SIGReg cluster-blur portrait for H2.
5. **Honest-null discipline:** if H0/H1 don't separate, report the null cleanly (your brand).
   Any deviation from this analysis is labeled exploratory.

## 10. Reproducibility & integrity

- All seeds, configs, and generated worlds **md5-pinned** in `evidence/manifest_md5.txt`.
- All runs logged to `results/` as JSONL with full config hash.
- Pre-registration **frozen + committed** before Stage 1 launch (this file).
- **No third-party item text** redistributed; worlds are procedurally generated by us.
- Cost: $0 (local RTX 4070); token spend logged to the cent.
- AI-assisted workflow disclosed in the paper (per ethics convention).

## 11. Licensing & ethics

- Code: MIT. Paper/results: CC BY 4.0.
- Procedural worlds contain no human data; Turkish morphology ground truth from the
  TDK rule set (as in the Imla benchmark) — no item redistribution.

## 12. Open questions flagged for resolution BEFORE Stage 1 (not data-dependent)

- Exact α-grid spacing to match Fig 4b; whether to reuse LeWorldModel codebase or
  implement minimal JEPA-core from scratch (decision: implement minimal + validate against
  LeWorldModel on L0).
- Scaling rule for steps vs. latent dim (to keep per-run cost bounded).

---

## Appendix A — Amendments (append-only)

### A1 (2026-09-07) — Interventional/counterfactual reframe checked; core unchanged
A proposed reframe — *"predict what survives intervention, not the future"* (Interventional
Predictive Learning / Counterfactual JEPA) — was stress-tested live. **Verdict: pre-empted.**
It is the founding premise of Causal Representation Learning (von Kügelgen et al., 2022,
arXiv:2209.11924), already brought into the JEPA lane by **"On the Identifiability of
Controlled World Models"** (arXiv:2607.22430, Jul 2026, joint rep+transition identifiability
for action-conditioned LeJEPA world models), and the name "Counterfactual JEPA" collides with
**Causal-JEPA** (arXiv:2602.11389). **Therefore: v2's L0–L4 ladder remains the core; we do NOT
build the paper's identity on this reframe.** The defensible intersection that survives
contact with 2607.22430: that paper assumes **Gaussian** latents, so the **controlled
(action-conditioned) identifiability question in discrete/non-Gaussian/rule-governed worlds**
is the possible open residue — tracked as a *possible extension* (an action/intervention axis
on the discrete levels), NOT part of the frozen primary hypotheses. CRL, 2607.22430,
Causal-JEPA, UWM-JEPA, DSGE are cited prominently and framed as boundary-mapping context, never
as a new principle.

*(No change to H0/H1a/H1b/H2, §4 ladder, §5 models, §6 metrics, §7 design, §8 launch plan.)*

### A2 (2026-09-07) — Stage-1 operationalization (mixing, dim, seeds) per Gate 0b evidence
Gate 0b (pipeline validation, `notes/gate0b_status.md`) established three instrument facts that
operationalize §8 Stage 1 without changing any hypothesis, metric, or factorial structure:
1. **linear g is degenerate for L0** — under linear mixing, R² ≈ 0.99 flat across the entire
   α-grid (nothing to discriminate), because an affine embedding satisfies alignment trivially.
   The theorem's own Sec 6.2 uses **nonlinear** mixing. Stage 1 therefore runs `g = nonlinear`
   (the discriminating regime). The linear arm remains in the full-factorial design for
   L1–L4 (where discrete structure still breaks linear recovery) and is tracked separately.
2. **dim 8** is the calibration winner for L0 (emb 8 ≈ obs 8): heavier dims (64) give the model
   spare Gaussian dims to hide in, diluting the SIGReg distributional penalty; dim 4 is too
   constrained. Stage 1 uses obs_dim = latent_dim = 8 for L0/L1; the {8, 32} dim factorial
   remains in Stage 2.
3. **seeds 0–2 (n=3)** for Stage 1 instead of the §8 "1 seed" fast-pass text: Gate 0b showed
   per-seed variance is material (peak wanders α∈{2,8} between seeds/configs), and the Stage-1
   gate requires a CI, not a point estimate. n=3 triples Stage-1 cost but keeps it an overnight
   batch; n=5 enters at Stage 2 per §7. **Stage-1 gate (restated):** L0 α-sweep (nonlinear g,
   dim 8, n=3) must show the qualitative Fig-4b shape — recovery peaked in the Gaussian region
   (α∈[1,4]) and degraded at BOTH heavy-tail (α≤0.5) and uniform (α≥16) extremes — with
   mean±CI consistent with a peak at α≈2, NOT monotone-in-α and NOT flat.
4. **Recon objective fixed for discrete worlds:** input-space MSE on one-hot/featurized
   observations is the natural decoder for categorical x; no change to L0's setup.
5. **Pure-discrete worlds (L2/L3/L4) report discrete + collapse readouts ONLY.** The pilot
   exposed an artifact: an all-zero continuous stand-in scores ridge R² = 1.0 (perfect fit of
   a constant), which is meaningless and would mislead the boundary map. Continuous readouts
   (ridge/CCA/MLP) are therefore emitted only for L0/L1 (where z is genuinely continuous);
   L2-L4 recovery is measured by per-dim linear-probe accuracy/AMI/purity + effective rank.
*(End A2.)*

### A3 (2026-09-07) — L0 gate re-operationalized: 2D spiral + λ=10 JEPA / temp=0.3 contrastive — GATE NOW PASSES
The A2 config (dim 8, quadratic mixing, λ=1.0) produced a
   monotone curve (uniform > Gaussian) that FAILED the restated gate. Root cause (isolated by
   direct experiment, not inferred): (a) the quadratic mixing map x_j = z_j + amp·z_{j+1}² has
   unbounded scale — for the small MLP encoder the map is effectively uninvertible, so the
   encoder never reaches the theorem's positive (linear-identifiable) regime at ANY α, and the
   residual nonlinearity (MLP-probe 0.93 vs ridge 0.82 at α=2) swamps the α-effect; (b) SIGReg
   at λ=1.0 is too weak to force the Gaussian-izing distortion at the uniform extreme. Fixes
   verified by full-α sweeps with n=3 seeds: **mixing = "spiral"** (bounded smooth map
   x_j = z_j + amp·sin(z_{j+1}), triangular unit-diagonal = globally invertible, scale-bounded),
   **latent_dim = emb_dim = 2** (the theorem's App H.7 setup), **sigreg_lambda = 10** for JEPA.
   Result: JEPA R² peaks at **α=2 (0.85 mean; seed 0: 0.94)**, monotone rise 0.04→0.85 through
   heavy-tail region, decay to ~0.50 at uniform — the Fig-4b signature reproduced. Contrastive
   (temp=0.3) also peaks sharply at α=2 (0.965, seeds 0.94-0.98) with a wider InfoNCE plateau,
   matching Fig 4b's "InfoNCE retains a wider plateau" note. Recon (input-space autoencoder) is
   NOT a Fig-4b objective (the paper's third objective is VICReg); it remains our discrete-world
   control and is excluded from the L0 gate. **Recon benchmark note:** autoencoder recovery is
   trivially high at every α because the decoder inverts input-space directly — expected, not a
   violation.
*(No change to H0/H1a/H1b/H2, §4 ladder, §5 models, §6 metrics, §7 design.)*


### A4 (2026-09-08) — L3/L4 world redesign (non-degenerate + no literal z exposure); entropy-matched H1b control
Stage-2 results exposed a **world-validity defect** that invalidates the Stage-2 L3 arm and blocks
L4 (all 15 cells errored). Measured latent entropies (n=20,000): **L3 joint = 1.00 bit vs 2.0 iid**
— the harmony rule as implemented sets `affix = stem` *deterministically*, so z has only 2 real
states — **and the observation x[:,0:4] literally contains stem/affix**. Any model that copies x
columns to h scores ~1.0 without learning structure; the Stage-2 L3 "H1b candidate" (jepa 0.975)
is therefore **not admissible evidence** for H1b. L4 additionally carries a constant latent column
(`caret_required = 1` for every word — zero-variance → per-dim probe crashes: 15/15 Stage-3 cells
failed) and x leaks caret_idx/class/flag directly.

**Fixes (worlds.py, tested):**
1. **L3 v2** — grammar with *genuine* joint entropy: stem vowel-class (front/back) × affix vowel
   (2-way) × affix *slot type* drawn from a rule table where the affix vowel is constrained by
   harmony *but not equal to* the stem class column; z = (stem_class, affix_vowel, slot) with
   H(z) > 2 bits and rule-governed dependence. Observation x is a **featurized surface**: stem and
   affix letters one-hot over the vowel/consonant alphabet + length + noise — z must be *inferred*
   through the rule, never read as literal columns. Existing tests asserting `affix==stem` and
   `x[:,4]≈0` are **replaced** (they codified the defect).
2. **L4 v2** — lexicon gains a **negative class** (words that must NOT take a caret, matched on
   length/vowel structure) so caret_required has real variance (P(caret)≈0.5); observation x drops
   the direct caret/class/flag columns and keeps only surface-feature columns (letters, length,
   vowel positions). Ground truth z = (stem_class, caret_required, caret_vowel) remains fully known.
3. **Entropy-matched H1b control (operationalizes the pre-registered "matched marginal statistics"
   phrase in H1b):** every L3/L4 cell gets a twin **L2m** (entropy-matched Markov) drawn with the
   same latent_dim and S chosen so that a single dimension's marginal entropy ≈ the L3/L4 world's
   per-dim entropy (S = round(2^H̄), H̄ = mean per-dim entropy). L2m differs from L2 only in S;
   same generator, same observation map. **H1b decision becomes:** R_L3 > R_L2m with 95% CI
   excluding 0 (Δ>0.2 as frozen), i.e., structure beats *entropy-matched* unstructured categories.
4. **Stage-2 L3 rows (15 cells) are withdrawn from evidence** (invalid world); re-run under L3 v2
   in a new stage (4). No H0/H1a/H2 claims touched — L1/L2/L4-v1 rows unaffected (L2 entropy
   verified healthy: S=5,d=8 joint 14.2 bits).
*(End A4.)*

### A5 (2026-09-08) — VICReg as the faithful fourth objective; R1 moment-matched twins; R4 decoupling index
1. **VICReg (models.py `vicreg`, Bardes et al. 2021) added as model #4.** A3 noted recon is an
   input-space autoencoder and "the paper's third objective is VICReg". VICReg is now implemented
   (variance-invariance-covariance on the same (x, x_prime) paired views; standard weights
   λ_inv=25, λ_var=25, λ_cov=1, no EMA, no predictor — collapse prevented by the variance hinge).
   Its role: **Fig-4b-trio fidelity on the L0 gate** and a second non-predictive contrast on the
   discrete ladder (VICReg regularizes the embedding toward whitened *but not Gaussian-izing*
   structure — variance hinge forces std≥1 per dim, covariance decorrelates; it has no explicit
   Gaussian target, so the theorem's mechanism does not predict a sharp α=2 peak for it).
   Decision rule (exploratory, no H-test): report VICReg's curve; the pre-registered H0/H1 tests
   remain defined on jepa/contrastive (predictive/alignment objectives) with recon as
   input-space control. No hyperparameter tuning of the 25/25/1 weights (standard values frozen).
2. **R1 moment-matched continuous twins (operationalizes the H0 "moment-matching" language):**
   each pure-discrete stage gets a continuous twin with matched first and second moments.
   Concretely for L2m (S states, one-hot marginal p over S): twin = i.i.d. Gaussian mixture with
   the same per-dim mean/variance and state-weighted components — i.e., z_cont ~ N(μ_s, σ²) with
   s ~ p and per-state μ_s chosen so the marginal mixture reproduces p's mean/variance per dim;
   observation map identical to L2 (embedding + noise, same dim). Purpose: separate *discreteness*
   from *moment statistics* — if L2m recovery ≈ twin recovery at matched moments, the failure is
   attributable to discreteness (H1a); if twins recover much better, the confound is distributional.
   Twins are compared on the **continuous ridge R² readout** (valid for continuous z_cont).
3. **R4 decoupling index (normalized, comparable across worlds):** F3 needs a scale-free
   prediction-vs-recovery contrast. Define, per (stage, model), the **within-stage z-scored gap**
   d_i = z(pred_loss_i) − z(recovery_i) over that stage's cells (prediction measured by pred_loss,
   recovery by the stage's primary recovery metric: ridge R² for L0/L1, acc_mean for L2-L4). A
   *decoupled* model shows cells where prediction is better-than-stage-average while recovery is
   worse-than-stage-average (large positive d). Reported as scatter + per-model mean d with
   bootstrap CI; labeled descriptive (no H-test).
*(End A5.)*