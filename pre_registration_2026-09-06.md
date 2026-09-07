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
