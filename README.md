# jepa-identifiability-boundary

**Empirical boundary map of latent identifiability in JEPA-style models.**

Empiricizes the Klindt–LeCun–Balestriero identifiability theorem (JEPA recovers
true latent variables *iff* they are Gaussian, under stationary additive-noise
dynamics; Lean-4-verified, arXiv:2605.26379) using procedurally generated worlds
with **known ground-truth latents**, swept from Gaussian → heavy-tailed →
discrete → symbolic dynamics.

Three self-supervised objectives — JEPA-core (latent-space predictor with EMA
target), reconstruction (MAE-style), contrastive (InfoNCE) — are trained on each
world; identifiability is read out mechanically (ridge R² / CCA / MLP-probe of
the known latent z from the learned embedding h).

**Core hypothesis (decoupling):** a model can *predict well while failing to
understand* — low prediction error with collapsed identifiability — below the
Gaussian boundary.

## Status

Phase 0 (environment). See `.hermes/plans/2026-09-06_224832-jepa-identifiability-boundary.md`.

## Reproduce

(TBD — repro/ scripts ship with the paper.)

## License

- Code: MIT
- Paper/results (when published): CC BY 4.0

Research conducted with disclosed AI-assisted workflows; every number in this
repo re-derivable from pinned artifacts.
