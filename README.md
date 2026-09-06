# jepa-identifiability-boundary

**The Discrete Boundary: Identifiability of Symbolic and Rule-Governed Worlds in Joint-Embedding Predictive Models.**

Empiricizes the Klindt–LeCun–Balestriero identifiability theorem (LeJEPA recovers true latent variables *iff* Gaussian, under stationary additive-noise dynamics; Lean-4-verified, arXiv:2605.26379) **beyond the continuous regime** — into the discrete, categorical, rule-governed latent worlds that language actually lives in.

**Why not the continuous sweep?** The theorem paper's own Section 6.2 already maps the continuous non-Gaussian spectrum (generalized-normal family, α→0 heavy-tailed → α=2 Gaussian → α→∞ uniform; recovery peaks at α=2, their Fig 4b). ARYA/PGSA (arXiv:2606.12471) extended it to real continuous physical systems. **The discrete/symbolic regime is unmapped:** discrete latents have no smooth density and no score function — the entire machinery of both proofs does not apply.

## The ladder (L0–L4)

| Lvl | World | Purpose |
|---|---|---|
| L0 | Gaussian OU + gennorm sweep | **Reproduce their Fig 4b** — pipeline sanity gate |
| L1 | Discretized Gaussian (K bins) | Bridge: is discreteness "very non-Gaussian" or a different failure? |
| L2 | Markov-chain latents | Does "stationary but discrete" get any guarantee? |
| L3 | Toy rule-governed grammar | Rule-structure with complete ground truth |
| L4 | Turkish orthography/morphology | The real case — Imla's rule set as ground truth |

## Hypotheses (pre-registered)

- **H0:** discreteness = extreme of the existing axis (moment-matching predicts the degradation).
- **H1a:** recovery breaks *worse* than continuous moment-matching predicts — qualitative collapse.
- **H1b:** compositional/rule-governed structure buys identifiability independent of the Gaussian mechanism (the headline if true).
- **H2 (insurance):** SIGReg's Gaussian-forcing measurably blurs genuinely discrete clusters.

## Status

- Phase 0 (environment): **DONE** — RTX 4070 8 GB verified, torch 2.6.0+cu124, $0 compute.
- Direction: v2 settled after live gap-verification (see `notes/research_direction_v2_discrete_ladder.md`).
- Plan: `.hermes/plans/2026-09-06_224832-jepa-identifiability-boundary.md` (v2).

## Reproduce

(TBD — repro/ scripts ship with the paper.)

## License

- Code: MIT
- Paper/results (when published): CC BY 4.0

Research conducted with disclosed AI-assisted workflows; every number in this repo re-derivable from pinned artifacts. All arXiv citations verified via `ti:`/id_list/abs-page protocol (free-text `all:` search is unreliable for exact titles).