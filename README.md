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

Plus L2t (continuous AR(1) twin of L2) and L2m/L2mH/L2mS2 (statistics-matched unstructured controls, A4/A6).

## Hypotheses (pre-registered, frozen 2026-09-06)

- **H0:** discreteness = extreme of the existing axis (moment-matching predicts the degradation). — *PASS (gate reproduced Fig 4b).*
- **H1a:** recovery breaks *worse* than continuous moment-matching predicts — qualitative collapse. — *HONEST NULL* under the A6 same-metric re-scoring: discreteness adds little beyond readout quantization (max gap 0.186 < 0.3 bar).
- **H1b:** compositional/rule-governed structure buys identifiability independent of the Gaussian mechanism. — *HONEST NULL but robust:* LeJEPA gains **+0.111** (CI [+0.043, +0.179], 5/5 seeds) over the marginal-matched L2mH (2,4,4) control — real, directional, JEPA-specific, below the frozen 0.2 bar. The whole advantage lives in rule-governed affix dimensions.
- **H2 (insurance):** SIGReg's Gaussian-forcing measurably blurs genuinely discrete clusters. — *HONEST NULL* under AMI (directional recon > jepa in 9/9 cells, one clears 0.1).

## Results (live-verified, 2026-09-08)

- **672 cells / 0 errors** across 9 stage files (pilot–stage 7), local RTX 4070, $0 compute.
- Full audit of the instrument nulls (own-side metric mixing, control mismatch, H2 instrument blindness) in `notes/audit_null_diagnosis_2026-09-08.md`; corrections frozen as Amendment A6 before any control cell ran (`pre_registration_2026-09-06.md`, append-only A1–A6).
- Decoupling index: LeJEPA +1.15/+1.47/+1.44 stage-z — "predicting without understanding" on known ground truth.
- TDD suite: 80 passed. Deterministic seeds; raw sweep JSONL kept local (`.gitignore`d); all committed analyses re-runnable from `repro/`.

## Reproduce

```bash
uv sync                # or: uv pip install -e .
uv run pytest tests/ -q        # 80 passed
uv run python repro/figures.py # regenerate figures/ from results/*.jsonl (needs local data)
```

Pre-registration (frozen + amendments A1–A6), per-stage notes, and the audit trail live in the repo. Paper draft: `paper/main.tex` (compiles with tectonic), references live-verified via the arXiv `id_list` protocol.

## License

- Code: MIT (see LICENSE)
- Paper/results (when published): CC BY 4.0

Research conducted with disclosed AI-assisted workflows; every number in this repo is re-derivable from pinned artifacts. All arXiv citations verified via `ti:`/id_list/abs-page protocol (free-text `all:` search is unreliable for exact titles).
