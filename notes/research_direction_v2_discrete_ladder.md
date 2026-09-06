# Research Direction v2 — The Discrete/Symbolic Identifiability Ladder

**Date:** 2026-09-06 | **Status:** VERIFIED OPEN (gap survives live checking)
**Supersedes:** the continuous "boundary map" formulation of the 2026-09-06 plan (that exact experiment is already published — see §1)

---

## 1. Why the original "boundary map" is dead (verified live, 2026-09-06)

The continuous distance-from-Gaussian sweep we planned **already exists**:

| Claim | Verification |
|---|---|
| **Klindt, LeCun, Balestriero** arXiv:2605.26379, "When Does LeJEPA Learn a World Model?" | ✅ Real. Lean-4 verified (Mathlib, zero `sorry`; background lemmas assumed: Hermite, Mazur-Ulam). Theorem: LeJEPA linearly recovers world latents *(up to rotation)* **iff latents Gaussian under stationary OU transition**. |
| **Their Section 6.2 already runs our exact experiment** | ✅ Confirmed in full text: *"Latent-Distribution Sweep"* — generalized-normal family α→0 heavy-tailed, α=1 Laplace, **α=2 Gaussian**, α→∞ uniform; linear recovery **peaks at α=2 across all three objectives** (Fig 4b, App H.7). Plus DMC-Reacher RL pixel trajectories: R² collapses on real non-Gaussian data. |
| **ARYA Labs** arXiv:2606.12471, "Identifiability Without Gaussianity: Symbolic World Models…" | ✅ Real (2 weeks after). Extends to real continuous physical systems (quantum oscillator, phase transitions, turbulent flow; Lyapunov-exact predictability). Their "symbolic" = *causal-generator grounding* (PGSA architecture), **not** discrete latent testing — "discrete" appears once, "categorical" zero times. |

**Lesson:** a boundary map of *continuous* non-Gaussianity is done — by the theorem's own authors, with a figure. Our Phase-1 live check would have caught this; we now catch it before building.

---

## 2. The surviving gap (verified OPEN)

**Nobody has tested identifiability when the TRUE GENERATIVE latents are discrete/categorical/rule-governed** — in JEPA/predictive-SSL. Full-around searches (arXiv API, live) found:

| Neighbor | Why it is NOT our gap |
|---|---|
| "Discrete JEPA" (KAIST/NYU, arXiv:2506.14373) | **Quantizes the encoder output** into discrete tokens for reasoning/tokenization. Never asks whether recovery holds when the *generative* latents are discrete. |
| "Your Probabilistic JEPA Is Secretly a Hidden Markov Model" (arXiv:2608.13621, Aug 13 2026) — introduces **MCJEPA** | State-space **structural interpretation** of probabilistic JEPA (PIB-VJEPA). **Adjacent constructive work, NOT a straight scoop of L2**: MCJEPA *proposes a different architecture* (learned transition matrix instead of a latent predictor) to handle finite Markov states correctly — the same move ARYA made with PGSA (build an alternative that sidesteps the failure) rather than characterizing *how and where the standard LeJEPA/SIGReg objective breaks* on discrete latents, which is precisely what our L2 measures. **Cite + distinguish, don't treat as racing.** (Verification: id_list ✓, ti: exact-title search ✓, abs page ✓.) |
| "Probing the Latent World: Emergent Discrete Symbols" (arXiv:2603.20327, AIM) | Passive **quantization probe** to *inspect* continuous video encoders — interpretability, not recovery of known discrete generative latents. |
| "SJEPA" (arXiv:2608.04060), "RiJEPA" (arXiv:2603.13265) | **Symbolic dynamics induction / rule injection** — different goals (induce compact dynamics; inject logic), not measuring recovery of known discrete latents. |
| "No Gaussian Required" (arXiv:2608.17542, Aug 18 2026) | Replaces SIGReg Gaussian-prescription with **contrastive inverse dynamics** — strengthens our SIGReg-insurance sub-hypothesis (H2), does not test discrete recovery. (Verification: id_list ✓ exact ✓ — title search returned it on the first try when done as `ti:"No Gaussian Required"`; note: arXiv's free-text `all:` search can miss exact-title phrases → always use `ti:` or the abs page.) |
| Discrete-latent identifiability in *statistics/generative* models (Bayesian pyramids, discrete causal models, "Deep Discrete Encoders" arXiv:2501.01414) | The **mathematical tools exist** (algebraic identifiability, tensor methods) but for generative/probabilistic models — **not** predictive-SSL/JEPA. We can *borrow* the vocabulary. |

**Why the gap is a different regime, not a denser sample:** discrete latents have no smooth density and no score function — the entire machinery of both prior proofs (Sturm–Liouville spectral theory, Gaussian transport) does not apply. It cannot be reached by tilting α in a generalized normal.

---

## 3. The research question (v2)

> Does the JEPA identifiability guarantee — and its failure mode — extend to **discrete, categorical, rule-governed generative worlds** (the regime language lives in)?
> If it fails as theory suggests, **how exactly does it fail** — can a rule-verifiable probe (Imla-style) detect and characterize the failure precisely?

## 4. The world ladder (L0–L4)

| Level | World | Purpose | Knob |
|---|---|---|---|
| **L0** | Gaussian OU (reproduce their Fig 4a) | **Pipeline sanity — non-negotiable**: our numbers must match theirs before anything else | — |
| **L1** | Discretized Gaussian (bin into K categories) | Bridge: is discreteness "very non-Gaussian" or a different failure? | K ∈ {2,4,8,16,64} |
| **L2** | Markov-chain latents (categorical, stochastic transition matrix) | Discrete analog of OU; does "stationary but discrete" get any guarantee? | state count K |
| **L3** | Toy rule-governed grammar (stem+affix, vowel harmony, agreement — synthetic) | Structurally like real morphology, fully known ground truth | grammar complexity |
| **L4** | Turkish orthography/morphology rules (real case) | Ties the ladder to something that matters; Imla's rule-set as ground truth | TDK rule families |

All worlds: known ground truth z by construction, procedural (contamination-proof), val split fresh-seeded.

## 5. Metrics (adapted — a small methodological contribution in itself)

Their R² does not transfer to categorical variables. Our readout (frozen pre-registration):
1. **Linear-probe top-1 accuracy** — can a linear head recover the true category from the embedding (chance vs. ceiling)?
2. **Adjusted Mutual Information (AMI)** — geometry-agnostic cross-check (no lineage assumption).
3. **Cluster purity + silhouette** — for assessing partial structure.
4. **For L1 only:** keep R² on the underlying continuous z (bridges to their Fig 4) + accuracy on binned z.
5. **Imla-style rule-verification (L3/L4):** mechanical rule checker (P=1.00/R=1.00 discipline) — prediction-level and latent-level.

Designing the adapted metric = a real, citable contribution, not a footnote.

## 6. Hypotheses (frozen in pre-registration)

- **H0 (continuity):** discreteness is just an extreme on the existing axis — recovery degrades exactly as matching kurtosis/moments to their published continuous sweep predicts; no new failure mode.
- **H1 (qualitative difference):** recovery breaks in ways continuous moment-matching cannot predict —
  - H1a: **worse** — categorical structure collapses even where a moment-matched continuous analog still partially recovers; or
  - H1b: **better** — compositional/rule-governed states buy identifiability independent of the Gaussian mechanism (the headline branch if it happens: *composition itself is the resource*).
- **H2 (SIGReg insurance, reportable regardless):** SIGReg's Gaussian-forcing measurably blurs/merges genuinely discrete clusters — informs the live "anti-collapse without Gaussian prescription" debate (see "No Gaussian Required").

## 7. Compute (local, RTX 4070 8 GB — verified sufficient)

- Models ≤ ~500 MB params; fp16 optional. LeWorldModel-scale fits easily.
- Roster: 5 ladders × knob levels × {jepa (SIGReg on/off), recon, contrastive} × seeds ≥5 × horizons {1,4} — staged:
  - **Stage 1:** L0 (reproduce Fig 4a) + L1 quick pass (K sweep, 2 seeds) → gates the project, ~1 overnight.
  - **Stage 2:** L2 + L3 (full seeds) → 2–3 overnights.
  - **Stage 3:** L4 Turkish + L1 deep + ablations → 2–3 overnights.
- Total ≈ **60–120 GPU-h**, all local: $0 compute. CPU: worlds/readout/stats. RAM 16 GB fine. Disk 57 GB free.

## 8. Publication framing (interesting-to-read, not benchmark-boring)

- **Working title:** *"The Discrete Boundary: Identifiability of Symbolic and Rule-Governed Worlds in Joint-Embedding Predictive Models"*
- **Story arc:** (1) the field's guarantee is Gaussian-bound, provably; (2) the continuous extension is already mapped (reproduce + cite, one paragraph); (3) **we build the first ladder into the regime language actually lives in** — discreteness, categories, rules; (4) the decoupling result (predicting-vs-understanding) resurfaces sharper: *in discrete worlds, when does a model predict the future correctly while having learned the wrong state?*; (5) rule-verifiable probes (Imla methodology) detect the failure mode precisely; (6) implications: world-model eval below the boundary is blind; why JEPA-language struggles (JEPA-paradox paper); SIGReg trade-off.
- **Revolutionary branch (if H1b):** *compositional rule-structure itself buys identifiability without Gaussian machinery* → a new principle for representation learning in symbolic domains.
- **Safe branch regardless:** H2 + the ladder instrument + metric design + honest map = citable even if H0.

## 9. The race risk (fast-pass mandate)

Two papers landed within two weeks of the theorem in mid-2026; the **HMM-correspondence paper (Aug 2026)** is the live threat to L2. Mitigation: **L0+L1 fast-pass in week 1** (before polish), staged launches, and an early frozen pre-registration so a competitor landing later doesn't invalidate our claim of independent discovery.

## 10. Immediate next actions (pending approval)

1. **Phases 1/2 of the plan stay** (theory grounding, pre-reg, worlds) but the world list is replaced by the L0–L4 ladder (§2 of plan file).
2. **Write L0 first** — reproduce their Fig 4a gennorm sweep; gate: our R² peak at α=2 matches theirs (within CI).
3. Then L1 (K-sweep) — the fastest evidence on H0-vs-H1a-before-H1b.
4. Metric design documented in pre-registration (frozen) before L2.