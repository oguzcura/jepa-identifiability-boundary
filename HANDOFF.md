# HANDOFF — jepa-identifiability-boundary

**Durable cross-session handoff.** Last updated: 2026-09-08 (post-P7, post-integrity-audit, post-GitHub-release).
Repo: `C:/Users/oguzc/research/jepa-identifiability-boundary` · GitHub: https://github.com/oguzcura/jepa-identifiability-boundary (public)
Branch `main` @ `23e0e91` (42 commits). Paper release asset: https://github.com/oguzcura/jepa-identifiability-boundary/releases/download/v0.1.0-p7-draft/main.pdf

---

## 1. What this project is (one paragraph)

A PhD-grade, globally-relevant research program that **empiricizes the Klindt–LeCun–Balestriero JEPA identifiability theorem** (arXiv:2605.26379 — LeJEPA linearly recovers latents *iff* Gaussian under stationary OU dynamics, Lean-4-verified) **beyond the continuous regime**, into the **discrete/symbolic regime the theorem's machinery cannot reach** (discrete latents have no smooth density, no score function). We build an **L0→L4 world ladder** (Gaussian OU gennorm sweep → discretized Gaussian → Markov chain → toy harmony grammar → Turkish Imla orthography rules), train **four objectives** (JepaCore+SIGReg, Recon, Contrastive/InfoNCE, VICReg) on procedurally generated worlds with **known ground-truth latents**, and measure *how and where* latent identifiability breaks. The experiment ran on a local RTX 4070 8 GB at **$0 compute**. Deliverable: a full-length paper draft (10 pp, tectonic) + pre-registered protocol (frozen, append-only A1–A6) + public GitHub repo with full audit trail.

**Current phase: POST-P7** — experiment complete (672 cells, 0 errors), audit done, A6 corrections done, paper drafted to post-A6 truth, integrity audit done, GitHub live. **Next: arXiv submission prep + optional follow-up experiments.**

---

## 2. The scientific result (what we found — post-A6 truth)

| Hypothesis | Claim | Verdict (frozen bar) | Evidence |
|---|---|---|---|
| **H0** | Discreteness = extreme of continuous axis | **PASS** (A3 gate) | L0 α-sweep reproduces theorem Fig 4b: JEPA ridge R² peaks at α=2 (0.851, seed0 0.94); contrastive 0.965 |
| **H1a** | Discreteness breaks recovery *worse* than continuous moment-matching | **HONEST NULL** (bar 0.3) | Same-metric (zK acc) max gap 0.186 across 18 cells; jepa S=10 d8 +0.071 CI [0.049,0.092]; discreteness ≈ readout quantization, not a distinct failure |
| **H1b** | Rule/compositional structure buys identifiability | **HONEST NULL but robust** (bar 0.2) | jepa Δ = **+0.111** CI [+0.043, +0.179] vs marginal-matched L2mH (2,4,4), 5/5 seeds positive; effect lives entirely in rule-governed affix dims; recon −0.115, contrastive −0.103 (structure helps ONLY the predictive objective) |
| **H2** | SIGReg Gaussian-forcing blurs discrete clusters | **HONEST NULL** (bar 0.1) | AMI recon−jepa positive in 8/9 cells but ~0.005–0.05; only L2m S=3 clears 0.1 (+0.106). Directional mechanism support only |
| VICReg (A5, descriptive) | No α=2 peak — whitens, doesn't Gaussianize | Confirmed | L0 plateau 0.987@α=2 → 0.975@α=32; discrete ladder mirrors recon/contrastive (L3−L2m = −0.167) |
| Decoupling index (R4) | JEPA "predicts without understanding" | Confirmed | +1.15 (L0 gate) / +1.47 (L4) / +1.44 (L2t) stage-z; contrastive mirrors negative |

**Claimable headline:** *Genuinely discrete generative latents are recovered about as well as equiprobable bins of a moment-matched continuous twin (H1a null) — but rule-governed composition gives a small, real, JEPA-specific identifiability advantage (+0.111) that lives exactly in the rule-constrained dimensions (H1b), while VICReg whitens without Gaussianizing.* The honest-null framing is a feature: sub-bar directional effects on a pre-registered, powered design are the honest finding.

**The audit story (why the bars were frozen and stayed):** three initial "honest nulls" were diagnosed as **own-side instrument failures**, not experiment failures — (1) H1a's "partial pass" was acc-vs-R² metric mixing (accuracy chance 1/S vs ridge R² chance 0), (2) H1b's control (L2m S=3, 27 states) was state-space mismatched to L3 (8 states), (3) H2's purity instrument cannot see a >0.1 blur on L2–L4 geometry (KMeans floor) while AMI can (+0.106/+0.131). Amendment A6 corrected all three *before* running the control arm. **The power audit proved the bars were never underpowered**: MDE @80% power (n=5, sd=0.05, α=0.05) = 0.081 → frozen 0.2 bar had ≈100% power; observed +0.113 had 99% power. The nulls are real sub-bar effects, not noise.

---

## 3. Repository map (tracked: 52 files)

```
├── pre_registration_2026-09-06.md   FROZEN protocol + append-only amendments A1–A6 (the contract)
├── README.md                        Post-A6 overview (public-facing)
├── LICENSE                          MIT
├── pyproject.toml                   uv project; NO console scripts (pipeline = repro/*.py scripts)
├── src/jepa_id/                     Library: worlds.py, models.py, train.py, readout.py, analyze.py
├── repro/                           Scripts (the runnable pipeline + all re-runnable analyses)
│   ├── run_sweep.py                 Grid runner, stages pilot/1/1b/2/3/4/5/6/7, resumable, TDD-gated
│   ├── analyze_report.py            H1a/H1b/H2 decision-rule report with frozen bars
│   ├── figures.py                   F1–F5 (F4 regenerated on A6-correct L2mH control)
│   ├── a6_h1a_rescore.py, a6_h1b_retest.py, a6_h2_rescore.py   A6 re-scoring analyses
│   ├── paper_numbers.py             Live-verified headline numbers for the paper (rerun, don't re-read)
│   └── _bib_verify.py, _bib_strict.py, _key_check.py, _ref_check.py, _l1_check.py,
│       _vicreg_check.py, _ent_verify.py, _decoup_paper.py      integrity-audit tools (proof-of-audit)
├── tests/                           80 tests (worlds, models, analyze, A6 L2mH control)
├── figures/                         F1_l0_fig4b.png, F2_boundary_map.png, F3_decoupling.png,
│                                   F4_l3_vs_l2m.png (L2mH), F5_sigreg_blur.png
├── paper/                           main.tex (10 pp full draft) + custom.bib (20 entries, all verified)
├── notes/                           9 markdown docs — full audit trail (see §5)
├── evidence/_withdrawn/             Stage-2 L3 v1 world (invalid, withdrawn under A4)
└── results/                         GITIGNORED raw sweep JSONL (672 cells × 9 stage files) — local only
```

**Key invariants:**
- `results/` is gitignored by design (raw data stays local). Committed analyses in `repro/` + `notes/` are the reproducibility layer. **A fresh clone cannot rerun analyses without the local JSONL** — flagged as a release consideration (§9).
- Author/identity: commits use `oguzcura <83915035+oguzcura@users.noreply.github.com>` (commits only, allowed); **paper author line is real name Oğuz Emre Cura, Independent Researcher, oguzemrecura@gmail.com, github.com/oguzcura** — never a noreply address on the paper.

---

## 4. Pipeline & tools actually used (the "how")

**Research pipeline (Hermes agent skills — all actively applied):**
- `academic-pipeline` (orchestrator) — staged P0→P7: research → write → integrity → review flow
- `research-artifact-verification` — "numbers in notes are claims too: rerun, don't re-read" (drove the whole audit)
- `experiment-craft` / `experimental-design` — world-validity audits, control construction
- `statistical-power` / `statistical-analysis` — MDE power audit, paired stats, Wilson CIs, Holm correction, effect sizes, APA reporting
- `research-bib-verification` — live arXiv ID verification protocol (`id_list=` + title compare, never free-text `all:`), strict title+author variant
- `research-novelty-verification`, `systematic-literature-review`, `academic-outreach-emails` (used in earlier phases)
- `statistical-analysis`/reporting_standards for paper numbers

**Engineering stack:**
- **Pre-registration discipline**: protocol frozen 2026-09-06 (self-hash pinned in file), amendments **A1–A6 append-only, each committed BEFORE its cells ran** (A6 committed `af6c002` before any stage-7 code — the discipline held under audit pressure)
- **TDD gates**: 80 tests green (49→53→58→63→75→80 progression); tests added before new world/cell types
- **Deterministic seeds, pinned artifacts, resumable grid runner** (resume skips only rows WITH results; error rows never count as done)
- **Stats rigor**: Holm correction across cells, Wilson CIs, paired per-seed tests, bootstrap + t CIs, honest directional claims vs frozen bars
- **Zero-cost mode**: local RTX 4070 8 GB, torch 2.6.0+cu124; arXiv API/OpenAlex/web_extract for verification only
- **Build**: tectonic (single-binary LaTeX), `pypdf` for PDF verification

**Environment gotchas (for future sessions):**
- Windows host, bash (git-bash/MSYS) shell in terminal tool — POSIX syntax; native tools need `C:/...` paths
- GPU env vars do NOT persist across terminal calls — re-export inline: `PYTHONPATH=src .venv/Scripts/python.exe repro/figures.py`
- matplotlib/numpy live in `.venv` (not system python): use `.venv/Scripts/python.exe`
- Run tests: `.venv/Scripts/python.exe -m pytest tests/ -q` (or `uv run pytest`)

---

## 5. Documentation inventory (notes/ = the audit trail, all public)

| File | Content |
|---|---|
| `notes/research_direction_v2_discrete_ladder.md` | Direction v2 rationale (why discrete is unmapped; live gap-verified against 2606.12471) |
| `notes/theorem_notes.md` | Theorem notes (KLB 2605.26379, §6.2, Fig 4b) |
| `notes/gate0b_status.md` | Gate 0b honest status (peak at 8 not 2, calibration open — later resolved by A3 spiral fix) |
| `notes/stage2_results.md` | Stage 2 (L1/L2 collapse confirmed; L3 v1 H1b candidate flagged) |
| `notes/stage4_results.md` | Stage 4 (H1b +0.113 honest null) |
| `notes/analysis_2026-09-08.md` | Pre-A6 decision summary (H0 PASS, H1a PARTIAL, H1b/H2 null) — superseded by A6 re-scoring |
| `notes/audit_null_diagnosis_2026-09-08.md` | **The audit** — full honest-null diagnosis (3 own-side instrument failures, power audit, per-dim tables, process lesson) |
| `notes/a6_results_2026-09-08.md` | A6 re-scored results (H1a null, H1b +0.111 vs L2mH, H2 AMI directional) |
| `notes/p7_integrity_audit_2026-09-08.md` | P7 integrity audit close-out (bib defects, author line, figures policy) |

Pre-registration (`pre_registration_2026-09-06.md`) contains hypotheses, decision rules with frozen bars (H1a 0.3, H1b 0.2, H2 0.1), world specs, grid cells, seeds, and the **append-only amendments A1–A6** — the single most important document for anyone evaluating the claims.

---

## 6. Paper documentation

- **Draft**: `paper/main.tex` → `paper/main.pdf` (10 pp, tectonic), release asset `v0.1.0-p7-draft` on GitHub
- **Structure**: Abstract (post-A6 verdicts) → Intro (theorem boundary, contributions) → Related work → Protocol/amendments → Worlds → Objectives/readouts → Results (H0 gate Fig 1, L1 bridge, H1a table, H1b Table + Fig 2 = A6-corrected F4, L4/VICReg, H2 AMI, decoupling) → Power analysis → Discussion/limitations → Reproducibility/ethics → References
- **Bibliography**: 20 entries, **17 arXiv entries strict-verified live** (title + author) on 2026-09-08; 20 cited = 20 bib keys, 0 orphaned. Fixed during integrity audit: fabricated author list (interventionalCRL2022 → Ahuja/Mahajan/Wang/Bengio ICML 2023), wrong ID (leworldmodel 2403.06407 → 2603.19312, renamed `leworldmodel2026`), 12 Anonymous→real authors, 2 paraphrased titles → exact
- **Figures**: F1 (L0 gate, embedded), F2 (boundary map), F3 (decoupling), **F4 (H1b on L2mH — regenerated post-A6; embedded as Fig 2)**, F5 (L0 blur geometry, descriptive only)
- **Not yet done for submission**: abstract/title finalization for arXiv, `\date`/working-draft markers cleanup, author-block final confirmation, decide figure set for camera-ready

---

## 7. GitHub state (live, public)

- https://github.com/oguzcura/jepa-identifiability-boundary — public, MIT, default branch `main`, pushed_at 2026-09-08
- Verified via API post-push: visibility=public, license=MIT, HEAD `23e0e91` matches local, working tree clean
- Release **v0.1.0-p7-draft** with `main.pdf` asset (download URL stable)
- gh CLI authenticated as `oguzcura` (full scopes) — future pushes/releases work without re-auth

---

## 8. Why this wins (the defensible framing)

1. **It fills a real, live-verified gap.** The theorem's own §6.2 maps the continuous non-Gaussian spectrum; ARYA (2606.12471) covers continuous physical systems. The discrete/symbolic regime — where language actually lives — has no smooth density and no score function: *the entire machinery of both proofs does not apply.* We mapped it.
2. **The honest-null result IS the contribution.** A pre-registered, powered (MDE 0.081), frozen-bar design that returns clean sub-bar directional effects — and publicly documents the *three instrument failures on our own side* that the audit uncovered — is more credible than a padded positive. The self-audit (A6) is a worked example of the pre-registration machinery catching its own errors, which we explicitly frame as a methodological contribution.
3. **The positive thread is real and mechanism-located.** H1b's +0.111 survives the corrected control (5/5 seeds, CI excluding 0, ~unchanged from the flawed-control estimate — the state-count confound worry is refuted, not just patched), and per-dim readouts show it lives *exactly* in the rule-governed affix dimensions. That is a falsifiable, interpretable claim about where composition helps predictive objectives.
4. **Engineering integrity is the differentiator.** 672 cells/0 errors; TDD gates; append-only amendments committed *before* code; live-verified citations (the audit caught a fabricated author list and a wrong ID that would have shipped); deterministic seeds; $0 cost; full audit trail public.
5. **Reproducibility by construction.** Every number in the paper is re-derived from pinned artifacts via `repro/*.py`; the audit scripts that proved the claims are committed alongside them ("rerun, don't re-read" as a working rule).

---

## 9. Open items & next steps (for the next session)

**Immediate (arXiv submission prep):**
1. Abstract/title finalization for arXiv submission (title candidate: "Where Identifiability Breaks: A Discrete Ladder for Joint-Embedding Predictive Architectures")
2. Author block + `\date`/draft-marker cleanup on the paper
3. Decide: commit `main.pdf` to repo or keep gitignore + release assets only (currently release-asset-only)
4. Optional: README "Reproduce" section notes that raw JSONL is local-only (clone can't rerun analyses) — add a data-availability note or ship a small sample/derived tables

**Scientific follow-ups (need new amendments + user sign-off before any cells):**
5. H1b escalation: stronger grammars / deeper rule hierarchies, effect-size bars sized from the +0.111 estimate
6. H2 mechanism study at matched entropy (L2m S=3 was the only cell clearing 0.1 under AMI)
7. Decoupling index under action-conditioned objectives

**Standing constraints (never violate):**
- Zero-cost mode unless user signs off on paid paths
- Frozen bars do NOT move; any new decision rule/control/objective needs its own append-only amendment BEFORE cells run
- arXiv ID verification protocol: `id_list=` + `ti:` + abs page only, never free-text `all:`
- Author line on any publication: real name + Independent Researcher + oguzemrecura@gmail.com + github.com/oguzcura
- Honest nulls are first-class results; `[UNVERIFIED]` markers where applicable
- (Credentials note: NVIDIA NIM key was exposed in an early chat — treat as rotated/never reuse; not stored in this repo)

## 10. Post-P7 verification audit (2026-09-09)

Full re-verification of the paper against raw data ("rerun, don't re-read").
**Verdict: core methodology and statistics sound; every headline number
reproduces from raw JSONL.** Five defects found and fixed in place:

1. **H2 direction count:** paper/README/notes said "8/9 arms" — raw shows
   recon>jepa AMI in **9/9** arms. Corrected in main.tex, README, and
   a6_results notes (verdict unchanged: honest null, only L2m clears the bar).
2. **InfoNCE gate wording:** "peaks at the same shape (0.965 at α=2)" was
   false — the sweep max is 0.983 at α=32. Rewritten as a broad plateau with
   no sharp peak; the α=2 maximum is SIGReg-specific (sharpens the theorem
   story rather than weakening it).
3. **Reproducibility section** claimed "all results JSONL committed" — false
   (results/ gitignored by design). Rewritten: retained locally, analyses
   re-run from repro/.
4. **Decoupling prose** called the stages "clean single-world" — stage 1b
   pools L0+L1 (54+24 cells; both read out by ridge R², so the pooled
   standardization is metric-consistent). Prose corrected; the frozen +1.15
   value is unchanged (an L0-only re-filter gives +0.97 — do not confuse).
5. **repro tooling:** paper_numbers.py carried an inline decoupling copy with
   an inverted pred sign and a literal 'ridge.mean' key (contradicted the
   paper); replaced with the committed jepa_id.analyze.decoupling_index.
   a6_h1a_rescore.py's secondary printout printed seed-0 only (read as a
   mismatch vs the paper's seed-means 0.456/0.943); now prints 5-seed means.

Verified clean: 672 cells / 0 errors; H1a table cell-by-cell; H1b +0.111
CI [+0.043,+0.179] with stage-7↔stage-4 pairing exact (λ, τ, emb, dim, g
identical per model); L1 0.899→0.202 acc with 0.738 flat ridge; VICReg
plateau 0.975–0.987; power MDE 0.082≈0.081 and CI ±0.062; bib 17/17 strict
live + independent arXiv spot-check; 80/80 TDD; GitHub synced + release
asset live; author block correct.

Residual (low severity, judgment calls — not fixed):
- paper_numbers.py per-dim print is seed-0-only (cosmetic; paper numbers
  independently verified from seed means).
- "first discrete ladder" is a literature priority claim — supportable but
  reviewer-arguable.
- H1a S=10 d=8: jepa is descriptively *worse* on genuinely discrete latents
  (0.456 vs 0.527 same-metric), though not significantly — the null framing
  is correct, but a reviewer may probe this cell.

## 11. Professionalism/polish pass (2026-09-09, user-directed)

- Removed raw-filename leak from §3: protocol now cited as "the protocol"
  with a footnote URL to the repo (file name kept only inside the footnote).
- hyperref configured colorlinks (navy) — was red-box default (amateur tell).
- "(hash-pinned)" → "SHA-256 hash recorded in the repository".
- Reproducibility: "TDD suite green" → "automated test suite (80 checks)";
  deleted internal "Author line per author-profile rules" sentence.
- VICReg "(vicreg, A5)" → "(vicreg; introduced in amendment A5)".
- F4 figure: legend moved fully outside (below axes), title de-coded
  ("F4: H1b —" prefix dropped), pairs given a gap (±0.19 offset, width 0.34),
  axis label cleaned.
- **F4 DATA bug (found via pixel-scan during the layout check): the bar loop
  unpacked offsets with enumerate(...), so the offset index shadowed the
  intended ±positions AND the series-selection string test could never match
  — both bars plotted the CONTROL arm (L2mH) values, one shifted left as
  "blue". The figure therefore contradicted Table 3 (it showed L3 > control
  for recon, not LeJEPA). Fixed (explicit (off, key, col, lab) tuples);
  pixel-verified: blue > orange only for LeJEPA, matching the table.
- All float specifiers [h]→[ht]; build now 0 LaTeX warnings (tectonic).
- Release asset main.pdf re-uploaded post-fix (same tag, --clobber).

**Quick resume commands:**
```bash
cd /c/Users/oguzc/research/jepa-identifiability-boundary
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q     # 80 passed
PYTHONPATH=src .venv/Scripts/python.exe repro/figures.py        # regenerate figures (needs local results/)
cd paper && /c/Users/oguzc/bin/tectonic main.tex                # rebuild PDF
git log --oneline | head -42                                    # full history
```
