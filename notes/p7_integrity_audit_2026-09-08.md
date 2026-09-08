# P7 Integrity audit — 2026-09-08 (commits 15fedff, d7a1b83, b629206, +figs)

Status: bibliography + figures + author line audited pre-release. Everything below
was re-run live, not re-read.

## 1. Bibliography (paper/custom.bib) — 3 real defects found & fixed

Verification protocol: every arXiv eprint hit via `export.arxiv.org/api/query?id_list=`
and compared against the bib's title **and** author list (strict full-string, not
prefix). Scripts: `repro/_bib_verify.py` (fast), `repro/_bib_strict.py` (strict,
author check). Final state: **17/17 arXiv entries OK (title + author)**; cite-key
audit 20 cited = 20 bib keys, 0 missing, 0 orphaned (`repro/_key_check.py`).

Defects found (all would have shipped in the PDF):
1. **interventionalCRL2022 (2209.11924)** — bib carried a *fabricated author list*
   (von Kügelgen, Besserve, …). Live: Kartik Ahuja, Divyat Mahajan, Yixin Wang,
   Yoshua Bengio (ICML 2023). Real-ID/wrong-metadata trap. Fixed.
2. **leworldmodel2024 (2403.06407)** — wrong ID entirely: 2403.06407 is a medical
   multimodal-LLM paper. The real LeWorldModel (Maes, Le Lidec, Scieur, LeCun,
   Balestriero) is **2603.19312**, published 2026-03. Entry renamed
   `leworldmodel2026`, title/authors/year corrected. Citations updated in main.tex.
3. **12 "Anonymous" author fields** — resolved to live authors (all real papers,
   e.g. mcjepa2026 = Yongchao Huang; discretejepa2025 = Baek, Lee, Hoang, Ren, Ahn;
   causaljepa2026 = Nam, Le Lidec, Maes, LeCun, Balestriero; …).
4. **2 paraphrased titles** (jepaparadox2026, phylatent2026) — replaced with exact
   live titles; related-work prose updated to match (jepaparadox: "geometry of
   linguistic alternatives").
5. **Internal process notes leaked into the rendered PDF references** — the bib
   `note` fields contained working notes ("NOT a scoop of our L2 measurement",
   "DIRECT HIT", "name territory is taken") that plainnat renders. Purged all
   non-publishable notes; kept only factual annotations (e.g. Lean-4 verification
   note on klindt2026lejepa). Verified gone from PDF text.

## 2. Author line — OK as-is

Real name (Oğuz Emre Cura), affiliation "Independent Researcher", reachable email
oguzemrecura@gmail.com, GitHub github.com/oguzcura. No noreply address. (Git commit
identity remains the noreply address — commits only, per rule.)

## 3. Figures — F4 was plotting the A6-refuted control; regenerated

`repro/figures.py` read all stage files; F4 filtered `arm=="l2m"` (homogeneous S=3,
27-state — the control A6 showed to be state-space mismatched). Fixed to read
**L2mH (2,4,4)** from stage 7 (marginal-matched control), label updated. Regenerated
all five figures (PYTHONPATH=src .venv/Scripts/python.exe). F4 now shows jepa
L3 0.918 vs L2mH 0.807 (Δ+0.111). F4 embedded in main.tex as Figure 2 (H1b section);
text cross-ref added. 10 pp total. F1/F2/F3/F5 unchanged semantics; F5 remains
L0-only geometry (descriptive, not the H2 verdict — paper text says so).

## 4. Repo hygiene — OK
- `.gitignore`: results/, .venv/, __pycache__, *.aux/*.out/*.log, paper/*.pdf. Raw
  sweep JSONL stay local by design (672 cells across 9 stage files).
- Tracked: 50 files incl. figures/*.png (5), repro/*.py (15), paper/main.tex+custom.bib,
  pre-reg + notes, src/, tests/.
- PDF is a build artifact (gitignored); tectonic rebuilds from main.tex.

## 5. Release-readiness notes (deferred items for GitHub prep)
- main.pdf is not tracked; decide whether to attach to release or build-only.
- README.md does not yet exist at repo root (repo currently code+docs heavy).
- `repro/_*.py` scripts are the audit scratch (kept intentionally: they are the
  re-runnable proof of the audit claims, matching the "rerun don't re-read" doctrine).
- Figure captions reference "LeJEPA" for the JepaCore+SIGReg model (paper convention).
