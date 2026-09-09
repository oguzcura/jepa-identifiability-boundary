# Mock rejection letter — pre-submission red-team (2026-09-09)

Per paper-planning counterintuitive rule #1: draft the most likely rejection
comments BEFORE submission and verify each is pre-empted. Written against the
post-audit draft (commit `6347186`). Each objection: where the paper already
answers it + residual action if any.

---

## Reviewer 1 — "Limited novelty: an empirical case study of an existing theorem"

**Objection.** No new theory, no new method beyond a measurement protocol;
the theorem (klindt2026lejepa) and the continuous extension (arya2026) already
exist.

**Pre-empted by.** Framing: the theorem *cannot* answer the discrete-regime
question by construction (no density, no score function — both proofs do not
apply). The contribution is the first known-ground-truth discrete ladder +
three falsifiable pre-registered answers, including two instrument-level
inventions (same-metric readout discipline; decoupling index) that generalize
beyond this paper. The novelty-verification sweep (2026-09-06) found no prior
work measuring recovery of known discrete generative latents.

**Residual.** None needed; cite the verification date in §1 (already there).

## Reviewer 2 — "Toy worlds: does any of this transfer to real language?"

**Objection.** L0–L4 are synthetic; L4 is a lexicon, not corpus-scale language
modeling. External validity unproven.

**Pre-empted by.** §Limitations explicitly bounds the claim ("up to S=10 and
3-bit grammars; beyond is untested"); L4 grounds the ladder in a real rule
system (Imla caret-rules) with complete ground truth; the decoupling index
gives a cheap instrument that practitioners can run on *real* embeddings
without ground truth. The paper's thesis is boundary-mapping, not SOTA.

**Residual.** Strengthen the closing "What is open" paragraph to name the
corpus-scale follow-up explicitly (action-conditioned objectives; stronger
grammars) — already present; optionally add one sentence proposing the
decoupling index as a diagnostic on production JEPA embeddings.

## Reviewer 3 — "H1b is below the pre-registered bar — why report it as a finding?"

**Objection.** +0.111 < frozen 0.2 bar; calling it "real" smells like null-
result inflation.

**Pre-empted by.** The paper never claims H1b passes: it is labeled *honest
null* everywhere (abstract, contributions, table). The power analysis shows
the bar was detectable (MDE 0.081), so the effect is genuinely sub-bar — and
the per-dim localization (affix dims only) makes the mechanism falsifiable.
This is the paper's core integrity argument.

**Residual.** None. Do NOT soften the "real, JEPA-specific" phrasing in the
abstract — it is immediately qualified by "below the pre-registered bar".

## Reviewer 4 — "The H1a S=10 d=8 cell looks like the opposite sign (0.456 vs 0.527)"

**Objection.** Descriptively, JEPA does *worse* on genuinely discrete latents
at S=10 — doesn't that contradict "no worse than the twin"?

**Pre-empted by.** Same-metric CI includes 0 (paired analysis, n=5); the null
verdict is about the pre-registered 0.3 bar, not the sign of point estimates.
The paper reports the cell openly.

**Residual.** Add one sentence in §H1a flagging this cell as the largest
descriptive (non-significant) counter-directional gap, so a reviewer finds it
acknowledged rather than discoverable. ← **ACTION (only text change from this
exercise).**

## Reviewer 5 — "Single author, AI-assisted, self-audited — credibility?"

**Objection.** No independent verification; LLM tooling in the loop; the
self-audit (A6) is self-congratulation.

**Pre-empted by.** Full artifact trail: frozen pre-registration with
append-only amendments committed *before* the cells they govern; every number
re-derivable from pinned raw JSONL via committed `repro/` scripts; 80-check
test suite; AI assistance disclosed per the ethics convention with the author
bearing responsibility; the three instrument failures are documented as
failures, not hidden. The Lean-4 caveat (theorem claims cited, not
re-verified) is stated explicitly.

**Residual.** Submitting the raw JSONL as a supplementary artifact (or a
data note on Zenodo) would let reviewers re-run everything without cloning +
regenerating — consider for the arXiv version. ← **OPTIONAL ACTION.**

---

## Verdict

No objection survives as a *fatal* flaw; R4's is the only one the paper does
not currently name in its own text → one-sentence fix applied in this pass.
R5's data-availability upgrade is the highest-leverage optional move before
submission.

## Pre-flight checklist (arXiv)

- [x] Title finalized: "Where Identifiability Breaks: A Discrete Ladder for
      Joint-Embedding Predictive Architectures"
- [x] Author line: real name + Independent Researcher + reachable email +
      github handle (no noreply)
- [x] \date de-drafted ("September 2026")
- [x] 0 LaTeX warnings; all figures/tables data-verified (pixel-level for F4)
- [x] Bib 17/17 live-verified (strict title+author)
- [x] Release asset refreshed on tag v0.1.0-p7-draft
- [ ] arXiv account + endorsement check (new-submitter endorsement may be
      required — needs Oğuz's account; license field: arXiv non-exclusive
      license, repo stays MIT/CC-BY-4.0)
- [ ] Optional: raw-JSONL supplementary upload (R5 residual)
