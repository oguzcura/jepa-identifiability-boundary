# Stage 2 (discrete core) — RESULTS + L4 blocker (2026-09-08, overnight)

**Data:** `results/sweep_stage2.jsonl` — 225/225 cells, 0 errors (L1 120 + L2 90 + L3 15).
**Config:** spiral mixing (A3), emb=dim, λ=10 JEPA, τ=0.3 contrastive, seeds 0-4.

## L1 discrete bridge — clean monotone collapse (re-confirmed, 5 seeds)
Linear-probe acc_mean, avg over seeds × dims {8,32}:
| K | jepa | recon | contrastive |
|---|---|---|---|
| 2 | 0.899 | 0.972 | 0.965 |
| 4 | 0.679 | 0.902 | 0.865 |
| 8 | 0.403 | 0.669 | 0.565 |
| 16 | 0.202 | 0.330 | 0.271 |

Discretization granularity K destroys recoverability monotonically; recon > contrastive > jepa ordering is stable.

## L2 Markov chain (S-dial) — the S-collapse is REAL and steep
acc_mean (avg seeds):
| dim=8 | S=3 | S=5 | S=10 |
|---|---|---|---|
| jepa | 0.855 | 0.716 | 0.456 |
| recon | 0.971 | 0.852 | 0.580 |
| contrastive | 0.884 | 0.711 | 0.539 |

| dim=32 | S=3 | S=5 | S=10 |
|---|---|---|---|
| jepa | 0.697 | 0.465 | 0.241 |
| recon | 0.932 | 0.784 | 0.447 |
| contrastive | 0.920 | 0.784 | 0.479 |

Larger state spaces (S↑) and wider latents (dim↑) both degrade recovery. JEPA degrades hardest.

## L3 harmony grammar — FIRST POTENTIAL H1b SIGNAL (needs R2 control!)
acc/AMI/purity (avg seeds):
| model | acc | AMI | purity |
|---|---|---|---|
| **jepa** | **0.975** | 0.697 | 0.924 |
| recon | 1.000 | 1.000 | 1.000 |
| contrastive | 0.625 | 0.001 | 0.508 |

**JEPA recovers the rule-governed grammar world at 0.975** — near recon ceiling, far above
contrastive. Compare L2 S=5 dim8 jepa = 0.716: grammar structure is RECOVERED where
unstructured Markov chains at similar state counts are NOT. This is the "structure buys
identifiability" (H1b) signature — **but it must survive the entropy/moment-matched control
(R2) before we claim it** (grammar may simply have fewer effective states or easier geometry).

## Stage 3 L4 — 15/15 ERRORED (deterministic; blocked pending design fix)
All cells: `ValueError: needs samples of at least 2 classes... only one class: np.int64(1)`.
Root cause (confirmed by reading worlds.py):
1. **`caret_required = 1` hardcoded for every word** (line 394) — z[:,1] is constant-1 by
   construction, so the per-dim linear probe on that column has a single class → crash.
   CARET_WORDS only contains caret-requiring words; the negative class doesn't exist.
2. **x leaks z** — x[:,2]=caret_idx, x[:,3]=cls, x[:,4]=caret_flag ARE the latents as literal
   columns; even if the crash were fixed, recovery would be ~ceiling for all models (worthless).

**Runner bug also fixed** (commit after this): error rows were added to the resume `done` set,
so re-running stage 3 after an L4 fix would silently skip the 15 failed cells and falsely
report "complete". Now only error-free rows count as done.

## Morning decisions (from companion plan §5.3 / §R)
1. **L4 redesign** (needs sign-off): add non-caret words (negative class), remove the z-leak
   from x so z is recoverable only through rule structure. Small worlds.py change + TDD.
2. **R1/R2 controls** before claiming H1b on L3: moment-matched continuous twin + entropy-matched
   L2 baseline. This is the make-or-break analysis for the "revolutionary" framing.
3. Amendment A4 before running any new cells.

## Verification
```
wc -l results/sweep_stage2.jsonl results/sweep_stage3.jsonl   # 225 / 15
git log --oneline -5
uv run pytest tests/ -q                                        # expect 49 passed
```
