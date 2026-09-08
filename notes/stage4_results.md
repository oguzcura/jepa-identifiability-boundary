# Stage 4 — L3 v2 vs L2m (entropy-matched) H1b test (2026-09-08)

Worlds redesigned per A4: L3 v2 = non-degenerate rule grammar (joint H = 3.0 bits,
no literal z in x), L2m = entropy-matched iid Markov (S = round(2^H̄) = 3, same obs
map). 30 cells = 2 arms × 3 models × 5 seeds, spiral-free scalar-per-dim obs, emb=6.

## Result (linear-probe acc_mean, paired by seed, n=5)

| model      | R_L3  | R_L2m | Δ      | per-seed Δ                        |
|------------|-------|-------|--------|-----------------------------------|
| **jepa**   | 0.918 | 0.805 | +0.113 | +0.167 +0.081 +0.130 +0.146 +0.044 |
| recon      | 0.808 | 0.878 | −0.070 | −0.015 −0.029 −0.085 −0.174 −0.047 |
| contrastive| 0.805 | 0.876 | −0.071 | −0.015 −0.118 −0.116 −0.090 −0.015 |

## Honest classification vs frozen rule (H1b: R_L3 − R_L2m > 0.2, 95% CI excl. 0)

- **JEPA: NOT PASSED at the frozen bar.** Δ = +0.113, 95% CI [0.057, 0.169]
  (se = 0.020, t_4), all five seeds positive. Directionally consistent but effect
  is roughly half the pre-registered 0.2 threshold. → **directional H1b evidence,
  no claim.** CI excludes 0 → the direction is real; magnitude falls short.
- **Recon / contrastive: negative** (−0.07 both, 4-5/5 seeds negative). Structure
  does NOT help non-predictive objectives at matched entropy; if anything the
  rule-constrained joint is *harder* for them. Consistent with "composition is a
  resource specifically for the predictive objective" — but H1b as frozen tests
  JEPA, and JEPA fails the 0.2 bar.

## Interpretation (exploratory)

1. The v1 L3 "candidate" (jepa 0.975, column-copying) is replaced by a *measured*
   +0.113 that survives the no-leak redesign — the direction was real, the v1
   magnitude was inflated by the leak.
2. Recon ≈ contrastive ≈ flat-to-negative Δ while JEPA is +0.11 → the effect is
   specific to the predictive/EMA objective, as H1b's mechanism story would want.
3. Do NOT rescale the bar post hoc. Report: "H1b not confirmed at the frozen
   effect size; directional evidence (CI excludes 0, 5/5 seeds positive)."

## Next
- Stage 5: L2t moment-matched twin S-dial (H1a discreteness control) + VICReg arms.
- L4 v2 stage-3 done (15/15, 0 errors): jepa acc 0.60-0.79 — report separately.
