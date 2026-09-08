# Audit: Why did we get honest nulls? — experiment vs own-side diagnosis

**Date:** 2026-09-08 (post-P7). **Scope:** H1a partial-pass re-examination, H1b null,
H2 null. **Method:** live re-analysis of raw JSONL + code reading (readout.py,
worlds.py, pre-reg + amendments A4/A5). Every number below was recomputed from
`results/sweep_stage{2,3,4,5}.jsonl` in this session, not copied from notes.

---

## TL;DR

Three honest nulls, three different causes — **and one of them (H1a's "partial
pass") was substantially a metric artifact on our own side, not a real effect.**

| Verdict | Actual cause | Our-side mistake? |
|---|---|---|
| **H1a PARTIAL (S=10 pass)** | Compared **acc vs ridge-R²** (different scales) | **YES — metric mixing (A5 §2 wording)** |
| **H1b null (Δ=+0.113 < 0.2)** | Real but small directional effect; bar calibrated on *v1-world intuition* | Partial — control confounded (state counts 8 vs 27) |
| **H2 null (purity gap ≈ 0)** | SIGReg-blur *is* visible in AMI at low S (recon−jepa +0.106/+0.131); purity can't see it | **YES — instrument too weak (purity)** |

The experiment (worlds, training, readouts, seeds) ran correctly — 0 errors, gate
passed, VICReg behaved exactly as A5 predicted. The nulls are NOT a broken rig. But
**two of the three decision rules used instruments that could not see the effect
they were designed to detect**, and the H1a rule mixed incompatible metrics. That
is the honest, uncomfortable answer to "was this our mistake": partly yes.

---

## 1. H1a — the "partial pass at S=10" is a metric artifact (OUR SIDE)

### What A5 froze (§2, line 286)
> "Twins are compared on the **continuous ridge R² readout** (valid for continuous z_cont)."

And the pre-reg H1a rule: *R_L2 below the moment-matched continuous prediction by
> 0.3 (absolute) with CI excluding that bound.*

### The problem
R_L2 is measured as **discrete linear-probe accuracy** (chance = 1/S; bounded [0,1]).
The twin's prediction was measured as **ridge R²** (chance = 0; unbounded below; on a
*continuous* target the linear probe gets credit for getting close, not just exact).
Comparing acc(S=10)=0.456 (chance 0.1) against ridge R²=0.943 is comparing a
10-way classification score to a continuous regression score — **different scales,
different chance floors, different ceilings.** The +0.487 "gap" that produced the
S=10 pass is largely this scale difference.

### The same-metric recomputation (Audit 1/7, raw rows)
Twin zK (S equiprobable bins of the continuous z_cont) is the *discrete readout of
the twin* — the apples-to-apples comparison. Paired per seed:

```
jepa        S=10 d= 8: same-metric gap = +0.071   (vs +0.487 with ridge-R²)
jepa        S=10 d=32: same-metric gap = +0.015   (vs +0.473)
jepa        S= 5 d=32: same-metric gap = −0.001   (vs +0.240)
contrastive S=10 d= 8: same-metric gap = +0.186   (vs +0.452)
```

**Under the same metric, NOTHING approaches the frozen 0.3 bar.** The largest
same-metric gap anywhere is contrastive S=10 d=8 at +0.186; jepa never exceeds
+0.071; at dim=32 jepa's gaps are −0.031/+0.015. Max across all 18 cells: 0.186.
95% CIs on the big ones: jepa S=10 d=8 gap CI = [0.049, 0.092] — excludes 0 but
**nowhere near 0.3**.

### What this means
The honest H1a verdict under a *same-metric, same-information* comparison is:
**H1a does NOT pass at S=10 — the discreteness "collapse" is real but is a
readout-quantization effect, and a moment-matched continuous latent quantized to
the same number of bins collapses almost identically.** Genuinely discrete z
recovers ≈ as well as binned continuous z at the same bin count and same metric.
The "partial pass" headline was an artifact of comparing accuracy to R².

**Fix:** Report H1a on the same-metric readout (zK acc for the twin) as the
primary; keep ridge-R² as a *secondary, labeled* comparison. Amend A5 §2 wording
in the paper's analysis section (or add A6) noting the metric-mixing pitfall and
re-scoring H1a as an honest null with a documented directional-but-sub-bar signal
at d=8. The L1 bridge (continuous R² flat ≈0.9 while zK acc collapses with K)
tells the same story consistently: the collapse is in the *discrete readout*, not
in the latent's discreteness per se.

---

## 2. H1b — the +0.113 is real but the control has a state-count confound

### Verified numbers (stage 4, 5 seeds, raw rows)
jepa per-seed deltas: +0.167, +0.081, +0.130, +0.146, +0.044 → mean **+0.113**
(sd 0.050, paired t₄ = 5.05, p = 0.0072, t CI [0.051, 0.176]; bootstrap CI
[0.072, 0.151] per h1b_test) — **5/5 seeds positive, CI excludes 0.** recon
−0.070, contrastive −0.071 (both negative).

### The effect is real and *specific* (Audit 8, per-dim)
```
Stage-4 per-dim acc (means over 5 seeds)
              stem(2way)  suf1(4way)  suf2(4way)   mean
jepa    L3    0.876       0.947       0.932        0.918
recon   L3    0.914       0.758       0.752        0.808
contras L3    0.929       0.735       0.751        0.805
              L2m: 3-way dims ×3                 mean
jepa    L2m   0.901       0.654       0.859        0.805
recon   L2m   0.996       0.665       0.972        0.878
contras L2m   0.984       0.680       0.963        0.876
```
All three models recover the *stem class* (surface-visible, 2-way) equally
(0.88–0.93). **JEPA's entire +0.113 advantage lives in the two rule-governed
affix dims** (0.947/0.932 vs recon/contrastive 0.735–0.758) — exactly where
composition/prediction should help. So H1b's directional signal is genuine,
mechanistically interpretable, and JEPA-specific. It is simply below the frozen
Δ>0.2 bar.

### BUT the control is not actually entropy-matched (Audit 4)
A4 chose L2m S = round(2^H̄) with H̄ = *mean per-dim* entropy. Verified:

```
L3 v2 : joint states = 8    joint H = 3.00 bits   per-dim H = [1.0, 2.0, 2.0]
L2m   : joint states = 27   joint H = 4.75 bits   per-dim H = [1.58,1.58,1.58]
       (S=3 homogeneous 3-way dims)
```

- L3's dims are **heterogeneous**: 2-way stem (1 bit) + two 4-way suffixes (2 bits
  each), with the harmony rule *coupling* them (joint 3 bits < sum 5 bits).
- L2m's dims are **homogeneous 3-way**, *independent* (joint 4.75 = sum).
- Mean *marginal* entropy matched (1.67 vs 1.58 — approx), but **joint state count
  differs 8 vs 27** and per-dim cardinality differs (2/4/4 vs 3/3/3). Per-dim
  chance also differs: L3 mean chance = (0.5+0.25+0.25)/3 = 0.333; L2m = 0.333 —
  equal by coincidence, but the *shape* differs (binary dim is easy, 4-way dims hard).

**Direction of bias:** a 27-state control is harder than an 8-state world *for
everyone*, so a naive reading says "L3 advantage partly = fewer states." But the
per-dim evidence cuts against that: the stem dim (equal difficulty for all) shows
**no** jepa advantage, and the 4-way dims (harder than 3-way) show the *biggest*
jepa advantage. State-count alone cannot explain the pattern; rule-structure can.
Still, the control is imperfect — the honest fix is a *joint-entropy-matched*
control (e.g., L2m with per-dim S ∈ {2,4,4} = same marginal structure, or 3 dims
× 2 states = 8 states) to fully rule out cardinality confounds.

### Why the bar was set where it was
The 0.2 bar predates the A4 redesign; it was frozen in the original pre-reg
(before any L3 data existed). The only L3 pilot number available when the ladder
was designed was v1's inflated 0.975 (later withdrawn as column-copying). An
effect of +0.113 with CI excluding 0 on a *clean* world is a real but modest
composition benefit — the bar was effectively calibrated on v1-world intuition.

### Bottom line for H1b
**Honest null stands at the frozen bar — but the paper must say: (a) the effect is
real (5/5 seeds, CI excl 0), (b) it is JEPA-specific and lives exactly in the
rule-governed dims, (c) the control was marginal-entropy-matched but not
joint-state-matched (8 vs 27 states), and (d) a joint-matched control is the
clean follow-up.** This turns "H1b null" from a shrug into a precise, testable
claim about effect size, not effect existence.

---

## 3. H2 — null because purity is too weak to see the blur that AMI detects

### Pre-reg rule
> cluster-purity gap between JEPA-core (SIGReg) and reconstruction > 0.1 with CI
> excluding 0, in L2/L3/L4.

### Verified (Audit 6 + 7, paired per seed, AMI cross-check)
```
AMI (recon − jepa), paired by seed:
  L2  S=3 d=3 (l2m):  +0.106  t=+4.68   ← blur visible
  L3  d=3:            +0.131  t=+1.39   ← blur visible (direction), n.s.
  L2  S=3 d=8:        +0.045  t=+11.7
  L4  d=3:            +0.045  t=+1.12
  everything else:    +0.001 … +0.019   (no blur)
Purity (recon − jepa): ±0.04 at most, frequently NEGATIVE (jepa slightly higher)
```

So the SIGReg-blur mechanism **does** show up — recon clusters are cleaner than
jepa's by up to +0.106/+0.131 AMI at the low-S cells where discrete structure is
recoverable at all — but the frozen instrument (KMeans **purity**) cannot see it:
purity saturates/behaves erratically on these h geometries (both models sit near
the KMeans floor at high S; purity even flips sign at d=8). The H2 null is
therefore **an instrument failure, not evidence that SIGReg doesn't blur**: at the
only cells where the effect could appear (low S, small d), the AMI direction
agrees with the hypothesis; purity just wasn't sensitive enough to register a
>0.1 gap.

### Fix
Report H2 on **AMI (or adjusted Rand), not purity**, with purity as secondary;
re-score H2 as *directional support at low S, honest null at high S / on purity*.
Optionally add a λ escalation arm (the H2 mechanism is λ-driven; λ=10 may be too
weak to measurably blur 3–10 well-separated states, which is itself an
interesting null).

---

## 4. What this means for the paper and the process

### Claims to correct in main.tex before release
1. **H1a "partial pass at S=10" → re-score on same-metric zK readout.** Headline
   becomes: discreteness collapse ≈ quantization collapse; no evidence of a
   >0.3 same-metric gap at any S. (Directional d=8 signal survives; the 0.487
   number does not.)
2. **H2 → AMI-based reporting**, purity demoted; directional support at low S.
3. **H1b stays an honest null at the frozen bar but is upgraded in prose** to a
   precise effect-size claim with the state-count caveat + joint-matched follow-up.

### Process lessons (this is where "are we still using our pipelines" lands)
- **The pipeline worked.** Pre-reg + append-only amendments + TDD + honest-null
  reporting caught the v1 degeneracy (A4), the one-class crash, the emb-dim bug,
  and — this audit — the H1a metric mixing. The discipline is why these are
  *correctable findings* instead of published artifacts.
- **The pipeline's blind spot:** we froze *instruments* (purity; acc-vs-R²) inside
  decision rules without validating that the instrument can see the effect it is
  supposed to detect. That is a **construct-validity** failure, not a
  statistical one — and it is exactly what the `statistical-power` /
  `experimental-design` / `research-artifact-verification` skills warn about
  (effect-size calibration + instrument validation before freezing bars).
- The 0.2/0.3/0.1 bars were set **before any pilot data on the relevant worlds**
  and never revisited (frozen = frozen). The correct response is not to move the
  bars but to *report effect sizes + CIs alongside the binary verdicts*, which the
  paper now does.

### Concrete next steps (offer, needs user sign-off)
1. **A6 amendment** (append-only): same-metric H1a scoring; AMI-based H2;
   joint-entropy-matched H1b control arm. Small compute: L2m heterogeneous
   (S per-dim 2/4/4) ≈ 15–30 cells ≈ 20–40 min on the 4070.
2. Paper re-score + prose updates per §4.1–4.3.
3. Add an "audit trail" appendix entry pointing at this file + the audit scripts.

**Scripts:** results/_audit1_samemetric.py, _audit2b_perdim.py, _audit4b_entropy.py,
_audit6_ami.py, _audit7_paired.py, _audit8_stage4clean.py (all re-runnable).
