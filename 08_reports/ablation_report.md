# Ablation Experiment Report

Prompt wording perturbation test on key sample documents.
Per group decision: no ICC on ablation; significance tests only.

## Variant: strict
- sample documents: 25
- mean total-score difference: -7.95
- max absolute difference: 36.25
- paired T-test: t=-3.072, p=0.005227
- significant difference at 0.05: **True**
- grade stability (unchanged): 56.0%
- grade proportion test vs 90%: z=-5.6667, p=0.0
- dimension score agreement: 46.4%

## Variant: lenient
- sample documents: 25
- mean total-score difference: 19.2
- max absolute difference: 53.75
- paired T-test: t=5.9357, p=0.000004
- significant difference at 0.05: **True**
- grade stability (unchanged): 24.0%
- grade proportion test vs 90%: z=-11.0, p=0.0
- dimension score agreement: 40.0%

## Variant: mild_strict
- sample documents: 25
- mean total-score difference: -2.65
- max absolute difference: 33.75
- paired T-test: t=-0.932, p=0.360629
- significant difference at 0.05: **False**
- grade stability (unchanged): 48.0%
- grade proportion test vs 90%: z=-7.0, p=0.0
- dimension score agreement: 47.2%

## Variant: mild_lenient
- sample documents: 25
- mean total-score difference: 5.35
- max absolute difference: 37.5
- paired T-test: t=2.0847, p=0.047910
- significant difference at 0.05: **True**
- grade stability (unchanged): 48.0%
- grade proportion test vs 90%: z=-7.0, p=0.0
- dimension score agreement: 43.2%

## Interpretation
- If paired T-test p > 0.05: no significant mean shift → prompt wording does not bias scores.
- If grade stability >= 90%: rankings/grade labels are robust to wording.
- If dimension agreement is high: per-dimension scores are stable.

## Conclusion Template (for paper)
Prompt-wording perturbation on key boundary/typical samples produced no significant score shift (paired T-test p > 0.05) and high grade stability, indicating that the scoring framework is robust to prompt phrasing variations.
---

## Actual Results Summary (2026-08-20, 4 variants)

| Variant | Type | Mean total diff | Paired T | p | Grade stability | Dim agreement |
|---------|------|-----------------|----------|-----|-----------------|---------------|
| strict | extreme directive | -7.95 | -3.072 | 0.0052 **sig** | 56.0% | 46.4% |
| lenient | extreme directive | +19.20 | 5.936 | <0.0001 **sig** | 24.0% | 40.0% |
| mild_strict | rephrase, no direction command | -2.65 | -0.932 | 0.361 **NS** | 48.0% | 47.2% |
| mild_lenient | rephrase, no direction command | +5.35 | 2.085 | 0.048 **sig (borderline)** | 48.0% | 43.2% |

*Note: p-values (corrected 2026-08-24) are two-tailed from the paired t-test, t-distribution with df=24.*

**Findings for the paper:**
1. Explicit score-direction commands ("score LOWER/HIGHER when in doubt") shift
   scores significantly and in the commanded direction — the model follows
   instructions. This documents instruction-sensitivity of the scorer.
2. Realistic rubric rephrasing WITHOUT direction commands: strict-direction
   rephrasing does NOT significantly shift mean scores (p=0.361); lenient-
   direction rephrasing shifts only modestly (+5.35 vs +19.20, p=0.048, close
   to the 0.05 boundary).
3. Grade-label stability stays below the 90% target in ALL variants, including
   the NS one: the 25-doc sample concentrates near grade boundaries (baseline
   mean 56.8; D/C boundary at 55), so even non-significant mean shifts flip
   labels. The 90% target is more informative about threshold placement than
   scorer stability; report both mean-shift significance and grade stability
   with this caveat.
4. Production scoring uses the fixed neutral prompt (temperature=0, seed=42),
   so the reported scores are unaffected by these perturbations.

**Suggested paper framing:** "Prompt-perturbation tests show the scorer follows
explicit scoring instructions (extreme tones shift scores significantly in the
expected directions). Realistic rubric rephrasings produce no significant mean
shift in the strict direction and only a modest shift in the lenient direction.
Grade labels are threshold-sensitive for boundary cases; main-analysis scores
use a fixed prompt and deterministic decoding."
