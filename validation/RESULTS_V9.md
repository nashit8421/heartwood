# V9 results — the statics finally add, and the model fails to combine

Apnea-ECG, 17,023 one-minute ECG segments, 35 subjects, subject-disjoint splits,
3 seeds, n=1000. Headline ROC-AUC, fixed in `VALIDATION_V9.md` §3.

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| **D** — heartwood + statics | 0.702 | 0.899 | 0.879 | **0.827** |
| **C** — heartwood, no statics | 0.666 | 0.880 | 0.875 | 0.807 |
| **S** — `static_only` (age, sex, height, weight, BMI) | 0.744 | 0.893 | 0.867 | **0.835** |
| `agg` aggregate workaround | 0.739 | 0.828 | 0.872 | 0.813 |
| best `aeon` MiniROCKET | 0.716 | 0.887 | 0.839 | 0.814 |

## Verdicts

**Precondition — met, emphatically.** `static_only` reaches 0.835, **+33.5 points above
chance**. For the first time in six studies the static block carries real, substantial
signal that the series does not already encode.

**H-V9.1 (statics add) — MARGINAL.** Paired D−C is **+2.0** points (+3.6, +2.0, +0.4),
landing a hair under the ≥2 bar. The static block does help — the first time it has ever
measurably helped — but not by the margin required, and the per-seed spread is wide.

**H-V9.2 (the series still matters) — FAIL.** Heartwood beats `agg` by only +1.4 points,
where ≥5 was required and under 2 is a failure.

**H-V9.3 (competitive) — PASS.** +1.3 over the better MiniROCKET.

## The result that matters, and it is not one of the hypotheses

The pre-registered degeneracy check asked whether the ECG adds anything over the statics.
It does not:

```
    ECG alone (arm C)      0.807
    statics alone (arm S)  0.835
    both together (arm D)  0.827   <- worse than statics alone
```

**D − static_only = −0.8 points.** Two informative and largely independent sources — a
physiological signal worth 0.807 and a body-size prior worth 0.835 — and combining them
produces *less* than the better one alone. A competent combination should clear 0.87.

This is not a dataset problem. It is a defect in this library, and V9 is the first
experiment able to see it, because it is the first dataset where both halves were strong at
once.

**The likely cause is structural and specific.** The ridge base is fitted on the
**series only** (`booster._augment` passes `X_series` to `_dense_bank`; the static block
never reaches it). So the strong linear layer — the part that added +3.8 over MiniROCKET in
V7 — cannot see BMI at all. The statics enter solely through greedy tree splits on top of a
series-only base, and greedy per-node selection is the very mechanism `HEADROOM.md` measured
as this architecture's ceiling. The model is forced to learn its best-predicting feature
through its weakest machinery.

## Where six studies leave the premise

The static block has now, once, measurably helped (+2.0). But the combination has still
never beaten the better half on its own. The honest summary is unchanged in substance and
sharper in detail: **this is a strong time-series classifier that does not yet combine
static and temporal information competently**, and V9 localises why.

That is a better position than V7's, where the statics did nothing and there was nothing to
fix. Here there is a specific, testable defect with an obvious remedy: put the static block
in the ridge base, beside the convolution features, so the linear layer can weigh BMI and
ECG jointly and the trees are left to add interactions rather than to discover the main
effect by greedy search.

Also worth stating: 35 subjects, ~10 in each held-out split, produce a seed spread from
0.702 to 0.899. Three seeds is too few for a dataset this small, and any follow-up needs
more.
