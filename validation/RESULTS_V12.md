# V12 results — the V10 win does not survive an honest check

## The headline

**H-V12.2 FAILS, and it invalidates V10's claim.**

Apnea-ECG, n=1000, 5 seeds. The only difference between these two rows is *how the ridge
base validates itself* — row-wise leave-one-out, or leaving out whole subjects:

| | seeds | mean |
|---|---|---|
| V10 — base checks itself row-wise | 0.785 0.906 0.905 0.887 0.797 | **0.856** |
| V12 — base checks itself by subject | 0.722 0.910 0.918 0.863 0.717 | **0.826** |
| `static_only` (age, sex, height, weight, BMI) | 0.744 0.893 0.867 0.874 0.797 | **0.835** |

Paired difference: **−3.0 points** (−6.2, +0.4, +1.3, −2.4, −8.0).

With the honest check, the full model scores **0.826 against 0.835 for the static block
alone**. It no longer beats its best half. **The V10 result — reported here as the first
demonstration in seven studies of this library's founding claim — was substantially an
artefact of a validation procedure that could not see the thing being tested.**

## Why the row-wise check was wrong

The base chooses its own penalty strength, and decides whether to exist at all, by asking
"how well would I predict a row I had not seen?" Leave-one-out answers that by hiding **one
minute** of one patient — while nine hundred–odd other minutes from that same patient remain
in the fit, carrying their age, weight and BMI.

Every benchmark here splits Apnea by **subject**. So the base was tuning itself against a
question far easier than the one it would be graded on, choosing too weak a penalty and
producing an overconfident fit. The trees then boosted from it.

Leaving out whole subjects removes ~4% of the data per fold rather than 0.1%, and it is the
question that matches the test. The resulting base is more conservative, and the honest
score is three points lower.

## The other hypotheses

**H-V12.1 — MARGINAL.** With the guard able to see extrapolation, bounded rank-products
score **0.828**, against **0.478** for V11's raw products. The bounding worked exactly as
designed — the catastrophe is gone — but the products add nothing over leaving them out
(0.826 → 0.828).

**H-V12.3 — MARGINAL.** `static_control` improves from V10's −1.1/−2.7/−3.1/−2.0 to
−1.1/−1.2/−1.5/−1.2. Better, no cell now worse than −2, still not within the 1 point a pass
required.

**H-V12.4 — FAIL.** `amp_regression` still collapses, −11.3 at n=100 against V11's −13.4.
Bounding did not rescue it, which is itself informative: that failure was never
extrapolation. Its target is `transient_height × static_coefficient`, a product of
*magnitudes*, and a product of *ranks* throws away exactly the information it needs. The
base fits something confidently wrong and the trees inherit it.

**H-V12.5.** Group-aware validation ships **on** whenever groups are supplied — it fails
H-V12.2 in the sense of costing accuracy, but the rule it fails is about preserving a number
that should not have been believed. A check that matches the benchmark is not optional
because its answer is unwelcome. Bounded interactions ship **off**: they fail H-V12.4 and
buy nothing where they are safe.

## What this does to the record

The claim in the README that this library combines static covariates with a raw series and
beats either alone **is withdrawn**. On the one dataset able to test it, with a validation
procedure that matches the benchmark, the combination scores 0.826 against 0.835 for the
statics by themselves.

Where seven studies actually leave things:

* A strong time-series classifier. It beats `aeon` MiniROCKET by +3.7 and +4.1 on CPSC-2018,
  replicated on data it had never seen (V7), and that result does not depend on any of this.
* The convolution bank matches MiniROCKET's exactly; the edge is the trees over it.
* **The static block has never been shown to earn its place.** V7 measured it at −0.1 and
  +0.0. V9 found it helped by +2.0 but the combination still lost to statics alone. V10
  appeared to fix that and did not. Three attempts, no demonstration.

That is the third time in this project a confident result has been undone by a measurement
defect rather than a modelling one — after the v0.3 aggregate baseline and the V7 arm-C
aggregation bug. The difference is that this one was predicted: V12 §5 named H-V12.2 as
"the outcome I would least like and the one this plan exists to make findable", before it
was run.
