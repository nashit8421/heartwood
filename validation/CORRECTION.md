# Correction to the v0.3 real-data validation

**What this is.** An audit of `VALIDATION.md`'s study found two defects in the harness.
One of them decided the headline result. This document records what was wrong, how much it
was worth, and what the pre-registered hypotheses say once it is fixed.

**The pre-registered record is not edited.** `validation/RESULTS.md` and the raw
`results.json` files under `validation/{credit,har,icu,uea,uea2,uea3}/` are left exactly as
they were committed. Re-running a pre-registered study with a changed baseline and quietly
replacing the tables is the one move that would genuinely damage this project's
credibility; `VALIDATION.md` §2 rule 3 says post-hoc work must be labelled, so it is.
Corrected numbers live in `validation/rerun/` and `validation/RESULTS_CORRECTED.md`.

**Headline: H1 fails.** The claim this library was built to support — that it beats the
aggregate workaround on real data — is not supported once the aggregate workaround is
implemented the same way Heartwood implements it.

---

## Defect 1 — the aggregate baseline propagated NaN; Heartwood never did

`benchmarks/baselines.py::_block_stats` summarised each window with bare NumPy reductions:
`block.mean(1)`, `block.min(1)`, `np.median(block, axis=1)`. Those return NaN if *any* cell
in the window is missing. `heartwood/features.py::interval_stat` has always skipped missing
cells by explicit masking — it is documented as doing so in its own module docstring.

On dense data the difference is invisible, which is why it survived 267 tests and a full
synthetic benchmark grid: all six scenarios have exactly 0.000 missingness, as do credit
and HAR. On PhysioNet ICU, which is 79.7% missing, it is not invisible:

```
agg design matrix on ICU:  (7990, 375),  94.1% NaN
                           176 of 370 series columns entirely NaN
                           mean/std/min/max/slope/median/mean_abs_change: 0.0% finite
```

So `agg`'s lift over `static_only` was not coming from aggregate *values* at all. It was
coming from XGBoost treating near-empty columns as missingness indicators. The benchmark
was varying two things at once — the representation under study, and the missing-data
convention — and attributing the whole difference to the first.

### What it was worth

Same ten statistics, same 375 columns, same budget, same seeds; the only change is
skipping missing cells instead of propagating them. `agg_naive` is the old behaviour,
retained so the v0.3 numbers still reproduce.

| ICU, ROC-AUC | n=100 | n=250 | n=500 | n=1000 | n=3997 |
|---|---|---|---|---|---|
| heartwood | 0.689±0.038 | 0.787±0.017 | 0.814±0.010 | 0.833±0.011 | 0.862 |
| `agg` (fixed) | 0.682±0.047 | 0.772±0.023 | 0.810±0.009 | 0.818±0.011 | 0.858 |
| `agg_naive` (v0.3) | 0.615±0.037 | 0.669±0.012 | 0.700±0.016 | 0.733±0.012 | 0.783 |
| **published margin** | **+10.5** | **+10.0** | **+11.2** | **+9.7** | **+8.0** |
| **corrected margin** | **+0.7** | **+1.4** | **+0.3** | **+1.5** | **+0.5** |

Roughly 93% of the ICU margin was the summary function, not the representation. Under H1's
own ≥2-point rule, ICU goes from **5/5 cells won to 0/5**.

### Why this matters more than the number

The README explained the ICU win by saying the trajectory holds structure that summary
statistics throw away — *when* something happened, and missingness as a first-class
feature. Neither explanation survives. `agg` is a **global** summary: it has no timing
information whatsoever, and it models missingness only incidentally. It lands within 1.5
points at every training size. Whatever Heartwood was doing on ICU, a NaN-aware mean of
each channel was doing nearly all of it.

## Defect 2 — one seed, reported as five

`run_validation.py` contained `n_seeds = 1 if official is not None`. An official train/test
split fixes the *split*; it does not fix the *subsample*, and `VALIDATION.md` §5 asks for
five repeats of the small-data curve regardless. The effect is that ICU, HAR and all seven
UEA rows in `RESULTS.md` are single draws, and every `±0.000` in that file means *n=1*, not
*zero variance*.

It mattered. Measured across five subsample seeds, the published cells are not typical
draws:

* ICU n=100: published +10.5; five-seed mean +7.4 (sd 2.3) — the published cell was the
  *high* draw.
* HAR n=251: published +9.2; five-seed mean +5.1 (sd 3.2) — the luckiest of five.
* UEA AtrialFibrillation, the single largest number in the H3 table: published balanced
  accuracy 0.467, giving "+33.3 points over MiniROCKET". Across seeds it is 0.467 / 0.267 /
  0.200 — at or *below* the 0.333 chance line on two of three draws. The test set has 15
  samples, so one sample is worth 6.7 points of balanced accuracy and the headline number
  was five samples of luck.

---

## What the corrected study says

Full tables: [`validation/RESULTS_CORRECTED.md`](RESULTS_CORRECTED.md). Verdicts are
recomputed mechanically by the same `report.py`, against the same frozen thresholds.

| pre-registered hypothesis | v0.3 verdict | corrected verdict |
|---|---|---|
| **H1** — beats `agg` by ≥2 pt on ≥60% of cells | PASS, 10/15 (67%) | **FAIL, 5/15 (33%)** |
| **H2** — wins at the smallest informative size | PASS | PASS |
| **H3** — median within 5 pt of MiniROCKET | PASS, −2.8 pt | PASS, **−4.7 pt** (bar is −5.0) |
| **H4** — no harm on uninformative series | not testable | not testable |

H1 splits cleanly by dataset, and the split is the interesting part:

| dataset | corrected margin over `agg` | cells won |
|---|---|---|
| credit | −0.4, −0.0, +0.6, +0.8, −0.2 | 0/5 |
| **HAR** | **+3.5, +5.1, +4.6, +3.5, +5.2** | **5/5** |
| ICU | +0.7, +1.4, +0.3, +1.5, +0.5 | 0/5 |

**H3 survives, but only just, and for different reasons than before.** Proper seeds moved
the median from −2.8 to −4.7 against a −5.0 pass bar. The two datasets that carried the
v0.3 story both shrank: StandWalkJump's "+6.7" is now −0.0, and AtrialFibrillation's
"+33.3" is now +16.0 — still large, still on a 15-sample test set where one sample is worth
6.7 points. And EigenWorms, the eighth dataset the locked selection rule named, remains
unfittable; scored as any loss worse than −7.5 it would drag the median to about −5.2 and
turn H3 marginal. A hypothesis that depends this heavily on which of eight datasets could
be run is not carrying much weight.

**The blast radius is exactly one dataset.** The post-hoc table in
`RESULTS_CORRECTED.md` shows `agg` gaining +6.7 to +11.1 points on ICU from NaN-skipping
and exactly **0.0 points** on credit, HAR, and all seven UEA datasets — none of which has a
missing cell. That is the cleanest possible confirmation that the defect touched one
result and that the synthetic grid never needed re-running.

### What the corrected picture actually is

Heartwood beats the aggregate workaround on **HAR and nowhere else** among the mixed
datasets — and HAR is the one whose series a global summary demonstrably cannot capture.
On ICU and credit, where ten statistics over the series are close to sufficient, it ties.
Meanwhile on HAR, MiniROCKET beats Heartwood at every training size by 6 to 13 points.

So the honest one-line summary of v0.3 is: *where aggregation is enough, Heartwood matches
it; where aggregation is not enough, Heartwood helps but a dedicated time-series method
helps more.* That is a much narrower claim than the one the README made, and whether
anything stronger is available is the question `VALIDATION_V5.md` was written to answer.

## What still stands

The audit checked more than it broke, and several things came through intact:

* **The pre-registration is real.** `VALIDATION.md` was committed at 17:01 and every
  results commit follows it. No hypothesis was edited, no dataset was swapped, and
  EigenWorms was reported as infeasible rather than dropped.
* **Every published number reproduces exactly.** Re-running seed 0 returns 0.682 / 0.779 /
  0.819 / 0.851 / 0.862 on ICU and reproduces every baseline to three decimals. The defects
  were in what was measured, never in the recording of it.
* **No leakage.** Comparison-split ranks are computed against frozen training grids
  (`features.py::ecdf`, `booster.py::_static_grids`), so predictions are not transductive.
* **H5 is genuine.** `validation/dump_icu_splits.py` — committed here, since the claim was
  previously README-only — reproduces the quoted splits verbatim from a `random_state=0`
  fit on set-a. The top families really are GCS (last, median, delta), BUN, urine output,
  temperature, minimum lactate and age. That result is unaffected by either defect and is
  the most defensible thing in the study.
* **The synthetic grid is untouched.** Zero missingness in all six scenarios, verified and
  pinned by `test_dense_data_is_unaffected_by_the_fix`. `benchmarks/results.md` did not
  need re-running.

Two further corrections to claims in the README, neither load-bearing:

* MiniROCKET ran at 2,000 kernels, not the "10,000 random kernels" the README stated. Both
  budgets are now run and the better is credited; on ICU the *smaller* bank is the stronger
  baseline, so the text overstated the opponent while under-running it.
* MiniROCKET never received the static block. Tested: giving it one gains ≤0.6 points on
  ICU, so this caveat dissolves — that comparison was fair.

## What was fixed, and what stops it recurring

`_block_stats` is NaN-skipping and matches `interval_stat` statistic for statistic. The
test that matters is not "handles NaN" but
`test_baseline_and_heartwood_agree_on_what_a_summary_of_gaps_means`, which asserts the two
agree column-for-column on a 70%-missing block. A benchmark that varies two things at once
is not a benchmark, and `VALIDATION_V5.md` §6 now carries that as a standing rule.

`run_validation.py` runs five subsample seeds; `--full-seeds` covers the one genuinely
redundant case, where an official split with no subsampling means every seed sees identical
rows. `report.py` grows `--out` so the pre-registered `RESULTS.md` cannot be overwritten by
a later, differently-configured run.

## How this was found

Not by the test suite, and not by re-reading the code. It was found by asking a question
the study never asked itself — *is the baseline implemented the way the thing it is
compared against is implemented?* — and then measuring the answer instead of arguing about
it. The tell was visible in the published tables all along: on ICU, `raw_flat` (1,776
NaN-carrying columns handed straight to XGBoost) beat `agg` by four points. A representation
that throws away all structure should not beat one that summarises it, and that inversion
was the symptom of a baseline whose features had been destroyed before XGBoost ever saw
them.
