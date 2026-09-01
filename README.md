# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is exactly what this library
does: gradient-boosted trees whose splits can read the trajectory a row carries, not just
the columns beside it.

Boosting for datasets that mix **static per-row features with raw time series** — without
collapsing the series into aggregates first.

> **Status: v0.8 — a strong time-series classifier; the static half remains unproven.** 299 tests and ten
> pre-registered real-data studies. The v0.3 headline was a
> [baseline bug](validation/CORRECTION.md) and **H1 fails** once corrected. The conditional
> restatement ([V5](VALIDATION_V5.md)) fails too: Heartwood beat aggregation but MiniROCKET
> beat Heartwood. Measuring *why* ([`validation/HEADROOM.md`](validation/HEADROOM.md))
> showed the ceiling was greedy per-node selection, not search budget — so
> [V6](VALIDATION_V6.md) put a ridge over a dilated-convolution bank underneath the trees.
> That closes the MiniROCKET gap to parity across 12 cells and **wins by 2.2 points at
> n=1000 on PTB-XL**, the one dataset here with both real static covariates and
> shape-regime series. H-V6.2 still fails on its own terms (1 of 4 sizes, a majority was
> required), so `dense_features="rocket"` ships **opt-in**.

## Validation on real data — the honest verdict

The hypotheses, thresholds, datasets and baselines were
[pre-registered](VALIDATION.md) and committed **before** any real dataset was downloaded,
then evaluated mechanically. That part worked exactly as intended, and it is the reason
this section can be written at all. What it did not protect against was the benchmark
measuring the wrong thing.

**The v0.3 headline was wrong.** `benchmarks/baselines.py` summarised each window with
bare NumPy reductions, so a single missing cell turned a whole statistic into NaN.
Heartwood's own `interval_stat` has always skipped missing cells. On PhysioNet ICU, which
is 80% missing, that left the `agg` design matrix 94% NaN with 176 of 370 series columns
entirely empty — and the reported "+10 points over the aggregate workaround" was mostly
the difference between two ways of averaging, not two ways of representing a series.

| ICU, ROC-AUC | n=100 | n=250 | n=500 | n=1000 | n=3997 |
|---|---|---|---|---|---|
| Heartwood | 0.689 | 0.787 | 0.814 | 0.833 | 0.862 |
| `agg`, NaN-aware (correct) | 0.682 | 0.772 | 0.810 | 0.818 | 0.858 |
| `agg`, NaN-propagating (v0.3) | 0.615 | 0.669 | 0.700 | 0.733 | 0.783 |
| **published margin** | +10.5 | +10.0 | +11.2 | +9.7 | +8.0 |
| **corrected margin** | +0.7 | +1.4 | +0.3 | +1.5 | +0.5 |

A second defect: `run_validation.py` collapsed every official-split dataset to a single
subsample seed, so each `±0.000` in the v0.3 tables meant *n=1*, not zero variance. Both
are fixed, both are pinned by tests, and the pre-registered results are left in place
rather than overwritten.

### What the corrected study says

| dataset | series | Heartwood vs `agg`, by training size | won |
|---|---|---|---|
| **ICU mortality** (PhysioNet 2012) | 37 channels × 48 h, 80% missing | +0.7, +1.4, +0.3, +1.5, +0.5 | 0/5 |
| **Human activity** (UCI HAR) | 9 channels × 128 steps | **+3.5, +5.1, +4.6, +3.5, +5.2** | **5/5** |
| **Credit default** (UCI) | 3 channels × 6 months | −0.4, −0.0, +0.6, +0.8, −0.2 | 0/5 |

**H1 — FAIL**: 5/15 cells at ≥2 points (33%; pass needed 60%, fail is under 50%).
**H2** — PASS. **H3** — PASS at −4.7 points against a −5.0 bar, having moved from −2.8 once
the seeds were fixed. **H4** — not testable. **H5** — PASS, and it is the most defensible
result in the study; `validation/dump_icu_splits.py` reproduces it.

The failures are not scattered, which is the one genuinely useful thing here. Heartwood
ties `agg` exactly where ten statistics over the series are nearly sufficient (ICU, credit)
and beats it exactly where they are not (HAR). Where it wins, though, MiniROCKET wins by
more: on HAR it beats Heartwood by 6 to 13 points at every size.

**So the claim this library was built on is currently unsupported.** What the evidence
supports is narrower: *where aggregation is enough, Heartwood matches it; where aggregation
is not enough, Heartwood helps, but a dedicated time-series method helps more.*
Whether anything stronger holds — in particular on data that has both real static
covariates and series a summary genuinely loses, a combination none of these three datasets
had — is the pre-registered question of [`VALIDATION_V5.md`](VALIDATION_V5.md).

### What it learned, in its own words

This part is unaffected by either defect and reproduces exactly
(`python validation/dump_icu_splits.py`). The ICU model's highest-gain splits, with channel
indices replaced by their clinical names and nothing else edited:

```
gain=90.5   series[BUN].last[t=0:48]    <= 30.5    [missing->left]
gain=66.2   series[GCS].min[t=18:23]    <= 9.25    [missing->right]
gain=59.2   series[Urine].mean[t=0:48]  <= 59.62   [missing->right]
gain=43.7   series[FiO2].min[t=27:33]   <= 0.29    [missing->left]
```

Top families by total gain: Glasgow Coma Scale (last, median, and **delta** — its change
across the stay), age, blood urea nitrogen, urine output, temperature, minimum lactate.

Read that as a clinician would. Low GCS is impaired consciousness, and the thresholds it
picked — 9.25 and 10.5 — sit right at the conventional coma boundary. Elevated BUN is
renal dysfunction. Urine output under ~60 mL/h is oliguria, a standard organ-failure
criterion. Lactate is the shock marker. Raised FiO2 means ventilator support. Nobody told
the model any of this; it found the standard ICU risk factors from raw hourly readings and
reported them in one line each. 0.862 AUROC on the challenge's own split is a respectable
number — it is just not a number only this library can reach.

One dataset from the locked list could not be run: **EigenWorms** (T=17,984) did not
complete a single fit in twelve minutes at default settings. That is a genuine scaling
limit, recorded rather than dropped.

Full tables: [`validation/RESULTS_CORRECTED.md`](validation/RESULTS_CORRECTED.md). The
pre-registered originals are preserved unedited in
[`validation/RESULTS.md`](validation/RESULTS.md).

## V5: the conditional claim, and the cell that was empty

The corrected failures were not scattered — Heartwood tied `agg` on the two datasets whose
series summarise well and beat it on the one whose series do not — so
[`VALIDATION_V5.md`](VALIDATION_V5.md) pre-registered the conditional version of the claim
and tested it on datasets v0.3 never touched. Regime is decided by a diagnostic computed
from **baselines only**, so it cannot be fitted to the answer.

Arm A is **PTB-XL** (12-lead ECG, 100 Hz × 10 s, plus age/sex/height/weight), chosen
because no v0.3 dataset occupied the cell this library was built for: ICU and credit have
real statics whose series summarise away, and HAR's "static block" is a subject id that is
disjoint across the split. Arm B is UEA ranks 9–16 by training-set size.

| | verdict | |
|---|---|---|
| **H-V5.1** conditional claim | **FAIL** | *helps where it should*: **7/8** shape-regime cells at ≥2 pt (+3.8 to +14.5). *harms nowhere*: 0/1 — and one cell is not a test |
| **H-V5.2** regime robustness | **FAIL** | mean rank: minirocket **1.50**, heartwood 2.33, wagg4 3.67, wagg8 4.00, agg 4.08, raw_flat 5.12 |
| **H-V5.3** the empty cell | **FAIL** | 0/4 sizes beat both `agg` and MiniROCKET |

On PTB-XL, Heartwood beats the fair aggregate baseline by **+3.8, +7.2, +9.9, +9.0** points
at n=100/250/500/1000 — the largest, most seed-stable win over `agg` anywhere in this
project, on a genuinely mixed dataset. It loses to MiniROCKET at every size: −6.6, −4.0,
−1.0, −2.1.

So the empty cell is occupied, Heartwood is genuinely good in it, and it does not own it.

### Why, and what that rules out

The obvious hypothesis was search: ~4 random shapelet draws per node cannot find what
10,000 dilated kernels find. [`validation/HEADROOM.md`](validation/HEADROOM.md) tests it by
multiplying the candidate budget ×4 and ×16. **It is signal, not search** — ×16 buys +1.5
points on a ~20-point deficit at 9× the cost, and on two of three datasets a bigger budget
is *worse*.

That last part is the interesting one, and it is structural rather than accidental. A node
takes the single highest-gain split from a pool of random candidates, so enlarging the pool
raises the winner's expected gain **whether or not anything in it is informative**.
Heartwood cannot buy accuracy with candidates, by construction. MiniROCKET avoids this
entirely by not selecting: it computes every kernel and lets a ridge shrink them jointly.

This retires the assumption the README has carried since v0.1 — that better *targeting* is
the open problem. The bottleneck is the selection rule, not the sampler feeding it.

## V6: stop selecting where selection is the bottleneck

`HEADROOM.md` found that Heartwood's deficit to MiniROCKET was not a search problem —
×16 the candidate budget moved a 20-point gap by 1.5 points and made two of three datasets
*worse*. The cause is structural: a node keeps the single highest-gain split from a pool of
random draws, so a bigger pool raises the winner's expected gain whether or not anything in
it is informative. MiniROCKET has no such ceiling because it never selects; it computes
~10,000 fixed dilated-kernel features and lets a ridge shrink all of them jointly.

So V6 stops selecting for that part of the problem. `heartwood/rocket.py` builds the
convolution bank (pure numpy), `DenseBase` puts a leave-one-out ridge over it, and the
trees boost from there — adding what a ridge structurally cannot: static covariates,
static × temporal interactions, and nonlinearity.

| dataset | n | default | **+rocket** | MiniROCKET | vs rocket | vs default |
|---|---|---|---|---|---|---|
| **PTB-XL** | 1000 | 0.490 | **0.533** | 0.511 | **+2.2** | **+4.3** |
| PTB-XL | 500 | 0.465 | 0.489 | 0.475 | +1.5 | +2.5 |
| PTB-XL | 250 | 0.399 | 0.438 | 0.439 | −0.1 | +3.9 |
| PTB-XL | 100 | 0.347 | 0.408 | 0.414 | −0.6 | +6.0 |
| HandMovementDirection | 160 | 0.429 | **0.459** | 0.387 | **+7.2** | +3.0 |
| Handwriting | 150 | 0.316 | **0.520** | 0.514 | +0.7 | **+20.5** |
| RacketSports | 151 | 0.892 | 0.892 | 0.866 | **+2.5** | −0.0 |
| NATOPS | 180 | 0.890 | 0.924 | 0.944 | −2.0 | +3.4 |

Against MiniROCKET across all 12 cells the median gap is **−0.1** — parity, up from a clear
deficit (V5 mean rank 2.33 against 1.50). Against Heartwood's own previous default the gain
is large and near-uniform: 8 of 12 cells up by ≥2.5 points, nothing down by more than 0.3.

**The pre-registered verdicts.** H-V6.1 (the bank is not broken) **PASS**, within 2 points
of `aeon` on 7 of 8 datasets. H-V6.2 (the win) **FAIL** — ≥2 points at 1 of 4 PTB-XL sizes
where a majority was required. H-V6.3 (no regression) **FAIL** on one scenario:
`amp_regression` loses 2.3–6.7 points, because its target is height × static coefficient,
so a linear base predicts the marginal part well and still misleads the trees. So by
H-V6.4 the rocket base **does not become the default**.

The PTB-XL margins are −0.6, −0.1, +1.5, +2.2 as n goes 100 → 1000. That is monotone, and
the crossing at the largest size tested is real and seed-stable (+2.1, +3.4, +1.2). It is
still one of four sizes, and calling a trend a win would be picking the summary after
seeing the data — the exact habit that produced the v0.3 headline. n=2000 is the obvious
next run and has not been done.

## V7: it replicates, and the mechanism is the static block

V6's win was one dataset, one size, three seeds. [V7](VALIDATION_V7.md) attacked it with
two datasets it had never seen — **CPSC-2018** (a different ECG cohort: different country,
label set, sampling rate) and **Sleep-EDF** (a different modality and task entirely: EEG
sleep staging).

| dataset | n=100 | n=1000 | n=2000 |
|---|---|---|---|
| **CPSC-2018** | +2.3 | **+3.7** | **+4.1** |
| **Sleep-EDF** | +1.3 | +1.3 | **+2.4** |

*(points over the better `aeon` MiniROCKET, 3 seeds, balanced accuracy)*

**H-V7.1 PASS** — CPSC clears the bar at every size and every seed. Sleep-EDF clears it at
n=2000 but not n=1000. **H-V7.2 FAIL** — the margin grows with training size (+1.8 and +1.1)
but not by the 2 points required. **H-V7.3 FAIL** — the static block does not help.

### Where the win actually comes from

The contribution was decomposed into four arms at n=1000. The same pattern appears on both
datasets, despite one being a heart and the other a brain:

| arm | CPSC | Sleep-EDF | |
|---|---|---|---|
| A — `aeon` MiniROCKET | 0.654 | 0.660 | the reference |
| B — ridge over **our** bank | 0.654 | 0.661 | bank quality **+0.0 / +0.1** |
| C — + trees, no statics | 0.692 | 0.673 | **trees add +3.8 / +1.2** |
| D — + statics | 0.691 | 0.673 | **statics add −0.1 / +0.0** |

> **These numbers were reported wrongly the first time.** An earlier version of
> `validation/report_v7.py` averaged every model in the no-static results file — `agg` and
> both MiniROCKETs alongside Heartwood — into "arm C", which dragged it down about five
> points and produced a fictitious "trees cost 5 points, statics are worth 8" story. V8 was
> then pre-registered to fix that non-existent tax. The table above is the corrected
> version, verified per-seed and paired. See [`validation/RESULTS_V8.md`](validation/RESULTS_V8.md).

Our convolution bank is *exactly* as good as MiniROCKET's — the reimplementation is not
where any edge lives. **The trees over that base are the edge**, worth +3.8 and +1.2:
nonlinearity a ridge cannot express, on top of a base that already matches the best
available method.

**And the static block is worth nothing measurable** — −0.1 and +0.0, with paired per-seed
differences of `+0.0, −0.4, +0.2` and `−0.1, −0.2, +0.3`. That is this library's founding
premise, isolated for the first time, contributing zero on both datasets where it could be
measured. It is consistent with the rest of the record rather than surprising: Heartwood
also beats MiniROCKET on Handwriting and HandMovementDirection, which have no static block
at all.

So the honest mechanism is narrower than the pitch: **this is a competitive time-series
classifier whose edge is boosted trees over a convolution base, and the "mixed static plus
series" premise is not currently earning its place.**

### V9: the first fair test of the premise, and a defect it exposed

Every dataset above fails to test the founding claim, for a reason visible in the data:
strong static blocks only ever appeared beside series a plain average already captures, and
rich series only ever came with statics that were weak or that the signal itself encodes
(an ECG partly reveals your age). [V9](VALIDATION_V9.md) fixed the selection with a rule
written first — a static column counts only if the signal cannot encode it — and landed on
**Apnea-ECG**, where body weight and BMI predict apnea and no one-minute ECG contains them.

| | ROC-AUC |
|---|---|
| ECG alone | 0.807 |
| statics alone (age, sex, height, weight, BMI) | **0.835** |
| **both together** | 0.827 |

The statics finally help the model (+2.0 points over the same model without them, the first
time in six studies). But **the combination is worse than the better half alone.** Two
strong, largely independent sources, and putting them together loses information.

That is a defect in this library, not in the dataset, and V9 is the first experiment able to
see it. The likely cause is specific: the ridge base is fitted on the **series only** — the
static block never reaches it — so BMI can only enter through greedy tree splits, which
`validation/HEADROOM.md` already measured as this architecture's weakest machinery. The
model is made to learn its single best predictor through the one mechanism it is worst at.
Full numbers and the remedy: [`validation/RESULTS_V9.md`](validation/RESULTS_V9.md).

## V9 and V10: the founding claim, tested and then earned

Five datasets could not test the premise, for a reason visible in the data: strong static
blocks only ever appeared beside series a plain average already captures, and rich series
only ever came with statics that were weak or that the signal itself encodes — an ECG partly
reveals your age. So [V9](VALIDATION_V9.md) fixed the selection with a rule written first —
**a static column counts only if the signal cannot encode it** — and landed on Apnea-ECG,
where body weight and BMI predict apnea and a one-minute ECG contains neither.

V9 found the statics finally mattered, and that the model could not use them:

| | V9 — base blind to statics | **V10 — base sees statics** |
|---|---|---|
| series alone | 0.807 | 0.790 |
| statics alone | 0.835 | 0.835 |
| **both together** | 0.827 — *worse than either* | **0.856** |
| what the statics are worth | +2.0 | **+6.6** |

The defect was specific: the ridge base was fitted on `X_series` alone, so the linear layer —
the component worth +3.8 over MiniROCKET — never saw BMI, and the statics could only enter
through greedy per-node tree splits, which [`HEADROOM.md`](validation/HEADROOM.md) had already
measured as this architecture's ceiling. The model was made to learn its single best
predictor through the one mechanism it is worst at.

V10 fits the base on the statics **and** the convolution bank, with the statics unpenalised
(Frisch–Waugh; five columns sharing a penalty tuned for ten thousand kernel responses would
see BMI shrunk as hard as an arbitrary convolution). The leave-one-out machinery stays exact
— verified to 1.2e-14 against literally refitting without each row, since this is where the
ridge base nearly shipped broken once before.

**Every metric moves, not just the headline** — F1 +2.1, accuracy +3.4, balanced accuracy
+2.2 over statics alone. Full table: [`validation/RESULTS_V10.md`](validation/RESULTS_V10.md).

**This result was withdrawn by V12.** The +2.1 depended on how the ridge base checked its
own work. It used leave-one-*row*-out — hiding one minute of a patient while ~900 other
minutes from that same patient, carrying their age and BMI, stayed in the fit. Every
benchmark here splits Apnea by *subject*, so the base was tuning itself against a far
easier question than the one it is graded on, choosing too weak a penalty and handing the
trees an overconfident fit.

With the base leaving out whole subjects — a closed-form block hold-out, verified exact to
4.9e-15 against literally refitting without each group — the same model scores:

| | AUC |
|---|---|
| statics alone | **0.835** |
| **full model, honest check (V13)** | **0.842** |
| full model, honest check (V12, mistuned) | 0.826 |
| full model, row-wise check (V10) | 0.856 |

**The combination does not beat its best half.** +0.7 against a pre-registered bar of +2,
negative on two of five seeds. Four attempts now: V7 measured the static block at −0.1 and
+0.0, V9 found +2.0 with the combination still losing to statics alone, V10 appeared to fix
it and did not, and V13 recovers half the loss without clearing the bar.

**V12's own number was partly a second bug of mine.** V12 made the base *validate* by
subject but left its *penalty search* row-wise, so it tuned for one question and was graded
on another — effective dof 441 of 1000 rows, and the base declined to exist on two of three
seeds. [V13](validation/RESULTS_V13.md) makes the search use the same criterion as the
judgement. **V10's inflation was ~1.4 points, not the 3.0 recorded in V12.**

The group-aware check ships on regardless. A check that matches the benchmark is not
optional because its answer is unwelcome.

### V13: what the statics are actually worth, and where the series half loses

Against the identical model with the static block dropped (paired, same seeds):

| Apnea, n=1000 | AUC |
|---|---|
| with statics | **0.842** |
| series only | 0.750 |
| **the statics are worth** | **+9.2** — positive on all 5 seeds |

The fusion mechanism *works*. The headline is still +0.7 because **the series half is weak
on this dataset**: series-only scores 0.750 against MiniROCKET's 0.790 — **−3.6, the first
dataset in the project where the time-series half loses.** On 12-lead CPSC the same code is
+3.7. Everything distinguishing this bank from MiniROCKET's is cross-channel, and Apnea is
single-lead.

> ### Correction (2026-09-01): that explanation is wrong, and it was tested
>
> The paragraph above was published as a hypothesis and V14 tested it, holding the dataset,
> task, split, seed and label fixed and varying only the number of ECG leads on CPSC-2018.
>
> | arm | leads | margin over MiniROCKET-10k |
> |---|---|---|
> | A | 12 | +2.8 |
> | B | 3 | +2.6 |
> | C | **1** | **+3.0** |
>
> **Twelve leads score worse than one.** The gap is −0.3 points against a +3.0 bar, the
> ordering is not even monotone, and the margin survives intact at a single lead. Channel
> width is not the mechanism. Full result in `RESULTS_V14.md`; pre-registration, including
> §5 naming this as the outcome it would least like, in `VALIDATION_V14.md`.
>
> The counterexample was already on this page: **Sleep-EDF is single-channel** (EEG Fpz-Cz)
> and the series half wins there by +1.2, two paragraphs below. It was overlooked when the
> cross-channel story was written.
>
> Two later results say the same thing from other directions. **V15** found that none of the
> four things this bank adds to MiniROCKET's — virtual channels, comparison splits, window
> statistics, Lévy areas — clears a +0.5 bar on the UEA suite, and that MiniROCKET's plain
> bank beats the full configuration on 3 of 8 datasets. So there was less cross-channel
> machinery doing work than this paragraph assumed. And the **dataset screen** offers a
> replacement explanation with no channels in it at all: Apnea's regime gap is **−0.010**, so
> a global summary of that one-minute ECG loses nothing a finer representation recovers
> (`RESULTS_SCREEN.md`). A series half with little temporal structure to find would lose to a
> large fixed bank there at any width.
>
> **The honest description of this library is now "MiniROCKET's bank under our trees."**

The Sleep-EDF result re-run group-aware survives: **0.672 vs 0.660** for MiniROCKET-10k and
0.645 for 2k, winning on all three seeds. `static_only` there is 0.202 against a chance
floor of 0.200 — a pure series win, which is why it is the cleanest evidence in the project
that the time-series half is real.

**Where four attempts leave the founding claim: unproven, for a reason about data.** Beating
either half alone needs both halves strong on one dataset. Apnea has exogenous statics and a
weak series; Sleep-EDF has the series and statics at chance. No dataset yet has given us
both.

### The lesson worth keeping

The most useful finding in this study is not about Heartwood. It is that on a dataset which
is 80% missing, **whether your aggregate skips NaNs is worth about ten points of AUC — more
than any representation choice tested here, including this one.** If you are pasting
summary statistics next to static columns, check that a single missing cell is not voiding
the whole statistic before you conclude anything about representations.

The tell was in the published tables the whole time: on ICU, `raw_flat` — every timestep as
its own column, no structure at all — beat `agg` by four points. A representation that
throws everything away should not beat one that summarises it, and that inversion was the
symptom.

## The problem

A very common dataset shape is "some columns per row, plus a trajectory per row":
customer attributes plus 12 months of transactions, machine specs plus sensor traces,
patient demographics plus vitals. There is no mainstream algorithm for it. So people
compute mean/std/min/max/slope of each series, paste those next to the static columns,
and run XGBoost. That works, and it throws away exactly the information that usually
matters: *when* something happened, what shape it had, in what order things occurred.
Sequence models handle the series but need far more data than these datasets have and
treat static covariates as an afterthought.

## The approach

Heartwood keeps XGBoost's machinery unchanged — same second-order gain, same leaf
weights `−G/(H+λ)`, same shrinkage, same sparsity-aware missing-value handling — and
enlarges what a split is allowed to ask. At every node, three kinds of candidate compete
on one gain scale:

| candidate | the question it asks |
|---|---|
| static threshold | `static[3] <= 0.5` — ordinary tabular split |
| interval statistic | `slope of channel 0 between t=12 and t=40 <= 0.31` |
| shapelet distance / position | *does this shape occur?* and *how early does it occur?* |
| **comparison** | *did this happen before that?* — a learned event time versus a static column |
| **banked feature** | anything that already won a split, offered again for free |
| **cross-channel area** | *which channel moved first?* — a signed area no per-channel statistic can express (multichannel only) |
| matched filter (opt-in) | a template fitted to the node's own residuals, at several time scales |
| dense ridge base (opt-in) | a leave-one-out-honest linear model over a wide window bank, used as a starting point |
| **convolution base (opt-in)** | *the same, over ~10,000 dilated kernels — MiniROCKET as a starting point rather than a competitor* |

The temporal candidates are **redrawn at every node of every round**, so the window that
matters is discovered at whatever position and resolution the gradients call for, rather
than fixed by an up-front aggregation. Interval candidates include the whole series with
some probability, which keeps the classical global aggregate permanently inside the
hypothesis space: the model can never be *less* expressive than the baseline it replaces.

Every split stays human-readable, so the model explains itself. Here is the model on the
quickstart task, where the label is which of two transients came first XOR a static flag —
and the top split says exactly that, in one line:

```
rank(series[ch=0].shapelet_pos(len=18)) - rank(static[0]) <= -0.483   gain=59.69
series[ch=0].slope[t=60:99] <= -0.0014                                gain=34.62
```

## Install

```bash
pip install -e .            # numpy only
pip install -e '.[bench]'   # + scikit-learn, xgboost for the benchmark baselines
```

## Usage

```python
from heartwood import HeartwoodClassifier

model = HeartwoodClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                              random_state=0)
model.fit(X_static, X_series, y)          # X_static (n, p), X_series (n, C, T)
model.predict_proba(X_static_te, X_series_te)

model.feature_importances()               # total gain per feature family
model.dump_splits(top=10)                 # every split, readable, ranked by gain
```

- `X_series` accepts `(n, C, T)`, `(n, T)`, or a list of per-sample arrays with
  **variable lengths** (right-padded with NaN internally).
- Either block may be `None`: with no series it is an ordinary gradient-boosted tree,
  with no static block it is a pure time-series learner.
- NaN means missing everywhere — in static columns, inside series, and in padding. Each
  split learns which way to route missing values, so partially observed rows are fine.
- `eval_set=(X_static_val, X_series_val, y_val)` plus `early_stopping_rounds` works as
  you would expect, and prediction then uses the best iteration.

## Does it actually help?

`examples/quickstart.py` runs the comparison. Both models are the *same booster with the
same hyperparameters*; only the representation differs. The dataset is built so that
aggregation is provably lossy: every row contains two transients at random positions, one
up-then-down and one down-then-up, and the label is which came first, XOR a static flag.
Because both classes hold the same two shapes and each has zero net area, every global
statistic — mean, std, min, max, median, slope, mean-absolute-change — sees an identical
distribution either way. That property is pinned by a test, so it cannot rot.

```
aggregate + boost    test accuracy = 0.490     (chance)
Heartwood            test accuracy = 0.970     (+47.9 points)
```

### The benchmark

```bash
python benchmarks/run_benchmarks.py      # 6 scenarios × 4 sizes × 3 seeds, ~4 min
```

Six scenarios, training sizes 100/250/500/1000, test n=2000, three seeds. Every model
gets the same budget — 200 rounds, depth 4, learning rate 0.1 — and Heartwood runs on
library defaults with no per-scenario tuning. Five representations compete: `agg` (global
summaries), `wagg4`/`wagg8`/`wagg16` (the same summaries per equal window), and `raw_flat`
(every timestep as a column). Full tables in [`benchmarks/results.md`](benchmarks/results.md).

Two comparisons matter, and they answer different questions.

**Against `agg`, the workaround teams actually ship.** This is the claim the library makes.

| scenario | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|
| which transient came first | +13.9 pt | +46.0 pt | **+48.2 pt** | +48.7 pt |
| event timing vs. deadline | +33.4 pt | +32.4 pt | **+31.8 pt** | +32.2 pt |
| trend in one window | −1.3 pt | +29.1 pt | **+38.6 pt** | +35.1 pt |
| transient height × coef | 38.7% less error | 46.7% less | **48.4% less** | 49.2% less |
| which channel led (2 channels) | +0.0 pt | +2.2 pt | **+14.0 pt** | +12.3 pt |
| pure-static control | −0.7 pt | −0.4 pt | **−0.1 pt** | −0.5 pt |

The control ties — offered a pure-noise series and thousands of chances to use it, the
model declines and costs nothing.

**Against the best of all five representations** — an oracle that picks, per task and per
size, whichever one happened to win. Nobody can do that in advance, so this is a
deliberately unfair bar; it is here because it is the honest one.

| scenario | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|
| which transient came first | +11.7 pt | +37.6 pt | **+28.4 pt** | +17.9 pt |
| event timing vs. deadline | +8.4 pt | +5.6 pt | **+4.5 pt** | +3.7 pt |
| trend in one window | −7.2 pt | +0.7 pt | **+0.8 pt** | +0.5 pt |
| transient height × coef | 0.7% more error | 1.7% more | **1.6% less** | 2.4% less |
| which channel led (2 channels) | +0.0 pt | +2.2 pt | **+14.0 pt** | +11.3 pt |
| pure-static control | −0.7 pt | −0.4 pt | **−0.1 pt** | −0.5 pt |

Heartwood beats even that oracle on five of the six scenarios from n=250 up, and ties the
control. The one place it still loses is `slope_window` at n=100, where every method
including the baselines sits near chance — there is not enough data for anyone to find
the window.

Worth naming the strongest baseline honestly: on the trend scenario the 8-window
aggregate reaches 0.989, because its boundaries at 45 and 75 happen to bracket the
informative window `[46, 70)` almost exactly. The 4-window grid straddles it (0.787) and
the 16-window grid chops it up (0.812) — a 20-point spread from a hyperparameter you would
have to guess right. That is why the benchmark sweeps window counts instead of quoting
one, and why beating the luckiest grid rather than the average one is the bar we hold
ourselves to.

## What Phase B changed

Three upgrades were designed. **Two earned their place and one did not**, which is the
kind of thing a benchmark exists to tell you. Reproduce with
`python benchmarks/run_benchmarks.py --ablation`.

Accuracy on the ordering task, and the mean fit time across the whole grid:

| variant | n=100 | n=250 | n=500 | n=1000 | fit |
|---|---|---|---|---|---|
| Phase A only | 0.533 | 0.665 | 0.826 | 0.946 | 13.9 s |
| + feature bank | 0.502 | 0.641 | 0.787 | 0.960 | 17.4 s |
| **+ comparison splits (default)** | **0.641** | **0.958** | **0.983** | **0.996** | 18.2 s |
| + matched filters | 0.627 | 0.929 | 0.977 | 0.994 | 28.1 s |

**Comparison splits are the win.** `rank(event time) − rank(static column)` expresses
"did this happen before that" in a single split, where an axis-aligned tree needs a
staircase of them across several depths. They turned out to matter far beyond the
deadline-style task they were designed for: on the ordering and trend scenarios the label
is an interaction between a temporal quantity and a static one, and one comparison split
captures that directly. That is why the ordering task jumps from 0.665 to 0.958 at n=250.

**The feature bank is a supporting act.** On its own it is roughly a wash — it caches
whatever won a split so later nodes get it free, but caching does not help if the useful
feature was never drawn. Its real job is being the substrate comparison splits are built
from, since a comparison needs a position feature that already exists. One thing that
mattered a lot: the bank is *subsampled* per node (`bank_colsample=0.25`). Offering all of
it at every node made the model markedly worse, because every extra candidate is another
chance for noise to win the best-gain contest. That is the same reason boosting subsamples
columns.

**Matched filters did not earn their default.** Fitting a template in closed form to each
node's Newton residuals is a genuinely nice idea — it was the design panel's top-scoring
proposal — but measured against variable-length shapelets it is consistently a little
worse and about 50% slower. Nine-tap templates at dyadic scales seem to trade away the
length flexibility that matters here. The code ships, tested, behind
`n_filter_candidates=8`, and the negative result is recorded rather than buried.

## What Phase C changed

Two more optional pieces, and again the benchmark split them.

**Cross-channel areas ship on by default.** Every scenario above is single-channel, which
was itself a gap, so Phase C added one where two channels carry the same transient offset
by a small lag and the label is which moved first. Every per-channel statistic is blind to
that by construction — both channels look identical in both classes — and indeed every
baseline sits at chance. A signed Lévy area `½ Σ (xΔy − yΔx)` answers it in one number:

| variant | n=100 | n=250 | n=500 |
|---|---|---|---|
| without areas | 0.495 | 0.517 | 0.555 |
| **with areas (default)** | 0.511 | **0.535** | **0.652** |
| best baseline | 0.511 | 0.513 | 0.513 |

They are on by default because they are free everywhere else: on single-channel data they
produce no columns at all and predictions are *bit-identical* (there is a test for that),
and on a multichannel pure-noise control they cost nothing measurable (0.926 → 0.930,
fit time +5%). Strictly dominant, so the plan's "opt-in" was overridden by the numbers.

**The dense ridge base stays opt-in, because it is genuinely a trade.** A ridge over ~490
window statistics can add up a thousand weak signals in a way trees cannot — but it can
only do that when the temporal signal has real *marginal* structure. Three of the six
scenarios are XOR tasks where the series alone correlates with the label at exactly zero,
and there the base is flat and the trees must work around it:

| | bump_order | timing | slope_window | amp_regression | lead_lag |
|---|---|---|---|---|---|
| `dense_base=True` (n=250) | +0.6 pt | −0.4 pt | **+5.2 pt** | 7% more error | −1.8 pt |
| `dense_base=True` (n=500) | +0.3 pt | 0.0 pt | 0.0 pt | **10% more error** | −5.2 pt |

Turn it on when you expect the series to be linearly informative on its own; leave it off
when the signal lives in interactions. Costs nothing in fit time either way.

One implementation note, because it nearly shipped broken. Boosting from a linear base is
only sound if the trees are trained against *out-of-fold* predictions — otherwise the base
looks far better on training rows than it will on new ones and the trees find nothing left
to learn. Ridge gives leave-one-out predictions in closed form, but with more features
than rows it interpolates, every leverage goes to 1, and the formula divides a rounding
error by a rounding error — producing margins that still carry the sign of the label.
Training accuracy 1.00, test accuracy 0.47, nothing raised. The fix refuses any penalty
that lets the model interpolate, and two tests hold that line: one checks the closed form
against literally refitting without each row, the other checks that random labels stay at
chance.

## Known limitations (v0.4)

- **n=100 is still the frontier.** The library's pitch is small data, and Phase B helped
  there (the ordering task went 0.533 → 0.641, timing 0.908 → 0.959) but did not solve it:
  the trend scenario sits at chance for everyone, and the ordering task's spread across
  seeds is ±0.18, meaning the model either finds the signal or does not. Discovery is
  still a lottery; only *keeping* what was discovered has been fixed.
- **Greedy selection over random candidates is the ceiling, and it is structural.**
  Temporal candidates are drawn at random and each node keeps the single highest-gain one,
  so enlarging the pool raises the winner's expected gain whether or not anything in it is
  informative. Measured: ×16 the candidate budget moves a ~20-point deficit by +1.5 points
  at 9× the cost, and makes two of three datasets *worse*
  ([`validation/HEADROOM.md`](validation/HEADROOM.md)). Earlier versions of this section
  called better *targeting* the open problem; that is retired — targeting feeds the
  sampler, and the sampler is not the bottleneck.
- **The ceiling is worked around, not removed.** `dense_features="rocket"` puts a ridge
  over a convolution bank underneath the trees, which reaches parity with MiniROCKET and
  wins at n=1000 on PTB-XL. The greedy split search itself is unchanged and still cannot
  buy accuracy with candidates; the base simply routes around it. If your data has no
  static block and no interaction structure, a rocket-style classifier is simpler, roughly
  two orders of magnitude faster, and gives up little.
- **The convolution base is opt-in and costs fit time.** It fails H-V6.3 on
  `amp_regression` (−2.3 to −6.7), where the target is an interaction and a linear base
  misleads the trees. Turn it on for shape-regime data; leave it off when the signal lives
  in interactions.
- **Comparison splits are approximate.** Ranking two quantities against their own training
  distributions makes them comparable, but that mapping is monotone rather than exact, so
  a single comparison split does not reach the oracle rule when the two quantities have
  genuinely different distributions.
- Single-threaded, no GPU, pure NumPy. Series are padded to a common length.
- Roughly 20–100× slower to fit than XGBoost on pre-aggregated features (18 s vs
  0.1–0.8 s averaged over the benchmark grid), because the temporal features are searched
  rather than precomputed. Phase B added about 30% to fit time; the bank was expected to
  pay for itself by making features reusable and it did not.

## Tests

```bash
pip install -e '.[test]'
python -m pytest tests/ -q          # 299 tests, ~60 s
```

The suite is built around slow, obviously-correct references that share no code with the
implementations they check: interval statistics and shapelet distances against explicit
per-row loops, the split scan against an O(n²) search over 400 randomised problems, and
gradients and hessians against finite differences of each objective. NaN handling gets
more attention than the happy paths, because a window touching a NaN can otherwise
produce a finite-looking number, win a split, and leave every metric plausible.

`tests/test_datasets.py` also pins the benchmark scenarios from both sides: an oracle must
recover each signal, *and* every global aggregate must fail to. During development two
scenarios turned out to be solvable by aggregation — which would have made the headline
result measure the wrong thing — so those properties are now regression-tested.

## Layout

```
heartwood/       losses, features, splits, filters, bank, dense, tree, booster, api, datasets
tests/           299 tests: the library, the scenarios, and the benchmark harness
benchmarks/      baselines, scenario registry, the runner, and results.md
examples/        quickstart.py
PLAN.md          the full phased implementation plan
ARCHITECTURES.md the design panel that chose this architecture over four alternatives
```

## Honest positioning

The ingredients are known: XGBoost's gain machinery, interval features from Time Series
Forest / CIF, shapelets. What is new here is the packaging — one regularised booster with
a unified per-node split search over static and temporal candidates, guided by the
gradients.

**What nine pre-registered studies support.** Heartwood beats the aggregate workaround
reliably and substantially — +3.8 to +14.5 points — where a global summary demonstrably
loses information, and ties it where one does not. Its splits stay readable, and on ICU it
recovered the standard clinical mortality predictors unprompted. With the convolution base
it reaches parity with MiniROCKET across twelve cells and beats it by 2.2 points at n=1000
on PTB-XL. All of that reproduces.

**What they do not support** is a general claim to beat rocket-style methods. H-V6.2 asked
for a majority of PTB-XL sizes and got one of four; below n≈500 MiniROCKET is still ahead,
and on datasets with no static block it wins or ties more often than it loses. The margin
grows with training size, which is suggestive and untested past n=1000.

So the honest position: **this is a strong time-series classifier, and the "static plus
series" premise on the tin is still unproven.** The series half is solid and replicated —
+3.7 and +4.1 over MiniROCKET on a cohort it had never seen. The static half has been
attempted three times and has never been shown to earn its place; the one result that
appeared to demonstrate it did not survive a validation check that matches the benchmark.

Nine studies, three of them written specifically to try to falsify the project's own
headline. All three succeeded. That is the record, and it is the reason to trust what
remains.
