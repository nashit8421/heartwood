# Validation Plan V7 — is the win a regime, or one dataset?

**Status: PRE-REGISTERED.** Written and committed before any V7 dataset was fetched.
Fourth study in this repository; the first two produced results that looked good until the
baseline was checked, so the discipline is not optional.

---

## 1. Why this exists

V6 produced the project's first win over a strong external method: with a ridge over a
dilated-convolution bank underneath the trees, Heartwood beats the better `aeon` MiniROCKET
on PTB-XL at n=1000 by **+2.2 points** (three seeds: +2.1, +3.4, +1.2), and reaches parity
across twelve cells.

That win rests on **one dataset, at one training size, with three seeds.** That is roughly
the evidentiary thinness that made the v0.3 headline collapse under audit. V7 exists to
attack it.

Note what is *not* being done. H-V6.2 asked for ≥2 points at a majority of PTB-XL sizes and
got one of four; it **failed**, and it stays failed. Adding sizes until a failed hypothesis
passes is the exact habit this repository keeps having to unlearn. The size question is
re-asked below as a *new* hypothesis, stated before the data that would answer it exists.

---

## 2. Datasets (locked)

Criteria, unchanged from `VALIDATION.md` §3, plus: genuine per-row static covariates
shipped as data, and series in the shape regime by the `VALIDATION_V5.md` §2 diagnostic
(computed from baselines only).

| # | dataset | static | series | target | headline |
|---|---|---|---|---|---|
| V7-M1 | **CPSC-2018** (via PhysioNet/CinC 2020) | age, sex — in the WFDB header | 12-lead ECG, 500 Hz | SNOMED diagnostic class | balanced accuracy |
| V7-M2 | **Sleep-EDF Cassette** (PhysioNet) | age, sex — from `SC-subjects.xls` | single-channel EEG, 30 s epochs | sleep stage | balanced accuracy |

M1 is a **different ECG cohort** from PTB-XL — different country, different label set,
different sampling rate — so it tests replication while holding modality fixed. M2 is a
**different modality and task entirely**, and is the one that decides whether V6 found
something about time series or something about electrocardiograms. M2 needs an EDF reader;
if that proves infeasible within budget it is reported as `unavailable` with the reason, as
EigenWorms was, and H-V7.1 is then answered by M1 alone and labelled as such.

Preprocessing, fixed here:

* **M1**: 12 leads, resampled to 100 Hz by decimation to match PTB-XL's resolution, first
  10 s (1000 samples); records shorter than that are dropped and counted. Labels are the
  SNOMED `Dx` codes; a record is kept only if it maps to exactly one of the classes that
  hold ≥200 records, and the dropped counts are reported. Split is a patient-disjoint
  70/30 by record id hash, fixed by seed, never redrawn.
* **M2**: Fpz-Cz EEG, 30 s epochs at 100 Hz (3000 samples), labelled by the accompanying
  hypnogram. Wake epochs beyond 30 minutes either side of sleep are dropped, which is the
  standard convention for this dataset and is applied before any score is computed. Split
  is subject-disjoint.
  **Subject subset, fixed now:** the full cassette set is ~7.5 GB, so the study uses the
  **first 40 subjects by record id** — a mechanical rule chosen on size alone, recorded
  before any Sleep-EDF number was computed, and reported as a limitation. Forty subjects
  yields tens of thousands of epochs, far more than the n=2000 the size grid asks for; the
  binding constraint is subject count for a disjoint split, not epochs.

**Evaluation cost, fixed now.** Sleep-EDF yields ~43,000 epochs, so roughly 13,000 land in
a held-out split and *prediction*, not fitting, dominates every cell — measured at 2900 s
for n=100, where the fit is trivial. The test split is therefore capped at **4,000 rows by
stratified subsample**. Balanced accuracy over 4,000 stratified rows is precise to well
under the 2-point margins these hypotheses turn on, so the remaining 9,000 buy hours and
nothing else. Applied to Sleep-EDF only; CPSC-2018's test split is already ~1,900.

**Run scope, fixed now.** The full grid — 2 datasets x 5 sizes x 5 seeds x 4 arms — is
over twenty hours of compute, because a single ECG-sized fit takes 450-2500 s. So the
*hypotheses* stand as written and the *runs* are scoped: H-V7.1 is evaluated at n=1000 and
n=2000 with 3 seeds; H-V7.2 additionally needs n=100; the H-V7.3 decomposition runs at
n=1000 only. Every cell not run is reported as not run, with this reason, rather than
being left to look like it was never asked for.
* Both: statics stay missing where missing, no imputation.

---

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V7.1 — replication.** On each V7 mixed dataset, `heartwood_rocket` beats the better
`aeon` MiniROCKET by **≥ 2 points** at n = 1000 and at n = 2000.
*Fails if* it fails to beat it at either size on a majority of the V7 datasets. This is the
question: the V6 win either reproduces on data it has never seen, or it was PTB-XL.

**H-V7.2 — size dependence.** The margin over the better MiniROCKET at the largest size
tested exceeds the margin at n=100 by **≥ 2 points**, on PTB-XL and on each V7 dataset.
*Fails if* the margin at the largest size is not above the margin at n=100.
This is stated directionally rather than as "non-decreasing", because three noisy points
will violate monotonicity for reasons that have nothing to do with the claim.

**H-V7.3 — mechanism.** The win is currently unexplained: Heartwood also beats MiniROCKET
on Handwriting and HandMovementDirection, which have **no static block at all**, so the
story cannot simply be "we use the statics". The contribution is therefore decomposed, on
every mixed dataset and at every size:

| arm | what it is | isolates |
|---|---|---|
| A | `aeon` MiniROCKET | the reference |
| B | ridge over **our** bank, no trees | our bank vs theirs |
| C | `heartwood_rocket`, `X_static=None` | what the trees add |
| D | `heartwood_rocket`, with statics | what the statics add |

Pre-registered prediction: **D − C ≥ 1 point** wherever `static_only` beats the
majority-class rate by ≥2 points, and **C − B ≥ 1 point** on shape-regime data.
*Fails if* D − C < 0 — i.e. if removing the static block *helps*, which would mean the
library's entire premise is not what is producing its results.

Reported regardless of the verdict, because the decomposition is more useful than the
hypothesis.

---

## 4. Rules

As `VALIDATION.md` §2 and `VALIDATION_V5.md` §6, in force unchanged. No tuning; library
defaults; matched budgets; five subsample seeds; MiniROCKET credited with the better of
2,000 and 10,000 kernels; every dataset and baseline reported; splits fixed in advance;
comparisons against `aeon`, never against our own copy of the opponent.

Sizes: n ∈ {100, 250, 500, 1000, 2000}. Any cell exceeding a 45-minute-per-fit budget is
recorded as infeasible with the reason and is not quietly dropped.

**Algorithm changes are out of scope.** If V7 motivates one, it is pre-registered as V8.
The convolution base stays opt-in regardless of what V7 says about accuracy, because
H-V6.3 failed on `amp_regression` and that has not been revisited.

## 5. What counts as done

If H-V7.1 passes, "Heartwood with a convolution base beats MiniROCKET on mixed data with
enough rows" becomes a claim about a regime rather than about PTB-XL, and the README says
so. If it fails, the V6 result is one dataset's quirk, and the README says **that**, in the
same place and the same size type. The decomposition in H-V7.3 is worth having either way,
because "we do not know why we win" is not a state this project should stay in.
