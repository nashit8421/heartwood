# Validation Plan — real data

**Status: PRE-REGISTERED. Written and committed before any real dataset was downloaded
or any result computed.** The git history is the evidence for that ordering, and it is the
point of the document.

---

## 1. Why this exists

Every number Heartwood has produced so far comes from generators in `heartwood/datasets.py`
that I wrote myself, while also writing the algorithm. That is not a small caveat, and the
project record shows the failure mode concretely:

- `make_bump_interaction` went through three designs, each revised until the baseline
  failed and Heartwood won. Each revision was individually justified — the earlier versions
  really were broken — but the cumulative process still selected for a task we win.
- In M5 I implemented Lévy areas, found no scenario could test them, wrote
  `make_lead_lag` where "which channel led" *is* the label, measured +9.7 points, and
  turned the feature on by default. That is evidence the implementation is correct. It is
  not independent evidence that the feature is useful.

So the synthetic suite establishes: *when the signal is journey-shaped, aggregation
destroys it and Heartwood recovers it.* It cannot establish how often real data is that
shape, which is the question that decides whether any of this matters.

**This plan is designed to be able to fail.** If the pre-registered hypotheses do not hold,
that goes in the README as the headline, and the honest conclusion is that the idea did not
transfer.

---

## 2. Rules, fixed in advance

These exist because the natural drift of this work is toward flattering results.

1. **No tuning.** Heartwood runs on library defaults everywhere. Baselines get matched
   budgets (same rounds/depth/learning rate). If any tuning happens it is a *separate,
   clearly-labelled exploratory arm* and never the headline.
2. **No dataset dropping.** The list in §4 is locked below. A dataset that fails to
   download or parse is reported as **unavailable, with the reason** — not silently
   swapped for another. A dataset where we lose is reported as a loss.
3. **No hypothesis edits after seeing results.** §6 is frozen. Post-hoc observations are
   allowed but must be labelled "exploratory, not pre-registered".
4. **Report every dataset × every baseline.** No selective tables.
5. **One headline metric per dataset, chosen in §5 before running**, so we cannot pick the
   metric that happens to look best afterwards.
6. **Splits are fixed by seed and never re-drawn** after seeing a result.

---

## 3. Dataset selection criteria (stated before choosing)

A dataset qualifies for the **mixed** arm if:

- it has genuine per-row static covariates **and** a per-row raw time series;
- it is publicly downloadable without credentialed or manual-approval access;
- the prediction target is per-row (not forecasting the series itself);
- n is between ~500 and ~50,000 rows (the regime we claim).

A dataset qualifies for the **temporal-only** arm if it is a standard, published
time-series-classification benchmark. That arm has no static block (`X_static=None`, a mode
we support) and exists to answer a different question: does the temporal machinery hold up
against real time-series methods on real signals, rather than only against aggregation?

---

## 4. Datasets (locked)

**Mixed arm** — the actual product claim.

| # | dataset | static | series | target | why chosen |
|---|---|---|---|---|---|
| M1 | UCI Default of Credit Card Clients (Taiwan) | limit, sex, education, marriage, age | 6 months × 3 channels (repayment status, bill amount, payment amount) | binary default | genuinely mixed, easy access, **short T=6** — a hard case for us, deliberately kept |
| M2 | PhysioNet 2012 Challenge (ICU mortality) | age, gender, height, ICU type, weight | 48 h × ~37 clinical variables, irregular → hourly bins | binary in-hospital death | the archetypal target shape, heavy missingness, class imbalance |
| M3 | UCI Human Activity Recognition (smartphones) | subject-level attributes | 128 timesteps × 9 inertial channels | 6-class activity | multichannel, real sensors, tests the multiclass path |

**Temporal-only arm** — comparison against real TSC methods.

| # | source | selection rule (fixed now) |
|---|---|---|
| T1 | UEA multivariate archive | the **8 smallest by training-set size**, chosen by n alone, before looking at any accuracy. Small n is our claim, and picking by size is a rule I cannot bend afterwards. |

Any dataset that cannot be obtained is recorded in the results table as `unavailable`
with the failure reason.

---

## 5. Protocol

**Splits.** Where the source ships an official train/test split (UEA, HAR, PhysioNet
2012 sets a/b/c), use it — no re-splitting. Otherwise stratified 70/30, five repeats with
seeds 0–4. Grouping is respected where rows share a subject (HAR is grouped by subject).

**The small-data curve.** This is the central claim, so it gets a first-class test: for
each dataset, subsample the training set to n ∈ {100, 250, 500, 1000, all} (stratified,
5 seeds) and evaluate on the *full* held-out set every time. A method that only wins with
plenty of data has not supported the claim that motivated the project.

**Headline metrics, fixed now:** ROC-AUC for the imbalanced binary datasets (M1, M2),
balanced accuracy for multiclass (M3, T1), RMSE for any regression. Precision, recall, F1
and accuracy are reported alongside for all — those were the pain points in the original
brief — but the headline is the one named here.

**Missing data** stays missing. No imputation is applied for Heartwood; baselines get
median imputation where their model requires it, which is stated per table.

## 5b. Baselines

| baseline | what it tells us |
|---|---|
| `static_only` + XGBoost | **whether the series matters at all** on this dataset |
| `agg` + XGBoost | the workaround this project exists to replace — the primary comparison |
| `wagg4`, `wagg8` + XGBoost | the stronger fixed-window version |
| `raw_flat` + XGBoost | every timestep as a column |
| MiniROCKET + ridge | a real TSC method; installed from PyPI (`aeon`), or hand-rolled if unavailable |

`static_only` doing as well as `agg` means the dataset has no recoverable temporal signal,
and *nobody* can win there. Reporting how often that happens across real datasets is a
deliverable in its own right (§7), because it is the honest form of the question I said I
could not answer.

---

## 6. Hypotheses — frozen, with numeric pass/fail

**H1 — the core claim.** On mixed datasets where the series demonstrably carries signal
(`agg` beats `static_only` by ≥ 2 points on the headline metric), Heartwood beats `agg` by
**≥ 2 points on at least 60%** of such dataset × training-size cells.
*Fails if* it wins on < 50% of cells. Failure means the central claim does not transfer,
and that becomes the README headline.

**H2 — small data.** On those same datasets, Heartwood beats `agg` at the **smallest
training size where the series is informative at all**.
*Fails if* Heartwood's margin over `agg` is negative at n = 100 and n = 250 on a majority
of datasets — i.e. if it only wins once data is plentiful.

**H3 — versus real time-series methods.** On the temporal-only arm, Heartwood's median
headline metric is **within 5 points of MiniROCKET**.
*Fails if* the median gap exceeds 10 points. (Beating MiniROCKET is not expected and is not
required; being in the same league is what "a credible method" means here.)

**H4 — no harm.** On datasets where the series is uninformative (`agg` ≈ `static_only`
within 2 points), Heartwood stays **within 2 points of `static_only`**.
*Fails if* it is worse by > 4 points — that would mean the temporal machinery actively
costs accuracy on real data, which the synthetic control said it does not.

**H5 — interpretability sanity (qualitative, no pass/fail).** For each mixed dataset, dump
the top splits and record whether they are domain-plausible. Reported verbatim, including
when they are nonsense.

---

## 7. Deliverables

1. `validation/results.json` — every raw cell.
2. `validation/RESULTS.md` — full tables, all datasets, all baselines, all sizes.
3. **The headroom finding**: across all real datasets examined, the fraction where
   temporal structure measurably helps *any* method over `static_only`. This is the direct,
   honest answer to "how often is real data journey-shaped?"
4. A verdict section in the README stating which hypotheses passed and which failed, in
   the same prominence as the synthetic numbers.

---

## 8. Execution phases

- **V1 — harness + the easiest mixed dataset (M1).** Build `validation/` mirroring
  `benchmarks/`: loaders, the same metric code (already tested), the same runner shape.
  M1 first because its access is trivial, so any failure is ours and not the data's.
- **V2 — M3 and the temporal-only arm (T1).** Multichannel and multiclass paths, and the
  MiniROCKET comparison.
- **V3 — M2 (PhysioNet 2012).** Most work: irregular sampling must be binned to hourly,
  which is a real preprocessing decision and will be documented and version-controlled.
- **V4 — write up, including failures. Update README verdict.**

---

## 9. Threats to validity, stated up front

- **Preprocessing is a degree of freedom.** Binning PhysioNet to hourly, choosing which of
  its ~37 variables to keep, and encoding HAR's static block are all choices that could be
  nudged. Each is fixed in code, committed, and never revised after seeing a score.
- **Three mixed datasets is not a survey.** Whatever happens, the conclusion is about these
  datasets, not about "real data" in general. Say so.
- **XGBoost baselines get default hyperparameters too.** A well-tuned XGBoost on aggregates
  may beat an untuned Heartwood; the matched-budget comparison is the fair one for the
  representation question, and this limitation is stated rather than hidden.
- **Short series (M1, T=6) may be outside the regime where any of this helps.** It is kept
  deliberately, because reporting only the datasets suited to us is the exact bias this
  document exists to prevent.
