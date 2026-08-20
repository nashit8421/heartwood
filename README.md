# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is exactly what this library
does: gradient-boosted trees whose splits can read the trajectory a row carries, not just
the columns beside it.

Boosting for datasets that mix **static per-row features with raw time series** — without
collapsing the series into aggregates first.

> **Status: v0.1 (Phase A / milestone M1).** The core booster works end to end and is
> verified against brute-force references. The pytest suite (M2) and the full benchmark
> grid (M3) are not written yet, and the accuracy numbers below are from a
> single-seed development run, not the final benchmark. See `PLAN.md`.

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

The temporal candidates are **redrawn at every node of every round**, so the window that
matters is discovered at whatever position and resolution the gradients call for, rather
than fixed by an up-front aggregation. Interval candidates include the whole series with
some probability, which keeps the classical global aggregate permanently inside the
hypothesis space: the model can never be *less* expressive than the baseline it replaces.

Every split stays human-readable, so the model explains itself:

```
series[ch=0].shapelet_dist(len=29) <= 1.008   gain=20.45
static[0] <= 0.5                              gain=16.61
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
aggregation is provably lossy: every row contains one transient, either up-then-down or
down-then-up, and the label is that orientation XOR a static flag. The two orientations
are exact negations, so both classes have identical global statistics.

```
aggregate + boost   test accuracy = 0.535     (chance)
Heartwood         test accuracy = 0.742     (+20.7 points)
```

Development-run results at n_train=500, against a global-aggregate baseline (`agg`) and a
stronger 4-window aggregate baseline (`wagg`), both fed to XGBoost:

| scenario | agg | wagg | Heartwood |
|---|---|---|---|
| transient orientation XOR static flag | 0.536 | 0.526 | **0.742** |
| event timing vs. per-row deadline | 0.827 | 0.945 | **0.970** |
| trend inside one off-grid window | 0.637 | 0.879 | **0.928** |
| transient height × static coef (RMSE, lower better) | 0.743 | 0.628 | **0.553** |
| pure-static control (no temporal signal) | 0.943 | 0.945 | 0.945 |

The last row is the one that keeps the rest honest: when there is nothing temporal to
find, the temporal machinery costs nothing.

## Known limitations (v0.1)

- **Small-data candidate lottery.** Temporal candidates are drawn at random, and on the
  hardest scenario only about 8% of draws are useful — so at n≈300 the model can miss a
  signal it finds easily at n=500. Raising `n_shapelet_candidates` from 4 to 32 lifts that
  scenario from 0.742 to 0.896, at ~4× the fit time. Phase B replaces the lottery with
  templates fitted to the gradients plus a bank that reuses whatever worked.
- **Pure XOR at the root.** When neither modality has marginal signal, a greedy tree has
  to get lucky with its first split, and a large random candidate pool makes that harder
  than a small fixed one.
- Single-threaded, no GPU, pure NumPy. Series are padded to a common length.
- Fit time on the quickstart (n=500, T=100, 150 rounds) is about 25 s.

## Layout

```
heartwood/     losses, features, splits, tree, booster, api, datasets
examples/        quickstart.py
tests/smoke/     runnable brute-force cross-checks (M2 formalises these as pytest)
PLAN.md          the full phased implementation plan
ARCHITECTURES.md the design panel that chose this architecture over four alternatives
```

## Honest positioning

The ingredients are known: XGBoost's gain machinery, interval features from Time Series
Forest / CIF, shapelets. What is new here is the packaging — one regularised booster with
a unified per-node split search over static and temporal candidates, guided by the
gradients. Claims are meant to be demonstrated by benchmark, not asserted; the benchmark
grid lands in M3.
