# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is exactly what this library
does: gradient-boosted trees whose splits can read the trajectory a row carries, not just
the columns beside it.

Boosting for datasets that mix **static per-row features with raw time series** — without
collapsing the series into aggregates first.

> **Status: v0.1 (Phase A, milestones M1–M2 complete).** The core booster works end to
> end and has a 184-test suite that checks it against brute-force references, including
> the NaN paths. The full benchmark grid (M3) is not written yet: the numbers below come
> from a three-seed development run at one training size, not a final benchmark. See
> `PLAN.md`.

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
aggregation is provably lossy: every row contains two transients at random positions, one
up-then-down and one down-then-up, and the label is which came first, XOR a static flag.
Because both classes hold the same two shapes and each has zero net area, every global
statistic — mean, std, min, max, median, slope, mean-absolute-change — sees an identical
distribution either way. That property is pinned by a test, so it cannot rot.

```
aggregate + boost    test accuracy = 0.490     (chance)
Heartwood            test accuracy = 0.746     (+25.5 points)
```

Development-run results, mean ± sd over three seeds at n_train = 500, n_test = 2000.
`agg` is the global-aggregate workaround; `wagg` is a stronger 4-window aggregate; both
are fed to XGBoost.

| scenario | agg | wagg | Heartwood |
|---|---|---|---|
| which transient came first, XOR static flag | 0.511 ±.010 | **0.798 ±.040** | 0.764 ±.032 |
| event timing vs. per-row deadline | 0.663 ±.011 | 0.935 ±.005 | **0.968 ±.003** |
| trend inside one off-grid window | 0.609 ±.012 | 0.837 ±.068 | **0.886 ±.039** |
| transient height × static coef (RMSE, lower better) | 0.730 ±.004 | **0.384 ±.006** | 0.394 ±.002 |
| pure-static control (no temporal signal) | 0.934 ±.001 | 0.935 ±.004 | 0.938 ±.003 |

Read that honestly, in two parts.

**Against global aggregation — the thing people actually do — Heartwood wins everywhere**,
by 25 to 31 accuracy points on the classification tasks and 46% RMSE on the regression
one, while tying on the control. That is the result the library exists for.

**Against a well-chosen windowed aggregation it is a fairer fight.** Heartwood wins on
event timing and on the off-grid trend, essentially ties on the regression task, and
*loses* on the ordering task (0.764 vs 0.798, roughly one standard deviation). Fixed
windows are a stronger baseline than folklore suggests, and pretending otherwise would
make every other number here less believable. The ordering task is precisely where Phase B
is aimed — comparison splits express "this happened before that" in a single split — so it
is also the clearest target to beat in M4.

The control row is the one that keeps the rest honest: offered a pure-noise series and
thousands of chances to use it, the model declines, and costs nothing.

## Known limitations (v0.1)

- **Small-data candidate lottery.** Temporal candidates are drawn at random, and on the
  hardest scenario only about 8% of draws are useful. Raising `n_shapelet_candidates` from
  4 to 32 lifts that scenario substantially at roughly 4× the fit time — but it is a wash
  on the other four, so more candidates is the wrong lever and better targeting is the
  right one. Phase B replaces the lottery with templates fitted to the gradients plus a
  bank that reuses whatever worked.
- **Relations between two events are hard.** "Shape A occurred before shape B" needs the
  tree to compare two learned positions across several depths. This is the one place a
  fixed-window baseline currently wins.
- **Pure XOR at the root.** When neither modality has marginal signal, a greedy tree has
  to get lucky with its first split, and a large random candidate pool makes that harder
  than a small fixed one.
- Single-threaded, no GPU, pure NumPy. Series are padded to a common length.
- Fit time on the quickstart (n=500, T=100, 150 rounds) is about 10 s.

## Tests

```bash
pip install -e '.[test]'
python -m pytest tests/ -q          # 184 tests, ~30 s
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
heartwood/       losses, features, splits, tree, booster, api, datasets
tests/           the suite described above
examples/        quickstart.py
PLAN.md          the full phased implementation plan
ARCHITECTURES.md the design panel that chose this architecture over four alternatives
```

## Honest positioning

The ingredients are known: XGBoost's gain machinery, interval features from Time Series
Forest / CIF, shapelets. What is new here is the packaging — one regularised booster with
a unified per-node split search over static and temporal candidates, guided by the
gradients. Claims are meant to be demonstrated by benchmark, not asserted; the benchmark
grid lands in M3.
