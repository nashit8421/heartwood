# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is exactly what this library
does: gradient-boosted trees whose splits can read the trajectory a row carries, not just
the columns beside it.

Boosting for datasets that mix **static per-row features with raw time series** — without
collapsing the series into aggregates first.

> **Status: v0.1 — Phase A complete (M1–M3).** The core booster works end to end, has a
> 200-test suite that checks it against brute-force references including the NaN paths,
> and the full benchmark grid below is reproducible in about three minutes. Phase B (the
> three upgrades that address the limitations at the bottom of this page) is next. See
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

### The benchmark

```bash
python benchmarks/run_benchmarks.py      # 5 scenarios × 4 sizes × 3 seeds, ~3 min
```

Five scenarios, training sizes 100/250/500/1000, test n=2000, three seeds. Every model
gets the same budget — 200 rounds, depth 4, learning rate 0.1 — and Heartwood runs on
library defaults with no per-scenario tuning. Five representations compete: `agg` (global
summaries), `wagg4`/`wagg8`/`wagg16` (the same summaries per equal window), and `raw_flat`
(every timestep as a column). Full tables in [`benchmarks/results.md`](benchmarks/results.md).

Two comparisons matter, and they answer different questions.

**Against `agg`, the workaround teams actually ship.** This is the claim the library makes.

| scenario | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|
| which transient came first | +3.1 pt | +16.6 pt | **+32.5 pt** | +43.7 pt |
| event timing vs. deadline | +28.3 pt | +30.2 pt | **+29.9 pt** | +30.6 pt |
| trend in one window | +2.9 pt | +16.0 pt | **+28.4 pt** | +33.2 pt |
| transient height × coef | 32.5% less error | 45.0% less | **45.7% less** | 46.6% less |
| pure-static control | −0.3 pt | +0.6 pt | **+0.0 pt** | −0.2 pt |

Every target is met with room to spare, and the control ties — offered a pure-noise series
and thousands of chances to use it, the model declines and costs nothing.

**Against the best of all five representations** — an oracle that picks, per task and per
size, whichever one happened to win. Nobody can do that in advance, so this is a
deliberately unfair bar; it is here because it is the honest one.

| scenario | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|
| which transient came first | +0.9 pt | +8.3 pt | **+12.7 pt** | +12.9 pt |
| event timing vs. deadline | +3.3 pt | +3.4 pt | **+2.7 pt** | +2.1 pt |
| trend in one window | −3.0 pt | −12.3 pt | **−9.5 pt** | −1.3 pt |
| transient height × coef | 10.9% more error | 5.1% more | **3.5% more** | 2.5% more |
| pure-static control | −0.3 pt | +0.6 pt | **+0.0 pt** | −0.2 pt |

Heartwood wins three and loses two. Both losses have the same two causes, and neither is a
modelling limitation:

*A fixed grid sometimes gets lucky.* The trend scenario's informative window is
`[46, 70)`; the 8-window grid has boundaries at 45 and 75, so two of its windows bracket
it almost exactly and `wagg8` reaches 0.989. The 4-window grid straddles it and manages
0.787; the 16-window grid chops it up and manages 0.812. Same task, same data — a 20-point
spread depending on a hyperparameter you would have to guess right. This is why the
benchmark sweeps window counts rather than reporting one.

*The candidate lottery, again.* Both losses close when the temporal search gets a bigger
budget. Raising `n_interval_candidates` from 16 to 96 takes the trend scenario from 0.894
to **0.982** (against `wagg8`'s 0.989) and the regression from 0.393 to **0.377** (beating
`wagg4`'s 0.380) — at roughly double the fit time. Heartwood can express both signals
perfectly well; on defaults it just does not draw the right window often enough. That is
precisely what Phase B is for: fit templates to the gradients instead of sampling blindly,
and reuse whatever worked instead of rediscovering it.

## Known limitations (v0.1)

- **The candidate lottery is the dominant weakness.** Temporal candidates are drawn at
  random, so the model has to be lucky enough to draw a useful window or template. On the
  hardest scenario only about 8% of shapelet draws are informative. `n_interval_candidates`
  is consequently the main quality/time knob: raising it from 16 to 96 turns both benchmark
  losses into wins, at roughly double the fit time. Better targeting, not a bigger budget,
  is the real fix — that is Phase B.
- **Small data is where it hurts.** At n=100 two scenarios sit near chance. The
  library's pitch is small data, so this is the gap that matters most; note that the
  baselines are near chance there too.
- **Pure XOR at the root.** When neither modality has marginal signal, a greedy tree has
  to get lucky with its first split, and a large random candidate pool makes that harder
  than a small fixed one.
- Single-threaded, no GPU, pure NumPy. Series are padded to a common length.
- Roughly 25× slower to fit than XGBoost on pre-aggregated features (14 s vs 0.1–0.8 s
  averaged over the benchmark grid), because the temporal features are searched rather
  than precomputed. About 10 s for the quickstart; 29 s for the largest grid cell.

## Tests

```bash
pip install -e '.[test]'
python -m pytest tests/ -q          # 200 tests, ~45 s
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
tests/           200 tests: the library, the scenarios, and the benchmark harness
benchmarks/      baselines, scenario registry, the runner, and results.md
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
