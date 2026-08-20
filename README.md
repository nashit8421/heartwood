# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is exactly what this library
does: gradient-boosted trees whose splits can read the trajectory a row carries, not just
the columns beside it.

Boosting for datasets that mix **static per-row features with raw time series** — without
collapsing the series into aggregates first.

> **Status: v0.3 — all three phases complete (M1–M5).** 267 tests, six benchmark
> scenarios, and a reproducible grid. Heartwood beats every baseline on five of six
> scenarios and ties the control. Three of the five upgrades across Phases B and C earned
> a place in the defaults; two are opt-in or off, with the measurements that decided it
> recorded below. See `PLAN.md`.

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

## Known limitations (v0.1)

- **n=100 is still the frontier.** The library's pitch is small data, and Phase B helped
  there (the ordering task went 0.533 → 0.641, timing 0.908 → 0.959) but did not solve it:
  the trend scenario sits at chance for everyone, and the ordering task's spread across
  seeds is ±0.18, meaning the model either finds the signal or does not. Discovery is
  still a lottery; only *keeping* what was discovered has been fixed.
- **The candidate lottery underneath.** Temporal candidates are drawn at random, and on
  the hardest scenario fewer than one shapelet draw in ten is informative.
  `n_interval_candidates` remains the main quality/time knob. Better targeting rather than
  a bigger budget is the open problem — matched filters were the attempt, and they did not
  pay off.
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
python -m pytest tests/ -q          # 267 tests, ~60 s
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
tests/           267 tests: the library, the scenarios, and the benchmark harness
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
