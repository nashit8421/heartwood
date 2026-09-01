# When to use Heartwood, and when not to

The short version: **use it when your rows carry both a trajectory and their own
attributes, and the timing or shape of that trajectory matters.** Reach for something else
otherwise, and this page says which something else.

## Use it when

**Your data is series + static covariates, and the covariates are not derivable from the
series.** This is the case the library is shaped for. A patient's ECG plus their BMI; a
machine's vibration trace plus its model number; a wearer's accelerometry plus their body
mass. The trees can split on both in the same node, on the same gain.

**When something happens matters, not just how much of it there is.** If `mean(series)`
and `max(series)` would throw away the answer — because the informative window is somewhere
specific, or because the order of two events is the label — an aggregate table cannot
express it and this can.

**You want the model to tell you what it found.** `model.feature_importances()` and
`model.dump_splits()` return readable rules: `slope of channel 0 between t=12 and t=40 <=
0.31`. A convolution bank cannot do that.

## Do not use it when

**Your series has no static block and no interaction structure.** Use MiniROCKET, or
`aeon`'s implementation of it. It is simpler, roughly two orders of magnitude faster, and
gives up little — on sixteen UEA datasets it is **ahead of this library on most of them**
(see `RESULTS_V24.md`). That is measured, not modesty.

**A global summary already captures your signal.** Run
`python validation/screen_dataset.py --datasets <yours>` first. If the regime gap comes
back near zero, a fixed aggregate table loses nothing and XGBoost on `agg` features will be
faster and just as good.

**You need speed.** This is single-threaded pure NumPy, roughly 20–100× slower to fit than
XGBoost on pre-aggregated features. A CPSC-2018 cell at n=1000 takes ~22 minutes.

**Your dataset is large.** Everything here is designed and measured at n=100 to n=2000.
Nothing has been tested at a scale where the fixed convolution bank stops fitting in memory
alongside the trees.

## Check first, commit later

Before running a full study on a new dataset, screen it:

```bash
python validation/screen_dataset.py --datasets apnea sleepedf --seeds 3
```

```
apnea      REJECT  static_lift=+0.149 (ok)  exogeneity=+0.252 (FAIL)  regime_gap=-0.010 (FAIL)
sleepedf   REJECT  static_lift=-0.001 (FAIL) exogeneity=+0.000 (ok)   regime_gap=-0.008 (FAIL)
```

Three numbers, minutes rather than days, and no Heartwood model anywhere in the screen — so
it cannot be fitted to the answer it exists to give. It rejects every dataset this project
has ever studied, which is the point: it would have said in an afternoon what four full
studies took weeks to discover.

## A note on tuning

Don't, mostly. The estimator takes 42 parameters and about a quarter of them are switched
off because a pre-registered study failed them — turning one on is re-opening a question
that has an answer. The package docstring lists which, with the study and the number it
failed by. `validation/HEADROOM.md` also measured that a ×16 candidate budget buys +1.5
points on a 20-point gap and makes most datasets worse, so more search is not the lever it
looks like.

The knobs worth touching are the ordinary ones: `n_estimators`, `learning_rate`,
`max_depth`, and `dense_base=True` if your series is shape-regime.
