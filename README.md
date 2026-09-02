# Heartwood

**Gradient-boosted trees that read the rings.**

Heartwood is the dense inner wood of a tree, and it carries that tree's whole history as
growth rings — a time series recorded inside a tree. That is what this library does:
gradient-boosted trees whose splits can read the trajectory a row carries, not just the
columns beside it.

```python
from heartwood import HeartwoodClassifier

model = HeartwoodClassifier(n_estimators=200, dense_base=True, random_state=0)
model.fit(X_static, X_series, y)          # (n, p) attributes + (n, channels, T) series
model.predict_proba(X_static_test, X_series_test)
```

---

## What it is good at, and what it is not

Both halves are measured, on this code, in [`RESULTS_V24.md`](RESULTS_V24.md).

**On 12-lead and single-lead ECG it beats MiniROCKET.** On CPSC-2018 restricted to lead II —
one channel, n=1000, 5 seeds — balanced accuracy **0.588 against MiniROCKET-10k's 0.555**, a
margin of **+3.3 points**, positive on every seed (+4.0, +2.5, +3.5, +3.2, +3.4).

**On general time-series benchmarks it does not.** Across sixteen UEA datasets it is ahead
on **5 of 16**. Mean balanced accuracy is 0.798 against MiniROCKET-10k's 0.787 — but that
mean is carried by one 22-point win on EthanolConcentration, and the count is the more
honest summary. If your problem looks like a UEA classification task, **use MiniROCKET**: it
is simpler, roughly two orders of magnitude faster, and ahead of this library more often
than not.

| | Heartwood | MiniROCKET-10k |
|---|---|---|
| CPSC-2018, single lead | **0.588** | 0.555 |
| 16 UEA datasets (mean) | 0.798 | 0.787 |
| 16 UEA datasets (won) | 5 | **11** |

That is the whole claim. There is no larger one.

## Why it exists

Real datasets routinely pair per-row attributes with a raw trajectory per row: a patient's
ECG and their BMI, a machine's vibration trace and its model number. The standard workaround
summarises the series into aggregates and hands the result to XGBoost, which discards
exactly what often matters — *when* something happened, what shape it had, how the journey
unfolded.

Heartwood keeps XGBoost's second-order boosting machinery and enlarges what a split may ask
about, so a node can ask *"was the slope between t=12 and t=40 negative?"* or *"does this
shape occur, and how early?"* as readily as *"is the customer over 50?"* — all scored on the
same gain, all learned from the gradients rather than fixed up front.

It also stays readable. `model.dump_splits()` returns rules a human can check:

```
slope of channel 0 between t=12 and t=40 <= 0.31
```

A convolution bank cannot tell you that.

## Install

```bash
pip install heartwood
```

Or from a clone, for development:

```bash
pip install -e ".[test]"    # + pytest
```

Python 3.10+. Pure NumPy, single-threaded, no GPU.

## Documentation

| | |
|---|---|
| [How it works](docs/how-it-works.md) | the split kinds, the convolution bank underneath the trees, and why the base is a ridge |
| [When to use it](docs/when-to-use-it.md) | including when to reach for MiniROCKET instead |
| [How this was validated](docs/how-this-was-validated.md) | the pre-registration rule, and the ten studies that failed their bars |

## The unusual part: how this was validated

Every performance claim here comes from a study **written down and committed before it ran** —
arms fixed, a numeric bar per hypothesis, what each outcome would mean, and a section naming
the outcome the author would least like. A script then applies the bars mechanically. No
verdict in this repository was reached by reading a table.

**Ten studies. Every bar failed.** Five features were deleted as a result:

| deleted | why |
|---|---|
| virtual channels | −0.2 points over eight datasets |
| window-statistic bank | +0.4 points; 0.015% of RMSE on synthetics |
| comparison splits | 0 of 8 on two independent suites |
| Lévy areas | 0 of 8 on two independent suites |
| the no-regret fallback | picked the better component 48% of the time — a coin flip |

Three published claims were withdrawn, including one where the counterexample had been
sitting two paragraphs below it in this README for weeks. The full record — corrections
included, especially the corrections — is in [`RESEARCH_LOG.md`](RESEARCH_LOG.md) and the
`VALIDATION_V*.md` / `RESULTS_V*.md` files.

**What the library is, after all that:** MiniROCKET's kernel bank, with a ridge over it,
under gradient-boosted trees that can also split on static attributes. Every embellishment
this project added to that was measured and most were removed.

## Screen your dataset before you commit to it

```bash
python validation/screen_dataset.py --datasets apnea sleepedf --seeds 3
```

```
apnea      REJECT  static_lift=+0.149 (ok)   exogeneity=+0.252 (FAIL)  regime_gap=-0.010 (FAIL)
sleepedf   REJECT  static_lift=-0.001 (FAIL) exogeneity=+0.000 (ok)    regime_gap=-0.008 (FAIL)
```

Three numbers in minutes: are the static covariates informative on their own, are they
*exogenous* (can the series predict them — age and sex are recoverable from an ECG, which
disqualifies them), and is there temporal structure a fixed summary would lose. It fits no
Heartwood model, so it cannot be tuned to the answer it exists to give.

It rejects every dataset this project ever studied. That is the point.

## Known limitations

- **Behind MiniROCKET on most UEA benchmarks** — 5 of 16. Stated above; repeated here so it
  cannot be missed.
- **Slow.** 20–100× slower to fit than XGBoost on pre-aggregated features. A CPSC-2018 cell
  at n=1000 takes ~22 minutes.
- **Small data.** Everything here is designed and measured between n=100 and n=2000.
- **The founding claim is still unproven.** That a model seeing raw series *and* static
  covariates beats one seeing either alone remains untested after five attempts — not
  disproven, untested, for want of a dataset where both halves are strong and the statics
  are genuinely exogenous. `validation/screen_dataset.py` exists to make attempt six cheaper.
- **Most of the 42 constructor parameters should be left alone.** About a quarter are
  switched off because a pre-registered study failed them; the package docstring lists which,
  with the study and the margin it failed by.
- Single-threaded, pure NumPy, no GPU. Series are padded to a common length.

## Tests

```bash
python -m pytest tests -q     # 374 tests, ~75 s
```

The suite leans on properties rather than snapshots: exact leave-one-group-out is checked
against literally refitting without each group to ~5e-15, and that check has caught two
defects that had already produced confident wrong answers.

## Licence

MIT.
