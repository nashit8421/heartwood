# V22 — magnitude products, and a bound that is not a rank

Written and committed **before any V22 benchmark cell is run.** The mechanism has unit tests
(`tests/test_product_splits.py`, 10 passing). The diagnostic measurements in §2 are exploratory
and are labelled as such; no benchmark number exists.

## 1. Why

V12 diagnosed `amp_regression` (−11.3) exactly and the diagnosis has never been acted on. The
target is `transient_height × static_coefficient` — a product of **magnitudes** — and the only
products this model could form were products of **ranks**, which discard precisely the
information the target is made of.

There was a second half to the problem that V12 did not name. Those rank products were
`static × static`. The cross the target is actually built from is **series × static**, and it
had no representation anywhere in the model. No amount of fixing the currency would have
helped while the operands were wrong.

## 2. Two routes, and an exploratory measurement of what they can reach

**`prod_split`** — a new `product` candidate at the tree level: a banked temporal feature times
a static column, both standardised and both clipped to their training range.

**`prod_margin`** — the base's out-of-fold margin times each static, appended to the columns
the trees see.

**Why this is not V11 again.** V11's collapse (Apnea-ECG 0.856 → 0.478 AUC, below chance) came
from an *unpenalised linear* product column: products grow quadratically, so a subject outside
the training range produced an exploding term. Both routes here feed **trees**, and a tree
cannot extrapolate — its output is a leaf value however large the input. The failure mode
requires a linear extrapolation and there is not one. Clipping to the training range is carried
as well, so the bound does not depend on that argument being right.

**Exploratory, on the diagnostic scenario only** (`amp_regression`, n=500, one seed, RMSE):

| arm | RMSE | vs baseline |
|---|---|---|
| baseline | 0.4569 | — |
| `prod_split` (k=4) | 0.4485 | +1.8% |
| `prod_margin` | 0.4466 | +2.3% |
| `prod_both` | 0.4327 | +5.3% |
| **oracle product column** | **0.4184** | **+8.4%** |

The last row is the number that matters and it is the reason this document is cautious. An
oracle column — the true signal-window maximum times the true coefficient, correlated 0.925
with the target — buys only 8.4%. **So ~8% is the ceiling for this mechanism on this scenario,
and the mechanism recovers about two thirds of it.** Anything reported above that is not this
mechanism working; it is something else, and would need explaining rather than celebrating.

Two things were tried and are **not** shipped, recorded here so they are not silently
re-attempted: drawing bank entries in proportion to cumulative gain rather than uniformly
(+2.0% against +1.8%, and *worse* at sixteen candidates), and raising the candidate count,
which reverses the gain entirely (k=64 scored below baseline). The second is the winner's curse
arriving on schedule and is the reason `n_product_candidates` defaults to 4 in the arms.

These numbers are one scenario and one seed. They set expectations; they decide nothing.

## 3. Hypotheses

Arms: `rocket_static` against `prod_split`, `prod_margin`, `prod_both`.

* **H-V22.1 — the diagnostic improves.** On `amp_regression`, 5 seeds, at least one arm must
  reduce RMSE by **≥ 4%** against the baseline. Below the oracle's 8.4% by design: a bar above
  the ceiling would be unpassable and a bar at it would be a demand for luck.
* **H-V22.2 — Apnea-ECG does not collapse.** **This is a veto.** No arm may score below
  `rocket_static − 2.0` points of AUC on Apnea-ECG at n=1000. V11 failed here and failed
  silently, so this runs whatever H-V22.1 says, and a failure withdraws the change regardless
  of the diagnostic.
* **H-V22.3 — the synthetic suite does not regress.** No scenario in the synthetic set worse
  than baseline by more than 1.0 point / 2% RMSE. `amp_regression` is one of seven scenarios and
  a fix that costs the other six is not a fix.

## 4. What each outcome means

* **All three pass.** Magnitude products ship, on by default for the arm that won, and V12's
  diagnosis is closed with the correction it earned.
* **H-V22.1 passes, H-V22.2 fails.** The V11 failure mode was not what this document claims,
  the tree-cannot-extrapolate argument in §2 is wrong, and that argument is corrected before
  anything else is attempted. This is the outcome that teaches the most.
* **H-V22.1 fails.** Two routes to a magnitude product, neither reaching a 4% bar under an 8.4%
  ceiling, means the operands are still wrong rather than the currency — most likely that the
  bank does not hold a clean amplitude carrier for the signal window at all, which is a
  statement about the *bank* and sends this back to item 1 rather than forward.

## 5. The outcome I would least like

H-V22.1 passing on `prod_margin` alone.

`prod_margin` crosses the base's margin with a static, and the base's margin is already a
prediction of `amplitude × coef`. Multiplying it by `coef` again gives `amplitude × coef²`,
which is not the target and should not help for the stated reason. If it wins anyway, the
mechanism is not the one described here — most likely it is acting as a per-subject rescaling
of the base — and the write-up would have to say so instead of claiming V12's diagnosis was
confirmed.

## 6. Pre-committed analysis

The synthetic suite is scored on its own generators at 5 seeds; Apnea-ECG at n=1000, 5 seeds,
AUC, matching V10's protocol so the comparison is against a number that already exists. No arm,
seed or scenario is dropped. The exploratory table in §2 is not re-reported as a result and does
not count toward any bar.
