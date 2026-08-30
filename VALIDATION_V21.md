# V21 — a nonlinear base that keeps its exact hold-out

Written and committed **before any V21 cell is run.** The map has unit tests
(`tests/test_nonlinear_base.py`, 11 passing) and no scores.

## 1. Why, and what is being refused

The base should be nonlinear. It should **not** become LightGBM, and roadmap item 4 is as much
about the second clause as the first.

A gradient-boosted base would model more. It would also make the tree layer redundant —
boosting stacked on boosting — and it would destroy closed-form leave-one-group-out. That
machinery is not a convenience: it is what caught the V12 and V13 defects, and what lets a
block hold-out be verified against literal refits. This project has needed the ability to
detect its own mistakes four times. Trading it for a modelling gain is a bad trade at any
price.

Random Fourier features get the nonlinearity without the trade. `sqrt(2/D)·cos(Wx + b)` with
Gaussian `W` and uniform `b` approximates an RBF kernel, so the ridge on top is nonlinear in
the bank while remaining **linear in what it fits**.

## 2. Design

The map lives inside `DenseBase._prepare`. Everything downstream — the SVD, the lambda search,
`_leave_out_margins` — then sees an ordinary design matrix, and the exactness is preserved by
construction rather than by care.

**Measured, before any arm was chosen.** Leave-one-group-out against literal refits that
actually hold each group out:

| random features | max drift from a refit |
|---|---|
| 0 (linear) | 5.1e-15 |
| 24 | 7.1e-15 |
| 128 | 9.8e-15 |

That is the bar item 4 set for itself — ~1e-14 — and it is met before a single score exists.
`tests/test_nonlinear_base.py` re-runs the check, so it cannot silently stop being true.

**The linear block is kept alongside the random one.** The nonlinear base therefore *contains*
the linear one: the ridge can shrink the random block away and recover it. Without that, a win
could not be attributed to curvature — the arm would be a different model rather than a richer
one — and a loss could not be attributed to anything at all.

**Bandwidth is set from the design's own width.** Columns are standardised, so `E‖x − x'‖²` is
about `2d` and `gamma = 1/(2d)` puts the kernel argument near 1 whatever the bank contains.
A test checks that the mapped inner products actually converge on `exp(−gamma‖x − x'‖²)`, so
the construction is verified rather than cited.

## 3. Hypotheses

Arms: `rocket_static` (linear) against widths **D = 256, 1024, 4096**, named now. Only the
width is swept; sweeping bandwidth as well would be a two-dimensional search dressed as a
study, and the bandwidth is the parameter with a principled default.

Suite: the same eight UEA datasets as V15–V20, official splits, 5 seeds.

* **H-V21.1 — the nonlinear base clears the roadmap's bar.** For at least one pre-named width,
  `margin ≥ +1.0` points on **≥ 5 of 8** datasets against the linear ridge, same width
  everywhere.
* **H-V21.2 — the exact hold-out still holds.** PASS if the refit check in
  `tests/test_nonlinear_base.py` passes at every shipped width. **This is a veto, not a
  contribution:** if it fails, the base does not ship whatever H-V21.1 says.
* **H-V21.3 — width buys accuracy monotonically.** PASS if the mean margin is non-decreasing in
  D. A better kernel approximation should not be worse; if D=256 beats D=4096 then the gain is
  not the kernel and the arm is doing something else — most likely acting as extra
  regularisation, which `reg_lambda` would do more cheaply.

## 4. What each outcome means

* **H-V21.1 and H-V21.2 pass.** The base becomes nonlinear and this project keeps the tool that
  finds its own bugs. That is the whole thesis of item 4, and it would be the first Tier 2
  result.
* **H-V21.1 fails.** The map stays in the library at its no-op default of 0. A null result here
  is informative rather than empty: it would say the bank's ~10,000 dilated-convolution
  responses are already rich enough that curvature on top of them adds nothing, which is a real
  statement about why MiniROCKET-style banks work and one worth writing down.
* **H-V21.3 fails while H-V21.1 passes.** Reported as a regularisation effect, not a
  nonlinearity result, and the honest comparison then becomes a `reg_lambda` sweep rather than
  a wider map.

## 5. The outcome I would least like

H-V21.1 passing at D=4096 only, on a suite whose training splits are 137 to 204 rows.

At that width the random block alone has twenty times more columns than the largest of these
datasets has rows. The ridge is then interpolating in a space where the lambda search is doing
all the work, and "nonlinear base" would be the wrong name for what improved. §3's H-V21.3
would still pass in that case — monotone in D — which is exactly why it is not sufficient on
its own, and why any such result goes to PTB-XL at n=500 before it ships.

**This is also why V21 must run after V15.** The bank's composition is still unsettled until
V15 reports, and measuring curvature on top of a bank we may be about to delete a third of
would answer a question about a model that no longer exists.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. The bar, the majority rule
and the width list are applied mechanically by `validation/report_v21.py` from
`NONLINEAR_ARMS`. H-V21.2 is read from the test suite, not from the results file.
