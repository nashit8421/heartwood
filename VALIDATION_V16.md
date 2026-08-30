# V16 — does per-node bagging beat the winner's curse?

Written and committed **before any V16 cell is run.** The knob has unit tests
(`tests/test_candidate_bagging.py`, 11 passing) and no scores.

## 1. Why

`validation/HEADROOM.md` is the most useful negative result this project has, and it named
the ceiling precisely: *"a node picks the single highest-gain split from a pool of randomly
drawn candidates, which means a bigger pool raises the winner's expected gain whether or not
anything in it is informative."* It then measured the consequence — ×4 and ×16 candidate
budgets bought +1.5 points on a 20-point deficit and made two datasets of three **worse**.

That experiment only ever pushed the knob one way. If the winner's-curse account is right,
the interesting direction is **down**: fewer candidates per node means a maximum taken over a
smaller pool, less over-fitting to the node's own rows, and less of that error inherited by
every subtree below. HEADROOM's own data hints at it — the default already beat ×4 on all
three datasets — but nobody has gone below ×1.

This is roadmap item 2a, and it is first among the four attacks on selection because it is
the cheapest and the most standard: it is `colsample_bynode`, which ordinary gradient
boosting has had for years, applied to the one candidate pool in this library that has no
thinning at all.

## 2. Design

**The knob.** `candidate_colsample` thins the temporal draws — intervals, shapelets, filters,
comparisons — per node. Scope is deliberate and is the thing most likely to be misread later:
the static block and the feature bank **already** have `colsample` and `bank_colsample`, so a
third knob silently multiplying those two would make any measured effect impossible to
attribute. V16 measures the pool that currently has none.

There is a floor of one candidate per enabled source, so a small fraction degrades the search
rather than switching a source off; without it, `×0.125` would be measuring "shapelets
disabled", which is a different experiment.

**Arms.** `rocket_static` (the shipped base, `candidate_colsample=1.0`) against three
fractions named now and never added to: **×0.5, ×0.25, ×0.125**. Each arm differs from the
baseline in exactly one setting.

**Suite.** The same eight UEA datasets as V15 — Epilepsy, Handwriting, RacketSports,
HandMovementDirection, Libras, NATOPS, SelfRegulationSCP2, Heartbeat — official splits, 5
seeds, library defaults otherwise. Same suite as V15 on purpose: it is the suite HEADROOM's
ceiling was measured on, and reusing it means V15 and V16 are comparable without a new
selection step.

Bagging makes nodes cheaper, so this grid is faster than V15's.

## 3. Hypotheses

**H-V16.1 — bagging clears the roadmap's bar.** For at least one of the three pre-named
fractions, `margin ≥ +1.5` points on **≥ 5 of 8** datasets, where the margin is the paired
per-seed mean against `candidate_colsample=1.0`.

The fraction must be **the same on every dataset.** This is the whole methodological point of
V16 and it is worth stating plainly: a study that picks the best fraction per dataset and
averages the winners is not measuring bagging, it is measuring its own search, and
`HEADROOM.md` exists because this project has already been fooled once by exactly that. The
report computes the per-dataset-best number too, labels it *not a result*, and prints the gap
between the two as the tuning optimism it is.

**H-V16.2 — the direction is the predicted one.** PASS if the mean margin is non-negative for
at least one fraction. This is strictly weaker than H-V16.1 and exists so that a real but
sub-bar effect is recorded rather than rounded to "no effect": it would say the winner's-curse
account is right and the remedy is too blunt, which points at 2b (penalise the max by the pool
size) rather than away from the whole line of attack.

## 4. What each outcome means

* **H-V16.1 passes.** The winning fraction becomes the library default, the change is a
  three-character diff, and items 2b–2d are re-scoped to what bagging did not fix.
* **H-V16.1 fails, H-V16.2 passes.** Bagging helps but not enough. The knob **stays in the
  library at its no-op default of 1.0** — it is 20 lines, it is standard, and it is the
  control arm every later selection experiment needs — and the roadmap moves to 2b.
* **Both fail.** Shrinking the pool does not help, which is genuine evidence *against* the
  winner's-curse account of the ceiling and would make 2b and 2d much less promising. That
  would be the most informative outcome of the three and it changes the roadmap the most.
* **Bagging makes things worse monotonically in the fraction.** The pool is too small already
  and the deficit is a search problem after all — which would contradict HEADROOM directly and
  demand that document be revisited rather than this one.

## 5. The outcome I would least like

`×0.5` and `×0.125` failing while `×0.25` passes on exactly five datasets. Three fractions is
already a small search, and a single non-monotone winner on a bare majority is what that
search looks like when it finds noise. If that happens the result does **not** ship on this
evidence: it goes to the same PTB-XL confirmation V15 uses, and it is reported here as having
needed one.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. The bar, the majority rule
and the fraction list are applied mechanically by `validation/report_v16.py` from
`BAGGING_ARMS`, so no fraction can join the study after its score is known.
