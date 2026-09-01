# V24 — the release gate: does what is left still beat MiniROCKET?

Written and committed **before any V24 cell is run.**

## 1. Why this must be run before releasing anything

Every headline number this project has quoted — +3.8 then +3.0 on CPSC-2018, +2.2 on PTB-XL,
+1.8 on Sleep-EDF — was measured on a version of the library that **no longer exists.** Since
then, V15 and V23 removed all four of the things this bank added to MiniROCKET's:

* virtual channels (V15: −0.2 over eight datasets)
* the dyadic window-statistic block (V15: +0.4, and 0.015% of RMSE on the synthetics)
* comparison splits (V15 and V23: 0 of 8 on both suites)
* Lévy areas (V15 and V23: 0 of 8 on both suites)

and V20's no-regret guarantee went with them.

Each deletion was justified by a margin *against our own baseline*, which is the right test for
"does this extra earn its place" and the **wrong** test for "does the result survive." Four
individually-negligible removals can still add up, and nobody has measured the library that
actually remains against the library it claims to beat.

**Releasing on the strength of numbers from deleted code would be the worst thing this project
has done.** It has spent two weeks correcting claims that outran their evidence; shipping one
would undo the point of all of it.

## 2. Design

**Arm:** the shipped default with the convolution base — `rocket_static` — at library defaults.
One arm. This is not an ablation; it is the release candidate.

**Baselines:** MiniROCKET at 2,000 and 10,000 kernels, and the `agg` aggregate baseline for
context. Same splits, same seeds.

**Suites, both of them, because they answer different questions:**

1. **All sixteen UEA datasets** — V15's eight and V23's eight together. Official splits, 5
   seeds. This is breadth, and it is the first time these two halves are scored as one suite.
2. **CPSC-2018, single lead, n=1000, 5 seeds.** This is the headline claim. V14 measured +3.0
   here on the old library, and single-lead is the arm V14 established as the honest one after
   the channel-width explanation failed.

**Nothing is tuned.** No arm is added after a score is seen. If the number is worse than the
old library's, that is the result and it goes in the README.

## 3. Hypotheses

* **H-V24.1 — the library still beats MiniROCKET on breadth.** Mean balanced accuracy over the
  sixteen UEA datasets ≥ MiniROCKET-10k's, and ahead on **≥ 9 of 16** datasets.
* **H-V24.2 — the headline survives.** On single-lead CPSC-2018, `margin ≥ +2.0` points over
  MiniROCKET-10k. The old library measured +3.0; a bar at +2.0 allows the deletions to have
  cost something real without allowing the claim to evaporate.
* **H-V24.3 — no deletion did quiet damage.** On the eight V15 datasets, the current library
  must not score more than **1.0 point** below what `abl_min` scored in V15 — that arm was
  MiniROCKET's plain bank under our trees, which is what the library now *is*. This is a
  direct before/after on the same cells, and it is the check that four small removals did not
  compound.

## 4. What each outcome means

* **All three pass.** The library releases, with numbers measured on the code being released
  and a README that claims exactly them.
* **H-V24.2 fails.** The headline claim is withdrawn and the release either waits or ships with
  a smaller claim — "competitive with MiniROCKET" rather than "beats it". No re-running until
  it passes: that is the rule this project has held to ten times.
* **H-V24.3 fails.** One of the deletions cost more than it appeared to in isolation. The
  release stops and the guilty extra is restored, because a compounding effect is exactly what
  a per-extra bar cannot see.
* **H-V24.1 fails but H-V24.2 passes.** The library is a CPSC/ECG result rather than a general
  time-series result, and the README says so. That is a narrower claim, not a dead one.

## 5. The outcome I would least like

H-V24.2 passing at, say, +2.1 — above the bar, below the +3.0 the README currently quotes.

It would mean the deletions cost about a point of the headline, and I would be releasing a
library whose central number went down while I was improving it. The temptation would be to
quote the old +3.0 because it is "the same architecture". It is not the same architecture, the
old number came from code with four extras in it, and the README will quote whatever V24
measures.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. H-V24.3 compares against
`abl_min` cells already recorded in `validation/rerun/v15_uea/results.json` — those numbers
exist and cannot be re-run to taste.
