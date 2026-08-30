# V20 — never meaningfully worse than its best component

Written and committed **before any V20 cell is run.** The guarantee has unit tests
(`tests/test_no_regret.py`, 12 passing) and no scores.

## 1. Why

This is roadmap item 3, and it is the only item on the list whose value is **not** a benchmark
number. "Never meaningfully worse than its best component" is a property a library either has
or does not have, and it is a stronger thing to be able to say than any single win.

The concrete motivation is a regression this project has already shipped and measured: a
combination scoring below the static block on its own (`static_control`, and the Apnea cell
V10 was written to fix). Every study since has re-measured that class of failure by hand. A
guarantee makes it impossible by construction instead.

Both halves already existed. The ridge base can switch itself off against a permutation null
(`DenseBase._chance_r2`, V6). The regime diagnostic is computed from baselines alone, so it
cannot be fitted to the answer. What was missing was the decision that uses them.

## 2. Design

Three candidates, fitted on the same rows and judged on rows none of them saw:

| candidate | what it is |
|---|---|
| `combined` | the shipped architecture — trees boosted from the ridge's margins |
| `base_only` | the ridge alone, zero trees |
| `trees_only` | the trees alone, no ridge under them |

The hold-out is 25%, stratified for classification — a fold that happened to miss a class
would score the candidates on a different problem than the one being solved. Rare classes are
never emptied out of the fitting half to fill it.

**The asymmetry is the whole design.** `combined` wins ties and wins by default; a component
must beat it by more than `no_regret_margin` on held-out data to displace it. This selection
is itself a selection step and can over-fit like any other — the roadmap says so — and
requiring evidence to *deviate* rather than evidence to stay put is what bounds the damage it
can do. At `margin=0` it is a plain argmin; above that it is an argmin that must clear a bar.

Cost: four fits instead of one — three candidates on 75% of the rows, then the winner refitted
on all of them. That is the price of the guarantee and it is not amortised anywhere.

## 3. Hypotheses

Arms: `noregret` (guarded) and `rocket_static` (unguarded, the same architecture), plus
`comp_base` and `comp_trees` — the two components the bar is measured against.

Suite: the same eight UEA datasets as V15–V19, official splits, 5 seeds.

**V20 is not graded like a sweep.** The claim is a guarantee, so the unit is the cell and a
mean would hide exactly the cell the claim is about.

* **H-V20.1 — the guarantee holds.** PASS if **no cell** — dataset × seed, 40 of them — scores
  more than **0.5 points** below the better of `comp_base` and `comp_trees`. One violation
  falsifies it.
* **H-V20.2 — it costs little where it was not needed.** PASS if the guarded model's mean
  margin against the unguarded one is ≥ −0.5 points. A guarantee that is bought with a general
  loss of accuracy is a bad trade, and the extra selection step is the plausible source of one.
* **H-V20.3 — it fixes something.** PASS if the unguarded model violates the tolerance in at
  least one cell that the guarded one does not. This is the hypothesis most likely to fail, and
  §5 is about that.

## 4. What each outcome means

* **H-V20.1 and H-V20.2 pass, H-V20.3 fails.** The guarantee holds and costs nothing, but this
  suite never needed it. That is a real and reportable outcome: the property is worth having
  for the datasets where the components diverge, and this suite is not one of them. It ships,
  and the write-up says plainly that the evidence for it is the absence of a violation rather
  than the repair of one.
* **H-V20.1 fails.** The fallback picked wrong on held-out data — the selection step over-fitted,
  which is the failure mode the design anticipates. The response is a larger `no_regret_margin`,
  not a larger hold-out, and it is re-run once with the new margin named in advance.
* **H-V20.2 fails.** The guarantee costs accuracy where it was not needed, and the trade has to
  be argued rather than assumed. At that point the honest default is `no_regret=False` with the
  guarantee available to anyone who wants it.

## 5. The outcome I would least like

H-V20.3 failing with **zero** violations anywhere — guarded or unguarded — across all 40 cells.

That would mean V20 proved nothing on this suite: a guarantee that never bound is not a
guarantee that held. It is the likeliest single outcome, because these eight UEA datasets are
all series-dominated, and the failure the guarantee was built for (`static_control`, Apnea) is
a *static*-block failure that this suite structurally cannot produce.

The report is written to say so in those words rather than printing PASS and stopping. And the
right follow-up in that case is not another UEA run: it is Apnea-ECG, the dataset where the
regression actually happened, which is named here in advance as the confirmation for exactly
this outcome.

## 6. Pre-committed analysis

Regrets are computed per cell against the better of the two components in the *same* cell, so
no seed is compared against another seed's component. No arm, seed or dataset is dropped; a
cell that fails to run is reported as failed. The tolerance and the cell-level rule are applied
mechanically by `validation/report_v20.py`, which prints every per-seed regret for both models.
