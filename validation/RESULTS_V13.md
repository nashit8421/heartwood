# V13 results — the verdict holds, the number that produced it was partly my own bug

## What V13 was

V12 made the ridge base *validate* itself by leaving out whole subjects, because that is how
the benchmark splits Apnea. It left the base's **penalty search** row-wise. So the base was
still choosing λ by asking "how well would I predict an unseen minute?" and then being graded
on "how well do you predict an unseen subject" — tuned for one question, judged on another.

The consequence was measurable and bad: λ = 232, effective dof = 441, leave-one-group-out R²
**negative**, and the base declined to exist on 2 of 3 seeds. V12's 0.826 was produced by a
base that had been mistuned by my own half-finished change.

V13 is one fix — the search uses the same criterion as the judgement — and a rerun of the
same three cells. λ moved to 7.35e3 / 7.5e6, dof to 81 / 1 / 1, R² to +0.139 / +0.115 /
+0.160, and the base is accepted on every seed.

## Apnea-ECG, n=1000, 5 seeds, ROC-AUC

| | seeds | mean |
|---|---|---|
| V10 — blind eval, blind tuning | 0.785 0.906 0.905 0.887 0.797 | 0.856 |
| V12 — honest eval, blind tuning | 0.722 0.910 0.918 0.863 0.717 | 0.826 |
| **V13 — honest eval, honest tuning** | 0.738 0.901 0.938 0.885 0.747 | **0.842** |
| `static_only` (age, sex, height, weight, BMI) | 0.744 0.893 0.867 0.874 0.797 | **0.835** |
| MiniROCKET (10k kernels) | 0.716 0.855 0.826 0.778 0.757 | 0.786 |
| MiniROCKET (2k kernels) | 0.668 0.887 0.839 0.782 0.772 | 0.790 |
| `agg` | 0.739 0.828 0.872 0.891 0.742 | 0.814 |

**vs `static_only`: +0.7 points** (−0.6, +0.8, +7.1, +1.1, −5.0). The pre-registered bar was
+2, and two of five seeds are negative. **The combination still does not beat its best half.**

**V12's verdict stands. The magnitude V12 attributed to it does not.** V10's inflation was
about **1.4 points**, not the 3.0 recorded in `RESULTS_V12.md` — roughly half of what I
charged to blind validation was my own mistuning. The withdrawal of the V10 claim is
unaffected: 0.842 against 0.835 fails the same bar 0.826 did.

## What the statics are actually worth (arm D − arm C, paired)

Arm C is the identical model with the static block dropped, same seeds, same splits.

| | seeds | mean |
|---|---|---|
| D — with statics | 0.738 0.901 0.938 0.885 0.747 | **0.842** |
| C — series only | 0.611 0.841 0.889 0.733 0.677 | **0.750** |
| paired difference | +12.8 +6.0 +4.8 +15.2 +7.0 | **+9.2** |

The fusion mechanism works, and this is the clearest measurement of it in the project: the
static block is worth **+9.2 points**, positive on all five seeds. V7 measured the same
quantity at −0.1 and +0.0 on datasets whose statics the signal could encode.

The reason the headline is still +0.7 is not that the statics fail to contribute. It is that
**our series half is weak on this dataset** and the statics spend their +9.2 climbing back to
roughly where the statics started.

## The finding that is new here

| Apnea, series only | mean |
|---|---|
| Heartwood, statics dropped | **0.750** |
| MiniROCKET (2k) | 0.790 |
| MiniROCKET (10k) | 0.786 |

**−3.6 against MiniROCKET, behind on 4 of 5 seeds.** This is the first dataset in the project
where the time-series half loses. On CPSC-2018 the same code is +3.7 and +4.1.

The difference between those two runs is channel count: CPSC is 12-lead, Apnea is single-lead.
Everything that distinguishes this library's bank from MiniROCKET's — virtual channels,
comparison splits, cross-channel structure — has nothing to work with on one channel, while
the per-kernel cost of the trees stays the same. That is a hypothesis, not a result, and it is
the obvious thing to pre-register next.

## Sleep-EDF re-check, n=1000, 3 seeds, balanced accuracy (5 classes, chance = 0.200)

The V7 Sleep-EDF result was produced under row-wise validation. This re-runs it group-aware.

| | seeds | mean |
|---|---|---|
| **Heartwood** | 0.680 0.667 0.668 | **0.672** |
| MiniROCKET (10k) | 0.657 0.661 0.661 | 0.660 |
| MiniROCKET (2k) | 0.642 0.648 0.645 | 0.645 |
| `agg` | 0.456 0.425 0.437 | 0.439 |
| `static_only` | 0.206 0.200 0.200 | **0.202 — chance** |

**The result survives, smaller than published.** +2.7 over MiniROCKET-2k (all three seeds),
+1.2 over MiniROCKET-10k (all three seeds). V7 reported +2.4 against the 2k setting; the
honest figure against the stronger setting is +1.2.

`static_only` is 0.202 against a chance floor of 0.200, so this is a **pure series result** —
nothing here speaks to the fusion question either way, which is exactly why it is the cleanest
evidence in the project that the time-series half is genuinely good.

## Where this leaves the record

* The time-series classifier is real and now replicated under honest validation on two
  datasets — CPSC-2018 (+3.7/+4.1) and Sleep-EDF (+1.2/+2.7).
* It is **not uniformly better than MiniROCKET**. On single-channel ECG it is −3.6. The
  advantage appears to be multi-channel, and that is now the most valuable open question.
* The static block, given exogenous statics and a base that can see them, is worth **+9.2
  points** — the founding mechanism does work.
* **The founding claim is still unproven.** Four attempts. Beating either half alone requires
  both halves to be strong on the same dataset, and no dataset yet has given us that: Apnea
  has the statics and a weak series, Sleep-EDF has the series and no statics.

That last line is the whole project in one sentence, and it is a statement about the datasets
as much as about the code.

## Defects found in this study

One, mine: the penalty search and the acceptance test asking different questions (fixed in
`30caef9`). It is the fourth measurement defect in the project and the third of mine, and like
the others it was caught by a check written before the result it overturned — here, logging
the base's chosen λ and dof, which made an effective dof of 441 out of 1000 rows impossible
to read as anything but a mistuned fit.
