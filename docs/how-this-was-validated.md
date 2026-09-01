# How this library was validated

Every performance claim here comes from a study that was **written down and committed
before it was run.** That is the whole method, and it is worth explaining because it is
why the numbers in the README are smaller than the ones this project used to quote.

## The rule

Each study is a `VALIDATION_V*.md` file containing, before any cell runs:

1. **What is being tested** and why it is worth the compute.
2. **The arms**, fixed in advance — no arm may be added after a score is seen.
3. **A numeric bar per hypothesis.** Not "does it help" but "≥ +1.5 points on a majority of
   eight datasets".
4. **What each outcome means**, including the failures — written while the answer is still
   unknown, so the interpretation cannot be chosen to fit the result.
5. **The outcome I would least like.** This section exists specifically to name the
   flattering-but-wrong result in advance, so it cannot quietly become the headline.

Then the study runs, and a `report_v*.py` applies the bars **mechanically**. No verdict in
this repository was reached by reading a table.

## The scoreboard

Ten studies. Every bar failed.

| study | question | verdict |
|---|---|---|
| V14 | does channel width cause the MiniROCKET margin? | **FAIL** — 12 leads scored *worse* than 1 |
| V15 | are the bank's four extras worth their place? | **FAIL** — all four |
| V16 | does per-node bagging beat the winner's curse? | **FAIL** — 1 of 8 |
| V17 | does an analytic multiple-comparisons charge? | **FAIL** — 1 of 8 |
| V18 | does pre-screening the feature bank? | **FAIL** — 1 of 8 |
| V19 | does a recalibrated permutation null? | **FAIL** — 1 of 8 |
| V20 | can the model guarantee it is never worse than its best half? | **FAIL** — the guarantee cost 1.6 points |
| V21 | does a nonlinear base help? | **FAIL** — 0 of 8 at every width |
| V22 | do magnitude products fix `amp_regression`? | **FAIL** — +2.9% against a 4% bar |
| V23 | do comparison splits and Lévy areas work on real data? | **FAIL** — 0 of 8, twice |
| V24 | **does what remains still beat MiniROCKET?** | see `RESULTS_V24.md` |

Five features were deleted as a result. The library is smaller than it was a fortnight ago
and its claims are narrower.

## Three corrections worth reading

**`validation/HEADROOM.md`.** Its central claim — *"the bottleneck is the selection rule,
not the sampler feeding it"* — was the premise four studies were built on. All four failed.
The document now carries a correction saying so. A hypothesis stated confidently in an
exploratory write-up is still a hypothesis.

**The channel-width explanation.** The README claimed our margin over MiniROCKET came from
cross-channel machinery, and that Apnea-ECG's loss followed from being single-lead. V14
held the dataset, task, split, seed and label fixed and varied only lead count: 12 leads
+2.8, 3 leads +2.6, **1 lead +3.0**. The explanation was wrong, and a counterexample —
single-channel Sleep-EDF, which we win — had been sitting two paragraphs below it the whole
time.

**V20's own bar.** The study judged a model per cell against the per-cell *maximum* of two
alternatives. The maximum of two noisy estimates is biased upward, so the rule penalised
whoever was under test. Applying the identical rule to the components proved it: the ridge
alone violated as often as the combination. The pre-registered verdict is kept in
`RESULTS_V20.md` **with the evidence against it printed underneath**, because it was
pre-registered and deleting it would be the very thing this method exists to prevent.

## The dataset screen

`validation/screen_dataset.py` answers, in minutes, whether a dataset can test a claim at
all — before committing days of compute:

1. **Are the statics informative on their own?**
2. **Are they exogenous** — can the series predict them? (Age and sex are recoverable from
   an ECG, which disqualifies them.)
3. **Is the regime temporal** — does a fixed summary lose anything?

It rejects **every dataset this project ever ran a study on.** It reproduced Sleep-EDF's
known verdict — statics at chance — in about a minute, having previously cost a full study.

Building it surfaced the row-wise/group-wise confusion for the *third* time in this
project: a random row split let the model recover a subject-constant static by recognising
whose ECG it was, reading Apnea's exogeneity as 0.82 instead of 0.25. The tool built to
prevent bad studies contained the defect it was built to prevent.

## What is still unproven

The founding claim — that a model seeing raw series **and** static covariates beats one
seeing either alone — **remains unproven after five attempts.** Not disproven: untested, for
want of a dataset where both halves are strong and the statics are genuinely exogenous.
That is the honest state, and `validation/screen_dataset.py` exists to make the sixth
attempt cheaper than the first five.
