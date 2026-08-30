# Dataset screen — first run

Roadmap item 6. Produced by `validation/screen_dataset.py`, which fits **no Heartwood model**:
every number comes from the same baselines the studies are graded against, so the gate cannot
be fitted to the answer it exists to give.

Thresholds: statics informative if `static_lift ≥ 0.03`; exogenous if the series predicts them
at `R² < 0.25`; temporal regime if a finer representation beats a global summary by `≥ 0.05`.
Each metric is the mean of 3 splits.

## Results

| dataset | static lift | exogeneity | regime gap | verdict |
|---|---|---|---|---|
| Sleep-EDF | **−0.001** | +0.000 | −0.008 | **REJECT** — statics at chance |
| Apnea-ECG | +0.149 | **+0.252** | **−0.010** | **REJECT** — statics partly recoverable; tabular regime |
| PhysioNet ICU | **+0.011** | **+0.759** | **−0.022** | **REJECT** — all three |
| credit | **+0.003** | **+0.398** | **+0.005** | **REJECT** — all three |
| HAR | **−0.001** | **+0.281** | **−0.015** | **REJECT** — all three |
| **PAMAP2** (new, item 7) | **+0.005** | **+0.000** | **−0.078** | **REJECT** — statics irrelevant to the target |

Every dataset this project has ever run a study on is rejected by the screen, and so is the
first new candidate. That is the finding, not a bug in the tool: it is the same fact the
roadmap states — *the founding claim is still unproven after four attempts* — arriving in
minutes instead of weeks.

## What this confirms

**Sleep-EDF's statics score −0.001 against chance.** That is the verdict the roadmap said this
screen should have produced in an afternoon, reproduced in about a minute. Sleep-EDF could
never have tested the founding claim, and a full study was run on it anyway.

## What it found that was not expected

**Apnea-ECG's exogeneity is 0.252, against a ceiling of 0.25.** Apnea is the dataset
`VALIDATION_V9.md` calls the first fair test of this library's premise, on the grounds that BMI
"is not present in a one-minute single-lead ECG at any resolution". The screen puts age, sex,
height, weight and BMI as a block at R² 0.252 from that ECG — recoverable enough to sit exactly
on the line.

This does **not** overturn V9. The screen tests the static block jointly and reports its worst
column, and age and sex being partly readable from an ECG is precisely the V9 rule working as
intended; BMI specifically may still be exogenous. But it does mean the premise was assumed
rather than measured, and a 0.002 margin is not a margin.

**Apnea's regime gap is −0.010.** A global summary loses nothing a finer representation
recovers. That is consistent with Apnea being the one dataset where MiniROCKET beat this
library (−3.0), and it suggests that result was about the dataset rather than about channel
width — the question V14 is running to answer.

## The defect this run found in the screen itself

The first version split rows at random. On Apnea that read **exogeneity 0.821** and would have
disqualified the dataset outright.

That number was subject recall, not exogeneity. Apnea's statics are constant within a subject
and one-minute ECG segments identify their subject easily, so a row-wise hold-out lets the
model recover the static by recognising whose ECG it is. **This is the third time this project
has shipped the row-wise/group-wise confusion** — V12 and V13 were both defects of exactly this
shape — and the first version of the tool built to prevent bad studies contained it.

`_split` now splits by group wherever a dataset has them, and
`tests/test_screen_dataset.py` pins the failure: a static drawn independently of the series but
constant within a subject must not be called endogenous.

## Item 7: the first candidate screened, and what it cost

**PAMAP2** was chosen because the roadmap names its shape exactly — wearable accelerometry
where "body weight is genuinely exogenous to a step signal". Nine subjects, three IMUs, twelve
activities, and a subject table of sex, age, height, weight, resting and maximum heart rate.
Fetcher and loader written, 6,691 windows parsed, screened. **About an hour, against the days
a study would have cost.**

The verdict splits in a way worth reading carefully:

* **Exogeneity 0.000 — the roadmap's premise is confirmed.** Body measurements are *not*
  recoverable from an accelerometer trace, unlike age and sex from an ECG. These are the first
  genuinely exogenous statics this project has had.
* **Static lift +0.005 — and the target is wrong.** Knowing someone's height and weight tells
  you almost nothing about *which activity they are performing*. The covariates are right and
  the label is not.
* **Regime gap −0.078.** A global summary beats every finer representation. Mean and standard
  deviation of acceleration separate lying from running; there is no temporal structure here
  for the architecture to find.

The obvious next move is to keep PAMAP2 and change the target to something body size predicts.
**That is a search, and it is named as one here before it is run.** The dataset was chosen for
its covariates; picking a target afterwards because it makes those covariates look useful is
how a study gets built to confirm itself. Any such target must be named in a pre-registration
before it is scored, and the regime gap of −0.078 is a separate obstacle it would not fix.

The heart-rate channel was excluded from the series before screening, deliberately: two of the
six statics are resting and maximum heart rate, and a series carrying heart rate would make
them endogenous by construction. That decision is in the loader docstring, made before any
number existed.

## Honest limitations

* **The thresholds are calibrated on four known answers.** They reproduce Sleep-EDF's verdict,
  which is the point, but a screen tuned on four cases has four degrees of freedom. Apnea
  landing 0.002 the wrong side of a line is the visible cost of that.
* **Criterion 1 rejects interaction-only datasets.** A static flag XORed with a series feature
  has no marginal static signal, so the screen refuses it — even though that is arguably the
  purest possible test of the founding claim. A rejection on criterion 1 alone deserves a look
  by hand. This is pinned as a test rather than left as a caveat.
* **CPSC-2018 and PTB-XL are not screened here.** V14 is using the machine. They are the next
  two to run, and CPSC is the second calibration case — the roadmap predicts it fails on
  exogeneity, and that prediction is on the record before the number exists.
