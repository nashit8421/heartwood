# Validation Plan V9 — the founding claim, on statics a sensor cannot infer

**Status: PRE-REGISTERED.** Written and committed before the dataset was fetched or a
loader written. Sixth study here.

---

## 1. Why the previous four could not answer this

This library exists to combine **per-row static facts** with **raw time series**. V7 isolated
the static block for the first time and found it contributes **nothing**: −0.1 points on
CPSC-2018 and +0.0 on Sleep-EDF.

That is a real result but it is not a fair test of the premise, and the reason is visible in
the data:

| dataset | static signal alone | series regime | why it cannot test the claim |
|---|---|---|---|
| ICU | strong (+10.2) | summary | aggregation already suffices; nothing for the series machinery to add |
| Credit | strong (+7.3) | summary | six monthly numbers *are* their own summary |
| PTB-XL | weak (+4.1) | shape | age is partly readable **off the ECG itself** |
| CPSC-2018 | weak (+6.8) | shape | same — statics carried signal but added nothing, i.e. redundant |
| Sleep-EDF | **none** (+0.2) | shape | age and sex do not predict a sleep stage at all |

The pattern is the problem. Strong statics only ever appeared beside series a plain average
already captures. Rich series only ever came with statics that were weak, or that the
signal already encodes. **Age next to an ECG was never a fair test**, and I chose PTB-XL and
CPSC because their demographics were conveniently in the file rather than because they were
exogenous. That was a selection mistake, not a finding.

## 2. What "exogenous" means here, fixed before looking

A static column qualifies only if it is **a physical or administrative fact about the
subject that the recorded signal does not encode**. The operative test: could a good enough
model reconstruct this column from the series alone? If plausibly yes, it is redundant and
cannot test the premise.

* Age from an ECG — **plausibly yes**. Disqualified.
* Body weight from a one-minute ECG trace — **no**. Qualifies.

## 3. Dataset (locked)

**PhysioNet Apnea-ECG** (CinC Challenge 2000). 35 released recordings with per-minute
expert apnea annotations, and — in `additional-information.txt` — **age, sex, height and
weight** per subject.

Body-mass index is the primary risk factor for obstructive sleep apnea and spans **19.2 to
41.7** across these subjects, weight **56–135 kg**. None of that is present in a one-minute
single-lead ECG. This is the combination the previous five datasets never offered.

Decisions, fixed here:

* **Series**: the single ECG lead, one minute per row, 100 Hz decimated to 50 Hz by
  averaging pairs (T=3000). Apnea shows up in the ECG as respiratory modulation near
  0.2–0.3 Hz and in heart-rate variability, both comfortably preserved at 50 Hz.
* **Target**: the per-minute annotation, apnea vs non-apnea. Binary and imbalanced, so the
  headline is **ROC-AUC**, per `VALIDATION.md` §5.
* **Static block**: age, sex, height, weight, **and BMI**. BMI is included deliberately and
  is stated here rather than discovered later: it is *the* established risk factor, and a
  tree cannot form a ratio of two columns by splitting on them. Excluding it would
  handicap the arm under test as surely as inventing features would flatter it.
* **Split**: subject-disjoint, 70/30, redrawn per seed. Statics are constant within a
  subject by construction.

## 4. Hypotheses — frozen, with numeric pass/fail

**Precondition, reported either way**: `static_only` must beat 0.500 AUC by ≥2 points. If
the statics carry nothing, H-V9.1 is void rather than failed.

**H-V9.1 — the founding claim.** At n=1000, the model *with* the static block beats the same
model *without* it by **≥ 2 points** of AUC.
*Fails if* D − C ≤ 0. Five studies in, this is the first fair test of the sentence on the
tin, and it is the only hypothesis here that matters.

**H-V9.2 — the series still matters.** Heartwood beats the `agg` aggregate workaround by
**≥ 5 points**. *Fails if* under 2. Guards against the reverse degenerate outcome, where
the statics do all the work and the series machinery is irrelevant.

**H-V9.3 — still competitive.** Heartwood lands within **2 points** of the better `aeon`
MiniROCKET, which sees the series only.

**Both directions are reported**, not just the flattering one: D − C is what the statics add
to the series, and D − `static_only` is what the series adds to the statics. A win that is
really "BMI predicts apnea and the ECG is decoration" would show up in the second number,
and would be reported as such.

## 5. Rules and scope

`VALIDATION.md` §2 in force. No tuning, matched budgets, splits fixed by seed, every arm
reported, comparison against `aeon` rather than our own bank.

Runs: **n=1000, 3 seeds, arms C and D**, plus `agg`, `static_only` and both MiniROCKET
budgets. Larger sizes are not run — T=3000 makes a cell expensive, and n=1000 is where every
previous decomposition was measured, so it is where this one is measured too. Anything not
run is reported as not run.

## 6. What counts as done

If H-V9.1 passes, the premise this library was built on is supported for the first time, on
statics chosen by a rule written before the data was seen. If it fails, then across six
studies and six datasets the static block has never earned its place, and the honest
conclusion is that this is an excellent time-series classifier carrying a claim it cannot
support — which goes in the README, in those words.
