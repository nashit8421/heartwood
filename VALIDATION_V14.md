# V14 — does channel width cause the MiniROCKET margin?

Written and committed **before any V14 cell is run.** No V14 number exists at the time of
writing; the ablation flag has unit tests and no scores.

## 1. Why

Four datasets have now compared this library's series half against MiniROCKET with the static
block dropped:

| dataset | channels | ours | MiniROCKET-10k | margin |
|---|---|---|---|---|
| CPSC-2018 | 12 | 0.692 | 0.654 | **+3.8** |
| PTB-XL | 12 | 0.533 | 0.511 | **+2.2** |
| Sleep-EDF | 1 | 0.678 | 0.660 | **+1.8** |
| Apnea-ECG | 1 | 0.694 | 0.724 | **−3.0** |

After V13 I told the user the advantage looked multi-channel. **That statement was wrong on
the evidence I already had:** Sleep-EDF is single-channel (EEG Fpz-Cz) and we win there. I had
it filed as multi-channel and never checked. The ordering above is consistent with width
mattering, but Sleep-EDF breaks any clean story, and with four datasets and four different
tasks, "width" is confounded with everything else that differs between them.

This library's bank contains machinery MiniROCKET's does not — virtual channels, comparison
splits, cross-channel structure — all of which is inert on one channel while still costing its
share of the fitting budget. That is a *mechanism* for width mattering. It is not evidence.

The only way to ask the question is to hold the dataset, the task, the split, the seed and the
label fixed, and vary nothing but the number of leads.

## 2. Design

**Dataset:** CPSC-2018, n=1000, 5 seeds, `--drop-static` (the statics are age and sex, which
an ECG encodes, so they are disqualified under the V9 rule and would only add noise here).

**Arms**, selected by lead *name*:

| arm | leads |
|---|---|
| A | all 12: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 |
| B | 3: I, II, V2 |
| C | 1: II |

Lead II is named in advance as the single-lead arm because it is the standard rhythm lead and
the one a single-lead recorder captures — not because it scores best. The 3-lead subset is
named in advance as one limb pair plus one precordial lead. **Neither subset may be changed
after a score is seen.**

Every arm is compared against MiniROCKET at 2,000 and 10,000 kernels fitted on *the same
restricted series*, so the baseline degrades with width exactly as we do and the margin is the
only quantity read.

Metric: balanced accuracy, the CPSC headline fixed in VALIDATION_V5.md.

## 3. Hypotheses

**H-V14.1 — width causes the margin.** PASS if `margin(A) − margin(C) ≥ 3.0` points and the
sign of that difference is consistent on at least 4 of 5 seeds.

**H-V14.2 — dose response.** PASS if `margin(A) ≥ margin(B) ≥ margin(C)` in the means. This is
strictly weaker than H-V14.1 and can pass when it fails; a monotone trend with a small total
range would say width matters but explains little.

**H-V14.3 — the margin survives at one channel.** PASS if `margin(C) > 0`. This is the
*control*, and it is the outcome that would convict Apnea rather than width: if we still beat
MiniROCKET on single-lead CPSC, then one channel is not the problem and Apnea's loss is about
Apnea — most likely that a one-minute single-lead ECG segment simply carries little apnea
signal, which the `agg` baseline at 0.673 already hints at.

## 4. What each outcome means

* **H-V14.1 passes.** Width is the mechanism. The next work is making the bank degrade
  gracefully on narrow input — the cross-channel features should cost nothing when there is
  nothing to cross — and the library gets an honest scope statement: strongest on
  multi-channel series.
* **H-V14.1 fails, H-V14.3 passes.** Width is not the mechanism and my V13 write-up needs
  correcting a second time. Apnea's loss is dataset-specific and the search moves to what
  distinguishes it.
* **H-V14.1 fails, H-V14.3 fails.** We lose at one channel on CPSC too, but by less than the
  3 points the bar demands. Width matters, weakly, and something else matters as well.

## 5. The outcome I would least like

H-V14.3 passing. It would mean I published a mechanism in `RESULTS_V13.md` and in the README
on the strength of a correlation across four confounded datasets, having already overlooked a
counterexample sitting in my own results directory. This document exists so that outcome is
recorded rather than quietly dropped, and the README claim is written as a hypothesis pending
this test — not as a finding.

## 6. Pre-committed analysis

Margins are computed per seed and paired within seed. Means are reported with the full
per-seed vector, never alone. No arm is dropped, no seed is dropped, and if a cell fails to
run it is reported as failed rather than omitted.
