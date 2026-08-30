# V15 — what are our bank's extras actually worth?

Written and committed **before any V15 cell is run.** No V15 number exists at the time of
writing; the ablation flags have unit tests (`tests/test_bank_ablation.py`, 21 passing) and
no scores.

## 1. Why

`heartwood.rocket` reproduces MiniROCKET's kernel bank exactly — 84 length-9 kernels,
exponentially spaced dilations, per-(kernel, dilation) bias quantiles, proportion-of-positive
pooling — and then adds machinery MiniROCKET does not have:

| extra | where it lives | what it is |
|---|---|---|
| virtual channels | `rocket._channel_groups` | random unions of channels, summed, on top of the singletons |
| comparison splits | `tree._comparison_candidates` | splits comparing two banked position features |
| interval statistics | `dense.dense_bank` | every statistic over every dyadic window, appended to the bank |
| Lévy areas | `dense.levy_area_columns` | signed areas between channel pairs — who moves first |

**None of the four has ever been measured on its own.** They were each added against a
specific failure (the `_channel_groups` docstring names Handwriting's joint x-y trajectory
and an 11-point loss), and then never revisited. The V7 decomposition is the reason this is
now the top item: it attributed +0.0 and +0.1 to the bank against +3.8 and +1.2 to the
trees, which is a direct hint that the extras contribute nothing and are being carried for
free.

Every one of them costs. Virtual channels spend group diversity that could have gone to
singleton coverage; interval statistics widen the ridge's design matrix; comparison splits
and Lévy areas add candidates to a greedy selector that `HEADROOM.md` already identified as
the ceiling, where *every extra candidate is another chance for noise to win the best-gain
contest*.

## 2. Design

**Arms.** One baseline plus one arm per extra plus one all-on arm. `abl_min` is MiniROCKET's
bank and nothing else; each single-extra arm adds exactly one thing back. The arm table is
`VARIANTS` in `validation/run_validation.py`, and a test asserts that each `abl_*` arm
differs from `abl_min` in exactly one setting — so an arm cannot quietly change two things.

| arm | channel groups | comparison splits | dense bank | Lévy areas |
|---|---|---|---|---|
| `abl_min` | singletons | 0 | rocket | off |
| `abl_vchan` | **subsets** | 0 | rocket | off |
| `abl_cmp` | singletons | **4** | rocket | off |
| `abl_stats` | singletons | 0 | **both** | off |
| `abl_levy` | singletons | 0 | rocket | **on** |
| `abl_all` | subsets | 4 | both | on |

The ablation does **not** change the feature budget: a channel group is assigned per kernel
per dilation rather than multiplying the bank, so `abl_min` and `abl_vchan` produce the same
number of columns. This is asserted in
`test_the_ablation_does_not_change_the_feature_budget`, because if dropping the unions also
shrank the bank then `abl_min` would be a smaller *and* less diverse model and the arm would
isolate nothing.

**Suite.** The eight UEA multivariate datasets of V6b Arm B — Epilepsy, Handwriting,
RacketSports, HandMovementDirection, Libras, NATOPS, SelfRegulationSCP2, Heartbeat — official
splits, 5 seeds, 200 rounds, depth 4, library defaults otherwise.

That list is **taken from an existing locked run and not chosen now**, which is the point:
picking a suite after four studies' worth of knowing which datasets flatter us would be a
selection step in itself. All eight are multivariate (2 to 61 channels), so all four extras
are live on every dataset and no arm collapses onto the baseline by construction.

**Why not the physiological four.** CPSC-2018, PTB-XL, Sleep-EDF and Apnea-ECG cost ~2,400 s
per cell at n=1000; six arms × five seeds × four datasets is ~80 hours, not the one night the
roadmap budgeted. The UEA suite costs ~380 s per (seed, arm) across all eight, so the whole
grid is ~3-4 hours. The trade is being made in the open: **V15 answers the question on eight
datasets we did not pick today, at the price of not answering it on the four the recent
studies used.** Any extra that survives is re-tested on PTB-XL at n=500 before it is called
established (§6), and any extra that fails is deleted on this evidence.

**Metric.** Balanced accuracy, the classification headline fixed in `VALIDATION_V5.md`.
Margins are per-extra against `abl_min`, paired within seed.

## 3. Hypotheses

For each extra `E`, with `margin(E, d) = mean_seeds[ score(arm_E, d) − score(abl_min, d) ]`
in balanced-accuracy points:

* **H-V15.1 — virtual channels earn their place.** PASS if `margin ≥ +0.5` on **≥ 5 of 8**
  datasets.
* **H-V15.2 — comparison splits earn their place.** Same bar.
* **H-V15.3 — interval statistics earn their place.** Same bar.
* **H-V15.4 — Lévy areas earn their place.** Same bar.
* **H-V15.5 — the extras are additive.** PASS if `margin(abl_all) ≥ Σ margin(E)` − 1.0 point
  on ≥ 5 of 8. This is the check on the whole design: if the extras interact strongly then
  measuring them one at a time answers a different question than the one asked, and that has
  to be said rather than discovered later.

No multiplicity correction is applied, and the reason is that the bar is a **fixed effect
size on a mean**, not a p-value — there is no null to inflate. What four tests do buy is four
chances for a +0.5 to be seed noise, which is what H-V15.5 and §6 exist to catch.

## 4. What each outcome means

* **An extra passes.** It stays, and we finally know where our edge lives. It then goes to
  the §6 confirmation before any README claim is made about it.
* **An extra fails.** **The code is deleted.** Not flagged off, not left as a non-default —
  deleted, in a follow-up commit that cites this document. A failed extra that survives as a
  dead flag is how a library accumulates the thing this project keeps finding in its own
  measurements.
* **All four fail.** The honest claim becomes *"MiniROCKET's bank under our trees"*, which is
  still the +3.8 on CPSC-2018 and +2.2 on PTB-XL, from a smaller and faster library. This is
  a good outcome and it is written down here as one, in advance, so that it cannot later be
  reframed as a disappointment worth re-running until it improves.
* **H-V15.5 fails.** The one-at-a-time design is reported as inconclusive about the failing
  extras rather than as a verdict on them.

## 5. The outcome I would least like

Two of them, and neither is "the extras fail".

The first is an extra clearing +0.5 on exactly five of eight datasets with a mean near zero —
a pass on the letter of a bar I wrote, carrying code that does nothing. §6 exists so that
this cannot ship on the primary suite alone.

The second is `abl_min` beating `abl_all` outright. That would mean the shipped configuration
has been *worse* than MiniROCKET's plain bank for the whole V6-V14 line, and that every
margin in `RESULTS_V13.md` understates what this architecture can do. It would be good news
about the design and bad news about four studies' worth of write-ups, and it would require a
correction to the README rather than a footnote.

## 6. Confirmation, pre-committed

An extra that passes §3 is **not** established. It is re-run on PTB-XL, n=500, 5 seeds, six
arms (~7 hours), and must clear `+0.5` there as well. PTB-XL is named now, before any score,
because it is 12-channel (so every extra is live), it is in the recent studies' line of work,
and at ~817 s per cell it is the only one of the four whose grid fits a second night.

An extra that passes UEA and fails PTB-XL is reported as **dataset-dependent and not
shipped** — it does not get a third dataset to try on.

## 7. Pre-committed analysis

Margins are computed per seed and paired within seed. Means are reported with the full
per-seed vector, never alone. No arm is dropped, no seed is dropped, no dataset is dropped,
and if a cell fails to run it is reported as failed rather than omitted. The report script
(`validation/report_v15.py`) applies the bars mechanically from `ABLATION_EXTRAS`, so which
arm counts as "the +virtual-channels arm" cannot be changed after a score has been seen.

`minirocket10k` and the `agg` baseline are run for context and are **not** part of any bar.
