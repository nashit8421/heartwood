# Roadmap — what to do next, and why

Written 2026-08-30, after V13 and while V14 is still running. Ordered by
information-per-hour, not by how interesting the work is.

Every item states a **bar** in advance. An item that does not clear its bar does not
ship, and the failure is written up rather than retried until it passes. That rule is
the reason four measurement defects in this project were caught before they became
claims, and it applies to everything below.

Current state this plan responds to:

* The time-series half beats MiniROCKET on CPSC-2018 (+3.8), PTB-XL (+2.2) and
  Sleep-EDF (+1.8), and loses on Apnea-ECG (−3.0).
* The static block is worth +9.2 points on Apnea when the statics are exogenous and
  the base can see them — the fusion mechanism works.
* **The founding claim is still unproven after four attempts**, because no dataset yet
  has had both halves strong at once.
* `HEADROOM.md` located the ceiling at greedy per-node feature selection, not at
  search budget.

---

## Tier 1 — cheap, high-information, attacks the measured ceiling

### 1. Ablate the bank's extras individually

**The highest-value open item, and it should not have stayed open this long.**

Our bank reproduces MiniROCKET's exactly and adds virtual channels, comparison splits
and interval statistics. Those extras have **never been measured separately.** The V7
decomposition — bank +0.0/+0.1 against trees +3.8/+1.2 — hints they contribute nothing,
and if that is right they should be deleted.

* **Arms:** MiniROCKET-only bank / +virtual channels / +comparison splits / +interval
  stats / all four.
* **Bar:** each extra must clear **+0.5** on a majority of datasets to survive.
* **Both outcomes are wins.** Either we learn where our edge actually lives, or we
  delete code and the honest claim becomes "MiniROCKET's bank under our trees" — which
  is still a +3.8 result, from a smaller and faster library.
* **Cost:** ~1 night. Needs only a bank-composition flag.

### 2. Attack greedy per-node selection

The measured ceiling. With ~10,000 candidate features per node, max-gain frequently
picks the luckiest feature rather than the best one, and every subtree below inherits
the mistake. Four approaches, cheapest first:

* **2a. Per-node feature bagging** (`colsample_bynode`). ~20 lines, standard, attacks
  the winner's curse directly by shrinking the candidate pool at each node. **Start here.**
* **2b. Gain penalised by candidate count** — a multiple-comparisons correction applied
  to the maximum. Principled and cheap.
* **2c. Two-stage pre-screen** — rank the bank by marginal association on a held-out
  fold, let the trees split only among the top-k.
* **2d. Revisit `selection_null`** with a better-calibrated null. V8's attempt failed;
  that is one attempt at a hard problem, not a verdict on the idea.

* **Bar:** **+1.5** on a majority of datasets, or it does not ship.
* **Cost:** 2a is a day; the others a few days each.

### 3. The no-regret guarantee

**The change most worth having as a design property rather than a benchmark number.**

Fit the halves, compare them honestly on held-out data, and fall back to the better
half when the combination cannot beat it. Both parts already exist: the base can
switch itself off against a permutation null, and the regime diagnostic is computed
from baselines alone so it cannot be fitted to the answer.

This makes `static_control`-style regressions impossible by construction instead of
something re-measured every study. "Never meaningfully worse than its best component"
is a stronger claim than any single benchmark win.

* **Bar:** no cell in the suite worse than **−0.5** against its best component.
* **Cost:** ~2 days. Most of the work is validating the fallback decision honestly —
  it is itself a selection step and can overfit like any other.

---

## Tier 2 — architecture

### 4. Nonlinear base via random Fourier features or Nyström

The base should be nonlinear; it should **not** become LightGBM. Project the bank
through a random nonlinear map and ridge on top: real nonlinearity, still linear in
the transformed space, so closed-form leave-one-group-out survives.

That last clause is the whole point. Exact closed-form LOO/LOGO is what caught the V12
and V13 defects and what lets a block hold-out be verified to 4.9e-15 against literal
refits. A tree-model base would cost k-fold refits and lose exactness — trading a small
modelling gain for the ability to detect our own mistakes, on a project that has needed
that ability four times.

* **Bar:** **+1.0** over linear ridge on a majority of datasets, *and* the exact-LOGO
  refit check still passes to ~1e-14.
* **Cost:** ~3 days. Do it **after** item 1, or we will not know what we improved on.

### 5. Fix `amp_regression` (currently −11.3)

V12 diagnosed the cause: the target is `transient_height × static_coefficient`, a
product of **magnitudes**, and the model is fed products of **ranks**, which discard
exactly the information needed. Item 3 stops the bleeding; this fixes the cause. Needs
magnitude products with a bound that does not reintroduce the V11 blow-up (Apnea 0.478).

---

## Tier 3 — the dataset problem, which is the actual blocker

### 6. Build a dataset screening harness before committing to another study

Four attempts, four datasets, still no fair test. That is partly bad luck and partly
that **we commit to a full study before knowing whether the dataset can answer the
question at all.**

Three cheap numbers, hours rather than days:

1. **`static_only` score** — are the statics strong on their own?
2. **Exogeneity** — can the series predict the statics? Train signal → static; if that
   succeeds the statics are disqualified, the way age and sex are on an ECG.
3. **Regime** — the gap between raw series and aggregates; is there series information
   a summary loses?

**Run the full study only if all three pass.** This screen would have said in an
afternoon that Sleep-EDF cannot test the claim (statics at chance) and that CPSC's
statics are disqualified.

### 7. Screen candidate datasets

PhysioNet-2019 Sepsis; MIMIC waveform plus demographics; wearable accelerometry (body
weight is genuinely exogenous to a step signal); industrial sensor traces plus machine
specifications. Screen several cheaply, commit to whichever passes.

---

## Explicitly not doing

* **Optuna or any hyperparameter search.** Measured in `HEADROOM.md`: 16× the search
  budget bought +1.5 on a 20-point gap and made 2 of 3 datasets worse. Worse, it
  industrialises the validation-optimism failure mode that has already produced four
  defects here — it would find penalties that flatter the held-out score with no way to
  distinguish that from a real gain.
* **LightGBM or random forest as the base.** Makes the tree layer redundant (gradient
  boosting stacked on gradient boosting) and destroys closed-form LOGO. See item 4.

---

## Housekeeping

8. **Correct the README on V14's answer, whichever way it falls.** If we still beat
   MiniROCKET on single-lead CPSC then the multi-channel explanation is wrong, and that
   is the second correction to the V13 write-up. `VALIDATION_V14.md` §5 names this as
   the outcome I would least like, so it gets recorded either way.
9. **Speed.** CPSC runs ~3,150 s/seed. Irrelevant to the science, relevant to whether
   anyone would use the library. Low priority until the claims settle.

---

## Recommended starting point

Items **1** and **2a** together — both cheap, both aimed at things already measured,
and item 1 may let us delete a third of the bank. Item **6** in parallel, since it is
mostly a script and it unblocks the only question that finally matters.
