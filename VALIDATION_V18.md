# V18 — does ranking the bank beat offering it blind?

Written and committed **before any V18 cell is run.** The screen has unit tests
(`tests/test_bank_screening.py`, 11 passing) and no scores.

## 1. Why

The feature bank is this library's memory: a column that won a split in round 3 is offered
free at every later node instead of having to be rediscovered. But it is offered **blind** —
`FeatureBank.candidates` hands a node a random 25% of the bank and lets maximum gain choose,
which is the winner's curse in its purest form. The bank was built to stop good features being
forgotten; nothing in it stops a lucky feature being remembered.

Roadmap item 2c replaces the lottery with a shortlist: rank the bank by marginal association
and let the trees split only among the top-*k*.

## 2. The design decision that matters

A ranking is itself a selection step. Ranked on the rows the tree then fits, it would order
the bank by the very signal the split is about to exploit — training scores would improve, the
model would be *more* over-fitted, and nothing in a results table would show it. That is the
most expensive way this change could go wrong and the least visible.

So the screen is **out of fold, and the fold rotates**. Every round and every output, a
random 25% of that round's rows is held out, the bank is ranked against the Newton residuals
on those rows, and the tree is grown on the remaining 75%. No row is permanently spent — the
cost is that any single tree sees a quarter fewer rows, not that the model does.

Two consequences are deliberate and are the things to remember when reading the result:

* **A failing V18 is ambiguous by construction.** Each tree trains on 75% of its rounds's
  rows, so a null result could be a shortlist that does not help *or* a training set that got
  smaller. §3 puts an arm in the study specifically to separate those.
* **The screen replaces `bank_colsample` rather than composing with it.** Thinning a ranked
  shortlist at random again would undo the ranking, so a screened node sees exactly the top-*k*
  and no random draw.

A screen that would leave too few rows to grow a tree is skipped, and the bank is left
unscreened for that round — better an unscreened round than a round that silently measures a
starved fit.

## 3. Hypotheses

Arms: `rocket_static` (`screen_fraction=0`) against **top-4, top-8, top-16** at a fixed 25%
fold. Only the shortlist length is swept; sweeping the fold size as well would be a
two-dimensional search dressed as a study.

Suite: the same eight UEA datasets as V15–V17, official splits, 5 seeds.

* **H-V18.1 — the screen clears the roadmap's bar.** For at least one pre-named *k*,
  `margin ≥ +1.5` points on **≥ 5 of 8** datasets, with the same *k* on every dataset.
* **H-V18.2 — the shortlist, not the smaller fit.** The `subsample=0.75` arm — same fraction
  of rows per tree, no screening — must **not** clear the bar. If it does, V18 measured the
  effect of training on fewer rows and its own hypothesis is unfalsifiable as posed. This arm
  is why the study is worth running rather than merely worth reporting.
* **H-V18.3 — the shortlist is a shortlist.** PASS if the best *k* is smaller than the bank
  (`bank_max=32`). A top-16 win with 4 and 8 failing would say the useful thing is having
  *any* stable ordering, not the restriction — a different mechanism, and it should be named
  as one rather than reported as a screening result.

## 4. What each outcome means

* **H-V18.1 passes and H-V18.2 holds.** Screening ships and the bank stops being offered
  blind. This would also be the first of items 2a–2c to clear its bar, which would say the
  ceiling is about *which* candidates reach a node rather than about the arithmetic of the
  maximum.
* **H-V18.1 fails.** The screen stays in the library at its no-op default of 0 and the roadmap
  moves to 2d. Combined with V16 and V17 also failing, three independent attacks below bar
  would be strong evidence against `HEADROOM.md`'s account of the ceiling — and at that point
  the honest move is to correct that document, not to try 2d.
* **H-V18.2 fails.** The result is withdrawn regardless of H-V18.1. A screening win that a
  plain row subsample reproduces is not a screening win.

## 5. The outcome I would least like

H-V18.1 and H-V18.2 both passing, with the margin coming from Handwriting alone. Handwriting
is the dataset with 26 classes and the largest MiniROCKET deficit, so it is the one where a
bank shortlist has the most room to matter and the one whose per-seed variance is widest. A
majority driven by one extreme cell is a majority I would not trust, and the per-seed vectors
in §6 are what would show it.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. The bar, the majority rule
and the *k* list are applied mechanically by `validation/report_v18.py` from `SCREEN_ARMS`.
H-V18.2 is read from the `sub075` arm in the same run.
