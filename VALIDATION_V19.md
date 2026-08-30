# V19 — a permutation null that stays put

Written and committed **before any V19 cell is run.** The recalibration has unit tests
(`tests/test_selection_null.py`, 7 passing) and no scores.

## 1. Why

V8 gave a node a chance floor: permute the gradients, rescan the pool, and refuse to split
when the winning gain is no better than what noise reaches. It failed. Roadmap item 2d says
that is one attempt at a hard problem rather than a verdict on the idea, and this study starts
by finding out *what* failed.

**It was the calibration, and the defect is specific.** V8's floor was the maximum gain over
every permutation *and* every candidate. So the bar tightened as `selection_null` grew: asking
the null for more precision also asked it for more severity, and no sweep over that knob could
be interpreted. At one permutation — the value V8 actually ran — the test is barely a test at
all, because comparing an observed maximum against a single draw of the null maximum accepts
about half of pure noise.

Measured on a task with no signal whatsoever, 120 trials, false-split rate at the root:

| setting | 1 perm | 4 perms | 16 perms |
|---|---|---|---|
| quantile 1.0 (**V8**) | 0.53 | 0.21 | 0.06 |
| quantile 0.5 | — | 0.52 | 0.53 |
| quantile 0.95 | — | 0.23 | 0.07 |

The first row is the bug in one line: same test, three different bars. The second row is what a
fixed quantile buys — the same bar, estimated more precisely. This is not a new idea in this
library; it is the pattern `DenseBase._chance_r2` has used since V6, at 32 permutations and the
0.95 quantile, one level down from where V8 needed it.

## 2. The fix, and one thing it does not fix

Take the maximum *within* each permutation, then a fixed quantile *across* them.
`selection_null_quantile` defaults to 1.0, which reproduces V8 exactly at one permutation, so
the failing arm stays available for comparison rather than being quietly redefined.

What this does not fix: estimating a far tail from few draws. The 0.95 row above reads 0.23 at
four permutations against 0.07 at sixteen — a quantile is only as good as the sample behind it.
**So V19 fixes the permutation count at 16 and sweeps only the quantile.** Sweeping both would
confound "which bar" with "how well the bar is estimated", which is a version of the exact
mistake being corrected.

The cost was measured before the arms were chosen: 16 permutations is **3.0× the fit time**,
not 16×, because a rescan reuses the candidate columns and pays only for `scan_threshold`. That
is what makes a properly estimated null affordable at all, and it is the reason V8's economy of
one permutation was never necessary.

## 3. Hypotheses

Arms: `rocket_static` (no floor) against **q=0.5, q=0.9, q=0.95** at 16 permutations, plus
`null_v8` (one permutation, global maximum) as the reference for what failed.

Suite: the same eight UEA datasets as V15–V18, official splits, 5 seeds.

* **H-V19.1 — the recalibrated null clears the roadmap's bar.** For at least one pre-named
  quantile, `margin ≥ +1.5` points on **≥ 5 of 8** datasets, same quantile everywhere.
* **H-V19.2 — the recalibration is why.** PASS if the best quantile arm beats `null_v8` by
  ≥ 1.0 points on a majority. Without this, a V19 win is just "a floor helps" and says nothing
  about the diagnosis in §1 — the whole reason to revisit 2d rather than restate V8.
* **H-V19.3 — severity is monotone in the quantile.** PASS if mean margin is ordered
  q=0.5 ≤ q=0.9 ≤ q=0.95 or its reverse. A non-monotone winner across three values is what a
  small search looks like when it finds noise, and it would send the result to the PTB-XL
  confirmation before anything ships.

## 4. What each outcome means

* **H-V19.1 and H-V19.2 pass.** V8's verdict is overturned, the null ships, and the honest
  write-up says the idea was right and the instrument was mis-calibrated — a correction to the
  V8 write-up, not a new discovery.
* **H-V19.1 passes, H-V19.2 fails.** A floor helps and the calibration was not the point. V8
  should then have worked, and the discrepancy needs explaining before anything ships.
* **H-V19.1 fails.** Item 2 closes with four attempts — 2a bagging, 2b an analytic charge, 2c a
  pre-screen, 2d a recalibrated null — all below bar. Four independent attacks failing is not
  four failures; it is evidence against the account of the ceiling that `HEADROOM.md` proposed
  and that all four were built on. **The next commit then corrects `HEADROOM.md` rather than
  attempting a fifth attack.** That is written here, in advance, because the alternative — one
  more idea, and one more after that — is how a project spends a year on a premise nobody
  re-examined.

## 5. The outcome I would least like

H-V19.1 passing at q=0.5. That is the *permissive* bar — a floor that refuses only the bottom
half of noise splits — and if it beats both stricter quantiles then the benefit is not
"rejecting noise" at all; it is something about stopping trees a little earlier, which
`max_depth` or `gamma` would do more cheaply and more honestly. I would rather find that out
than ship a permutation null that is really a regularisation knob wearing a hypothesis test's
clothes, and §3's H-V19.3 is what would surface it.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. The bar, the majority rule
and the quantile list are applied mechanically by `validation/report_v19.py` from `NULL_ARMS`.
H-V19.2 is read from the `null_v8` arm in the same run.
