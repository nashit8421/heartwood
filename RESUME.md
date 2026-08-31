# Where this was left — paused 2026-08-31 09:35

The study queue was stopped mid-flight so the laptop could be shut down. Nothing is
mid-write; every completed study wrote its own `RESULTS_*.md` and those are committed.

## To resume

```bash
bash validation/run_v22.sh      # only the Apnea veto is outstanding (~10-13 h)
```

`validation/run_all.sh` is the full queue and is finished apart from V22. Re-running
`run_v22.sh` re-runs its synthetic stage too (8 minutes) and overwrites
`validation/rerun/v22_synth` — harmless, and it keeps the two halves from one run.

## What finished, and what it said

**Every pre-registered bar failed.** Eight studies, eight verdicts, no exceptions.

| study | headline | verdict |
|---|---|---|
| V14 channel width | 12-lead **+2.8**, 3-lead +2.6, 1-lead **+3.0** | **FAIL** — width is not the mechanism |
| V15 bank extras | all four ≤ +0.4 against a +0.5 bar | **FAIL** — delete all four |
| V16 per-node bagging | best 1/8, +0.4 | **FAIL** |
| V17 gain penalty | best 1/8, −0.1 | **FAIL** |
| V18 bank pre-screen | best 1/8, −0.1 | **FAIL** |
| V19 recalibrated null | best 1/8, −0.3 | **FAIL** |
| V21 nonlinear base | best 0/8, −0.3 | **FAIL** |
| V20 no-regret | 20 of 40 cells violate the tolerance | **FAIL** — and see below |
| V22 synthetic | `amp_regression` 0.4071 → **0.3953** (+2.9%) | bar was 4%; **Apnea veto not run** |

## The four things to do next, in this order

1. **Write up V20. It is the most important result here and it is not a bar failure.**
   The guarantee was expected to find nothing — `VALIDATION_V20.md` §5 predicted a suite
   that structurally could not produce the failure. Instead the *unguarded* model is worse
   than its own best component in **21 of 40 cells, worst case −13.5 points**. The shipped
   architecture is meaningfully worse than either half alone on half the suite. The
   guarantee then caught almost none of it (21 → 20), so the fallback does not work either.
   That is a defect in the model, found by a test that was expected to be idle, and it
   outranks everything on the original roadmap.

2. **Delete the four bank extras** (V15, pre-committed in `VALIDATION_V15.md` §4): virtual
   channels, comparison splits, the dyadic window-statistic block, Lévy areas. `abl_min` —
   MiniROCKET's plain bank — beat the full shipped configuration on 3 of 8 datasets. The
   honest claim becomes *"MiniROCKET's bank under our trees"*.

3. **Correct `validation/HEADROOM.md`** (V19 §4, pre-committed). Four independent attacks on
   greedy per-node selection all came in below bar and mostly negative. HEADROOM's account
   of the ceiling is the premise all four were built on, and it is now the thing to fix
   rather than attack a fifth time.

4. **Correct the README and `RESULTS_V13.md` on channel width** (roadmap item 8). V14 is
   final: twelve leads are *worse* than one. The replacement explanation for Apnea is
   already measured — its regime gap is −0.010, so a global summary of that one-minute ECG
   loses nothing a finer representation recovers (`RESULTS_SCREEN.md`).

## Caveats that must survive the pause

* **V21 measured curvature on the un-shrunk bank.** It ran after V15 in time, but nothing
  deleted the failed extras in between — that is item 2 above. Its verdict is negative at
  every width so the direction is safe, but the number belongs to a model we have decided
  not to ship. Do not quote it after the deletion without re-running.
* **V22 is half done.** The synthetic stage says `prod_margin` is the best arm at +2.9%,
  under its 4% bar and under the +8.4% oracle ceiling measured in `VALIDATION_V22.md` §2.
  `VALIDATION_V22.md` §5 named a `prod_margin`-only win as the outcome it would least like,
  because that arm multiplies the base's margin by a static the margin already contains.
  **H-V22.2, the Apnea veto, has not run at all** — no product change ships until it does.
* **V14's arms ran on two code versions.** Arm C on pre-speedup code, arms B and A on
  post-speedup. Scores are unaffected (that change was verified bit-identical across nine
  configurations); per-arm `fit_seconds` are not comparable.
* `validation/rerun/v22_apnea_INCOMPLETE/` is a partial grid, kept and renamed so it cannot
  be mistaken for a finished one.
