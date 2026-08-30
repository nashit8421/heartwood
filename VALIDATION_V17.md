# V17 — can a node price its own selection bias analytically?

Written and committed **before any V17 cell is run.** The charge has unit tests
(`tests/test_gain_penalty.py`, 11 passing) and no scores.

## 1. Why

V16 attacks the winner's curse by shrinking the pool. V17 attacks it by **charging for it**:
if a node's best-of-*m* gain is no larger than what *m* uninformative candidates would reach
anyway, the node does not split.

This is roadmap item 2b, and it exists because `selection_null` (V8, item 2d) already tried
the empirical version of this idea — permute the gradients, rescan the whole pool, take the
best noise gain as a floor — and paid for it: a full rescan of every candidate per permutation
per node. V17 asks whether the same floor can be had from a formula and one dot product.

## 2. The charge, and why it is not a tuned constant

With the parent term subtracted and the node's gradients centred against its own hessians, the
gain of a split on an *uninformative* feature is asymptotically `scale · χ²₁ / 2`, where
`scale` is the node's hessian-weighted gradient variance. The maximum of *m* independent
χ² grows like `2 log m`, so the best gain noise produces grows like `scale · log m`. The
charge is therefore

```
charge = mc_penalty · scale · (log m + 0.4 · log n_rows)
```

The second term prices the *within-candidate* search: `scan_threshold` maximises over every
cut point as well as over candidates, so the effective pool is larger than the candidate
count. Its coefficient is well under 1 because adjacent cut points are strongly correlated.

**All of this was measured against the real `scan_threshold`, not assumed.** Across centred,
shifted and heteroscedastic gradients and node sizes from 100 to 1600 rows, the `log m`
coefficient came out at 1.00–1.10 — the derivation's value — and the `log n` coefficient at
0.25–0.46. `tests/test_gain_penalty.py` re-measures both and fails if either the theory or
`scan_threshold` drifts away from the other.

The same fit produced an additive constant that ranged from −2.5 to −0.1 depending on the
gradient regime. **That constant is deliberately not in the formula.** An unstable number
dressed as theory is worse than a multiplier the study has to sweep in the open, and
`mc_penalty` is where that slack is put on the record.

Because every candidate at a node shares the same *m*, the charge cannot change *which*
candidate wins — only whether the node splits at all. A test asserts this, so the claim
cannot quietly stop being true.

## 3. Hypotheses

Arms: `rocket_static` (`mc_penalty=0`) against **0.5×, 1.0×, 2.0×**, named now and never added
to. 1.0 is the value the derivation implies; the other two bracket it.

Suite: the same eight UEA datasets as V15 and V16, official splits, 5 seeds.

* **H-V17.1 — the charge clears the roadmap's bar.** For at least one pre-named multiplier,
  `margin ≥ +1.5` points on **≥ 5 of 8** datasets against `mc_penalty=0`. The multiplier must
  be **the same on every dataset**; the report computes the per-dataset best, labels it *not a
  result*, and prints the gap as tuning optimism.
* **H-V17.2 — the theoretical value is the right one.** PASS if 1.0× is the best of the three
  multipliers. This is the check on §2: if the derivation is doing the work, its own value
  should win. If 2.0× wins instead, the missing additive constant is real and the formula is
  under-charging; if 0.5× wins, it is over-charging and the independence assumption behind
  `log m` is too generous.
* **H-V17.3 — it is cheaper than the permutation null.** PASS if a `mc_penalty=1.0` fit is
  faster than the same fit at `selection_null=1`. This is not an accuracy claim and it is the
  only reason to prefer 2b over 2d if both work.

## 4. What each outcome means

* **H-V17.1 passes.** The charge becomes a default and V8's verdict on `selection_null` is
  revisited — it would mean the idea was right and only the instrument was too expensive.
* **H-V17.1 fails, H-V16.1 also failed.** Two independent attacks on the winner's curse, both
  below bar, is real evidence **against** the account of the ceiling that `HEADROOM.md`
  proposed and that this roadmap's item 2 is built on. That would make 2c and 2d much less
  promising and it would demand a correction to `HEADROOM.md`'s closing paragraph rather than
  another attempt.
* **H-V17.1 fails, H-V16.1 passed.** Shrinking the pool works and pricing it does not, which
  says the damage is done by the *candidates a node scans*, not by the arithmetic of the
  maximum. Item 2c (pre-screening) inherits that and gets more attractive.
* **The charge stays in the library at 0 either way.** It is ~15 lines, it is the analytic
  counterpart to `selection_null`, and every later selection experiment wants both floors
  available for comparison.

## 5. The outcome I would least like

H-V17.1 passing while H-V17.2 fails at 2.0×. That combination would mean the charge works but
the derivation does not explain why it works at the size that works — that we are one
multiplier away from an arbitrary pruning knob with a proof attached to it. It would be a
benchmark win and a theory failure at once, and §2's care about the unstable constant is
exactly what would be implicated. If it happens, the write-up leads with the theory failure.

## 6. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. The bar, the majority rule
and the multiplier list are applied mechanically by `validation/report_v17.py` from `MC_ARMS`.
H-V17.3 is read from the `fit_seconds` already recorded per cell.
