# V6 results — the rocket base reaches parity with MiniROCKET

Complete grid, 5 seeds throughout, run after the calibration bug in
`DenseBase` was fixed (commit "Fix a base that helped in training and
confidently hurt at predict time"). `heartwood_rocket` is
`dense_base=True, dense_features="rocket"`. MiniROCKET is `aeon`, credited with
the better of 2,000 and 10,000 kernels.

| dataset | n | heartwood | +rocket | MiniROCKET | agg | vs rocket | vs default |
|---|---|---|---|---|---|---|---|
| ptbxl | 100 | 0.347 | 0.408 | 0.414 | 0.309 | −0.6 | **+6.0** |
| ptbxl | 250 | 0.399 | 0.438 | 0.439 | 0.327 | −0.1 | **+3.9** |
| ptbxl | 500 | 0.465 | 0.489 | 0.475 | 0.365 | +1.5 | **+2.5** |
| Epilepsy | 137 | 0.997 | 0.995 | 1.000 | 0.957 | −0.5 | −0.3 |
| HandMovementDirection | 160 | 0.429 | **0.459** | 0.387 | 0.289 | **+7.2** | +3.0 |
| Handwriting | 150 | 0.316 | **0.520** | 0.514 | 0.171 | +0.7 | **+20.5** |
| Heartbeat | 204 | 0.623 | 0.654 | 0.665 | 0.677 | −1.0 | +3.1 |
| Libras | 180 | 0.889 | 0.916 | 0.917 | 0.761 | −0.1 | +2.7 |
| NATOPS | 180 | 0.890 | 0.924 | 0.944 | 0.889 | −2.0 | +3.4 |
| RacketSports | 151 | 0.892 | 0.892 | 0.866 | 0.833 | **+2.5** | −0.0 |
| SelfRegulationSCP2 | 200 | 0.498 | 0.498 | 0.556 | 0.522 | −5.8 | +0.0 |

## Verdicts

**H-V6.1 (parity floor) — PASS.** Within 2 points of the better MiniROCKET on
**7 of 8** Arm B datasets; the bar was 75%. The bank is not broken.

**H-V6.2 (the win) — FAIL.** On PTB-XL, `heartwood_rocket` beats the better
MiniROCKET by ≥2 points at **0 of 3** training sizes (−0.6, −0.1, +1.5). A
majority was required.

**H-V6.3 (no regression) — FAIL**, on one scenario out of six. After the base
learned to decide itself against chance rather than against the mean, 20 of 24
synthetic cells are identical to the default or better. All four regressions are
`amp_regression`: −6.7, −3.5, −3.4, −2.3.

| scenario | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|
| bump_order | +0.0 | +0.0 | +0.3 | −0.1 |
| lead_lag | +0.0 | +0.0 | +0.0 | +0.0 |
| slope_window | +1.5 | +0.0 | −0.2 | +0.0 |
| timing | +0.4 | −0.3 | −0.1 | −0.0 |
| static_control | +0.0 | −0.3 | +0.0 | +0.0 |
| **amp_regression** | **−6.7** | **−3.5** | **−3.4** | **−2.3** |

No guard on base quality can fix that one, and it is worth being precise about
why. `amp_regression`'s ridge has a leave-one-out R² of 0.47 — genuine,
comfortably better than chance. The target is transient height *times* a static
coefficient, a pure interaction, so the linear base predicts it well on average
and still misleads the trees, which then have to spend capacity undoing it.
Deciding this correctly would mean fitting the booster both ways and comparing,
which doubles fit time. That is the same trade Phase C recorded for the
statistics bank, arrived at independently.

**H-V6.4 (earns its default) — NO.** The rule required H-V6.2 to pass and
H-V6.3 not to fail; neither held. `dense_features="rocket"` ships **opt-in**,
with these numbers, exactly as matched filters and the ridge base did before it.

## What actually happened

**The diagnosis in `HEADROOM.md` was right, and acting on it worked.** V5 measured
Heartwood at mean rank 2.33 against MiniROCKET's 1.50 — a clear deficit on
shape-regime data. Putting a ridge over a fixed convolution bank underneath the
trees, instead of asking a greedy per-node search to find those features, closes
it: median gap now **−0.1**, i.e. parity. Selection really was the ceiling.

**Against Heartwood's own previous default, this is a large and near-uniform
gain**: 7 of 11 cells improve by ≥2.5 points, the largest being Handwriting at
+20.5, and nothing regresses by more than 0.3. Whatever else is true, the base
should probably ship.

**Parity is not the target, and it is not a win.** H-V6.2 asked for ≥2 points on
the one dataset that has both a real static block and shape-regime series, and
it did not deliver. The margin does grow with training size (−0.6 → −0.1 → +1.5),
which is worth testing at n=1000 before drawing any conclusion from it, but three
points on a trend line is not evidence.

**The fix is visible in the table.** SelfRegulationSCP2 now reads 0.498 for both
`heartwood` and `heartwood_rocket` — identical, because the ridge finds nothing
there and the base is now declined outright rather than boosting from inverted
noise. Before the fix that cell read 0.422, below chance. The guard costs
nothing when the ridge does have signal and prevents a below-chance model when
it does not.

## Still open

* H-V6.3, the synthetic no-regression check.
* PTB-XL at n=1000, where the trend points.
* `dense_features="both"` — window statistics *and* convolutions in one bank —
  is implemented and has never been run on real data.
* Our bank is a shade behind `aeon`'s in a ridge-only comparison (median −0.7,
  worst NATOPS −4.4). Closing that would move every row here up.
