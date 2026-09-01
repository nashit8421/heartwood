# Benchmark results

Grid: scenarios × n_train [500] × 5 seeds, test n=2000. Every model gets 200 rounds, depth 4, learning rate 0.1; Heartwood runs on library defaults with no per-scenario tuning.

Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline at that size.

Two comparisons are reported because they answer different questions. **agg** is the global-aggregate workaround teams actually ship — beating it is the claim this library makes. **best-of-all baselines** is an oracle: it picks, per task and per training size, whichever of the five representations turned out best, which nobody can do in advance. Losing to that oracle on some task is expected; losing to `agg` would mean the premise is wrong.

### bump_order — accuracy (higher is better)
_which of two transients happened first, XOR a static flag_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.982±0.005*  |
| hw_v23_base  | 0.824±0.071   |
| hw_v23_cmp   | 0.983±0.004   |
| hw_v23_levy  | 0.824±0.071   |
| hw_v23_both  | 0.983±0.004   |
| agg          | 0.497±0.014   |
| wagg4        | 0.704±0.070   |
| wagg8        | 0.626±0.033   |
| wagg16       | 0.584±0.034   |
| raw_flat     | 0.568±0.026   |
| agg_naive    | 0.497±0.014   |
| wagg8_naive  | 0.626±0.033   |

vs agg (the standard workaround) — n=500: +48.4pt
vs best-of-all baselines (oracle choice) — n=500: +27.8pt

### timing — accuracy (higher is better)
_did the event happen before this row's static deadline_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.986±0.006*  |
| hw_v23_base  | 0.936±0.016   |
| hw_v23_cmp   | 0.983±0.004   |
| hw_v23_levy  | 0.936±0.016   |
| hw_v23_both  | 0.983±0.004   |
| agg          | 0.662±0.013   |
| wagg4        | 0.938±0.003   |
| wagg8        | 0.940±0.004   |
| wagg16       | 0.923±0.007   |
| raw_flat     | 0.875±0.009   |
| agg_naive    | 0.662±0.013   |
| wagg8_naive  | 0.940±0.004   |

vs agg (the standard workaround) — n=500: +32.4pt
vs best-of-all baselines (oracle choice) — n=500: +4.6pt

### slope_window — accuracy (higher is better)
_the direction of drift inside one off-grid window, XOR a static gate_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.997±0.001*  |
| hw_v23_base  | 0.972±0.017   |
| hw_v23_cmp   | 0.997±0.001   |
| hw_v23_levy  | 0.972±0.017   |
| hw_v23_both  | 0.997±0.001   |
| agg          | 0.602±0.026   |
| wagg4        | 0.784±0.083   |
| wagg8        | 0.988±0.004   |
| wagg16       | 0.788±0.044   |
| raw_flat     | 0.646±0.032   |
| agg_naive    | 0.602±0.026   |
| wagg8_naive  | 0.988±0.004   |

vs agg (the standard workaround) — n=500: +39.5pt
vs best-of-all baselines (oracle choice) — n=500: +0.9pt

### lead_lag — accuracy (higher is better)
_which of two channels moved first, XOR a static flag_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.620±0.056*  |
| hw_v23_base  | 0.500±0.006   |
| hw_v23_cmp   | 0.540±0.041   |
| hw_v23_levy  | 0.553±0.035   |
| hw_v23_both  | 0.620±0.056   |
| agg          | 0.509±0.011   |
| wagg4        | 0.513±0.018   |
| wagg8        | 0.506±0.011   |
| wagg16       | 0.495±0.008   |
| raw_flat     | 0.496±0.006   |
| agg_naive    | 0.509±0.011   |
| wagg8_naive  | 0.506±0.011   |

vs agg (the standard workaround) — n=500: +11.1pt
vs best-of-all baselines (oracle choice) — n=500: +10.7pt

### fit time (seconds, mean over the whole grid)

  hw_v23_both    17.69   (max 21.21)
  hw_v23_levy    17.65   (max 19.99)
  hw_v23_cmp     17.50   (max 20.56)
  hw_v23_base    17.27   (max 19.95)
  heartwood      16.14   (max 18.71)
  wagg16          1.07   (max 1.86)
  raw_flat        0.72   (max 1.18)
  wagg8_naive     0.53   (max 0.99)
  wagg8           0.53   (max 0.98)
  wagg4           0.30   (max 0.52)
  agg             0.12   (max 0.17)
  agg_naive       0.11   (max 0.17)

_macOS-26.5.2-arm64-arm-64bit, python 3.10.17, 6.4 min._
