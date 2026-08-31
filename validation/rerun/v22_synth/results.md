# Benchmark results

Grid: scenarios × n_train [500] × 5 seeds, test n=2000. Every model gets 200 rounds, depth 4, learning rate 0.1; Heartwood runs on library defaults with no per-scenario tuning.

Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline at that size.

Two comparisons are reported because they answer different questions. **agg** is the global-aggregate workaround teams actually ship — beating it is the claim this library makes. **best-of-all baselines** is an oracle: it picks, per task and per training size, whichever of the five representations turned out best, which nobody can do in advance. Losing to that oracle on some task is expected; losing to `agg` would mean the premise is wrong.

### bump_order — accuracy (higher is better)
_which of two transients happened first, XOR a static flag_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.982±0.005*  |
| hw_rocket_static | 0.983±0.004   |
| hw_prod_split | 0.985±0.007   |
| hw_prod_margin | 0.984±0.004   |
| hw_prod_both | 0.983±0.009   |
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
| hw_rocket_static | 0.983±0.004   |
| hw_prod_split | 0.983±0.004   |
| hw_prod_margin | 0.981±0.003   |
| hw_prod_both | 0.982±0.005   |
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
| hw_rocket_static | 0.997±0.001   |
| hw_prod_split | 0.995±0.002   |
| hw_prod_margin | 0.997±0.001   |
| hw_prod_both | 0.995±0.002   |
| agg          | 0.602±0.026   |
| wagg4        | 0.784±0.083   |
| wagg8        | 0.988±0.004   |
| wagg16       | 0.788±0.044   |
| raw_flat     | 0.646±0.032   |
| agg_naive    | 0.602±0.026   |
| wagg8_naive  | 0.988±0.004   |

vs agg (the standard workaround) — n=500: +39.5pt
vs best-of-all baselines (oracle choice) — n=500: +0.9pt

### amp_regression — rmse (lower is better)
_the height of a transient in one stretch, times a static coefficient_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.372±0.011*  |
| hw_rocket_static | 0.407±0.010   |
| hw_prod_split | 0.403±0.004   |
| hw_prod_margin | 0.395±0.005   |
| hw_prod_both | 0.397±0.007   |
| agg          | 0.722±0.019   |
| wagg4        | 0.380±0.002   |
| wagg8        | 0.421±0.012   |
| wagg16       | 0.450±0.016   |
| raw_flat     | 0.459±0.009   |
| agg_naive    | 0.722±0.019   |
| wagg8_naive  | 0.421±0.012   |

vs agg (the standard workaround) — n=500: +48.5%
vs best-of-all baselines (oracle choice) — n=500: +2.2%

### lead_lag — accuracy (higher is better)
_which of two channels moved first, XOR a static flag_

| model        | n=500         |
|--------------|---------------|
| heartwood    | 0.620±0.056*  |
| hw_rocket_static | 0.620±0.056   |
| hw_prod_split | 0.715±0.055   |
| hw_prod_margin | 0.620±0.056   |
| hw_prod_both | 0.715±0.055   |
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

  hw_prod_split   20.88   (max 24.62)
  hw_prod_both   20.35   (max 22.74)
  hw_prod_margin   18.05   (max 19.93)
  hw_rocket_static   17.68   (max 19.67)
  heartwood      15.92   (max 18.79)
  wagg16          1.01   (max 1.42)
  raw_flat        0.67   (max 0.92)
  wagg8_naive     0.51   (max 0.78)
  wagg8           0.50   (max 0.74)
  wagg4           0.28   (max 0.40)
  agg             0.11   (max 0.15)
  agg_naive       0.11   (max 0.13)

_macOS-26.5.2-arm64-arm-64bit, python 3.10.17, 8.2 min._
