# Benchmark results

Grid: scenarios × n_train [100, 250, 500, 1000] × 3 seeds, test n=2000. Every model gets 200 rounds, depth 4, learning rate 0.1; Heartwood runs on library defaults with no per-scenario tuning.

Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline at that size.

Two comparisons are reported because they answer different questions. **agg** is the global-aggregate workaround teams actually ship — beating it is the claim this library makes. **best-of-all baselines** is an oracle: it picks, per task and per training size, whichever of the five representations turned out best, which nobody can do in advance. Losing to that oracle on some task is expected; losing to `agg` would mean the premise is wrong.

### bump_order — accuracy (higher is better)
_which of two transients happened first, XOR a static flag_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.533±0.023*  | 0.665±0.025*  | 0.826±0.008*  | 0.946±0.022*  |
| agg          | 0.502±0.010   | 0.499±0.005   | 0.501±0.017   | 0.508±0.004   |
| wagg4        | 0.524±0.024   | 0.582±0.057   | 0.699±0.087   | 0.817±0.008   |
| wagg8        | 0.509±0.004   | 0.563±0.057   | 0.602±0.019   | 0.791±0.015   |
| wagg16       | 0.502±0.000   | 0.524±0.011   | 0.573±0.009   | 0.690±0.012   |
| raw_flat     | 0.508±0.007   | 0.517±0.011   | 0.557±0.024   | 0.648±0.021   |

vs agg (the standard workaround) — n=100: +3.1pt, n=250: +16.6pt, n=500: +32.5pt, n=1000: +43.7pt
vs best-of-all baselines (oracle choice) — n=100: +0.9pt, n=250: +8.3pt, n=500: +12.7pt, n=1000: +12.9pt

### timing — accuracy (higher is better)
_did the event happen before this row's static deadline_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.908±0.021*  | 0.955±0.006*  | 0.969±0.007*  | 0.975±0.003*  |
| agg          | 0.625±0.030   | 0.653±0.008   | 0.670±0.012   | 0.669±0.005   |
| wagg4        | 0.875±0.020   | 0.921±0.007   | 0.938±0.003   | 0.952±0.003   |
| wagg8        | 0.834±0.018   | 0.919±0.002   | 0.942±0.002   | 0.954±0.008   |
| wagg16       | 0.745±0.048   | 0.885±0.002   | 0.927±0.006   | 0.954±0.004   |
| raw_flat     | 0.709±0.046   | 0.821±0.021   | 0.876±0.010   | 0.921±0.003   |

vs agg (the standard workaround) — n=100: +28.3pt, n=250: +30.2pt, n=500: +29.9pt, n=1000: +30.6pt
vs best-of-all baselines (oracle choice) — n=100: +3.3pt, n=250: +3.4pt, n=500: +2.7pt, n=1000: +2.1pt

### slope_window — accuracy (higher is better)
_the direction of drift inside one off-grid window, XOR a static gate_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.551±0.017   | 0.747±0.034   | 0.894±0.041   | 0.981±0.006   |
| agg          | 0.521±0.008   | 0.587±0.013   | 0.610±0.012   | 0.648±0.013   |
| wagg4        | 0.503±0.007   | 0.594±0.018   | 0.787±0.104   | 0.898±0.038   |
| wagg8        | 0.581±0.078   | 0.870±0.079   | 0.989±0.001   | 0.994±0.002   |
| wagg16       | 0.507±0.008   | 0.600±0.075   | 0.812±0.014   | 0.852±0.018   |
| raw_flat     | 0.502±0.008   | 0.563±0.018   | 0.656±0.025   | 0.790±0.034   |

vs agg (the standard workaround) — n=100: +2.9pt, n=250: +16.0pt, n=500: +28.4pt, n=1000: +33.2pt
vs best-of-all baselines (oracle choice) — n=100: -3.0pt, n=250: -12.3pt, n=500: -9.5pt, n=1000: -1.3pt

### amp_regression — rmse (lower is better)
_the height of a transient in one stretch, times a static coefficient_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.572±0.008   | 0.439±0.010   | 0.393±0.008   | 0.368±0.003   |
| agg          | 0.848±0.010   | 0.798±0.003   | 0.724±0.024   | 0.689±0.006   |
| wagg4        | 0.516±0.023   | 0.418±0.007   | 0.380±0.002   | 0.359±0.002   |
| wagg8        | 0.640±0.036   | 0.494±0.032   | 0.418±0.012   | 0.382±0.005   |
| wagg16       | 0.702±0.041   | 0.520±0.007   | 0.447±0.020   | 0.396±0.008   |
| raw_flat     | 0.823±0.043   | 0.556±0.016   | 0.463±0.006   | 0.404±0.005   |

vs agg (the standard workaround) — n=100: +32.5%, n=250: +45.0%, n=500: +45.7%, n=1000: +46.6%
vs best-of-all baselines (oracle choice) — n=100: -10.9%, n=250: -5.1%, n=500: -3.5%, n=1000: -2.5%

### static_control — accuracy (higher is better)
_nothing temporal at all — the signal is entirely in the static columns_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.857±0.021   | 0.923±0.004*  | 0.943±0.004*  | 0.952±0.001   |
| agg          | 0.860±0.014   | 0.917±0.003   | 0.942±0.003   | 0.954±0.002   |
| wagg4        | 0.851±0.015   | 0.910±0.004   | 0.939±0.004   | 0.952±0.001   |
| wagg8        | 0.839±0.021   | 0.907±0.003   | 0.938±0.006   | 0.954±0.003   |
| wagg16       | 0.841±0.014   | 0.903±0.010   | 0.934±0.003   | 0.952±0.002   |
| raw_flat     | 0.841±0.013   | 0.914±0.004   | 0.936±0.004   | 0.954±0.002   |

vs agg (the standard workaround) — n=100: -0.3pt, n=250: +0.6pt, n=500: +0.0pt, n=1000: -0.2pt
vs best-of-all baselines (oracle choice) — n=100: -0.3pt, n=250: +0.6pt, n=500: +0.0pt, n=1000: -0.2pt

### fit time (seconds, mean over the whole grid)

  heartwood      12.92   (max 27.25)
  wagg16          0.79   (max 1.98)
  raw_flat        0.54   (max 1.47)
  wagg8           0.40   (max 0.99)
  wagg4           0.23   (max 0.56)
  agg             0.10   (max 0.22)

_macOS-26.5.2-arm64-arm-64bit, python 3.10.17, 2.6 min._
