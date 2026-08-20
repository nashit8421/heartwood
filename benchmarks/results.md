# Benchmark results

Grid: scenarios × n_train [100, 250, 500, 1000] × 3 seeds, test n=2000. Every model gets 200 rounds, depth 4, learning rate 0.1; Heartwood runs on library defaults with no per-scenario tuning.

Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline at that size.

Two comparisons are reported because they answer different questions. **agg** is the global-aggregate workaround teams actually ship — beating it is the claim this library makes. **best-of-all baselines** is an oracle: it picks, per task and per training size, whichever of the five representations turned out best, which nobody can do in advance. Losing to that oracle on some task is expected; losing to `agg` would mean the premise is wrong.

### bump_order — accuracy (higher is better)
_which of two transients happened first, XOR a static flag_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.641±0.181*  | 0.958±0.019*  | 0.983±0.005*  | 0.996±0.000*  |
| hw_phaseA    | 0.533±0.023   | 0.665±0.025   | 0.826±0.008   | 0.946±0.022   |
| hw_bank      | 0.502±0.011   | 0.641±0.089   | 0.787±0.070   | 0.960±0.010   |
| hw_filters   | 0.627±0.157   | 0.929±0.014   | 0.977±0.010   | 0.994±0.001   |
| agg          | 0.502±0.010   | 0.499±0.005   | 0.501±0.017   | 0.508±0.004   |
| wagg4        | 0.524±0.024   | 0.582±0.057   | 0.699±0.087   | 0.817±0.008   |
| wagg8        | 0.509±0.004   | 0.563±0.057   | 0.602±0.019   | 0.791±0.015   |
| wagg16       | 0.502±0.000   | 0.524±0.011   | 0.573±0.009   | 0.690±0.012   |
| raw_flat     | 0.508±0.007   | 0.517±0.011   | 0.557±0.024   | 0.648±0.021   |

vs agg (the standard workaround) — n=100: +13.9pt, n=250: +46.0pt, n=500: +48.2pt, n=1000: +48.7pt
vs best-of-all baselines (oracle choice) — n=100: +11.7pt, n=250: +37.6pt, n=500: +28.4pt, n=1000: +17.9pt

### timing — accuracy (higher is better)
_did the event happen before this row's static deadline_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.959±0.013*  | 0.977±0.005*  | 0.987±0.004*  | 0.991±0.001*  |
| hw_phaseA    | 0.908±0.021   | 0.955±0.006   | 0.969±0.007   | 0.975±0.003   |
| hw_bank      | 0.901±0.022   | 0.953±0.005   | 0.963±0.007   | 0.972±0.003   |
| hw_filters   | 0.952±0.013   | 0.973±0.005   | 0.987±0.004   | 0.989±0.002   |
| agg          | 0.625±0.030   | 0.653±0.008   | 0.670±0.012   | 0.669±0.005   |
| wagg4        | 0.875±0.020   | 0.921±0.007   | 0.938±0.003   | 0.952±0.003   |
| wagg8        | 0.834±0.018   | 0.919±0.002   | 0.942±0.002   | 0.954±0.008   |
| wagg16       | 0.745±0.048   | 0.885±0.002   | 0.927±0.006   | 0.954±0.004   |
| raw_flat     | 0.709±0.046   | 0.821±0.021   | 0.876±0.010   | 0.921±0.003   |

vs agg (the standard workaround) — n=100: +33.4pt, n=250: +32.4pt, n=500: +31.8pt, n=1000: +32.2pt
vs best-of-all baselines (oracle choice) — n=100: +8.4pt, n=250: +5.6pt, n=500: +4.5pt, n=1000: +3.7pt

### slope_window — accuracy (higher is better)
_the direction of drift inside one off-grid window, XOR a static gate_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.508±0.009   | 0.878±0.107*  | 0.997±0.001*  | 0.999±0.001*  |
| hw_phaseA    | 0.551±0.017   | 0.747±0.034   | 0.894±0.041   | 0.981±0.006   |
| hw_bank      | 0.516±0.008   | 0.672±0.063   | 0.971±0.022   | 0.999±0.001   |
| hw_filters   | 0.525±0.020   | 0.700±0.051   | 0.939±0.059   | 0.998±0.001   |
| agg          | 0.521±0.008   | 0.587±0.013   | 0.610±0.012   | 0.648±0.013   |
| wagg4        | 0.503±0.007   | 0.594±0.018   | 0.787±0.104   | 0.898±0.038   |
| wagg8        | 0.581±0.078   | 0.870±0.079   | 0.989±0.001   | 0.994±0.002   |
| wagg16       | 0.507±0.008   | 0.600±0.075   | 0.812±0.014   | 0.852±0.018   |
| raw_flat     | 0.502±0.008   | 0.563±0.018   | 0.656±0.025   | 0.790±0.034   |

vs agg (the standard workaround) — n=100: -1.3pt, n=250: +29.1pt, n=500: +38.6pt, n=1000: +35.1pt
vs best-of-all baselines (oracle choice) — n=100: -7.2pt, n=250: +0.7pt, n=500: +0.8pt, n=1000: +0.5pt

### amp_regression — rmse (lower is better)
_the height of a transient in one stretch, times a static coefficient_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.520±0.029   | 0.425±0.011   | 0.374±0.011*  | 0.350±0.003*  |
| hw_phaseA    | 0.572±0.008   | 0.439±0.010   | 0.393±0.008   | 0.368±0.003   |
| hw_bank      | 0.510±0.019   | 0.417±0.002   | 0.368±0.006   | 0.350±0.002   |
| hw_filters   | 0.536±0.031   | 0.434±0.024   | 0.374±0.011   | 0.358±0.003   |
| agg          | 0.848±0.010   | 0.798±0.003   | 0.724±0.024   | 0.689±0.006   |
| wagg4        | 0.516±0.023   | 0.418±0.007   | 0.380±0.002   | 0.359±0.002   |
| wagg8        | 0.640±0.036   | 0.494±0.032   | 0.418±0.012   | 0.382±0.005   |
| wagg16       | 0.702±0.041   | 0.520±0.007   | 0.447±0.020   | 0.396±0.008   |
| raw_flat     | 0.823±0.043   | 0.556±0.016   | 0.463±0.006   | 0.404±0.005   |

vs agg (the standard workaround) — n=100: +38.7%, n=250: +46.7%, n=500: +48.4%, n=1000: +49.2%
vs best-of-all baselines (oracle choice) — n=100: -0.7%, n=250: -1.7%, n=500: +1.6%, n=1000: +2.4%

### static_control — accuracy (higher is better)
_nothing temporal at all — the signal is entirely in the static columns_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.853±0.023   | 0.912±0.002   | 0.941±0.004   | 0.950±0.003   |
| hw_phaseA    | 0.857±0.021   | 0.923±0.004   | 0.943±0.004   | 0.952±0.001   |
| hw_bank      | 0.848±0.017   | 0.916±0.003   | 0.941±0.002   | 0.954±0.001   |
| hw_filters   | 0.848±0.015   | 0.914±0.002   | 0.934±0.004   | 0.951±0.002   |
| agg          | 0.860±0.014   | 0.917±0.003   | 0.942±0.003   | 0.954±0.002   |
| wagg4        | 0.851±0.015   | 0.910±0.004   | 0.939±0.004   | 0.952±0.001   |
| wagg8        | 0.839±0.021   | 0.907±0.003   | 0.938±0.006   | 0.954±0.003   |
| wagg16       | 0.841±0.014   | 0.903±0.010   | 0.934±0.003   | 0.952±0.002   |
| raw_flat     | 0.841±0.013   | 0.914±0.004   | 0.936±0.004   | 0.954±0.002   |

vs agg (the standard workaround) — n=100: -0.7pt, n=250: -0.4pt, n=500: -0.1pt, n=1000: -0.5pt
vs best-of-all baselines (oracle choice) — n=100: -0.7pt, n=250: -0.4pt, n=500: -0.1pt, n=1000: -0.5pt

### fit time (seconds, mean over the whole grid)

  hw_filters     28.14   (max 61.72)
  heartwood      18.24   (max 38.80)
  hw_bank        17.38   (max 35.04)
  hw_phaseA      13.91   (max 29.08)
  wagg16          0.83   (max 2.30)
  raw_flat        0.55   (max 1.62)
  wagg8           0.41   (max 1.11)
  wagg4           0.24   (max 0.58)
  agg             0.10   (max 0.28)

_macOS-26.5.2-arm64-arm-64bit, python 3.10.17, 13.4 min._
