# Benchmark results

Grid: scenarios × n_train [100, 250, 500, 1000] × 3 seeds, test n=2000. Every model gets 200 rounds, depth 4, learning rate 0.1; Heartwood runs on library defaults with no per-scenario tuning.

Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline at that size.

Two comparisons are reported because they answer different questions. **agg** is the global-aggregate workaround teams actually ship — beating it is the claim this library makes. **best-of-all baselines** is an oracle: it picks, per task and per training size, whichever of the five representations turned out best, which nobody can do in advance. Losing to that oracle on some task is expected; losing to `agg` would mean the premise is wrong.

### bump_order — accuracy (higher is better)
_which of two transients happened first, XOR a static flag_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.641±0.181*  | 0.958±0.019*  | 0.983±0.005*  | 0.996±0.000*  |
| hw_levy      | 0.641±0.181   | 0.958±0.019   | 0.983±0.005   | 0.996±0.000   |
| hw_dense     | 0.656±0.097   | 0.964±0.008   | 0.986±0.005   | 0.998±0.001   |
| hw_both      | 0.656±0.097   | 0.964±0.008   | 0.986±0.005   | 0.998±0.001   |
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
| hw_levy      | 0.959±0.013   | 0.977±0.005   | 0.987±0.004   | 0.991±0.001   |
| hw_dense     | 0.958±0.004   | 0.973±0.002   | 0.987±0.002   | 0.993±0.003   |
| hw_both      | 0.958±0.004   | 0.973±0.002   | 0.987±0.002   | 0.993±0.003   |
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
| hw_levy      | 0.508±0.009   | 0.878±0.107   | 0.997±0.001   | 0.999±0.001   |
| hw_dense     | 0.500±0.003   | 0.930±0.038   | 0.997±0.001   | 0.999±0.001   |
| hw_both      | 0.500±0.003   | 0.930±0.038   | 0.997±0.001   | 0.999±0.001   |
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
| hw_levy      | 0.520±0.029   | 0.425±0.011   | 0.374±0.011   | 0.350±0.003   |
| hw_dense     | 0.606±0.063   | 0.455±0.001   | 0.413±0.004   | 0.374±0.006   |
| hw_both      | 0.606±0.063   | 0.455±0.001   | 0.413±0.004   | 0.374±0.006   |
| agg          | 0.848±0.010   | 0.798±0.003   | 0.724±0.024   | 0.689±0.006   |
| wagg4        | 0.516±0.023   | 0.418±0.007   | 0.380±0.002   | 0.359±0.002   |
| wagg8        | 0.640±0.036   | 0.494±0.032   | 0.418±0.012   | 0.382±0.005   |
| wagg16       | 0.702±0.041   | 0.520±0.007   | 0.447±0.020   | 0.396±0.008   |
| raw_flat     | 0.823±0.043   | 0.556±0.016   | 0.463±0.006   | 0.404±0.005   |

vs agg (the standard workaround) — n=100: +38.7%, n=250: +46.7%, n=500: +48.4%, n=1000: +49.2%
vs best-of-all baselines (oracle choice) — n=100: -0.7%, n=250: -1.7%, n=500: +1.6%, n=1000: +2.4%

### lead_lag — accuracy (higher is better)
_which of two channels moved first, XOR a static flag_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.511±0.020*  | 0.535±0.004*  | 0.652±0.050*  | 0.628±0.040*  |
| hw_levy      | 0.511±0.020   | 0.535±0.004   | 0.652±0.050   | 0.628±0.040   |
| hw_dense     | 0.519±0.020   | 0.542±0.002   | 0.646±0.006   | 0.687±0.076   |
| hw_both      | 0.519±0.020   | 0.542±0.002   | 0.646±0.006   | 0.687±0.076   |
| agg          | 0.511±0.011   | 0.513±0.007   | 0.513±0.007   | 0.504±0.014   |
| wagg4        | 0.508±0.009   | 0.490±0.016   | 0.511±0.006   | 0.515±0.014   |
| wagg8        | 0.493±0.006   | 0.505±0.013   | 0.510±0.005   | 0.512±0.004   |
| wagg16       | 0.510±0.010   | 0.493±0.006   | 0.496±0.008   | 0.510±0.006   |
| raw_flat     | 0.498±0.007   | 0.506±0.015   | 0.499±0.005   | 0.494±0.013   |

vs agg (the standard workaround) — n=100: +0.0pt, n=250: +2.2pt, n=500: +14.0pt, n=1000: +12.3pt
vs best-of-all baselines (oracle choice) — n=100: +0.0pt, n=250: +2.2pt, n=500: +14.0pt, n=1000: +11.3pt

### static_control — accuracy (higher is better)
_nothing temporal at all — the signal is entirely in the static columns_

| model        | n=100         | n=250         | n=500         | n=1000        |
|--------------|---------------|---------------|---------------|---------------|
| heartwood    | 0.853±0.023   | 0.912±0.002   | 0.941±0.004   | 0.950±0.003   |
| hw_levy      | 0.853±0.023   | 0.912±0.002   | 0.941±0.004   | 0.950±0.003   |
| hw_dense     | 0.847±0.024   | 0.904±0.007   | 0.941±0.001   | 0.952±0.001   |
| hw_both      | 0.847±0.024   | 0.904±0.007   | 0.941±0.001   | 0.952±0.001   |
| agg          | 0.860±0.014   | 0.917±0.003   | 0.942±0.003   | 0.954±0.002   |
| wagg4        | 0.851±0.015   | 0.910±0.004   | 0.939±0.004   | 0.952±0.001   |
| wagg8        | 0.839±0.021   | 0.907±0.003   | 0.938±0.006   | 0.954±0.003   |
| wagg16       | 0.841±0.014   | 0.903±0.010   | 0.934±0.003   | 0.952±0.002   |
| raw_flat     | 0.841±0.013   | 0.914±0.004   | 0.936±0.004   | 0.954±0.002   |

vs agg (the standard workaround) — n=100: -0.7pt, n=250: -0.4pt, n=500: -0.1pt, n=1000: -0.5pt
vs best-of-all baselines (oracle choice) — n=100: -0.7pt, n=250: -0.4pt, n=500: -0.1pt, n=1000: -0.5pt

### fit time (seconds, mean over the whole grid)

  heartwood      18.24   (max 37.16)
  hw_dense       18.08   (max 36.88)
  hw_levy        17.99   (max 36.17)
  hw_both        17.97   (max 36.27)
  wagg16          0.97   (max 4.33)
  raw_flat        0.64   (max 2.55)
  wagg8           0.48   (max 2.13)
  wagg4           0.27   (max 1.03)
  agg             0.11   (max 0.33)

_macOS-26.5.2-arm64-arm-64bit, python 3.10.17, 15.7 min._
