# How Heartwood works

Three layers, and the interesting part is how the second and third divide the labour.

## 1. The split

An ordinary gradient-boosted tree asks `is column 7 above 0.42?`. Heartwood's nodes can ask
that, and can also ask questions *about a trajectory*:

| kind | the question it asks |
|---|---|
| `static` | is this attribute above a threshold? |
| `interval` | is the *slope* of channel 0 between t=12 and t=40 below 0.31? |
| `shapelet_dist` | does this shape occur anywhere in the series? |
| `shapelet_pos` | and how early does it occur? |
| `filter_resp` / `filter_pos` | does this learned template match, and where? |
| `product` | how large is this temporal feature, scaled by this static one? |

Every candidate is scored on the **same** second-order gain, so a node decides for itself
whether the next best question is about a customer attribute or the shape of their
trajectory. Temporal candidates are redrawn at every node of every round, so the window
that matters is discovered at whatever position and resolution the gradients call for,
rather than fixed by an up-front aggregation.

A split carries everything it needs to be re-evaluated — including its own copy of any
shapelet — so a fitted model never depends on the training arrays still existing.

## 2. The bank, and why the trees are not left to search alone

Greedy per-node selection has a structural problem. A node picks the single highest-gain
split from a pool of randomly drawn candidates, so **enlarging the pool raises the winner's
expected gain whether or not anything in it is informative.** That is the winner's curse,
and `validation/HEADROOM.md` measured it: ×16 the candidate budget moved a 20-point deficit
by 1.5 points and made two datasets of three *worse*.

MiniROCKET has no such ceiling because it never selects — it computes a large fixed bank of
dilated convolutions and lets a ridge shrink all of it jointly. So Heartwood builds that
bank (`heartwood/rocket.py`, following Dempster, Schmidt and Webb 2021) and puts a ridge
over it *underneath* the trees:

```
X_series ──▶ 10,000 dilated convolutions ──▶ ridge ──▶ margins ──┐
                                                                 ├──▶ trees ──▶ prediction
X_static ────────────────────────────────────────────────────────┘
```

The trees are boosted from the ridge's **leave-one-out** margins, not its in-sample fit.
That distinction is the difference between the trees learning what the ridge could not and
the trees learning nothing at all.

## 3. The ridge declines itself when it is not helping

Before its margins are used, the ridge asks whether it beat chance: permute the target,
refit, and see how large an R² this design produces from noise. If the honest
leave-one-out R² does not clear that null, the base switches off and the booster starts
from its ordinary initial score.

This is not decoration. On an XOR task where the series has zero marginal correlation with
the label, accepting a +0.008 R² cost 36 points of accuracy.

## Why leave-one-out is closed-form, and why that matters

The base is a ridge, so the leave-one-out and leave-one-**group**-out predictions are read
off the same spectrum the fit already produced — no refitting. For a group `G`:

```
LOGO_G = (I − H_GG)⁻¹ (fitted_G − H_GG · y_G)
```

This is verified against literally refitting without each group, to ~5e-15
(`tests/test_dense.py`).

That exactness is not a performance trick, it is this project's bug detector: it caught two
separate defects that had already produced confident wrong answers. It is also the reason
the base is a ridge rather than a second gradient-boosted model — a tree base would model
more and would cost the exact hold-out. `VALIDATION_V21.md` records that trade being made
deliberately, and tested: random Fourier features buy nonlinearity while keeping the fit
linear in what it fits, and were measured at 0 of 8 datasets and left off.

## Missing data

`NaN` is load-bearing almost everywhere. `interval_stat` skips it, threshold scans send it
down whichever branch scores better (the sparsity-aware trick), and a row with no series
data coexists with fully observed rows in the same model.

The one exception is the convolution bank, which has no NaN-aware form: values are imputed
per row and channel before convolving. Saying so is better than pretending the bank is
exact.
