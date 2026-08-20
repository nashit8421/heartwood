# Heartwood — Implementation Plan

> **Status:** architecture finalized after a 5-design × 3-judge exploration panel — see
> `ARCHITECTURES.md` for the candidates, scores, and rationale. The plan is now PHASED:
> **Phase A (v0.1)** = §1–§9 of this file, the core booster, implement first and exactly
> as written. **Phase B (v0.2)** = §10, three additive upgrades (matched-filter splits,
> winners-only feature bank, comparison splits). **Phase C (opt-in)** = §11.
> Do not start §10 until §6's Phase-A tests are green.

**Goal:** An XGBoost-style gradient boosting library that natively handles datasets with
**static (per-row) features + raw time series** together, without pre-aggregating the series.
Works well on small/medium data (hundreds to tens of thousands of rows), gives
interpretable models, and beats the industry-standard workaround
("aggregate the series → concat with static → XGBoost") on tasks where the signal
lives in the *temporal journey*.

**Package name:** `heartwood` · **Language:** Python 3.10+, NumPy only for the core
(scikit-learn / xgboost used only in benchmarks). No compiled code, no deep learning.

**Environment (verified):** Python 3.10.17, numpy 2.2.6, sklearn 1.7.1, xgboost 3.0.4.
`pytest` is NOT installed — install it before running tests (`pip install pytest`).

---

## 1. Core idea (why this works)

XGBoost is: second-order gradient boosting where each tree node greedily picks the
split with maximal gain over a set of candidate scalar features.

**Heartwood keeps the exact same boosting machinery** (same gain formula, leaf
weights, shrinkage, regularization) but enlarges the *split family*. At every tree
node, candidates come from three sources:

1. **Static splits** — ordinary `x[col] <= t` thresholds on static columns (identical to XGBoost).
2. **Interval-statistic splits** — sample random `(channel, window [s,e), statistic)` triples;
   the scalar feature is e.g. "slope of channel 2 over timesteps 12..40". Statistics:
   mean, std, min, max, slope, median, mean_abs_change, last, delta(last−first).
   Windows are sampled fresh at every node (log-uniform lengths, random positions,
   with some probability the full series so classical global aggregates are always in
   the hypothesis space — guarantees we can never do *worse* than the aggregate baseline
   in expressiveness).
3. **Shapelet splits** — sample short subsequences (shapelets) from the training series
   *inside the current node*; the scalar features are (a) the minimum sliding
   z-normalized distance from each row's series to the shapelet ("does this shape occur?")
   and (b) the argmin position normalized to [0,1] ("*when* does it occur?").
   Both are threshold-scanned like any numeric feature.

Because window/shapelet candidates are re-sampled per node per boosting round, the
model *adaptively learns which part of the temporal journey matters*, at whatever
resolution the gradients demand — instead of committing to fixed aggregates up front.
Randomized candidates + shrinkage + subsampling = variance control, which is exactly
why it stays strong on small data (same mechanism as random forests / MiniROCKET, but
gain-guided instead of blind).

Interpretability falls out for free: every split is human-readable, e.g.
`series[ch=0].slope[t=12:40] <= 0.31 (gain 12.4)`.

**Honest positioning** (put this in README): this is a synthesis of known ingredients
(XGBoost gain machinery; interval features from Time Series Forest/CIF; shapelets) into
one coherent, regularized booster with a unified split search over static + temporal
candidates. The novelty is the packaging and the per-node gradient-guided temporal
feature search; benchmark claims must be demonstrated, not asserted.

---

## 2. The math (must be implemented exactly)

Second-order boosting, identical to XGBoost:

- Model output after M rounds: `raw_i = base_score + η · Σ_m f_m(x_i)` (η = learning_rate).
- Per round, compute gradient `g_i = ∂l/∂raw_i` and hessian `h_i = ∂²l/∂raw²_i` at current raw.
- For a node with sample set S: `G = Σ g_i`, `H = Σ h_i`.
- **Leaf weight:** `w = −G / (H + λ)` (λ = reg_lambda). Booster adds `η·w`.
- **Split gain:** `gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ`.
  Accept a split only if `gain > 1e-12` and constraints hold:
  `H_L, H_R ≥ min_child_weight`, `n_L, n_R ≥ min_samples_leaf`.
- **Missing values (NaN feature values):** sparsity-aware like XGBoost — try sending the
  NaN group left and right, keep whichever gives higher gain; store `missing_left: bool`
  on the split and use it at predict time too.

### Losses

| Loss | init base_score | g | h | metric |
|---|---|---|---|---|
| SquaredError (K=1) | mean(y) | raw − y | 1 | rmse |
| Logistic (K=1, y∈{0,1}) | log(p̄/(1−p̄)), p̄=clip(mean(y),1e-6,1−1e-6) | p − y, p=σ(raw) | clip(p(1−p), 1e-16, ∞) | logloss (clip p to [1e-15, 1−1e-15]) |
| Softmax (K classes) | log(clip(class priors,1e-6,·)) | p − onehot(y) | clip(p(1−p), 1e-16, ∞) | mlogloss |

Softmax must be numerically stable: subtract rowwise max before exp.
Raw predictions always have shape `(n, K)` with K=1 for regression/binary — keeps the
booster loop uniform. Multiclass fits K trees per round (one per class, standard
diagonal-hessian softmax boosting).

---

## 3. Data model & public API

```python
from heartwood import HeartwoodClassifier, HeartwoodRegressor

model = HeartwoodClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, ...)
model.fit(X_static, X_series, y, eval_set=None, verbose=False)
model.predict(X_static, X_series)
model.predict_proba(X_static, X_series)          # classifier only
model.feature_importances()                       # dict: readable family -> total gain
model.dump_splits()                               # list of (description, gain), sorted desc
```

- `X_static`: `(n, p)` float array, NaN allowed. May be `None` (pure time-series mode → treated as `(n, 0)`).
- `X_series`: one of
  - `(n, C, T)` float array (canonical form),
  - `(n, T)` array → reshaped to `(n, 1, T)`,
  - Python list of per-sample arrays `(C, T_i)` or `(T_i,)` with **variable lengths** →
    right-pad with NaN to max T (NaN = missing everywhere downstream),
  - `None` (degenerate pure-static mode → temporal candidates skipped; model becomes a
    plain XGBoost-style booster; must still work).
- At least one of `X_static` / `X_series` must be non-None. Everything is converted to
  float64 C-contiguous once at fit time.
- `y`: classifier does `classes_ = np.unique(y)` and label-encodes; 2 classes → Logistic,
  ≥3 → Softmax. Regressor casts to float64.
- `eval_set=(Xst_val, Xse_val, y_val)` + `early_stopping_rounds` → track best iteration
  by eval metric, stop after patience exhausted, predict with `best_iteration`.
- Reproducibility: `random_state` seeds one `np.random.default_rng` master; every tree
  gets `np.random.default_rng(master.integers(2**63))`.

### Hyperparameters (defaults chosen small-data-friendly)

```
n_estimators=200, learning_rate=0.1, max_depth=4,
reg_lambda=1.0, gamma=0.0, min_child_weight=1e-3, min_samples_leaf=5,
subsample=1.0,                    # row subsample per round, without replacement
colsample=1.0,                    # fraction of static cols considered per node
n_interval_candidates=16,         # per node
n_shapelet_candidates=4,          # per node (each yields TWO features: dist + position)
interval_stats=("mean","std","min","max","slope","median","mean_abs_change","last","delta"),
full_interval_prob=0.25,          # prob an interval candidate is the whole series
min_interval_len=3,
shapelet_min_len=3, shapelet_max_frac=0.5, shapelet_znorm=True,
early_stopping_rounds=None, random_state=None
```

---

## 4. Package layout

```
TS_XGBoost/
  heartwood/
    __init__.py        # exports Classifier, Regressor, __version__="0.1.0"
    _util.py           # small helpers (validation, rng spawn)
    losses.py          # SquaredError, Logistic, Softmax
    features.py        # interval_stat(), shapelet_features(), eval_split_feature()
    splits.py          # SplitSpec dataclass, scan_threshold(), candidate generation
    filters.py         # Phase B (§10.1): pyramid, NCC, matched-filter fit — empty in v0.1
    tree.py            # TemporalTree (recursive builder + predict + dump)
    booster.py         # _BoosterCore: fit loop, early stopping, predict_raw
    api.py             # HeartwoodClassifier / HeartwoodRegressor (validation, encoding)
    datasets.py        # synthetic generators (used by tests AND benchmarks)
  tests/               # pytest; see §6
  benchmarks/
    baselines.py       # aggregate-features + XGBoost/HistGB baselines
    scenarios.py       # benchmark scenario registry built on heartwood.datasets
    run_benchmarks.py  # CLI: runs all scenarios × sample sizes, prints/saves table
  examples/quickstart.py
  README.md
  PLAN.md              # this file
  pyproject.toml       # name=heartwood, requires numpy>=1.24; [test] pytest; [bench] scikit-learn,xgboost
```

Implementation order: `losses → features → splits → tree → booster → api → datasets → tests → benchmarks → README`.

---

## 5. File-by-file spec

### 5.1 `losses.py`

Each loss class: `n_outputs(y) -> int`, `init_score(y) -> (K,)`, `grad_hess(y, raw) -> (g, h)`
both `(n, K)`, `transform(raw)` (identity / sigmoid / softmax), `metric_name`,
`eval_metric(y, raw) -> float`. Keep pure NumPy, no loops over rows.

### 5.2 `features.py`

```python
STAT_NAMES = ("mean","std","min","max","slope","median","mean_abs_change","last","delta")

def interval_stat(sub: np.ndarray, stat: str) -> np.ndarray
    # sub: (m, L) possibly containing NaN. Returns (m,), NaN where undefined.
```

**All stats must be NaN-aware via explicit masks** (`m = np.isfinite(sub)`,
`n = m.sum(1)`), never bare `np.nanmean` on possibly all-NaN rows (RuntimeWarning spam).
Rows with 0 valid points → NaN (→ handled as missing by the split). Details:

- mean: `sum(where(m,sub,0),1)/n` where n>0.
- std: population std via `E[x²]−mean²`, clip variance at 0 before sqrt.
- min/max: `where(m, sub, ±inf)` then min/max; NaN where n==0.
- slope: OLS slope vs t=0..L−1 using masked sums:
  `slope=(n·Σtx − Σt·Σx)/(n·Σt² − (Σt)²)`; NaN where n<2 or denom ≤ 1e-12.
- median: `np.nanmedian` wrapped in `warnings.catch_warnings()` suppressing RuntimeWarning; rows all-NaN → NaN.
- mean_abs_change: `d=|diff(sub,axis=1)|` (NaN propagates), masked mean of d; NaN if <1 valid diff.
- last: value at last finite index per row (`L−1−argmax(m[:,::-1],1)`); **only assign for rows with n>0** (argmax on all-False returns 0 → wrong index; use `rows=np.where(n>0)[0]`).
- delta: `last − first` (first finite via `argmax(m,1)`), same guard.

```python
def shapelet_features(X2d, shp, znorm=True, chunk_bytes=64<<20) -> (dist, pos)
    # X2d: (m, T); shp: (l,); returns two (m,) arrays.
```

- Sliding windows via `np.lib.stride_tricks.sliding_window_view(X2d, l, axis=1)` → `(m, P, l)`, `P = T−l+1`.
- **Chunk over rows** so the window tensor stays under `chunk_bytes` (`rows_per_chunk = max(1, chunk_bytes // (P*l*8))`).
- z-norm: window mean/std along last axis; if `std ≤ 1e-12` the normalized window is all zeros (same for the shapelet, normalized once).
- distance = mean squared difference between normalized window and normalized shapelet → `(m, P)`.
- **CRITICAL NaN pitfall:** with z-norm, a window touching NaN gets NaN mean/std, and
  `np.where(sd>eps, ...)` on NaN sd is False → window silently becomes zeros → finite
  bogus distance. You MUST explicitly mask: `nanwin = np.isnan(W.sum(-1))` and set
  `d[nanwin] = np.inf` BEFORE taking the min. If all positions are inf for a row →
  dist=NaN, pos=NaN (missing).
- `dist = min over P`, `pos = argmin/(P−1)` (if P==1, pos=0.0).

```python
def eval_split_feature(spec, X_static, X_series, rows) -> np.ndarray  # (len(rows),)
```
Shared by tree fit and predict: dispatch on `spec.kind`
(`static` → `X_static[rows, spec.col]`; `interval` → `interval_stat(X_series[rows, spec.channel, spec.start:spec.end], spec.stat)`;
`shapelet_dist`/`shapelet_pos` → `shapelet_features(...)` picking element 0 or 1).

### 5.3 `splits.py`

```python
@dataclass
class SplitSpec:
    kind: str            # 'static' | 'interval' | 'shapelet_dist' | 'shapelet_pos'
    threshold: float = np.nan
    missing_left: bool = True
    gain: float = -np.inf
    col: int = -1                                   # static
    channel: int = -1; start: int = -1; end: int = -1; stat: str = ""   # interval + shapelet channel
    shapelet: np.ndarray | None = None; znorm: bool = True              # shapelet
    def describe(self) -> str   # e.g. "series[ch=1].slope[t=12:40] <= 0.310"
```

```python
def scan_threshold(f, g, h, reg_lambda, gamma, min_child_weight, min_samples_leaf)
    -> (gain, threshold, missing_left) | None
```

Vectorized exact scan (no Python loop over thresholds):

1. `miss = ~np.isfinite(f)`; `Gm, Hm, nm` = sums/count over missing.
2. Sort non-missing `fv` (stable argsort); cumsum sorted g and h → `cg, ch`.
3. Candidate cut positions: `cut = np.nonzero(fs[:-1] < fs[1:])[0]` (split *after* index i;
   threshold = midpoint `(fs[i]+fs[i+1])/2`). No cuts → None.
4. `G = cg[-1]+Gm`, `H = ch[-1]+Hm`, `parent = G²/(H+λ)`.
5. For `missing_left in (True, False)` (skip False if nm==0):
   `GL = cg[cut] (+Gm)`, `HL = ch[cut] (+Hm)`, `nL = cut+1 (+nm)`; `GR = G−GL` etc.;
   validity mask from constraints; `gains = ½(GL²/(HL+λ)+GR²/(HR+λ)−parent) − γ`,
   invalid → −inf; take argmax; keep best over the two scenarios.
6. Return None unless best gain > 1e-12.

Candidate generation (a generator method on the tree, but the samplers live here):

```python
def sample_interval(T, rng, min_len, full_prob) -> (start, end)
    # with prob full_prob return (0, T); else length log-uniform in [min_len, T]:
    # l = clip(round(exp(uniform(log(min_len), log(T)))), min_len, T); start = rng.integers(0, T-l+1)

def sample_shapelet(X_series, rows, rng, min_len, max_frac) -> (channel, shp) | None
    # donor row from rows, channel uniform, length log-uniform in [min_len, max(min_len+1, int(max_frac*T))],
    # position uniform; RETRY up to 10 times if the cut subsequence has any non-finite
    # value or std < 1e-12 (constant); return None after 10 failures.
    # If T < min_len + 1: return None (shapelets disabled for very short series).
```

### 5.4 `tree.py`

```python
class TemporalTree:
    def fit(self, X_static, X_series, g, h, rows, rng): ...   # g,h are (n,) for this output
    def predict(self, X_static, X_series) -> (n,)             # raw leaf weights (no η)
    def iter_splits(self) -> iterator of SplitSpec            # for importances
```

Recursive `_build(rows, depth)`:

1. `G, H = g[rows].sum(), h[rows].sum()`; leaf value `= −G/(H+λ)`.
2. Leaf if: `depth ≥ max_depth` or `len(rows) < 2·min_samples_leaf` or `H < 2·min_child_weight`.
3. Enumerate candidates, each `(f_values_over_rows, partial SplitSpec)`:
   - static: columns (all, or a `colsample` fraction sampled without replacement, ≥1);
   - `n_interval_candidates` × (random channel, `sample_interval`, random stat from `interval_stats`);
   - `n_shapelet_candidates` × `sample_shapelet`, each yielding **two** candidates
     (dist and pos features share one distance computation — compute once, yield twice).
4. `scan_threshold` each; keep the best spec (fill threshold/gain/missing_left) **and its
   feature values** (avoid recomputation for routing).
5. No positive-gain split → leaf. Else route:
   `go_left = np.where(np.isfinite(f), f <= thr, missing_left)`; guard: if either side
   is empty → leaf (shouldn't happen given constraints, but guard anyway); recurse.

Predict: recursive with row-index subsets; at each internal node compute the feature via
`eval_split_feature` only for the rows that reached that node; leaves write `value` into
the output array. No per-row Python loop.

### 5.5 `booster.py`

```python
class _BoosterCore:
    def fit(self, X_static, X_series, y_enc, loss, params, eval_set, verbose): ...
    def predict_raw(self, X_static, X_series, iteration=None) -> (n, K)
```

- `raw = np.tile(loss.init_score(y), (n, 1))`.
- Per round: `(g, h) = loss.grad_hess(y, raw)`; row subsample (without replacement,
  same rows for all K trees this round); fit K trees on `g[:,k], h[:,k]`;
  `raw[:,k] += η · tree.predict(train)`. Store trees as `list[round][k]`.
- Eval set: maintain `raw_val` incrementally the same way; after each round compute
  `loss.eval_metric`; track `best_score/best_iteration`; stop when no improvement for
  `early_stopping_rounds` rounds. `predict_raw` uses `trees[:best_iteration+1]` when
  early stopping happened, else all.
- `verbose=True` → print `[round] metric` every 10 rounds.

### 5.6 `api.py`

Validation rules from §3. Classifier: `predict_proba` returns `(n, 2)` for binary
(`column_stack([1−p, p])`) and `(n, K)` for multiclass; `predict` maps argmax back
through `classes_`. Importances: aggregate `spec.gain` by readable family key —
static → `static[<col>]`, interval → `interval(ch=<c>, <stat>)`,
shapelet → `shapelet_dist(ch=<c>)` / `shapelet_pos(ch=<c>)`; `dump_splits()` returns
every split's full `describe()` + gain, sorted descending.

### 5.7 `datasets.py` — synthetic generators (tests + benchmarks share these)

Each returns `(X_static, X_series, y)`, deterministic under `seed`. Static blocks always
include a few pure-noise distractor columns. Design targets: the temporal signal must be
**destroyed or diluted by global aggregation** so the aggregate baseline measurably fails,
while remaining learnable from a few hundred rows.

**Design rule learned in M1 (do not violate):** a scenario is only valid if the
aggregate baseline actually *fails* on it. Two of the first drafts were solvable by
global statistics and had to be redesigned — always verify against the baseline before
trusting a scenario (see §8.1).

1. `make_bump_interaction(n, T=100, noise=1.0, seed, amp=3.0)` — binary. Every series
   contains one **doublet** transient (two opposite Gaussian lobes, σ=T/50, separation
   2σ) at a random position, oriented either up-then-down or down-then-up. Static flag
   `s∈{0,1}`; label `y = is_down XOR s`.
   *Why aggregation provably fails:* the two orientations are exact negations, so both
   classes have identical global value distributions (mean, std, min, max, median,
   mean_abs_change all uninformative); the doublet has zero net area so it does not move
   the mean; the only leak is its dipole into the global slope, which is an order of
   magnitude below the slope's own noise. Correlation between classes is −1, so one
   z-normalised template separates them. Verified: agg 0.536 / wagg 0.526 (chance).
2. `make_timing_task(n, T=100, noise=0.5, seed)` — binary. Every series has the same
   bump; label depends on **when**: `y = 1 iff bump_center < deadline`, where `deadline`
   is a static column (uniform in [0.2T, 0.8T]). Global aggregates carry no timing;
   shapelet_pos plus the static column recover it. Verified: agg 0.827, wagg 0.945.
3. `make_slope_window(n, T=120, noise=0.8, seed)` — binary. Piecewise series; label =
   sign of slope **only within t∈[0.38T, 0.58T)**; other segments have random, steeper
   distracting slopes so the *global* slope aggregate is diluted. The window is
   deliberately **off-grid** — an earlier draft used exactly [T/2, 3T/4), which a
   4-equal-window baseline lands on precisely (it scored 0.997). Interacts with one
   static column (label flips when static > 0). Verified: agg 0.637, wagg 0.879.
4. `make_shape_amplitude_regression(n, T=100, noise=0.5, seed)` — regression.
   `y = amplitude_of_signal_bump × static_coef + ε`; amplitude U[0.5, 3], and the signal
   bump always sits in the off-grid stretch [0.56T, 0.66T]. A **larger nuisance bump**
   (amplitude U[3, 5]) sits anywhere, so the global max reports the nuisance and says
   nothing about the target — the model must learn *where* to measure, not just what.
   Verified (RMSE): agg 0.743, wagg 0.628.
5. `make_pure_static(n, p=10, seed)` — control, no series (X_series=None). Plus
   `make_static_plus_noise_series` — same labels with a pure-noise series attached, to
   check that temporal candidates cost nothing when there is nothing to find.

### 5.8 Benchmarks

`baselines.py`:
- `aggregate_features(X_series)` → per channel: mean, std, min, max, slope, median,
  mean_abs_change, first, last, delta (the classic practitioner move).
- `windowed_aggregate_features(X_series, n_windows=4)` → same stats per equal window
  (a *stronger* baseline — be fair).
- Baseline models: `xgboost.XGBClassifier/Regressor` (installed; fall back to sklearn
  `HistGradientBoosting*` if import fails) on `[static ‖ aggregates]` and
  `[static ‖ windowed aggregates]`, plus `[static ‖ flattened raw series]` as a third baseline.

`run_benchmarks.py`:
- Grid: every scenario × n ∈ {100, 250, 500, 1000} (train), fixed test n=2000,
  3 seeds each; report mean ± std.
- Metrics: classification → accuracy, F1, ROC-AUC, precision, recall (the user's pain
  points); regression → RMSE, MAE, R².
- Heartwood run with defaults (no per-scenario tuning; baselines get equivalent
  budget: same n_estimators/depth/lr).
- Output: aligned text table to stdout AND `benchmarks/results.json` +
  `benchmarks/results.md`. Also record wall-clock fit time per model.
- Runtime budget: full grid should complete in ≲15 min on a laptop; if a scenario is
  slow, drop to n_estimators=150 for ALL models equally.

**Acceptance targets:** on scenarios 1–4 Heartwood beats the best baseline by a
clear margin (≥ +5 accuracy/F1 points or ≥ 10% RMSE reduction at n=500); on the pure
static control it is within noise of XGBoost (±2 points). If a target is missed,
investigate before shipping — likely knobs: n_interval/shapelet candidates, max_depth,
learning_rate, subsample.

---

## 6. Test plan (`tests/`, pytest)

`test_losses.py`
- grad/hess of each loss matches finite-difference gradients of the metric/objective
  (rtol 1e-4) at random raw values; softmax rows of `transform` sum to 1; stability at
  extreme raw (±50) — no inf/NaN.

`test_features.py`
- interval stats vs brute-force per-row loops on random data WITH NaNs (allclose, equal_nan=True).
- all-NaN rows → NaN for every stat; no warnings emitted (use `warnings.catch_warnings(record=True)`).
- slope of exact line = its slope; slope of constant = 0.
- shapelet: planting the shapelet exactly in a series → dist ≈ 0 and pos ≈ plant
  position/(P−1); NaN-padded windows never win (the §5.2 pitfall — regression-test it
  explicitly); chunking path (tiny chunk_bytes) equals unchunked result.

`test_splits.py`
- scan_threshold vs brute-force O(n²) reference on random g/h/f including NaNs and
  duplicated feature values; verifies gain, threshold, missing_left.
- constraints respected (min_samples_leaf, min_child_weight); constant feature → None.

`test_tree.py`
- single tree on separable data reaches pure leaves; max_depth honored; leaf values
  equal −G/(H+λ) computed by hand on a tiny fixture; routing consistent between fit
  and predict (predict on train reproduces the leaf assignment).

`test_booster_api.py`
- overfit sanity: 60-sample bump task, train logloss → < 0.05 with enough rounds.
- determinism: same random_state → identical predictions; different → different.
- early stopping triggers on noisy eval set and best_iteration < n_estimators.
- binary + multiclass (3-class) + regression shapes and `classes_` round-trip.
- variable-length list input == same data pre-padded with NaN manually.
- X_static=None and X_series=None modes both fit and predict.
- static-only mode sanity vs sklearn HistGB on a static dataset (within a few points —
  not a strict bound, smoke-level).
- importances/dump: non-empty, gains positive, descriptions contain expected substrings
  (e.g. "slope" appears for `make_slope_window`).

`test_datasets.py` — shapes, determinism per seed, label balance sane (0.25–0.75),
an oracle using the true generative feature achieves high accuracy (validates the
generators actually contain signal).

### Phase-B mandatory tests (`test_filters.py`, `test_bank.py`, `test_comparison.py`)

These encode the judge panel's implementation warnings (ARCHITECTURES.md §5). They are
non-negotiable — do not run Phase-B benchmarks until they are green.

- **NaN masking at every scale:** plant NaNs in series; at every pyramid scale, verify
  a window touching NaN can never be the argmax of |NCC| (mask to −inf BEFORE argmax);
  all-NaN pooling windows yield NaN (never 0).
- **Iteration-0 equivalence:** with `n_alt=0` and unit-normed windows, verify
  `dist = 2·(1 − ρ)` reproduces `shapelet_features`' distances to allclose tolerance
  (same shapelet, scale 0). This validates the NCC kernel itself.
- **Planted template:** insert a known smooth template into a series at a known
  position at each scale → signed response ≈ 1 (or ≈ −1 for inverted polarity) at the
  plant position; position feature ≈ planted position/(P−1).
- **Gain-leak ban:** assert the fitted filter's split is chosen by `scan_threshold` on
  its emitted feature column only; unit-test that the stored β is a `.copy()` (mutating
  the training array after fit must not change predictions).
- **DC exclusion:** the DCT basis contains no DC component; a constant-window refit
  must not produce NaN/inf or a degenerate β.
- **Fit/predict symmetry:** predict-on-train reproduces fit-time leaf assignments
  exactly for trees containing filter, bank, and comparison splits.
- **Bank integrity:** a promoted feature's column equals recomputing its spec from
  scratch on all rows; eviction never drops a spec referenced by any tree; bank respects
  its cap; model with bank enabled and disabled produce identical predictions when no
  candidate ever wins twice (degenerate case).
- **Comparison splits:** on `make_timing_task`, a single comparison split achieves the
  oracle rule (event position ECDF vs deadline ECDF) on clean data; ECDFs computed on
  train only and reused frozen at predict.
- **Pure-static control:** with ALL Phase-B machinery enabled on a static-only signal
  dataset (noise series attached), accuracy within ±2 points of xgboost.

### Phase-C mandatory tests (only if §11 is implemented)

- **LOO correctness:** train-time base margin equals closed-form
  `(ŷᵢ − hᵢᵢ·yᵢ)/(1 − hᵢᵢ)` computed by explicit refit-without-row-i on a tiny dataset.
- **Leak test:** with pure-noise labels, eval metric stays at chance level (a leaky
  stack shows train-like performance on eval).
- **Lévy areas:** closed-form check on a synthetic pair of paths (e.g., quarter-circle)
  against the analytic signed area; invariance to NaN padding.

Smoke bar for CI: `python -m pytest tests/ -x -q` green, and
`python examples/quickstart.py` runs end-to-end in < 60 s.

---

## 7. Performance notes (keep it honest, don't optimize prematurely)

- Everything vectorized NumPy; the only Python-level loops are per-node/per-candidate.
- Cost per boosting round ≈ Σ_levels [ n · (16 interval·L̄ + 4 shapelet·P·l̄) ] — for
  n=500, T=100, depth 4, 200 rounds this is seconds-to-~1 min. Fine for v0.1.
- Knobs already in the API (candidates counts, subsample) are the perf levers; document
  them in README. Numba/multithreading is explicitly OUT of scope for v0.1.

## 8. Milestones

1. **M1 core (Phase A):** losses, features, splits, tree, booster, api, `__init__`,
   pyproject — quickstart runs on `make_bump_interaction`, train loss decreases
   monotonically-ish.
2. **M2 quality (Phase A):** full §6 Phase-A test suite green (incl. finite-difference
   and brute-force refs).
3. **M3 proof (Phase A):** benchmark suite runs; results table meets §5.8 acceptance
   targets; README with usage, algorithm explanation, results table, honest limitations
   (no GPU, single-thread, equal-length-after-padding, v0.1 scope).
4. **M4 upgrades (Phase B):** implement §10 in order (filters → bank → comparison
   splits); §6 Phase-B mandatory tests green after each; re-run benchmarks with the
   ablation grid (core / +filters / +bank / +comparison) — each addition must not
   regress any scenario by more than noise, and the pure-static control must stay
   within ±2 points of XGBoost with everything enabled.
5. **M5 extras (Phase C, optional):** §11 behind flags, each with its own ablation row
   and the §6 Phase-C tests.

## 8.1 M1 results and findings (recorded 2026-08-20) — **read before M2/M3**

**Status: M1 COMPLETE.** All modules implemented; `examples/quickstart.py` runs in 28 s
and shows train logloss 0.674 → 0.040. Two runnable brute-force suites live in
`tests/smoke/` (plain `python`, no pytest needed) and pass with zero failures:
`smoke_core.py` (interval stats, shapelet distances and `scan_threshold` vs slow
references, incl. all the NaN cases) and `smoke_model.py` (estimator mechanics).

**Measured at n_train=500, n_test=2000, 150 rounds, single seed** (development run, not
the M3 benchmark):

| scenario | agg | wagg | Heartwood |
|---|---|---|---|
| bump_xor (accuracy) | 0.536 | 0.526 | **0.742** (+20.7) |
| timing (accuracy) | 0.827 | 0.945 | **0.970** (+2.5) |
| slope_window (accuracy) | 0.637 | 0.879 | **0.928** (+4.9) |
| amp_regression (RMSE ↓) | 0.743 | 0.628 | **0.553** (−12%) |
| static control (accuracy) | 0.943 | 0.945 | 0.945 (tie ✓) |

Against §5.8 acceptance targets: bump ✓✓, slope ✓ (+4.9 ≈ +5), regression ✓ (−12% > 10%),
control ✓. **Timing is +2.5, short of +5 vs the windowed baseline — an M3 item.** The
model must currently staircase "position vs deadline" across several depths; Phase B's
comparison splits (§10.3) are exactly the fix, so expect this to close in M4.

**Finding 1 — the candidate lottery is real and quantified.** On the bump scenario the
shapelet family is *correct* (the true template gives AUC 0.960; the best template drawn
from an actual training row gives 0.953) but only **8.5% of random draws reach AUC > 0.80**
and the median draw is 0.536, i.e. useless. Raising `n_shapelet_candidates` 4 → 32 (with
intervals reduced to 8) lifts bump accuracy 0.742 → **0.896** at ~4× fit time. This is
direct empirical support for Phase B: fit templates to the gradients (§10.1) and *reuse*
whatever won (§10.2) instead of redrawing blindly.
Do **not** simply raise the default budget: across the other four scenarios 4 / 8 / 16
candidates are a wash (differences ≤ 0.02), so more candidates is pure cost there. The
right lever is targeting, not volume. Plan defaults were left unchanged.

**Finding 2 — pure XOR at the root is the weak spot.** At n=300 the bump scenario sits at
chance (0.50) while n=500 reaches 0.742. With an XOR, *no* candidate has marginal gain at
the root, so the first split is chosen essentially at random from ~29 candidates — a
larger random pool makes greedy XOR discovery harder than plain XGBoost's small fixed
pool. Worth an explicit M3 ablation (`colsample`, candidate counts, deeper first tree).

**Implementation note — `shapelet_features` deviates from §5.2's letter.** It computes
the distance through the algebraic expansion `mean(w²) − 2·w·s/l + mean(s²)` over
cumulative sums plus a loop over the template's taps, instead of materialising the
`(m, P, l)` window tensor. Mathematically identical, and verified against a brute-force
reference for both `znorm=True/False` including NaN rows — but it avoids multi-GB
temporaries. `chunk_bytes` still chunks (over the `(m, P)` arrays) so the M2 chunking
test remains meaningful. Caveat documented in the docstring: variance via `E[x²] − E[x]²`
would lose precision on series whose values dwarf their fluctuations.

**Deviations from spec: none other than the above.** Defaults, math, API, and the pitfalls
checklist in §9 were followed as written; every §9 item was exercised by the smoke suites.

## 9. Known pitfalls checklist (re-read before coding each file)

- [ ] Shapelet z-norm NaN-window bug (§5.2) — the one true silent-corruption risk.
- [ ] `argmax` on all-False mask returns 0 — guard every first/last-valid-index computation.
- [ ] Missing-direction must be applied identically at fit routing and predict routing.
- [ ] Gain must use the SAME G/H totals including the missing group in parent and children.
- [ ] Duplicate feature values: never split between equal values (`fs[i] < fs[i+1]` strictly).
- [ ] `np.argsort(..., kind="stable")` for determinism.
- [ ] Store the shapelet ARRAY in the SplitSpec (predict needs it), copy it (`.copy()`)
      so it doesn't alias the training matrix.
- [ ] Bake nothing about η into trees; booster applies η when accumulating raw.
- [ ] float64 throughout; C-contiguous conversion once in api.py.

---

# PHASE B (v0.2) — three additive upgrades

Origin: the architecture panel (ARCHITECTURES.md). All three keep §1–§9 untouched and
plug into the existing candidate enumeration in `tree.py`. Implement in this order,
running the §6 Phase-B tests after each. Each lives behind a parameter and defaults ON
in v0.2. New module: `heartwood/filters.py`; bank and comparison logic extend
`splits.py`/`tree.py`/`booster.py`.

## 10.1 Matched-filter candidate family (replaces blind shapelet sampling)

**Idea:** instead of sampling shapelets and hoping, each node FITS a small temporal
template to its own Newton residuals in closed form, at multiple time scales. The
family strictly contains the v0.1 shapelet family (see equivalence below), so risk is
monotone.

**Pyramid (build once per fit and once per predict, in `filters.py`):**
`pyramid[s]` for s = 0,1,2,… is the series NaN-aware average-pooled by factor 2^s:
level s is built from level s−1 by pairwise means where a pair's mean uses only finite
members and an all-NaN pair yields NaN (never 0). Keep scales while pooled length
`T_s ≥ filter_len + 1` (at least 2 sliding positions).

**NCC definition (one function, reused everywhere):** for window w and filter β,
`ρ = ((w−mean(w)) · (β−mean(β))) / (‖w−mean(w)‖ · ‖β−mean(β)‖)`; if either norm
≤ 1e-12 → ρ = 0. Windows containing any NaN are INVALID: they must be excluded from
the argmax entirely (mask before argmax — §5.2 pitfall, now at every scale). For
z-normalized w and β this equals the cosine `(w_n·β_n)/l`, hence the exact identity
**`z-norm distance = 2·(1 − ρ)`** — the §6 iteration-0 equivalence test.

**Per-node candidates (`n_filter_candidates = 8`, of which `n_fitted = 4`):**
For each candidate:
1. Sample channel c uniform; scale s uniform over valid scales.
2. **SEED:** sample a donor row from the node with probability ∝ `h_i·r̃_i²` where
   `r_i = −g_i/(h_i+λ)` and `r̃ = r − (h-weighted mean of r)`; fall back to uniform if
   the weights are degenerate (all ~0). Cut a random fully-finite window of length
   `filter_len = 9` from `pyramid[s][donor, c]`; retry up to 10 times if non-finite or
   constant; give up → skip candidate.
   - *Unfitted candidate (4 of 8):* β = the raw z-normalized snippet. STOP here
     (this is the incumbent shapelet behavior, at pyramid scales).
   - *Fitted candidate (4 of 8):* project the z-normed snippet onto the smooth basis:
     β⁰ = B Bᵀ w_z, normalized (B below).
3. **ALIGN:** via `sliding_window_view` on `pyramid[s][rows, c]`, compute ρ for every
   valid window; `t*_i = argmax_t |ρ_i(t)|` over valid windows only;
   `f_resp_i = ρ_i(t*_i)` (SIGNED — opposite-polarity occurrences matter),
   `f_pos_i = t*_i/(P_s−1)` (0.0 if P_s = 1). Rows with no valid window → both NaN.
4. **REFIT (fitted candidates only, `n_alt = 1` pass):** let V = node rows with valid
   alignment; if |V| < K+2 skip refit. Stack the aligned z-normed windows W (|V|×l).
   Basis B (l×K): DCT-II components k = 1..K (**k=0 / DC EXCLUDED** — windows are
   mean-zero; including DC silently degenerates the solve), columns orthonormalized.
   Solve the h-weighted ridge with unpenalized intercept:
   `(Zᵀ diag(h) Z + λ_β·diag(0,1,…,1)) θ = Zᵀ diag(h) r̃`, Z = [1 ‖ W B],
   θ = (a, c₁..c_K), defaults K = 5, λ_β = 1.0. Set β ← B c / ‖B c‖ (if ‖B c‖ ≤ 1e-12
   keep the previous β). Re-run ALIGN once with the new β.
5. **EMIT two features** into the ordinary candidate tournament: `f_resp` (kind
   `filter_resp`) and `f_pos` (kind `filter_pos`). They are scored by `scan_threshold`
   EXACTLY like any feature — never by the refit's residual fit (§6 gain-leak ban).
   Winning specs store `(kind, channel, scale, β.copy(), threshold, missing_left)`.

**Predict:** at a filter node, build (or reuse a cached) pyramid level for the rows
reaching the node, one ALIGN pass with the stored β, take resp or pos, threshold with
stored `missing_left`. Same ε rules and signed convention as fit — bit-for-bit.

**Defaults change:** with the filter family ON, `n_shapelet_candidates` drops to 0
(the code path stays; the ablation grid re-enables it once to prove nothing was lost).
`n_interval_candidates = 16` stays verbatim — intervals own amplitude/level signals;
filters are z-normed and amplitude-blind by design.

**describe():** `series[ch=0]·filter(scale=2, span≈36).resp <= 0.42` and
`….pos <= 0.31`; span = filter_len·2^s raw timesteps.

## 10.2 Winners-only feature bank (training-time reuse cache)

**Idea (GRAFT stripped to its safe core):** any temporal candidate (interval, shapelet,
filter_resp/filter_pos) that WINS a node split gets its feature column materialized on
ALL n rows and re-offered as a zero-marginal-cost candidate at every later node and
round — discovered features are refined across the ensemble instead of rediscovered.

**Mechanics:**
- `BankEntry = (spec_without_threshold, column (n,) float64, argsort cache,
  cumulative_gain, last_win_round)`.
- On a node win by a temporal candidate: if an entry with an identical spec already
  exists, just add gain credit; else if `|pearson corr|` with any existing bank column
  > 0.995 → skip; else materialize on all rows and append.
- Every node's candidate list includes ALL bank columns (they cost one gather + scan).
  A bank column that wins a node adds its gain to `cumulative_gain`.
- Cap `bank_max = 128` entries. The bank is a TRAINING-TIME CACHE ONLY: every tree
  split stores its own self-contained spec copy, so evicting a bank entry (lowest
  cumulative_gain, oldest last_win_round as tiebreak) can never affect predict. This is
  the §6 "bank integrity" test.
- Predict is unchanged (each node evaluates its own spec). Optional, correctness-second
  optimization: memoize identical specs per batch during predict.
- Interpretability bonus: `feature_importances()` gains a "banked feature" view —
  the shared dictionary with cumulative gains and birth rounds.

## 10.3 Comparison-split family (event time vs per-row threshold)

**Idea (from RocketFuse):** tasks like "did the event happen before this row's
deadline" need a staircase of axis-aligned splits; a single split on
`rank(event position) − rank(static column)` solves them in one shot.

**Mechanics:**
- Only active once the bank contains ≥1 position-type entry (`filter_pos` /
  `shapelet_pos`) and `p ≥ 1`.
- Per node, `n_comparison_candidates = 8` random pairs (banked position entry, static
  column j). Feature: `f_i = ECDF_pos(v_i) − ECDF_stat(x_ij)` where each ECDF is
  `np.searchsorted(sorted_train_values, ·, side='right') / n_train`, computed ONCE on
  the full training column at promotion time (position entry) / at fit start (static
  cols), stored as the sorted value arrays, and reused FROZEN at predict. NaN in either
  input → NaN feature (missing path).
- Scored by `scan_threshold` like everything else. Winning spec stores the position
  spec copy, static col index, and both sorted-value tables.
- `describe()`: `rank(pos(filter ch=0, scale=1)) − rank(static[2]) <= 0.13`.

## Phase-B parameter additions

```
n_filter_candidates=8, n_fitted_filters=4, filter_len=9, dct_K=5, lambda_beta=1.0,
n_filter_alt=1, bank_enabled=True, bank_max=128, n_comparison_candidates=8,
n_shapelet_candidates=0   # changed default; family still implemented
```

---

# PHASE C (v0.3, opt-in flags) — only after Phase B ships and benchmarks pass

## 11.1 `dense_base=True` — exact-LOO ridge base margin (dense/diffuse-signal engine)

Trees cannot aggregate thousands of individually weak temporal features into one vote;
a regularized linear layer can. Opt-in because it is dead weight on pure-interaction
(XOR-type) tasks and adds a stacking-hygiene failure mode.

- **Deterministic bank (label-free):** dyadic interval pyramid — windows of length
  T, T/2, T/4, T/8 with 50% overlap (1+3+7+15 = 26 windows) × the §5.2 nine stats ×
  each channel and its first difference; float32; NaN-aware (all masks, §5.2 rules).
  This bank deterministically CONTAINS the aggregate baseline.
- **Ridge:** standardize by train mean/std; median-impute NaNs (ridge copy only).
  Economy SVD; λ from `logspace(−3, 3, 13)` by closed-form LOOCV using hat diagonals
  `h_ii(λ) = Σ_j u_ij² s_j²/(s_j²+λ)`. Targets: centered y (regression) or ±1
  one-vs-rest per class (classification).
- **The non-negotiable part:** train-time base margins are the exact LOO predictions
  `m_i = (ŷ_i − h_ii·y_i)/(1 − h_ii)`; predict-time margins use the full-data fit.
  Classification margins go through a 2-parameter Platt map fitted on the LOO margins.
  §6 Phase-C tests enforce both.
- **Boosting integration:** `raw` is initialized to the (calibrated) margin instead of
  the constant `init_score`; the margin is also appended as one extra static-like
  candidate column so trees can gate/flip it (static×temporal XOR handling).
- n > 10,000: replace exact LOO with 5-fold cross-fitted margins (documented switch).
- Skip the MiniROCKET convolution bank in this phase; add later ONLY if benchmarks show
  dense-signal scenarios losing (decision recorded in ARCHITECTURES.md §4).

## 11.2 `levy_areas=True` — cross-channel lead-lag columns (only meaningful when C ≥ 2)

For sampled channel pairs (a,b) and each dyadic window w (same 26-window pyramid):
center both channels at the window start, then
`A_ab(w) = ½ Σ_t (x_a,t·Δx_b,t − x_b,t·Δx_a,t)` over observed points (drop NaN
timesteps first — observed-point path; NaN-safe by construction). Computable with
cumsums, ~50 LOC, no signature machinery. Offered as extra static-like columns.
Positive A_ab ⇒ "a leads b" within the window. §6 Phase-C analytic test (quarter
circle → area πr²/8-style closed form) is mandatory.

---

# 12. Deferred / rejected (with revisit triggers)

- **GRAFT saliency, mutation foundry, bandits:** rejected — marginal (root-level)
  probe is blind on interaction-gated signals; biggest bookkeeping risk.
- **RegimeBoost Kalman/HMM canonicalization + LLR probes:** deferred — best-in-class
  for heavy informative missingness; revisit if a target domain has it.
- **ChenBoost signature engine:** deferred — revisit for multichannel lead-lag-heavy
  domains; the Lévy-area transplant (§11.2) covers the irreplaceable part.
- **MiniROCKET conv bank inside dense_base:** deferred pending benchmark evidence.
- **TabPFN-style prior-fitted models:** disqualified by CPU/no-pretraining constraints.
