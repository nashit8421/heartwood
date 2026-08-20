"""The workarounds Heartwood is measured against.

Every baseline is the same idea: turn the series into a fixed table of numbers,
paste it next to the static columns, and hand the result to a gradient-boosted
tree.  They differ only in how much temporal detail survives that step —
which is exactly the variable under study.

**Missing data is skipped, not propagated** — see ``_block_stats``.  The first
version of this module used bare NumPy reductions, which turn a whole summary
into NaN if a single cell in the window is missing.  On dense data that is
invisible; on PhysioNet ICU (80% missing) it left the ``agg`` design matrix 94%
NaN with 176 of 370 columns entirely empty, and cost the baseline about ten
points of AUC.  Since Heartwood's own ``interval_stat`` has always skipped NaN,
the comparison was varying two things at once — the representation under study
and the missing-data convention.  ``validation/CORRECTION.md`` records what that
did to the published result.
"""

from __future__ import annotations

import warnings

import numpy as np

STAT_NAMES = (
    "mean", "std", "min", "max", "slope", "median", "mean_abs_change",
    "first", "last", "delta",
)


def _block_stats_naive(block: np.ndarray) -> list[np.ndarray]:
    """The ten summaries via bare NumPy reductions — NaN in, NaN out.

    Kept only so the pre-registered v0.3 validation numbers stay reproducible
    (representations ``agg_naive`` / ``wagg8_naive``).  Do not use for new work:
    a single missing cell destroys the whole statistic.
    """
    t = np.arange(block.shape[1], dtype=np.float64)
    tc = t - t.mean()
    denom = (tc * tc).sum()
    slope = (block * tc).sum(1) / denom if denom > 0 else np.zeros(len(block))
    changes = (
        np.abs(np.diff(block, axis=1)).mean(1)
        if block.shape[1] > 1
        else np.zeros(len(block))
    )
    return [
        block.mean(1), block.std(1), block.min(1), block.max(1), slope,
        np.median(block, axis=1), changes,
        block[:, 0], block[:, -1], block[:, -1] - block[:, 0],
    ]


def _block_stats(block: np.ndarray) -> list[np.ndarray]:
    """The ten classic summaries of a ``(n, L)`` block, skipping missing cells.

    This is what ``pandas`` gives a practitioner by default and what Heartwood's
    own ``interval_stat`` computes, so the two sides of the benchmark differ only
    in *which windows and statistics get searched* — the actual question.

    A row with no finite value in the window yields NaN, which is honest: there
    is nothing to summarise.  ``first``/``last``/``delta`` use the first and last
    *observed* value rather than the first and last cell.
    """
    m, L = block.shape
    mask = np.isfinite(block)
    count = mask.sum(1)
    has = count > 0
    z = np.where(mask, block, 0.0)
    blank = np.full(m, np.nan)

    def defined(values: np.ndarray) -> np.ndarray:
        """Keep ``values`` only for rows that had something to average."""
        out = blank.copy()
        out[has] = values[has]
        return out

    mean = np.zeros(m)
    mean[has] = z.sum(1)[has] / count[has]
    mean_square = np.zeros(m)
    mean_square[has] = (z * z).sum(1)[has] / count[has]
    std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    minimum = np.where(mask, block, np.inf).min(1)
    maximum = np.where(mask, block, -np.inf).max(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        median = np.nanmedian(np.where(mask, block, np.nan), axis=1)

    # least squares over the observed points only
    t = np.arange(L, dtype=np.float64)
    n = count.astype(np.float64)
    sum_t, sum_tt = (mask * t).sum(1), (mask * t * t).sum(1)
    sum_x, sum_tx = z.sum(1), (z * t).sum(1)
    denom = n * sum_tt - sum_t * sum_t
    usable = (count >= 2) & (np.abs(denom) > 1e-12)
    slope = blank.copy()
    slope[usable] = (
        n[usable] * sum_tx[usable] - sum_t[usable] * sum_x[usable]
    ) / denom[usable]

    changes = blank.copy()
    if L >= 2:
        step = np.abs(np.diff(block, axis=1))
        step_mask = np.isfinite(step)
        step_count = step_mask.sum(1)
        stepped = step_count > 0
        changes[stepped] = (
            np.where(step_mask, step, 0.0).sum(1)[stepped] / step_count[stepped]
        )

    first, last, delta = blank.copy(), blank.copy(), blank.copy()
    rows = np.nonzero(has)[0]
    if rows.size:
        first_index = np.argmax(mask, axis=1)
        last_index = L - 1 - np.argmax(mask[:, ::-1], axis=1)
        first[rows] = block[rows, first_index[rows]]
        last[rows] = block[rows, last_index[rows]]
        delta[rows] = last[rows] - first[rows]

    return [
        defined(mean), defined(std), defined(minimum), defined(maximum), slope,
        defined(median), changes, first, last, delta,
    ]


def aggregate_features(X_series: np.ndarray, naive: bool = False) -> np.ndarray:
    """The standard move: collapse each channel to ten global summaries.

    This is what most teams ship, and it is the representation Heartwood is
    meant to replace.

    Written as its own loop rather than delegating to the windowed version, so
    that ``test_one_window_aggregation_equals_global_aggregation`` still checks
    something instead of being true by construction.
    """
    stats = _block_stats_naive if naive else _block_stats
    columns: list[np.ndarray] = []
    for channel in range(X_series.shape[1]):
        columns.extend(stats(X_series[:, channel, :]))
    return np.column_stack(columns)


def windowed_aggregate_features(
    X_series: np.ndarray, n_windows: int = 4, naive: bool = False
) -> np.ndarray:
    """A deliberately stronger baseline: the same summaries per equal window.

    Included because it is the obvious next thing a good practitioner tries, and
    because on some tasks it is genuinely hard to beat.  Reporting only the
    global version would flatter us.

    ``naive=True`` selects the NaN-propagating reductions the first version of
    this module used; it exists to reproduce the v0.3 validation run and should
    not be used for new comparisons.
    """
    stats = _block_stats_naive if naive else _block_stats
    columns: list[np.ndarray] = []
    for channel in range(X_series.shape[1]):
        series = X_series[:, channel, :]
        for index in np.array_split(np.arange(series.shape[1]), n_windows):
            if len(index):
                columns.extend(stats(series[:, index]))
    return np.column_stack(columns)


def flatten_series(X_series: np.ndarray) -> np.ndarray:
    """Every timestep as its own column — no aggregation, no alignment either.

    Keeps all the information and throws away all the structure: column t means
    "the value at time t", so a pattern that moves looks like a different
    feature entirely.
    """
    return X_series.reshape(len(X_series), -1)


#: name -> (transform, kwargs); ``None`` means static columns only.
REPRESENTATIONS: dict[str, object] = {
    "static_only": None,
    "agg": (aggregate_features, {}),
    "wagg4": (windowed_aggregate_features, {"n_windows": 4}),
    "wagg8": (windowed_aggregate_features, {"n_windows": 8}),
    "wagg16": (windowed_aggregate_features, {"n_windows": 16}),
    "raw_flat": (flatten_series, {}),
    # the NaN-propagating originals, kept so the v0.3 numbers reproduce
    "agg_naive": (aggregate_features, {"naive": True}),
    "wagg8_naive": (windowed_aggregate_features, {"n_windows": 8, "naive": True}),
}


def build_design_matrix(name: str, X_static: np.ndarray, X_series) -> np.ndarray:
    """Assemble ``[static | transformed series]`` for one baseline."""
    spec = REPRESENTATIONS[name]
    if spec is None or X_series is None:
        return X_static
    transform, kwargs = spec
    return np.hstack([X_static, transform(X_series, **kwargs)])


def make_baseline_model(task: str, n_estimators: int, max_depth: int,
                        learning_rate: float, seed: int):
    """A gradient-boosted tree with the same budget Heartwood gets.

    Prefers XGBoost; falls back to scikit-learn so the suite still runs without
    it.  Threads are pinned to one so parallel benchmark workers do not fight
    over cores.
    """
    try:
        import xgboost as xgb

        cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        return cls(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed,
            n_jobs=1, verbosity=0, tree_method="exact",
        ), "xgboost"
    except ImportError:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        cls = (
            HistGradientBoostingRegressor
            if task == "regression"
            else HistGradientBoostingClassifier
        )
        return cls(
            max_iter=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed,
        ), "sklearn-histgb"
