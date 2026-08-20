"""The workarounds Heartwood is measured against.

Every baseline is the same idea: turn the series into a fixed table of numbers,
paste it next to the static columns, and hand the result to a gradient-boosted
tree.  They differ only in how much temporal detail survives that step —
which is exactly the variable under study.
"""

from __future__ import annotations

import numpy as np

STAT_NAMES = (
    "mean", "std", "min", "max", "slope", "median", "mean_abs_change",
    "first", "last", "delta",
)


def _block_stats(block: np.ndarray) -> list[np.ndarray]:
    """The ten classic summaries of a ``(n, L)`` block."""
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


def aggregate_features(X_series: np.ndarray) -> np.ndarray:
    """The standard move: collapse each channel to ten global summaries.

    This is what most teams ship, and it is the representation Heartwood is
    meant to replace.
    """
    columns: list[np.ndarray] = []
    for channel in range(X_series.shape[1]):
        columns.extend(_block_stats(X_series[:, channel, :]))
    return np.column_stack(columns)


def windowed_aggregate_features(X_series: np.ndarray, n_windows: int = 4) -> np.ndarray:
    """A deliberately stronger baseline: the same summaries per equal window.

    Included because it is the obvious next thing a good practitioner tries, and
    because on some tasks it is genuinely hard to beat.  Reporting only the
    global version would flatter us.
    """
    columns: list[np.ndarray] = []
    for channel in range(X_series.shape[1]):
        series = X_series[:, channel, :]
        for index in np.array_split(np.arange(series.shape[1]), n_windows):
            if len(index):
                columns.extend(_block_stats(series[:, index]))
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
