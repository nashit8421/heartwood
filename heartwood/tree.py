"""The temporal regression tree — where static and temporal splits compete.

A node draws a fresh pool of candidate splits: every (subsampled) static column,
a batch of randomly placed interval statistics, and a batch of shapelets cut
from the series of the rows that reached this node.  All of them are scored on
the *same* second-order gain, so the tree decides per node whether the next best
question is about a customer attribute or about the shape of their trajectory.

Because the temporal candidates are redrawn at every node of every round, the
window that matters is discovered at whatever position and resolution the
gradients call for, instead of being fixed by an up-front aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import STAT_NAMES, eval_split_feature, interval_stat, shapelet_features
from .splits import SplitSpec, sample_interval, sample_shapelet, scan_threshold


@dataclass
class TreeParams:
    """Everything a single tree needs to know."""

    max_depth: int = 4
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1e-3
    min_samples_leaf: int = 5
    colsample: float = 1.0
    n_interval_candidates: int = 16
    n_shapelet_candidates: int = 4
    interval_stats: tuple[str, ...] = STAT_NAMES
    full_interval_prob: float = 0.25
    min_interval_len: int = 3
    shapelet_min_len: int = 3
    shapelet_max_frac: float = 0.5
    shapelet_znorm: bool = True


class TemporalTree:
    """A depth-limited regression tree over static + temporal split candidates."""

    def __init__(self, params: TreeParams):
        self.params = params
        self.nodes: list[dict] = []

    # ------------------------------------------------------------------ fit

    def fit(self, X_static, X_series, g, h, rows, rng) -> "TemporalTree":
        """Grow the tree on ``rows`` against gradients ``g`` and hessians ``h``.

        ``g``/``h`` are 1-D and indexed by the *global* row id, so subsampling is
        just a smaller ``rows`` array.
        """
        self.nodes = []
        rows = np.asarray(rows, dtype=np.intp)
        if rows.size == 0:
            raise ValueError("cannot fit a tree on an empty row set")
        self._build(X_static, X_series, g, h, rows, 0, rng)
        return self

    def _add_leaf(self, value: float) -> int:
        self.nodes.append({"leaf": True, "value": float(value)})
        return len(self.nodes) - 1

    def _build(self, X_static, X_series, g, h, rows, depth, rng) -> int:
        p = self.params
        gr = g[rows]
        hr = h[rows]
        G = float(gr.sum())
        H = float(hr.sum())
        value = -G / (H + p.reg_lambda)

        if (
            depth >= p.max_depth
            or rows.size < 2 * p.min_samples_leaf
            or H < 2 * p.min_child_weight
        ):
            return self._add_leaf(value)

        best = None
        for f, spec in self._candidates(X_static, X_series, rows, rng):
            found = scan_threshold(
                f, gr, hr, p.reg_lambda, p.gamma, p.min_child_weight, p.min_samples_leaf
            )
            if found is None:
                continue
            gain, threshold, missing_left = found
            if best is None or gain > best[0]:
                best = (gain, f, spec, threshold, missing_left)

        if best is None:
            return self._add_leaf(value)

        gain, f, spec, threshold, missing_left = best
        spec.gain = gain
        spec.threshold = threshold
        spec.missing_left = missing_left

        go_left = np.where(np.isfinite(f), f <= threshold, missing_left)
        left_rows = rows[go_left]
        right_rows = rows[~go_left]
        if left_rows.size == 0 or right_rows.size == 0:
            return self._add_leaf(value)

        idx = len(self.nodes)
        self.nodes.append({"leaf": False, "spec": spec, "left": -1, "right": -1})
        self.nodes[idx]["left"] = self._build(
            X_static, X_series, g, h, left_rows, depth + 1, rng
        )
        self.nodes[idx]["right"] = self._build(
            X_static, X_series, g, h, right_rows, depth + 1, rng
        )
        return idx

    def _candidates(self, X_static, X_series, rows, rng):
        """Yield ``(feature_values_over_rows, partial_spec)`` pairs."""
        p = self.params

        n_cols = X_static.shape[1]
        if n_cols:
            k = max(1, int(np.ceil(p.colsample * n_cols)))
            cols = (
                np.arange(n_cols)
                if k >= n_cols
                else rng.choice(n_cols, size=k, replace=False)
            )
            for col in cols:
                yield X_static[rows, col], SplitSpec(kind="static", col=int(col))

        if X_series is None:
            return

        _, n_channels, T = X_series.shape
        stats = p.interval_stats

        for _ in range(p.n_interval_candidates):
            channel = int(rng.integers(n_channels))
            start, end = sample_interval(T, rng, p.min_interval_len, p.full_interval_prob)
            stat = str(stats[int(rng.integers(len(stats)))])
            values = interval_stat(X_series[rows, channel, start:end], stat)
            yield values, SplitSpec(
                kind="interval", channel=channel, start=start, end=end, stat=stat
            )

        for _ in range(p.n_shapelet_candidates):
            drawn = sample_shapelet(
                X_series, rows, rng, p.shapelet_min_len, p.shapelet_max_frac
            )
            if drawn is None:
                continue
            channel, shapelet = drawn
            # One distance pass answers both "does this shape occur?" and
            # "where?" — compute once, offer as two independent candidates.
            dist, pos = shapelet_features(
                X_series[rows, channel, :], shapelet, znorm=p.shapelet_znorm
            )
            yield dist, SplitSpec(
                kind="shapelet_dist",
                channel=channel,
                shapelet=shapelet,
                znorm=p.shapelet_znorm,
            )
            yield pos, SplitSpec(
                kind="shapelet_pos",
                channel=channel,
                shapelet=shapelet,
                znorm=p.shapelet_znorm,
            )

    # -------------------------------------------------------------- predict

    def predict(self, X_static, X_series) -> np.ndarray:
        """Raw leaf values (the learning rate is applied by the booster)."""
        n = X_static.shape[0]
        out = np.zeros(n, dtype=np.float64)
        if not self.nodes:
            return out
        self._predict_into(0, np.arange(n, dtype=np.intp), X_static, X_series, out)
        return out

    def _predict_into(self, idx, rows, X_static, X_series, out) -> None:
        if rows.size == 0:
            return
        node = self.nodes[idx]
        if node["leaf"]:
            out[rows] = node["value"]
            return
        spec = node["spec"]
        f = eval_split_feature(spec, X_static, X_series, rows)
        go_left = np.where(np.isfinite(f), f <= spec.threshold, spec.missing_left)
        self._predict_into(node["left"], rows[go_left], X_static, X_series, out)
        self._predict_into(node["right"], rows[~go_left], X_static, X_series, out)

    def apply(self, X_static, X_series) -> np.ndarray:
        """Leaf index reached by each row — used to verify fit/predict routing."""
        n = X_static.shape[0]
        out = np.full(n, -1, dtype=np.intp)
        if not self.nodes:
            return out
        self._apply_into(0, np.arange(n, dtype=np.intp), X_static, X_series, out)
        return out

    def _apply_into(self, idx, rows, X_static, X_series, out) -> None:
        if rows.size == 0:
            return
        node = self.nodes[idx]
        if node["leaf"]:
            out[rows] = idx
            return
        spec = node["spec"]
        f = eval_split_feature(spec, X_static, X_series, rows)
        go_left = np.where(np.isfinite(f), f <= spec.threshold, spec.missing_left)
        self._apply_into(node["left"], rows[go_left], X_static, X_series, out)
        self._apply_into(node["right"], rows[~go_left], X_static, X_series, out)

    # ----------------------------------------------------------- inspection

    def iter_splits(self):
        for node in self.nodes:
            if not node["leaf"]:
                yield node["spec"]

    @property
    def n_leaves(self) -> int:
        return sum(1 for node in self.nodes if node["leaf"])
