"""The temporal regression tree — where static and temporal splits compete.

A node draws a fresh pool of candidate splits: every (subsampled) static column,
a batch of randomly placed interval statistics, templates matched against the
series, anything the bank has already found useful, and comparisons between a
learned event time and a static column.  All of them are scored on the *same*
second-order gain, so the tree decides per node whether the next best question
is about a customer attribute or about the shape of their trajectory.

Because the temporal candidates are redrawn at every node of every round, the
window that matters is discovered at whatever position and resolution the
gradients call for, instead of being fixed by an up-front aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .features import STAT_NAMES, ecdf, eval_split_feature, interval_stat, shapelet_features
from .filters import (
    Pyramid,
    align,
    dct_basis,
    gather_windows,
    refit_template,
    znorm_snippet,
)
from .splits import SplitSpec, sample_interval, sample_shapelet, scan_threshold

_EPS = 1e-12
TEMPORAL_KINDS = (
    "interval", "shapelet_dist", "shapelet_pos", "filter_resp", "filter_pos", "comparison",
)


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
    # Phase B
    n_filter_candidates: int = 0
    n_fitted_filters: int = 4
    filter_len: int = 9
    dct_components: int = 5
    ridge_beta: float = 1.0
    n_filter_alt: int = 1
    n_comparison_candidates: int = 4
    bank_colsample: float = 0.25


@dataclass
class FitContext:
    """Per-fit structures shared across trees: pooled series, bank, rank grids."""

    pyramid: Pyramid | None = None
    bank: object | None = None
    static_grids: list[np.ndarray] | None = None
    round_index: int = 0


class TemporalTree:
    """A depth-limited regression tree over static + temporal split candidates."""

    def __init__(self, params: TreeParams):
        self.params = params
        self.nodes: list[dict] = []

    # ------------------------------------------------------------------ fit

    def fit(self, X_static, X_series, g, h, rows, rng, context: FitContext | None = None):
        """Grow the tree on ``rows`` against gradients ``g`` and hessians ``h``.

        ``g``/``h`` are 1-D and indexed by the *global* row id, so subsampling is
        just a smaller ``rows`` array.
        """
        self.nodes = []
        rows = np.asarray(rows, dtype=np.intp)
        if rows.size == 0:
            raise ValueError("cannot fit a tree on an empty row set")
        self._context = context or FitContext()
        self._X_static = X_static
        self._X_series = X_series
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
        for f, spec in self._candidates(X_static, X_series, rows, gr, hr, rng):
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

        self._bank_winner(spec, gain)

        idx = len(self.nodes)
        self.nodes.append({"leaf": False, "spec": spec, "left": -1, "right": -1})
        self.nodes[idx]["left"] = self._build(
            X_static, X_series, g, h, left_rows, depth + 1, rng
        )
        self.nodes[idx]["right"] = self._build(
            X_static, X_series, g, h, right_rows, depth + 1, rng
        )
        return idx

    def _bank_winner(self, spec: SplitSpec, gain: float) -> None:
        """Offer a winning temporal split to the bank, materialised on all rows."""
        bank = self._context.bank
        if bank is None or spec.kind not in TEMPORAL_KINDS:
            return
        all_rows = np.arange(self._X_static.shape[0], dtype=np.intp)
        column = eval_split_feature(
            spec, self._X_static, self._X_series, all_rows, self._context.pyramid
        )
        bank.promote(spec, column, gain, self._context.round_index)

    # ---------------------------------------------------------- candidates

    def _candidates(self, X_static, X_series, rows, gr, hr, rng):
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

        bank = self._context.bank
        if bank is not None:
            yield from bank.candidates(rows, rng, p.bank_colsample)
            yield from self._comparison_candidates(X_static, rows, rng)

        if X_series is None:
            return

        yield from self._interval_candidates(X_series, rows, rng)
        yield from self._shapelet_candidates(X_series, rows, rng)
        yield from self._filter_candidates(X_series, rows, gr, hr, rng)

    def _interval_candidates(self, X_series, rows, rng):
        p = self.params
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

    def _shapelet_candidates(self, X_series, rows, rng):
        p = self.params
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
            for kind, values in (("shapelet_dist", dist), ("shapelet_pos", pos)):
                yield values, SplitSpec(
                    kind=kind, channel=channel, shapelet=shapelet, znorm=p.shapelet_znorm
                )

    def _filter_candidates(self, X_series, rows, gr, hr, rng):
        """Templates matched against the series, half of them fitted to the residuals."""
        p = self.params
        pyramid = self._context.pyramid
        if pyramid is None or p.n_filter_candidates <= 0:
            return

        # Newton residuals: what this node still gets wrong, in the units the
        # leaf value is expressed in.
        residual = -gr / (hr + p.reg_lambda)
        weight_total = float(hr.sum())
        centred = residual - (
            float(hr @ residual) / weight_total if weight_total > _EPS else residual.mean()
        )
        donor_weights = hr * centred**2
        donor_total = float(donor_weights.sum())
        probabilities = (
            donor_weights / donor_total
            if np.isfinite(donor_total) and donor_total > _EPS
            else None
        )

        basis = dct_basis(p.filter_len, p.dct_components)
        n_channels = X_series.shape[1]

        for index in range(p.n_filter_candidates):
            is_fitted = index < p.n_fitted_filters
            channel = int(rng.integers(n_channels))
            scale = int(rng.integers(pyramid.n_scales))
            block = pyramid.block(scale, channel, rows)
            if block.shape[1] < p.filter_len + 1:
                continue

            template = self._seed_template(block, probabilities, rng)
            if template is None:
                continue

            if is_fitted:
                projected = basis @ (basis.T @ template)
                norm = float(np.linalg.norm(projected))
                if norm <= _EPS:
                    continue
                template = projected / norm

            response, position = align(block, template)

            if is_fitted:
                for _ in range(p.n_filter_alt):
                    matched, windows = gather_windows(block, position, p.filter_len)
                    refitted = refit_template(
                        windows, centred[matched], hr[matched], basis, p.ridge_beta
                    )
                    if refitted is None:
                        break
                    template = refitted
                    response, position = align(block, template)

            stored = template.copy()
            for kind, values in (("filter_resp", response), ("filter_pos", position)):
                yield values, SplitSpec(
                    kind=kind, channel=channel, scale=scale, template=stored
                )

    def _seed_template(self, block, probabilities, rng):
        """Cut a starting template from a row this node still gets wrong."""
        p = self.params
        n_rows, length = block.shape
        for _ in range(10):
            donor = (
                int(rng.choice(n_rows, p=probabilities))
                if probabilities is not None
                else int(rng.integers(n_rows))
            )
            start = int(rng.integers(0, length - p.filter_len + 1))
            snippet = znorm_snippet(block[donor, start : start + p.filter_len])
            if snippet is not None:
                return snippet
        return None

    def _comparison_candidates(self, X_static, rows, rng):
        """"Did this happen before that" — a learned event time versus a static column.

        Axis-aligned splits need a staircase of thresholds to approximate this;
        ranking both sides against their training distributions turns it into a
        single split.
        """
        p = self.params
        bank = self._context.bank
        grids = self._context.static_grids
        if p.n_comparison_candidates <= 0 or not grids:
            return
        positions = bank.position_entries()
        if not positions:
            return

        for _ in range(p.n_comparison_candidates):
            entry = positions[int(rng.integers(len(positions)))]
            if entry.grid is None:
                continue
            col = int(rng.integers(len(grids)))
            if grids[col].size == 0:
                continue
            values = ecdf(entry.column[rows], entry.grid) - ecdf(
                X_static[rows, col], grids[col]
            )
            yield values, SplitSpec(
                kind="comparison",
                col=col,
                position_spec=replace(entry.spec),
                position_grid=entry.grid,
                static_grid=grids[col],
            )

    # -------------------------------------------------------------- predict

    def predict(self, X_static, X_series, pyramid=None) -> np.ndarray:
        """Raw leaf values (the learning rate is applied by the booster)."""
        n = X_static.shape[0]
        out = np.zeros(n, dtype=np.float64)
        if not self.nodes:
            return out
        self._walk(0, np.arange(n, dtype=np.intp), X_static, X_series, pyramid, out, False)
        return out

    def apply(self, X_static, X_series, pyramid=None) -> np.ndarray:
        """Leaf index reached by each row — used to verify fit/predict routing."""
        n = X_static.shape[0]
        out = np.full(n, -1, dtype=np.intp)
        if not self.nodes:
            return out
        self._walk(0, np.arange(n, dtype=np.intp), X_static, X_series, pyramid, out, True)
        return out

    def _walk(self, idx, rows, X_static, X_series, pyramid, out, want_index) -> None:
        if rows.size == 0:
            return
        node = self.nodes[idx]
        if node["leaf"]:
            out[rows] = idx if want_index else node["value"]
            return
        spec = node["spec"]
        f = eval_split_feature(spec, X_static, X_series, rows, pyramid)
        go_left = np.where(np.isfinite(f), f <= spec.threshold, spec.missing_left)
        self._walk(node["left"], rows[go_left], X_static, X_series, pyramid, out, want_index)
        self._walk(node["right"], rows[~go_left], X_static, X_series, pyramid, out, want_index)

    # ----------------------------------------------------------- inspection

    def iter_splits(self):
        for node in self.nodes:
            if not node["leaf"]:
                yield node["spec"]

    @property
    def n_leaves(self) -> int:
        return sum(1 for node in self.nodes if node["leaf"])
