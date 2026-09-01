"""The temporal regression tree — where static and temporal splits compete.

A node draws a fresh pool of candidate splits: every (subsampled) static column,
a batch of randomly placed interval statistics, templates matched against the
series, and anything the bank has already found useful.  All of them are scored on the *same*
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

#: How much of the null maximum gain the within-candidate threshold search adds,
#: as a coefficient on ``log(n_rows)``.  ``scan_threshold`` maximises over every
#: cut point as well as over candidates, so a node's pool is effectively larger
#: than its candidate count; the coefficient is well under 1 because adjacent
#: cut points are highly correlated.  Measured against the real
#: ``scan_threshold`` in ``tests/test_gain_penalty.py``, which fails if either
#: this constant or that function drifts away from the other.
MC_THRESHOLD_LOG = 0.4

TEMPORAL_KINDS = (
    "interval", "shapelet_dist", "shapelet_pos", "filter_resp", "filter_pos",
    "product",
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
    #: Magnitude products of a banked temporal feature with a static column
    #: (V22, roadmap item 5).  0 disables.
    n_product_candidates: int = 0
    bank_colsample: float = 0.25
    #: Per-node bagging over the temporal draws (V16, roadmap item 2a).  1.0 is
    #: the shipped behaviour.  See :meth:`TemporalTree._draw_count`.
    candidate_colsample: float = 1.0
    #: Permutations used to price a node's own selection bias (V8). 0 disables.
    selection_null: int = 0
    #: Which quantile of the per-permutation null maxima a split must clear
    #: (V19, item 2d).  1.0 is V8's behaviour and is kept as the default so that
    #: arm reproduces exactly.  See :meth:`TemporalTree._chance_gain`.
    selection_null_quantile: float = 1.0
    #: Analytic multiple-comparisons charge on the winning gain (V17, item 2b).
    #: 0 disables.  See :meth:`TemporalTree._selection_charge`.
    mc_penalty: float = 0.0


@dataclass
class FitContext:
    """Per-fit structures shared across trees: pooled series, bank, rank grids."""

    pyramid: Pyramid | None = None
    bank: object | None = None
    static_grids: list[np.ndarray] | None = None
    static_names: list[str] | None = None
    round_index: int = 0
    #: ``(center, scale, low, high)`` per static column, from the training rows.
    #: Product splits need magnitudes on a common scale and a training range to
    #: clip to; ``static_grids`` carries ranks, which is the currency a product
    #: split must not use.
    static_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


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
        scored: list[np.ndarray] = []
        n_scanned = 0
        for f, spec in self._candidates(X_static, X_series, rows, gr, hr, rng):
            found = scan_threshold(
                f, gr, hr, p.reg_lambda, p.gamma, p.min_child_weight, p.min_samples_leaf
            )
            if found is None:
                continue
            n_scanned += 1
            gain, threshold, missing_left = found
            if p.selection_null:
                scored.append(f)
            if best is None or gain > best[0]:
                best = (gain, f, spec, threshold, missing_left)

        if best is None:
            return self._add_leaf(value)

        if p.mc_penalty and best[0] <= self._selection_charge(n_scanned, gr, hr, H):
            # The winner is no larger than what a pool this size reaches on
            # noise, so it is priced as noise. Stop here.
            return self._add_leaf(value)

        if p.selection_null and best[0] <= self._chance_gain(scored, gr, hr, rng):
            # The winner is no better than the best this same pool reaches on
            # shuffled gradients, so it is the winner's curse rather than a
            # question worth asking. Stop here.
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

    def _draw_count(self, requested: int) -> int:
        """How many temporal candidates this node actually draws.

        ``candidate_colsample`` is per-node feature bagging aimed at the winner's
        curse, which ``validation/HEADROOM.md`` measured as this model's ceiling:
        a node takes the maximum gain over its pool, so a larger pool raises the
        winner's expected gain *whether or not anything in it is informative*,
        and every subtree below inherits the over-fitted split.  HEADROOM pushed
        this knob upwards -- x4 and x16 -- and found the default already past the
        optimum on two datasets of three.  Downwards is the untested direction
        and the one the argument actually predicts.

        Scope is deliberate.  This thins the *temporal draws* only, because the
        static block and the bank already have ``colsample`` and
        ``bank_colsample``; a third knob silently multiplying those two would
        make any measured effect impossible to attribute.

        The floor of one keeps a node's pool non-empty at any fraction, so a
        small ``candidate_colsample`` degrades the search rather than switching
        a whole candidate source off.
        """
        if requested <= 0:
            return 0
        if not 0.0 < self.params.candidate_colsample < 1.0:
            return requested
        return max(1, int(np.ceil(self.params.candidate_colsample * requested)))

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
            names = self._context.static_names
            for col in cols:
                yield X_static[rows, col], SplitSpec(
                    kind="static", col=int(col),
                    name_hint=names[col] if names else "",
                )

        bank = self._context.bank
        if bank is not None:
            yield from bank.candidates(rows, rng, p.bank_colsample)
            yield from self._product_candidates(X_static, rows, rng)

        if X_series is None:
            return

        yield from self._interval_candidates(X_series, rows, rng)
        yield from self._shapelet_candidates(X_series, rows, rng)
        yield from self._filter_candidates(X_series, rows, gr, hr, rng)

    def _interval_candidates(self, X_series, rows, rng):
        p = self.params
        _, n_channels, T = X_series.shape
        stats = p.interval_stats
        for _ in range(self._draw_count(p.n_interval_candidates)):
            channel = int(rng.integers(n_channels))
            start, end = sample_interval(T, rng, p.min_interval_len, p.full_interval_prob)
            stat = str(stats[int(rng.integers(len(stats)))])
            values = interval_stat(X_series[rows, channel, start:end], stat)
            yield values, SplitSpec(
                kind="interval", channel=channel, start=start, end=end, stat=stat
            )

    def _shapelet_candidates(self, X_series, rows, rng):
        p = self.params
        for _ in range(self._draw_count(p.n_shapelet_candidates)):
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

        # Both counts are thinned, so bagging changes how many templates a node
        # tries without changing the fitted/seeded mix it tries them in.
        n_filters = self._draw_count(p.n_filter_candidates)
        n_fitted = min(n_filters, self._draw_count(p.n_fitted_filters))
        for index in range(n_filters):
            is_fitted = index < n_fitted
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

    def _selection_charge(self, n_scanned: int, gr, hr, H: float) -> float:
        """What a pool of ``n_scanned`` candidates reaches on noise alone.

        Roadmap item 2b: a multiple-comparisons correction applied to the
        maximum, rather than the permutation null of ``selection_null``.  It
        costs one dot product per node instead of ``selection_null`` full
        rescans of the pool, which is the entire reason to prefer it.

        The derivation.  With the parent term subtracted and the node's
        gradients centred against its own hessians, the gain of a split against
        an *uninformative* feature is asymptotically ``scale * chi2_1 / 2``,
        where ``scale`` is this node's hessian-weighted gradient variance.  The
        maximum of ``m`` independent chi-squares grows like ``2 log m``, so the
        expected best gain from noise grows like ``scale * log m``.  Every
        candidate at a node shares the same ``m``, so this charge cannot change
        *which* candidate wins -- only whether the node splits at all.

        Measured, not assumed: against the real ``scan_threshold`` the
        ``log m`` coefficient comes out at 1.00-1.10 across centred, shifted and
        heteroscedastic gradients, and ``tests/test_gain_penalty.py`` fails if
        that stops being true.  The additive constant from the same fit ranged
        from -2.5 to -0.1 depending on the regime, so it is deliberately **not**
        baked in here: an unstable constant dressed as theory is worse than a
        multiplier the study has to sweep, which is what ``mc_penalty`` is.
        """
        if n_scanned < 1:
            return 0.0
        G = float(gr.sum())
        centred = gr - hr * (G / H) if H > _EPS else gr
        scale = float(centred @ centred) / max(H + self.params.reg_lambda, _EPS)
        pool = np.log(max(n_scanned, 1)) + MC_THRESHOLD_LOG * np.log(max(gr.size, 2))
        return float(self.params.mc_penalty * scale * pool)

    def _chance_gain(self, scored, gr, hr, rng) -> float:
        """The best gain this node's own candidates reach on shuffled gradients.

        Permuting the (g, h) pairs across rows breaks any relationship between a
        feature and the target while leaving both marginal distributions alone,
        so the best gain that survives is what this pool produces from noise.

        This is the instrument that fixed the ridge base in V6, one level down.
        There it stopped a null ridge being boosted into confident nonsense;
        here it stops a node taking a noise split seriously -- measured in
        validation/HEADROOM.md as the ceiling on this architecture, and in V7 as
        a five-point tax wherever the trees had no static block to work with.

        **What V8 got wrong** (roadmap item 2d).  It took the maximum gain over
        every permutation *and* every candidate, so the bar tightened as
        ``selection_null`` grew: asking for a more precise null also asked for a
        harsher one, and the knob could not be turned up without changing the
        test.  A single permutation, meanwhile, is not a test at all -- comparing
        an observed maximum against one draw of the null maximum accepts roughly
        half of pure noise.

        The fix is the pattern this library already uses one level down, in
        ``DenseBase._chance_r2``: take the maximum *within* each permutation,
        then a fixed quantile *across* them.  More permutations now estimate the
        same bar more precisely instead of moving it.
        ``selection_null_quantile`` defaults to 1.0, which reproduces V8 exactly
        at ``selection_null=1``, so the old arm is still available to compare
        against rather than being quietly redefined.
        """
        p = self.params
        if not scored:
            return 0.0
        maxima = []
        for _ in range(p.selection_null):
            order = rng.permutation(gr.size)
            gp, hp = gr[order], hr[order]
            best = 0.0
            for f in scored:
                found = scan_threshold(
                    f, gp, hp, p.reg_lambda, p.gamma,
                    p.min_child_weight, p.min_samples_leaf,
                )
                if found is not None and found[0] > best:
                    best = found[0]
            maxima.append(best)
        if not maxima:
            return 0.0
        return float(np.quantile(maxima, p.selection_null_quantile))

    def _product_candidates(self, X_static, rows, rng):
        """A banked temporal feature times a static column, both as magnitudes.

        Roadmap item 5.  ``amp_regression`` is ``transient_height *
        static_coefficient`` and V12 measured this model 11.3 points behind on
        it.  The diagnosis was exact: the only products available were products
        of *ranks*, which throw away the magnitudes the target is built from --
        and they were static-by-static, so the cross the target actually needs,
        series by static, had no representation at all.

        A rank is the wrong currency for this question.  A comparison split --
        "did this happen before that" -- wanted ranks and used them; it was
        deleted after V23.  "How big was this, scaled by that" needs magnitudes,
        which is the whole point of the split kind below.

        The bound that keeps this from being V11 again is clipping, not ranking:
        each side is standardised on its training distribution and clipped to
        the range seen there, so an unfamiliar row contributes the edge of the
        training range instead of an unbounded term.  V11's Apnea collapse to
        0.478 came from an *unpenalised linear* product extrapolating; this
        product is a tree split, and a tree returns a leaf value however large
        its input.
        """
        p = self.params
        bank = self._context.bank
        if p.n_product_candidates <= 0 or bank is None or not X_static.shape[1]:
            return
        entries = bank.entries
        if not entries:
            return
        stats = self._context.static_stats
        if stats is None:
            return
        center, scale, low, high = stats

        # Entries are drawn uniformly.  Drawing them in proportion to
        # cumulative gain -- the bank's own ordering, the quantity it evicts by
        # -- was tried and did not help on ``amp_regression`` (+2.0% against
        # +1.8% at four candidates, and worse at sixteen), so the complication
        # is not carried on the strength of one scenario and one seed.
        for _ in range(self._draw_count(p.n_product_candidates)):
            entry = entries[int(rng.integers(len(entries)))]
            col = int(rng.integers(X_static.shape[1]))
            if not (scale[col] > _EPS):
                continue
            finite = entry.column[np.isfinite(entry.column)]
            if finite.size < 2:
                continue
            inner_center = float(finite.mean())
            inner_scale = float(finite.std())
            if inner_scale <= _EPS:
                continue
            standard = (finite - inner_center) / inner_scale
            spec = SplitSpec(
                kind="product",
                col=col,
                name_hint=(self._context.static_names[col]
                           if self._context.static_names else ""),
                inner_spec=replace(entry.spec),
                inner_center=inner_center,
                inner_scale=inner_scale,
                inner_bounds=(float(standard.min()), float(standard.max())),
                static_center=float(center[col]),
                static_scale=float(scale[col]),
                static_bounds=(float(low[col]), float(high[col])),
            )
            values = eval_split_feature(spec, X_static, self._X_series, rows,
                                        self._context.pyramid)
            yield values, spec

    #: ``_comparison_candidates`` lived here: "did this happen before that", a
    #: banked event time ranked against a static column.  V15 failed it on eight
    #: UEA datasets; it was kept once because removing it cost 9.9 points on
    #: ``bump_order``, the scenario that is an XOR of which transient came first.
    #: V23 then ran the same arms on eight further UEA datasets and on
    #: ``bump_order`` itself: +9.9 on the scenario, 0 of 8 and a mean of -0.2 on
    #: the real suite.  Deleted -- it only ever worked on the task written for it.


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
