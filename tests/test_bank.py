"""The feature bank, comparison splits, and the guards that keep them honest.

The bank is a training-time cache, and the property that makes it safe is that
nothing in a fitted model depends on it: every split carries a self-contained
copy of what it needs.  Several tests here exist only to hold that line, because
if it ever broke, predictions would drift with the cache rather than fail
loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.bank import FeatureBank
from heartwood.datasets import make_bump_interaction, make_timing_task
from heartwood.features import ecdf, eval_split_feature
from heartwood.filters import Pyramid
from heartwood.splits import SplitSpec
from heartwood.tree import FitContext, TemporalTree, TreeParams


def fitted_model(scenario=make_bump_interaction, n=250, rounds=25, **kwargs):
    X_static, X_series, y = scenario(n=n, seed=0)
    model = HeartwoodClassifier(n_estimators=rounds, random_state=0, **kwargs)
    model.fit(X_static, X_series, y)
    return model, X_static, X_series, y


# ------------------------------------------------------------------- basics


def test_promotes_a_new_feature_once(rng):
    bank = FeatureBank(max_entries=8)
    spec = SplitSpec(kind="interval", channel=0, start=0, end=10, stat="mean")
    column = rng.normal(size=50)

    assert bank.promote(spec, column, gain=1.0, round_index=0) is True
    assert len(bank) == 1
    # the same feature again is credit, not a second copy
    assert bank.promote(spec, column, gain=2.0, round_index=1) is False
    assert len(bank) == 1
    assert bank.entries[0].cumulative_gain == 3.0
    assert bank.entries[0].last_win_round == 1


def test_rejects_a_near_duplicate_column(rng):
    bank = FeatureBank(max_entries=8)
    column = rng.normal(size=100)
    bank.promote(SplitSpec(kind="interval", stat="mean", start=0, end=5), column, 1.0, 0)

    twin = column * 3.0 + 7.0  # a different spec, the same information
    stored = bank.promote(
        SplitSpec(kind="interval", stat="mean", start=1, end=6), twin, 1.0, 0
    )
    assert stored is False and len(bank) == 1
    assert bank.n_rejected_duplicate == 1


def test_respects_its_cap_and_evicts_the_least_useful(rng):
    bank = FeatureBank(max_entries=3)
    for i in range(6):
        spec = SplitSpec(kind="interval", channel=0, start=i, end=i + 4, stat="mean")
        bank.promote(spec, rng.normal(size=60), gain=float(i), round_index=i)

    assert len(bank) == 3
    assert bank.n_evicted == 3
    # the survivors are the highest-gain ones
    assert sorted(e.cumulative_gain for e in bank.entries) == [3.0, 4.0, 5.0]


def test_ignores_an_all_missing_column():
    bank = FeatureBank()
    stored = bank.promote(
        SplitSpec(kind="interval", stat="mean"), np.full(40, np.nan), 1.0, 0
    )
    assert stored is False and len(bank) == 0


def test_candidates_hand_out_copies_not_the_stored_spec(rng):
    """A tree stamps a threshold onto the spec it wins with; ours must stay clean."""
    bank = FeatureBank()
    bank.promote(SplitSpec(kind="interval", stat="mean", start=0, end=4), rng.normal(size=30), 1.0, 0)

    rows = np.arange(30)
    _, handed_out = next(iter(bank.candidates(rows)))
    handed_out.threshold = 42.0
    assert np.isnan(bank.entries[0].spec.threshold)
    assert handed_out is not bank.entries[0].spec


def test_position_entries_are_identified_for_comparison_splits(rng):
    bank = FeatureBank()
    bank.promote(SplitSpec(kind="filter_resp", channel=0, scale=0,
                           template=np.arange(9.0)), rng.normal(size=40), 1.0, 0)
    bank.promote(SplitSpec(kind="filter_pos", channel=0, scale=0,
                           template=np.arange(9.0)), rng.random(40), 1.0, 0)

    positions = bank.position_entries()
    assert [e.spec.kind for e in positions] == ["filter_pos"]
    assert positions[0].grid is not None, "a position entry needs its frozen rank grid"
    assert np.all(np.diff(positions[0].grid) >= 0)


# -------------------------------------------------------------- integration


def test_bank_fills_up_during_fitting():
    model, *_ = fitted_model()
    bank = model._core.bank
    assert bank is not None and len(bank) > 0
    assert bank.n_promoted >= len(bank)
    assert all(e.cumulative_gain > 0 for e in bank.entries)


def test_banked_columns_match_recomputing_their_spec_from_scratch():
    """The cached column must be exactly what the spec means, or reuse is a lie."""
    model, X_static, X_series, _ = fitted_model()
    pyramid = Pyramid(X_series, model.filter_len)
    rows = np.arange(len(X_static))

    for entry in model._core.bank.entries:
        recomputed = eval_split_feature(entry.spec, X_static, X_series, rows, pyramid)
        assert np.allclose(entry.column, recomputed, equal_nan=True, atol=1e-12), (
            f"cached column drifted from its spec: {entry.spec.feature_name()}"
        )


def test_predictions_do_not_depend_on_the_bank_surviving():
    """Every split is self-contained, so throwing the cache away changes nothing."""
    model, X_static, X_series, _ = fitted_model()
    before = model.predict_proba(X_static, X_series)

    model._core.bank = None  # as if every entry had been evicted
    assert np.array_equal(model.predict_proba(X_static, X_series), before)


def test_a_tiny_bank_still_produces_a_valid_model():
    model, X_static, X_series, y = fitted_model(bank_max=2)
    assert len(model._core.bank) <= 2
    assert model._core.bank.n_evicted > 0
    accuracy = (model.predict(X_static, X_series) == y).mean()
    assert accuracy > 0.6 and np.isfinite(model.predict_proba(X_static, X_series)).all()


def test_disabling_the_bank_still_fits():
    model, X_static, X_series, y = fitted_model(bank_enabled=False)
    assert model._core.bank is None
    assert (model.predict(X_static, X_series) == y).mean() > 0.6
    assert not any(s.kind == "comparison" for s in model._core.iter_splits()), (
        "comparison splits need banked positions, so they cannot appear without a bank"
    )


def test_bank_summary_is_readable():
    model, *_ = fitted_model()
    summary = model._core.bank.summary()
    assert summary and all(isinstance(name, str) for name, _, _ in summary)
    assert [gain for _, gain, _ in summary] == sorted(
        (g for _, g, _ in summary), reverse=True
    )


# ------------------------------------------------------ comparison splits


def test_comparison_feature_is_a_difference_of_frozen_ranks(rng):
    values = rng.normal(size=200)
    grid = np.sort(values)
    ranks = ecdf(values, grid)
    assert np.all((ranks > 0) & (ranks <= 1.0))
    assert np.isnan(ecdf(np.array([np.nan]), grid))[0]
    # a value beyond everything seen in training saturates rather than extrapolates
    assert ecdf(np.array([values.max() + 10]), grid)[0] == 1.0
    assert ecdf(np.array([values.min() - 10]), grid)[0] == 0.0


def best_single_threshold(feature: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the best possible single threshold on one feature."""
    finite = feature[np.isfinite(feature)]
    if finite.size == 0:
        return 0.5
    grid = np.quantile(finite, np.linspace(0, 1, 201))
    return max(
        max(((feature <= t).astype(int) == y).mean(), ((feature > t).astype(int) == y).mean())
        for t in grid
    )


def test_one_comparison_split_beats_any_split_on_position_alone():
    """Position-versus-deadline, in one threshold instead of a staircase.

    The rank difference is an *approximation* of "did it happen before the
    deadline": ranking puts two differently-distributed quantities on a common
    scale, but that mapping is monotone rather than exact, so a single split
    does not reach the oracle rule.  What it must do is beat what any single
    axis-aligned split can manage, since that is the work it exists to save.
    """
    X_static, X_series, y = make_timing_task(n=600, seed=0)

    template = X_series[0, 0, 8:24].copy()
    from heartwood.features import shapelet_features

    _, position = shapelet_features(X_series[:, 0, :], template)

    spec = SplitSpec(
        kind="comparison",
        col=0,
        position_spec=SplitSpec(kind="shapelet_pos", channel=0, shapelet=template),
        position_grid=np.sort(position[np.isfinite(position)]),
        static_grid=np.sort(X_static[:, 0]),
    )
    feature = eval_split_feature(spec, X_static, X_series, np.arange(len(y)))

    comparison = best_single_threshold(feature, y)
    position_only = best_single_threshold(position, y)
    deadline_only = best_single_threshold(X_static[:, 0], y)

    assert comparison > 0.75, f"comparison split only reached {comparison:.3f}"
    assert comparison > max(position_only, deadline_only) + 0.05, (
        f"comparison {comparison:.3f} vs position alone {position_only:.3f} "
        f"and deadline alone {deadline_only:.3f}"
    )


def test_comparison_splits_get_used_on_the_timing_task():
    model, X_static, X_series, y = fitted_model(make_timing_task, n=400, rounds=40)
    kinds = {spec.kind for spec in model._core.iter_splits()}
    assert "comparison" in kinds, sorted(kinds)

    described = [d for d, _ in model.dump_splits() if "rank(" in d]
    assert described and "static[" in described[0]


def test_comparison_ranks_are_frozen_at_fit_time():
    """Ranks recomputed on the batch at hand would mean something different."""
    model, X_static, X_series, y = fitted_model(make_timing_task, n=300, rounds=30)
    specs = [s for s in model._core.iter_splits() if s.kind == "comparison"]
    if not specs:
        pytest.skip("no comparison split was selected in this draw")

    grids = [(s.position_grid.copy(), s.static_grid.copy()) for s in specs]
    # predicting on a wildly different batch must not touch the stored grids
    model.predict(X_static[:10] * 100.0, X_series[:10])
    for spec, (position_grid, static_grid) in zip(specs, grids):
        assert np.array_equal(spec.position_grid, position_grid)
        assert np.array_equal(spec.static_grid, static_grid)


# ------------------------------------------------- fit / predict agreement


def test_leaf_values_still_match_under_predict_time_routing(rng):
    """The strongest routing check, now with filters, bank and comparisons live."""
    X_static, X_series, y = make_timing_task(n=200, seed=0)
    g = rng.normal(size=len(y))
    h = np.ones(len(y))

    pyramid = Pyramid(X_series, 9)
    bank = FeatureBank()
    context = FitContext(
        pyramid=pyramid,
        bank=bank,
        static_grids=[np.sort(X_static[:, j]) for j in range(X_static.shape[1])],
    )
    params = TreeParams(max_depth=4, n_filter_candidates=8, n_comparison_candidates=8)

    # a first tree fills the bank so the second sees banked and comparison candidates
    TemporalTree(params).fit(X_static, X_series, g, h, np.arange(len(y)), rng, context)
    tree = TemporalTree(params).fit(
        X_static, X_series, g, h, np.arange(len(y)), rng, context
    )

    values = tree.predict(X_static, X_series, pyramid)
    leaves = tree.apply(X_static, X_series, pyramid)
    for leaf in np.unique(leaves):
        rows = leaves == leaf
        expected = -g[rows].sum() / (h[rows].sum() + params.reg_lambda)
        assert np.allclose(values[rows], expected, atol=1e-9)


def test_stored_templates_do_not_alias_the_training_series():
    """Overwriting the training data after fitting must not change predictions."""
    X_static, X_series, y = make_bump_interaction(n=200, seed=0)
    held = X_series.copy()
    model = HeartwoodClassifier(
        n_estimators=20, random_state=0, n_filter_candidates=8
    ).fit(X_static, X_series, y)
    before = model.predict_proba(X_static, held)

    X_series[:] = 0.0
    assert np.array_equal(model.predict_proba(X_static, held), before)
    for spec in model._core.iter_splits():
        if spec.template is not None:
            assert np.isfinite(spec.template).all()


def test_filters_are_scored_only_through_the_split_scan():
    """A fitted template must earn its place on the gain scan, not on its own fit."""
    model, *_ = fitted_model(n_filter_candidates=8)
    assert any(s.kind.startswith("filter") for s in model._core.iter_splits())
    for spec in model._core.iter_splits():
        assert spec.gain > 0
        assert np.isfinite(spec.threshold)
        if spec.kind.startswith("filter"):
            assert np.isclose(np.linalg.norm(spec.template), 1.0, atol=1e-9)
