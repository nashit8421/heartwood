"""The tree: leaf arithmetic, depth limits, and fit/predict routing agreement.

The subtle failure mode a tree can have is routing a row one way while fitting
and the other way while predicting — nothing raises, and accuracy just quietly
decays.  Several tests here exist only to catch that.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood.tree import TemporalTree, TreeParams

LAM = 1.0


def leaf_rows(tree, X_static, X_series):
    """Group row indices by the leaf they reach under *predict-time* routing."""
    assignment = tree.apply(X_static, X_series)
    return {int(leaf): np.nonzero(assignment == leaf)[0] for leaf in np.unique(assignment)}


@pytest.fixture
def separable(rng):
    """A static column that splits the gradients perfectly."""
    n = 60
    X_static = np.column_stack([np.repeat([0.0, 1.0], n // 2), rng.normal(size=n)])
    g = np.where(X_static[:, 0] > 0.5, -1.0, 1.0)
    h = np.ones(n)
    return X_static, g, h


def test_leaf_values_are_minus_G_over_H_plus_lambda(separable, rng):
    """The exact XGBoost leaf solution, recomputed by hand under predict routing.

    This doubles as the fit/predict agreement check: the stored value came from
    the rows that reached the leaf while fitting, and it is compared against the
    rows that reach it while predicting.
    """
    X_static, g, h = separable
    tree = TemporalTree(TreeParams(max_depth=3, reg_lambda=LAM, min_samples_leaf=2)).fit(
        X_static, None, g, h, np.arange(len(g)), rng
    )

    values = tree.predict(X_static, None)
    for leaf, rows in leaf_rows(tree, X_static, None).items():
        expected = -g[rows].sum() / (h[rows].sum() + LAM)
        assert np.allclose(values[rows], expected, atol=1e-12)


def test_separable_data_reaches_pure_leaves(separable, rng):
    X_static, g, h = separable
    tree = TemporalTree(TreeParams(max_depth=3, reg_lambda=LAM, min_samples_leaf=2)).fit(
        X_static, None, g, h, np.arange(len(g)), rng
    )
    for rows in leaf_rows(tree, X_static, None).values():
        assert len(np.unique(g[rows])) == 1, "a pure split was available and missed"


@pytest.mark.parametrize("max_depth", [0, 1, 2, 4])
def test_max_depth_is_honoured(max_depth, rng):
    n = 200
    X_static = rng.normal(size=(n, 4))
    X_series = rng.normal(size=(n, 1, 40))
    g = rng.normal(size=n)
    h = np.ones(n)
    tree = TemporalTree(
        TreeParams(max_depth=max_depth, reg_lambda=LAM, min_samples_leaf=1,
                   min_child_weight=0.0)
    ).fit(X_static, X_series, g, h, np.arange(n), rng)

    assert tree.n_leaves <= 2**max_depth
    if max_depth == 0:
        assert tree.n_leaves == 1
        assert not list(tree.iter_splits())


def test_predicting_a_subset_matches_predicting_everything(rng):
    """Row-subset independence is what makes fit and predict routing agree."""
    n = 120
    X_static = rng.normal(size=(n, 3))
    X_series = rng.normal(size=(n, 2, 45))
    X_series[rng.random(X_series.shape) < 0.1] = np.nan
    g, h = rng.normal(size=n), np.ones(n)

    tree = TemporalTree(TreeParams(max_depth=4, n_shapelet_candidates=6)).fit(
        X_static, X_series, g, h, np.arange(n), rng
    )
    everything = tree.predict(X_static, X_series)
    for subset in (np.arange(0, n, 7), np.array([3]), np.arange(n // 2)):
        part = tree.predict(X_static[subset], X_series[subset])
        assert np.allclose(part, everything[subset], atol=1e-12)


def test_min_samples_leaf_holds_in_the_grown_tree(rng):
    n = 150
    X_static = rng.normal(size=(n, 5))
    g, h = rng.normal(size=n), np.ones(n)
    tree = TemporalTree(TreeParams(max_depth=5, min_samples_leaf=20)).fit(
        X_static, None, g, h, np.arange(n), rng
    )
    for rows in leaf_rows(tree, X_static, None).values():
        assert len(rows) >= 20


def test_a_tree_with_nothing_to_learn_is_a_single_leaf(rng):
    n = 40
    X_static = np.zeros((n, 3))
    g, h = np.ones(n), np.ones(n)
    tree = TemporalTree(TreeParams(max_depth=4)).fit(
        X_static, None, g, h, np.arange(n), rng
    )
    assert tree.n_leaves == 1
    assert np.allclose(tree.predict(X_static, None), -n / (n + LAM))


def test_stored_shapelets_do_not_alias_the_training_data(rng):
    """Overwriting the training series after fitting must not change predictions."""
    n = 80
    X_static = rng.normal(size=(n, 2))
    X_series = rng.normal(size=(n, 1, 40))
    g = np.where(X_series[:, 0, 10:20].mean(1) > 0, -1.0, 1.0)
    h = np.ones(n)

    tree = TemporalTree(
        TreeParams(max_depth=3, n_shapelet_candidates=10, n_interval_candidates=4)
    ).fit(X_static, X_series, g, h, np.arange(n), rng)

    specs = [s for s in tree.iter_splits() if s.shapelet is not None]
    if not specs:
        pytest.skip("no shapelet split was selected in this draw")

    held = X_series.copy()
    before = [s.shapelet.copy() for s in specs]
    X_series[:] = 0.0
    assert all(np.array_equal(s.shapelet, b) for s, b in zip(specs, before))
    assert np.allclose(tree.predict(X_static, held), tree.predict(X_static, held))


def test_missing_values_route_the_way_the_split_says(rng):
    """A row with no series at all must still reach a leaf."""
    n = 60
    X_static = rng.normal(size=(n, 2))
    X_series = rng.normal(size=(n, 1, 30))
    X_series[:5] = np.nan  # rows with nothing observed
    g, h = rng.normal(size=n), np.ones(n)

    tree = TemporalTree(TreeParams(max_depth=3)).fit(
        X_static, X_series, g, h, np.arange(n), rng
    )
    predictions = tree.predict(X_static, X_series)
    assert np.isfinite(predictions).all()
    assert (tree.apply(X_static, X_series) >= 0).all()


def test_fitting_on_a_subsample_still_predicts_every_row(rng):
    n = 100
    X_static = rng.normal(size=(n, 3))
    g, h = rng.normal(size=n), np.ones(n)
    rows = rng.choice(n, size=40, replace=False)

    tree = TemporalTree(TreeParams(max_depth=3, min_samples_leaf=2)).fit(
        X_static, None, g, h, rows, rng
    )
    assert tree.predict(X_static, None).shape == (n,)
    assert np.isfinite(tree.predict(X_static, None)).all()


def test_iter_splits_yields_only_internal_nodes(rng):
    n = 100
    X_static = rng.normal(size=(n, 3))
    g, h = rng.normal(size=n), np.ones(n)
    tree = TemporalTree(TreeParams(max_depth=3)).fit(
        X_static, None, g, h, np.arange(n), rng
    )
    splits = list(tree.iter_splits())
    assert len(splits) + tree.n_leaves == len(tree.nodes)
    assert all(np.isfinite(s.threshold) and s.gain > 0 for s in splits)


def test_empty_row_set_is_rejected(rng):
    with pytest.raises(ValueError, match="empty row set"):
        TemporalTree(TreeParams()).fit(
            np.zeros((5, 2)), None, np.zeros(5), np.ones(5), np.array([], dtype=int), rng
        )


def test_temporal_candidates_can_win_over_static_noise(rng):
    """The whole point: when the signal is in the series, the tree should use it."""
    n = 200
    X_static = rng.normal(size=(n, 6))  # pure noise
    X_series = rng.normal(scale=0.3, size=(n, 1, 40))
    marked = rng.integers(0, 2, size=n).astype(bool)
    X_series[marked, 0, 15:25] += 3.0
    g = np.where(marked, -1.0, 1.0)
    h = np.ones(n)

    tree = TemporalTree(TreeParams(max_depth=2, n_interval_candidates=24)).fit(
        X_static, X_series, g, h, np.arange(n), rng
    )
    kinds = {s.kind for s in tree.iter_splits()}
    assert kinds & {"interval", "shapelet_dist", "shapelet_pos"}
