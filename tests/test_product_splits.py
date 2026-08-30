"""A product split must carry magnitudes, and must stay bounded off-range.

Roadmap item 5.  V12 diagnosed ``amp_regression`` exactly: the target is
``transient_height * static_coefficient``, a product of magnitudes, and the only
products the model could form were products of *ranks* -- which discard the
magnitudes -- between two *statics*, when the cross the target needs is series
by static.

Two things can go wrong, and they pull in opposite directions.  Use ranks and
the feature is bounded but cannot express the target.  Use raw magnitudes
unbounded and V11 comes back: an out-of-range row produced an exploding product
and Apnea-ECG fell to 0.478 AUC, below chance.  These tests hold both ends.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodRegressor
from heartwood.datasets import make_shape_amplitude_regression
from heartwood.features import eval_split_feature
from heartwood.splits import SplitSpec


def product_spec(**kwargs):
    inner = SplitSpec(kind="interval", channel=0, start=0, end=4, stat="max")
    defaults = dict(kind="product", col=0, inner_spec=inner,
                    inner_center=0.0, inner_scale=1.0, inner_bounds=(-2.0, 2.0),
                    static_center=0.0, static_scale=1.0, static_bounds=(-2.0, 2.0))
    defaults.update(kwargs)
    return SplitSpec(**defaults)


def test_the_feature_is_a_product_of_magnitudes_not_ranks():
    """Doubling either side must double the feature. A rank could not do this."""
    X_series = np.zeros((2, 1, 4))
    X_series[0, 0, :] = 1.0
    X_series[1, 0, :] = 2.0
    X_static = np.array([[1.0], [1.0]])
    values = eval_split_feature(product_spec(), X_static, X_series, np.arange(2))
    assert values[1] == pytest.approx(2.0 * values[0])


def test_scaling_the_static_side_scales_the_feature():
    X_series = np.ones((2, 1, 4))
    X_static = np.array([[1.0], [2.0]])
    values = eval_split_feature(product_spec(), X_static, X_series, np.arange(2))
    assert values[1] == pytest.approx(2.0 * values[0])


def test_an_out_of_range_row_is_clipped_rather_than_exploding():
    """The V11 guard. A wildly out-of-range static must not blow the term up."""
    X_series = np.ones((3, 1, 4))
    X_static = np.array([[1.0], [2.0], [1e6]])
    values = eval_split_feature(product_spec(), X_static, X_series, np.arange(3))
    assert values[2] == pytest.approx(values[1])          # both clipped to +2
    assert abs(values[2]) <= 2.0 * 2.0 + 1e-9


def test_both_sides_are_clipped():
    X_series = np.full((2, 1, 4), 1.0)
    X_series[1, 0, :] = 1e9
    X_static = np.array([[1.0], [1.0]])
    values = eval_split_feature(product_spec(), X_static, X_series, np.arange(2))
    assert abs(values[1]) <= 2.0 * 2.0 + 1e-9


def test_a_zero_spread_side_does_not_divide_by_zero():
    X_series = np.ones((2, 1, 4))
    X_static = np.array([[1.0], [1.0]])
    values = eval_split_feature(product_spec(static_scale=0.0), X_static,
                                X_series, np.arange(2))
    assert np.isfinite(values).all()


def test_the_spec_is_self_contained():
    """A fitted tree must not depend on the training arrays staying alive."""
    Xs, Xt, y = make_shape_amplitude_regression(n=120, seed=0)
    model = HeartwoodRegressor(n_estimators=20, max_depth=3, random_state=0,
                               n_product_candidates=8).fit(Xs, Xt, y)
    specs = [node["spec"] for rt in model._core.trees_ for t in rt
             for node in t.nodes
             if not node["leaf"] and node["spec"].kind == "product"]
    assert specs, "no product split was ever chosen"
    for spec in specs:
        assert spec.inner_spec is not None
        assert spec.inner_bounds is not None and spec.static_bounds is not None


def test_predictions_are_reproducible_from_the_specs_alone():
    Xs, Xt, y = make_shape_amplitude_regression(n=150, seed=1)
    model = HeartwoodRegressor(n_estimators=20, max_depth=3, random_state=0,
                               n_product_candidates=8).fit(Xs, Xt, y)
    first = model.predict(Xs, Xt)
    second = model.predict(Xs.copy(), Xt.copy())
    assert np.allclose(first, second)


def test_product_splits_are_off_by_default():
    Xs, Xt, y = make_shape_amplitude_regression(n=100, seed=0)
    model = HeartwoodRegressor(n_estimators=10, max_depth=3,
                               random_state=0).fit(Xs, Xt, y)
    kinds = {node["spec"].kind for rt in model._core.trees_ for t in rt
             for node in t.nodes if not node["leaf"]}
    assert "product" not in kinds


def test_the_count_is_validated():
    Xs, Xt, y = make_shape_amplitude_regression(n=60, seed=0)
    with pytest.raises(ValueError, match="n_product_candidates"):
        HeartwoodRegressor(n_product_candidates=-1).fit(Xs, Xt, y)


def test_products_need_a_static_block_to_cross_with():
    """Series-only data has nothing to multiply by; the source must stay quiet."""
    Xs, Xt, y = make_shape_amplitude_regression(n=100, seed=0)
    model = HeartwoodRegressor(n_estimators=10, max_depth=3, random_state=0,
                               n_product_candidates=8).fit(None, Xt, y)
    kinds = {node["spec"].kind for rt in model._core.trees_ for t in rt
             for node in t.nodes if not node["leaf"]}
    assert "product" not in kinds
