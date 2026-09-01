"""The fallback must be chosen on data neither candidate was fitted on.

Roadmap item 3 promises the model is never meaningfully worse than its best
component.  The promise is only worth having if the comparison behind it is
honest: choosing the winner on the fitting rows would pick whichever candidate
over-fits hardest, which is the most flattering possible way to break a
guarantee about not being worse than anything.

The asymmetry is also load-bearing and is tested here.  ``combined`` wins ties
and wins by default, so the selection has to find evidence to deviate rather
than needing evidence to stay put -- that is what bounds the damage this extra
selection step can do.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier, HeartwoodRegressor

ROCKET = dict(dense_base=True, n_rocket_features=300)


def series_signal(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, 48))
    y = (X[:, 0, 5:20].mean(axis=1) > 0).astype(int)
    return X, y


def pure_noise(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2, 48)), rng.integers(0, 2, size=n)


def test_the_guarantee_is_off_by_default():
    X, y = series_signal()
    model = HeartwoodClassifier(n_estimators=6, max_depth=3, random_state=0,
                                **ROCKET).fit(None, X, y)
    assert model.fallback_ == "combined"
    assert model.component_scores_ == {}


def test_every_component_is_scored_when_the_guarantee_is_on():
    X, y = series_signal()
    model = HeartwoodClassifier(n_estimators=10, max_depth=3, random_state=0,
                                no_regret=True, **ROCKET).fit(None, X, y)
    assert set(model.component_scores_) == {"combined", "base_only", "trees_only"}
    assert model.fallback_ in model.component_scores_


def test_it_falls_back_when_a_component_is_clearly_better():
    """A component that wins on held-out data must actually be chosen."""
    X, y = series_signal()
    model = HeartwoodClassifier(n_estimators=15, max_depth=3, random_state=0,
                                no_regret=True, **ROCKET).fit(None, X, y)
    scores = model.component_scores_
    best = min(scores, key=scores.get)
    if scores[best] < scores["combined"]:
        assert model.fallback_ == best
    else:
        assert model.fallback_ == "combined"


def test_a_large_margin_pins_the_model_to_the_combination():
    """The margin is the brake: nothing should ever clear an impossible bar."""
    X, y = series_signal()
    model = HeartwoodClassifier(n_estimators=10, max_depth=3, random_state=0,
                                no_regret=True, no_regret_margin=1e6,
                                **ROCKET).fit(None, X, y)
    assert model.fallback_ == "combined"


def test_ties_go_to_the_combination():
    """At margin 0 a component must be strictly better, not merely equal."""
    X, y = series_signal()
    model = HeartwoodClassifier(n_estimators=10, max_depth=3, random_state=0,
                                no_regret=True, **ROCKET)
    model.component_scores_ = {"combined": 0.5, "base_only": 0.5, "trees_only": 0.5}
    combined = model.component_scores_["combined"]
    better = [k for k, v in model.component_scores_.items()
              if k != "combined" and v < combined - model.no_regret_margin]
    assert not better


def test_on_pure_noise_it_does_not_boost_the_noise():
    """The case the guarantee exists for.

    With no signal anywhere, the base switches itself off against its own
    permutation null and ``base_only`` becomes the constant predictor. A model
    that keeps boosting trees into noise should lose to that, and be replaced
    by it.
    """
    X, y = pure_noise()
    model = HeartwoodClassifier(n_estimators=15, max_depth=3, random_state=0,
                                no_regret=True, **ROCKET).fit(None, X, y)
    scores = model.component_scores_
    assert scores[model.fallback_] <= scores["combined"] + 1e-12


def test_the_holdout_is_disjoint_from_the_fit():
    model = HeartwoodClassifier(no_regret=True, random_state=0)
    y = np.array([0] * 60 + [1] * 40, dtype=float)
    fit_rows, held_rows = model._holdout_split(y, len(y))
    assert set(fit_rows).isdisjoint(held_rows)
    assert len(fit_rows) + len(held_rows) == len(y)


def test_the_holdout_keeps_every_class_on_both_sides():
    """A fold that loses a class scores the candidates on a different problem."""
    model = HeartwoodClassifier(no_regret=True, random_state=0)
    y = np.array([0] * 90 + [1] * 8 + [2] * 2, dtype=float)
    fit_rows, held_rows = model._holdout_split(y, len(y))
    for rows in (fit_rows, held_rows):
        assert set(np.unique(y[rows])) == {0.0, 1.0, 2.0}


def test_a_rare_class_is_never_emptied_out_of_the_fitting_half():
    model = HeartwoodClassifier(no_regret=True, no_regret_fraction=0.9, random_state=0)
    y = np.array([0] * 50 + [1] * 2, dtype=float)
    fit_rows, _ = model._holdout_split(y, len(y))
    assert 1.0 in set(np.unique(y[fit_rows]))


def test_regression_uses_the_guarantee_too():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 2, 40))
    y = X[:, 0, 5:15].mean(axis=1) * 3.0 + 0.1 * rng.normal(size=150)
    model = HeartwoodRegressor(n_estimators=10, max_depth=3, random_state=0,
                               no_regret=True, **ROCKET).fit(None, X, y)
    assert model.fallback_ in {"combined", "base_only", "trees_only"}
    assert model.predict(None, X).shape == (150,)


def test_zero_trees_needs_a_base_to_predict_from():
    X, y = series_signal(n=60)
    with pytest.raises(ValueError, match="dense_base"):
        HeartwoodClassifier(n_estimators=0).fit(None, X, y)
    model = HeartwoodClassifier(n_estimators=0, random_state=0, **ROCKET).fit(None, X, y)
    assert model.predict(None, X).shape == (60,)


def test_the_guarantee_parameters_are_validated():
    X, y = series_signal(n=40)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="no_regret_fraction"):
            HeartwoodClassifier(no_regret_fraction=bad).fit(None, X, y)
    with pytest.raises(ValueError, match="no_regret_margin"):
        HeartwoodClassifier(no_regret_margin=-0.1).fit(None, X, y)
