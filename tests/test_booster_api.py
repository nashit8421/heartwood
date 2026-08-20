"""The public estimators end to end: learning, plumbing, and input handling."""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier, HeartwoodRegressor
from heartwood.datasets import (
    make_bump_interaction,
    make_pure_static,
    make_slope_window,
    make_static_plus_noise_series,
)

FAST = dict(n_estimators=15, random_state=0)


# ------------------------------------------------------------------ learning


def test_can_overfit_a_small_problem():
    """If the booster cannot drive training loss down, nothing else matters."""
    X_static, X_series, y = make_bump_interaction(n=60, seed=0)
    model = HeartwoodClassifier(
        n_estimators=120, learning_rate=0.3, min_samples_leaf=1,
        min_child_weight=0.0, random_state=0,
    ).fit(X_static, X_series, y)

    history = model.train_history_
    assert history[-1] < 0.05, f"train logloss stalled at {history[-1]:.4f}"
    assert history[-1] < history[0]


def test_training_loss_decreases_overall():
    X_static, X_series, y = make_bump_interaction(n=150, seed=1)
    history = HeartwoodClassifier(n_estimators=40, random_state=0).fit(
        X_static, X_series, y
    ).train_history_
    assert history[-1] < 0.5 * history[0]
    increases = sum(b > a for a, b in zip(history, history[1:]))
    assert increases <= 3, "loss should fall steadily, not oscillate"


def test_beats_the_aggregate_workaround_on_a_temporal_signal():
    """The library's reason to exist, as a test.

    The same booster is given the same data twice: once as global aggregates
    (the industry-standard workaround, which is provably lossy on this scenario)
    and once as the raw series.
    """
    def aggregate(series):
        X = series[:, 0, :]
        t = np.arange(X.shape[1], dtype=float)
        tc = t - t.mean()
        return np.column_stack([
            X.mean(1), X.std(1), X.min(1), X.max(1), (X * tc).sum(1) / (tc * tc).sum(),
            np.median(X, axis=1), np.abs(np.diff(X, axis=1)).mean(1),
            X[:, 0], X[:, -1], X[:, -1] - X[:, 0],
        ])

    X_static, X_series, y = make_bump_interaction(n=500, seed=0)
    X_static_te, X_series_te, y_te = make_bump_interaction(n=1000, seed=99)
    common = dict(n_estimators=120, random_state=0)

    baseline = HeartwoodClassifier(**common).fit(
        np.hstack([X_static, aggregate(X_series)]), None, y
    )
    baseline_acc = (
        baseline.predict(np.hstack([X_static_te, aggregate(X_series_te)])) == y_te
    ).mean()

    model = HeartwoodClassifier(**common).fit(X_static, X_series, y)
    acc = (model.predict(X_static_te, X_series_te) == y_te).mean()

    assert baseline_acc < 0.60, "the scenario is supposed to defeat aggregation"
    assert acc > baseline_acc + 0.10, f"raw={acc:.3f} vs aggregate={baseline_acc:.3f}"


def test_temporal_machinery_does_not_hurt_a_purely_static_problem():
    """The control: offered a noise series, the model must decline to use it."""
    X_static, X_series, y = make_static_plus_noise_series(n=400, seed=0)
    X_static_te, X_series_te, y_te = make_static_plus_noise_series(n=1000, seed=7)

    with_series = HeartwoodClassifier(n_estimators=60, random_state=0).fit(
        X_static, X_series, y
    )
    without = HeartwoodClassifier(n_estimators=60, random_state=0).fit(X_static, None, y)

    acc_with = (with_series.predict(X_static_te, X_series_te) == y_te).mean()
    acc_without = (without.predict(X_static_te, None) == y_te).mean()
    assert acc_with > acc_without - 0.05, f"{acc_with:.3f} vs {acc_without:.3f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_regression_learns_a_temporal_target(seed):
    from heartwood.datasets import make_shape_amplitude_regression

    X_static, X_series, y = make_shape_amplitude_regression(n=300, seed=seed)
    X_static_te, X_series_te, y_te = make_shape_amplitude_regression(n=600, seed=seed + 50)

    model = HeartwoodRegressor(n_estimators=60, random_state=0).fit(X_static, X_series, y)
    rmse = np.sqrt(np.mean((model.predict(X_static_te, X_series_te) - y_te) ** 2))
    baseline = np.sqrt(np.mean((y_te - y.mean()) ** 2))
    assert rmse < 0.7 * baseline


# ------------------------------------------------------------------ plumbing


def test_same_seed_reproduces_predictions_exactly():
    X_static, X_series, y = make_bump_interaction(n=150, seed=0)
    a = HeartwoodClassifier(n_estimators=20, random_state=7).fit(X_static, X_series, y)
    b = HeartwoodClassifier(n_estimators=20, random_state=7).fit(X_static, X_series, y)
    c = HeartwoodClassifier(n_estimators=20, random_state=8).fit(X_static, X_series, y)

    assert np.array_equal(a.predict_proba(X_static, X_series), b.predict_proba(X_static, X_series))
    assert not np.array_equal(a.predict_proba(X_static, X_series), c.predict_proba(X_static, X_series))


def test_early_stopping_stops_and_predicts_from_the_best_round():
    X_static, X_series, y = make_bump_interaction(n=120, seed=0)
    X_val, S_val, y_val = make_bump_interaction(n=200, seed=5)

    model = HeartwoodClassifier(
        n_estimators=200, early_stopping_rounds=5, random_state=0
    ).fit(X_static, X_series, y, eval_set=(X_val, S_val, y_val))

    assert model.best_iteration_ < 199
    assert len(model.eval_history_) < 200, "training should have stopped early"
    assert model.eval_history_[model.best_iteration_] == min(model.eval_history_)

    # predictions must come from best_iteration_, not from every round fitted
    raw_best = model._core.predict_raw(
        *model._check_predict_inputs(X_val, S_val)[:2], iteration=model.best_iteration_
    )
    raw_default = model._core.predict_raw(*model._check_predict_inputs(X_val, S_val)[:2])
    assert np.allclose(raw_best, raw_default)


def test_subsample_changes_the_model_but_still_learns():
    X_static, X_series, y = make_bump_interaction(n=200, seed=0)
    full = HeartwoodClassifier(n_estimators=25, subsample=1.0, random_state=0).fit(
        X_static, X_series, y
    )
    part = HeartwoodClassifier(n_estimators=25, subsample=0.5, random_state=0).fit(
        X_static, X_series, y
    )
    assert not np.array_equal(
        full.predict_proba(X_static, X_series), part.predict_proba(X_static, X_series)
    )
    assert part.train_history_[-1] < part.train_history_[0]


# ------------------------------------------------------------ task varieties


def test_binary_classification_shapes_and_labels():
    X_static, X_series, y = make_bump_interaction(n=120, seed=0)
    labels = np.where(y == 1, "churn", "stay")
    model = HeartwoodClassifier(**FAST).fit(X_static, X_series, labels)

    assert list(model.classes_) == ["churn", "stay"]
    proba = model.predict_proba(X_static, X_series)
    assert proba.shape == (120, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(model.predict(X_static, X_series)) <= {"churn", "stay"}


def test_multiclass_shapes_and_labels(rng):
    n = 180
    X_static = rng.normal(size=(n, 3))
    X_series = rng.normal(size=(n, 1, 40))
    y = np.repeat([10, 20, 30], n // 3)
    X_series[y == 20, 0, 10:20] += 3.0
    X_static[y == 30, 0] += 4.0

    model = HeartwoodClassifier(n_estimators=25, random_state=0).fit(X_static, X_series, y)
    assert list(model.classes_) == [10, 20, 30]
    proba = model.predict_proba(X_static, X_series)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (model.predict(X_static, X_series) == y).mean() > 0.8


def test_regression_returns_one_value_per_row(rng):
    n = 100
    X_static = rng.normal(size=(n, 3))
    y = 2.0 * X_static[:, 0] + rng.normal(scale=0.1, size=n)
    model = HeartwoodRegressor(n_estimators=40, random_state=0).fit(X_static, None, y)
    predictions = model.predict(X_static, None)
    assert predictions.shape == (n,)
    assert np.corrcoef(predictions, y)[0, 1] > 0.9


# --------------------------------------------------------------- input forms


def test_ragged_input_equals_manually_padded_input():
    X_static, X_series, y = make_bump_interaction(n=120, seed=0)
    ragged = [X_series[i, :, : 60 + (i % 40)] for i in range(len(X_series))]

    longest = max(part.shape[1] for part in ragged)
    padded = np.full((len(ragged), X_series.shape[1], longest), np.nan)
    for i, part in enumerate(ragged):
        padded[i, :, : part.shape[1]] = part

    a = HeartwoodClassifier(**FAST).fit(X_static, ragged, y).predict_proba(X_static, ragged)
    b = HeartwoodClassifier(**FAST).fit(X_static, padded, y).predict_proba(X_static, padded)
    assert np.array_equal(a, b)


def test_two_dimensional_series_is_treated_as_one_channel():
    X_static, X_series, y = make_bump_interaction(n=100, seed=0)
    flat = X_series[:, 0, :]
    a = HeartwoodClassifier(**FAST).fit(X_static, flat, y).predict_proba(X_static, flat)
    b = HeartwoodClassifier(**FAST).fit(X_static, X_series, y).predict_proba(X_static, X_series)
    assert np.array_equal(a, b)


def test_series_only_mode():
    _, X_series, y = make_bump_interaction(n=120, seed=0)
    model = HeartwoodClassifier(**FAST).fit(None, X_series, y)
    assert model.n_static_features_ == 0
    assert model.predict(None, X_series).shape == (120,)


def test_static_only_mode():
    X_static, _, y = make_pure_static(n=200, seed=0)
    model = HeartwoodClassifier(n_estimators=40, random_state=0).fit(X_static, None, y)
    assert model.series_shape_ is None
    assert (model.predict(X_static, None) == y).mean() > 0.9


def test_static_only_is_competitive_with_sklearn():
    """Smoke-level: with no series, this should behave like an ordinary GBM."""
    ensemble = pytest.importorskip("sklearn.ensemble")

    X_static, _, y = make_pure_static(n=400, seed=0)
    X_te, _, y_te = make_pure_static(n=1000, seed=11)

    ours = HeartwoodClassifier(n_estimators=100, random_state=0).fit(X_static, None, y)
    theirs = ensemble.HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, random_state=0
    ).fit(X_static, y)

    ours_acc = (ours.predict(X_te, None) == y_te).mean()
    theirs_acc = (theirs.predict(X_te) == y_te).mean()
    assert ours_acc > theirs_acc - 0.05, f"ours={ours_acc:.3f} sklearn={theirs_acc:.3f}"


def test_shorter_series_at_predict_time_is_padded():
    X_static, X_series, y = make_bump_interaction(n=100, seed=0)
    model = HeartwoodClassifier(**FAST).fit(X_static, X_series, y)
    assert model.predict(X_static, X_series[:, :, :70]).shape == (100,)


def test_longer_series_at_predict_time_is_refused():
    X_static, X_series, y = make_bump_interaction(n=100, seed=0)
    model = HeartwoodClassifier(**FAST).fit(X_static, X_series, y)
    longer = np.concatenate([X_series, X_series[:, :, :10]], axis=2)
    with pytest.raises(ValueError, match="exceeds the fitted length"):
        model.predict(X_static, longer)


# -------------------------------------------------------------- inspection


def test_importances_and_dump_are_readable_and_ranked():
    X_static, X_series, y = make_bump_interaction(n=200, seed=0)
    model = HeartwoodClassifier(n_estimators=30, random_state=0).fit(X_static, X_series, y)

    importances = model.feature_importances()
    assert importances and all(gain > 0 for gain in importances.values())
    assert list(importances.values()) == sorted(importances.values(), reverse=True)

    dump = model.dump_splits()
    assert len(dump) >= len(importances)
    assert [gain for _, gain in dump] == sorted((g for _, g in dump), reverse=True)
    assert all("<=" in description for description, _ in dump)
    assert len(model.dump_splits(top=3)) == 3


def test_a_windowed_trend_signal_surfaces_slope_splits():
    """Interpretability with teeth: the reported reason should match the truth."""
    X_static, X_series, y = make_slope_window(n=300, seed=0)
    model = HeartwoodClassifier(n_estimators=40, random_state=0).fit(X_static, X_series, y)
    families = model.feature_importances()

    assert any("slope" in name for name in families), list(families)[:6]
    assert any(name.startswith("static") for name in families), "the gate should show up too"


# -------------------------------------------------------------- validation


def test_using_a_model_before_fitting_it_is_an_error():
    model = HeartwoodClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.feature_importances()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.dump_splits()


def test_shape_mismatches_at_predict_time_are_caught():
    X_static, X_series, y = make_bump_interaction(n=100, seed=0)
    model = HeartwoodClassifier(**FAST).fit(X_static, X_series, y)

    with pytest.raises(ValueError, match="columns"):
        model.predict(X_static[:, :2], X_series)
    with pytest.raises(ValueError, match="channels"):
        model.predict(X_static, np.repeat(X_series, 2, axis=1))
    with pytest.raises(ValueError, match="X_series is required"):
        model.predict(X_static, None)


@pytest.mark.parametrize(
    "params, message",
    [
        (dict(subsample=0.0), "subsample"),
        (dict(subsample=1.5), "subsample"),
        (dict(colsample=0.0), "colsample"),
        (dict(n_estimators=0), "n_estimators"),
        (dict(learning_rate=0.0), "learning_rate"),
        (dict(min_samples_leaf=0), "min_samples_leaf"),
        (dict(interval_stats=("mean", "kurtosis")), "unknown interval statistics"),
        (dict(interval_stats=()), "must not be empty"),
    ],
)
def test_bad_parameters_are_rejected_with_a_useful_message(params, message):
    X_static, X_series, y = make_bump_interaction(n=60, seed=0)
    with pytest.raises(ValueError, match=message):
        HeartwoodClassifier(**params).fit(X_static, X_series, y)


def test_degenerate_inputs_are_rejected():
    X_static, X_series, y = make_bump_interaction(n=60, seed=0)

    with pytest.raises(ValueError, match="at least 2 classes"):
        HeartwoodClassifier(**FAST).fit(X_static, X_series, np.zeros(60))
    with pytest.raises(ValueError, match="y has"):
        HeartwoodClassifier(**FAST).fit(X_static, X_series, y[:10])
    with pytest.raises(ValueError, match="at least one of"):
        HeartwoodClassifier(**FAST).fit(None, None, y)
    with pytest.raises(ValueError, match="must be 1-D"):
        HeartwoodRegressor(**FAST).fit(X_static, X_series, np.zeros((60, 2)))


def test_eval_set_with_unseen_classes_is_refused():
    X_static, X_series, y = make_bump_interaction(n=80, seed=0)
    with pytest.raises(ValueError, match="unseen classes"):
        HeartwoodClassifier(**FAST).fit(
            X_static, X_series, y, eval_set=(X_static, X_series, np.full(80, 9))
        )


def test_get_params_round_trips():
    model = HeartwoodClassifier(n_estimators=33, max_depth=7)
    params = model.get_params()
    assert params["n_estimators"] == 33 and params["max_depth"] == 7
    assert HeartwoodClassifier(**params).get_params() == params
