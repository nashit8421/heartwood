"""The benchmark harness itself.

Every headline number rests on this code, so it gets the same treatment as the
library: metrics checked against scikit-learn's implementations, feature
transforms checked against hand-computed values, and one end-to-end cell run so
the runner cannot silently stop producing results.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.baselines import (
    aggregate_features,
    build_design_matrix,
    flatten_series,
    make_baseline_model,
    windowed_aggregate_features,
)
from benchmarks.run_benchmarks import (
    classification_metrics,
    regression_metrics,
    roc_auc,
    run_cell,
    summarise,
)
from benchmarks.scenarios import DEFAULT_ORDER, SCENARIOS


# ------------------------------------------------------------------ metrics


def test_roc_auc_matches_sklearn(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    for _ in range(20):
        n = int(rng.integers(20, 200))
        y = rng.integers(0, 2, size=n)
        if len(np.unique(y)) < 2:
            continue
        # rounded scores deliberately create ties, which naive rank AUC gets wrong
        score = rng.normal(size=n).round(1)
        assert np.isclose(roc_auc(y, score), metrics.roc_auc_score(y, score), atol=1e-12)


def test_roc_auc_is_undefined_for_one_class():
    assert np.isnan(roc_auc(np.ones(10, dtype=int), np.arange(10.0)))


def test_classification_metrics_match_sklearn(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    for _ in range(20):
        n = int(rng.integers(30, 200))
        y = rng.integers(0, 2, size=n)
        pred = rng.integers(0, 2, size=n)
        score = rng.normal(size=n)
        got = classification_metrics(y, pred, score)
        assert np.isclose(got["accuracy"], metrics.accuracy_score(y, pred))
        assert np.isclose(got["precision"], metrics.precision_score(y, pred, zero_division=0))
        assert np.isclose(got["recall"], metrics.recall_score(y, pred, zero_division=0))
        assert np.isclose(got["f1"], metrics.f1_score(y, pred, zero_division=0))


def test_regression_metrics_match_sklearn(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.5, size=200)
    got = regression_metrics(y, pred)
    assert np.isclose(got["rmse"], np.sqrt(metrics.mean_squared_error(y, pred)))
    assert np.isclose(got["mae"], metrics.mean_absolute_error(y, pred))
    assert np.isclose(got["r2"], metrics.r2_score(y, pred))


def test_perfect_and_useless_predictions_score_as_expected():
    y = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                       "roc_auc": 1.0}
    inverted = classification_metrics(y, 1 - y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert inverted["accuracy"] == 0.0 and inverted["roc_auc"] == 0.0


# --------------------------------------------------------------- transforms


def test_aggregate_features_are_the_ten_classic_summaries(rng):
    X = rng.normal(size=(15, 1, 30))
    got = aggregate_features(X)
    assert got.shape == (15, 10)

    series = X[:, 0, :]
    assert np.allclose(got[:, 0], series.mean(1))
    assert np.allclose(got[:, 1], series.std(1))
    assert np.allclose(got[:, 2], series.min(1))
    assert np.allclose(got[:, 3], series.max(1))
    assert np.allclose(got[:, 5], np.median(series, axis=1))
    assert np.allclose(got[:, 7], series[:, 0])
    assert np.allclose(got[:, 8], series[:, -1])
    assert np.allclose(got[:, 9], series[:, -1] - series[:, 0])


def test_aggregate_slope_is_the_least_squares_slope():
    ramp = (np.arange(40, dtype=float) * 0.25 + 3.0)[None, None, :]
    assert np.isclose(aggregate_features(ramp)[0, 4], 0.25)


def test_one_window_aggregation_equals_global_aggregation(rng):
    """A useful invariant: the windowed transform generalises the global one."""
    X = rng.normal(size=(12, 2, 24))
    assert np.allclose(windowed_aggregate_features(X, n_windows=1), aggregate_features(X))


def test_more_windows_means_more_columns(rng):
    X = rng.normal(size=(12, 2, 32))
    widths = [windowed_aggregate_features(X, k).shape[1] for k in (1, 4, 8)]
    assert widths == [20, 80, 160]


def test_flatten_keeps_every_timestep(rng):
    X = rng.normal(size=(7, 3, 11))
    flat = flatten_series(X)
    assert flat.shape == (7, 33)
    assert np.allclose(flat[4], X[4].ravel())


def test_design_matrix_puts_static_columns_first(rng):
    X_static = rng.normal(size=(9, 4))
    X_series = rng.normal(size=(9, 1, 20))
    design = build_design_matrix("agg", X_static, X_series)
    assert design.shape == (9, 14)
    assert np.allclose(design[:, :4], X_static)
    assert np.allclose(build_design_matrix("static_only", X_static, X_series), X_static)
    assert np.allclose(build_design_matrix("agg", X_static, None), X_static)


# ------------------------------------------------------------------ wiring


def test_every_scenario_is_registered_and_generates():
    assert set(DEFAULT_ORDER) == set(SCENARIOS)
    for key, scenario in SCENARIOS.items():
        X_static, X_series, y = scenario.generator(n=40, seed=0)
        assert len(y) == 40 and X_static.shape[0] == 40
        assert scenario.task in ("binary", "regression")
        assert scenario.question and scenario.why_aggregation_fails


def test_baseline_model_is_available():
    model, backend = make_baseline_model("binary", 10, 3, 0.1, 0)
    assert backend in ("xgboost", "sklearn-histgb")
    assert hasattr(model, "fit") and hasattr(model, "predict")


@pytest.mark.parametrize("key", ["timing", "amp_regression"])
def test_one_benchmark_cell_runs_end_to_end(key):
    config = {
        "test_size": 200, "rounds": 5, "depth": 2, "learning_rate": 0.3,
        "representations": ["agg", "wagg4"], "heartwood_variants": {"heartwood": {}},
    }
    results = run_cell((key, 60, 0, config))

    assert [r.model for r in results] == ["heartwood", "agg", "wagg4"]
    for row in results:
        assert row.scenario == key and row.n_train == 60
        assert row.fit_seconds > 0
        expected = (
            {"rmse", "mae", "r2"}
            if row.task == "regression"
            else {"accuracy", "precision", "recall", "f1", "roc_auc"}
        )
        assert set(row.metrics) == expected
        assert all(np.isfinite(v) for v in row.metrics.values())


def test_summarise_averages_over_seeds():
    config = {"test_size": 150, "rounds": 3, "depth": 2, "learning_rate": 0.3,
              "representations": ["agg"], "heartwood_variants": {"heartwood": {}}}
    results = run_cell(("timing", 60, 0, config)) + run_cell(("timing", 60, 1, config))
    stats = summarise(results, "accuracy")
    assert stats[("timing", "heartwood", 60)]["n"] == 2
    assert 0.0 <= stats[("timing", "agg", 60)]["mean"] <= 1.0
