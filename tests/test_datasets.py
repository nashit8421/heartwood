"""The synthetic scenarios must contain the signal they claim, and only that.

Two failure modes matter here and both actually happened during development:
a generator with *no* recoverable signal (nothing can score above chance, so the
benchmark is meaningless), and a generator whose signal survives aggregation
(the baseline solves it, so the benchmark flatters us for the wrong reason).
Each scenario is therefore pinned from both sides: an oracle must succeed, and
global aggregates must fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood.datasets import (
    make_bump_interaction,
    make_pure_static,
    make_shape_amplitude_regression,
    make_slope_window,
    make_static_plus_noise_series,
    make_timing_task,
)

CLASSIFIERS = [
    make_bump_interaction,
    make_timing_task,
    make_slope_window,
    make_static_plus_noise_series,
]
ALL_GENERATORS = CLASSIFIERS + [make_shape_amplitude_regression, make_pure_static]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC; 0.5 means the score carries no information about the label."""
    keep = np.isfinite(score)
    score, label = score[keep], label[keep]
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def matched_filter(series: np.ndarray, template: np.ndarray, signed: bool = True):
    """Peak response to ``template`` and where it occurs — the oracle detector.

    ``signed`` locates the best *same-polarity* match, which is what the
    library's shapelet distance does (it is minimised at correlation +1).  With
    ``signed=False`` the strongest match of either polarity wins, which cannot
    tell an up-then-down transient from a down-then-up one.
    """
    template = template - template.mean()
    template = template / np.linalg.norm(template)
    X = series[:, 0, :]
    length = len(template)
    n_positions = X.shape[1] - length + 1
    response = np.zeros((len(X), n_positions))
    for j in range(length):
        response += X[:, j : j + n_positions] * template[j]
    peak = np.argmax(response if signed else np.abs(response), axis=1)
    return response[np.arange(len(X)), peak], peak


def global_aggregates(series: np.ndarray) -> dict[str, np.ndarray]:
    X = series[:, 0, :]
    t = np.arange(X.shape[1], dtype=float)
    tc = t - t.mean()
    return {
        "mean": X.mean(1), "std": X.std(1), "min": X.min(1), "max": X.max(1),
        "median": np.median(X, axis=1), "slope": (X * tc).sum(1) / (tc * tc).sum(),
        "mean_abs_change": np.abs(np.diff(X, axis=1)).mean(1),
        "first": X[:, 0], "last": X[:, -1], "delta": X[:, -1] - X[:, 0],
    }


# ------------------------------------------------------------------- basics


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_shapes_are_consistent(generator):
    X_static, X_series, y = generator(n=64, seed=0)
    assert X_static.shape[0] == 64 and X_static.ndim == 2
    assert len(y) == 64
    if X_series is None:
        assert generator is make_pure_static
    else:
        assert X_series.ndim == 3 and X_series.shape[0] == 64
        assert np.isfinite(X_series).all()
    assert np.isfinite(X_static).all()


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_seeds_are_reproducible_and_distinct(generator):
    a = generator(n=40, seed=3)
    b = generator(n=40, seed=3)
    c = generator(n=40, seed=4)
    for left, right in zip(a, b):
        if left is None:
            assert right is None
        else:
            assert np.array_equal(left, right)
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("generator", CLASSIFIERS + [make_pure_static])
def test_classes_are_reasonably_balanced(generator):
    _, _, y = generator(n=800, seed=0)
    rate = y.mean()
    assert 0.25 < rate < 0.75, f"{generator.__name__} is unbalanced at {rate:.2f}"


def test_regression_target_has_spread():
    _, _, y = make_shape_amplitude_regression(n=400, seed=0)
    assert y.std() > 0.5 and np.isfinite(y).all()


# ------------------------------------- the signal is there (oracles succeed)


def doublet_template(T=100):
    """The same zero-area transient the generators plant."""
    sigma = max(1.5, T / 50.0)
    sep = 2.0 * sigma
    length = int(round(4 * sigma + 2 * sep))
    u = np.arange(length, dtype=float) - (length - 1) / 2.0
    return np.exp(-0.5 * ((u + sep) / sigma) ** 2) - np.exp(-0.5 * ((u - sep) / sigma) ** 2)


def test_bump_order_is_recoverable_and_xors_with_the_flag():
    X_static, X_series, y = make_bump_interaction(n=600, seed=0)
    up_first = (y.astype(bool) ^ (X_static[:, 0] == 1)).astype(int)

    # Locating both transients recovers the order exactly.
    template = doublet_template()
    _, up_at = matched_filter(X_series, template)
    _, down_at = matched_filter(X_series, -template)
    assert ((up_at < down_at).astype(int) == up_first).mean() > 0.95

    # Knowing when the up-then-down shape occurred is nearly enough on its own —
    # which is the point: the recoverable quantity is a *position*, and it takes
    # shape matching to read one off.
    assert auc(up_at, up_first) < 0.10

    # But the label is an XOR, so neither modality predicts it alone.
    assert 0.42 < auc(up_at, y) < 0.58
    assert 0.42 < auc(X_static[:, 0], y) < 0.58


def test_timing_label_is_exactly_position_versus_deadline():
    X_static, X_series, y = make_timing_task(n=600, seed=0)
    template = doublet_template()
    _, position = matched_filter(X_series, template)

    centre = position + (len(template) - 1) / 2.0
    predicted = (centre < X_static[:, 0]).astype(int)
    assert (predicted == y).mean() > 0.95


def test_slope_label_lives_in_its_window_and_not_in_the_global_slope():
    X_static, X_series, y = make_slope_window(n=600, T=120, seed=0)
    gate = X_static[:, 0]
    target_sign = (y.astype(bool) ^ (gate > 0)).astype(int)

    X = X_series[:, 0, :]
    window = X[:, int(0.38 * 120) : int(0.58 * 120)]
    t = np.arange(window.shape[1], dtype=float)
    tc = t - t.mean()
    local_slope = (window * tc).sum(1) / (tc * tc).sum()

    assert auc(local_slope, target_sign) > 0.95, "the window should carry the sign"
    assert auc(global_aggregates(X_series)["slope"], target_sign) < 0.80, (
        "the global slope should be diluted by the distractor segments"
    )


def test_regression_amplitude_is_recoverable_only_from_the_right_window():
    X_static, X_series, y = make_shape_amplitude_regression(n=600, seed=0)
    coefficient = X_static[:, 0]
    amplitude = y / coefficient

    X = X_series[:, 0, :]
    local_max = X[:, int(0.50 * 100) : int(0.72 * 100)].max(1)
    global_max = X.max(1)

    assert np.corrcoef(local_max, amplitude)[0, 1] > 0.75
    assert np.corrcoef(global_max, amplitude)[0, 1] < 0.4, (
        "the nuisance transient should dominate the global maximum"
    )


def test_static_control_signal_is_static_only():
    X_static, X_series, y = make_static_plus_noise_series(n=600, seed=0)
    assert auc(X_static[:, 0], y) > 0.7, "static columns should carry the signal"
    for name, values in global_aggregates(X_series).items():
        assert abs(auc(values, y) - 0.5) < 0.10, f"the noise series leaked through {name}"


# -------------------------- the signal is hidden from aggregation (by design)


@pytest.mark.parametrize("statistic", list(global_aggregates(np.zeros((2, 1, 8)))))
def test_no_global_aggregate_predicts_the_bump_order(statistic):
    """The property that makes this benchmark meaningful, pinned as a test.

    Both classes contain the same two transients and differ only in their order,
    so every global statistic sees the same distribution either way.  If this
    ever starts failing, the scenario has stopped being a fair test of temporal
    modelling and the headline result would be measuring something else.
    """
    X_static, X_series, y = make_bump_interaction(n=2000, seed=0)
    up_first = (y.astype(bool) ^ (X_static[:, 0] == 1)).astype(int)
    score = global_aggregates(X_series)[statistic]
    assert abs(auc(score, up_first) - 0.5) < 0.10, (
        f"global {statistic} separates the classes at auc="
        f"{auc(score, up_first):.3f}; aggregation is supposed to be blind here"
    )


@pytest.mark.parametrize("statistic", list(global_aggregates(np.zeros((2, 1, 8)))))
def test_no_global_aggregate_predicts_the_timing_label(statistic):
    """Every series holds the same transient; only its position varies."""
    _, X_series, y = make_timing_task(n=2000, seed=0)
    score = global_aggregates(X_series)[statistic]
    assert abs(auc(score, y) - 0.5) < 0.10, f"global {statistic} leaked the timing label"


def test_pure_static_scenario_has_no_series_at_all():
    X_static, X_series, y = make_pure_static(n=100, seed=0)
    assert X_series is None
    assert X_static.shape == (100, 10)


def test_static_blocks_carry_noise_distractors():
    """A scenario with only informative columns would be unrealistically easy."""
    for generator in CLASSIFIERS[:3] + [make_shape_amplitude_regression]:
        X_static, _, y = generator(n=500, seed=0)
        assert X_static.shape[1] >= 5
        uninformative = [
            column for column in range(1, X_static.shape[1])
            if abs(auc(X_static[:, column], (y > np.median(y)).astype(int)) - 0.5) < 0.08
        ]
        assert len(uninformative) >= 3, f"{generator.__name__} lacks distractor columns"
