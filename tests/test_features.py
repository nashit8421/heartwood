"""Feature extraction, with the NaN edge cases treated as first-class.

Missing data is where this library is most likely to be quietly wrong: a window
that touches a NaN can still produce a finite-looking number, win a split, and
leave every metric plausible.  So the NaN paths get more attention here than the
happy paths.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from conftest import brute_interval_stat, brute_shapelet
from heartwood.features import (
    STAT_NAMES,
    eval_split_feature,
    interval_stat,
    shapelet_features,
)
from heartwood.splits import SplitSpec


@pytest.fixture
def messy_block(rng):
    """A block exercising every awkward row shape at once."""
    X = rng.normal(size=(40, 25))
    X[rng.random(X.shape) < 0.25] = np.nan
    X[0, :] = np.nan  # nothing observed
    X[1, :] = 7.0  # constant
    X[2, :] = np.nan
    X[2, 6] = 1.0  # exactly one observation
    X[3, :] = np.arange(25) * 0.5 + 2.0  # an exact line
    X[4, :2] = [1.0, 3.0]
    X[4, 2:] = np.nan  # exactly two observations
    return X


# ------------------------------------------------------------- interval stats


@pytest.mark.parametrize("stat", STAT_NAMES)
def test_interval_stat_matches_brute_force(stat, messy_block):
    got = interval_stat(messy_block, stat)
    want = np.array([brute_interval_stat(row, stat) for row in messy_block])
    assert np.allclose(got, want, equal_nan=True, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("stat", STAT_NAMES)
def test_interval_stat_returns_nan_when_nothing_is_observed(stat, messy_block):
    assert np.isnan(interval_stat(messy_block, stat)[0])


@pytest.mark.parametrize("stat", STAT_NAMES)
def test_interval_stat_emits_no_warnings(stat, messy_block):
    """All-NaN rows must be handled by masking, not by numpy complaining."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interval_stat(messy_block, stat)
    assert [str(w.message) for w in caught] == []


def test_slope_is_the_ordinary_least_squares_slope(messy_block):
    slopes = interval_stat(messy_block, "slope")
    assert np.isclose(slopes[3], 0.5)  # the exact line
    assert np.isclose(slopes[1], 0.0)  # constant row
    assert np.isnan(slopes[2])  # a single point has no slope


def test_std_is_population_not_sample(rng):
    block = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.isclose(interval_stat(block, "std")[0], np.std(block[0]))


def test_last_and_delta_ignore_trailing_padding():
    block = np.array([[3.0, 5.0, 9.0, np.nan, np.nan]])
    assert np.isclose(interval_stat(block, "last")[0], 9.0)
    assert np.isclose(interval_stat(block, "delta")[0], 6.0)


def test_stats_of_an_empty_window_are_nan():
    empty = np.empty((4, 0))
    for stat in STAT_NAMES:
        assert np.isnan(interval_stat(empty, stat)).all()


def test_unknown_stat_is_rejected():
    with pytest.raises(ValueError, match="unknown statistic"):
        interval_stat(np.zeros((2, 3)), "kurtosis")


# ------------------------------------------------------------------ shapelets


@pytest.mark.parametrize("znorm", [True, False])
def test_shapelet_matches_brute_force(znorm, rng):
    X = rng.normal(size=(20, 40))
    X[rng.random(X.shape) < 0.10] = np.nan
    X[0, :] = np.nan
    X[1, 10:20] = 5.0  # a constant stretch: z-norm degenerates here
    shp = rng.normal(size=7)

    got_dist, got_pos = shapelet_features(X, shp, znorm=znorm)
    want_dist, want_pos = brute_shapelet(X, shp, znorm=znorm)
    assert np.allclose(got_dist, want_dist, equal_nan=True, atol=1e-9)
    assert np.allclose(got_pos, want_pos, equal_nan=True, atol=1e-9)


def test_planted_shapelet_is_found_exactly(rng):
    """A rescaled, offset copy still matches: the distance is z-normalised."""
    X = rng.normal(size=(5, 50))
    template = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    X[2, 12:19] = template * 3.0 + 4.0

    dist, pos = shapelet_features(X, template)
    assert dist[2] < 1e-9
    assert np.isclose(pos[2], 12 / (50 - 7))


def test_windows_touching_nan_never_win(rng):
    """The core silent-corruption guard: a NaN window must be excluded, not zeroed."""
    template = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    X = np.full((3, 30), np.nan)
    X[0, 5:12] = template  # the only usable window in row 0
    X[1, :] = np.nan  # nothing usable at all

    dist, pos = shapelet_features(X, template)
    assert dist[0] < 1e-9 and np.isclose(pos[0], 5 / (30 - 7))
    assert np.isnan(dist[1]) and np.isnan(pos[1])


def test_nan_padding_does_not_change_the_answer(rng):
    """Right-padding a short series must not invent a better match."""
    X = rng.normal(size=(6, 30))
    shp = X[0, 4:12].copy()
    dist_short, pos_short = shapelet_features(X, shp)

    padded = np.full((6, 45), np.nan)
    padded[:, :30] = X
    dist_pad, pos_pad = shapelet_features(padded, shp)

    assert np.allclose(dist_short, dist_pad, atol=1e-9)
    # positions are normalised by a different window count, so compare indices
    assert np.allclose(pos_short * (30 - 8), pos_pad * (45 - 8), atol=1e-9)


@pytest.mark.parametrize("chunk_bytes", [1, 64, 1 << 20])
def test_chunking_does_not_change_results(chunk_bytes, rng):
    X = rng.normal(size=(17, 40))
    X[rng.random(X.shape) < 0.1] = np.nan
    shp = rng.normal(size=6)

    ref = shapelet_features(X, shp, chunk_bytes=1 << 26)
    got = shapelet_features(X, shp, chunk_bytes=chunk_bytes)
    assert np.allclose(got[0], ref[0], equal_nan=True, atol=1e-12)
    assert np.allclose(got[1], ref[1], equal_nan=True, atol=1e-12)


def test_shapelet_longer_than_series_is_missing_not_an_error(rng):
    dist, pos = shapelet_features(rng.normal(size=(4, 5)), rng.normal(size=9))
    assert np.isnan(dist).all() and np.isnan(pos).all()


def test_non_finite_shapelet_is_missing_not_an_error(rng):
    shp = np.array([1.0, np.nan, 2.0])
    dist, pos = shapelet_features(rng.normal(size=(4, 10)), shp)
    assert np.isnan(dist).all() and np.isnan(pos).all()


def test_single_window_series_has_position_zero(rng):
    X = rng.normal(size=(3, 6))
    dist, pos = shapelet_features(X, X[0, :6].copy())
    assert np.allclose(pos, 0.0)


def test_distances_are_never_negative(rng):
    """The algebraic expansion can round below zero; it must be clipped."""
    X = rng.normal(size=(30, 40)) * 1e-3
    for shp in (X[0, :9].copy(), rng.normal(size=9)):
        dist, _ = shapelet_features(X, shp)
        assert np.nanmin(dist) >= 0.0


# ------------------------------------------------------------ dispatch helper


def test_eval_split_feature_dispatches_to_the_right_extractor(rng):
    X_static = rng.normal(size=(12, 3))
    X_series = rng.normal(size=(12, 2, 30))
    rows = np.array([0, 3, 7, 11])

    spec = SplitSpec(kind="static", col=2)
    assert np.allclose(eval_split_feature(spec, X_static, X_series, rows), X_static[rows, 2])

    spec = SplitSpec(kind="interval", channel=1, start=4, end=17, stat="slope")
    assert np.allclose(
        eval_split_feature(spec, X_static, X_series, rows),
        interval_stat(X_series[rows, 1, 4:17], "slope"),
    )

    shp = X_series[0, 0, 3:11].copy()
    want = shapelet_features(X_series[rows, 0, :], shp)
    for kind, which in (("shapelet_dist", 0), ("shapelet_pos", 1)):
        spec = SplitSpec(kind=kind, channel=0, shapelet=shp, znorm=True)
        assert np.allclose(eval_split_feature(spec, X_static, X_series, rows), want[which])


def test_eval_split_feature_rejects_unknown_kinds(rng):
    with pytest.raises(ValueError, match="unknown split kind"):
        eval_split_feature(
            SplitSpec(kind="wavelet"), np.zeros((2, 2)), None, np.array([0, 1])
        )
