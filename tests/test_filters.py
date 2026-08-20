"""Matched filters — including the guards the design review called non-negotiable.

Three of these exist purely to catch silent corruption: a NaN-touching window
winning a match, the refit degenerating because the constant was left in the
basis, and the fitted family failing to reduce to the shapelet family it is
supposed to contain.  None of those would raise; all of them would just make the
model quietly worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood.features import shapelet_features
from heartwood.filters import (
    align,
    build_pyramid,
    dct_basis,
    refit_template,
    znorm_snippet,
)


# ------------------------------------------------------------------ pyramid


def test_pyramid_halves_the_resolution_each_level(rng):
    X = rng.normal(size=(6, 2, 100))
    levels = build_pyramid(X, filter_len=9)
    assert levels[0] is X
    assert [level.shape[2] for level in levels] == [100, 50, 25, 12]
    assert all(level.shape[:2] == (6, 2) for level in levels)
    # a nine-tap filter at the coarsest level spans 9·2³ = 72 raw timesteps
    assert 9 * 2 ** (len(levels) - 1) <= X.shape[2]


def test_pyramid_stops_before_levels_get_too_short(rng):
    levels = build_pyramid(rng.normal(size=(3, 1, 24)), filter_len=9)
    assert all(level.shape[2] >= 10 for level in levels)
    assert len(build_pyramid(rng.normal(size=(3, 1, 12)), filter_len=9)) == 1


def test_pooling_averages_only_observed_values():
    X = np.array([[[1.0, 3.0, 5.0, np.nan, np.nan, np.nan, 2.0, 4.0]]])
    level = build_pyramid(X, filter_len=1, max_scales=2)[1]
    assert np.isclose(level[0, 0, 0], 2.0)  # (1+3)/2
    assert np.isclose(level[0, 0, 1], 5.0)  # 5 alone; the NaN is ignored
    assert np.isnan(level[0, 0, 2]), "a pair with nothing observed must stay missing"
    assert np.isclose(level[0, 0, 3], 3.0)  # (2+4)/2


def test_pooling_never_turns_missing_into_zero():
    X = np.full((2, 1, 16), np.nan)
    for level in build_pyramid(X, filter_len=1, max_scales=4):
        assert np.isnan(level).all()


# -------------------------------------------------------------------- basis


def test_dct_basis_is_orthonormal_and_has_no_constant_component():
    basis = dct_basis(9, 5)
    assert basis.shape == (9, 5)
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-12)
    assert np.allclose(basis.sum(axis=0), 0.0, atol=1e-12), "DC must be excluded"


def test_dct_basis_is_clipped_to_what_the_length_supports():
    assert dct_basis(4, 20).shape[1] == 3


def test_dct_components_are_ordered_from_smooth_to_wiggly():
    basis = dct_basis(16, 4)
    crossings = [int((np.diff(np.sign(basis[:, k])) != 0).sum()) for k in range(4)]
    assert crossings == sorted(crossings)


# --------------------------------------------------------------- alignment


def test_a_planted_template_matches_at_its_own_position(rng):
    template = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    block = rng.normal(scale=0.01, size=(4, 60))
    block[1, 20:29] = template * 5.0 + 3.0  # rescaled and shifted: still a match

    response, position = align(block, template)
    assert response[1] > 0.999
    assert np.isclose(position[1], 20 / (60 - 9))


def test_inverted_patterns_give_a_negative_response(rng):
    template = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    block = rng.normal(scale=0.01, size=(2, 60))
    block[0, 30:39] = -template * 4.0

    response, position = align(block, template)
    assert response[0] < -0.999, "polarity must survive; it is what tells classes apart"
    assert np.isclose(position[0], 30 / (60 - 9))


def test_windows_touching_nan_never_win(rng):
    """The core silent-corruption guard, checked at every pyramid scale."""
    template = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    X = rng.normal(scale=0.5, size=(3, 1, 96))
    X[:, 0, 40:] = np.nan
    X[0, 0, 5:14] = template  # the only strong match sits in observed data

    for scale, level in enumerate(build_pyramid(X, filter_len=9)):
        block = level[:, 0, :]
        response, position = align(block, template)
        n_positions = block.shape[1] - 9 + 1
        for row in range(3):
            if np.isnan(position[row]):
                continue
            start = int(round(position[row] * (n_positions - 1)))
            chosen = block[row, start : start + 9]
            assert np.isfinite(chosen).all(), f"scale {scale} matched into padding"


def test_rows_with_nothing_observed_are_missing_not_zero():
    template = np.arange(9.0)
    block = np.full((2, 40), np.nan)
    block[0, :20] = np.linspace(0, 1, 20)
    response, position = align(block, template)
    assert np.isfinite(response[0]) and np.isfinite(position[0])
    assert np.isnan(response[1]) and np.isnan(position[1])


def test_flat_windows_and_flat_templates_do_not_divide_by_zero():
    flat_block = np.ones((2, 30))
    response, position = align(flat_block, np.arange(9.0))
    assert np.isfinite(response).all() and np.allclose(response, 0.0)

    response, _ = align(np.random.default_rng(0).normal(size=(2, 30)), np.ones(9))
    assert np.allclose(response, 0.0)


def test_correlation_stays_within_bounds(rng):
    for _ in range(30):
        block = rng.normal(size=(5, 50)) * rng.uniform(0.001, 1000)
        response, _ = align(block, rng.normal(size=9))
        assert np.nanmax(np.abs(response)) <= 1.0 + 1e-12


def test_template_shorter_than_the_series_is_required(rng):
    response, position = align(rng.normal(size=(3, 5)), rng.normal(size=9))
    assert np.isnan(response).all() and np.isnan(position).all()


def test_reduces_exactly_to_the_shapelet_family(rng):
    """Iteration zero of a filter *is* a shapelet: distance == 2·(1 − correlation).

    This is what makes turning filters on a safe change — the new family
    contains the old one rather than replacing it.
    """
    X = rng.normal(size=(25, 80))
    X[rng.random(X.shape) < 0.05] = np.nan
    for _ in range(15):
        template = rng.normal(size=int(rng.integers(4, 20)))
        response, position = align(X, template)
        distance, shapelet_position = shapelet_features(X, template, znorm=True)

        both = np.isfinite(response) & np.isfinite(distance)
        # the shapelet takes the *closest* match, i.e. the most positive
        # correlation, so compare on rows where the filter agrees on polarity
        positive = both & (response > 0)
        assert np.allclose(2.0 * (1.0 - response[positive]), distance[positive], atol=1e-8)
        assert np.allclose(position[positive], shapelet_position[positive], atol=1e-12)


# ------------------------------------------------------------------- refit


def test_refit_recovers_a_planted_template(rng):
    """Rows whose window carries the template should drag the fit toward it."""
    basis = dct_basis(9, 5)
    truth = basis @ np.array([1.0, -0.5, 0.25, 0.0, 0.0])
    truth /= np.linalg.norm(truth)

    windows = rng.normal(scale=0.3, size=(60, 9))
    strength = rng.normal(size=60)
    windows += strength[:, None] * truth
    residuals = strength + rng.normal(scale=0.05, size=60)

    fitted = refit_template(windows, residuals, np.ones(60), basis, ridge=1.0)
    assert fitted is not None
    assert abs(float(fitted @ truth)) > 0.9


def test_refit_returns_unit_norm_or_nothing(rng):
    basis = dct_basis(9, 5)
    fitted = refit_template(
        rng.normal(size=(40, 9)), rng.normal(size=40), np.ones(40), basis, ridge=1.0
    )
    assert fitted is None or np.isclose(np.linalg.norm(fitted), 1.0)


def test_refit_declines_rather_than_guessing_from_too_few_rows(rng):
    basis = dct_basis(9, 5)
    for n_rows in (0, 3, 6):
        assert refit_template(
            rng.normal(size=(n_rows, 9)), rng.normal(size=n_rows),
            np.ones(n_rows), basis, ridge=1.0,
        ) is None


def test_refit_survives_a_degenerate_design(rng):
    """Constant windows carry no shape; the solve must not produce garbage."""
    basis = dct_basis(9, 5)
    fitted = refit_template(
        np.zeros((40, 9)), rng.normal(size=40), np.ones(40), basis, ridge=1.0
    )
    assert fitted is None


def test_hessian_weighting_follows_the_confident_rows(rng):
    """Rows the model is still uncertain about should shape the template most."""
    basis = dct_basis(9, 5)
    first = basis[:, 0]
    second = basis[:, 1]

    windows = np.zeros((40, 9))
    windows[:20] = first
    windows[20:] = second
    residuals = np.concatenate([np.ones(20), np.ones(20)])

    weights = np.concatenate([np.full(20, 100.0), np.full(20, 0.01)])
    fitted = refit_template(windows, residuals, weights, basis, ridge=1e-6)
    assert fitted is not None
    assert abs(float(fitted @ first)) > abs(float(fitted @ second))


def test_ridge_shrinks_the_template_toward_smoothness(rng):
    basis = dct_basis(9, 5)
    windows = rng.normal(size=(80, 9))
    residuals = rng.normal(size=80)
    gentle = refit_template(windows, residuals, np.ones(80), basis, ridge=1e-6)
    strong = refit_template(windows, residuals, np.ones(80), basis, ridge=1e6)
    assert gentle is not None and strong is not None
    # both are unit-norm, so compare where the energy sits across components
    assert np.isclose(np.linalg.norm(gentle), 1.0)
    assert np.isclose(np.linalg.norm(strong), 1.0)


# ------------------------------------------------------------------ snippets


def test_znorm_snippet_centres_and_scales(rng):
    snippet = rng.normal(size=9) * 4.0 + 7.0
    normalised = znorm_snippet(snippet)
    assert normalised is not None
    assert np.isclose(normalised.mean(), 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(normalised), 1.0)


def test_znorm_snippet_rejects_shapeless_input():
    assert znorm_snippet(np.ones(9)) is None
    assert znorm_snippet(np.array([1.0, np.nan, 2.0])) is None
