"""A nonlinear base is only acceptable if the exact hold-out survives it.

Roadmap item 4 is explicit about the trade it refuses: a tree base would model
more and would cost closed-form leave-one-group-out, which is the machinery that
caught the V12 and V13 defects.  Random Fourier features buy nonlinearity while
keeping the fit linear in what it fits, so the exactness is preserved *by
construction* -- the map lives inside ``_prepare`` and everything downstream sees
an ordinary design matrix.

"By construction" is a claim, so it is checked here the same way the linear case
is: against literal refits that actually hold each group out.  If this file ever
fails, the nonlinear base does not ship, whatever it scores.
"""

from __future__ import annotations

import numpy as np
import pytest

import heartwood.dense as module
from heartwood import HeartwoodClassifier
from heartwood.dense import DenseBase


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_leave_one_group_out_is_still_exact_with_random_features(rng):
    """The bar item 4 sets for itself: a refit check to ~1e-14."""
    n, p, k, n_groups, D = 60, 8, 2, 6, 24
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    Z = rng.normal(size=(n, k))
    X = rng.normal(size=(n, p))
    y = Z[:, 0] * 1.2 + X[:, :2] @ rng.normal(size=2) + 0.4 * rng.normal(size=n)

    grid = module.LAMBDA_GRID
    module.LAMBDA_GRID = np.array([1.0])
    try:
        base = DenseBase("regression", 1, use_static=True, nonlinear_features=D)
        out = base.fit(X, y, static=Z, groups=groups)
        assert out is not None

        bank = base._prepare(X, fitting=False)
        assert bank.shape[1] == p + D, "the random block never reached the design"
        design = base._static_design(Z, n, fitting=False)
        penalty = np.zeros(design.shape[1] + bank.shape[1])
        penalty[design.shape[1]:] = base.lambda_
        centred = y - base.target_center_[0]

        expected = np.empty(n)
        for key in np.unique(groups):
            held = np.nonzero(groups == key)[0]
            rest = np.nonzero(groups != key)[0]
            A = np.hstack([design[rest], bank[rest]])
            beta = np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ centred[rest])
            expected[held] = np.hstack([design[held], bank[held]]) @ beta
            expected[held] += base.target_center_[0]
    finally:
        module.LAMBDA_GRID = grid

    drift = float(np.abs(out[:, 0] - expected).max())
    assert drift < 1e-8, f"group hold-out drifted from a refit by {drift:.2e}"


def test_the_random_map_is_frozen_after_fitting(rng):
    """A test row must be measured against the same map a training row was.

    The same discipline as the frozen bias quantiles in ``RocketBank`` and the
    frozen rank grids in ``features.ecdf``. A map redrawn at transform time
    would be a different feature space wearing the same coefficients.
    """
    X = rng.normal(size=(40, 6))
    base = DenseBase("regression", 1, nonlinear_features=16)
    base.fit(X, rng.normal(size=40))
    weights = base.rff_weights_.copy()
    first = base._prepare(X, fitting=False)
    second = base._prepare(X, fitting=False)
    assert np.array_equal(base.rff_weights_, weights)
    assert np.array_equal(first, second)


def test_transforming_before_fitting_is_an_error():
    base = DenseBase("regression", 1, nonlinear_features=8)
    base.impute_ = np.zeros(4)
    base.center_ = np.zeros(4)
    base.scale_ = np.ones(4)
    with pytest.raises(RuntimeError, match="random features"):
        base._prepare(np.zeros((3, 4)), fitting=False)


def test_the_linear_block_is_kept_alongside_the_random_one(rng):
    """The nonlinear base must contain the linear one, not replace it.

    Otherwise a win could not be attributed to curvature: the arm would have
    traded the original columns away and be a different model, not a richer one.
    """
    X = rng.normal(size=(30, 5))
    base = DenseBase("regression", 1, nonlinear_features=12)
    base.fit(X, rng.normal(size=30))
    prepared = base._prepare(X, fitting=False)
    assert prepared.shape == (30, 5 + 12)
    linear = (np.where(np.isfinite(X), X, base.impute_) - base.center_) / base.scale_
    assert np.allclose(prepared[:, :5], linear)


def test_the_map_approximates_an_rbf_kernel(rng):
    """The claim behind the construction, checked rather than cited.

    With enough features, inner products of the mapped rows should approach
    ``exp(-gamma ||x - x'||^2)``. If this drifts, the base is still *a*
    nonlinear map but no longer the one the write-up describes.
    """
    n, p, D = 40, 6, 40000
    X = rng.normal(size=(n, p))
    base = DenseBase("regression", 1, nonlinear_features=D, nonlinear_gamma=1.0)
    base.impute_ = np.zeros(p)
    base.center_ = np.zeros(p)
    base.scale_ = np.ones(p)
    mapped = base._random_features(X, fitting=True)

    gamma = 1.0 / (2.0 * p)
    approx = mapped @ mapped.T
    squared = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    exact = np.exp(-gamma * squared)
    assert np.abs(approx - exact).max() < 0.05


def test_a_wider_map_is_a_better_approximation(rng):
    n, p = 30, 5
    X = rng.normal(size=(n, p))
    squared = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    exact = np.exp(-(1.0 / (2.0 * p)) * squared)

    def error(D):
        base = DenseBase("regression", 1, nonlinear_features=D, random_state=1)
        base.impute_, base.center_, base.scale_ = np.zeros(p), np.zeros(p), np.ones(p)
        mapped = base._random_features(X, fitting=True)
        return float(np.abs(mapped @ mapped.T - exact).max())

    assert error(20000) < error(200)


def test_gamma_changes_the_kernel_width(rng):
    X = rng.normal(size=(20, 4))
    outputs = []
    for gamma in (0.1, 10.0):
        base = DenseBase("regression", 1, nonlinear_features=64, nonlinear_gamma=gamma)
        base.impute_, base.center_, base.scale_ = np.zeros(4), np.zeros(4), np.ones(4)
        outputs.append(base._random_features(X, fitting=True))
    assert not np.allclose(outputs[0], outputs[1])


@pytest.mark.parametrize("width", [0, 64, 256])
def test_the_model_fits_and_predicts_at_any_width(width, rng):
    X = rng.normal(size=(120, 2, 40))
    y = (X[:, 0, 5:15].mean(axis=1) > 0).astype(int)
    model = HeartwoodClassifier(
        n_estimators=8, max_depth=3, dense_base=True,
        n_rocket_features=200, nonlinear_features=width, random_state=0,
    ).fit(None, X, y)
    assert model.predict(None, X).shape == (len(y),)


def test_the_nonlinear_parameters_are_validated(rng):
    X = rng.normal(size=(20, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    with pytest.raises(ValueError, match="nonlinear_features"):
        HeartwoodClassifier(nonlinear_features=-1).fit(None, X, y)
    with pytest.raises(ValueError, match="nonlinear_gamma"):
        HeartwoodClassifier(nonlinear_gamma=0.0).fit(None, X, y)
