"""Losses: gradients and hessians must be the real derivatives of the objective.

Everything the booster does rests on g and h being right.  A sign error or a
factor of two here would still *train* — just badly — so these are checked
against finite differences of the objective rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood.losses import Logistic, Softmax, SquaredError, sigmoid, softmax

FD_EPS = 1e-5


def objective_squared(y, raw):
    return 0.5 * (raw[:, 0] - y) ** 2


def objective_logistic(y, raw):
    p = np.clip(sigmoid(raw[:, 0]), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def objective_softmax(y, raw):
    p = np.clip(softmax(raw), 1e-15, 1.0)
    return -np.log(p[np.arange(len(y)), y.astype(int)])


def finite_diff(objective, y, raw, k):
    """Numeric ∂/∂raw[:, k] and ∂²/∂raw[:, k]² of a per-sample objective."""
    step = np.zeros_like(raw)
    step[:, k] = FD_EPS
    f_plus = objective(y, raw + step)
    f_minus = objective(y, raw - step)
    f_zero = objective(y, raw)
    grad = (f_plus - f_minus) / (2 * FD_EPS)
    hess = (f_plus - 2 * f_zero + f_minus) / (FD_EPS**2)
    return grad, hess


def build(kind, rng, n=40):
    """Return (loss, y, raw, objective) for one task."""
    if kind == "squared":
        y = rng.normal(size=n)
        return SquaredError(), y, rng.normal(size=(n, 1)), objective_squared
    if kind == "logistic":
        y = rng.integers(0, 2, size=n).astype(float)
        return Logistic(), y, rng.normal(scale=1.5, size=(n, 1)), objective_logistic
    y = rng.integers(0, 4, size=n).astype(float)
    return Softmax(4), y, rng.normal(scale=1.5, size=(n, 4)), objective_softmax


@pytest.mark.parametrize("kind", ["squared", "logistic", "softmax"])
def test_grad_matches_finite_difference(kind, rng):
    loss, y, raw, objective = build(kind, rng)
    g, _ = loss.grad_hess(y, raw)
    for k in range(raw.shape[1]):
        expected, _ = finite_diff(objective, y, raw, k)
        assert np.allclose(g[:, k], expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("kind", ["squared", "logistic", "softmax"])
def test_hess_matches_finite_difference(kind, rng):
    loss, y, raw, objective = build(kind, rng)
    _, h = loss.grad_hess(y, raw)
    for k in range(raw.shape[1]):
        _, expected = finite_diff(objective, y, raw, k)
        # Second differences are noisy; a loose tolerance still catches a wrong
        # formula, which is what this is for.
        assert np.allclose(h[:, k], expected, rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize("kind", ["squared", "logistic", "softmax"])
def test_shapes_are_n_by_k(kind, rng):
    loss, y, raw, _ = build(kind, rng)
    g, h = loss.grad_hess(y, raw)
    assert g.shape == raw.shape == h.shape
    assert loss.n_outputs(y) == raw.shape[1]
    assert loss.init_score(y).shape == (raw.shape[1],)


@pytest.mark.parametrize("kind", ["squared", "logistic", "softmax"])
@pytest.mark.parametrize("extreme", [-50.0, 50.0])
def test_stable_at_extreme_raw(kind, rng, extreme):
    """Confident-but-wrong predictions must not produce inf/NaN anywhere."""
    loss, y, raw, _ = build(kind, rng)
    raw = np.full_like(raw, extreme)
    g, h = loss.grad_hess(y, raw)
    assert np.isfinite(g).all()
    assert np.isfinite(h).all()
    assert (h > 0).all(), "hessian must stay strictly positive for the leaf solve"
    assert np.isfinite(loss.eval_metric(y, raw))
    assert np.isfinite(loss.transform(raw)).all()


def test_softmax_transform_is_a_distribution(rng):
    raw = rng.normal(scale=30, size=(25, 5))
    p = softmax(raw)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_sigmoid_matches_definition_in_the_stable_range():
    z = np.linspace(-30, 30, 101)
    assert np.allclose(sigmoid(z), 1.0 / (1.0 + np.exp(-z)))
    assert sigmoid(np.array([-800.0, 800.0])).tolist() == [0.0, 1.0]


def test_init_score_predicts_the_base_rate():
    y = np.array([1.0] * 30 + [0.0] * 70)
    assert np.isclose(sigmoid(Logistic().init_score(y))[0], 0.3)
    assert np.isclose(SquaredError().init_score(np.arange(5.0))[0], 2.0)

    y3 = np.array([0.0] * 20 + [1.0] * 30 + [2.0] * 50)
    priors = softmax(Softmax(3).init_score(y3)[None, :])[0]
    assert np.allclose(priors, [0.2, 0.3, 0.5])


def test_init_score_handles_degenerate_labels():
    """All-one-class input must not produce infinite logits."""
    assert np.isfinite(Logistic().init_score(np.zeros(10))).all()
    assert np.isfinite(Softmax(3).init_score(np.zeros(10))).all()


def test_eval_metrics_match_their_definitions(rng):
    y = rng.normal(size=30)
    raw = rng.normal(size=(30, 1))
    assert np.isclose(
        SquaredError().eval_metric(y, raw), np.sqrt(np.mean((raw[:, 0] - y) ** 2))
    )

    yb = rng.integers(0, 2, size=30).astype(float)
    rawb = rng.normal(size=(30, 1))
    p = sigmoid(rawb[:, 0])
    assert np.isclose(
        Logistic().eval_metric(yb, rawb),
        -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)),
    )


def test_metric_improves_as_predictions_improve(rng):
    """A metric that does not order predictions correctly would break early stopping."""
    y = rng.integers(0, 2, size=200).astype(float)
    good = np.where(y[:, None] > 0, 2.0, -2.0)
    bad = -good
    assert Logistic().eval_metric(y, good) < Logistic().eval_metric(y, bad)


def test_softmax_rejects_binary():
    with pytest.raises(ValueError):
        Softmax(2)
