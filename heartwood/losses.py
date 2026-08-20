"""Second-order losses.

Each loss exposes the four things the booster needs: how many outputs to fit,
where to start (``init_score``), the gradient/hessian pair at the current raw
prediction, and an evaluation metric (lower is always better).

Raw predictions are always ``(n, K)`` with ``K == 1`` for regression and binary
classification, so the boosting loop never special-cases the task.
"""

from __future__ import annotations

import numpy as np

_EPS_H = 1e-16
_EPS_P = 1e-15


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-subtracted for stability."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Loss:
    """Interface shared by all losses."""

    metric_name = ""

    def n_outputs(self, y: np.ndarray) -> int:
        raise NotImplementedError

    def init_score(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_hess(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval_metric(self, y: np.ndarray, raw: np.ndarray) -> float:
        raise NotImplementedError


class SquaredError(Loss):
    """l(y, r) = ½ (r − y)² — plain least squares regression."""

    metric_name = "rmse"

    def n_outputs(self, y):
        return 1

    def init_score(self, y):
        return np.array([float(np.mean(y))])

    def grad_hess(self, y, raw):
        g = raw - y[:, None]
        h = np.ones_like(g)
        return g, h

    def transform(self, raw):
        return raw[:, 0]

    def eval_metric(self, y, raw):
        return float(np.sqrt(np.mean((raw[:, 0] - y) ** 2)))


class Logistic(Loss):
    """Binary cross-entropy on a single logit; y ∈ {0, 1}."""

    metric_name = "logloss"

    def n_outputs(self, y):
        return 1

    def init_score(self, y):
        p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return np.array([np.log(p / (1.0 - p))])

    def grad_hess(self, y, raw):
        p = sigmoid(raw)
        g = p - y[:, None]
        h = np.clip(p * (1.0 - p), _EPS_H, None)
        return g, h

    def transform(self, raw):
        return sigmoid(raw[:, 0])

    def eval_metric(self, y, raw):
        p = np.clip(sigmoid(raw[:, 0]), _EPS_P, 1.0 - _EPS_P)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


class Softmax(Loss):
    """Multiclass cross-entropy; y holds integer class indices 0..K-1.

    Uses the standard diagonal-hessian approximation, so the booster fits one
    tree per class per round.
    """

    metric_name = "mlogloss"

    def __init__(self, n_classes: int):
        if n_classes < 3:
            raise ValueError("Softmax expects at least 3 classes; use Logistic for 2")
        self.n_classes = int(n_classes)

    def n_outputs(self, y):
        return self.n_classes

    def init_score(self, y):
        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes)
        priors = np.clip(counts / max(len(y), 1), 1e-6, None)
        return np.log(priors)

    def grad_hess(self, y, raw):
        p = softmax(raw)
        g = p.copy()
        g[np.arange(len(y)), y.astype(np.int64)] -= 1.0
        h = np.clip(p * (1.0 - p), _EPS_H, None)
        return g, h

    def transform(self, raw):
        return softmax(raw)

    def eval_metric(self, y, raw):
        p = np.clip(softmax(raw), _EPS_P, 1.0)
        return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(np.int64)])))
