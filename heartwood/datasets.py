"""Synthetic generators where the signal lives in the temporal journey.

Each scenario is built so that the industry-standard workaround — summarise the
series into global aggregates, concatenate with the static columns, fit a GBM —
throws away most of the signal, while a model that can look at *where* and *when*
something happened recovers it.  They are shared by the tests (which check the
signal is really there) and the benchmarks (which check we can find it).

Every generator returns ``(X_static, X_series, y)`` and is deterministic in
``seed``.  The static block always carries a few pure-noise distractor columns.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_bump_interaction",
    "make_timing_task",
    "make_slope_window",
    "make_shape_amplitude_regression",
    "make_pure_static",
    "make_static_plus_noise_series",
]


def _bump(T: int, centers: np.ndarray, sigma: float, amplitude: np.ndarray) -> np.ndarray:
    """Gaussian humps of given centre/amplitude, one row per sample."""
    t = np.arange(T, dtype=np.float64)
    z = (t[None, :] - np.asarray(centers, dtype=np.float64)[:, None]) / sigma
    return np.asarray(amplitude, dtype=np.float64)[:, None] * np.exp(-0.5 * z * z)


def _doublet(sigma: float, sep: float) -> np.ndarray:
    """An up-then-down transient: two opposite Gaussian lobes, zero net area."""
    length = int(round(4 * sigma + 2 * sep))
    u = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    return np.exp(-0.5 * ((u + sep) / sigma) ** 2) - np.exp(-0.5 * ((u - sep) / sigma) ** 2)


def _noise_cols(rng, n: int, k: int) -> np.ndarray:
    return rng.normal(size=(n, k))


def make_bump_interaction(n=500, T=100, noise=1.0, seed=0, n_noise_static=4, amp=3.0):
    """Binary XOR of a transient's *orientation* and a static flag.

    Every series contains one transient at a random position: either up-then-down
    or down-then-up.  The two are exact negations of one another, which makes the
    scenario adversarial to aggregation by construction:

    * the two classes have **identical** global value distributions, so mean,
      std, min, max, median and mean-absolute-change carry no information at all;
    * the transient has zero net area, so it does not move the mean;
    * the only leak — its dipole, which tilts the global slope — is an order of
      magnitude below the slope's own noise.

    An aggregate-and-concatenate baseline therefore sits at chance, while a
    z-normalised shape detector separates the classes almost perfectly (the two
    orientations correlate at −1).  The XOR with the static flag removes the
    marginal signal from *both* modalities, so only their interaction predicts.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))

    sigma = max(1.5, T / 50.0)
    shape = _doublet(sigma, 2.0 * sigma)
    length = len(shape)

    is_down = rng.integers(0, 2, size=n).astype(bool)
    starts = rng.integers(0, T - length + 1, size=n)
    sign = np.where(is_down, -1.0, 1.0)
    for i in range(n):
        series[i, 0, starts[i] : starts[i] + length] += amp * noise * sign[i] * shape

    flag = rng.integers(0, 2, size=n)
    X_static = np.column_stack([flag.astype(np.float64), _noise_cols(rng, n, n_noise_static)])
    y = (is_down ^ (flag == 1)).astype(np.int64)
    return X_static, series, y


def make_timing_task(n=500, T=100, noise=0.5, seed=0, n_noise_static=4):
    """Binary "did the event happen before this row's deadline".

    Every series contains exactly one identical transient, so *no* global
    aggregate carries any information at all — only its position does, compared
    against a per-row static deadline.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))

    centers = rng.uniform(0.15 * T, 0.85 * T, size=n)
    series[:, 0, :] += _bump(T, centers, sigma=T / 40.0, amplitude=np.full(n, 3.0 * noise))

    deadline = rng.uniform(0.2 * T, 0.8 * T, size=n)
    X_static = np.column_stack([deadline, _noise_cols(rng, n, n_noise_static)])
    y = (centers < deadline).astype(np.int64)
    return X_static, series, y


def make_slope_window(n=500, T=120, noise=0.8, seed=0, n_noise_static=4):
    """Binary sign-of-trend inside one window, diluted by distractor trends.

    The label depends only on the direction of drift over a window covering
    t ∈ [0.38 T, 0.58 T); the rest of the series drifts randomly and more
    steeply, so the *global* slope is dominated by the distractors.  The window
    is deliberately **not** aligned to any halves-or-quarters grid, so a
    fixed-window aggregation scheme straddles it and dilutes the signal too —
    the model has to find where to look.  A static column flips the label,
    forcing an interaction.
    """
    rng = np.random.default_rng(seed)
    lo = int(round(0.38 * T))
    hi = int(round(0.58 * T))
    edges = [0, lo, hi, int(round(0.79 * T)), T]

    target_sign = rng.choice([-1.0, 1.0], size=n)
    step = np.zeros((n, T), dtype=np.float64)
    for k in range(4):
        a, b = edges[k], edges[k + 1]
        slope = target_sign * 0.12 if k == 1 else rng.uniform(-0.3, 0.3, size=n)
        step[:, a:b] = np.asarray(slope).reshape(-1, 1)

    series = np.cumsum(step, axis=1) + rng.normal(scale=noise, size=(n, T))
    series = series[:, None, :]

    gate = rng.uniform(-1.0, 1.0, size=n)
    X_static = np.column_stack([gate, _noise_cols(rng, n, n_noise_static)])
    y = ((target_sign > 0) ^ (gate > 0)).astype(np.int64)
    return X_static, series, y


def make_shape_amplitude_regression(n=500, T=100, noise=0.5, seed=0, n_noise_static=4):
    """Regression on a transient's height times a static coefficient.

    The informative transient always occurs in a narrow stretch of the series,
    but a *larger* nuisance transient occurs anywhere.  So the global maximum
    reports the nuisance and says nothing about the target, while the maximum
    over the right window recovers it: the model has to learn where to measure,
    not just what to measure.  The window is off-grid so equal-width windowing
    does not land on it either.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))
    sigma = max(1.5, T / 40.0)

    amplitude = rng.uniform(0.5, 3.0, size=n)
    centers = rng.uniform(0.56 * T, 0.66 * T, size=n)
    series[:, 0, :] += _bump(T, centers, sigma=sigma, amplitude=amplitude)

    nuisance = rng.uniform(3.0, 5.0, size=n)
    nuisance_at = rng.uniform(0.05 * T, 0.95 * T, size=n)
    series[:, 0, :] += _bump(T, nuisance_at, sigma=sigma, amplitude=nuisance)

    coef = rng.uniform(0.5, 2.0, size=n)
    X_static = np.column_stack([coef, _noise_cols(rng, n, n_noise_static)])
    y = amplitude * coef + rng.normal(scale=0.1, size=n)
    return X_static, series, y


def make_pure_static(n=500, p=10, seed=0):
    """Control: all signal is static, no series at all (``X_series is None``).

    Used to check that the temporal machinery costs nothing when there is
    nothing temporal to find.
    """
    rng = np.random.default_rng(seed)
    X_static = rng.normal(size=(n, p))
    score = 1.5 * X_static[:, 0] - 1.2 * X_static[:, 1] + 1.0 * X_static[:, 0] * X_static[:, 2]
    y = (score > np.median(score)).astype(np.int64)
    return X_static, None, y


def make_static_plus_noise_series(n=500, p=10, T=100, noise=1.0, seed=0):
    """Control: static signal with a pure-noise series attached.

    The harder version of :func:`make_pure_static` — the temporal candidates are
    offered thousands of chances to find something, and must decline every time.
    """
    X_static, _, y = make_pure_static(n=n, p=p, seed=seed)
    rng = np.random.default_rng(seed + 99991)
    series = rng.normal(scale=noise, size=(n, 1, T))
    return X_static, series, y
