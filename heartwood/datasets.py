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


def _two_positions(rng, n: int, T: int, length: int):
    """Two well-separated start positions per row, the first strictly earlier."""
    gap = length
    first = rng.integers(0, max(1, T - 2 * length - gap), size=n)
    room = (T - length) - (first + length + gap)
    second = first + length + gap + (rng.random(n) * np.maximum(room + 1, 1)).astype(int)
    return first, np.minimum(second, T - length)


def make_bump_interaction(n=500, T=100, noise=1.0, seed=0, n_noise_static=4, amp=3.0):
    """Binary XOR of *which transient came first* and a static flag.

    Every series contains **both** transients — one up-then-down and one
    down-then-up — at two well-separated but otherwise random positions.  Only
    their order differs between classes, which defeats aggregation at two levels.

    Global statistics are blind by construction:

    * both classes hold exactly the same two shapes, so the global distribution
      of values is identical — mean, std, min, max, median and
      mean-absolute-change carry nothing;
    * each transient has zero net area, so its position contributes nothing to
      the global slope, and the two internal dipoles are equal and opposite, so
      they cancel regardless of order;
    * the transients sit away from the endpoints, so first/last/delta see noise.

    Fixed-window aggregation fails too, and this is why the positions are random
    rather than tidy early/late slots: with the shapes pinned to predictable
    stretches, a four-window summary reads the answer straight off the window
    that happens to contain the first shape (measured: 0.94 accuracy that way,
    versus 0.70 once the positions scatter).  What survives is a question no
    fixed window can answer — *where* did this particular shape occur — which
    needs shape matching plus a position readout.

    The XOR with the static flag then removes the marginal signal from *both*
    modalities, so only their interaction predicts the label.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))

    sigma = max(1.5, T / 50.0)
    shape = _doublet(sigma, 2.0 * sigma)
    length = len(shape)
    first_at, second_at = _two_positions(rng, n, T, length)

    up_first = rng.integers(0, 2, size=n).astype(bool)
    for i in range(n):
        sign = 1.0 if up_first[i] else -1.0
        series[i, 0, first_at[i] : first_at[i] + length] += amp * noise * sign * shape
        series[i, 0, second_at[i] : second_at[i] + length] -= amp * noise * sign * shape

    flag = rng.integers(0, 2, size=n)
    X_static = np.column_stack([flag.astype(np.float64), _noise_cols(rng, n, n_noise_static)])
    y = (up_first ^ (flag == 1)).astype(np.int64)
    return X_static, series, y


def make_timing_task(n=500, T=100, noise=0.5, seed=0, n_noise_static=4, amp=3.0):
    """Binary "did the event happen before this row's deadline".

    Every series contains exactly one identical transient; only *when* it happens
    varies, and the label compares that instant to a per-row static deadline.

    The transient is a zero-area doublet on purpose.  A plain bump would betray
    its own position through the global slope — a localised feature of area A at
    position c contributes A·(c − t̄) to the overall trend, so the slope
    aggregate would read out the timing directly.  With zero area that term
    vanishes and the remaining internal dipole is the same for every row, so no
    global aggregate carries the label at all.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))

    shape = _doublet(max(1.5, T / 50.0), 2.0 * max(1.5, T / 50.0))
    length = len(shape)
    starts = rng.integers(int(0.08 * T), max(int(0.08 * T) + 1, T - length), size=n)
    for i in range(n):
        series[i, 0, starts[i] : starts[i] + length] += amp * noise * shape

    centers = starts + (length - 1) / 2.0
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

    The informative transient always occurs in a narrow stretch of the series;
    a *taller* nuisance transient occurs somewhere outside it.  So the global
    maximum reports the nuisance and says nothing about the target, while the
    maximum over the right window recovers it cleanly: the model has to learn
    where to measure, not just what to measure.  The window is off-grid so
    equal-width windowing does not land on it either.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=(n, 1, T))
    sigma = max(1.5, T / 40.0)

    amplitude = rng.uniform(0.5, 3.0, size=n)
    centers = rng.uniform(0.56 * T, 0.66 * T, size=n)
    series[:, 0, :] += _bump(T, centers, sigma=sigma, amplitude=amplitude)

    # The nuisance stays clear of the signal window, so the target is recoverable
    # there and nowhere else.
    nuisance = rng.uniform(3.0, 5.0, size=n)
    before = rng.uniform(0.05 * T, 0.44 * T, size=n)
    after = rng.uniform(0.78 * T, 0.95 * T, size=n)
    nuisance_at = np.where(rng.random(n) < 0.5, before, after)
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
