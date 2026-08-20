"""Matched filters: temporal templates *solved for* rather than sampled.

A shapelet candidate is a snippet cut out of some training row and hoped to be
discriminative.  Most are not — measured on the benchmark scenarios, fewer than
one draw in ten is useful — so the search wastes most of its budget and small
nodes rarely get lucky.

A matched filter starts the same way but then fits the template, in closed form,
to the node's own Newton residuals.  The fit has five degrees of freedom (a
smooth DCT basis under ridge), which is the point: a template with five
coefficients physically cannot memorise a noisy snippet the way a raw cut can.
Multiple pooling scales let a nine-tap filter cover a long, slow pattern as
easily as a brief one.

Two scalars come out of each filter, and both go into the ordinary split
tournament: the signed correlation at its best match ("does this shape occur,
and with what polarity") and where that match landed ("when").

With ``n_alt=0`` the refit is skipped and the family reduces *exactly* to the
shapelet family — ``z-normalised distance == 2·(1 − correlation)`` — which is
checked by a test.  So enabling filters can never shrink the hypothesis space.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12

__all__ = [
    "Pyramid",
    "build_pyramid",
    "dct_basis",
    "align",
    "gather_windows",
    "refit_template",
    "znorm_snippet",
]


class Pyramid:
    """The pooling pyramid for one batch of series, built once and reused.

    Fitting touches this thousands of times, so it is built per fit and per
    predict batch rather than per node.  Total memory is about twice the series
    array, since each level is half the size of the one above.
    """

    __slots__ = ("levels", "filter_len")

    def __init__(self, X_series: np.ndarray, filter_len: int, max_scales: int = 8):
        self.filter_len = int(filter_len)
        self.levels = build_pyramid(X_series, self.filter_len, max_scales)

    @property
    def n_scales(self) -> int:
        return len(self.levels)

    def level(self, scale: int) -> np.ndarray:
        return self.levels[min(int(scale), len(self.levels) - 1)]

    def block(self, scale: int, channel: int, rows) -> np.ndarray:
        return self.level(scale)[rows, channel, :]


def build_pyramid(X_series: np.ndarray, filter_len: int, max_scales: int = 8) -> list[np.ndarray]:
    """NaN-aware dyadic average-pooling pyramid.

    Level ``s`` halves the resolution of level ``s-1``: each pair averages only
    its finite members, and a pair with nothing observed yields NaN rather than
    zero — pooling must not invent data where there is none.  Scales stop once a
    level is too short to slide the filter over.
    """
    levels = [X_series]
    while len(levels) < max_scales:
        current = levels[-1]
        length = current.shape[2]
        if length // 2 < filter_len + 1:
            break
        left = current[:, :, 0 : 2 * (length // 2) : 2]
        right = current[:, :, 1 : 2 * (length // 2) : 2]
        finite_left = np.isfinite(left)
        finite_right = np.isfinite(right)
        count = finite_left.astype(np.float64) + finite_right
        total = np.where(finite_left, left, 0.0) + np.where(finite_right, right, 0.0)
        levels.append(np.where(count > 0, total / np.maximum(count, 1.0), np.nan))
    return levels


def dct_basis(length: int, n_components: int) -> np.ndarray:
    """Orthonormal smooth basis, ``(length, n_components)``.

    DCT-II components 1..K.  Component 0 — the constant — is deliberately
    excluded: the windows being fitted are mean-centred, so including it would
    make the solve degenerate and silently return a meaningless template.
    """
    n_components = max(1, min(n_components, length - 1))
    j = np.arange(length, dtype=np.float64)[:, None]
    k = np.arange(1, n_components + 1, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * (j + 0.5) * k / length)
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def znorm_snippet(snippet: np.ndarray) -> np.ndarray | None:
    """Mean-centre and scale to unit norm; ``None`` if there is no shape to keep."""
    if not np.isfinite(snippet).all():
        return None
    centred = snippet - snippet.mean()
    norm = float(np.linalg.norm(centred))
    if norm <= _EPS:
        return None
    return centred / norm


def align(block: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best normalised cross-correlation of ``template`` against each row.

    ``block`` is ``(m, T)`` at one channel and scale.  Returns the *signed*
    correlation at the best-matching offset and that offset normalised to
    [0, 1].  Rows with no usable window get NaN for both.

    Windows touching a non-finite value are excluded *before* the arg-max.  This
    is the one place where a wrong line would not announce itself: such a window
    still produces finite-looking sums, so it would win matches on padding and
    every downstream number would stay plausible.
    """
    block = np.asarray(block, dtype=np.float64)
    m, T = block.shape
    length = len(template)
    if length > T or m == 0:
        return np.full(m, np.nan), np.full(m, np.nan)

    centred = template - template.mean()
    template_norm = float(np.linalg.norm(centred))
    n_positions = T - length + 1

    finite = np.isfinite(block)
    filled = np.where(finite, block, 0.0)
    zero = np.zeros((m, 1), dtype=np.float64)
    cumulative = np.concatenate([zero, np.cumsum(filled, axis=1)], axis=1)
    squares = np.concatenate([zero, np.cumsum(filled * filled, axis=1)], axis=1)
    missing = np.concatenate([zero, np.cumsum((~finite).astype(np.float64), axis=1)], axis=1)

    window_sum = cumulative[:, length:] - cumulative[:, :n_positions]
    window_sqsum = squares[:, length:] - squares[:, :n_positions]
    touches_nan = (missing[:, length:] - missing[:, :n_positions]) > 0

    dot = np.zeros((m, n_positions), dtype=np.float64)
    for j in range(length):
        weight = centred[j]
        if weight != 0.0:
            dot += filled[:, j : j + n_positions] * weight

    spread = np.sqrt(np.maximum(window_sqsum - window_sum**2 / length, 0.0))
    usable = (spread > _EPS) & ~touches_nan & (template_norm > _EPS)
    correlation = np.where(usable, dot / np.where(usable, spread * template_norm, 1.0), 0.0)
    correlation = np.clip(correlation, -1.0, 1.0)

    ranked = np.where(touches_nan, -np.inf, np.abs(correlation))
    has_window = np.isfinite(ranked).any(axis=1)
    best = np.argmax(ranked, axis=1)

    response = np.where(has_window, correlation[np.arange(m), best], np.nan)
    position = (
        np.where(has_window, best / (n_positions - 1), np.nan)
        if n_positions > 1
        else np.where(has_window, 0.0, np.nan)
    )
    return response, position


def gather_windows(block: np.ndarray, position: np.ndarray, length: int):
    """Collect each row's best-matching window, mean-centred and unit-norm.

    Returns ``(rows, windows)`` for the rows that had a usable match — the ones
    the refit is allowed to learn from.
    """
    m, T = block.shape
    n_positions = T - length + 1
    if n_positions < 1:
        return np.empty(0, dtype=np.intp), np.empty((0, length))

    usable = np.nonzero(np.isfinite(position))[0]
    if usable.size == 0:
        return usable, np.empty((0, length))

    start = (
        np.rint(position[usable] * (n_positions - 1)).astype(np.intp)
        if n_positions > 1
        else np.zeros(usable.size, dtype=np.intp)
    )
    start = np.clip(start, 0, n_positions - 1)
    windows = block[usable[:, None], start[:, None] + np.arange(length)[None, :]]

    windows = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(windows, axis=1, keepdims=True)
    keep = (norms[:, 0] > _EPS) & np.isfinite(windows).all(axis=1)
    return usable[keep], windows[keep] / norms[keep]


def refit_template(
    windows: np.ndarray,
    residuals: np.ndarray,
    hessians: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> np.ndarray | None:
    """Solve for the template that best explains the node's Newton residuals.

    A hessian-weighted ridge regression of the residuals on the aligned windows,
    restricted to the smooth basis: ``K+1`` unknowns, one small linear solve, no
    iteration and no learning rate.  The intercept is left unpenalised.

    Returns a unit-norm template, or ``None`` if the solve is degenerate — in
    which case the caller keeps the template it already had.
    """
    if windows.shape[0] < basis.shape[1] + 2:
        return None

    design = np.column_stack([np.ones(len(windows)), windows @ basis])
    weighted = design * hessians[:, None]
    normal = design.T @ weighted
    penalty = np.eye(normal.shape[0])
    penalty[0, 0] = 0.0  # the intercept is not a shape coefficient
    try:
        coefficients = np.linalg.solve(normal + ridge * penalty, weighted.T @ residuals)
    except np.linalg.LinAlgError:
        return None

    template = basis @ coefficients[1:]
    norm = float(np.linalg.norm(template))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return template / norm
