"""Temporal feature extractors.

Two families turn a raw series into the scalar columns a tree can threshold:

* ``interval_stat`` — a summary statistic over a time window, e.g. "slope of
  channel 2 between t=12 and t=40".
* ``shapelet_features`` — the minimum sliding z-normalised distance to a short
  template ("does this shape occur?") together with the position of that best
  match ("*when* does it occur?").

Every function is NaN-aware by explicit masking.  NaN is the library's single
representation of "missing/padded", and a missing feature value is a first-class
citizen downstream: the split scan learns which way to route it.
"""

from __future__ import annotations

import warnings

import numpy as np

STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "slope",
    "median",
    "mean_abs_change",
    "last",
    "delta",
)

_EPS_SD = 1e-12


def interval_stat(sub: np.ndarray, stat: str) -> np.ndarray:
    """Summary statistic of each row of ``sub`` ``(m, L)``; returns ``(m,)``.

    Rows with too few finite observations for the statistic to be defined yield
    NaN rather than raising or silently substituting zero.
    """
    sub = np.asarray(sub, dtype=np.float64)
    if sub.ndim != 2:
        raise ValueError(f"interval_stat expects a 2-D block, got shape {sub.shape}")

    m, L = sub.shape
    out = np.full(m, np.nan, dtype=np.float64)
    if L == 0:
        return out

    mask = np.isfinite(sub)
    cnt = mask.sum(axis=1)
    has = cnt > 0
    z = np.where(mask, sub, 0.0)

    if stat == "mean":
        out[has] = z.sum(axis=1)[has] / cnt[has]

    elif stat == "std":
        mu = np.zeros(m)
        mu[has] = z.sum(axis=1)[has] / cnt[has]
        ex2 = np.zeros(m)
        ex2[has] = (z * z).sum(axis=1)[has] / cnt[has]
        out[has] = np.sqrt(np.maximum(ex2[has] - mu[has] ** 2, 0.0))

    elif stat == "min":
        vals = np.where(mask, sub, np.inf).min(axis=1)
        out[has] = vals[has]

    elif stat == "max":
        vals = np.where(mask, sub, -np.inf).max(axis=1)
        out[has] = vals[has]

    elif stat == "slope":
        t = np.arange(L, dtype=np.float64)
        n = cnt.astype(np.float64)
        st = (mask * t).sum(axis=1)
        stt = (mask * t * t).sum(axis=1)
        sx = z.sum(axis=1)
        stx = (z * t).sum(axis=1)
        denom = n * stt - st * st
        ok = (cnt >= 2) & (np.abs(denom) > _EPS_SD)
        out[ok] = (n[ok] * stx[ok] - st[ok] * sx[ok]) / denom[ok]

    elif stat == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(mask, sub, np.nan), axis=1)
        out[has] = med[has]

    elif stat == "mean_abs_change":
        if L >= 2:
            d = np.abs(np.diff(sub, axis=1))
            dmask = np.isfinite(d)
            dcnt = dmask.sum(axis=1)
            dhas = dcnt > 0
            out[dhas] = np.where(dmask, d, 0.0).sum(axis=1)[dhas] / dcnt[dhas]

    elif stat in ("last", "delta"):
        # argmax on an all-False row returns 0, which would silently point at a
        # NaN; only rows with at least one finite value are ever assigned.
        rows = np.nonzero(has)[0]
        if rows.size:
            last_idx = L - 1 - np.argmax(mask[:, ::-1], axis=1)
            if stat == "last":
                out[rows] = sub[rows, last_idx[rows]]
            else:
                first_idx = np.argmax(mask, axis=1)
                out[rows] = sub[rows, last_idx[rows]] - sub[rows, first_idx[rows]]

    else:
        raise ValueError(f"unknown statistic {stat!r}; expected one of {STAT_NAMES}")

    return out


def _sliding_sums(X2d: np.ndarray, l: int):
    """Sliding window sums, sums-of-squares and NaN counts, via cumulative sums.

    Returns ``(Z, S1, S2, bad)`` where ``Z`` is ``X2d`` with non-finite entries
    replaced by 0, ``S1``/``S2`` are ``(m, P)`` window sums of ``Z`` and ``Z²``,
    and ``bad`` marks windows touching a non-finite value (their sums are
    meaningless and must never be used).
    """
    m, T = X2d.shape
    P = T - l + 1
    finite = np.isfinite(X2d)
    Z = np.where(finite, X2d, 0.0)
    zero = np.zeros((m, 1), dtype=np.float64)

    c1 = np.concatenate([zero, np.cumsum(Z, axis=1)], axis=1)
    c2 = np.concatenate([zero, np.cumsum(Z * Z, axis=1)], axis=1)
    cn = np.concatenate([zero, np.cumsum((~finite).astype(np.float64), axis=1)], axis=1)

    S1 = c1[:, l:] - c1[:, :P]
    S2 = c2[:, l:] - c2[:, :P]
    bad = (cn[:, l:] - cn[:, :P]) > 0
    return Z, S1, S2, bad


def shapelet_features(
    X2d: np.ndarray,
    shp: np.ndarray,
    znorm: bool = True,
    chunk_bytes: int = 64 << 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum sliding distance to ``shp`` and the position of the best match.

    ``X2d`` is ``(m, T)``, ``shp`` is ``(l,)``.  Returns ``(dist, pos)``, each
    ``(m,)``, with ``pos`` normalised to [0, 1].  Rows where no window is usable
    (all windows touch NaN, or the series is shorter than the shapelet) get NaN
    for both.

    The distance is the mean squared difference between the z-normalised window
    and the z-normalised shapelet.  It is evaluated through the algebraic
    expansion ``mean(w²) − 2·w·s/l + mean(s²)`` over cumulative sums, which is
    mathematically identical to materialising every window but avoids building
    an ``(m, P, l)`` tensor — the difference between a few MB and a few GB on
    realistic inputs.  (The variance is computed as ``E[x²] − E[x]²``, so series
    whose values dwarf their fluctuations would lose precision; real-world
    scales are far from that regime.)
    """
    X2d = np.asarray(X2d, dtype=np.float64)
    shp = np.asarray(shp, dtype=np.float64)
    if X2d.ndim != 2:
        raise ValueError(f"shapelet_features expects (m, T), got {X2d.shape}")
    if shp.ndim != 1 or shp.size < 1:
        raise ValueError(f"shapelet must be a non-empty 1-D array, got {shp.shape}")

    m, T = X2d.shape
    l = shp.size
    nan_out = (np.full(m, np.nan), np.full(m, np.nan))
    if l > T or not np.isfinite(shp).all():
        return nan_out

    # Normalise the template once.
    if znorm:
        mu_s = shp.mean()
        sd_s = np.sqrt(max(np.mean(shp * shp) - mu_s * mu_s, 0.0))
        if sd_s > _EPS_SD:
            shp_n = (shp - mu_s) / sd_s
            s2 = 1.0
        else:
            shp_n = np.zeros_like(shp)
            s2 = 0.0
    else:
        shp_n = shp
        s2 = float(np.mean(shp * shp))
    s_sum = float(shp_n.sum())

    P = T - l + 1
    dist_out = np.full(m, np.nan)
    pos_out = np.full(m, np.nan)

    # (m, P) temporaries dominate; keep a handful of them within the budget.
    rows_per_chunk = max(1, int(chunk_bytes) // max(P * 8 * 8, 1))

    for lo in range(0, m, rows_per_chunk):
        hi = min(lo + rows_per_chunk, m)
        Z, S1, S2, bad = _sliding_sums(X2d[lo:hi], l)

        dot = np.zeros((hi - lo, P), dtype=np.float64)
        for j in range(l):
            w = shp_n[j]
            if w != 0.0:
                dot += Z[:, j : j + P] * w

        if znorm:
            mu = S1 / l
            sd = np.sqrt(np.maximum(S2 / l - mu * mu, 0.0))
            ok = sd > _EPS_SD
            sd_safe = np.where(ok, sd, 1.0)
            # (w − mu)·s / sd, with the mean term kept explicit so the identity
            # holds even if the template's mean is not exactly zero.
            dotn = np.where(ok, (dot - mu * s_sum) / sd_safe, 0.0)
            w2 = np.where(ok, 1.0, 0.0)
            d = w2 - 2.0 * dotn / l + s2
        else:
            d = S2 / l - 2.0 * dot / l + s2

        # The one silent-corruption risk in this file: a window touching NaN
        # produces finite-looking sums, so it must be excluded *before* argmin.
        d = np.where(bad | ~np.isfinite(d), np.inf, np.maximum(d, 0.0))

        j_best = np.argmin(d, axis=1)
        d_best = d[np.arange(hi - lo), j_best]
        usable = np.isfinite(d_best)

        dist_out[lo:hi][usable] = d_best[usable]
        if P > 1:
            pos_out[lo:hi][usable] = j_best[usable] / (P - 1)
        else:
            pos_out[lo:hi][usable] = 0.0

    return dist_out, pos_out


def ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Where each value falls in a *frozen* training distribution, in [0, 1].

    The grid is the sorted training column, captured once and never recomputed —
    a rank computed against the batch at hand would mean something different at
    predict time than it did during fitting.
    """
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if grid.size and finite.any():
        out[finite] = np.searchsorted(grid, values[finite], side="right") / grid.size
    return out


def eval_split_feature(spec, X_static, X_series, rows: np.ndarray, pyramid=None) -> np.ndarray:
    """Compute a split's scalar feature for ``rows`` — shared by fit and predict.

    Using one function for both directions is what guarantees a row routes the
    same way at predict time as it did during fitting.
    """
    kind = spec.kind
    if kind == "static":
        return X_static[rows, spec.col]

    if kind == "interval":
        block = X_series[rows, spec.channel, spec.start : spec.end]
        return interval_stat(block, spec.stat)

    if kind in ("shapelet_dist", "shapelet_pos"):
        dist, pos = shapelet_features(
            X_series[rows, spec.channel, :], spec.shapelet, znorm=spec.znorm
        )
        return dist if kind == "shapelet_dist" else pos

    if kind in ("filter_resp", "filter_pos"):
        from .filters import Pyramid, align

        if pyramid is None:
            pyramid = Pyramid(X_series, len(spec.template))
        response, position = align(
            pyramid.block(spec.scale, spec.channel, rows), spec.template
        )
        return response if kind == "filter_resp" else position

    if kind == "comparison":
        inner = eval_split_feature(spec.position_spec, X_static, X_series, rows, pyramid)
        return ecdf(inner, spec.position_grid) - ecdf(
            X_static[rows, spec.col], spec.static_grid
        )

    raise ValueError(f"unknown split kind {kind!r}")
