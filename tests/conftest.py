"""Shared fixtures and slow references used across the suite.

The references here are deliberately naive — plain Python loops that are obviously
correct at a glance.  Their whole job is to disagree with the fast vectorised
implementations when those are wrong, so they must never share code with them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20250820)


def brute_interval_stat(row: np.ndarray, stat: str) -> float:
    """Slow, obvious reference for one row of :func:`heartwood.features.interval_stat`."""
    finite = np.isfinite(row)
    v = row[finite]
    if stat == "mean":
        return v.mean() if v.size else np.nan
    if stat == "std":
        return v.std() if v.size else np.nan
    if stat == "min":
        return v.min() if v.size else np.nan
    if stat == "max":
        return v.max() if v.size else np.nan
    if stat == "median":
        return np.median(v) if v.size else np.nan
    if stat == "slope":
        idx = np.nonzero(finite)[0].astype(float)
        if idx.size < 2:
            return np.nan
        denom = idx.size * (idx**2).sum() - idx.sum() ** 2
        if abs(denom) <= 1e-12:
            return np.nan
        return (idx.size * (idx * v).sum() - idx.sum() * v.sum()) / denom
    if stat == "mean_abs_change":
        d = np.abs(np.diff(row))
        d = d[np.isfinite(d)]
        return d.mean() if d.size else np.nan
    if stat == "last":
        return v[-1] if v.size else np.nan
    if stat == "delta":
        return v[-1] - v[0] if v.size else np.nan
    raise AssertionError(f"unknown stat {stat!r}")


def brute_shapelet(X2d: np.ndarray, shp: np.ndarray, znorm: bool = True):
    """Slow reference for :func:`heartwood.features.shapelet_features`.

    Materialises every window explicitly and skips any window touching a
    non-finite value — the behaviour the fast path has to reproduce.
    """
    m, T = X2d.shape
    l = len(shp)
    dist = np.full(m, np.nan)
    pos = np.full(m, np.nan)
    if l > T:
        return dist, pos
    P = T - l + 1

    if znorm:
        sd = shp.std()
        template = (shp - shp.mean()) / sd if sd > 1e-12 else np.zeros_like(shp)
    else:
        template = shp

    for i in range(m):
        best, best_j = np.inf, -1
        for p in range(P):
            w = X2d[i, p : p + l]
            if not np.isfinite(w).all():
                continue
            if znorm:
                sd = w.std()
                wn = (w - w.mean()) / sd if sd > 1e-12 else np.zeros_like(w)
            else:
                wn = w
            d = float(np.mean((wn - template) ** 2))
            if d < best:
                best, best_j = d, p
        if best_j >= 0:
            dist[i] = best
            pos[i] = best_j / (P - 1) if P > 1 else 0.0
    return dist, pos


def brute_scan(f, g, h, reg_lambda, gamma, min_child_weight, min_samples_leaf):
    """O(n²) reference for :func:`heartwood.splits.scan_threshold`.

    Tries every candidate threshold and both missing-directions explicitly.
    """
    n = f.size
    finite = np.isfinite(f)
    G, H = g.sum(), h.sum()
    parent = G * G / (H + reg_lambda)
    best = None
    values = np.unique(f[finite])
    for missing_left in (True, False):
        for i in range(len(values) - 1):
            threshold = 0.5 * (values[i] + values[i + 1])
            if threshold >= values[i + 1]:
                threshold = values[i]
            left = np.where(finite, f <= threshold, missing_left)
            n_left = int(left.sum())
            GL, HL = g[left].sum(), h[left].sum()
            GR, HR = G - GL, H - HL
            if (
                HL < min_child_weight
                or HR < min_child_weight
                or n_left < min_samples_leaf
                or n - n_left < min_samples_leaf
            ):
                continue
            gain = 0.5 * (
                GL**2 / (HL + reg_lambda) + GR**2 / (HR + reg_lambda) - parent
            ) - gamma
            if best is None or gain > best[0] + 1e-15:
                best = (gain, threshold, missing_left)
    if best is None or best[0] <= 1e-12:
        return None
    return best


def noisy_feature(rng, n, nan_frac=0.0, decimals=1):
    """A feature column with deliberate duplicate values and optional NaNs."""
    f = rng.normal(size=n).round(decimals)
    if nan_frac:
        f[rng.random(n) < nan_frac] = np.nan
    return f
