"""Input validation and small shared helpers.

Everything the library consumes is normalised here, once, at ``fit``/``predict``
time: float64, C-contiguous, canonical shapes.  Downstream modules assume that
normalisation has already happened and never re-check.
"""

from __future__ import annotations

import numpy as np

MAX_SEED = 2**63


def spawn_rng(master: np.random.Generator) -> np.random.Generator:
    """Derive an independent child generator from ``master``.

    Deriving children this way (rather than sharing one generator) keeps a tree's
    random draws independent of how many draws its siblings happened to make.
    """
    return np.random.default_rng(int(master.integers(MAX_SEED)))


def as_static(X_static, n_rows: int | None = None) -> np.ndarray:
    """Coerce the static block to a C-contiguous float64 ``(n, p)`` array.

    ``None`` becomes an ``(n, 0)`` array so that the rest of the library has a
    single code path (``p == 0`` simply yields no static split candidates).
    """
    if X_static is None:
        if n_rows is None:
            raise ValueError("X_static=None requires X_series to determine n_rows")
        return np.empty((n_rows, 0), dtype=np.float64)

    arr = np.ascontiguousarray(np.asarray(X_static, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X_static must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def as_series(X_series, n_rows: int | None = None, pad_to: int | None = None) -> np.ndarray | None:
    """Coerce the series block to a C-contiguous float64 ``(n, C, T)`` array.

    Accepts ``(n, C, T)``, ``(n, T)``, a list of per-sample ``(C, T_i)`` / ``(T_i,)``
    arrays with **variable lengths** (right-padded with NaN), or ``None``.

    ``pad_to`` right-pads with NaN up to that length; series longer than
    ``pad_to`` are an error, since a model fitted on length T stores split
    windows that are only meaningful within that length.
    """
    if X_series is None:
        return None

    if isinstance(X_series, (list, tuple)):
        arr = _pad_ragged(X_series)
    else:
        arr = np.asarray(X_series, dtype=np.float64)
        if arr.dtype == object:
            arr = _pad_ragged(list(X_series))
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        elif arr.ndim != 3:
            raise ValueError(
                f"X_series must be (n, C, T), (n, T) or a list of per-sample arrays, "
                f"got shape {arr.shape}"
            )

    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(
            f"X_series has {arr.shape[0]} rows but X_static/y has {n_rows}"
        )

    if pad_to is not None:
        T = arr.shape[2]
        if T > pad_to:
            raise ValueError(
                f"series length {T} exceeds the fitted length {pad_to}; the model's "
                f"split windows are defined on length {pad_to}. Truncate or refit."
            )
        if T < pad_to:
            pad = np.full((arr.shape[0], arr.shape[1], pad_to - T), np.nan)
            arr = np.concatenate([arr, pad], axis=2)

    return np.ascontiguousarray(arr, dtype=np.float64)


def _pad_ragged(seq) -> np.ndarray:
    """Right-pad a list of per-sample series with NaN to a common length."""
    if len(seq) == 0:
        raise ValueError("X_series is an empty sequence")

    mats = []
    for i, item in enumerate(seq):
        a = np.asarray(item, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2:
            raise ValueError(
                f"X_series[{i}] must be 1-D (T,) or 2-D (C, T), got shape {a.shape}"
            )
        mats.append(a)

    channels = {m.shape[0] for m in mats}
    if len(channels) != 1:
        raise ValueError(f"all samples must have the same channel count, got {sorted(channels)}")

    C = mats[0].shape[0]
    T = max(m.shape[1] for m in mats)
    out = np.full((len(mats), C, T), np.nan, dtype=np.float64)
    for i, m in enumerate(mats):
        out[i, :, : m.shape[1]] = m
    return out


def check_inputs(X_static, X_series, n_rows: int | None = None, pad_to: int | None = None):
    """Normalise both blocks together and return ``(X_static, X_series, n)``."""
    if X_static is None and X_series is None:
        raise ValueError("at least one of X_static / X_series must be provided")

    Xt = as_series(X_series, pad_to=pad_to)
    n = n_rows if n_rows is not None else (Xt.shape[0] if Xt is not None else None)
    Xs = as_static(X_static, n_rows=n)
    n = Xs.shape[0]

    if Xt is not None and Xt.shape[0] != n:
        raise ValueError(f"X_series has {Xt.shape[0]} rows but X_static has {n}")
    if n == 0:
        raise ValueError("got 0 rows")
    return Xs, Xt, n
