"""Dense precomputed columns: a regularised linear base, and cross-channel areas.

Trees are good at carving out interactions and hopeless at adding up a thousand
individually weak signals — each split has to justify itself alone.  A ridge over
a wide fixed feature bank does exactly the opposite.  So :class:`DenseBase` fits
one, and hands the booster its predictions as a starting point rather than a
constant.

The whole thing rests on one piece of hygiene.  If the trees were trained against
the ridge's *own* fitted values, the ridge would look far better on training rows
than it will on new ones, the trees would find nothing left to learn, and the
model would fall apart at prediction time with every training metric looking
excellent.  The fix is to train them against leave-one-out predictions instead,
which ridge gives in closed form for free — no refitting, no folds.
"""

from __future__ import annotations

import numpy as np

from .features import STAT_NAMES, interval_stat

_EPS = 1e-12
LAMBDA_GRID = np.logspace(-3, 3, 13)


def dyadic_windows(T: int, levels: int = 4) -> list[tuple[int, int]]:
    """Whole series, halves, quarters, eighths — each at 50% overlap.

    Deterministic and label-free by design: this bank must not be chosen using
    the labels, or the ridge inherits a selection bias that leave-one-out cannot
    undo.
    """
    windows: list[tuple[int, int]] = []
    for level in range(levels):
        length = T // (2**level)
        if length < 2:
            break
        stride = max(1, length // 2)
        start = 0
        while start + length <= T:
            windows.append((start, start + length))
            start += stride
    seen: dict[tuple[int, int], None] = {}
    for window in windows:
        seen.setdefault(window, None)
    return list(seen)


def dense_bank(X_series: np.ndarray, stats: tuple[str, ...] = STAT_NAMES) -> np.ndarray:
    """Every statistic over every dyadic window, for each channel and its first difference.

    Wide on purpose — this is the input the linear layer is meant to aggregate.
    It deterministically contains the global-aggregate baseline as a subset.
    """
    n, n_channels, T = X_series.shape
    windows = dyadic_windows(T)
    columns: list[np.ndarray] = []
    for channel in range(n_channels):
        base = X_series[:, channel, :]
        differenced = np.diff(base, axis=1) if T > 1 else base
        for block in (base, differenced):
            for start, end in windows:
                piece = block[:, min(start, block.shape[1] - 1) : min(end, block.shape[1])]
                if piece.shape[1] == 0:
                    continue
                for stat in stats:
                    columns.append(interval_stat(piece, stat))
    return np.column_stack(columns).astype(np.float64) if columns else np.zeros((n, 0))


def levy_area_columns(X_series: np.ndarray, max_pairs: int = 6) -> np.ndarray:
    """Signed areas between channel pairs — who moves first.

    The Lévy area ``½ Σ (xΔy − yΔx)`` of a two-channel path is positive when the
    first channel leads the second and negative when it lags.  No per-channel
    statistic can express that, however many windows you give it, because the
    information is in the *joint* trajectory.

    Returns an empty block for single-channel data, where the notion is vacuous.
    """
    n, n_channels, T = X_series.shape
    if n_channels < 2 or T < 2:
        return np.zeros((n, 0))

    pairs = [(a, b) for a in range(n_channels) for b in range(a + 1, n_channels)][:max_pairs]
    windows = dyadic_windows(T, levels=3)
    columns: list[np.ndarray] = []

    for a, b in pairs:
        for start, end in windows:
            x = X_series[:, a, start:end]
            y = X_series[:, b, start:end]
            if x.shape[1] < 2:
                continue
            # Observed points only; a step touching a gap contributes nothing.
            dx = np.diff(x, axis=1)
            dy = np.diff(y, axis=1)
            x0 = x[:, :-1] - x[:, :1]
            y0 = y[:, :-1] - y[:, :1]
            term = 0.5 * (x0 * dy - y0 * dx)
            valid = np.isfinite(term)
            counts = valid.sum(axis=1)
            area = np.where(counts > 0, np.where(valid, term, 0.0).sum(axis=1), np.nan)
            columns.append(area)
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def _platt(margins: np.ndarray, y: np.ndarray, iterations: int = 50) -> tuple[float, float]:
    """Map least-squares margins onto logits by fitting ``sigmoid(a·m + b)``."""
    a, b = 1.0, 0.0
    for _ in range(iterations):
        z = a * margins + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-9)
        residual = p - y
        grad = np.array([float(residual @ margins), float(residual.sum())])
        design = np.column_stack([margins, np.ones_like(margins)])
        hessian = design.T @ (design * w[:, None]) + 1e-9 * np.eye(2)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


class DenseBase:
    """Ridge over a wide temporal bank, with exact leave-one-out training margins.

    ``fit`` returns the margins the *booster* should train against: for each row,
    what the ridge would have predicted had that row been left out.  ``transform``
    returns what the full-data fit predicts, which is the right thing for rows
    the model has never seen.
    """

    def __init__(self, task: str, n_outputs: int):
        self.task = task
        self.n_outputs = int(n_outputs)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.impute_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.target_center_: np.ndarray | None = None
        self.lambda_: float = 1.0
        self.calibration_: list[tuple[float, float]] = []

    # ---------------------------------------------------------------- fitting

    def _targets(self, y: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return y[:, None].astype(np.float64)
        if self.n_outputs == 1:  # binary, as ±1
            return np.where(y[:, None] > 0.5, 1.0, -1.0)
        onehot = np.full((len(y), self.n_outputs), -1.0)
        onehot[np.arange(len(y)), y.astype(int)] = 1.0
        return onehot

    def _prepare(self, bank: np.ndarray, fitting: bool) -> np.ndarray:
        if fitting:
            finite = np.isfinite(bank)
            with np.errstate(invalid="ignore"):
                self.impute_ = np.where(
                    finite.any(axis=0), np.nanmedian(np.where(finite, bank, np.nan), axis=0), 0.0
                )
            self.impute_ = np.nan_to_num(self.impute_)
        filled = np.where(np.isfinite(bank), bank, self.impute_)
        if fitting:
            self.center_ = filled.mean(axis=0)
            spread = filled.std(axis=0)
            self.scale_ = np.where(spread > _EPS, spread, 1.0)
        return (filled - self.center_) / self.scale_

    def fit(self, bank: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the ridge and return the leave-one-out margins, ``(n, K)``."""
        X = self._prepare(bank, fitting=True)
        Y = self._targets(y)
        self.target_center_ = Y.mean(axis=0)
        Yc = Y - self.target_center_

        U, singular, Vt = np.linalg.svd(X, full_matrices=False)
        s2 = singular**2
        UtY = U.T @ Yc
        n_rows = len(X)

        # The grid is scaled to the data's own spectrum, so it means the same
        # thing whatever the features happen to be measured in.
        grid = LAMBDA_GRID * max(float(s2.mean()), _EPS)

        best = fallback = None
        for lam in grid:
            shrink = s2 / (s2 + lam)
            leverage = (U**2) @ shrink
            fitted = U @ (shrink[:, None] * UtY)
            denominator = (1.0 - leverage)[:, None]
            candidate = (lam, shrink, leverage, fitted)

            # With more features than rows, a weak penalty lets the ridge
            # interpolate: every leverage goes to 1, the leave-one-out
            # denominator goes to 0, and the ratio becomes floating-point
            # residue that still carries the sign of the label.  That looks like
            # a perfect training fit and predicts at chance.  Such a lambda is
            # not merely a poor choice, it is uninformative, so it is refused.
            effective_dof = float(shrink.sum())
            if effective_dof > 0.9 * n_rows or float(leverage.max()) > 0.99:
                if fallback is None:
                    fallback = candidate
                continue

            error = float(np.mean(((Yc - fitted) / denominator) ** 2))
            if best is None or error < best[0]:
                best = (error, *candidate)

        if best is None:  # every lambda interpolated; take the strongest penalty
            lam, shrink, leverage, fitted = (
                fallback if fallback is not None else (grid[-1], None, None, None)
            )
            shrink = s2 / (s2 + grid[-1])
            leverage = (U**2) @ shrink
            fitted = U @ (shrink[:, None] * UtY)
            best = (np.inf, grid[-1], shrink, leverage, fitted)

        _, self.lambda_, shrink, leverage, fitted = best
        self.coefficients_ = Vt.T @ ((singular * shrink / np.maximum(s2, _EPS))[:, None] * UtY)

        # The leave-one-out prediction, in closed form: what the model would have
        # said about row i had row i not been in the fit.
        denominator = np.clip(1.0 - leverage, 1e-3, None)[:, None]
        loo = (fitted - leverage[:, None] * Yc) / denominator + self.target_center_
        self.effective_dof_ = float(shrink.sum())

        if self.task != "regression":
            self.calibration_ = [
                _platt(loo[:, k], self._class_indicator(y, k)) for k in range(loo.shape[1])
            ]
        return self._calibrate(loo)

    def _class_indicator(self, y: np.ndarray, k: int) -> np.ndarray:
        """1/0 membership of class ``k`` — what Platt scaling is calibrated against."""
        if self.n_outputs == 1:
            return (y > 0.5).astype(np.float64)
        return (y.astype(int) == k).astype(np.float64)

    def _calibrate(self, margins: np.ndarray) -> np.ndarray:
        if self.task == "regression" or not self.calibration_:
            return margins
        out = np.empty_like(margins)
        for k, (a, b) in enumerate(self.calibration_):
            out[:, k] = a * margins[:, k] + b
        return out

    # -------------------------------------------------------------- inference

    def transform(self, bank: np.ndarray) -> np.ndarray:
        """Full-data-fit margins for unseen rows, ``(n, K)``."""
        if self.coefficients_ is None:
            raise RuntimeError("DenseBase is not fitted")
        X = self._prepare(bank, fitting=False)
        return self._calibrate(X @ self.coefficients_ + self.target_center_)
