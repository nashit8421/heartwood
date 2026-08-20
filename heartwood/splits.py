"""Split representation, the exact threshold scan, and candidate samplers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MIN_GAIN = 1e-12
_EPS_DEN = 1e-16
_EPS_SD = 1e-12


@dataclass
class SplitSpec:
    """A single learned split: what to compute, and where to send each row.

    The spec is self-contained — it carries its own copy of any shapelet — so a
    fitted tree never depends on the training arrays staying alive or unchanged.
    """

    kind: str  # 'static' | 'interval' | 'shapelet_dist' | 'shapelet_pos'
    threshold: float = np.nan
    missing_left: bool = True
    gain: float = -np.inf
    # static
    col: int = -1
    # interval (and the channel for shapelets)
    channel: int = -1
    start: int = -1
    end: int = -1
    stat: str = ""
    # shapelet
    shapelet: np.ndarray | None = field(default=None, repr=False)
    znorm: bool = True

    def feature_name(self) -> str:
        """Human-readable name of the scalar this split thresholds."""
        if self.kind == "static":
            return f"static[{self.col}]"
        if self.kind == "interval":
            return f"series[ch={self.channel}].{self.stat}[t={self.start}:{self.end}]"
        length = 0 if self.shapelet is None else len(self.shapelet)
        which = "dist" if self.kind == "shapelet_dist" else "pos"
        return f"series[ch={self.channel}].shapelet_{which}(len={length})"

    def family(self) -> str:
        """Coarser key used to aggregate importances."""
        if self.kind == "static":
            return f"static[{self.col}]"
        if self.kind == "interval":
            return f"interval(ch={self.channel}, {self.stat})"
        return f"{self.kind}(ch={self.channel})"

    def describe(self) -> str:
        side = "left" if self.missing_left else "right"
        return f"{self.feature_name()} <= {self.threshold:.4g}  [missing->{side}]"


def scan_threshold(
    f: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
):
    """Best threshold on feature ``f`` under the XGBoost gain criterion.

    Returns ``(gain, threshold, missing_left)`` or ``None`` when no split
    satisfies the constraints.  Rows whose feature value is not finite form a
    "missing" group that is tried on both sides, keeping whichever scores
    higher — the sparsity-aware trick, which is also how a row with no series
    data coexists with fully observed rows in the same model.
    """
    n_total = f.size
    finite = np.isfinite(f)
    n_missing = int(n_total - finite.sum())

    if n_missing:
        miss = ~finite
        g_miss = float(g[miss].sum())
        h_miss = float(h[miss].sum())
    else:
        g_miss = h_miss = 0.0

    fv = f[finite]
    if fv.size < 2:
        return None

    order = np.argsort(fv, kind="stable")
    fs = fv[order]
    cg = np.cumsum(g[finite][order])
    ch = np.cumsum(h[finite][order])

    # Only cut strictly between distinct values: splitting a tie is not a split.
    cut = np.nonzero(fs[:-1] < fs[1:])[0]
    if cut.size == 0:
        return None

    G = float(cg[-1]) + g_miss
    H = float(ch[-1]) + h_miss
    parent = G * G / max(H + reg_lambda, _EPS_DEN)

    lo, hi = fs[cut], fs[cut + 1]
    mid = 0.5 * (lo + hi)
    # Guard the rounding case where the midpoint lands on the upper value, which
    # would route that row differently than the gain computation assumed.
    thresholds = np.where(mid >= hi, lo, mid)

    best = None
    scenarios = (True, False) if n_missing else (True,)
    for missing_left in scenarios:
        GL = cg[cut] + (g_miss if missing_left else 0.0)
        HL = ch[cut] + (h_miss if missing_left else 0.0)
        nL = (cut + 1) + (n_missing if missing_left else 0)
        GR = G - GL
        HR = H - HL
        nR = n_total - nL

        ok = (
            (HL >= min_child_weight)
            & (HR >= min_child_weight)
            & (nL >= min_samples_leaf)
            & (nR >= min_samples_leaf)
        )
        if not ok.any():
            continue

        gains = 0.5 * (
            GL * GL / np.maximum(HL + reg_lambda, _EPS_DEN)
            + GR * GR / np.maximum(HR + reg_lambda, _EPS_DEN)
            - parent
        ) - gamma
        gains = np.where(ok, gains, -np.inf)

        i = int(np.argmax(gains))
        if best is None or gains[i] > best[0]:
            best = (float(gains[i]), float(thresholds[i]), missing_left)

    if best is None or best[0] <= _MIN_GAIN:
        return None
    return best


def sample_interval(T: int, rng: np.random.Generator, min_len: int, full_prob: float):
    """Sample a window ``(start, end)``.

    Lengths are log-uniform so short and long windows get comparable attention,
    and with probability ``full_prob`` the whole series is proposed — which is
    what keeps the classical global aggregate permanently inside the hypothesis
    space, so the model can never be *less* expressive than the aggregate-and-
    concatenate baseline it is meant to beat.
    """
    if T <= min_len:
        return 0, T
    if rng.random() < full_prob:
        return 0, T
    lo, hi = np.log(min_len), np.log(T)
    length = int(round(float(np.exp(rng.uniform(lo, hi)))))
    length = int(np.clip(length, min_len, T))
    start = int(rng.integers(0, T - length + 1))
    return start, start + length


def sample_shapelet(
    X_series: np.ndarray,
    rows: np.ndarray,
    rng: np.random.Generator,
    min_len: int,
    max_frac: float,
    max_tries: int = 10,
):
    """Cut a candidate shapelet out of a series belonging to the current node.

    Returns ``(channel, shapelet)`` or ``None``.  Non-finite or constant cuts are
    rejected and retried: a constant template has no shape to match.
    """
    n, C, T = X_series.shape
    if T < min_len + 1 or rows.size == 0:
        return None

    max_len = min(max(min_len + 1, int(max_frac * T)), T)
    if max_len < min_len:
        return None

    log_lo, log_hi = np.log(min_len), np.log(max_len)
    for _ in range(max_tries):
        i = int(rows[rng.integers(rows.size)])
        c = int(rng.integers(C))
        length = int(round(float(np.exp(rng.uniform(log_lo, log_hi)))))
        length = int(np.clip(length, min_len, max_len))
        start = int(rng.integers(0, T - length + 1))
        shp = X_series[i, c, start : start + length]
        if not np.isfinite(shp).all():
            continue
        if float(shp.std()) < _EPS_SD:
            continue
        return c, shp.copy()
    return None
