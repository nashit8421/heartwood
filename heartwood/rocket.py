"""A dilated-convolution feature bank, in the MiniROCKET style.

Why this exists, in one paragraph.  Heartwood picks the single highest-gain split
from a pool of randomly drawn candidates, which means a bigger pool raises the
winner's expected gain whether or not anything in it is informative — measured in
``validation/HEADROOM.md``, where a 16x candidate budget moved a 20-point deficit
by 1.5 points and made two of three datasets worse.  Selection is the ceiling.
MiniROCKET has no such ceiling because it never selects: it computes a large
fixed bank and lets a ridge shrink all of it jointly.  This module builds that
bank so :class:`~heartwood.dense.DenseBase` can do the same underneath the trees.

The construction follows Dempster, Schmidt and Webb (2021):

* 84 kernels of length 9 — every way to choose 3 of the 9 weights to be ``2``
  with the remaining 6 at ``-1``, so every kernel sums to zero.
* Exponentially spaced dilations, capped so the dilated kernel still fits.
* Per (kernel, dilation) biases drawn from quantiles of that kernel's own
  convolution output on a training row, so thresholds sit where the data is.
* Pooling by *proportion of positive values* — the fraction of timesteps whose
  convolution exceeds the bias.

Two deliberate departures, both because this has to hold Heartwood's contracts
rather than reproduce a paper:

* **Multivariate handling goes through channel groups.**  A group is a subset of
  channels summed into one virtual channel.  Every singleton is included, so
  per-channel structure is never lost, plus random subsets of size 2, 4, ... so
  joint structure is available too — Handwriting's signal is the joint x-y pen
  trajectory rather than anything in either axis alone, and per-channel-only cost
  11 points there.  Crucially the channel choice is folded *into* the feature
  budget rather than multiplying it: each kernel is assigned one group per
  dilation.  Multiplying the budget by the group count starves every (kernel,
  dilation) down to a single bias, which cost another 5 points.
* **NaN is imputed per row and channel** before convolving, because a
  convolution has no NaN-aware form.  Rows that are entirely missing in a channel
  contribute a constant, and their features fall out in the ridge's centring.
  Everywhere else in this library NaN is load-bearing; here it cannot be, and
  saying so is better than pretending the bank is exact.

The one number to hold onto: this is a *fixed, label-free* bank.  Nothing here
looks at ``y``.  That is what lets the ridge on top of it be honest.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

KERNEL_LEN = 9
#: Every way to pick which 3 of 9 weights are +2; the other 6 are -1.
ALPHA_INDICES = np.array(list(combinations(range(KERNEL_LEN), 3)), dtype=np.intp)
N_KERNELS = len(ALPHA_INDICES)  # 84
DEFAULT_FEATURES = 10_000
_EPS = 1e-12


def _dilations(length: int, n_features_per_kernel: int, max_dilations: int = 32):
    """Exponentially spaced dilations, and how many biases each one gets.

    Spacing runs to the largest dilation whose length-9 kernel still fits inside
    the series, so no feature is computed from padding alone.
    """
    if length <= KERNEL_LEN:
        return np.array([1], dtype=np.intp), np.array([n_features_per_kernel], dtype=np.intp)

    n_dilations = max(1, min(n_features_per_kernel, max_dilations))
    max_exponent = np.log2((length - 1) / (KERNEL_LEN - 1))
    dilations, multiplicity = np.unique(
        np.logspace(0, max_exponent, n_dilations, base=2).astype(np.intp), return_counts=True
    )

    share = multiplicity / multiplicity.sum()
    per_dilation = np.maximum(1, (share * n_features_per_kernel).astype(np.intp))
    # hand out (or claw back) the rounding remainder one at a time
    index = 0
    while per_dilation.sum() != n_features_per_kernel:
        step = 1 if per_dilation.sum() < n_features_per_kernel else -1
        if step == -1 and per_dilation[index] <= 1:
            index = (index + 1) % len(per_dilation)
            continue
        per_dilation[index] += step
        index = (index + 1) % len(per_dilation)
    return dilations, per_dilation


def _channel_groups(n_channels: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Singletons, plus random subsets of size 2, 4, ... summed together.

    Singletons are always present so no per-channel structure is lost; the
    subsets exist because some signals live only in the joint trajectory.  Sizes
    are powers of two, following MiniROCKET's channel-combination scheme.
    """
    if n_channels == 1:
        return [np.array([0], dtype=np.intp)]
    groups = [np.array([c], dtype=np.intp) for c in range(n_channels)]
    max_exponent = int(np.log2(min(n_channels, 9)))
    for _ in range(n_channels):
        size = int(2 ** rng.integers(1, max_exponent + 1)) if max_exponent >= 1 else 1
        groups.append(rng.choice(n_channels, size=min(size, n_channels), replace=False))
    return groups


def _virtual_channels(X_series: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Sum each channel group into one series, imputing NaN first."""
    out = np.empty((len(X_series), len(groups), X_series.shape[2]), dtype=np.float64)
    for index, group in enumerate(groups):
        out[:, index, :] = sum(_impute(X_series[:, c, :]) for c in group)
    return out


def _impute(block: np.ndarray) -> np.ndarray:
    """Replace NaN with each row's own mean for this channel, or 0 if all-NaN."""
    finite = np.isfinite(block)
    if finite.all():
        return block
    counts = finite.sum(axis=1)
    totals = np.where(finite, block, 0.0).sum(axis=1)
    means = np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
    return np.where(finite, block, means[:, None])


def _shifted(block: np.ndarray, dilation: int, pad: bool) -> list[np.ndarray] | None:
    """The nine dilated taps of a length-9 kernel, as views into one array.

    Slicing gives views rather than copies, which is what keeps this affordable:
    the bank never materialises more than one ``(n, T)`` temporary at a time.

    ``pad`` selects zero-padded ("same") or valid-only taps, and the caller
    alternates between them.  It matters more than it looks: at dilation 18 on a
    152-step series, padding appends 144 fabricated zeros, so nearly every window
    straddles a boundary that is not in the data.  Padding everywhere cost 15
    points on Handwriting.  Returns ``None`` when the unpadded kernel does not
    fit at all.
    """
    length = block.shape[1]
    span = (KERNEL_LEN - 1) * dilation
    if pad:
        margin = span // 2
        padded = np.pad(block, ((0, 0), (margin, span - margin)))
        return [padded[:, j * dilation: j * dilation + length] for j in range(KERNEL_LEN)]
    valid = length - span
    if valid < 1:
        return None
    return [block[:, j * dilation: j * dilation + valid] for j in range(KERNEL_LEN)]


class RocketBank:
    """Fixed dilated-convolution features; fit once on train, applied unchanged after.

    ``fit_transform`` draws the biases from the given rows and remembers them.
    ``transform`` reuses those biases, so a test row is measured against the same
    thresholds a training row was — the same discipline as the frozen rank grids
    in ``features.ecdf``.
    """

    def __init__(self, n_features: int = DEFAULT_FEATURES, random_state: int = 0,
                 max_dilations: int = 32):
        self.n_features = int(n_features)
        self.random_state = int(random_state)
        self.max_dilations = int(max_dilations)
        self.plan_: list[tuple[int, int, np.ndarray]] | None = None  # group, dilation, biases
        self.groups_: list[np.ndarray] | None = None
        self.n_channels_: int | None = None
        self.length_: int | None = None

    # ------------------------------------------------------------------ plan

    def _budget(self) -> int:
        """Features per kernel, at least one.

        Not divided by the channel-group count: a group is assigned per kernel
        per dilation, so groups add diversity without consuming budget.
        """
        return max(1, self.n_features // N_KERNELS)

    def fit(self, X_series: np.ndarray) -> "RocketBank":
        rng = np.random.default_rng(self.random_state)
        n, n_channels, length = X_series.shape
        self.n_channels_, self.length_ = n_channels, length
        self.groups_ = _channel_groups(n_channels, rng)
        virtual = _virtual_channels(X_series, self.groups_)
        per_kernel = self._budget()
        dilations, per_dilation = _dilations(length, per_kernel, self.max_dilations)

        plan: list[tuple[np.ndarray, int, np.ndarray]] = []
        for dilation, count in zip(dilations, per_dilation):
            quantiles = np.linspace(0, 1, int(count) + 2)[1:-1]
            biases = np.full((N_KERNELS, int(count)), np.nan)
            assignment = rng.integers(0, len(self.groups_), size=N_KERNELS)
            donors = rng.integers(0, n, size=N_KERNELS)
            cache: dict[tuple[int, bool], list[np.ndarray] | None] = {}
            for k, alpha in enumerate(ALPHA_INDICES):
                pad = bool(k % 2 == 0)
                key = (int(assignment[k]), pad)
                if key not in cache:
                    cache[key] = _shifted(virtual[:, key[0], :], int(dilation), pad)
                taps = cache[key]
                if taps is None:  # unpadded kernel does not fit at this dilation
                    continue
                row = donors[k]
                response = (
                    3.0 * (taps[alpha[0]][row] + taps[alpha[1]][row] + taps[alpha[2]][row])
                    - sum(tap[row] for tap in taps)
                )
                biases[k] = np.quantile(response, quantiles)
            plan.append((assignment, int(dilation), biases))
        self.plan_ = plan
        return self

    # ------------------------------------------------------------- transform

    def transform(self, X_series: np.ndarray) -> np.ndarray:
        if self.plan_ is None:
            raise RuntimeError("RocketBank is not fitted")
        n, n_channels, length = X_series.shape
        if n_channels != self.n_channels_ or length != self.length_:
            raise ValueError(
                f"expected series of shape (*, {self.n_channels_}, {self.length_}), "
                f"got (*, {n_channels}, {length})"
            )

        columns: list[np.ndarray] = []
        virtual = _virtual_channels(X_series, self.groups_)
        for assignment, dilation, biases in self.plan_:
            cache: dict[tuple[int, bool], tuple[list[np.ndarray], np.ndarray] | None] = {}
            for k, alpha in enumerate(ALPHA_INDICES):
                if not np.isfinite(biases[k]).any():
                    continue  # this kernel had no usable window at this dilation
                key = (int(assignment[k]), bool(k % 2 == 0))
                if key not in cache:
                    taps = _shifted(virtual[:, key[0], :], dilation, key[1])
                    cache[key] = (taps, sum(taps)) if taps is not None else None
                entry = cache[key]
                if entry is None:
                    continue
                taps, total = entry
                response = 3.0 * (taps[alpha[0]] + taps[alpha[1]] + taps[alpha[2]]) - total
                for bias in biases[k]:
                    # proportion of positive values
                    columns.append((response > bias).mean(axis=1))
        return np.column_stack(columns) if columns else np.empty((n, 0))

    def fit_transform(self, X_series: np.ndarray) -> np.ndarray:
        return self.fit(X_series).transform(X_series)

    @property
    def n_output_features(self) -> int:
        if self.plan_ is None:
            return 0
        return sum(int(np.isfinite(biases).any(axis=1).sum()) * biases.shape[1]
                   for _, _, biases in self.plan_)
