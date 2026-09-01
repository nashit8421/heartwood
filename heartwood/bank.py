"""A cache of temporal features that have already proved themselves.

Without it, every node re-runs the same lottery: draw windows and templates at
random and hope one is discriminative.  A feature that won a split in round 3
has to be rediscovered from scratch in round 4, and usually is not.

The bank keeps the winners.  When a temporal candidate wins a node, its column
is materialised over all training rows and offered — free, no sampling — at
every later node in every later round.  Discovery stays random; *keeping* what
was discovered does not.

Deliberately kept to winners only.  The design review's verdict on the richer
version (saliency maps, mutation, bandit credit) was that its bookkeeping was
the largest silent-corruption surface on the table, so none of that is here.

This is strictly a training-time cache: every tree split stores its own
self-contained spec, so evicting an entry can never change a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .splits import SplitSpec

_EPS = 1e-12


def _abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """|Pearson r| over the rows where both are observed."""
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 3:
        return 0.0
    x, y = a[both], b[both]
    x_spread, y_spread = float(x.std()), float(y.std())
    if x_spread <= _EPS or y_spread <= _EPS:
        return 0.0
    return abs(float(np.mean((x - x.mean()) * (y - y.mean())) / (x_spread * y_spread)))


@dataclass(eq=False)  # entries hold arrays; identity is the only sane equality
class BankEntry:
    spec: SplitSpec
    column: np.ndarray
    cumulative_gain: float = 0.0
    last_win_round: int = 0
    born_round: int = 0
    grid: np.ndarray | None = field(default=None, repr=False)

    def fresh_spec(self) -> SplitSpec:
        """A copy for the tree to stamp a threshold onto, leaving ours pristine."""
        return replace(self.spec)


class FeatureBank:
    """Winners-only store of temporal feature columns."""

    def __init__(self, max_entries: int = 128, correlation_cutoff: float = 0.995):
        self.max_entries = int(max_entries)
        self.correlation_cutoff = float(correlation_cutoff)
        self.entries: list[BankEntry] = []
        self._by_identity: dict[tuple, BankEntry] = {}
        self.n_promoted = 0
        self.n_evicted = 0
        self.n_rejected_duplicate = 0
        #: Top-k survivors of the most recent out-of-fold screen (V18), or None
        #: when no screen is in force.  Never persisted: a screen is a per-round
        #: view of the bank, not a change to what the bank stores.
        self.screened: list[BankEntry] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def candidates(self, rows: np.ndarray, rng=None, fraction: float = 1.0):
        """Yield ``(values, spec)`` for banked features, restricted to rows.

        ``fraction`` subsamples the bank per node, and it matters more than it
        looks: every candidate a node scans is another chance for noise to win
        the best-gain contest, so offering a large bank in full makes the
        multiple-comparison problem worse than the reuse is worth.  This is the
        same reasoning behind column subsampling in ordinary boosting.
        """
        if not self.entries:
            return
        if self.screened is not None:
            # A screen replaces the random subsample rather than composing with
            # it: the point of ranking the bank is to stop offering it blind, and
            # thinning the ranked list at random again would undo exactly that.
            for entry in self.screened:
                yield entry.column[rows], entry.fresh_spec()
            return
        chosen = self.entries
        if rng is not None and 0.0 < fraction < 1.0:
            k = max(1, int(round(fraction * len(self.entries))))
            if k < len(self.entries):
                picks = rng.choice(len(self.entries), size=k, replace=False)
                chosen = [self.entries[i] for i in picks]
        for entry in chosen:
            yield entry.column[rows], entry.fresh_spec()

    def screen(self, rows: np.ndarray, target: np.ndarray, top_k: int) -> None:
        """Keep only the ``top_k`` entries most associated with ``target``.

        Roadmap item 2c.  A node currently chooses among banked features by
        maximum gain on its *own* rows, which is the winner's curse in its purest
        form: the bank is offered blind and the luckiest column wins.  Ranking it
        first, on rows the trees of this round will not see, replaces that
        lottery with a shortlist.

        ``rows`` must be held out from the fit that follows, and enforcing that
        is the caller's job (``_BoosterCore.fit`` rotates the fold every round so
        no data is permanently spent).  Screening on the fitting rows would rank
        the bank by the same signal the split then exploits, which is the
        selection step this is meant to remove, wearing a shortlist's clothes.
        """
        if not self.entries:
            self.screened = None
            return
        ranked = sorted(
            self.entries,
            key=lambda e: _abs_correlation(e.column[rows], target),
            reverse=True,
        )
        self.screened = ranked[: max(1, int(top_k))]

    def clear_screen(self) -> None:
        self.screened = None


    def promote(self, spec: SplitSpec, column: np.ndarray, gain: float, round_index: int) -> bool:
        """Offer a winning feature to the bank; True if it was newly stored."""
        key = spec.identity()
        existing = self._by_identity.get(key)
        if existing is not None:
            existing.cumulative_gain += float(gain)
            existing.last_win_round = round_index
            return False

        if not np.isfinite(column).any():
            return False
        for entry in self.entries:
            if _abs_correlation(column, entry.column) > self.correlation_cutoff:
                entry.cumulative_gain += float(gain)
                entry.last_win_round = round_index
                self.n_rejected_duplicate += 1
                return False

        stored = replace(spec, threshold=np.nan, gain=-np.inf, missing_left=True)
        finite = column[np.isfinite(column)]
        entry = BankEntry(
            spec=stored,
            column=column,
            cumulative_gain=float(gain),
            last_win_round=round_index,
            born_round=round_index,
            grid=np.sort(finite) if stored.is_position() and finite.size else None,
        )
        self.entries.append(entry)
        self._by_identity[key] = entry
        self.n_promoted += 1
        self._evict_if_full()
        return True

    def _evict_if_full(self) -> None:
        while len(self.entries) > self.max_entries:
            victim = min(self.entries, key=lambda e: (e.cumulative_gain, e.last_win_round))
            # by identity, never by equality: these hold numpy arrays
            self.entries = [entry for entry in self.entries if entry is not victim]
            self._by_identity.pop(victim.spec.identity(), None)
            self.n_evicted += 1

    def summary(self) -> list[tuple[str, float, int]]:
        """Readable dictionary of what the model found, best first."""
        return sorted(
            ((e.spec.feature_name(), e.cumulative_gain, e.born_round) for e in self.entries),
            key=lambda row: row[1],
            reverse=True,
        )
