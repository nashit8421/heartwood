"""The boosting loop: Newton steps, one shallow temporal tree at a time."""

from __future__ import annotations

import numpy as np

from ._util import spawn_rng
from .losses import Loss
from .tree import TemporalTree, TreeParams


class _BoosterCore:
    """Task-agnostic gradient booster over :class:`TemporalTree` learners.

    Knows nothing about labels or classes — ``api.py`` encodes those and hands
    over an already-normalised problem plus a :class:`~heartwood.losses.Loss`.
    """

    def __init__(self, tree_params: TreeParams, n_estimators=200, learning_rate=0.1,
                 subsample=1.0, early_stopping_rounds=None, random_state=None):
        self.tree_params = tree_params
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.trees_: list[list[TemporalTree]] = []
        self.base_score_: np.ndarray | None = None
        self.n_outputs_ = 1
        self.best_iteration_ = -1
        self.best_score_ = np.inf
        self.train_history_: list[float] = []
        self.eval_history_: list[float] = []

    # ------------------------------------------------------------------ fit

    def fit(self, X_static, X_series, y, loss: Loss, eval_set=None, verbose=False):
        n = X_static.shape[0]
        K = loss.n_outputs(y)
        self.n_outputs_ = K
        self.base_score_ = np.asarray(loss.init_score(y), dtype=np.float64)
        self.trees_ = []
        self.train_history_ = []
        self.eval_history_ = []
        self.best_iteration_ = -1
        self.best_score_ = np.inf

        raw = np.tile(self.base_score_, (n, 1))

        has_eval = eval_set is not None
        if has_eval:
            Xs_val, Xt_val, y_val = eval_set
            raw_val = np.tile(self.base_score_, (Xs_val.shape[0], 1))

        master = np.random.default_rng(self.random_state)
        n_sub = max(1, int(round(self.subsample * n))) if self.subsample < 1.0 else n
        rounds_without_gain = 0

        for m in range(self.n_estimators):
            g, h = loss.grad_hess(y, raw)
            rows = (
                np.arange(n, dtype=np.intp)
                if n_sub >= n
                else np.sort(master.choice(n, size=n_sub, replace=False)).astype(np.intp)
            )

            round_trees = []
            for k in range(K):
                tree = TemporalTree(self.tree_params)
                tree.fit(X_static, X_series, g[:, k], h[:, k], rows, spawn_rng(master))
                raw[:, k] += self.learning_rate * tree.predict(X_static, X_series)
                if has_eval:
                    raw_val[:, k] += self.learning_rate * tree.predict(Xs_val, Xt_val)
                round_trees.append(tree)
            self.trees_.append(round_trees)

            train_score = loss.eval_metric(y, raw)
            self.train_history_.append(train_score)

            if has_eval:
                score = loss.eval_metric(y_val, raw_val)
                self.eval_history_.append(score)
                if score < self.best_score_ - 1e-12:
                    self.best_score_ = score
                    self.best_iteration_ = m
                    rounds_without_gain = 0
                else:
                    rounds_without_gain += 1
            else:
                self.best_iteration_ = m

            if verbose and (m % 10 == 0 or m == self.n_estimators - 1):
                msg = f"[{m}] train-{loss.metric_name}: {train_score:.5f}"
                if has_eval:
                    msg += f"  eval-{loss.metric_name}: {self.eval_history_[-1]:.5f}"
                print(msg)

            if (
                has_eval
                and self.early_stopping_rounds
                and rounds_without_gain >= self.early_stopping_rounds
            ):
                if verbose:
                    print(
                        f"early stopping at round {m}; best iteration "
                        f"{self.best_iteration_} ({loss.metric_name}={self.best_score_:.5f})"
                    )
                break

        return self

    # -------------------------------------------------------------- predict

    def predict_raw(self, X_static, X_series, iteration: int | None = None) -> np.ndarray:
        """Raw scores ``(n, K)`` using rounds up to and including ``iteration``.

        Defaults to the best iteration found on the eval set, so a model fitted
        with early stopping predicts with the rounds that actually generalised.
        """
        if self.base_score_ is None:
            raise RuntimeError("booster is not fitted")

        if iteration is None:
            iteration = (
                self.best_iteration_ if self.best_iteration_ >= 0 else len(self.trees_) - 1
            )
        n_rounds = int(np.clip(iteration + 1, 0, len(self.trees_)))

        n = X_static.shape[0]
        raw = np.tile(self.base_score_, (n, 1))
        for round_trees in self.trees_[:n_rounds]:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(X_static, X_series)
        return raw

    def n_rounds_used(self) -> int:
        if self.best_iteration_ >= 0:
            return min(self.best_iteration_ + 1, len(self.trees_))
        return len(self.trees_)

    def iter_splits(self):
        """Splits of the rounds that are actually used at predict time."""
        for round_trees in self.trees_[: self.n_rounds_used()]:
            for tree in round_trees:
                yield from tree.iter_splits()
