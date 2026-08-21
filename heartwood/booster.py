"""The boosting loop: Newton steps, one shallow temporal tree at a time."""

from __future__ import annotations

import numpy as np

from ._util import spawn_rng
from .bank import FeatureBank
from .dense import DenseBase, dense_bank, levy_area_columns
from .rocket import RocketBank
from .filters import Pyramid
from .losses import Loss
from .tree import FitContext, TemporalTree, TreeParams


class _BoosterCore:
    """Task-agnostic gradient booster over :class:`TemporalTree` learners.

    Knows nothing about labels or classes — ``api.py`` encodes those and hands
    over an already-normalised problem plus a :class:`~heartwood.losses.Loss`.
    """

    def __init__(self, tree_params: TreeParams, n_estimators=200, learning_rate=0.1,
                 subsample=1.0, early_stopping_rounds=None, random_state=None,
                 bank_enabled=True, bank_max=32, dense_base=False, levy_areas=False,
                 dense_features="stats", n_rocket_features=10000):
        self.tree_params = tree_params
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.bank_enabled = bool(bank_enabled)
        self.bank_max = int(bank_max)
        self.bank: FeatureBank | None = None
        self.dense_base = bool(dense_base)
        if dense_features not in ("stats", "rocket", "both"):
            raise ValueError(
                f"dense_features must be 'stats', 'rocket' or 'both', got {dense_features!r}"
            )
        self.dense_features = dense_features
        self.n_rocket_features = int(n_rocket_features)
        self.rocket_: RocketBank | None = None
        self.levy_areas = bool(levy_areas)
        self.dense_: DenseBase | None = None
        self.static_names_: list[str] = []

        self.trees_: list[list[TemporalTree]] = []
        self.base_score_: np.ndarray | None = None
        self.n_outputs_ = 1
        self.best_iteration_ = -1
        self.best_score_ = np.inf
        self.train_history_: list[float] = []
        self.eval_history_: list[float] = []

    def _pyramid(self, X_series):
        """Pooled copies of a batch's series, built once and shared by every tree."""
        if X_series is None or self.tree_params.n_filter_candidates <= 0:
            return None
        return Pyramid(X_series, self.tree_params.filter_len)

    def _augment(self, X_static, X_series, y=None, loss=None):
        """Attach the optional dense columns, identically at fit and predict time.

        Returns ``(X_static_augmented, base_raw)``.  ``base_raw`` is the starting
        point for boosting: leave-one-out ridge margins while fitting, full-fit
        margins afterwards.  Using the honest out-of-fold value during training is
        the difference between the trees learning what the ridge could not and
        the trees learning nothing at all.
        """
        fitting = y is not None
        blocks, names = [X_static], [f"static[{j}]" for j in range(X_static.shape[1])]
        base_raw = None

        if self.levy_areas and X_series is not None:
            areas = levy_area_columns(X_series)
            if areas.shape[1]:
                blocks.append(areas)
                names += [f"levy_area[{i}]" for i in range(areas.shape[1])]

        if self.dense_base and X_series is not None:
            bank = self._dense_bank(X_series, fitting)
            if fitting:
                self.dense_ = DenseBase(loss.task, loss.n_outputs(y))
                base_raw = self.dense_.fit(bank, y)
            elif self.dense_ is not None:
                base_raw = self.dense_.transform(bank)
            if base_raw is not None:
                blocks.append(base_raw)
                names += [f"dense_margin[{k}]" for k in range(base_raw.shape[1])]

        if fitting:
            self.static_names_ = names
        return (np.hstack(blocks) if len(blocks) > 1 else X_static), base_raw

    def _dense_bank(self, X_series, fitting: bool) -> np.ndarray:
        """The feature bank the ridge base sees.

        ``stats`` is the original dyadic window-statistic bank.  ``rocket`` is the
        dilated-convolution bank, which exists because greedy per-node selection
        is the measured ceiling on shape-regime data and a ridge over a large
        fixed bank does not select at all (``validation/HEADROOM.md``).  ``both``
        concatenates them and lets the ridge decide.

        The rocket bank is stateful — its biases are quantiles of the *training*
        convolutions — so it is fitted once and reused unchanged afterwards, the
        same discipline as the frozen rank grids in ``features.ecdf``.
        """
        parts = []
        if self.dense_features in ("stats", "both"):
            parts.append(dense_bank(X_series))
        if self.dense_features in ("rocket", "both"):
            if fitting:
                self.rocket_ = RocketBank(
                    n_features=self.n_rocket_features,
                    random_state=0 if self.random_state is None else self.random_state,
                ).fit(X_series)
            if self.rocket_ is None:
                raise RuntimeError("rocket bank is not fitted")
            parts.append(self.rocket_.transform(X_series))
        return parts[0] if len(parts) == 1 else np.hstack(parts)

    @staticmethod
    def _static_grids(X_static):
        """Frozen sorted training columns, so a rank means the same thing later."""
        return [
            np.sort(X_static[np.isfinite(X_static[:, j]), j])
            for j in range(X_static.shape[1])
        ]

    # ------------------------------------------------------------------ fit

    def fit(self, X_static, X_series, y, loss: Loss, eval_set=None, verbose=False):
        n = X_static.shape[0]
        K = loss.n_outputs(y)
        X_static, base_raw = self._augment(X_static, X_series, y=y, loss=loss)
        self.n_outputs_ = K
        self.base_score_ = np.asarray(loss.init_score(y), dtype=np.float64)
        self.trees_ = []
        self.train_history_ = []
        self.eval_history_ = []
        self.best_iteration_ = -1
        self.best_score_ = np.inf

        raw = base_raw.copy() if base_raw is not None else np.tile(self.base_score_, (n, 1))

        pyramid = self._pyramid(X_series)
        self.bank = FeatureBank(self.bank_max) if self.bank_enabled else None
        context = FitContext(
            pyramid=pyramid, bank=self.bank,
            static_grids=self._static_grids(X_static),
            static_names=self.static_names_,
        )

        has_eval = eval_set is not None
        if has_eval:
            Xs_val, Xt_val, y_val = eval_set
            Xs_val, base_val = self._augment(Xs_val, Xt_val)
            raw_val = (
                base_val.copy() if base_val is not None
                else np.tile(self.base_score_, (Xs_val.shape[0], 1))
            )
            pyramid_val = self._pyramid(Xt_val)

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
            context.round_index = m

            round_trees = []
            for k in range(K):
                tree = TemporalTree(self.tree_params)
                tree.fit(
                    X_static, X_series, g[:, k], h[:, k], rows, spawn_rng(master), context
                )
                raw[:, k] += self.learning_rate * tree.predict(X_static, X_series, pyramid)
                if has_eval:
                    raw_val[:, k] += self.learning_rate * tree.predict(
                        Xs_val, Xt_val, pyramid_val
                    )
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

        X_static, base_raw = self._augment(X_static, X_series)
        n = X_static.shape[0]
        raw = base_raw.copy() if base_raw is not None else np.tile(self.base_score_, (n, 1))
        pyramid = self._pyramid(X_series)
        for round_trees in self.trees_[:n_rounds]:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(X_static, X_series, pyramid)
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
