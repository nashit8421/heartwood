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
                 dense_features="stats", n_rocket_features=10000,
                 rocket_channel_groups="subsets",
                 dense_include_static=False, dense_static_interactions=False,
                 screen_fraction=0.0, screen_top_k=8,
                 nonlinear_features=0, nonlinear_gamma=1.0,
                 base_static_products=False):
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
        self.rocket_channel_groups = str(rocket_channel_groups)
        self.dense_include_static = bool(dense_include_static)
        self.dense_static_interactions = bool(dense_static_interactions)
        self.rocket_: RocketBank | None = None
        self.levy_areas = bool(levy_areas)
        self.dense_: DenseBase | None = None
        self.screen_fraction = float(screen_fraction)
        self.screen_top_k = int(screen_top_k)
        self.nonlinear_features = int(nonlinear_features)
        self.nonlinear_gamma = float(nonlinear_gamma)
        self.base_static_products = bool(base_static_products)
        self.product_impute_: np.ndarray | None = None
        self.product_bounds_: tuple[np.ndarray, np.ndarray] | None = None
        self.product_center_: np.ndarray | None = None
        self.product_scale_: np.ndarray | None = None
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

    def _augment(self, X_static, X_series, y=None, loss=None, groups=None):
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
            # V10: the static block joins the base, unpenalised. Without this the
            # linear layer never sees it and the statics can only reach the model
            # through greedy tree splits -- which on Apnea-ECG left a combination
            # scoring below the static block on its own.
            statics = X_static if self.dense_include_static else None
            if fitting:
                self.dense_ = DenseBase(
                    loss.task, loss.n_outputs(y),
                    use_static=self.dense_include_static,
                    static_interactions=self.dense_static_interactions,
                    nonlinear_features=self.nonlinear_features,
                    nonlinear_gamma=self.nonlinear_gamma,
                )
                base_raw = self.dense_.fit(bank, y, static=statics, groups=groups)
            elif self.dense_ is not None:
                base_raw = self.dense_.transform(bank, static=statics)
            if base_raw is not None:
                blocks.append(base_raw)
                names += [f"dense_margin[{k}]" for k in range(base_raw.shape[1])]

                if self.base_static_products and X_static.shape[1]:
                    products = self._base_static_products(X_static, base_raw, fitting)
                    if products.shape[1]:
                        blocks.append(products)
                        names += [
                            f"margin[{k}]*static[{j}]"
                            for k in range(base_raw.shape[1])
                            for j in range(X_static.shape[1])
                        ]

        if fitting:
            self.static_names_ = names
        return (np.hstack(blocks) if len(blocks) > 1 else X_static), base_raw

    def _base_static_products(self, X_static, base_raw, fitting: bool) -> np.ndarray:
        """``base margin x static``, in magnitudes, offered to the trees.

        Roadmap item 5.  V12 diagnosed ``amp_regression`` (-11.3) exactly: the
        target is ``transient_height * static_coefficient``, a product of
        *magnitudes*, and the only products the model could see were products of
        *ranks*, which discard the magnitudes the target is made of.  Worse, the
        rank products were static-by-static, so the cross the target is built
        from -- series by static -- had no representation anywhere.

        **Why this does not reinstate the V11 blow-up.** V11's failure was
        specific to the *linear* layer: an unpenalised product column grows
        quadratically, so a held-out subject outside the training range produced
        an exploding term and Apnea-ECG fell to 0.478 AUC, below chance.  These
        products are features for the *trees*, and a tree cannot extrapolate --
        its output is a leaf value however large the input gets.  The failure
        mode needs a linear extrapolation to exist and there is not one here.

        Belt as well as braces: the statics are clipped to their training range
        before multiplying, so the feature is bounded by construction too and
        does not rely on that argument being right.

        The margins used while fitting are the base's *out-of-fold* ones -- the
        same values the trees are already boosted from -- so a product column is
        no more able to see its own row's label than the margin it is built from.
        """
        if fitting:
            finite = np.isfinite(X_static)
            with np.errstate(invalid="ignore"):
                median = np.nanmedian(np.where(finite, X_static, np.nan), axis=0)
            self.product_impute_ = np.nan_to_num(median)
            filled = np.where(finite, X_static, self.product_impute_)
            self.product_bounds_ = (filled.min(axis=0), filled.max(axis=0))
            self.product_center_ = filled.mean(axis=0)
            spread = filled.std(axis=0)
            self.product_scale_ = np.where(spread > 1e-12, spread, 1.0)
        if self.product_bounds_ is None:
            raise RuntimeError("base-static products are not fitted")

        filled = np.where(np.isfinite(X_static), X_static, self.product_impute_)
        bounded = np.clip(filled, *self.product_bounds_)
        standard = (bounded - self.product_center_) / self.product_scale_
        return (base_raw[:, :, None] * standard[:, None, :]).reshape(len(X_static), -1)

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
                    channel_groups=self.rocket_channel_groups,
                ).fit(X_series)
            if self.rocket_ is None:
                raise RuntimeError("rocket bank is not fitted")
            parts.append(self.rocket_.transform(X_series))
        return parts[0] if len(parts) == 1 else np.hstack(parts)

    def _screen_bank(self, rows, g, h, master):
        """Rank the bank out of fold, and return the rows the tree may fit on.

        Roadmap item 2c.  The fold rotates every round and every output, so the
        screen is always out of sample for the tree that uses it and no row is
        permanently spent -- the cost is that any single tree sees a fraction
        fewer rows, not that the model does.

        Returns ``rows`` unchanged, and leaves the bank unscreened, whenever
        screening is off or the bank has nothing in it yet.
        """
        bank = self.bank
        if bank is None:
            return rows
        if self.screen_fraction <= 0.0 or not len(bank):
            bank.clear_screen()
            return rows

        n_screen = int(round(self.screen_fraction * rows.size))
        # Leave enough rows to grow a tree on; a screen that starves the fit is
        # measuring the smaller training set, not the shortlist.
        if n_screen < 1 or rows.size - n_screen < 2 * self.tree_params.min_samples_leaf:
            bank.clear_screen()
            return rows

        order = master.permutation(rows.size)
        screen_rows = rows[order[:n_screen]]
        fit_rows = np.sort(rows[order[n_screen:]]).astype(np.intp)

        # Newton residuals: what this round still gets wrong, which is what the
        # tree is about to fit -- not y, which it is not.
        residual = -g[screen_rows] / (h[screen_rows] + self.tree_params.reg_lambda)
        bank.screen(screen_rows, residual, self.screen_top_k)
        return fit_rows

    @staticmethod
    def _static_stats(X_static):
        """Per-column centre, spread and observed range, for product splits."""
        if not X_static.shape[1]:
            return None
        finite = np.isfinite(X_static)
        with np.errstate(invalid="ignore"):
            filled = np.where(finite, X_static, 0.0)
        center = np.where(finite.any(axis=0), filled.sum(axis=0)
                          / np.maximum(finite.sum(axis=0), 1), 0.0)
        deviation = np.where(finite, X_static - center, 0.0)
        scale = np.sqrt(deviation.__pow__(2).sum(axis=0)
                        / np.maximum(finite.sum(axis=0), 1))
        safe = np.where(scale > 1e-12, scale, 1.0)
        standard = np.where(finite, (X_static - center) / safe, np.nan)
        with np.errstate(invalid="ignore"):
            low = np.nanmin(np.where(finite, standard, np.nan), axis=0)
            high = np.nanmax(np.where(finite, standard, np.nan), axis=0)
        return center, scale, np.nan_to_num(low), np.nan_to_num(high)

    @staticmethod
    def _static_grids(X_static):
        """Frozen sorted training columns, so a rank means the same thing later."""
        return [
            np.sort(X_static[np.isfinite(X_static[:, j]), j])
            for j in range(X_static.shape[1])
        ]

    # ------------------------------------------------------------------ fit

    def fit(self, X_static, X_series, y, loss: Loss, eval_set=None, verbose=False,
            groups=None):
        n = X_static.shape[0]
        K = loss.n_outputs(y)
        X_static, base_raw = self._augment(X_static, X_series, y=y, loss=loss,
                                           groups=groups)
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
            static_stats=self._static_stats(X_static),
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
                fit_rows = self._screen_bank(rows, g[:, k], h[:, k], master)
                tree = TemporalTree(self.tree_params)
                tree.fit(
                    X_static, X_series, g[:, k], h[:, k], fit_rows,
                    spawn_rng(master), context,
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
