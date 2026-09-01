"""Public estimators: :class:`HeartwoodClassifier` and :class:`HeartwoodRegressor`.

Both take the two blocks separately — ``fit(X_static, X_series, y)`` — because
keeping the series raw is the entire point.  Either block may be ``None``: with
no series the model degrades to an ordinary gradient-boosted tree, and with no
static block it is a pure time-series learner.
"""

from __future__ import annotations

import numpy as np

from ._util import check_inputs
from .booster import _BoosterCore
from .features import STAT_NAMES
from .losses import Logistic, SquaredError, Softmax, sigmoid, softmax
from .tree import TreeParams


class _BaseHeartwood:
    """Shared parameter handling, validation and inspection."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1e-3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        colsample: float = 1.0,
        n_interval_candidates: int = 16,
        n_shapelet_candidates: int = 4,
        interval_stats: tuple[str, ...] = STAT_NAMES,
        full_interval_prob: float = 0.25,
        min_interval_len: int = 3,
        shapelet_min_len: int = 3,
        shapelet_max_frac: float = 0.5,
        shapelet_znorm: bool = True,
        n_filter_candidates: int = 0,
        n_fitted_filters: int = 4,
        filter_len: int = 9,
        dct_components: int = 5,
        ridge_beta: float = 1.0,
        n_filter_alt: int = 1,
        bank_enabled: bool = True,
        bank_max: int = 32,
        bank_colsample: float = 0.25,
        screen_fraction: float = 0.0,
        screen_top_k: int = 8,
        candidate_colsample: float = 1.0,
        n_product_candidates: int = 0,
        dense_base: bool = False,
        selection_null: int = 0,
        selection_null_quantile: float = 1.0,
        mc_penalty: float = 0.0,
        n_rocket_features: int = 10000,
        dense_include_static: bool = False,
        dense_static_interactions: bool = False,
        nonlinear_features: int = 0,
        nonlinear_gamma: float = 1.0,
        base_static_products: bool = False,
        no_regret: bool = False,
        no_regret_fraction: float = 0.25,
        no_regret_margin: float = 0.0,
        early_stopping_rounds: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.n_interval_candidates = n_interval_candidates
        self.n_shapelet_candidates = n_shapelet_candidates
        self.interval_stats = tuple(interval_stats)
        self.full_interval_prob = full_interval_prob
        self.min_interval_len = min_interval_len
        self.shapelet_min_len = shapelet_min_len
        self.shapelet_max_frac = shapelet_max_frac
        self.shapelet_znorm = shapelet_znorm
        self.n_filter_candidates = n_filter_candidates
        self.n_fitted_filters = n_fitted_filters
        self.filter_len = filter_len
        self.dct_components = dct_components
        self.ridge_beta = ridge_beta
        self.n_filter_alt = n_filter_alt
        self.bank_enabled = bank_enabled
        self.bank_max = bank_max
        self.bank_colsample = bank_colsample
        self.screen_fraction = screen_fraction
        self.screen_top_k = screen_top_k
        self.candidate_colsample = candidate_colsample
        self.n_product_candidates = n_product_candidates
        self.dense_base = dense_base
        self.selection_null = selection_null
        self.selection_null_quantile = selection_null_quantile
        self.mc_penalty = mc_penalty
        self.n_rocket_features = n_rocket_features
        self.dense_include_static = dense_include_static
        self.dense_static_interactions = dense_static_interactions
        self.nonlinear_features = nonlinear_features
        self.nonlinear_gamma = nonlinear_gamma
        self.base_static_products = base_static_products
        self.no_regret = no_regret
        self.no_regret_fraction = no_regret_fraction
        self.no_regret_margin = no_regret_margin
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._core: _BoosterCore | None = None
        self.n_static_features_ = 0
        #: Which component the no-regret guarantee kept.  Always "combined"
        #: when the guarantee is off, so downstream code can read it either way.
        self.fallback_ = "combined"
        self.component_scores_: dict[str, float] = {}
        self.series_shape_: tuple[int, int] | None = None

    # --------------------------------------------------------------- params

    def get_params(self) -> dict:
        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and not k.endswith("_")
        }

    def _validate_params(self) -> None:
        unknown = set(self.interval_stats) - set(STAT_NAMES)
        if unknown:
            raise ValueError(f"unknown interval statistics: {sorted(unknown)}")
        if not self.interval_stats:
            raise ValueError("interval_stats must not be empty")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < self.colsample <= 1:
            raise ValueError("colsample must be in (0, 1]")
        if not 0 < self.candidate_colsample <= 1:
            raise ValueError("candidate_colsample must be in (0, 1]")
        if not 0 < self.bank_colsample <= 1:
            raise ValueError("bank_colsample must be in (0, 1]")
        if self.n_estimators < 0:
            raise ValueError("n_estimators must be >= 0")
        if self.n_estimators == 0 and not self.dense_base:
            # Zero trees over no base is a constant predictor, which is a
            # configuration error rather than a model. Zero trees *over* a base
            # is the base on its own -- the component the no-regret guarantee
            # falls back to -- so that combination is allowed.
            raise ValueError("n_estimators=0 needs dense_base=True to predict anything")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.n_product_candidates < 0:
            raise ValueError("n_product_candidates must be >= 0")
        if self.mc_penalty < 0:
            raise ValueError("mc_penalty must be >= 0")
        if not 0 < self.selection_null_quantile <= 1:
            raise ValueError("selection_null_quantile must be in (0, 1]")
        if not 0 < self.no_regret_fraction < 1:
            raise ValueError("no_regret_fraction must be in (0, 1)")
        if self.no_regret_margin < 0:
            raise ValueError("no_regret_margin must be >= 0")
        if self.nonlinear_features < 0:
            raise ValueError("nonlinear_features must be >= 0")
        if self.nonlinear_gamma <= 0:
            raise ValueError("nonlinear_gamma must be > 0")
        if not 0 <= self.screen_fraction < 1:
            raise ValueError("screen_fraction must be in [0, 1)")
        if self.screen_top_k < 1:
            raise ValueError("screen_top_k must be >= 1")

    def _tree_params(self) -> TreeParams:
        return TreeParams(
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            colsample=self.colsample,
            n_interval_candidates=self.n_interval_candidates,
            n_shapelet_candidates=self.n_shapelet_candidates,
            interval_stats=self.interval_stats,
            full_interval_prob=self.full_interval_prob,
            min_interval_len=self.min_interval_len,
            shapelet_min_len=self.shapelet_min_len,
            shapelet_max_frac=self.shapelet_max_frac,
            shapelet_znorm=self.shapelet_znorm,
            n_filter_candidates=self.n_filter_candidates,
            n_fitted_filters=self.n_fitted_filters,
            filter_len=self.filter_len,
            dct_components=self.dct_components,
            ridge_beta=self.ridge_beta,
            n_filter_alt=self.n_filter_alt,
            n_product_candidates=self.n_product_candidates,
            selection_null=self.selection_null,
            selection_null_quantile=self.selection_null_quantile,
            mc_penalty=self.mc_penalty,
            bank_colsample=self.bank_colsample,
            candidate_colsample=self.candidate_colsample,
        )

    # ------------------------------------------------------------------ fit

    def _fit_core(self, X_static, X_series, y, loss, eval_set, verbose, groups=None):
        self._validate_params()
        Xs, Xt, n = check_inputs(X_static, X_series)
        if len(y) != n:
            raise ValueError(f"y has {len(y)} rows but X has {n}")

        self.n_static_features_ = Xs.shape[1]
        self.series_shape_ = None if Xt is None else (Xt.shape[1], Xt.shape[2])

        prepared_eval = None
        if eval_set is not None:
            if len(eval_set) != 3:
                raise ValueError("eval_set must be (X_static, X_series, y)")
            Xs_v, Xt_v, y_v = eval_set
            Xs_v, Xt_v, _ = self._align_inputs(Xs_v, Xt_v)
            prepared_eval = (Xs_v, Xt_v, self._encode_eval_target(y_v))

        if self.no_regret:
            self.fallback_ = self._choose_component(Xs, Xt, y, loss, groups)
        self._core = self._make_core(**self._component_overrides(self.fallback_))
        self._core.fit(Xs, Xt, y, loss, eval_set=prepared_eval, verbose=verbose,
                       groups=groups)
        return self

    # ------------------------------------------------------- no-regret (V20)

    #: The candidates the no-regret guarantee chooses between, and what each
    #: one turns off.  ``combined`` is the shipped architecture and is the
    #: default: a component has to *earn* the fallback on held-out data.
    _COMPONENTS = {
        "combined": {},
        "base_only": {"n_estimators": 0},
        "trees_only": {"dense_base": False},
    }

    def _component_overrides(self, component: str) -> dict:
        return dict(self._COMPONENTS[component])

    def _make_core(self, **overrides):
        settings = dict(
            tree_params=self._tree_params(),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            bank_enabled=self.bank_enabled,
            bank_max=self.bank_max,
            screen_fraction=self.screen_fraction,
            screen_top_k=self.screen_top_k,
            dense_base=self.dense_base,
            n_rocket_features=self.n_rocket_features,
            dense_include_static=self.dense_include_static,
            dense_static_interactions=self.dense_static_interactions,
            nonlinear_features=self.nonlinear_features,
            nonlinear_gamma=self.nonlinear_gamma,
            base_static_products=self.base_static_products,
        )
        settings.update(overrides)
        return _BoosterCore(**settings)

    def _holdout_split(self, y, n):
        """Rows to fit the candidates on, and rows to judge them on.

        Stratified for classification, because a fold that happens to miss a
        class would score every candidate on a different problem than the one
        being solved.
        """
        rng = np.random.default_rng(
            0 if self.random_state is None else self.random_state
        )
        held = np.zeros(n, dtype=bool)
        strata = ([np.arange(n)] if self.__class__.__name__.endswith("Regressor")
                  else [np.flatnonzero(y == value) for value in np.unique(y)])
        for group in strata:
            take = int(round(self.no_regret_fraction * group.size))
            # Never empty a class out of the fitting half to fill the hold-out.
            take = min(max(take, 1), max(group.size - 1, 0))
            if take:
                held[rng.choice(group, size=take, replace=False)] = True
        if held.all() or not held.any():
            return None
        return np.flatnonzero(~held), np.flatnonzero(held)

    def _choose_component(self, Xs, Xt, y, loss, groups) -> str:
        """Pick the model that a held-out fold actually prefers.

        Roadmap item 3.  "Never meaningfully worse than its best component" is a
        stronger thing to be able to say than any single benchmark win, and it
        is the property that makes a ``static_control``-style regression
        impossible by construction rather than something re-measured every
        study.

        The asymmetry is the design.  ``combined`` wins ties and wins by
        default; a component has to beat it by more than ``no_regret_margin`` on
        data neither of them was fitted on.  This selection is itself a
        selection step and can over-fit like any other, so requiring evidence to
        *deviate* is what keeps its worst case small: at ``margin=0`` it is a
        plain argmin, and above that it is an argmin that must clear a bar.
        """
        split = self._holdout_split(y, len(y))
        if split is None:
            return "combined"
        fit_rows, held_rows = split

        def score(component: str) -> float:
            if component == "base_only" and not self.dense_base:
                return np.inf   # there is no base to fall back to
            core = self._make_core(**self._component_overrides(component))
            core.fit(
                Xs[fit_rows], None if Xt is None else Xt[fit_rows], y[fit_rows], loss,
                groups=None if groups is None else np.asarray(groups)[fit_rows],
            )
            raw = core.predict_raw(Xs[held_rows],
                                   None if Xt is None else Xt[held_rows])
            return float(loss.eval_metric(y[held_rows], raw))

        self.component_scores_ = {name: score(name) for name in self._COMPONENTS}
        combined = self.component_scores_["combined"]
        best = min(
            (name for name in self._COMPONENTS if name != "combined"),
            key=lambda name: self.component_scores_[name],
        )
        if self.component_scores_[best] < combined - self.no_regret_margin:
            return best
        return "combined"

    def _encode_eval_target(self, y_val):
        return np.asarray(y_val, dtype=np.float64)

    def _check_predict_inputs(self, X_static, X_series):
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._align_inputs(X_static, X_series)

    def _align_inputs(self, X_static, X_series):
        """Coerce a batch to the shapes this model was fitted on."""
        pad_to = None if self.series_shape_ is None else self.series_shape_[1]
        if self.series_shape_ is not None and X_series is None:
            raise ValueError("model was fitted with series data; X_series is required")
        if self.series_shape_ is None and X_series is not None:
            raise ValueError("model was fitted without series data; X_series must be None")

        Xs, Xt, n = check_inputs(X_static, X_series, pad_to=pad_to)
        if Xs.shape[1] != self.n_static_features_:
            raise ValueError(
                f"X_static has {Xs.shape[1]} columns but the model was fitted with "
                f"{self.n_static_features_}"
            )
        if Xt is not None and Xt.shape[1] != self.series_shape_[0]:
            raise ValueError(
                f"X_series has {Xt.shape[1]} channels but the model was fitted with "
                f"{self.series_shape_[0]}"
            )
        return Xs, Xt, n

    # ----------------------------------------------------------- inspection

    @property
    def best_iteration_(self) -> int:
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._core.best_iteration_

    @property
    def train_history_(self) -> list[float]:
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return list(self._core.train_history_)

    @property
    def eval_history_(self) -> list[float]:
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return list(self._core.eval_history_)

    def feature_importances(self) -> dict[str, float]:
        """Total gain per readable feature family, highest first."""
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        totals: dict[str, float] = {}
        for spec in self._core.iter_splits():
            totals[spec.family()] = totals.get(spec.family(), 0.0) + float(spec.gain)
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def dump_splits(self, top: int | None = None) -> list[tuple[str, float]]:
        """Every split as ``(description, gain)``, highest gain first."""
        if self._core is None:
            raise RuntimeError("model is not fitted; call fit() first")
        rows = [(spec.describe(), float(spec.gain)) for spec in self._core.iter_splits()]
        rows.sort(key=lambda kv: kv[1], reverse=True)
        return rows if top is None else rows[:top]


class HeartwoodClassifier(_BaseHeartwood):
    """Gradient boosting over static + temporal splits, for classification."""

    def fit(self, X_static, X_series, y, eval_set=None, verbose=False, groups=None):
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("classification needs at least 2 classes in y")

        loss = (
            Logistic() if self.n_classes_ == 2 else Softmax(self.n_classes_)
        )
        y_enc = y_enc.astype(np.float64)
        return self._fit_core(X_static, X_series, y_enc, loss, eval_set, verbose, groups)

    def _encode_eval_target(self, y_val):
        y_val = np.asarray(y_val)
        unknown = set(np.unique(y_val)) - set(self.classes_.tolist())
        if unknown:
            raise ValueError(f"eval_set contains unseen classes: {sorted(unknown)}")
        return np.searchsorted(self.classes_, y_val).astype(np.float64)

    def predict_proba(self, X_static, X_series=None) -> np.ndarray:
        Xs, Xt, _ = self._check_predict_inputs(X_static, X_series)
        raw = self._core.predict_raw(Xs, Xt)
        if self.n_classes_ == 2:
            p = sigmoid(raw[:, 0])
            return np.column_stack([1.0 - p, p])
        return softmax(raw)

    def predict(self, X_static, X_series=None) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X_static, X_series), axis=1)]


class HeartwoodRegressor(_BaseHeartwood):
    """Gradient boosting over static + temporal splits, for regression."""

    def fit(self, X_static, X_series, y, eval_set=None, verbose=False, groups=None):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D for regression, got shape {y.shape}")
        return self._fit_core(X_static, X_series, y, SquaredError(), eval_set, verbose,
                              groups)

    def predict(self, X_static, X_series=None) -> np.ndarray:
        Xs, Xt, _ = self._check_predict_inputs(X_static, X_series)
        return self._core.predict_raw(Xs, Xt)[:, 0]
