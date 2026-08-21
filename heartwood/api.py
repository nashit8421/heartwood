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
        n_comparison_candidates: int = 4,
        dense_base: bool = False,
        dense_features: str = "stats",
        n_rocket_features: int = 10000,
        levy_areas: bool = True,
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
        self.n_comparison_candidates = n_comparison_candidates
        self.dense_base = dense_base
        self.dense_features = dense_features
        self.n_rocket_features = n_rocket_features
        self.levy_areas = levy_areas
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._core: _BoosterCore | None = None
        self.n_static_features_ = 0
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
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

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
            n_comparison_candidates=self.n_comparison_candidates,
            bank_colsample=self.bank_colsample,
        )

    # ------------------------------------------------------------------ fit

    def _fit_core(self, X_static, X_series, y, loss, eval_set, verbose):
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

        self._core = _BoosterCore(
            tree_params=self._tree_params(),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            bank_enabled=self.bank_enabled,
            bank_max=self.bank_max,
            dense_base=self.dense_base,
            dense_features=self.dense_features,
            n_rocket_features=self.n_rocket_features,
            levy_areas=self.levy_areas,
        )
        self._core.fit(Xs, Xt, y, loss, eval_set=prepared_eval, verbose=verbose)
        return self

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

    def fit(self, X_static, X_series, y, eval_set=None, verbose=False):
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("classification needs at least 2 classes in y")

        loss = (
            Logistic() if self.n_classes_ == 2 else Softmax(self.n_classes_)
        )
        y_enc = y_enc.astype(np.float64)
        return self._fit_core(X_static, X_series, y_enc, loss, eval_set, verbose)

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

    def fit(self, X_static, X_series, y, eval_set=None, verbose=False):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D for regression, got shape {y.shape}")
        return self._fit_core(X_static, X_series, y, SquaredError(), eval_set, verbose)

    def predict(self, X_static, X_series=None) -> np.ndarray:
        Xs, Xt, _ = self._check_predict_inputs(X_static, X_series)
        return self._core.predict_raw(Xs, Xt)[:, 0]
