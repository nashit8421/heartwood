"""Heartwood — gradient-boosted trees that read the rings.

A tree records time inside itself, one ring per season. That is the idea here:
gradient-boosted trees whose splits can read the trajectory a row carries, not
just the columns beside it.


Real datasets routinely pair per-row attributes with a raw trajectory per row.
The standard workaround is to summarise the series into aggregates and hand the
result to XGBoost, which discards exactly the information that matters: when an
event happened, what shape it had, how the journey unfolded.

Heartwood keeps XGBoost's second-order boosting machinery and enlarges what a
split is allowed to ask about, so a tree node can just as easily ask "was the
slope between t=12 and t=40 negative?" or "does this shape occur, and how early?"
as "is the customer over 50?" — all scored on the same gain, all learned from
the gradients rather than fixed up front.

    from heartwood import HeartwoodClassifier

    model = HeartwoodClassifier(n_estimators=200, random_state=0)
    model.fit(X_static, X_series, y)
    model.predict_proba(X_static_test, X_series_test)
"""

from .api import HeartwoodClassifier, HeartwoodRegressor
from .features import STAT_NAMES, interval_stat, shapelet_features
from .splits import SplitSpec

__version__ = "0.1.0"

__all__ = [
    "HeartwoodClassifier",
    "HeartwoodRegressor",
    "SplitSpec",
    "STAT_NAMES",
    "interval_stat",
    "shapelet_features",
    "__version__",
]
