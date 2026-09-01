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

Which knobs to touch
--------------------

The estimators take a large number of parameters and **most of them should be
left alone.** They are not all equal, and the difference is measured rather than
stylistic:

*The ordinary ones*, which behave the way they do in any gradient booster:
``n_estimators``, ``learning_rate``, ``max_depth``, ``reg_lambda``, ``gamma``,
``min_child_weight``, ``min_samples_leaf``, ``subsample``, ``colsample``,
``early_stopping_rounds``, ``random_state``.

*The one that changes the architecture*: ``dense_base=True`` puts a ridge over a
bank of dilated convolutions underneath the trees, and ``dense_include_static``
lets that ridge see the static block.  This is where this library's measured
advantage over MiniROCKET comes from; see ``README.md``.

*The ones that are switched off and stayed switched off.*  Each is a hypothesis
this project pre-registered, ran, and failed to confirm.  They are kept at
no-op defaults because the studies that failed them pre-committed to keeping
them as controls for later work -- **not because they are worth tuning.**
Turning one on is re-opening a question that has an answer:

===============================  ========  ==========================
parameter                        study     what was measured
===============================  ========  ==========================
``candidate_colsample``          V16       best 1 of 8 datasets, +0.4
``mc_penalty``                   V17       best 1 of 8, -0.1
``screen_fraction``/``top_k``    V18       best 1 of 8, -0.1
``selection_null``/``quantile``  V8, V19   best 1 of 8, -0.3
``nonlinear_features``/``gamma`` V21       0 of 8 at every width
``n_product_candidates``         V22       +1.0% against a 4% bar
``base_static_products``         V22       +2.9% against a 4% bar
``dense_static_interactions``    V11       Apnea 0.856 -> 0.478 AUC
===============================  ========  ==========================

The full write-ups are in ``VALIDATION_V*.md`` and ``RESULTS_V*.md``.  Four
further features -- virtual channels, a window-statistic bank, comparison splits
and Levy areas -- were measured the same way and **deleted** rather than left as
flags; that history is in ``RESULTS_V15.md`` and ``RESULTS_V23.md``.
"""

from .api import HeartwoodClassifier, HeartwoodRegressor
from .features import STAT_NAMES, interval_stat, shapelet_features
from .splits import SplitSpec

__version__ = "1.0.0"

__all__ = [
    "HeartwoodClassifier",
    "HeartwoodRegressor",
    "SplitSpec",
    "STAT_NAMES",
    "interval_stat",
    "shapelet_features",
    "__version__",
]
