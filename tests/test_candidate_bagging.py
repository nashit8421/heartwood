"""Per-node bagging must actually shrink the pool, and shrink it per node.

Roadmap item 2a attacks the winner's curse by giving each node a smaller
candidate pool.  Two ways that can silently fail to be what it claims: the knob
could thin the pool once per *tree* rather than once per *node*, which is a
different (and much weaker) estimator than the one being tested; or it could
empty a candidate source entirely at small fractions, so that what is measured
is "shapelets switched off" rather than "fewer shapelets".  Both would still
train, still score, and still look like a bagging result.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.tree import FitContext, TemporalTree, TreeParams


def counted(params, requested=16):
    tree = TemporalTree(params)
    return tree._draw_count(requested)


def test_full_fraction_is_the_shipped_behaviour():
    assert counted(TreeParams(candidate_colsample=1.0)) == 16


def test_a_fraction_shrinks_the_pool():
    assert counted(TreeParams(candidate_colsample=0.5)) == 8
    assert counted(TreeParams(candidate_colsample=0.25)) == 4


def test_a_disabled_source_stays_disabled():
    """Thinning zero candidates must not conjure one into existence."""
    assert counted(TreeParams(candidate_colsample=0.5), requested=0) == 0


def test_a_tiny_fraction_degrades_the_search_rather_than_removing_a_source():
    """The floor of one is what keeps this an ablation of pool *size*."""
    assert counted(TreeParams(candidate_colsample=1e-6)) == 1
    assert counted(TreeParams(candidate_colsample=1e-6), requested=1) == 1


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25, 0.05])
def test_the_model_still_fits_and_predicts_at_any_fraction(fraction):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2, 48))
    y = (X[:, 0, :16].mean(axis=1) > 0).astype(int)
    model = HeartwoodClassifier(
        n_estimators=8, max_depth=3, candidate_colsample=fraction, random_state=0
    ).fit(None, X, y)
    assert model.predict(None, X).shape == (len(y),)


def test_the_fraction_is_validated():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="candidate_colsample"):
            HeartwoodClassifier(candidate_colsample=bad).fit(None, X, y)


def test_every_node_draws_its_own_thinned_pool():
    """The pool must be thinned once per *node*, not once per tree.

    A per-tree draw would make this a random-subspace method -- every node in a
    tree searching the same restricted space -- which is a different estimator
    from the one item 2a is testing, and would score differently for reasons
    nothing in the write-up would explain.  Counting the pool a node is actually
    offered is the direct check; counting winning splits is not, because a good
    split legitimately wins at many nodes.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2, 64))
    y = (X[:, 0, 10:30].mean(axis=1) > 0).astype(int)

    class Counting(TemporalTree):
        def __init__(self, params):
            super().__init__(params)
            self.pool_sizes = []

        def _candidates(self, *args, **kwargs):
            drawn = list(super()._candidates(*args, **kwargs))
            self.pool_sizes.append(len(drawn))
            yield from drawn

    def pools(fraction):
        params = TreeParams(max_depth=4, candidate_colsample=fraction)
        tree = Counting(params)
        g = np.asarray(y, dtype=float) - 0.5
        h = np.full(len(y), 0.25)
        tree.fit(np.zeros((len(y), 0)), X, g, h,
                 np.arange(len(y)), np.random.default_rng(0), FitContext())
        return tree.pool_sizes

    full, thinned = pools(1.0), pools(0.25)
    assert len(full) > 3, "not enough nodes to tell per-node from per-tree"
    # Defaults draw 16 intervals and 4 shapelets, and a shapelet yields two
    # candidates -- distance and position -- so a full pool is 16 + 8.  A
    # quarter of each draw is 4 + 2.  Every node, not just the root.
    assert all(size == 24 for size in full)
    assert all(size == 6 for size in thinned)
    # Tree shape may differ between the two: a thinned node can fail to find a
    # split its full-pool counterpart found, so the node counts need not match.


def test_a_smaller_pool_actually_changes_the_fitted_model():
    """Guards against a knob that is stored, validated, and never read."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2, 64))
    y = (X[:, 0, 10:30].mean(axis=1) > 0).astype(int)

    def splits(fraction):
        model = HeartwoodClassifier(
            n_estimators=10, max_depth=3, candidate_colsample=fraction, random_state=0
        ).fit(None, X, y)
        return [node["spec"].feature_name()
                for round_trees in model._core.trees_ for tree in round_trees
                for node in tree.nodes if not node["leaf"]]

    assert splits(1.0) != splits(0.25)
