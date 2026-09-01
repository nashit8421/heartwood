"""The multiple-comparisons charge must track the null it claims to model.

``_selection_charge`` is not a heuristic with a tuned constant; it is a claim
about how the maximum gain over an uninformative pool grows -- like
``scale * log m``.  If that claim is wrong, or if ``scan_threshold`` changes in
a way that breaks it, the charge silently becomes an arbitrary pruning knob
while still looking principled in the write-up.  These tests measure the null
against the real ``scan_threshold`` and fail when the two drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.splits import scan_threshold
from heartwood.tree import MC_THRESHOLD_LOG, FitContext, TemporalTree, TreeParams


def null_max_gain(rng, n, m, shift=0.0, hetero=False, reg_lambda=1.0):
    """Best gain a pool of ``m`` uninformative features reaches, over ``scale``."""
    g = rng.normal(size=n) + shift
    h = rng.gamma(2.0, 0.5, size=n) if hetero else np.full(n, 1.0)
    best = 0.0
    for _ in range(m):
        found = scan_threshold(rng.normal(size=n), g, h, reg_lambda, 0.0, 1e-3, 5)
        if found is not None and found[0] > best:
            best = found[0]
    G, H = float(g.sum()), float(h.sum())
    centred = g - h * (G / H)
    return best / (float(centred @ centred) / (H + reg_lambda))


@pytest.mark.parametrize("shift,hetero", [(0.0, False), (3.0, False),
                                          (0.0, True), (3.0, True)])
def test_the_null_maximum_grows_like_log_of_the_pool(shift, hetero):
    """The load-bearing claim: slope 1.0 in ``log m``, in every gradient regime.

    The charge is derived from the maximum of ``m`` chi-squares, whose
    expectation grows like ``2 log m``; a gain is half a chi-square times the
    node's gradient scale, so the slope should be 1.  A slope materially away
    from 1 would mean the pool's candidates are not behaving independently and
    the correction is mis-shaped, not merely mis-scaled.
    """
    rng = np.random.default_rng(7)
    counts = [4, 16, 64, 256]
    means = [float(np.mean([null_max_gain(rng, 400, m, shift, hetero)
                            for _ in range(40)])) for m in counts]
    design = np.vstack([np.log(counts), np.ones(len(counts))]).T
    slope, _ = np.linalg.lstsq(design, np.array(means), rcond=None)[0]
    assert 0.8 <= slope <= 1.3, f"log-m slope drifted to {slope:.2f}"


def test_the_threshold_search_coefficient_is_the_measured_one():
    """``MC_THRESHOLD_LOG`` prices the within-candidate scan over cut points.

    It is well under 1 because adjacent cut points are strongly correlated, so
    a node's effective pool is much smaller than ``candidates x rows``.  This
    test is what keeps that number attached to ``scan_threshold``'s actual
    behaviour rather than to the day it was fitted.
    """
    rng = np.random.default_rng(11)
    sizes = [100, 400, 1600]
    means = [float(np.mean([null_max_gain(rng, n, 16) for _ in range(40)]))
             for n in sizes]
    design = np.vstack([np.log(sizes), np.ones(len(sizes))]).T
    slope, _ = np.linalg.lstsq(design, np.array(means), rcond=None)[0]
    assert abs(slope - MC_THRESHOLD_LOG) < 0.25, (
        f"threshold-scan coefficient measured {slope:.2f}, "
        f"MC_THRESHOLD_LOG is {MC_THRESHOLD_LOG}"
    )


def test_the_charge_grows_with_the_pool_and_shrinks_with_the_node():
    tree = TemporalTree(TreeParams(mc_penalty=1.0))
    rng = np.random.default_rng(0)
    g = rng.normal(size=200)
    h = np.full(200, 1.0)
    H = float(h.sum())
    small = tree._selection_charge(4, g, h, H)
    large = tree._selection_charge(256, g, h, H)
    assert large > small > 0

    # A node with less gradient mass to explain is charged less.
    quiet = tree._selection_charge(64, 0.1 * g, h, H)
    loud = tree._selection_charge(64, g, h, H)
    assert quiet < loud


def test_an_empty_pool_is_charged_nothing():
    tree = TemporalTree(TreeParams(mc_penalty=1.0))
    g, h = np.ones(10), np.ones(10)
    assert tree._selection_charge(0, g, h, 10.0) == 0.0


def test_the_charge_is_off_by_default():
    assert TreeParams().mc_penalty == 0.0


def test_the_charge_prunes_monotonically():
    """More charge, fewer splits -- and eventually none."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2, 64))
    y = (X[:, 0, 10:30].mean(axis=1) > 0).astype(int)

    def splits(mc):
        model = HeartwoodClassifier(
            n_estimators=20, max_depth=4, mc_penalty=mc, random_state=0
        ).fit(None, X, y)
        return sum(1 for r in model._core.trees_ for t in r
                   for node in t.nodes if not node["leaf"])

    # 16.0 rather than 8.0 for the refuse-everything end. Deleting comparison
    # splits after V23 made each node's pool smaller, and the charge is
    # proportional to log(pool size), so the same multiplier now buys a smaller
    # charge: at 8.0 a single split survives where none used to. The property
    # under test is that a large enough charge refuses everything, not that any
    # particular number does.
    counts = [splits(mc) for mc in (0.0, 1.0, 2.0, 16.0)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[-1] == 0, "a large enough charge must refuse every split"


def test_the_charge_cannot_change_which_candidate_wins():
    """Every candidate at a node shares one pool size, so the charge is a floor.

    If it ever became a per-candidate adjustment it would reorder the pool, and
    the write-up's claim that this only decides *whether* a node splits would
    stop being true.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2, 48))
    y = (X[:, 0, 5:20].mean(axis=1) > 0).astype(int)

    def first_split(mc):
        params = TreeParams(max_depth=1, mc_penalty=mc)
        tree = TemporalTree(params)
        g = np.asarray(y, dtype=float) - 0.5
        h = np.full(len(y), 0.25)
        tree.fit(np.zeros((len(y), 0)), X, g, h, np.arange(len(y)),
                 np.random.default_rng(0), FitContext())
        root = tree.nodes[0]
        return None if root["leaf"] else root["spec"].feature_name()

    unpriced = first_split(0.0)
    assert unpriced is not None
    # A charge small enough not to veto the root must leave the same winner.
    assert first_split(1e-6) == unpriced


def test_mc_penalty_is_validated():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    with pytest.raises(ValueError, match="mc_penalty"):
        HeartwoodClassifier(mc_penalty=-1.0).fit(None, X, y)
