"""The permutation null must not tighten when you ask it for more precision.

V8 built a chance floor for split acceptance and it did not work.  Roadmap item
2d says that was one attempt at a hard problem rather than a verdict on the
idea, and these tests pin down what was actually wrong: the floor was the
maximum gain over every permutation *and* every candidate, so raising
``selection_null`` raised the bar.  The knob that was supposed to sharpen the
estimate silently changed the test, which makes any sweep over it
uninterpretable.

Measured false-split rates on a pure-noise task, 120 trials:

    quantile 1.0 (V8):  n=1 -> 0.53   n=4 -> 0.21   n=16 -> 0.06
    quantile 0.5:       n=4 -> 0.52   n=16 -> 0.53
    quantile 0.95:      n=4 -> 0.23   n=16 -> 0.07

The first row is the bug.  The second is what a fixed quantile buys: the same
bar, estimated better.  The third shows the cost of estimating a far tail from
few draws, which is why V19 does not run high quantiles at low permutation
counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood.tree import FitContext, TemporalTree, TreeParams


def floors(quantile, permutations, n=200, seed=0):
    """``_chance_gain`` on a fixed pool, varying only the null's settings."""
    rng = np.random.default_rng(seed)
    gr = rng.normal(size=n)
    hr = np.full(n, 0.25)
    scored = [rng.normal(size=n) for _ in range(24)]
    tree = TemporalTree(TreeParams(selection_null=permutations,
                                   selection_null_quantile=quantile))
    return tree._chance_gain(scored, gr, hr, np.random.default_rng(seed + 1))


def test_v8s_floor_rises_with_the_permutation_count():
    """The defect, kept as a test so the fix cannot regress into it."""
    few = np.mean([floors(1.0, 2, seed=s) for s in range(8)])
    many = np.mean([floors(1.0, 16, seed=s) for s in range(8)])
    assert many > few * 1.1, (
        "quantile 1.0 is the global maximum; more permutations must make it "
        f"harsher ({few:.4f} -> {many:.4f}) or this test no longer describes V8"
    )


def test_a_fixed_quantile_is_stable_in_the_permutation_count():
    """The fix: more permutations estimate the same bar, they do not move it."""
    few = np.mean([floors(0.5, 4, seed=s) for s in range(8)])
    many = np.mean([floors(0.5, 32, seed=s) for s in range(8)])
    assert abs(many - few) < 0.25 * max(few, 1e-9), (
        f"median floor moved from {few:.4f} to {many:.4f} with more permutations"
    )


def test_a_higher_quantile_is_a_harsher_bar():
    at_half = np.mean([floors(0.5, 16, seed=s) for s in range(8)])
    at_high = np.mean([floors(0.95, 16, seed=s) for s in range(8)])
    assert at_high > at_half


def test_the_default_reproduces_v8_exactly():
    """One permutation at quantile 1.0 is V8, so that arm is still comparable."""
    assert TreeParams().selection_null_quantile == 1.0
    assert floors(1.0, 1, seed=3) == floors(1.0, 1, seed=3)


def test_an_empty_pool_has_no_floor():
    tree = TemporalTree(TreeParams(selection_null=4))
    assert tree._chance_gain([], np.ones(4), np.ones(4),
                             np.random.default_rng(0)) == 0.0


def test_the_null_still_rejects_pure_noise_more_often_than_it_accepts_it():
    """End to end on a task with no signal, at a bar that should mostly refuse."""
    rng = np.random.default_rng(0)
    splits = 0
    trials = 40
    for trial in range(trials):
        X = rng.normal(size=(150, 2, 32))
        y = rng.integers(0, 2, size=150).astype(float)
        tree = TemporalTree(TreeParams(max_depth=1, selection_null=16,
                                       selection_null_quantile=0.95))
        tree.fit(np.zeros((150, 0)), X, y - 0.5, np.full(150, 0.25),
                 np.arange(150), np.random.default_rng(trial), FitContext())
        splits += 0 if tree.nodes[0]["leaf"] else 1
    assert splits / trials < 0.35, f"accepted {splits}/{trials} noise splits"


def test_the_quantile_is_validated():
    from heartwood import HeartwoodClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="selection_null_quantile"):
            HeartwoodClassifier(selection_null_quantile=bad).fit(None, X, y)
