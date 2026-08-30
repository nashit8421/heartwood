"""The pre-screen must be out of fold, or it is the bug it was built to fix.

Roadmap item 2c replaces "offer the bank blind and let max-gain pick" with
"rank the bank first, then offer a shortlist".  The ranking is itself a
selection step, so it is only an improvement if it happens on rows the tree does
not then fit.  A screen computed on the fitting rows would rank features by the
same signal the split exploits and would look *better* in training while being
exactly the winner's curse it claims to remove -- the most expensive way this
change could go wrong, and an invisible one.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.bank import FeatureBank
from heartwood.splits import SplitSpec


def bank_with(columns):
    bank = FeatureBank(max_entries=64)
    for index, column in enumerate(columns):
        bank.promote(SplitSpec(kind="interval", channel=0, start=index,
                               end=index + 2, stat="mean"),
                     np.asarray(column, dtype=float), gain=1.0, round_index=0)
    return bank


def test_the_screen_keeps_the_associated_columns():
    rng = np.random.default_rng(0)
    target = rng.normal(size=200)
    # Correlated with the target but not with each other: the bank rejects a
    # column above 0.995 correlation with one it already holds, so three near
    # copies of the target would be stored as one entry and this test would be
    # checking the de-duplicator instead of the screen.
    useful = [target + 0.5 * rng.normal(size=200) for _ in range(3)]
    noise = [rng.normal(size=200) for _ in range(9)]
    bank = bank_with(useful + noise)
    assert len(bank) == 12, "de-duplication swallowed a column; test is invalid"

    rows = np.arange(200)
    bank.screen(rows, target, top_k=3)
    assert {id(e) for e in bank.screened} == {id(e) for e in bank.entries[:3]}


def test_the_screen_replaces_the_random_subsample():
    """A ranked shortlist thinned again at random would undo the ranking."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=100)
    bank = bank_with([target] + [rng.normal(size=100) for _ in range(15)])
    rows = np.arange(100)
    bank.screen(rows, target, top_k=4)
    drawn = list(bank.candidates(rows, np.random.default_rng(0), fraction=0.25))
    assert len(drawn) == 4


def test_clearing_the_screen_restores_the_whole_bank():
    rng = np.random.default_rng(0)
    bank = bank_with([rng.normal(size=50) for _ in range(10)])
    bank.screen(np.arange(50), rng.normal(size=50), top_k=2)
    assert len(list(bank.candidates(np.arange(50), None, 1.0))) == 2
    bank.clear_screen()
    assert len(list(bank.candidates(np.arange(50), None, 1.0))) == 10


def test_screening_an_empty_bank_is_a_no_op():
    bank = FeatureBank()
    bank.screen(np.arange(5), np.zeros(5), top_k=3)
    assert bank.screened is None
    assert list(bank.candidates(np.arange(5), None, 1.0)) == []


def test_top_k_is_at_least_one():
    rng = np.random.default_rng(0)
    bank = bank_with([rng.normal(size=30) for _ in range(5)])
    bank.screen(np.arange(30), rng.normal(size=30), top_k=0)
    assert len(bank.screened) == 1


def test_the_screen_rows_are_held_out_of_the_fit():
    """The load-bearing property: no row both ranks the bank and trains the tree.

    Checked by instrumenting the booster's own fold split rather than by
    inspecting a fitted model, because a leak here would not change any score in
    a way a test could recognise -- it would only make training look better.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2, 48))
    y = (X[:, 0, 5:20].mean(axis=1) > 0).astype(int)

    seen = []
    model = HeartwoodClassifier(n_estimators=12, max_depth=3, screen_fraction=0.25,
                                screen_top_k=4, random_state=0)
    model.fit(None, X, y)
    core = model._core

    original = core.bank.screen
    core.bank.screen = lambda rows, target, top_k: (
        seen.append(np.asarray(rows).copy()), original(rows, target, top_k))[1]

    rows = np.arange(300, dtype=np.intp)
    master = np.random.default_rng(1)
    g = np.full(300, 0.1)
    h = np.full(300, 0.25)
    fit_rows = core._screen_bank(rows, g, h, master)
    assert seen, "the screen never ran"
    screen_rows = seen[-1]
    assert set(screen_rows.tolist()).isdisjoint(set(fit_rows.tolist()))
    assert len(screen_rows) + len(fit_rows) == len(rows)


def test_a_screen_that_would_starve_the_fit_is_skipped():
    """Better to leave the bank unscreened than to measure a smaller training set."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    model = HeartwoodClassifier(n_estimators=4, max_depth=2, screen_fraction=0.95,
                                min_samples_leaf=5, random_state=0).fit(None, X, y)
    core = model._core
    rows = np.arange(12, dtype=np.intp)
    fit_rows = core._screen_bank(rows, np.full(12, 0.1), np.full(12, 0.25),
                                 np.random.default_rng(0))
    assert np.array_equal(fit_rows, rows)
    assert core.bank.screened is None


@pytest.mark.parametrize("fraction,top_k", [(0.0, 8), (0.2, 4), (0.3, 8)])
def test_the_model_fits_and_predicts_under_screening(fraction, top_k):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2, 48))
    y = (X[:, 0, 5:20].mean(axis=1) > 0).astype(int)
    model = HeartwoodClassifier(n_estimators=10, max_depth=3, random_state=0,
                                screen_fraction=fraction, screen_top_k=top_k)
    model.fit(None, X, y)
    assert model.predict(None, X).shape == (len(y),)


def test_the_screen_parameters_are_validated():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1, 16))
    y = (X[:, 0, :4].mean(axis=1) > 0).astype(int)
    with pytest.raises(ValueError, match="screen_fraction"):
        HeartwoodClassifier(screen_fraction=1.0).fit(None, X, y)
    with pytest.raises(ValueError, match="screen_top_k"):
        HeartwoodClassifier(screen_top_k=0).fit(None, X, y)
