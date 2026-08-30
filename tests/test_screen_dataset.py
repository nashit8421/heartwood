"""The screen must reject the three ways a dataset cannot answer the question.

Roadmap item 6 exists because this project has committed to four full studies
without first asking whether the dataset could answer the claim at all.  A
screen is only worth trusting if it fires on each failure separately, so these
tests build one dataset per failure -- statics at chance, statics reconstructable
from the series, a purely tabular regime -- and one that should pass, and check
the verdict is the one the construction guarantees.

The dangerous failure is the opposite of a false alarm: a screen that passes
everything would send the project into another week of compute with the same
confidence as before, which is the situation it was built to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.loaders import Dataset
from validation.screen_dataset import screen


def make(X_static, X_series, y, task="binary", key="synthetic"):
    headline = "r2" if task == "regression" else "balanced_accuracy"
    return Dataset(key=key, X_static=X_static, X_series=X_series, y=y,
                   task=task, headline=headline, groups=None)


def bumps(rng, n, T=48, amplitude=None):
    """A series whose informative feature is the height of a bump."""
    amplitude = rng.uniform(0.5, 3.0, size=n) if amplitude is None else amplitude
    t = np.arange(T)
    centre = T // 2
    shape = np.exp(-0.5 * ((t - centre) / 3.0) ** 2)
    return (amplitude[:, None] * shape[None, :] +
            0.3 * rng.normal(size=(n, T)))[:, None, :], amplitude


def windowed(rng, n, T=64):
    """A series a fixed summary cannot read.

    The informative bump sits in a narrow window; a *taller* nuisance bump sits
    outside it. So the global maximum reports the nuisance and says nothing
    about the target, while the maximum over the right window recovers it -- the
    same construction as ``make_shape_amplitude_regression``, and the reason it
    is a temporal regime rather than a tabular one.
    """
    t = np.arange(T)
    def bump(centre, amplitude):
        return amplitude[:, None] * np.exp(-0.5 * ((t - centre[:, None]) / 2.5) ** 2)

    amplitude = rng.uniform(0.5, 3.0, size=n)
    signal_at = rng.uniform(0.56 * T, 0.66 * T, size=n)
    # Much larger and much more variable than the signal, so it dominates any
    # whole-series statistic. With a merely comparable nuisance the global mean
    # still leaks the amplitude and the gap comes out at 0.03 -- measured, and
    # the reason this number is what it is.
    nuisance = rng.uniform(5.0, 20.0, size=n)
    nuisance_at = np.where(rng.random(n) < 0.5,
                           rng.uniform(0.05 * T, 0.40 * T, size=n),
                           rng.uniform(0.80 * T, 0.95 * T, size=n))
    series = bump(signal_at, amplitude) + bump(nuisance_at, nuisance)
    series += 0.3 * rng.normal(size=(n, T))
    return series[:, None, :], amplitude


def test_a_dataset_that_can_answer_the_question_passes():
    """Informative statics, exogenous, and a temporal regime.

    The statics contribute *additively*, not through an XOR, because criterion 1
    asks whether they are informative on their own -- see the module docstring
    on why an interaction-only dataset is screened out despite being the purest
    test of the claim.
    """
    rng = np.random.default_rng(0)
    n = 800
    series, amplitude = windowed(rng, n)
    statics = rng.normal(size=(n, 3))          # independent of the series
    score = 2.0 * (amplitude - 1.75) + 1.5 * statics[:, 0]
    y = (score > 0).astype(int)
    verdict = screen(make(statics, series, y))
    assert verdict.informative, verdict
    assert verdict.exogenous, verdict
    assert verdict.temporal, verdict
    assert verdict.passes


def test_statics_at_chance_are_rejected():
    """Sleep-EDF's failure: there is no static half to combine."""
    rng = np.random.default_rng(1)
    n = 600
    series, amplitude = bumps(rng, n)
    statics = rng.normal(size=(n, 3))
    y = (amplitude > 1.75).astype(int)         # statics carry nothing
    verdict = screen(make(statics, series, y))
    assert not verdict.informative
    assert not verdict.passes
    assert "nothing to fuse" in verdict.notes


def test_statics_reconstructable_from_the_series_are_rejected():
    """CPSC's failure: age and sex are written into the ECG."""
    rng = np.random.default_rng(2)
    n = 600
    series, amplitude = bumps(rng, n)
    # The static *is* the amplitude, so the series predicts it almost exactly.
    statics = np.column_stack([amplitude + 0.01 * rng.normal(size=n)])
    y = (statics[:, 0] > 1.75).astype(int)
    verdict = screen(make(statics, series, y))
    assert not verdict.exogenous
    assert not verdict.passes
    assert "disqualified" in verdict.notes


def test_a_tabular_regime_is_rejected():
    """A fixed summary loses nothing, so the architecture has no work to do."""
    rng = np.random.default_rng(3)
    n = 800
    # Flat series: the mean is a sufficient summary, so raw adds nothing.
    level = rng.normal(size=n)
    series = (level[:, None] + 0.05 * rng.normal(size=(n, 48)))[:, None, :]
    statics = rng.normal(size=(n, 3))
    y = ((level + 1.5 * statics[:, 0]) > 0).astype(int)
    verdict = screen(make(statics, series, y))
    assert verdict.informative, "the statics must be informative or this tests two things"
    assert not verdict.temporal, verdict
    assert not verdict.passes
    assert "tabular regime" in verdict.notes


def test_a_dataset_with_no_statics_is_rejected_without_crashing():
    rng = np.random.default_rng(4)
    n = 300
    series, amplitude = bumps(rng, n)
    verdict = screen(make(np.zeros((n, 0)), series, (amplitude > 1.75).astype(int)))
    assert not verdict.passes
    assert "no static block" in verdict.notes


def test_the_screen_never_fits_a_heartwood_model(monkeypatch):
    """Every number must come from baselines, or the gate can be fitted to it."""
    import heartwood

    def forbidden(*args, **kwargs):
        raise AssertionError("the screen fitted a Heartwood model")

    monkeypatch.setattr(heartwood.HeartwoodClassifier, "fit", forbidden)
    monkeypatch.setattr(heartwood.HeartwoodRegressor, "fit", forbidden)
    rng = np.random.default_rng(5)
    n = 300
    series, amplitude = bumps(rng, n)
    screen(make(rng.normal(size=(n, 2)), series, (amplitude > 1.75).astype(int)))


def test_a_large_dataset_is_subsampled_so_the_screen_stays_cheap():
    rng = np.random.default_rng(6)
    n = 4000
    series, amplitude = bumps(rng, n, T=16)
    verdict = screen(make(rng.normal(size=(n, 2)), series,
                          (amplitude > 1.75).astype(int)))
    assert verdict.n <= 1500


def test_regression_targets_are_screened_too():
    rng = np.random.default_rng(7)
    n = 500
    series, amplitude = bumps(rng, n)
    coef = rng.uniform(0.5, 2.0, size=n)
    statics = np.column_stack([coef, rng.normal(size=(n, 2))])
    verdict = screen(make(statics, series, amplitude * coef, task="regression"))
    assert verdict.task == "regression"
    assert np.isfinite(verdict.static_lift)


def test_an_interaction_only_dataset_is_screened_out():
    """The documented limitation, pinned so it cannot be forgotten.

    Neither half has any marginal signal, so criterion 1 rejects it -- even
    though a static flag XORed with a series feature is arguably the purest test
    of the founding claim there is. The module docstring says a rejection on
    criterion 1 alone deserves a second look by hand; this is why.
    """
    rng = np.random.default_rng(8)
    n = 800
    series, amplitude = bumps(rng, n)
    statics = rng.normal(size=(n, 3))
    y = ((amplitude > 1.75).astype(int) ^ (statics[:, 0] > 0).astype(int))
    verdict = screen(make(statics, series, y))
    assert not verdict.informative
    assert not verdict.passes


def test_the_split_respects_groups_when_there_are_any():
    """The V12/V13 lesson, applied to the screen itself.

    A row-wise hold-out on a dataset whose statics are constant within a subject
    measures subject recall, not exogeneity. On Apnea-ECG that read 0.82 and
    would have disqualified the dataset the roadmap calls this library's first
    fair test.
    """
    from validation.screen_dataset import _split

    groups = np.repeat(np.arange(10), 20)
    train, test = _split(len(groups), seed=0, groups=groups)
    assert set(groups[train]).isdisjoint(set(groups[test]))
    assert len(test) > 0 and len(train) > 0


def test_a_constant_within_group_static_is_not_called_endogenous():
    """The exact failure above, reproduced end to end and then not fired.

    The static is pure subject identity noise -- unrelated to the series shape --
    but every row of a subject shares it. A row-wise screen recovers it from the
    series (the segments of one subject look alike) and calls it endogenous; a
    subject-wise screen cannot, and does not.
    """
    rng = np.random.default_rng(9)
    n_subjects, per_subject = 40, 20
    n = n_subjects * per_subject
    groups = np.repeat(np.arange(n_subjects), per_subject)

    # Each subject has its own series offset, so subjects are identifiable...
    offset = rng.normal(size=n_subjects)[groups]
    series, amplitude = bumps(rng, n)
    series = series + offset[:, None, None]
    # ...and its own static, drawn independently of anything in the series.
    statics = rng.normal(size=n_subjects)[groups][:, None]
    y = (amplitude > 1.75).astype(int)

    dataset = make(statics, series, y)
    dataset.groups = groups
    verdict = screen(dataset, n_seeds=2)
    assert verdict.exogenous, (
        f"a static independent of the series was called endogenous "
        f"({verdict.exogeneity:+.3f}) -- the split is leaking subjects"
    )
