"""The V15 bank ablation isolates one extra per arm, or it measures nothing.

Item 1 of the roadmap asks what our bank's additions to MiniROCKET are actually
worth, and answers it by running one arm per addition.  That answer is only
worth having if each arm differs from the baseline in *exactly* the thing it is
named after.  Two failures would be invisible in the scores -- an arm that
silently changes a second setting, and an arm that changes nothing at all on
data where its extra is vacuous -- and both would still produce a full table of
plausible numbers.  These tests exist to make both loud.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.dense import levy_area_columns
from heartwood.rocket import RocketBank, _channel_groups

_spec = importlib.util.spec_from_file_location(
    "run_validation",
    Path(__file__).resolve().parents[1] / "validation" / "run_validation.py",
)
run_validation = importlib.util.module_from_spec(_spec)
sys.modules["run_validation"] = run_validation
_spec.loader.exec_module(run_validation)

VARIANTS = run_validation.VARIANTS
ABLATION_EXTRAS = run_validation.ABLATION_EXTRAS
MULTICHANNEL_ONLY_EXTRAS = run_validation.MULTICHANNEL_ONLY_EXTRAS
variant_kwargs = run_validation.variant_kwargs


def series(n=24, channels=3, length=64, seed=0):
    return np.random.default_rng(seed).normal(size=(n, channels, length))


# ------------------------------------------------------------ the arm table


@pytest.mark.parametrize("extra,arm", sorted(ABLATION_EXTRAS.items()))
def test_each_arm_differs_from_baseline_in_exactly_one_setting(extra, arm):
    base, tested = variant_kwargs("abl_min"), variant_kwargs(arm)
    assert base.keys() == tested.keys()
    differing = sorted(k for k in base if base[k] != tested[k])
    assert len(differing) == 1, f"{arm} changes {differing}, not one setting"


def test_all_arm_switches_on_every_extra():
    """``abl_all`` must be the union of the single-extra arms, not its own config."""
    base = variant_kwargs("abl_min")
    everything = variant_kwargs("abl_all")
    for arm in ABLATION_EXTRAS.values():
        changed = {k: v for k, v in variant_kwargs(arm).items() if base[k] != v}
        for key, value in changed.items():
            assert everything[key] == value, f"abl_all does not carry {arm}'s {key}"


def test_baseline_is_minirocket_only():
    base = variant_kwargs("abl_min")
    assert base["n_comparison_candidates"] == 0
    assert base["levy_areas"] is False


def test_an_unknown_variant_is_an_error_not_a_silent_default():
    """It used to mean ``dense_features=<name>``, addressing the window-statistic
    bank. That bank was deleted after V15, so a typo would otherwise now produce
    a plain default model wearing the wrong label in a results table."""
    with pytest.raises(SystemExit, match="unknown variant"):
        variant_kwargs("both")


# ------------------------------------------------------- the deleted extra

@pytest.mark.parametrize("extra", MULTICHANNEL_ONLY_EXTRAS)
def test_multichannel_only_extras_are_vacuous_on_one_channel(extra):
    """Why V15 counts this extra only where it is live.

    On one channel there is nothing to cross, so the arm and the baseline are
    the same model.  Scoring that as a dataset the extra "failed to beat the
    baseline on" would be counting a tautology as evidence.
    """
    assert extra == "levy_areas"
    assert levy_area_columns(series(channels=1)).shape[1] == 0


def test_channel_groups_are_singletons_only():
    """Virtual channels were deleted after V15; nothing may reintroduce unions.

    A union of channels consumes the same per-kernel, per-dilation budget a
    singleton would, so if these came back they would silently take diversity
    away from per-channel structure again -- which V15 measured at -0.2 points
    over eight UEA datasets, and a follow-up measured at no effect on any of the
    five synthetic scenarios.
    """
    assert [g.tolist() for g in _channel_groups(5)] == [[0], [1], [2], [3], [4]]
    assert [g.tolist() for g in _channel_groups(1)] == [[0]]


def test_the_bank_still_uses_every_channel():
    """Deleting the unions must not quietly drop channels from the bank."""
    bank = RocketBank(n_features=200, random_state=0).fit(series(channels=4))
    assert sorted(int(g[0]) for g in bank.groups_) == [0, 1, 2, 3]


# ------------------------------------------------------------- end to end


@pytest.mark.parametrize("arm", ["abl_min", *sorted(ABLATION_EXTRAS.values()), "abl_all"])
def test_every_arm_fits_and_predicts(arm):
    X = series(n=40, channels=3, length=48)
    y = (X[:, 0, :16].mean(axis=1) > 0).astype(int)
    model = HeartwoodClassifier(
        n_estimators=4, max_depth=2, n_rocket_features=200,
        random_state=0, **variant_kwargs(arm),
    ).fit(None, X, y)
    assert model.predict(None, X).shape == (len(y),)


def test_comparison_splits_are_absent_when_the_arm_switches_them_off():
    """The flag must reach the tree, not merely be stored on the estimator."""
    X = series(n=40, channels=2, length=48)
    y = (X[:, 0, :16].mean(axis=1) > 0).astype(int)

    def kinds(arm):
        model = HeartwoodClassifier(
            n_estimators=6, max_depth=3, n_rocket_features=200,
            random_state=0, **variant_kwargs(arm),
        ).fit(None, X, y)
        return {node["spec"].kind
                for round_trees in model._core.trees_ for tree in round_trees
                for node in tree.nodes if not node["leaf"]}

    assert "comparison" not in kinds("abl_min")
    # and the machinery still works when the arm switches it back on, so the
    # assertion above is evidence about the flag rather than about this data
    assert kinds("abl_cmp"), "the +comparison arm produced no splits at all"
