"""The swept-knob report must judge one value, and must print the optimism.

`validation/sweep_report.py` exists to stop a three-value sweep being reported
at its per-dataset best.  That guard is only real if the verdict genuinely comes
from a single value applied everywhere, so these tests build a case where the
two disagree by construction: no single multiplier clears the bar on a majority,
but picking the best one per dataset would clear it everywhere.  A report that
passed that case would be the exact failure `HEADROOM.md` documents.
"""

from __future__ import annotations

import numpy as np

from validation import sweep_report

ARMS = {"k1": 0.5, "k2": 0.25, "k3": 0.125}
DATASETS = [f"uea:D{i}" for i in range(6)]
SEEDS = range(5)
BASELINE = "heartwood_base"


def build(lifts):
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            rows.append({"dataset": dataset, "model": BASELINE, "n_train": 100,
                         "seed": seed, "metrics": {"balanced_accuracy": 0.5}})
            for arm in ARMS:
                rows.append({"dataset": dataset, "model": f"heartwood_{arm}",
                             "n_train": 100, "seed": seed,
                             "metrics": {"balanced_accuracy":
                                         0.5 + lifts[arm].get(dataset, 0.0) / 100}})
    return rows


def render(lifts, bar=1.5):
    return sweep_report.render(
        results=build(lifts), title="T", run="r", arms=ARMS, baseline=BASELINE,
        bar=bar, knob_label="knob", value_format="x{:g}",
        baseline_note="off", fail_note="moves on.")


def test_a_knob_that_does_nothing_fails():
    text = render({a: {} for a in ARMS})
    assert "**FAIL.**" in text
    assert "**PASS.**" not in text


def test_a_knob_that_works_everywhere_passes():
    lifts = {a: {} for a in ARMS}
    lifts["k2"] = {d: 3.0 for d in DATASETS}
    text = render(lifts)
    assert "**PASS.**" in text
    assert "| x0.25 | 6 / 6 |" in text


def test_a_per_dataset_winner_does_not_pass():
    """Each value wins on a third of the datasets; none wins on a majority.

    Choosing per dataset would report +4.0 everywhere. This is the case the
    whole module exists for.
    """
    lifts = {a: {} for a in ARMS}
    for index, dataset in enumerate(DATASETS):
        lifts[list(ARMS)[index % 3]][dataset] = 4.0
    text = render(lifts)
    assert "**FAIL.**" in text
    for arm in ARMS:
        assert f"| 2 / 6 |" in text
    # Cherry-picked +4.0 everywhere; the best single value averages 4.0/3.
    assert "Cherry-picked mean **+4.0**" in text
    assert "tuning optimism of **+2.7** points" in text


def test_the_optimism_is_zero_when_one_value_dominates():
    lifts = {a: {} for a in ARMS}
    lifts["k1"] = {d: 3.0 for d in DATASETS}
    text = render(lifts)
    assert "tuning optimism of **+0.0** points" in text


def test_per_seed_margins_are_always_printed():
    lifts = {a: {} for a in ARMS}
    lifts["k1"] = {d: 3.0 for d in DATASETS}
    text = render(lifts)
    assert "## Per-seed margins" in text
    assert text.count("+3.0") >= len(DATASETS) * len(SEEDS)
