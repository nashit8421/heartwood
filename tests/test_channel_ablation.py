"""The V14 channel ablation selects leads by name, or fails loudly.

The whole point of H-V14.1 is that the *only* thing differing between arms is
the width of the series.  If the selection silently took the wrong leads --
positional indices against a reordered loader, or a typo quietly ignored -- the
arms would differ in content as well as width and the comparison would measure
nothing.  These tests exist because that failure would be invisible in the
scores: three arms would still run, still differ, and still look like a result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "run_validation",
    Path(__file__).resolve().parents[1] / "validation" / "run_validation.py",
)
run_validation = importlib.util.module_from_spec(_spec)
sys.modules["run_validation"] = run_validation  # dataclasses resolves __module__
_spec.loader.exec_module(run_validation)
restrict_channels = run_validation.restrict_channels

from validation.loaders import Dataset  # noqa: E402


def make(names):
    n, t = 6, 5
    series = np.stack([
        np.full((n, t), float(i)) for i in range(len(names))
    ], axis=1)
    return Dataset(
        key="fake",
        X_static=np.zeros((n, 0)),
        X_series=series,
        y=np.arange(n) % 2,
        task="binary",
        headline="balanced_accuracy",
        channel_names=list(names),
    )


def test_selection_takes_the_named_lead_not_the_position():
    # Each channel is filled with its own index, so taking the wrong one is
    # visible in the values rather than only in the shape.
    dataset = make(["I", "II", "III", "aVR"])
    restrict_channels(dataset, ["III"])
    assert dataset.X_series.shape[1] == 1
    assert np.all(dataset.X_series[:, 0, :] == 2.0)
    assert dataset.channel_names == ["III"]


def test_order_follows_the_request_not_the_loader():
    dataset = make(["I", "II", "III"])
    restrict_channels(dataset, ["III", "I"])
    assert np.all(dataset.X_series[:, 0, :] == 2.0)
    assert np.all(dataset.X_series[:, 1, :] == 0.0)
    assert dataset.channel_names == ["III", "I"]


def test_a_typo_is_an_error_and_not_a_silent_ablation():
    dataset = make(["I", "II", "III"])
    with pytest.raises(SystemExit) as caught:
        restrict_channels(dataset, ["II", "V2"])
    assert "V2" in str(caught.value)


def test_a_loader_without_channel_names_cannot_be_ablated_by_name():
    dataset = make(["I", "II", "III"])
    dataset.channel_names = []
    with pytest.raises(SystemExit):
        restrict_channels(dataset, ["II"])


def test_the_restriction_is_recorded_in_the_notes():
    # The results file carries notes; the ablation must be legible from it
    # without re-deriving it from the command line.
    dataset = make(["I", "II", "III"])
    restrict_channels(dataset, ["II"])
    assert "channels restricted to ['II']" in dataset.notes


def test_asking_for_every_channel_changes_nothing():
    dataset = make(["I", "II", "III"])
    before = dataset.X_series.copy()
    restrict_channels(dataset, ["I", "II", "III"])
    assert np.array_equal(dataset.X_series, before)
