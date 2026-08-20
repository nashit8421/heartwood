"""The benchmark scenarios, and what each one is actually testing.

Each entry names the temporal question the task turns on, and how aggregation
loses it.  A scenario only belongs here if an oracle can solve it *and* global
aggregation cannot — both properties are regression-tested in
``tests/test_datasets.py``, because during development two scenarios turned out
to be quietly solvable by aggregation, which would have made the whole
comparison measure the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from heartwood import datasets


@dataclass(frozen=True)
class Scenario:
    key: str
    generator: Callable
    task: str  # 'binary' | 'regression'
    question: str  # the temporal question the label depends on
    why_aggregation_fails: str


SCENARIOS: dict[str, Scenario] = {
    "bump_order": Scenario(
        key="bump_order",
        generator=datasets.make_bump_interaction,
        task="binary",
        question="which of two transients happened first, XOR a static flag",
        why_aggregation_fails=(
            "both classes contain the same two shapes and each has zero net area, "
            "so every global statistic sees an identical distribution; the positions "
            "are random, so no fixed window isolates them either"
        ),
    ),
    "timing": Scenario(
        key="timing",
        generator=datasets.make_timing_task,
        task="binary",
        question="did the event happen before this row's static deadline",
        why_aggregation_fails=(
            "every series holds the same transient, so only its position varies, and "
            "the transient has zero net area so its position does not tilt the "
            "global slope"
        ),
    ),
    "slope_window": Scenario(
        key="slope_window",
        generator=datasets.make_slope_window,
        task="binary",
        question="the direction of drift inside one off-grid window, XOR a static gate",
        why_aggregation_fails=(
            "steeper distractor trends elsewhere dominate the global slope, and the "
            "informative window straddles any halves-or-quarters grid"
        ),
    ),
    "amp_regression": Scenario(
        key="amp_regression",
        generator=datasets.make_shape_amplitude_regression,
        task="regression",
        question="the height of a transient in one stretch, times a static coefficient",
        why_aggregation_fails=(
            "a taller nuisance transient elsewhere owns the global maximum, so the "
            "target is only readable in the right window"
        ),
    ),
    "lead_lag": Scenario(
        key="lead_lag",
        generator=datasets.make_lead_lag,
        task="binary",
        question="which of two channels moved first, XOR a static flag",
        why_aggregation_fails=(
            "both channels carry the same transient in both classes, so every "
            "per-channel statistic is blind; the information is in the joint "
            "trajectory, which only a cross-channel quantity can see"
        ),
    ),
    "static_control": Scenario(
        key="static_control",
        generator=datasets.make_static_plus_noise_series,
        task="binary",
        question="nothing temporal at all — the signal is entirely in the static columns",
        why_aggregation_fails=(
            "it does not: this is the control, and aggregation should do fine. What "
            "is under test is whether the temporal machinery costs anything when "
            "offered a pure-noise series and thousands of chances to use it"
        ),
    ),
}

DEFAULT_ORDER = [
    "bump_order", "timing", "slope_window", "amp_regression", "lead_lag",
    "static_control",
]
