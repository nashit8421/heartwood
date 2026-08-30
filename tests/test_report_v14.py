"""The V14 report must apply the bar it was given, including the sign clause.

H-V14.1 has two halves: a 3.0-point gap between the 12-lead and 1-lead margins,
*and* the sign of that gap agreeing on at least 4 of 5 seeds. The second half is
the one that disappears when a mean is read off a table by eye, and it is the
half that stops a gap driven by one extreme seed from counting.

These tests build each verdict by construction: a clean width effect, a gap
large enough but inconsistent across seeds, and the outcome VALIDATION_V14.md §5
names as the one it would least like -- the margin surviving at one lead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A": ("v14_cpsc_ch12", 12), "B": ("v14_cpsc_ch3", 3), "C": ("v14_cpsc_ch1", 1)}


def write(root, margins):
    """``margins[arm]`` is a per-seed list of heartwood-minus-baseline points."""
    for arm, (run, _leads) in ARMS.items():
        if arm not in margins:
            continue
        directory = root / run
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed, margin in enumerate(margins[arm]):
            for model, score in (("heartwood_rocket_static", 0.60 + margin / 100),
                                 ("minirocket10k", 0.60), ("minirocket", 0.58),
                                 ("agg", 0.30)):
                rows.append({"dataset": "cpsc2018", "model": model, "n_train": 1000,
                             "seed": seed, "metrics": {"balanced_accuracy": score}})
        (directory / "results.json").write_text(json.dumps({"results": rows}))
    return root


def run(root, out):
    subprocess.run([sys.executable, str(ROOT / "validation" / "report_v14.py"),
                    "--root", str(root), "--out", str(out)],
                   check=True, capture_output=True, cwd=ROOT)
    return out.read_text()


def test_a_clean_width_effect_passes(tmp_path):
    text = run(write(tmp_path, {"A": [6.0] * 5, "B": [3.0] * 5, "C": [0.5] * 5}),
               tmp_path / "o.md")
    assert "| H-V14.1 | width causes it | margin(A)−margin(C) = **+5.5**" in text
    assert "**PASS**" in text
    assert "Width is the mechanism." in text


def test_a_big_gap_driven_by_one_seed_fails_the_sign_clause(tmp_path):
    """Mean gap clears 3.0, but A beats C on only one seed of five."""
    text = run(write(tmp_path, {"A": [40.0, 0.0, 0.0, 0.0, 0.0],
                                "B": [1.0] * 5,
                                "C": [1.0, 1.0, 1.0, 1.0, 1.0]}), tmp_path / "o.md")
    assert "sign agrees on 1/5 seeds" in text
    assert "| H-V14.1 | width causes it" in text
    assert "**FAIL**" in text


def test_the_margin_surviving_at_one_lead_is_reported_as_the_disliked_outcome(tmp_path):
    text = run(write(tmp_path, {"A": [3.0] * 5, "B": [2.5] * 5, "C": [3.0] * 5}),
               tmp_path / "o.md")
    assert "| H-V14.3 | margin survives at one lead | margin(C) = **+3.0** | **PASS**" in text
    assert "needs correcting a second time" in text
    assert "outcome it would least like" in text


def test_dose_response_is_judged_on_the_ordering(tmp_path):
    rising = run(write(tmp_path / "a", {"A": [6.0] * 5, "B": [3.0] * 5, "C": [1.0] * 5}),
                 tmp_path / "a.md")
    assert "| H-V14.2 | dose response | +6.0 ≥ +3.0 ≥ +1.0 | **PASS**" in rising
    jumbled = run(write(tmp_path / "b", {"A": [1.0] * 5, "B": [6.0] * 5, "C": [3.0] * 5}),
                  tmp_path / "b.md")
    assert "| H-V14.2 | dose response | +1.0 ≥ +6.0 ≥ +3.0 | **FAIL**" in jumbled


def test_a_missing_arm_is_reported_not_judged(tmp_path):
    """§6: a cell that fails to run is reported as failed, never omitted."""
    text = run(write(tmp_path, {"B": [2.0] * 5, "C": [3.0] * 5}), tmp_path / "o.md")
    assert "INCOMPLETE — 2 of 3 arms" in text
    assert "arms A and C not both complete" in text
    assert "Withheld until every arm has run." in text


def test_unpaired_seeds_are_flagged(tmp_path):
    root = write(tmp_path, {"A": [1.0] * 5, "B": [1.0] * 5, "C": [1.0] * 5})
    path = root / "v14_cpsc_ch1" / "results.json"
    payload = json.loads(path.read_text())
    payload["results"] = [r for r in payload["results"]
                          if not (r["model"] == "minirocket10k" and r["seed"] == 4)]
    path.write_text(json.dumps(payload))
    text = run(root, tmp_path / "o.md")
    assert "unpaired seeds [4]" in text
