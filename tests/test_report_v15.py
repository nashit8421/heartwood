"""The V15 report must apply the pre-registered bar, not something near it.

`VALIDATION_V15.md` §7 puts the bars in the report script precisely so they
cannot drift once scores exist.  That only helps if the script implements them
correctly, so these tests feed it fabricated results whose verdict is known by
construction: an extra worth exactly nothing, an extra worth a clear +2, and an
extra that clears the bar on a minority.  A report that rounded a 4-of-8 up to a
pass, or averaged unpaired seeds, would produce a table that looks entirely
ordinary.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "report_v15", ROOT / "validation" / "report_v15.py")
report_v15 = importlib.util.module_from_spec(_spec)
sys.modules["report_v15"] = report_v15
_spec.loader.exec_module(report_v15)

EXTRAS = report_v15.ABLATION_EXTRAS
DATASETS = [f"uea:D{i}" for i in range(8)]
SEEDS = range(5)


def cell(dataset, model, seed, score):
    return {"dataset": dataset, "model": model, "n_train": 100, "seed": seed,
            "fit_seconds": 1.0, "metrics": {"balanced_accuracy": score}}


def write(tmp_path, lifts):
    """``lifts[extra][dataset]`` in points; baseline is a flat 0.500."""
    rows = []
    for d in DATASETS:
        for s in SEEDS:
            rows.append(cell(d, "heartwood_abl_min", s, 0.500))
            for extra, arm in EXTRAS.items():
                rows.append(cell(d, f"heartwood_{arm}", s,
                                 0.500 + lifts[extra].get(d, 0.0) / 100.0))
            total = sum(lifts[e].get(d, 0.0) for e in EXTRAS) / 100.0
            rows.append(cell(d, "heartwood_abl_all", s, 0.500 + total))
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    return tmp_path


def run(tmp_path, out):
    subprocess.run(
        [sys.executable, str(ROOT / "validation" / "report_v15.py"),
         "--run", str(tmp_path), "--out", str(out)],
        check=True, capture_output=True, cwd=ROOT,
    )
    return out.read_text()


def test_a_worthless_extra_is_marked_for_deletion(tmp_path):
    lifts = {e: {} for e in EXTRAS}          # every extra worth exactly zero
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    for extra in EXTRAS:
        assert f"| `{extra}` | 0 / 8 |" in text
    assert "FAIL — delete" in text
    assert "PASS — survives" not in text
    assert "Extras surviving §3: **none**" in text


def test_a_clearly_useful_extra_survives(tmp_path):
    first = next(iter(EXTRAS))
    lifts = {e: {} for e in EXTRAS}
    lifts[first] = {d: 2.0 for d in DATASETS}
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    assert f"| `{first}` | 8 / 8 | +2.0 | **PASS — survives**" in text
    assert f"`{first}`" in text.split("Extras surviving §3:")[1].split("\n")[0]


def test_a_bare_minority_does_not_pass(tmp_path):
    """Four of eight is not a majority, and the bar says majority."""
    first = next(iter(EXTRAS))
    lifts = {e: {} for e in EXTRAS}
    lifts[first] = {d: 3.0 for d in DATASETS[:4]}
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    assert f"| `{first}` | 4 / 8 |" in text
    assert "PASS — survives" not in text


def test_five_of_eight_does_pass(tmp_path):
    first = next(iter(EXTRAS))
    lifts = {e: {} for e in EXTRAS}
    lifts[first] = {d: 3.0 for d in DATASETS[:5]}
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    assert f"| `{first}` | 5 / 8 |" in text
    assert "PASS — survives" in text


def test_a_score_just_under_the_bar_does_not_count(tmp_path):
    first = next(iter(EXTRAS))
    lifts = {e: {} for e in EXTRAS}
    lifts[first] = {d: 0.49 for d in DATASETS}
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    assert f"| `{first}` | 0 / 8 |" in text


def test_unpaired_seeds_are_reported_not_averaged(tmp_path):
    lifts = {e: {} for e in EXTRAS}
    write(tmp_path, lifts)
    payload = json.loads((tmp_path / "results.json").read_text())
    arm = f"heartwood_{next(iter(EXTRAS.values()))}"
    payload["results"] = [
        r for r in payload["results"]
        if not (r["model"] == arm and r["dataset"] == DATASETS[0] and r["seed"] == 4)
    ]
    (tmp_path / "results.json").write_text(json.dumps(payload))
    text = run(tmp_path, tmp_path / "out.md")
    assert "Incomplete cells" in text
    assert "unpaired seeds [4]" in text


def test_a_duplicated_cell_is_a_hard_error(tmp_path):
    lifts = {e: {} for e in EXTRAS}
    write(tmp_path, lifts)
    payload = json.loads((tmp_path / "results.json").read_text())
    payload["results"].append(cell(DATASETS[0], "heartwood_abl_min", 0, 0.9))
    (tmp_path / "results.json").write_text(json.dumps(payload))
    with pytest.raises(subprocess.CalledProcessError):
        run(tmp_path, tmp_path / "out.md")


def test_perfectly_additive_extras_pass_h5(tmp_path):
    """``abl_all`` is built as the exact sum here, so additivity must hold."""
    lifts = {e: {d: 1.0 for d in DATASETS} for e in EXTRAS}
    text = run(write(tmp_path, lifts), tmp_path / "out.md")
    assert "| H-V15.5 | additivity | 8 / 8 | — | **PASS** |" in text
