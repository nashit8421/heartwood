"""Paired per-seed margins, shared by the ablation reports.

The guarantees here are the ones this project has had to learn twice: never
average an arm against a baseline over seeds they do not share, and never
silently accept the same cell twice.  Both produce a table that looks entirely
ordinary and is wrong, so they live in one place rather than being reimplemented
per study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRIC = "balanced_accuracy"   # VALIDATION_V5.md headline


def load(run: str) -> list[dict]:
    """Results for a run name under ``validation/rerun``, or an explicit path."""
    candidate = Path(run)
    path = (candidate if candidate.is_dir() else HERE / "rerun" / run) / "results.json"
    if not path.exists():
        raise SystemExit(f"no results at {path} -- has the run finished?")
    return json.loads(path.read_text())["results"]


def by_seed(results, dataset: str, model: str, metric: str = METRIC) -> dict[int, float]:
    """Seed -> score, and a loud failure if a cell was somehow run twice."""
    out: dict[int, float] = {}
    for r in results:
        if r["dataset"] != dataset or r["model"] != model or metric not in r["metrics"]:
            continue
        if r["seed"] in out:
            raise SystemExit(f"duplicate cell: {dataset} {model} seed {r['seed']}")
        out[r["seed"]] = r["metrics"][metric]
    return out


def paired_margin(results, dataset: str, arm: str, baseline: str,
                  metric: str = METRIC) -> tuple[list[float], str]:
    """Per-seed ``arm - baseline`` in points, plus a note on what is missing."""
    base = by_seed(results, dataset, baseline, metric)
    test = by_seed(results, dataset, arm, metric)
    shared = sorted(set(base) & set(test))
    dropped = sorted((set(base) | set(test)) - set(shared))
    note = "" if not dropped else f"unpaired seeds {dropped}"
    return [100.0 * (test[s] - base[s]) for s in shared], note


def mean_score(results, dataset: str, model: str, metric: str = METRIC) -> float:
    values = list(by_seed(results, dataset, model, metric).values())
    return float(np.mean(values)) if values else float("nan")
