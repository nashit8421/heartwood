"""Can this dataset answer the founding claim at all?  Three numbers, hours not days.

    python validation/screen_dataset.py --datasets apnea cpsc2018 sleepedf
    python validation/screen_dataset.py --datasets uea:NATOPS --json out.json

Roadmap item 6, and the roadmap is blunt about why it exists: four attempts,
four datasets, still no fair test of the claim that a model seeing raw series
*and* static covariates beats one seeing either alone.  That is partly bad luck
and partly that this project commits to a full study -- days of compute, a
pre-registration, a write-up -- before knowing whether the dataset can answer
the question.

A dataset can only answer it if three things hold at once:

1. **The statics are informative.**  If they score at chance there is no static
   half to combine, and the study measures the series half twice.  This is
   Sleep-EDF.
2. **The statics are exogenous.**  If the series predicts them, they add nothing
   the series did not already carry, and any "fusion" gain is the series being
   counted twice.  This is age and sex on an ECG -- CPSC-2018.
3. **The regime is temporal.**  If a fixed aggregate table scores as well as the
   raw series, there is no temporal information for the architecture to find and
   the comparison is about tabular modelling.

**Failing any one of them disqualifies the dataset**, and the point is to learn
that in an afternoon rather than after a week of compute and a write-up.

## On the thresholds

They are calibrated to reproduce verdicts this project already paid for --
Sleep-EDF's statics at chance, CPSC's statics predictable from the ECG -- and
then applied to datasets it has not seen.  That is a real limitation and it is
stated rather than hidden: a screen tuned on four known answers is a screen with
four degrees of freedom.  It is still worth far more than committing to a study
on a hunch, which is the alternative it replaces.

The screen never sees a Heartwood model.  Every number is computed from
baselines alone, so it cannot be fitted to the answer it is meant to gate.

## The limitation to keep in mind

Criterion 1 asks whether the statics are informative **on their own**, which is
what the roadmap specifies and what Sleep-EDF failed.  It therefore rejects a
dataset whose statics matter *only through an interaction* -- an XOR of a static
flag and a series feature, where neither side has any marginal signal at all.
That is not a corner case: it is the shape of ``bump_interaction``, and
arguably the purest test of the founding claim there is.

Such a dataset would be screened out.  The screen is a filter for the failures
this project actually hit, not a proof of suitability, and a dataset it rejects
on criterion 1 alone is worth a second look by hand before being dropped.

Every metric is averaged over ``--seeds`` splits, because a single split of a
few hundred rows moves the regime gap by more than the threshold that decides
it -- which was measured while building this, not assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines import build_design_matrix, make_baseline_model
from validation.loaders import MIXED, load_uea

#: A static block must beat chance by this much to count as informative.
INFORMATIVE_LIFT = 0.03
#: If the series predicts a static this well, that static is not exogenous.
EXOGENEITY_CEILING = 0.25
#: Raw series must beat a fixed aggregate table by this much to be a temporal
#: regime rather than a tabular one, in the metric's own units.  Larger than it
#: looks it should be: on a flat series where the mean is a sufficient summary,
#: single splits of 600 rows produced gaps of +/-0.04 from noise alone.
REGIME_GAP = 0.05
#: Splits each metric is averaged over.  Three is enough to put the noise
#: measured above well below the thresholds it would otherwise swamp.
DEFAULT_SEEDS = 3

MAX_ROWS = 1500          # the screen is meant to take minutes
ROUNDS, DEPTH, RATE = 100, 4, 0.1


@dataclass
class Verdict:
    dataset: str
    task: str
    n: int
    n_static: int
    n_channels: int
    static_lift: float
    exogeneity: float
    regime_gap: float
    informative: bool
    exogenous: bool
    temporal: bool
    passes: bool
    notes: str


def _balanced_accuracy(y_true, y_pred) -> float:
    return float(np.mean([
        float((y_pred[y_true == c] == c).mean())
        for c in np.unique(y_true) if (y_true == c).any()
    ]))


def _r2(y_true, y_pred) -> float:
    total = float(((y_true - y_true.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / total


def _split(n, seed=0, holdout=0.3, groups=None):
    """Train/test rows, splitting by *group* whenever the dataset has them.

    This is the third time this project has had to learn the same lesson.  V12
    and V13 were both defects where a row-wise hold-out validated against other
    rows of subjects the model had already seen, and the first version of this
    screen repeated it: on Apnea-ECG, where the statics are constant within a
    subject and one-minute ECG segments identify their subject easily, a row-wise
    exogeneity test read 0.82 and would have disqualified the one dataset the
    roadmap calls this library's first fair test.

    What that number measured was subject recall, not exogeneity.  Splitting by
    subject asks the question actually intended: can the series predict the
    static of a subject the model has never seen?
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(n)
        cut = int(round((1 - holdout) * n))
        return order[:cut], order[cut:]

    groups = np.asarray(groups)
    keys = rng.permutation(np.unique(groups))
    cut = max(1, int(round((1 - holdout) * len(keys))))
    if cut >= len(keys):
        cut = len(keys) - 1
    held = set(keys[cut:].tolist())
    mask = np.array([g in held for g in groups])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def _fit_score(task, design, y, train, test, seed=0) -> float:
    """One baseline tree, fitted and scored. The only model this screen uses."""
    model, _backend = make_baseline_model(task, ROUNDS, DEPTH, RATE, seed)
    model.fit(np.nan_to_num(design[train]), y[train])
    predicted = model.predict(np.nan_to_num(design[test]))
    if task == "regression":
        return _r2(y[test], predicted)
    return _balanced_accuracy(y[test], predicted)


def _chance(task, y, test) -> float:
    """What the metric gives for knowing nothing: the floor each lift is over."""
    if task == "regression":
        return 0.0
    return 1.0 / len(np.unique(y))


def screen(dataset, seed: int = 0, n_seeds: int = DEFAULT_SEEDS) -> Verdict:
    """Screen one dataset, averaging every metric over ``n_seeds`` splits."""
    measurements = [_screen_once(dataset, seed + offset) for offset in range(n_seeds)]
    static_lift = float(np.mean([m[0] for m in measurements]))
    exogeneity = float(np.mean([m[1] for m in measurements]))
    regime_gap = float(np.mean([m[2] for m in measurements]))
    notes = sorted({note for m in measurements for note in m[3]})

    informative = static_lift >= INFORMATIVE_LIFT
    exogenous = exogeneity < EXOGENEITY_CEILING
    temporal = regime_gap >= REGIME_GAP
    if not informative:
        notes.append("statics at chance: nothing to fuse")
    if not exogenous:
        notes.append("statics reconstructable from the series: disqualified")
    if not temporal:
        notes.append("a fixed summary loses nothing: tabular regime")

    series = dataset.X_series
    return Verdict(
        dataset=dataset.key, task=dataset.task, n=len(dataset.y),
        n_static=int(dataset.X_static.shape[1]),
        n_channels=0 if series is None else int(series.shape[1]),
        static_lift=round(static_lift, 4),
        exogeneity=round(exogeneity, 4),
        regime_gap=round(regime_gap, 4),
        informative=bool(informative), exogenous=bool(exogenous),
        temporal=bool(temporal),
        passes=bool(informative and exogenous and temporal),
        notes="; ".join(notes),
    )


def _screen_once(dataset, seed: int):
    """One split's ``(static_lift, exogeneity, regime_gap, notes)``."""
    y = dataset.y
    n = len(y)
    if n > MAX_ROWS:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n, size=MAX_ROWS, replace=False))
        dataset.X_static = dataset.X_static[keep]
        dataset.X_series = dataset.X_series[keep]
        if dataset.groups is not None:
            dataset.groups = dataset.groups[keep]
        y = dataset.y = y[keep]
        n = MAX_ROWS

    task = dataset.task
    statics, series = dataset.X_static, dataset.X_series
    groups = dataset.groups
    train, test = _split(n, seed, groups=groups)
    notes = []
    if groups is not None:
        notes.append("split by subject")

    # 1. Are the statics informative on their own?
    if statics.shape[1] == 0:
        static_lift = 0.0
        notes.append("no static block at all")
    else:
        static_lift = (_fit_score(task, statics, y, train, test, seed)
                       - _chance(task, y, test))

    # 2. Can the series predict the statics?  One regression per static column,
    #    from the aggregate table; the worst offender is what counts, because a
    #    single reconstructable covariate is enough to spoil the comparison.
    exogeneity = 0.0
    if statics.shape[1] and series is not None:
        aggregates = build_design_matrix("agg", np.zeros((n, 0)), series)
        for column in range(statics.shape[1]):
            target = statics[:, column]
            observed = np.isfinite(target)
            if observed.sum() < 50 or np.nanstd(target) <= 0:
                continue
            rows = np.flatnonzero(observed)
            inner_train = np.intersect1d(rows, train)
            inner_test = np.intersect1d(rows, test)
            if len(inner_train) < 30 or len(inner_test) < 15:
                continue
            score = _fit_score("regression", aggregates, target,
                               inner_train, inner_test, seed)
            exogeneity = max(exogeneity, score)

    # 3. Is there temporal information a fixed summary loses?
    if series is None:
        regime_gap = 0.0
        notes.append("no series")
    else:
        # Against the *best* finer representation, not against raw_flat alone.
        # raw_flat hands a tree one column per timestep, which at a few hundred
        # rows is a weak learner however much temporal signal there is: measured
        # here, it lost to the coarse aggregate on a construction where the
        # informative bump is invisible to a global summary by design. Judging
        # the regime by it would have called that dataset tabular.
        blank = np.zeros((n, 0))
        agg = _fit_score(task, build_design_matrix("agg", blank, series),
                         y, train, test, seed)
        finer = max(
            _fit_score(task, build_design_matrix(name, blank, series),
                       y, train, test, seed)
            for name in ("wagg4", "wagg8", "raw_flat")
        )
        regime_gap = finer - agg

    return static_lift, exogeneity, regime_gap, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="splits each metric is averaged over")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    verdicts = []
    for key in args.datasets:
        try:
            dataset = (load_uea(key.split(":", 1)[1]) if key.startswith("uea:")
                       else MIXED[key]())
        except Exception as error:
            print(f"{key}: UNAVAILABLE ({type(error).__name__}: {error})")
            continue
        verdict = screen(dataset, args.seed, args.seeds)
        verdicts.append(verdict)
        mark = "PASS" if verdict.passes else "REJECT"
        print(
            f"{verdict.dataset:<14} {mark:<7} "
            f"static_lift={verdict.static_lift:+.3f} "
            f"({'ok' if verdict.informative else 'FAIL'})  "
            f"exogeneity={verdict.exogeneity:+.3f} "
            f"({'ok' if verdict.exogenous else 'FAIL'})  "
            f"regime_gap={verdict.regime_gap:+.3f} "
            f"({'ok' if verdict.temporal else 'FAIL'})"
            + (f"\n{'':<22}{verdict.notes}" if verdict.notes else "")
        )

    if args.json:
        args.json.write_text(json.dumps({
            "thresholds": {"informative_lift": INFORMATIVE_LIFT,
                           "exogeneity_ceiling": EXOGENEITY_CEILING,
                           "regime_gap": REGIME_GAP},
            "verdicts": [asdict(v) for v in verdicts],
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
