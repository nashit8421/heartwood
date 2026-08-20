"""Run the benchmark grid: scenarios × training sizes × seeds × representations.

    python benchmarks/run_benchmarks.py              # the full grid
    python benchmarks/run_benchmarks.py --quick      # a fast smoke run
    python benchmarks/run_benchmarks.py --scenarios timing --sizes 500

Writes an aligned table to stdout plus ``results.json`` (every raw cell, so
later milestones can diff against this one) and ``results.md``.

Every model gets the same budget — same number of rounds, same depth, same
learning rate — and Heartwood runs on its library defaults with no per-scenario
tuning.  The point is to vary the representation and nothing else.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines import REPRESENTATIONS, build_design_matrix, make_baseline_model
from benchmarks.scenarios import DEFAULT_ORDER, SCENARIOS
from heartwood import HeartwoodClassifier, HeartwoodRegressor

HEARTWOOD = "heartwood"
PRIMARY = {"binary": "accuracy", "regression": "rmse"}
LOWER_IS_BETTER = {"rmse", "mae"}

#: --ablation adds these, to show what each Phase-B piece is worth.
ABLATIONS: dict[str, dict] = {
    "hw_phaseA": dict(bank_enabled=False, n_comparison_candidates=0, n_filter_candidates=0),
    "hw_bank": dict(bank_enabled=True, n_comparison_candidates=0, n_filter_candidates=0),
    "hw_filters": dict(bank_enabled=True, n_comparison_candidates=4, n_filter_candidates=8),
}

#: --phasec adds the opt-in Phase C extras.
PHASE_C: dict[str, dict] = {
    "hw_levy": dict(levy_areas=True),
    "hw_dense": dict(dense_base=True),
    "hw_both": dict(levy_areas=True, dense_base=True),
}


# ------------------------------------------------------------------- metrics


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC, so no sklearn dependency for the metric itself."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties, otherwise ties bias the estimate
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float(
        (ranks[y_true == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def classification_metrics(y_true, y_pred, score) -> dict[str, float]:
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    predicted_positive = int((y_pred == 1).sum())
    actual_positive = int((y_true == 1).sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc(y_true, score),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    error = y_pred - y_true
    variance = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - (error**2).sum() / variance) if variance > 0 else float("nan"),
    }


# ---------------------------------------------------------------- one cell


@dataclass
class Result:
    scenario: str
    task: str
    model: str
    n_train: int
    seed: int
    fit_seconds: float
    metrics: dict


def run_cell(args) -> list[Result]:
    """Fit every model on one (scenario, size, seed) draw of the data."""
    scenario_key, n_train, seed, config = args
    scenario = SCENARIOS[scenario_key]

    X_static, X_series, y = scenario.generator(n=n_train, seed=seed)
    X_static_te, X_series_te, y_te = scenario.generator(
        n=config["test_size"], seed=10_000 + seed
    )

    results: list[Result] = []

    def score(model_name, predictions, scores, elapsed):
        metrics = (
            regression_metrics(y_te, predictions)
            if scenario.task == "regression"
            else classification_metrics(y_te, predictions, scores)
        )
        results.append(
            Result(scenario_key, scenario.task, model_name, n_train, seed,
                   elapsed, metrics)
        )

    # --- Heartwood, on library defaults (plus any ablation variants requested)
    estimator = HeartwoodRegressor if scenario.task == "regression" else HeartwoodClassifier
    for name, overrides in config["heartwood_variants"].items():
        started = time.perf_counter()
        model = estimator(
            n_estimators=config["rounds"], max_depth=config["depth"],
            learning_rate=config["learning_rate"], random_state=seed, **overrides,
        ).fit(X_static, X_series, y)
        elapsed = time.perf_counter() - started
        predictions = model.predict(X_static_te, X_series_te)
        scores = (
            None if scenario.task == "regression"
            else model.predict_proba(X_static_te, X_series_te)[:, 1]
        )
        score(name, predictions, scores, elapsed)

    # --- the workarounds
    for name in config["representations"]:
        if name == "static_only" and X_series is None:
            continue
        design = build_design_matrix(name, X_static, X_series)
        design_te = build_design_matrix(name, X_static_te, X_series_te)

        baseline, _ = make_baseline_model(
            scenario.task, config["rounds"], config["depth"],
            config["learning_rate"], seed,
        )
        started = time.perf_counter()
        baseline.fit(design, y)
        elapsed = time.perf_counter() - started
        predictions = baseline.predict(design_te)
        scores = (
            None if scenario.task == "regression"
            else baseline.predict_proba(design_te)[:, 1]
        )
        score(name, predictions, scores, elapsed)

    return results


# ------------------------------------------------------------------ report


def summarise(results: list[Result], metric: str) -> dict:
    """(scenario, model, n_train) -> mean/sd/n over seeds."""
    buckets: dict[tuple, list[float]] = {}
    for row in results:
        buckets.setdefault((row.scenario, row.model, row.n_train), []).append(
            row.metrics[metric] if metric in row.metrics else float("nan")
        )
    return {
        key: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
        for key, v in buckets.items()
    }


def _is_better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def render_tables(results, sizes, models, baselines=None) -> str:
    """One table per scenario: models down, training sizes across.

    ``baselines`` names the competing representations.  Ablation variants of
    Heartwood are *not* baselines: scoring ourselves against our own
    variants would quietly turn every comparison into a much softer one.
    """
    baselines = list(baselines if baselines is not None else
                     [m for m in models if m != HEARTWOOD])
    lines: list[str] = []
    for key in DEFAULT_ORDER:
        rows = [r for r in results if r.scenario == key]
        if not rows:
            continue
        scenario = SCENARIOS[key]
        metric = PRIMARY[scenario.task]
        stats = summarise(rows, metric)
        arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"

        lines.append("")
        lines.append(f"### {key} — {metric} ({arrow})")
        lines.append(f"_{scenario.question}_")
        lines.append("")
        header = f"| {'model':<12s} | " + " | ".join(f"n={s:<11d}" for s in sizes) + " |"
        lines.append(header)
        lines.append("|" + "-" * 14 + "|" + "|".join(["-" * 15] * len(sizes)) + "|")

        best_baseline = {}
        for size in sizes:
            candidates = [
                stats[(key, m, size)]["mean"]
                for m in baselines
                if (key, m, size) in stats
            ]
            if candidates:
                best_baseline[size] = (
                    min(candidates) if metric in LOWER_IS_BETTER else max(candidates)
                )

        for model in models:
            cells = []
            for size in sizes:
                entry = stats.get((key, model, size))
                if entry is None:
                    cells.append(f"{'—':<13s}")
                    continue
                text = f"{entry['mean']:.3f}±{entry['sd']:.3f}"
                if model == HEARTWOOD and size in best_baseline:
                    marker = "*" if _is_better(metric, entry["mean"], best_baseline[size]) else " "
                    text += marker
                cells.append(f"{text:<13s}")
            lines.append(f"| {model:<12s} | " + " | ".join(cells) + " |")

        def deltas_against(reference: dict[int, float]) -> list[str]:
            out = []
            for size in sizes:
                entry = stats.get((key, HEARTWOOD, size))
                if entry is None or size not in reference:
                    continue
                delta = entry["mean"] - reference[size]
                out.append(
                    f"n={size}: {-100 * delta / reference[size]:+.1f}%"
                    if metric in LOWER_IS_BETTER
                    else f"n={size}: {100 * delta:+.1f}pt"
                )
            return out

        plain = {
            size: stats[(key, "agg", size)]["mean"]
            for size in sizes
            if (key, "agg", size) in stats
        }
        lines.append("")
        lines.append("vs agg (the standard workaround) — " + ", ".join(deltas_against(plain)))
        lines.append("vs best-of-all baselines (oracle choice) — "
                     + ", ".join(deltas_against(best_baseline)))
    return "\n".join(lines)


def render_timing(results: list[Result]) -> str:
    per_model: dict[str, list[float]] = {}
    for row in results:
        per_model.setdefault(row.model, []).append(row.fit_seconds)
    lines = ["", "### fit time (seconds, mean over the whole grid)", ""]
    for model, times in sorted(per_model.items(), key=lambda kv: -np.mean(kv[1])):
        lines.append(f"  {model:<12s} {np.mean(times):7.2f}   (max {max(times):.2f})")
    return "\n".join(lines)


# -------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_ORDER)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=0, help="0 = auto")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--quick", action="store_true", help="tiny grid, for smoke tests")
    parser.add_argument("--ablation", action="store_true",
                        help="also run the Phase-B ablation variants")
    parser.add_argument("--phasec", action="store_true",
                        help="also run the opt-in Phase-C variants")
    args = parser.parse_args()

    if args.quick:
        args.sizes, args.seeds, args.rounds = [100, 250], 1, 40

    representations = [name for name in REPRESENTATIONS if name != "static_only"]
    variants = {HEARTWOOD: {}}
    if args.ablation:
        variants.update(ABLATIONS)
    if args.phasec:
        variants.update(PHASE_C)
    models = list(variants) + representations
    config = {
        "test_size": args.test_size, "rounds": args.rounds, "depth": args.depth,
        "learning_rate": args.learning_rate, "representations": representations,
        "heartwood_variants": variants,
    }

    cells = [
        (key, size, seed, config)
        for key in args.scenarios
        for size in args.sizes
        for seed in range(args.seeds)
    ]
    jobs = args.jobs or max(1, min(8, (__import__("os").cpu_count() or 2) - 2))

    print(f"grid: {len(args.scenarios)} scenarios × {len(args.sizes)} sizes × "
          f"{args.seeds} seeds = {len(cells)} cells; {len(models)} models each")
    print(f"budget: {args.rounds} rounds, depth {args.depth}, lr {args.learning_rate}; "
          f"{jobs} worker processes\n")

    started = time.perf_counter()
    results: list[Result] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for done, batch in enumerate(pool.map(run_cell, cells), start=1):
            results.extend(batch)
            print(f"  [{done}/{len(cells)}] {batch[0].scenario} "
                  f"n={batch[0].n_train} seed={batch[0].seed}", flush=True)
    wall_clock = time.perf_counter() - started

    report = (render_tables(results, args.sizes, models, representations)
              + "\n" + render_timing(results))
    print(report)
    print(f"\ntotal wall clock: {wall_clock / 60:.1f} min")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "platform": f"{platform.platform()} / python {platform.python_version()}",
        "wall_clock_seconds": wall_clock,
        "results": [asdict(r) for r in results],
    }, indent=1, default=str))

    (args.out / "results.md").write_text(
        "# Benchmark results\n\n"
        f"Grid: scenarios × n_train {args.sizes} × {args.seeds} seeds, "
        f"test n={args.test_size}. Every model gets {args.rounds} rounds, depth "
        f"{args.depth}, learning rate {args.learning_rate}; Heartwood runs on library "
        "defaults with no per-scenario tuning.\n\n"
        "Cells are mean±sd over seeds. A `*` marks Heartwood beating every baseline "
        "at that size.\n\n"
        "Two comparisons are reported because they answer different questions. **agg** is "
        "the global-aggregate workaround teams actually ship — beating it is the claim "
        "this library makes. **best-of-all baselines** is an oracle: it picks, per task "
        "and per training size, whichever of the five representations turned out best, "
        "which nobody can do in advance. Losing to that oracle on some task is expected; "
        "losing to `agg` would mean the premise is wrong.\n"
        + report
        + f"\n\n_{platform.platform()}, python {platform.python_version()}, "
        f"{wall_clock / 60:.1f} min._\n"
    )
    print(f"wrote {args.out / 'results.json'} and {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
