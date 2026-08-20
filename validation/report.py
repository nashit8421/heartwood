"""Turn validation results into the tables and hypothesis verdicts VALIDATION.md promised.

    python validation/report.py validation/credit validation/har ...

Evaluates the pre-registered hypotheses mechanically, so the verdict does not
depend on which numbers I choose to look at.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HEADLINE = {"credit": "roc_auc", "har": "balanced_accuracy"}
DEFAULT_HEADLINE = "balanced_accuracy"
SERIES_MATTERS_MARGIN = 2.0  # points, from VALIDATION.md §6
WIN_MARGIN = 2.0


def headline_for(dataset: str) -> str:
    return HEADLINE.get(dataset, DEFAULT_HEADLINE)


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads((path / "results.json").read_text())
        rows.extend(payload["results"])
        for key, reason in payload.get("unavailable", {}).items():
            print(f"UNAVAILABLE {key}: {reason}")
    return rows


def summarise(rows: list[dict]):
    """(dataset, model, n_train) -> list of headline values across seeds."""
    buckets = defaultdict(list)
    for row in rows:
        metric = headline_for(row["dataset"])
        value = row["metrics"].get(metric)
        if value is not None and np.isfinite(value):
            buckets[(row["dataset"], row["model"], row["n_train"])].append(value)
    return buckets


def render(rows: list[dict]) -> str:
    buckets = summarise(rows)
    datasets = sorted({key[0] for key in buckets})
    lines: list[str] = []

    for dataset in datasets:
        sizes = sorted({key[2] for key in buckets if key[0] == dataset})
        models = sorted({key[1] for key in buckets if key[0] == dataset})
        models = ["heartwood"] + [m for m in models if m != "heartwood"]
        metric = headline_for(dataset)

        lines += ["", f"### {dataset} — {metric}", ""]
        header = f"| {'model':<14s} | " + " | ".join(f"n={s:<9d}" for s in sizes) + " |"
        lines.append(header)
        lines.append("|" + "-" * 16 + "|" + "|".join(["-" * 13] * len(sizes)) + "|")
        for model in models:
            cells = []
            for size in sizes:
                values = buckets.get((dataset, model, size))
                cells.append(
                    f"{np.mean(values):.3f}±{np.std(values):.3f}" if values else "—".ljust(11)
                )
            lines.append(f"| {model:<14s} | " + " | ".join(f"{c:<11s}" for c in cells) + " |")

        deltas = []
        for size in sizes:
            hw = buckets.get((dataset, "heartwood", size))
            agg = buckets.get((dataset, "agg", size))
            if hw and agg:
                deltas.append(f"n={size}: {100 * (np.mean(hw) - np.mean(agg)):+.1f}pt")
        if deltas:
            lines += ["", "Heartwood vs agg — " + ", ".join(deltas)]
    return "\n".join(lines)


def nominal(n_train: int) -> int:
    """Bucket an actual training size to the nominal one it was drawn for.

    Stratified rounding turns a requested 250 into 251, which would otherwise
    silently drop those cells out of the size-specific hypotheses.
    """
    for target in (100, 250, 500, 1000):
        if abs(n_train - target) <= max(5, 0.05 * target):
            return target
    return 0  # "full training split"


def verdicts(rows: list[dict]) -> str:
    """Evaluate the frozen hypotheses. No judgement calls here on purpose.

    H1/H2/H4 are stated in VALIDATION.md for the **mixed** arm (datasets with a
    static block); H3 is stated for the **temporal-only** arm as a single median.
    Applying them to the wrong arm would be a different, easier test.
    """
    buckets = summarise(rows)
    datasets = sorted({key[0] for key in buckets})
    mixed = [d for d in datasets if any(k[1] == "static_only" for k in buckets if k[0] == d)]
    temporal_only = [d for d in datasets if d not in mixed]

    lines = ["", "## Hypothesis verdicts (pre-registered, VALIDATION.md §6)", ""]
    lines.append(f"Mixed arm: {', '.join(mixed) or 'none'}. "
                 f"Temporal-only arm: {', '.join(temporal_only) or 'none'}.")
    lines.append("")

    per_dataset: dict[str, list[float]] = defaultdict(list)
    h1_wins = h1_total = 0
    small_margins, uninformative_gaps = [], []

    for dataset in mixed:
        for size in sorted({k[2] for k in buckets if k[0] == dataset}):
            hw = buckets.get((dataset, "heartwood", size))
            agg = buckets.get((dataset, "agg", size))
            static = buckets.get((dataset, "static_only", size))
            if not (hw and agg and static):
                continue
            hw_mean, agg_mean, static_mean = np.mean(hw), np.mean(agg), np.mean(static)
            margin = (hw_mean - agg_mean) * 100

            if (agg_mean - static_mean) * 100 >= SERIES_MATTERS_MARGIN:
                h1_total += 1
                h1_wins += int(margin >= WIN_MARGIN)
                per_dataset[dataset].append(margin)
                if nominal(size) in (100, 250):
                    small_margins.append((dataset, size, margin))
            else:
                uninformative_gaps.append((dataset, size, (hw_mean - static_mean) * 100))

    rate = h1_wins / h1_total if h1_total else float("nan")
    h1 = "PASS" if rate >= 0.60 else ("FAIL" if rate < 0.50 else "INCONCLUSIVE")
    lines.append(
        f"**H1 (core claim, mixed arm only)** — {h1}. Heartwood beat `agg` by "
        f">= {WIN_MARGIN:.0f} points on {h1_wins}/{h1_total} cells where the series is "
        f"informative ({rate:.0%}); pass needs >=60%, fail is <50%."
    )
    for dataset, margins in sorted(per_dataset.items()):
        wins = sum(1 for m in margins if m >= WIN_MARGIN)
        lines.append(
            f"  {dataset}: {wins}/{len(margins)} cells won, margins "
            + ", ".join(f"{m:+.1f}" for m in margins)
        )

    if small_margins:
        negative = sum(1 for _, _, d in small_margins if d < 0)
        h2 = "FAIL" if negative > len(small_margins) / 2 else "PASS"
        lines.append(
            f"**H2 (small data)** — {h2}. At n=100/250, Heartwood's margin over `agg` was "
            f"negative on {negative}/{len(small_margins)} cells "
            + "(" + ", ".join(f"{d}@{s}:{m:+.1f}" for d, s, m in small_margins) + ")."
        )

    gaps = []
    for dataset in temporal_only:
        hw = [np.mean(v) for k, v in buckets.items()
              if k[0] == dataset and k[1] == "heartwood"]
        mr = [np.mean(v) for k, v in buckets.items()
              if k[0] == dataset and k[1] == "minirocket"]
        if hw and mr:
            gaps.append((dataset, 100 * (np.median(hw) - np.median(mr))))
    if gaps:
        median_gap = float(np.median([g for _, g in gaps]))
        h3 = "PASS" if median_gap >= -5 else ("FAIL" if median_gap < -10 else "MARGINAL")
        lines.append(
            f"**H3 (temporal-only arm vs MiniROCKET)** — {h3}. Median gap over "
            f"{len(gaps)} datasets: {median_gap:+.1f}pt; pass needs >=-5, fail is <-10."
        )
        lines.append("  per dataset: "
                     + ", ".join(f"{d.replace('uea:', '')} {g:+.1f}" for d, g in sorted(gaps, key=lambda x: x[1])))

    if uninformative_gaps:
        harmed = sum(1 for _, _, d in uninformative_gaps if d < -4)
        h4 = "FAIL" if harmed else "PASS"
        lines.append(
            f"**H4 (no harm)** — {h4}. On {len(uninformative_gaps)} cells where the series "
            f"carries nothing, Heartwood was worse than `static_only` by >4 points on "
            f"{harmed}."
        )
    else:
        lines.append("**H4 (no harm)** — not testable: no cell had an uninformative series.")

    # The headroom question: does temporal structure help ANY method?
    lines += ["", "### Headroom: does the series matter at all?", ""]
    for dataset in datasets:
        for size in sorted({k[2] for k in buckets if k[0] == dataset}):
            static = buckets.get((dataset, "static_only", size))
            best_series = [
                np.mean(v) for k, v in buckets.items()
                if k[0] == dataset and k[2] == size and k[1] != "static_only"
            ]
            if static and best_series:
                lift = 100 * (max(best_series) - np.mean(static))
                lines.append(
                    f"  {dataset} n={size}: best series-using method is {lift:+.1f}pt "
                    f"over static-only"
                )
    return "\n".join(lines)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [Path(__file__).parent]
    rows = load(paths)
    if not rows:
        print("no results found")
        return 1
    report = render(rows) + "\n" + verdicts(rows)
    print(report)
    out = Path(__file__).parent / "RESULTS.md"
    out.write_text("# Validation results — real data\n\n"
                   "Generated by `validation/report.py`. Hypotheses and thresholds were "
                   "frozen in `VALIDATION.md` before any of this was run.\n" + report + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
