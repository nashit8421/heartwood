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

#: Fixed in VALIDATION.md §5 before anything was run: ROC-AUC for the imbalanced
#: binary datasets, balanced accuracy for the multiclass ones.
HEADLINE = {
    "credit": "roc_auc",      # M1, imbalanced binary
    "icu": "roc_auc",         # M2, imbalanced binary
    "har": "balanced_accuracy",  # M3, multiclass
    "ptbxl": "balanced_accuracy",  # V5-A1, multiclass; VALIDATION_V5.md §3
    "cpsc2018": "balanced_accuracy",  # V7-M1, multiclass; VALIDATION_V7.md §2
    "sleepedf": "balanced_accuracy",  # V7-M2, multiclass; VALIDATION_V7.md §2
}
SERIES_MATTERS_MARGIN = 2.0  # points, from VALIDATION.md §6
WIN_MARGIN = 2.0


def headline_for(dataset: str) -> str:
    """The pre-registered metric, or an error — never a silent default.

    An earlier version defaulted unknown datasets to balanced accuracy, which
    quietly scored ICU on the wrong metric and understated it by ~8 points.
    Failing loudly is the only safe behaviour here.
    """
    if dataset.startswith("uea:"):
        return "balanced_accuracy"  # T1 arm, multiclass
    if dataset not in HEADLINE:
        raise KeyError(
            f"no pre-registered headline metric for {dataset!r}; add it to HEADLINE "
            "with the value fixed in VALIDATION.md §5"
        )
    return HEADLINE[dataset]


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
        # Neither kernel budget dominates, so MiniROCKET is credited with its
        # better one -- the comparison should not be won by under-running it.
        mr = [np.mean(v) for k, v in buckets.items()
              if k[0] == dataset and k[1] in ("minirocket", "minirocket10k")]
        if hw and mr:
            gaps.append((dataset, 100 * (np.median(hw) - max(mr))))
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

    # What the NaN fix was worth -- labelled post-hoc, per VALIDATION.md rule 3.
    repairs = []
    for dataset in datasets:
        for size in sorted({k[2] for k in buckets if k[0] == dataset}):
            fixed = buckets.get((dataset, "agg", size))
            naive = buckets.get((dataset, "agg_naive", size))
            if fixed and naive:
                repairs.append((dataset, size, 100 * (np.mean(fixed) - np.mean(naive))))
    if repairs:
        lines += ["", "### Post-hoc (NOT pre-registered): what NaN-skipping was "
                  "worth to `agg`", "",
                  "The v0.3 harness summarised windows with bare NumPy reductions, so one "
                  "missing cell voided the whole statistic. These are the same ten "
                  "statistics computed NaN-aware, which is what Heartwood's own "
                  "`interval_stat` has always done. Zero on a dataset with no missing "
                  "values is the expected result, not a null finding.", ""]
        for dataset, size, delta in repairs:
            lines.append(f"  {dataset} n={size}: `agg` gains {delta:+.1f}pt over `agg_naive`")

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
    # ``--out`` keeps the pre-registered RESULTS.md from being overwritten by a
    # later, differently-configured run: that file is evidence, not a scratch
    # file.  The corrected re-run writes elsewhere (validation/CORRECTION.md).
    argv = list(sys.argv[1:])
    out = Path(__file__).parent / "RESULTS.md"
    if "--out" in argv:
        index = argv.index("--out")
        out = Path(argv[index + 1])
        del argv[index:index + 2]

    paths = [Path(p) for p in argv] or [Path(__file__).parent]
    rows = load(paths)
    if not rows:
        print("no results found")
        return 1
    report = render(rows) + "\n" + verdicts(rows)
    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Validation results — real data\n\n"
                   "Generated by `validation/report.py`. Hypotheses and thresholds were "
                   "frozen in `VALIDATION.md` before any of this was run.\n" + report + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
