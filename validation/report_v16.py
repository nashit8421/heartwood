"""Evaluate the V16 hypotheses (VALIDATION_V16.md §3) mechanically.

    python validation/report_v16.py [--run v16_uea] [--out RESULTS_V16.md]

The fractions come from ``BAGGING_ARMS`` in ``run_validation.py``, so one cannot
join the study after its score is known.  The bar is judged on a single fraction
applied to every dataset; see ``validation/sweep_report.py`` for why.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from validation import margins as margins_lib, sweep_report

_spec = importlib.util.spec_from_file_location("run_validation", HERE / "run_validation.py")
run_validation = importlib.util.module_from_spec(_spec)
sys.modules["run_validation"] = run_validation
_spec.loader.exec_module(run_validation)

BAR = 1.5   # roadmap item 2's bar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v16_uea")
    parser.add_argument("--out", default="RESULTS_V16.md")
    args = parser.parse_args()

    sweep_report.write(args.out, sweep_report.render(
        results=margins_lib.load(args.run),
        title="V16 results — per-node bagging against the winner's curse",
        run=args.run,
        arms=run_validation.BAGGING_ARMS,
        baseline=f"heartwood_{run_validation.BAGGING_BASELINE}",
        bar=BAR,
        knob_label="bagging fraction",
        value_format="x{:g}",
        baseline_note="`candidate_colsample=1.0`",
        fail_note="Per `VALIDATION_V16.md` §4 the knob stays in the library at its "
                  "no-op default of 1.0 — it is the control arm every later "
                  "selection experiment needs — and the roadmap moves to item 2b.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
