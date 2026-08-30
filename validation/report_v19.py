"""Evaluate the V19 hypotheses (VALIDATION_V19.md §3) mechanically.

    python validation/report_v19.py [--run v19_uea] [--out RESULTS_V19.md]

The quantiles come from ``NULL_ARMS`` in ``run_validation.py``, so one cannot
join the study after its score is known.  The bar is judged on a single quantile
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
    parser.add_argument("--run", default="v19_uea")
    parser.add_argument("--out", default="RESULTS_V19.md")
    args = parser.parse_args()

    sweep_report.write(args.out, sweep_report.render(
        results=margins_lib.load(args.run),
        title="V19 results — a permutation null that stays put",
        run=args.run,
        arms=run_validation.NULL_ARMS,
        baseline=f"heartwood_{run_validation.NULL_BASELINE}",
        bar=BAR,
        knob_label="null quantile",
        value_format="q={:g}",
        baseline_note="`selection_null=0`, no floor",
        fail_note="Per `VALIDATION_V19.md` §4 this closes item 2 with four "
                  "attempts below bar, and the next commit corrects "
                  "`validation/HEADROOM.md` rather than attempting a fifth.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
