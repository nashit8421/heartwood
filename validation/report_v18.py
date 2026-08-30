"""Evaluate the V18 hypotheses (VALIDATION_V18.md §3) mechanically.

    python validation/report_v18.py [--run v18_uea] [--out RESULTS_V18.md]

The shortlist lengths come from ``SCREEN_ARMS`` in ``run_validation.py``, so one
cannot join the study after its score is known.  The bar is judged on a single
length applied to every dataset; see ``validation/sweep_report.py`` for why.
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
    parser.add_argument("--run", default="v18_uea")
    parser.add_argument("--out", default="RESULTS_V18.md")
    args = parser.parse_args()

    sweep_report.write(args.out, sweep_report.render(
        results=margins_lib.load(args.run),
        title="V18 results — screening the bank before offering it",
        run=args.run,
        arms=run_validation.SCREEN_ARMS,
        baseline=f"heartwood_{run_validation.SCREEN_BASELINE}",
        bar=BAR,
        knob_label="shortlist length",
        value_format="top-{:g}",
        baseline_note="`screen_fraction=0`, the bank offered blind",
        fail_note="Per `VALIDATION_V18.md` §4 the screen stays in the library at "
                  "its no-op default of 0 and the roadmap moves to item 2d.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
