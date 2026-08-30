"""Evaluate the V21 hypotheses (VALIDATION_V21.md §3) mechanically.

    python validation/report_v21.py [--run v21_uea] [--out RESULTS_V21.md]

The widths come from ``NONLINEAR_ARMS`` in ``run_validation.py``, so one cannot
join the study after its score is known.  The bar is judged on a single width
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

BAR = 1.0   # roadmap item 4's bar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v21_uea")
    parser.add_argument("--out", default="RESULTS_V21.md")
    args = parser.parse_args()

    sweep_report.write(args.out, sweep_report.render(
        results=margins_lib.load(args.run),
        title="V21 results — a nonlinear base that keeps its exact hold-out",
        run=args.run,
        arms=run_validation.NONLINEAR_ARMS,
        baseline=f"heartwood_{run_validation.NONLINEAR_BASELINE}",
        bar=BAR,
        knob_label="random-feature width",
        value_format="D={:g}",
        baseline_note="`nonlinear_features=0`, the linear ridge",
        fail_note="Per `VALIDATION_V21.md` §4 the map stays in the library at its "
                  "no-op default of 0 and the linear base remains the shipped one.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
