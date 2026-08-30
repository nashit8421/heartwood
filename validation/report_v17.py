"""Evaluate the V17 hypotheses (VALIDATION_V17.md §3) mechanically.

    python validation/report_v17.py [--run v17_uea] [--out RESULTS_V17.md]

The multipliers come from ``MC_ARMS`` in ``run_validation.py``, so one cannot
join the study after its score is known.  The bar is judged on a single
multiplier applied to every dataset; see ``validation/sweep_report.py`` for why.
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
    parser.add_argument("--run", default="v17_uea")
    parser.add_argument("--out", default="RESULTS_V17.md")
    args = parser.parse_args()

    sweep_report.write(args.out, sweep_report.render(
        results=margins_lib.load(args.run),
        title="V17 results — pricing a node's own selection bias",
        run=args.run,
        arms=run_validation.MC_ARMS,
        baseline=f"heartwood_{run_validation.MC_BASELINE}",
        bar=BAR,
        knob_label="penalty multiplier",
        value_format="{:g}x",
        baseline_note="`mc_penalty=0`, no charge",
        fail_note="Per `VALIDATION_V17.md` §4 the charge stays in the library at its "
                  "no-op default of 0 and the roadmap moves to item 2c.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
