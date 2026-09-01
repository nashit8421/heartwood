#!/usr/bin/env bash
# V21 -- the nonlinear base (VALIDATION_V21.md).
#
# Run this AFTER V15: the bank's composition is unsettled until V15 reports, and
# curvature measured on top of a bank we may be about to shrink answers a
# question about a model that no longer exists. VALIDATION_V21.md section 5.
#
# The refit check (H-V21.2) is a veto and lives in the test suite, so it runs
# first and the grid does not start if it fails.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V21 exactness veto"
if ! python -m pytest tests/test_nonlinear_base.py -q; then
  say "FAILED exactness veto -- not running the grid (H-V21.2)"
  exit 1
fi
say "OK exactness veto"

say "BEGIN V21 nonlinear grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static rff0256 rff1024 rff4096 \
      --representations agg --no-minirocket \
      --out validation/rerun/v21_uea; then
  say "OK V21 grid"
else
  say "FAILED V21 grid"
fi

say "BEGIN V21 report"
python validation/report_v21.py --run v21_uea --out RESULTS_V21.md \
  && say "OK report" || say "FAILED report"
say "V21_COMPLETE"
