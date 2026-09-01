#!/usr/bin/env bash
# V17 -- the analytic multiple-comparisons charge (VALIDATION_V17.md).
#
# selection_null=1 is run as the H-V17.3 cost comparison: the charge only earns
# its place over the permutation null if it is cheaper as well as no worse.
#
# Do NOT start this while another study is running.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V17 penalty grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static mc050 mc100 mc200 rocket_null \
      --representations agg --no-minirocket \
      --out validation/rerun/v17_uea; then
  say "OK V17 grid"
else
  say "FAILED V17 grid"
fi

say "BEGIN V17 report"
python validation/report_v17.py --run v17_uea --out RESULTS_V17.md \
  && say "OK report" || say "FAILED report"
say "V17_COMPLETE"
