#!/usr/bin/env bash
# V18 -- the out-of-fold bank pre-screen (VALIDATION_V18.md).
#
# sub075 is H-V18.2's control and is not optional: each screened tree fits on
# 75% of its round's rows, so without it a win cannot be told apart from the
# effect of a smaller training set.
#
# Do NOT start this while another study is running.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V18 screening grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static screen04 screen08 screen16 sub075 \
      --representations agg --no-minirocket \
      --out validation/rerun/v18_uea; then
  say "OK V18 grid"
else
  say "FAILED V18 grid"
fi

say "BEGIN V18 report"
python validation/report_v18.py --run v18_uea --out RESULTS_V18.md \
  && say "OK report" || say "FAILED report"
say "V18_COMPLETE"
