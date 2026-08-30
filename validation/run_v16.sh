#!/usr/bin/env bash
# V16 -- per-node bagging against the winner's curse (VALIDATION_V16.md).
#
# Same eight UEA datasets as V15, cheapest-first, so a broken arm shows in the
# first minute.  Bagging makes nodes cheaper, so this grid is faster than V15's.
#
# Do NOT start this while another study is running: the arms are compared to one
# another and CPU contention would land unevenly across them.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V16 bagging grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static bag500 bag250 bag125 \
      --representations agg --no-minirocket \
      --out validation/rerun/v16_uea; then
  say "OK V16 grid"
else
  say "FAILED V16 grid"
fi

say "BEGIN V16 report"
python validation/report_v16.py --run v16_uea --out RESULTS_V16.md \
  && say "OK report" || say "FAILED report"
say "V16_COMPLETE"
