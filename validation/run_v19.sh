#!/usr/bin/env bash
# V19 -- the recalibrated permutation null (VALIDATION_V19.md).
#
# null_v8 is H-V19.2's reference and is not optional: without it a win says
# only "a floor helps", not that the calibration diagnosis was right.
#
# 16 permutations is ~3.0x the fit time (measured), so budget ~3x a V16 grid.
#
# Do NOT start this while another study is running.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V19 null grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static null_q50 null_q90 null_q95 null_v8 \
      --representations agg --no-minirocket \
      --out validation/rerun/v19_uea; then
  say "OK V19 grid"
else
  say "FAILED V19 grid"
fi

say "BEGIN V19 report"
python validation/report_v19.py --run v19_uea --out RESULTS_V19.md \
  && say "OK report" || say "FAILED report"
say "V19_COMPLETE"
