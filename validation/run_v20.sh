#!/usr/bin/env bash
# V20 -- the no-regret guarantee (VALIDATION_V20.md).
#
# comp_base and comp_trees are not optional: the bar is measured against them,
# per cell. The guarded arm costs four fits instead of one, so budget ~4x for
# that arm alone.
#
# Do NOT start this while another study is running.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V20 guarantee grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static comp_base comp_trees noregret \
      --representations agg --no-minirocket \
      --out validation/rerun/v20_uea; then
  say "OK V20 grid"
else
  say "FAILED V20 grid"
fi

say "BEGIN V20 report"
python validation/report_v20.py --run v20_uea --out RESULTS_V20.md \
  && say "OK report" || say "FAILED report"

# VALIDATION_V20.md §5 names Apnea-ECG as the confirmation for the likeliest
# outcome (no cell in the UEA suite ever needed guarding). It is the dataset
# where the regression this guarantee exists for actually happened.
say "V20_UEA_COMPLETE -- see VALIDATION_V20.md section 5 before running Apnea"
