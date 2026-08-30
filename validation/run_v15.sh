#!/usr/bin/env bash
# V15 -- ablate the bank's extras individually (VALIDATION_V15.md).
#
# One invocation, one results.json: run_validation.py checkpoints after every
# cell, so an interruption costs the cell in flight and nothing else, and
# report_v15.py has a single file to read.
#
# Datasets are ordered cheapest-first (measured per-cell fit times from
# validation/rerun/v6b_ueaB): Epilepsy 0.4s, RacketSports 10s, NATOPS 20s,
# Libras 23s, Heartbeat 27s, HandMovementDirection 80s, SelfRegulationSCP2 92s,
# Handwriting 131s.  If an arm is broken it shows in the first minute rather
# than four hours in.
#
# Do NOT start this while another study is running: the arms are compared to one
# another, and CPU contention with a concurrent grid would land unevenly across
# them.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

say "BEGIN V15 UEA ablation grid"
if python validation/run_validation.py \
      --datasets uea:Epilepsy uea:RacketSports uea:NATOPS uea:Libras \
                 uea:Heartbeat uea:HandMovementDirection \
                 uea:SelfRegulationSCP2 uea:Handwriting \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants abl_min abl_vchan abl_cmp abl_stats abl_levy abl_all \
      --representations agg \
      --out validation/rerun/v15_uea; then
  say "OK V15 grid"
else
  say "FAILED V15 grid"
fi

say "BEGIN V15 report"
python validation/report_v15.py --run v15_uea --out RESULTS_V15.md \
  && say "OK report" || say "FAILED report"
say "V15_COMPLETE"
