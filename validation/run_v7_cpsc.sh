#!/usr/bin/env bash
# V7, cpsc branch. Split from the single orchestrator so the two datasets run
# concurrently: each job is single-threaded on an 11-core box, so sequential
# execution left ~10 cores idle and turned a 12h night into an 18h one.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local name="$1"; shift
          say "BEGIN $name"
          if "$@"; then say "OK    $name"; else say "FAILED $name (recorded, continuing)"; fi }
stage "cpsc arm D (statics)"  python validation/run_validation.py \
      --datasets cpsc2018 --sizes 100 1000 2000 --seeds 3 --variants rocket \
      --representations agg static_only --out validation/rerun/v7_cpsc
stage "cpsc arm C (no statics)" python validation/run_validation.py \
      --datasets cpsc2018 --sizes 1000 --seeds 3 --variants rocket --drop-static \
      --representations agg --out validation/rerun/v7_cpsc_nostatic
stage "cpsc arm B (ridge only)" python validation/arm_b.py cpsc2018 1000
say "V7_CPSC_COMPLETE"
