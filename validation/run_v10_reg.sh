#!/usr/bin/env bash
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local n="$1"; shift; say "BEGIN $n"
          if "$@"; then say "OK    $n"; else say "FAILED $n (recorded, continuing)"; fi }
stage "cpsc regression"     python validation/run_validation.py --datasets cpsc2018 \
      --sizes 1000 --seeds 3 --variants rocket_static --representations agg static_only \
      --out validation/rerun/v10_cpsc
stage "sleepedf regression" python validation/run_validation.py --datasets sleepedf \
      --sizes 1000 --seeds 3 --variants rocket_static --max-test 4000 \
      --representations agg static_only --out validation/rerun/v10_sleepedf
say "V10_REG_COMPLETE"
