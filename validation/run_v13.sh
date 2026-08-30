#!/usr/bin/env bash
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }
stage "apnea D (statics)" python validation/run_validation.py --datasets apnea --sizes 1000 \
      --seeds 5 --variants rocket_static --max-test 4000 --representations agg static_only \
      --out validation/rerun/v13_apnea
stage "apnea C (no statics)" python validation/run_validation.py --datasets apnea --sizes 1000 \
      --seeds 5 --variants rocket_static --drop-static --max-test 4000 --representations agg \
      --out validation/rerun/v13_apnea_nostatic
stage "sleepedf recheck" python validation/run_validation.py --datasets sleepedf --sizes 1000 \
      --seeds 3 --variants rocket_static --max-test 4000 --representations agg static_only \
      --out validation/rerun/v13_sleepedf
say "V13_COMPLETE"
