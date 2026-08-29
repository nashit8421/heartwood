#!/usr/bin/env bash
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }
# H-V12.2 first: does honest group validation cost the V10 win?
stage "apnea, interactions OFF" python validation/run_validation.py --datasets apnea \
      --sizes 1000 --seeds 5 --variants rocket_static --max-test 4000 \
      --representations agg static_only --out validation/rerun/v12_apnea_off
# H-V12.1: with the guard able to see it, are bounded interactions safe?
stage "apnea, interactions ON"  python validation/run_validation.py --datasets apnea \
      --sizes 1000 --seeds 5 --variants rocket_inter --max-test 4000 \
      --representations agg --out validation/rerun/v12_apnea_on
say "V12_APNEA_COMPLETE"
