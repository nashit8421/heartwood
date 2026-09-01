#!/usr/bin/env bash
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }
stage "apnea V11" python validation/run_validation.py --datasets apnea --sizes 1000 \
      --seeds 5 --variants rocket_inter --max-test 4000 \
      --representations agg static_only --out validation/rerun/v11_apnea
stage "cpsc V11" python validation/run_validation.py --datasets cpsc2018 --sizes 1000 \
      --seeds 3 --variants rocket_inter --representations agg --out validation/rerun/v11_cpsc
stage "sleepedf V11" python validation/run_validation.py --datasets sleepedf --sizes 1000 \
      --seeds 3 --variants rocket_inter --max-test 4000 --representations agg \
      --out validation/rerun/v11_sleepedf
say "V11_COMPLETE"
