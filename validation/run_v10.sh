#!/usr/bin/env bash
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local n="$1"; shift; say "BEGIN $n"
          if "$@"; then say "OK    $n"; else say "FAILED $n (recorded, continuing)"; fi }
stage "apnea arm D (base sees statics)" python validation/run_validation.py \
      --datasets apnea --sizes 1000 --seeds 5 --variants rocket_static --max-test 4000 \
      --representations agg static_only --out validation/rerun/v10_apnea
stage "apnea arm C (no statics)" python validation/run_validation.py \
      --datasets apnea --sizes 1000 --seeds 5 --variants rocket_static --drop-static \
      --max-test 4000 --representations agg --out validation/rerun/v10_apnea_nostatic
say "V10_APNEA_COMPLETE"
