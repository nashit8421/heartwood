#!/usr/bin/env bash
# V9: does a static block a sensor cannot infer finally earn its place?
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local n="$1"; shift; say "BEGIN $n"
          if "$@"; then say "OK    $n"; else say "FAILED $n (recorded, continuing)"; fi }
stage "apnea arm D (with statics)" python validation/run_validation.py \
      --datasets apnea --sizes 1000 --seeds 3 --variants rocket --max-test 4000 \
      --representations agg static_only --out validation/rerun/v9_apnea
stage "apnea arm C (no statics)"   python validation/run_validation.py \
      --datasets apnea --sizes 1000 --seeds 3 --variants rocket --drop-static --max-test 4000 \
      --representations agg --out validation/rerun/v9_apnea_nostatic
say "V9_COMPLETE"
