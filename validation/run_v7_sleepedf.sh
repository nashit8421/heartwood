#!/usr/bin/env bash
# V7, sleepedf branch. Split from the single orchestrator so the two datasets run
# concurrently: each job is single-threaded on an 11-core box, so sequential
# execution left ~10 cores idle and turned a 12h night into an 18h one.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local name="$1"; shift
          say "BEGIN $name"
          if "$@"; then say "OK    $name"; else say "FAILED $name (recorded, continuing)"; fi }
stage "sleepedf arm D (statics)"  python validation/run_validation.py \
      --datasets sleepedf --sizes 100 1000 2000 --seeds 3 --variants rocket --max-test 4000 \
      --representations agg static_only --out validation/rerun/v7_sleepedf
stage "sleepedf arm C (no statics)" python validation/run_validation.py \
      --datasets sleepedf --sizes 1000 --seeds 3 --variants rocket --drop-static --max-test 4000 \
      --representations agg --out validation/rerun/v7_sleepedf_nostatic
stage "sleepedf arm B (ridge only)" python validation/arm_b.py sleepedf 1000
say "V7_SLEEPEDF_COMPLETE"
