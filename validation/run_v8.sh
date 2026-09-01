#!/usr/bin/env bash
# V8: does pricing selection bias remove the five-point tax, and convert it?
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local n="$1"; shift; say "BEGIN $n"
          if "$@"; then say "OK    $n"; else say "FAILED $n (recorded, continuing)"; fi }
DS=$1
stage "$DS arm D (statics)"    python validation/run_validation.py \
      --datasets $DS --sizes 1000 --seeds 3 --variants rocket_null \
      ${MAXTEST:+--max-test $MAXTEST} --representations agg static_only \
      --out validation/rerun/v8_${DS}
stage "$DS arm C (no statics)" python validation/run_validation.py \
      --datasets $DS --sizes 1000 --seeds 3 --variants rocket_null --drop-static \
      ${MAXTEST:+--max-test $MAXTEST} --representations agg \
      --out validation/rerun/v8_${DS}_nostatic
say "V8_${DS}_COMPLETE"
