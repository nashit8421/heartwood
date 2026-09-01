#!/usr/bin/env bash
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }

# Cheapest arm first: if the ablation is broken, it shows in six minutes, not ten hours.
stage "C 1-lead" python validation/run_validation.py --datasets cpsc2018 --sizes 1000 \
      --seeds 5 --variants rocket_static --drop-static --representations agg \
      --channels II --out validation/rerun/v14_cpsc_ch1
stage "B 3-lead" python validation/run_validation.py --datasets cpsc2018 --sizes 1000 \
      --seeds 5 --variants rocket_static --drop-static --representations agg \
      --channels I II V2 --out validation/rerun/v14_cpsc_ch3
stage "A 12-lead" python validation/run_validation.py --datasets cpsc2018 --sizes 1000 \
      --seeds 5 --variants rocket_static --drop-static --representations agg \
      --channels I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 \
      --out validation/rerun/v14_cpsc_ch12
say "V14_COMPLETE"
