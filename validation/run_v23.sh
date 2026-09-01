#!/usr/bin/env bash
# V23 -- comparison splits and Levy areas on a suite that did not condemn them
# (VALIDATION_V23.md).
#
# The synthetic control runs FIRST and takes minutes. If the arms cannot
# reproduce the effects on scenarios built for them, the real suite's result is
# uninterpretable (H-V23.3) and there is no point spending hours on it.
#
# Do NOT start this while another study is running.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }

stage "control" python benchmarks/run_benchmarks.py \
      --scenarios bump_order lead_lag timing slope_window \
      --sizes 500 --seeds 5 --v23 \
      --out validation/rerun/v23_synth

# Held-out UEA suite, cheapest first so a broken arm shows in minutes.
stage "uea" python validation/run_validation.py \
      --datasets uea:BasicMotions uea:ERing uea:UWaveGestureLibrary \
                 uea:ArticularyWordRecognition uea:StandWalkJump \
                 uea:CharacterTrajectories uea:Cricket \
                 uea:EthanolConcentration \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants v23_base v23_cmp v23_levy v23_both \
      --representations agg --no-minirocket \
      --out validation/rerun/v23_uea

say "BEGIN V23 report"
python validation/report_v23.py --run v23_uea --control v23_synth --out RESULTS_V23.md \
  && say "OK report" || say "FAILED report"
say "V23_COMPLETE"
