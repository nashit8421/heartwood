#!/usr/bin/env bash
# V24 -- the release gate (VALIDATION_V24.md). Nothing ships until this passes.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }

# Breadth first: sixteen UEA datasets, cheapest first.
stage "uea" python validation/run_validation.py \
      --datasets uea:Epilepsy uea:BasicMotions uea:ERing uea:RacketSports \
                 uea:Libras uea:NATOPS uea:UWaveGestureLibrary \
                 uea:ArticularyWordRecognition uea:StandWalkJump \
                 uea:Heartbeat uea:CharacterTrajectories \
                 uea:HandMovementDirection uea:SelfRegulationSCP2 \
                 uea:Handwriting uea:Cricket uea:EthanolConcentration \
      --sizes 0 --seeds 5 --full-seeds 5 \
      --variants rocket_static --representations agg \
      --out validation/rerun/v24_uea

# The headline: single-lead CPSC, the arm V14 established as the honest one.
stage "cpsc" python validation/run_validation.py \
      --datasets cpsc2018 --sizes 1000 --seeds 5 \
      --variants rocket_static --drop-static --representations agg \
      --channels II --out validation/rerun/v24_cpsc

say "V24_COMPLETE"
