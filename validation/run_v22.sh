#!/usr/bin/env bash
# V22 -- magnitude products (VALIDATION_V22.md).
#
# The Apnea arm is H-V22.2's veto and is NOT optional: V11 collapsed there to
# 0.478 AUC, below chance, and it did so silently. It runs whatever the
# diagnostic says, and a failure withdraws the change.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }
stage(){ local n="$1"; shift; say "BEGIN $n"; if "$@"; then say "OK $n"; else say "FAILED $n"; fi; }

# Cheap first: the synthetic scenarios settle H-V22.1 and H-V22.3 in minutes.
# They live in benchmarks/, not in the validation loaders, because they are
# generators rather than datasets.
stage "synthetic" python benchmarks/run_benchmarks.py \
      --scenarios amp_regression bump_order timing slope_window lead_lag \
      --sizes 500 --seeds 5 --products \
      --out validation/rerun/v22_synth

# The veto. Hours, and it runs regardless of the above.
stage "apnea veto" python validation/run_validation.py \
      --datasets apnea --sizes 1000 --seeds 5 \
      --variants rocket_static prod_split prod_margin prod_both \
      --representations agg \
      --out validation/rerun/v22_apnea

say "V22_COMPLETE -- H-V22.2 is a veto; check the apnea cell before reading the rest"
