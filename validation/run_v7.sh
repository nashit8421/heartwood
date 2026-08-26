#!/usr/bin/env bash
# V7, start to finish, unattended. Every stage is isolated: a stage that fails
# is recorded and the rest still run, so a bad hour does not cost the night.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
R=validation/rerun

say()   { echo "[$(date '+%F %T')] $*"; }
stage() { local name="$1"; shift
          say "BEGIN $name"
          if "$@"; then say "OK    $name"; else say "FAILED $name (recorded, continuing)"; fi }
waitfor() { local what="$1" pat="$2"
            while pgrep -f "$pat" >/dev/null 2>&1; do sleep 60; done
            say "download finished: $what"; }

say "V7 orchestrator starting"

# ---- Sleep-EDF first: its download finishes long before CPSC's ----
waitfor "sleep-edf" fetch_sleepedf.py
stage "sleepedf arm D (statics)"  python validation/run_validation.py \
      --datasets sleepedf --sizes 100 1000 2000 --seeds 3 --variants rocket \
      --representations agg static_only --out $R/v7_sleepedf
stage "sleepedf arm C (no statics)" python validation/run_validation.py \
      --datasets sleepedf --sizes 1000 --seeds 3 --variants rocket --drop-static \
      --representations agg --out $R/v7_sleepedf_nostatic
stage "sleepedf arm B (ridge only)" python validation/arm_b.py sleepedf 1000

# ---- CPSC-2018 ----
waitfor "cpsc-2018" fetch_cpsc.py
stage "cpsc arm D (statics)"  python validation/run_validation.py \
      --datasets cpsc2018 --sizes 100 1000 2000 --seeds 3 --variants rocket \
      --representations agg static_only --out $R/v7_cpsc
stage "cpsc arm C (no statics)" python validation/run_validation.py \
      --datasets cpsc2018 --sizes 1000 --seeds 3 --variants rocket --drop-static \
      --representations agg --out $R/v7_cpsc_nostatic
stage "cpsc arm B (ridge only)" python validation/arm_b.py cpsc2018 1000

# ---- reports ----
stage "report" python validation/report.py $R/v7_sleepedf $R/v7_cpsc \
      --out validation/RESULTS_V7_RAW.md

say "V7_ORCHESTRATOR_COMPLETE"
