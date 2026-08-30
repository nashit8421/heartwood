#!/usr/bin/env bash
# Run every pre-registered study end to end, unattended (roadmap items 1-5).
#
# Sequential by construction. Every run script in this project carries the same
# rule -- arms are graded against the other arms in their own study, so two
# grids must never share the machine -- and the way that rule was broken once
# already was a chain that watched the wrong pid. Here there is only ever one
# study in flight because the next line does not start until the previous one
# returns.
#
# Order is fixed by VALIDATION_V21.md section 5: V21 runs after V15, because
# curvature measured on a bank we may be about to shrink answers a question
# about a model that no longer exists.
#
# KNOWN DEVIATION, recorded rather than hidden: V21 runs after V15 in *time*,
# but nothing deletes a failed bank extra in between -- that is a human decision
# and a separate commit. So V21 measures curvature on the un-shrunk bank. The
# ordering constraint is met; the intent behind it is only partly met, and
# whoever reads RESULTS_V21.md needs to know that.
#
# A failing study is reported and the queue continues. One grid that dies at
# hour three must not cost the other seven.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

V14_PID="${1:-}"
if [ -n "$V14_PID" ]; then
  if ! ps -p "$V14_PID" -o command= 2>/dev/null | grep -q "run_v14.sh"; then
    say "note: pid $V14_PID is not run_v14.sh; assuming V14 is already done"
  else
    say "waiting on V14 shell (pid $V14_PID)"
    while kill -0 "$V14_PID" 2>/dev/null; do sleep 60; done
    say "V14 shell exited"
  fi
fi
while pgrep -f "run_validation.py .*v14_cpsc" >/dev/null 2>&1; do
  say "a V14 cell is still running; waiting"; sleep 60
done

say "BEGIN V14 report"
python validation/report_v14.py --out RESULTS_V14.md \
  && say "OK V14 report" || say "FAILED V14 report"

FAILED=""
stage(){
  local name="$1" script="$2"
  say "=========== BEGIN $name"
  if bash "$script"; then
    say "=========== OK $name"
  else
    say "=========== FAILED $name"
    FAILED="$FAILED $name"
  fi
}

# V15 first: it gates V21 and may shrink the library.
stage V15 validation/run_v15.sh
stage V16 validation/run_v16.sh
stage V17 validation/run_v17.sh
stage V18 validation/run_v18.sh
stage V19 validation/run_v19.sh
stage V20 validation/run_v20.sh
stage V21 validation/run_v21.sh
# Last: its Apnea veto is the longest single grid in the queue by a wide margin,
# so everything cheaper lands before it starts.
stage V22 validation/run_v22.sh

say "==================== ALL STUDIES COMPLETE"
[ -n "$FAILED" ] && say "studies that failed:$FAILED" || say "no study failed"
say "results written:"
ls -1 RESULTS_V1*.md RESULTS_V2*.md 2>/dev/null | sed 's/^/  /'
say "RUN_ALL_COMPLETE"
