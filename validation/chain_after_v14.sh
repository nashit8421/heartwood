#!/usr/bin/env bash
# Wait for the running V14 grid to finish, then start V15 (VALIDATION_V15.md).
#
# The waiting is not politeness. Every arm in V15 is compared against the other
# arms in V15, so CPU contention with a concurrent grid would land unevenly
# across them and the margins would be partly a scheduling artefact. Every run
# script in this project carries the same rule; this enforces it.
# Repository root, wherever this clone happens to live.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

# Pass the pid of the *run_v14.sh shell*, never of a python it spawned.
#
# The first version of this watched the python. run_v14.sh runs each arm as a
# separate python invocation, so that pid died when arm C finished and this
# script launched V15 on top of a V14 that had simply moved on to arm B. The two
# then ran concurrently for 75 minutes -- the exact thing every run script here
# forbids, done by the script written to prevent it. The shell is the process
# that spans all three arms, so the shell is what to wait on.
V14_PID="${1:?usage: chain_after_v14.sh <pid-of-run_v14.sh>}"
if ! ps -p "$V14_PID" -o command= 2>/dev/null | grep -q "run_v14.sh"; then
  say "FATAL: pid $V14_PID is not run_v14.sh -- refusing to chain off it"
  ps -p "$V14_PID" -o pid,command= 2>/dev/null || say "  (no such process)"
  exit 1
fi
say "waiting on V14 shell (pid $V14_PID)"
while kill -0 "$V14_PID" 2>/dev/null; do
  sleep 60
done
say "V14 shell exited"

# Belt as well as braces: nothing of V14's may still be running.
while pgrep -f "run_validation.py .*v14_cpsc" >/dev/null 2>&1; do
  say "a V14 cell is still running; waiting"
  sleep 60
done

# Report V14's own outcome before moving on, so the chain cannot silently
# swallow a failed grid.
if grep -q "V14_COMPLETE" validation/v14.log 2>/dev/null; then
  say "V14 reached V14_COMPLETE"
else
  say "WARNING: V14 exited without V14_COMPLETE -- check validation/v14.log"
fi
grep -E "^\[|FAILED|OK " validation/v14.log | tail -8

# Grade V14 mechanically the moment it lands, before V15's numbers arrive and
# compete for attention. The report refuses to judge on incomplete arms.
say "BEGIN V14 report"
python validation/report_v14.py --out RESULTS_V14.md \
  && say "OK V14 report" || say "FAILED V14 report"

say "starting V15"
bash validation/run_v15.sh
say "CHAIN_COMPLETE"
