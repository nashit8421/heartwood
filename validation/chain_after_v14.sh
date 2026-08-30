#!/usr/bin/env bash
# Wait for the running V14 grid to finish, then start V15 (VALIDATION_V15.md).
#
# The waiting is not politeness. Every arm in V15 is compared against the other
# arms in V15, so CPU contention with a concurrent grid would land unevenly
# across them and the margins would be partly a scheduling artefact. Every run
# script in this project carries the same rule; this enforces it.
cd /Users/dogmatixs/Desktop/TS_XGBoost
export PYTHONUNBUFFERED=1
say(){ echo "[$(date '+%F %T')] $*"; }

V14_PID="${1:?usage: chain_after_v14.sh <pid-of-v14>}"
say "waiting on V14 (pid $V14_PID)"
while kill -0 "$V14_PID" 2>/dev/null; do
  sleep 60
done
say "V14 process exited"

# Report V14's own outcome before moving on, so the chain cannot silently
# swallow a failed grid.
if grep -q "V14_COMPLETE" validation/v14.log 2>/dev/null; then
  say "V14 reached V14_COMPLETE"
else
  say "WARNING: V14 exited without V14_COMPLETE -- check validation/v14.log"
fi
grep -E "^\[|FAILED|OK " validation/v14.log | tail -8

say "starting V15"
bash validation/run_v15.sh
say "CHAIN_COMPLETE"
