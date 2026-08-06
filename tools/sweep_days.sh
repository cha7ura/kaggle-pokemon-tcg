#!/bin/bash
# Walk backwards from a start date, fetch_day.py each day into replays.sqlite.
# Stop after 3 consecutive days that CLEANLY enumerate 0 ids (404 = no dataset = pre-comp floor).
# A crash (nonzero exit, e.g. persistent 429) does NOT count as empty: back off and retry same day.
# ponytail: self-terminating on real floor, resilient to rate-limit blips.
set -u
cd "$(dirname "$0")/.."
d="${1:-2026-07-06}"
empty=0
while [ "$empty" -lt 3 ]; do
  echo "=== $d ($(date +%H:%M)) ==="
  out=$(python tools/fetch_day.py "$d" 100000 2>&1); rc=$?
  echo "$out" | grep -E "got [0-9]+ episode|DONE" || echo "$out" | tail -2
  if [ "$rc" -ne 0 ]; then
    echo "  crash (rc=$rc) — backoff 120s, retry same day"; sleep 120; continue
  fi
  ids=$(echo "$out" | grep -oE "got [0-9]+ episode" | grep -oE "[0-9]+" | head -1)
  if [ "${ids:-0}" = "0" ]; then empty=$((empty+1)); else empty=0; fi
  d=$(date -j -v-1d -f "%Y-%m-%d" "$d" +%Y-%m-%d)
  sleep 20   # gentle on the enum API between days
done
echo "=== SWEEP DONE: 3 empty days, floor reached ==="
python tools/replays_db.py stats
