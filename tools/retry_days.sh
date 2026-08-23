#!/bin/bash
# Retry the days the forward gap_sweep skipped on the 429 wall. Wide spacing
# (120s between days) to stay under Kaggle's enum limiter; dedup makes the 5
# already-done days (not listed) irrelevant. ponytail: explicit list, edit when
# the backlog changes. socket-timeout guard in fetch_day now turns a throttle
# hang into a fast 60s fail instead of an indefinite stall.
set -u
cd "$(dirname "$0")/.."
for d in 2026-08-01 2026-07-31 2026-07-29 2026-07-28 2026-07-27 2026-07-26 2026-07-25 \
         2026-07-24 2026-07-23 2026-07-22 2026-07-21 2026-07-20 2026-07-19 2026-07-18 \
         2026-07-17 2026-07-16 2026-07-15 2026-07-14 2026-07-13 2026-07-12 2026-07-11 \
         2026-07-10 2026-07-09 2026-07-08; do
  echo "=== $d ($(date +%H:%M)) ==="
  python tools/fetch_day.py "$d" 100000 12 2>&1 | grep -E "got [0-9]+ episode|DONE|Error" || echo "  (crash/no output)"
  sleep 120
done
echo "=== RETRY SWEEP DONE ==="
python tools/replays_db.py stats
