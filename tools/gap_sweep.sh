#!/bin/bash
# One-shot: ingest the backlog gap between the last stored day and today.
# Waits for any in-flight fetch_day to finish, then runs each day 08-04..07-08
# sequentially (concurrent fetches just multiply Kaggle's 429s). Dedup is
# automatic via replays_db.has(), so overlaps are cheap. ponytail: explicit
# date list, no clever range math; edit the list when the gap moves.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "fetch_day.py" >/dev/null; do sleep 30; done   # let the running day finish

for d in 2026-08-04 2026-08-03 2026-08-02 2026-08-01 \
         2026-07-31 2026-07-30 2026-07-29 2026-07-28 2026-07-27 2026-07-26 2026-07-25 \
         2026-07-24 2026-07-23 2026-07-22 2026-07-21 2026-07-20 2026-07-19 2026-07-18 \
         2026-07-17 2026-07-16 2026-07-15 2026-07-14 2026-07-13 2026-07-12 2026-07-11 \
         2026-07-10 2026-07-09 2026-07-08; do
  echo "=== $d ($(date +%H:%M)) ==="
  python tools/fetch_day.py "$d" 100000 12 2>&1 | grep -E "got [0-9]+ episode|DONE" || echo "  (no output)"
  sleep 15   # gentle on the enum API between days
done
echo "=== GAP SWEEP DONE ==="
python tools/replays_db.py stats
