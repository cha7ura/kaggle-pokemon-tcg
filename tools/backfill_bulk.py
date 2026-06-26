"""Daily-BULK backfill: for each day (newest -> oldest), download the WHOLE episode
dataset zip in one shot (~21GB raw), unzip into json/, compress every file into
replays.sqlite (verified roundtrip) and DELETE the raw, then step to the previous day.

Per-file download (fetch_day.py) is too slow at ~10k files/day; this pulls the day's
single dataset archive once. Dedup is automatic (ingest skips episodes already in the db),
so re-running a day or resuming after a crash is safe.

  python tools/backfill_bulk.py                      # start yesterday-ish, walk back
  python tools/backfill_bulk.py 2026-06-25           # explicit start day, walk back
  python tools/backfill_bulk.py 2026-06-25 2026-06-19  # start..stop (inclusive), walk back

Kaggle only keeps a rolling ~week of these datasets published; older days 403 (gone) and
the current day may 403 (not published yet). The loop stops after MAX_MISS consecutive
unavailable days.
"""
import sys, os, datetime, shutil, subprocess, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import replays_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = replays_db.JSON_DIR                       # ingest reads from here
DS = "kaggle/pokemon-tcg-ai-battle-episodes-"        # + YYYY-MM-DD
MIN_FREE_GB = 35.0                                    # need room for one day's zip + unzip (~30GB peak)
MAX_MISS = 3                                          # stop after this many consecutive unavailable days
DL_RETRIES = 4                                        # retry a day's bulk download (429/transient) before giving up


def free_gb():
    return shutil.disk_usage(ROOT).free / 1e9


def download_day(day_s):
    """Bulk-download+unzip one day's dataset into json/. Returns 'ok' | 'gone' | 'fail'.
    'gone' = 403 (unpublished/expired); 'fail' = transient (429/network) after retries."""
    slug = f"{DS}{day_s}"
    last = ""
    for attempt in range(1, DL_RETRIES + 1):
        r = subprocess.run(
            [sys.executable, "-m", "kaggle.cli", "datasets", "download", slug,
             "-p", JSON_DIR, "--unzip"],
            capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        last = out.strip().splitlines()[-1] if out.strip() else f"exit {r.returncode}"
        if r.returncode == 0 and glob.glob(f"{JSON_DIR}/*.json"):
            return "ok"
        if "403" in out or "Forbidden" in out or "not found" in out.lower():
            return "gone"
        print(f"    download attempt {attempt}/{DL_RETRIES} failed: {last[:90]}", flush=True)
    return "fail"


def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    today = datetime.date(*map(int, os.environ.get("BACKFILL_TODAY", "2026-06-27").split("-")))
    start = (datetime.date(*map(int, sys.argv[1].split("-"))) if len(sys.argv) > 1
             else today - datetime.timedelta(days=1))
    stop = datetime.date(*map(int, sys.argv[2].split("-"))) if len(sys.argv) > 2 else None

    replays_db.stats()
    day, miss = start, 0
    while True:
        if stop and day < stop:
            print(f"reached stop day {stop}; done.", flush=True); break
        if miss >= MAX_MISS:
            print(f"{MAX_MISS} consecutive unavailable days; data exhausted. done.", flush=True); break
        if free_gb() < MIN_FREE_GB:
            print(f"LOW DISK {free_gb():.1f}GB < {MIN_FREE_GB}GB; stopping.", flush=True); break

        day_s = day.isoformat()
        print(f"\n=== {day_s} (free {free_gb():.0f}GB) ===", flush=True)
        status = download_day(day_s)
        if status == "gone":
            print(f"  {day_s}: unavailable (403).", flush=True); miss += 1
        elif status == "fail":
            print(f"  {day_s}: download failed after retries; skipping.", flush=True); miss += 1
        else:
            miss = 0
            n = len(glob.glob(f"{JSON_DIR}/*.json"))
            print(f"  {day_s}: downloaded {n} files; ingesting (compress+delete)...", flush=True)
            replays_db.ingest(JSON_DIR, source="leader", delete=True)
        day -= datetime.timedelta(days=1)

    print("\nFINAL:", flush=True)
    replays_db.stats()


if __name__ == "__main__":
    main()
