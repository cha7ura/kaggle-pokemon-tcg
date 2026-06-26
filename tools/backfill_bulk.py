"""Daily-BULK backfill: for each day (newest -> oldest), download the WHOLE episode
dataset ZIP in one shot (~20GB raw), unzip into json/, compress every file into
replays.sqlite (verified roundtrip) and DELETE the raw, then step to the previous day.

Per-file download (fetch_day.py) is too slow at ~10k files/day; this pulls the day's
single dataset archive once. Dedup is automatic (ingest skips episodes already in the db),
so re-running a day or resuming after a crash is safe. If json/ already holds files at
startup (a prior run was interrupted mid-ingest), those are ingested first WITHOUT
re-downloading.

  python tools/backfill_bulk.py                      # start yesterday-ish, walk back
  python tools/backfill_bulk.py 2026-06-25           # explicit start day, walk back
  python tools/backfill_bulk.py 2026-06-25 2026-06-19  # start..stop (inclusive), walk back

Logging: a status line is emitted at every step boundary and at least every LOG_EVERY_S
seconds during long ingests, reporting db rows, db size, new/dup, and rate. All lines are
timestamped and appended to backfill.log.

Kaggle only keeps a rolling ~week of these datasets published; older days 403 (gone) and
the current day may 403 (not published yet). The loop stops after MAX_MISS consecutive
unavailable days.
"""
import sys, os, time, datetime, shutil, subprocess, glob, sqlite3, zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import replays_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = replays_db.JSON_DIR                       # ingest reads from here
DB = replays_db.DB
DS = "kaggle/pokemon-tcg-ai-battle-episodes-"        # + YYYY-MM-DD
MIN_FREE_GB = 35.0                                    # room for one day's zip + unzip (~30GB peak)
MAX_MISS = 3                                          # stop after this many consecutive unavailable days
DL_RETRIES = 4                                        # retry a day's bulk download before giving up
LOG_EVERY_S = 300                                     # emit an ingest status line at least this often (5 min)
LOGFILE = f"{ROOT}/backfill.log"
STATE = f"{ROOT}/.backfill_state"                    # day whose zip is unzipped in json/ (ingest maybe partial)
DONE = f"{ROOT}/.backfill_done"                      # newline list of fully-ingested days (skip re-download)
MANIFEST = f"{ROOT}/backfill_manifest.csv"           # persistent human-readable log of processed zips


def read_state():
    return open(STATE).read().strip() if os.path.exists(STATE) else None


def set_state(day_s):
    with open(STATE, "w") as fh:
        fh.write(day_s)


def clear_state():
    if os.path.exists(STATE):
        os.remove(STATE)


def load_done():
    return set(open(DONE).read().split()) if os.path.exists(DONE) else set()


def mark_done(day_s, files=0, added=0, minutes=0.0):
    """Record a fully-ingested day in the done-list AND the persistent manifest CSV."""
    with open(DONE, "a") as fh:
        fh.write(day_s + "\n")
    new_manifest = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        if new_manifest:
            fh.write("processed_at,day,zip_slug,zip_size_bytes_approx,files_ingested,"
                     "new_rows,db_rows_after,db_size_gb_after,minutes\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{day_s},{DS}{day_s},,"
                 f"{files},{added},{db_rows()},{db_gb():.3f},{minutes:.1f}\n")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def free_gb():
    return shutil.disk_usage(ROOT).free / 1e9


def db_rows():
    if not os.path.exists(DB):
        return 0
    c = sqlite3.connect(DB)
    try:
        return c.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    finally:
        c.close()


def db_gb():
    return os.path.getsize(DB) / 1e9 if os.path.exists(DB) else 0.0


def download_day(day_s):
    """Bulk-download+unzip one day's ZIP into json/. Returns 'ok' | 'gone' | 'fail'.
    On success, records STATE=day_s so an interrupted ingest resumes the correct day."""
    slug = f"{DS}{day_s}"
    last = ""
    for attempt in range(1, DL_RETRIES + 1):
        log(f"  {day_s}: downloading daily zip ({slug}) attempt {attempt}/{DL_RETRIES}...")
        r = subprocess.run(
            [sys.executable, "-m", "kaggle.cli", "datasets", "download", slug,
             "-p", JSON_DIR, "--unzip"],
            capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        last = out.strip().splitlines()[-1] if out.strip() else f"exit {r.returncode}"
        if r.returncode == 0 and glob.glob(f"{JSON_DIR}/*.json"):
            set_state(day_s)
            return "ok"
        if "403" in out or "Forbidden" in out or "not found" in out.lower():
            return "gone"
        log(f"    download failed: {last[:90]}")
    return "fail"


BATCH = 1000          # rows per transaction (fewer fsyncs = much faster bulk load)
ZLEVEL = 3            # zlib level; 3 ~3x faster than 6 for ~15% bigger blobs (disk is plentiful)
WORKERS = max(2, (os.cpu_count() or 4) - 1)            # parallel compressors (zlib releases the GIL)

INSERT_SQL = ("INSERT OR IGNORE INTO replays "
              "(episode_id, team0, team1, reward0, reward1, source, blob) VALUES (?,?,?,?,?,?,?)")


def _compress_one(f):
    """Read+meta+compress one file in a worker thread. Returns (path, row) or (path, None)."""
    try:
        with open(f, "rb") as fh:
            raw = fh.read()
        t0, t1, r0, r1 = replays_db._meta(raw)
        ep = os.path.splitext(os.path.basename(f))[0]
        return f, (ep, t0, t1, r0, r1, "leader", zlib.compress(raw, ZLEVEL))
    except Exception as e:
        return f, None


def ingest_dir(day_s):
    """PARALLEL bulk ingest: compress files across WORKERS threads (zlib frees the GIL) while the
    main thread does serial batched INSERT OR IGNORE + commit, deleting raw only after its batch
    commits. One tuned connection, synchronous=OFF/WAL. Crash-safe + dedup by episode_id PK.
    Returns (rows_added, files_seen, minutes)."""
    files = sorted(glob.glob(f"{JSON_DIR}/*.json"))
    total = len(files)
    start_rows, start_t = db_rows(), time.time()
    last_log = start_t
    log(f"  {day_s}: ingesting {total} files (parallel x{WORKERS}, zlevel={ZLEVEL}, batch={BATCH}) "
        f"| db start {start_rows} rows, {db_gb():.2f}G")

    replays_db._connect().close()                       # ensure schema/migration exists once
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")                 # safe here: backfill is fully re-runnable
    db.execute("PRAGMA temp_store=MEMORY")
    batch, pending, new = [], [], 0

    def flush():
        nonlocal new
        if not batch:
            return
        before = db.total_changes
        db.executemany(INSERT_SQL, batch)
        db.commit()
        new += db.total_changes - before
        for pf in pending:                               # delete only what's now committed
            try: os.remove(pf)
            except OSError: pass
        batch.clear(); pending.clear()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (f, row) in enumerate(ex.map(_compress_one, files, chunksize=16), 1):
            if row is None:
                log(f"    skip {os.path.basename(f)} (read/compress failed)")
            else:
                batch.append(row); pending.append(f)
                if len(batch) >= BATCH:
                    flush()
            now = time.time()
            if now - last_log >= LOG_EVERY_S or i == total:
                rate = i / max(now - start_t, 1e-6)
                eta = (total - i) / rate if rate else 0
                log(f"  {day_s}: {i}/{total} ({100*i//total}%) | +{new} new | "
                    f"db {start_rows + new} rows, {db_gb():.2f}G | {rate:.0f} f/s | ETA {eta/60:.1f}m | "
                    f"free {free_gb():.0f}G")
                last_log = now
    flush()
    db.close()
    minutes = (time.time() - start_t) / 60
    added_total = db_rows() - start_rows
    log(f"  {day_s}: DONE ingest in {minutes:.1f}m | +{added_total} new rows | "
        f"db now {db_rows()} rows, {db_gb():.2f}G | {total/max(minutes*60,1e-6):.0f} f/s")
    return added_total, total, minutes


def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    today = datetime.date(*map(int, os.environ.get("BACKFILL_TODAY", "2026-06-27").split("-")))
    start = (datetime.date(*map(int, sys.argv[1].split("-"))) if len(sys.argv) > 1
             else today - datetime.timedelta(days=1))
    stop = datetime.date(*map(int, sys.argv[2].split("-"))) if len(sys.argv) > 2 else None

    run_start = time.time()
    grand_added = 0
    done = load_done()
    log(f"=== BACKFILL START | db {db_rows()} rows, {db_gb():.2f}G | free {free_gb():.0f}G | "
        f"start {start} -> back, stop {stop or 'auto'} | {len(done)} days already done ===")

    # Resume: if json/ holds files from an interrupted run, ingest them under the day that
    # downloaded them (STATE), not whatever day the loop is on — prevents re-downloading.
    leftover = glob.glob(f"{JSON_DIR}/*.json")
    if leftover:
        rday = read_state() or start.isoformat()
        log(f"RESUME: {len(leftover)} leftover files in json/ belong to {rday}; ingesting first")
        added, files, mins = ingest_dir(rday)
        grand_added += added
        mark_done(rday, files, added, mins); clear_state(); done.add(rday)

    day, miss = start, 0
    while True:
        if stop and day < stop:
            log(f"reached stop day {stop}; done."); break
        if miss >= MAX_MISS:
            log(f"{MAX_MISS} consecutive unavailable days; published window exhausted. done."); break
        if free_gb() < MIN_FREE_GB:
            log(f"LOW DISK {free_gb():.1f}GB < {MIN_FREE_GB}GB; stopping."); break

        day_s = day.isoformat()
        if day_s in done:
            log(f"=== DAY {day_s}: already ingested, skip ==="); day -= datetime.timedelta(days=1); continue

        log(f"=== DAY {day_s} | free {free_gb():.0f}G | db {db_rows()} rows ===")
        status = download_day(day_s)
        if status == "gone":
            log(f"  {day_s}: unavailable (403)."); miss += 1
        elif status == "fail":
            log(f"  {day_s}: download failed after retries; skipping."); miss += 1
        else:
            miss = 0
            added, files, mins = ingest_dir(day_s)
            grand_added += added
            mark_done(day_s, files, added, mins); clear_state(); done.add(day_s)
        day -= datetime.timedelta(days=1)

    log(f"=== BACKFILL FINAL | +{grand_added} rows this run in {(time.time()-run_start)/60:.1f}m | "
        f"db {db_rows()} rows, {db_gb():.2f}G | free {free_gb():.0f}G ===")


if __name__ == "__main__":
    main()
