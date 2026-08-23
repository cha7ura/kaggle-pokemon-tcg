"""Recompress replays.sqlite blobs from legacy zlib -> lzma/xz preset 9|EXTREME (~6x smaller:
measured 18.85G -> ~3.1G). Lossless: every blob is verified to round-trip before it is written.

  PYTHONUTF8=1 python tools/recompress_db.py            # migrate all zlib blobs, then VACUUM
  PYTHONUTF8=1 python tools/recompress_db.py --no-vacuum

Design:
- THREADED: zlib.decompress + lzma.compress both release the GIL, so a thread pool gives real
  parallelism with zero inter-process blob copying. Main thread does all UPDATEs on one connection.
- RESUMABLE / idempotent: only rows whose blob is NOT already xz are processed, so a re-run after
  an interruption just finishes the remainder. Uniqueness by rowid.
- SAFE: synchronous=OFF is fine because the migration is fully re-runnable; per-batch commits keep
  the rollback journal bounded. Nothing is deleted; blobs are only rewritten in place.
"""
import os, sys, sqlite3, zlib, lzma, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/replays.sqlite"
_XZ = b"\xfd7zXZ\x00"                                  # xz stream magic
NOT_XZ = "substr(blob,1,6) != x'FD377A585A00'"        # SQL predicate: blob is not yet lzma
WORKERS = max(2, (os.cpu_count() or 4) - 1)
BATCH = 400                                            # rows per transaction / per parallel wave
PRESET = 9 | lzma.PRESET_EXTREME


def _recompress(item):
    """(rowid, zlib_blob) -> (rowid, xz_blob), verifying the round-trip. GIL-free hot path."""
    rowid, blob = item
    raw = zlib.decompress(blob)
    nb = lzma.compress(raw, preset=PRESET)
    if lzma.decompress(nb) != raw:
        raise AssertionError(f"roundtrip failed rowid={rowid}")
    return rowid, nb


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    do_vacuum = "--no-vacuum" not in sys.argv
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")              # stay in WAL (no exclusive lock needed)
    db.execute("PRAGMA synchronous=OFF")               # safe: re-runnable migration
    db.execute("PRAGMA temp_store=MEMORY")

    total = db.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    todo = [r for (r,) in db.execute(f"SELECT rowid FROM replays WHERE {NOT_XZ}")]
    start_sz = os.path.getsize(DB) / 1e9
    print(f"recompress: {len(todo)}/{total} rows need migration (rest already lzma) | "
          f"db {start_sz:.2f}G | {WORKERS} threads, preset 9e", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
    else:
        old_b = new_b = done = 0
        t0 = last = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for chunk in _chunks(todo, BATCH):
                qs = ",".join("?" * len(chunk))
                rows = db.execute(f"SELECT rowid, blob FROM replays WHERE rowid IN ({qs})",
                                  chunk).fetchall()
                old_b += sum(len(b) for _, b in rows)
                out = list(ex.map(_recompress, rows))
                new_b += sum(len(b) for _, b in out)
                db.executemany("UPDATE replays SET blob=? WHERE rowid=?",
                               [(nb, rid) for rid, nb in out])
                db.commit()
                done += len(out)
                if done % (BATCH * 12) < BATCH:        # bound WAL growth on this 18G rewrite
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                now = time.time()
                if now - last >= 20 or done >= len(todo):
                    rate = done / max(now - t0, 1e-6)
                    eta = (len(todo) - done) / rate if rate else 0
                    print(f"  {done}/{len(todo)} ({100*done//len(todo)}%) | "
                          f"blobs {old_b/1e9:.2f}G -> {new_b/1e9:.2f}G ({100*new_b/max(old_b,1):.0f}%) | "
                          f"{rate:.0f} rows/s | ETA {eta/60:.1f}m", flush=True)
                    last = now
        print(f"migrated {done} rows in {(time.time()-t0)/60:.1f}m | "
              f"blobs {old_b/1e9:.2f}G -> {new_b/1e9:.2f}G", flush=True)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()

    if do_vacuum:
        print("VACUUM (physically reclaiming freed pages)...", flush=True)
        t0 = time.time()
        v = sqlite3.connect(DB)
        v.execute("VACUUM")
        v.close()
        print(f"VACUUM done in {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"FINAL db size: {os.path.getsize(DB)/1e9:.2f}G (was {start_sz:.2f}G)", flush=True)


if __name__ == "__main__":
    main()
