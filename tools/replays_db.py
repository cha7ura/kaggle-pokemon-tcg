"""Compress json/ replays into a single SQLite db (zlib blobs) and read them back.

Replaces 12G of json/*.json with replays.sqlite (~1G). Metadata (teams/elo/reward)
is stored as columns so extract_field/extract_decisions can filter without
decompressing. Raw JSON is kept as a zlib-compressed blob.

  python tools/replays_db.py ingest            # dry: compress all, KEEP json (measure ratio)
  python tools/replays_db.py ingest --delete   # compress + delete each json after verify (frees disk)
  python tools/replays_db.py ingest --delete -n 50   # only first 50 (batch)
  python tools/replays_db.py stats             # row count, db size

Reader API (import this module):
  for ep_id, game in iter_replays():  ...      # game = decompressed dict
"""
import json, glob, os, sys, sqlite3, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/replays.sqlite"
JSON_DIR = f"{ROOT}/json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS replays (
  episode_id TEXT PRIMARY KEY,   -- json filename stem
  team0 TEXT, team1 TEXT,        -- info.TeamNames
  reward0 INTEGER, reward1 INTEGER,
  source TEXT DEFAULT 'leader',  -- 'leader' (top-Elo) or 'ours'
  blob BLOB NOT NULL             -- zlib.compress(raw json bytes)
);
"""


def _connect():
    db = sqlite3.connect(DB)
    db.execute(SCHEMA)
    # migrate older dbs that predate the source column
    cols = [r[1] for r in db.execute("PRAGMA table_info(replays)")]
    if "source" not in cols:
        db.execute("ALTER TABLE replays ADD COLUMN source TEXT DEFAULT 'leader'")
        db.commit()
    return db


def has(ep):
    db = _connect()
    hit = db.execute("SELECT 1 FROM replays WHERE episode_id=?", (ep,)).fetchone()
    db.close()
    return hit is not None


def ingest_file(path, source="leader", delete=True):
    """Compress one json file into the db (verified) and optionally delete it.
    Returns False if the episode is already stored."""
    ep = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as fh:
        raw = fh.read()
    blob = zlib.compress(raw, 6)
    assert zlib.decompress(blob) == raw, f"roundtrip failed {ep}"
    t0, t1, r0, r1 = _meta(raw)
    db = _connect()
    new = db.execute("SELECT 1 FROM replays WHERE episode_id=?", (ep,)).fetchone() is None
    db.execute("INSERT OR REPLACE INTO replays "
               "(episode_id, team0, team1, reward0, reward1, source, blob) "
               "VALUES (?,?,?,?,?,?,?)",
               (ep, t0, t1, r0, r1, source, blob))
    db.commit()
    db.close()
    if delete:
        os.remove(path)
    return new


def _meta(raw):
    """Pull cheap columns from raw json bytes. Best-effort; missing -> None."""
    try:
        d = json.loads(raw)
    except Exception:
        return (None, None, None, None)
    teams = d.get("info", {}).get("TeamNames", [None, None])
    rew = d.get("rewards", [None, None])
    g = lambda a, i: a[i] if isinstance(a, list) and len(a) > i else None
    return (g(teams, 0), g(teams, 1), g(rew, 0), g(rew, 1))


def ingest(src_dir=JSON_DIR, source="leader", delete=False, limit=None):
    files = sorted(glob.glob(f"{src_dir}/*.json"))
    if limit:
        files = files[:limit]
    done = raw = 0
    for f in files:
        raw += os.path.getsize(f)
        ingest_file(f, source=source, delete=delete)  # verifies roundtrip before any delete
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(files)}")
    sz = os.path.getsize(DB) if os.path.exists(DB) else 0
    print(f"ingested {done} {source} files (raw {raw/1e9:.2f}G); db now {sz/1e9:.2f}G delete={delete}")


def stats():
    db = _connect()
    n = db.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    by = db.execute("SELECT source, COUNT(*) FROM replays GROUP BY source").fetchall()
    sz = os.path.getsize(DB) if os.path.exists(DB) else 0
    print(f"{n} replays in db; {sz/1e9:.2f}G; by source: {dict(by)}")
    db.close()


def iter_replays(where=None, params=()):
    """Yield (episode_id, decompressed_game_dict). Optional SQL WHERE on metadata cols."""
    db = _connect()
    q = "SELECT episode_id, blob FROM replays"
    if where:
        q += f" WHERE {where}"
    for ep, blob in db.execute(q, params):
        yield ep, json.loads(zlib.decompress(blob))
    db.close()


def _selftest():
    # roundtrip a tiny fake game through compress/decompress
    g = {"info": {"TeamNames": ["a", "b"]}, "rewards": [1, 0], "steps": []}
    raw = json.dumps(g).encode()
    assert json.loads(zlib.decompress(zlib.compress(raw))) == g
    assert _meta(raw) == ("a", "b", 1, 0)
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "selftest":
        _selftest()
    elif a[0] == "ingest":
        n = int(a[a.index("-n") + 1]) if "-n" in a else None
        if "--ours" in a:
            ingest(f"{ROOT}/json_ours", "ours", delete="--delete" in a, limit=n)
        else:
            ingest(JSON_DIR, "leader", delete="--delete" in a, limit=n)
    elif a[0] == "stats":
        stats()
    else:
        print(__doc__)
