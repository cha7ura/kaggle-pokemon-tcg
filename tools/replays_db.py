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
import json, glob, os, sys, sqlite3, zlib, lzma, csv, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/replays.sqlite"
JSON_DIR = f"{ROOT}/json"

# Blob codec: replays are stored as either lzma/xz (new, ~6x smaller) or legacy zlib.
# Decompress auto-detects by the xz stream magic so both coexist (during/after migration).
# lzma is stdlib -> works on the Windows host AND inside the linux/amd64 Docker engine image.
_XZ_MAGIC = b"\xfd7zXZ\x00"


def _decompress(blob):
    """Decompress a replay blob, auto-detecting codec (lzma/xz new, zlib legacy)."""
    if blob[:6] == _XZ_MAGIC:
        return lzma.decompress(blob)
    return zlib.decompress(blob)


def _compress(raw):
    """Compress raw json bytes for storage: lzma preset 9|EXTREME (smallest)."""
    return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)

SCHEMA = """
CREATE TABLE IF NOT EXISTS replays (
  episode_id TEXT PRIMARY KEY,   -- json filename stem
  team0 TEXT, team1 TEXT,        -- info.TeamNames
  reward0 INTEGER, reward1 INTEGER,
  source TEXT DEFAULT 'leader',  -- 'leader' (top-Elo) or 'ours'
  blob BLOB NOT NULL             -- zlib.compress(raw json bytes)
);
CREATE TABLE IF NOT EXISTS cards (
  card_id INTEGER PRIMARY KEY, name TEXT, card_type TEXT,
  row_json TEXT                  -- full cards_full.csv row
);
CREATE TABLE IF NOT EXISTS decks (
  sig TEXT PRIMARY KEY, fname TEXT, archetype TEXT, count INTEGER,
  cards_json TEXT                -- the 60 card ids
);
CREATE TABLE IF NOT EXISTS policies (
  deck_sig TEXT, version INTEGER, created REAL, accuracy REAL, note TEXT,
  tree_json TEXT,               -- stdlib-walkable tree
  PRIMARY KEY (deck_sig, version)
);
CREATE TABLE IF NOT EXISTS league_games (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch TEXT, deck_a TEXT, deck_b TEXT, pilot_a TEXT, pilot_b TEXT,
  winner INTEGER,               -- 0=deck_a, 1=deck_b, 2=draw
  steps INTEGER, created REAL,
  trace BLOB                    -- reserved: zlib(move trace) for self-play; NULL for now
);
"""


def _connect():
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
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
    blob = _compress(raw)
    assert _decompress(blob) == raw, f"roundtrip failed {ep}"
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
        yield ep, json.loads(_decompress(blob))
    db.close()


def store_cards(path=f"{ROOT}/autoresearch/cards_full.csv"):
    db = _connect()
    n = 0
    for r in csv.DictReader(open(path)):
        db.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?)",
                   (int(r["cardId"]), r["name"], r.get("cardType"), json.dumps(r)))
        n += 1
    db.commit(); db.close()
    print(f"cards: stored {n}")


def store_decks(field_dir=f"{ROOT}/autoresearch/decks/field"):
    weights = json.load(open(f"{field_dir}/weights.json"))
    db = _connect()
    for w in weights:
        deck = [int(x) for x in open(f"{field_dir}/{w['slug']}.csv") if x.strip().isdigit()]
        sig = "_".join(str(x) for x in sorted(deck))
        db.execute("INSERT OR REPLACE INTO decks VALUES (?,?,?,?,?)",
                   (sig, w["slug"], w.get("archetype"), w.get("count"), json.dumps(deck)))
    db.commit(); db.close()
    print(f"decks: stored {len(weights)}")


def store_policy(deck_sig, tree, accuracy=None, note="", created=None):
    """Append a new policy version for a deck (version = prev max + 1)."""
    db = _connect()
    v = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM policies WHERE deck_sig=?",
                   (deck_sig,)).fetchone()[0]
    db.execute("INSERT INTO policies VALUES (?,?,?,?,?,?)",
               (deck_sig, v, created if created is not None else time.time(),
                accuracy, note, json.dumps(tree)))
    db.commit(); db.close()
    return v


def get_policy(deck_sig, version=None):
    """Latest policy tree for a deck (or a specific version). None if absent."""
    db = _connect()
    if version is None:
        row = db.execute("SELECT tree_json FROM policies WHERE deck_sig=? "
                         "ORDER BY version DESC LIMIT 1", (deck_sig,)).fetchone()
    else:
        row = db.execute("SELECT tree_json FROM policies WHERE deck_sig=? AND version=?",
                         (deck_sig, version)).fetchone()
    db.close()
    return json.loads(row[0]) if row else None


def sync_policies(pol_dir=f"{ROOT}/tools/imitation/policies",
                  manifest=f"{ROOT}/tools/imitation/data/manifest.json", note="", created=None):
    """Ingest current policies/*.json as a new version batch, mapping file-stem -> deck_sig."""
    by_file = {m["file"]: m for m in json.load(open(manifest))}
    n = 0
    for f in glob.glob(f"{pol_dir}/*.json"):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = by_file.get(stem)
        if not m:
            continue
        store_policy(m["deck_sig"], json.load(open(f)),
                     accuracy=m.get("accuracy"), note=note, created=created)
        n += 1
    print(f"policies: stored {n} (new version batch)")


def build_artifacts(note=""):
    store_cards(); store_decks(); sync_policies(note=note)


def store_league_games(rows, batch, created=None):
    """Bulk-insert per-game league results. rows: dicts with
    deck_a, deck_b, pilot_a, pilot_b, winner, steps (trace optional zlib blob)."""
    db = _connect()
    t = created if created is not None else time.time()
    db.executemany(
        "INSERT INTO league_games (batch,deck_a,deck_b,pilot_a,pilot_b,winner,steps,created,trace) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [(batch, r["deck_a"], r["deck_b"], r.get("pilot_a"), r.get("pilot_b"),
          r["winner"], r.get("steps"), t, r.get("trace")) for r in rows])
    db.commit(); db.close()
    return len(rows)


def league_matrix(batch=None):
    """Win counts per ordered (deck_a, deck_b): {(a,b): {'a':wins, 'b':wins, 'd':draws, 'n':games}}."""
    db = _connect()
    q = "SELECT deck_a,deck_b,winner FROM league_games"
    params = ()
    if batch:
        q += " WHERE batch=?"; params = (batch,)
    m = {}
    for a, b, win in db.execute(q, params):
        cell = m.setdefault((a, b), {"a": 0, "b": 0, "d": 0, "n": 0})
        cell["n"] += 1
        cell["a" if win == 0 else "b" if win == 1 else "d"] += 1
    db.close()
    return m


def _selftest():
    # roundtrip a tiny fake game through compress/decompress
    g = {"info": {"TeamNames": ["a", "b"]}, "rewards": [1, 0], "steps": []}
    raw = json.dumps(g).encode()
    assert json.loads(zlib.decompress(zlib.compress(raw))) == g
    assert _meta(raw) == ("a", "b", 1, 0)
    # policy versioning: two stores -> v1, v2; get_policy returns latest
    sig = "__selftest_sig__"
    db = _connect(); db.execute("DELETE FROM policies WHERE deck_sig=?", (sig,)); db.commit(); db.close()
    assert store_policy(sig, {"v": 1}, created=0.0) == 1
    assert store_policy(sig, {"v": 2}, created=0.0) == 2
    assert get_policy(sig) == {"v": 2}
    assert get_policy(sig, version=1) == {"v": 1}
    db = _connect(); db.execute("DELETE FROM policies WHERE deck_sig=?", (sig,)); db.commit(); db.close()
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
    elif a[0] == "artifacts":
        note = a[a.index("--note") + 1] if "--note" in a else ""
        build_artifacts(note=note)
    else:
        print(__doc__)
