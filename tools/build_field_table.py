"""Materialize the ladder field into an INDEXED SQL table so the meta read is a fast query
instead of a 45-min Python decompress loop. Decode each replay's two decks ONCE (parallel across
processes — json.loads is GIL-bound, so processes, not threads), tag the archetype by signature
card, and store one row per (episode, player) in `replay_field`, indexed on archetype and deck_sig.

  python tools/build_field_table.py            # (re)build the table from all replays

After this, the meta map is pure SQL, e.g.:
  SELECT archetype, COUNT(*) FROM replay_field GROUP BY archetype ORDER BY 2 DESC;
  SELECT deck_sig, archetype, COUNT(*) c, MAX(team) FROM replay_field GROUP BY deck_sig ORDER BY c DESC LIMIT 20;
"""
import os, sys, csv, json, zlib, sqlite3, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/replays.sqlite"
WORKERS = max(2, (os.cpu_count() or 4) - 1)
CHUNKS = WORKERS * 6                                  # more chunks than workers = better load balance

ARCH = None                                           # per-process archetype map (signature card id -> name)


def build_arch():
    names = {int(r["cardId"]): r["name"]
             for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}
    arch = {743: "Alakazam", 678: "MegaLucario", 1031: "MegaStarmie", 345: "Crustle"}
    for cid, nm in names.items():
        if nm == "Hop's Trevenant": arch.setdefault(cid, "Trevenant")
        if nm == "Dragapult ex": arch.setdefault(cid, "Dragapult")
        if "Bellibolt" in nm: arch.setdefault(cid, "Bellibolt")
        if "Cinderace" in nm: arch.setdefault(cid, "MegaStarmie")
    return arch


def _init():
    global ARCH
    ARCH = build_arch()


def deck_of(steps, p):
    for s in steps:
        a = s[p].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None


def arch_of(deck):
    s = set(deck)
    for cid, nm in ARCH.items():
        if cid in s:
            return nm
    return "OTHER"


def process_chunk(bounds):
    """Decode a rowid range in a worker process. Returns list of (ep, player, archetype, sig, team)."""
    lo, hi = bounds
    db = sqlite3.connect(DB)
    out = []
    for ep, blob in db.execute("SELECT episode_id, blob FROM replays WHERE rowid BETWEEN ? AND ?", (lo, hi)):
        try:
            d = json.loads(zlib.decompress(blob))
        except Exception:
            continue
        teams = d.get("info", {}).get("TeamNames", ["?", "?"])
        steps = d.get("steps", [])
        for p in (0, 1):
            dk = deck_of(steps, p)
            if not dk:
                continue
            sig = "_".join(map(str, sorted(dk)))
            team = str(teams[p])[:24] if p < len(teams) else "?"
            out.append((ep, p, arch_of(dk), sig, team))
    db.close()
    return out


def main():
    t0 = time.time()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # team names contain non-cp1252 chars
    except Exception:
        pass
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS replay_field")
    con.execute("""CREATE TABLE replay_field (
        episode_id TEXT, player INTEGER, archetype TEXT, deck_sig TEXT, team TEXT,
        PRIMARY KEY (episode_id, player))""")
    con.commit()
    max_rowid = con.execute("SELECT MAX(rowid) FROM replays").fetchone()[0] or 0
    step = max(1, max_rowid // CHUNKS + 1)
    bounds = [(lo, min(lo + step - 1, max_rowid)) for lo in range(1, max_rowid + 1, step)]
    print(f"decoding {max_rowid} replays across {len(bounds)} chunks x {WORKERS} processes...", flush=True)

    rows = decoded = 0
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as ex:
        for i, chunk_rows in enumerate(ex.map(process_chunk, bounds), 1):
            con.executemany("INSERT OR IGNORE INTO replay_field VALUES (?,?,?,?,?)", chunk_rows)
            con.commit()
            rows += len(chunk_rows)
            print(f"  chunk {i}/{len(bounds)} | +{len(chunk_rows)} rows | total {rows}", flush=True)

    con.execute("CREATE INDEX idx_field_arch ON replay_field(archetype)")
    con.execute("CREATE INDEX idx_field_sig ON replay_field(deck_sig)")
    con.commit()
    print(f"\nbuilt replay_field: {rows} rows in {(time.time()-t0)/60:.1f}m (indexed on archetype, deck_sig)\n", flush=True)

    print("=== META: field by archetype (weighted by appearances) ===", flush=True)
    for arch, c in con.execute("SELECT archetype, COUNT(*) FROM replay_field GROUP BY archetype ORDER BY 2 DESC"):
        print(f"  {arch:12} {c:6d}", flush=True)
    print("\n=== top 15 distinct decks ===", flush=True)
    for sig, arch, c, team in con.execute(
            "SELECT deck_sig, archetype, COUNT(*) c, MAX(team) FROM replay_field "
            "GROUP BY deck_sig ORDER BY c DESC LIMIT 15"):
        print(f"  x{c:5d} {arch:12} {team:24} {sig[:40]}...", flush=True)
    con.close()


if __name__ == "__main__":
    main()
