"""Imitation league + oracle upgrade. Builds a roster of (deck + learned policy) units and runs a
round-robin: every deck plays every deck N games (seat-swapped; the unseeded engine varies shuffle
and coin each game). Every game is stored in replays.sqlite (league_games). Thin decks with no
learned policy fall back to the generic typh pilot.

  python -m tools.imitation.league roundrobin 30          # all >=30g decks, 30 games/pairing
  python -m tools.imitation.league roundrobin 30 f00 f01  # specific field slugs

Live games need docker up (engine is linux/amd64 only). opponent_pilot() routing is pure and unit-
tested without docker.
"""
import os, json, glob, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from tools.imitation.extract_decisions import sig_to_fname
from tools import replays_db

_DBLOCK = threading.Lock()  # sqlite single-writer; serialize stores across shard threads

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POL = os.path.join(os.path.dirname(__file__), "policies")
FIELD_DIR = f"{ROOT}/autoresearch/decks/field"


def opponent_pilot(deck_sig):
    """('imitation', policy_path) if a learned policy exists for this deck, else ('typh', None)."""
    p = os.path.join(POL, f"{sig_to_fname(deck_sig)}.json")
    if os.path.exists(p):
        return "imitation", p
    return "typh", None


def _deck_of(slug):
    return [int(x) for x in open(f"{FIELD_DIR}/{slug}.csv") if x.strip().isdigit()][:60]


def _sig(deck):
    return "_".join(str(x) for x in sorted(deck))


def build_roster(slugs):
    """[{slug, deck, policy(tree|None), pilot}] for the given field slugs."""
    roster = []
    for slug in slugs:
        deck = _deck_of(slug)
        kind, polpath = opponent_pilot(_sig(deck))
        tree = json.load(open(polpath)) if kind == "imitation" else None
        roster.append({"slug": slug, "deck": deck, "policy": tree,
                       "pilot": "imitation" if tree else "typh"})
    return roster


def _learned_slugs():
    """Field slugs whose deck has a learned policy."""
    out = []
    for f in sorted(glob.glob(f"{FIELD_DIR}/f*.csv")):
        slug = os.path.splitext(os.path.basename(f))[0]
        if opponent_pilot(_sig(_deck_of(slug)))[0] == "imitation":
            out.append(slug)
    return out


def _run_shard(rpath, games, shard, nshards, batch):
    """One docker container playing pairings where pair_idx % nshards == shard. Streams to db."""
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
         "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk", "python:3.11-slim",
         "python", "/app/tools/imitation/league_play.py", "/app/.fetch_tmp/roster.json",
         "--games", str(games), "--shard", str(shard), "--nshards", str(nshards)],
        stdout=subprocess.PIPE, text=True)
    buf, total = [], 0
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            buf.append(json.loads(line))
        except Exception:
            continue
        if len(buf) >= 200:
            with _DBLOCK:
                replays_db.store_league_games(buf, batch)
            total += len(buf); buf = []
    proc.wait()
    if buf:
        with _DBLOCK:
            replays_db.store_league_games(buf, batch)
        total += len(buf)
    return total


def roundrobin(slugs=None, games=30, workers=6, batch=None):
    """Round-robin across `workers` parallel docker shards; store every game; return win matrix.
    Default roster = all field decks that have a learned policy."""
    slugs = slugs or _learned_slugs()
    if len(slugs) < 2:
        raise SystemExit(f"need >=2 decks with policies; got {len(slugs)} ({slugs})")
    batch = batch or f"rr-{int(time.time())}"
    npairs = len(slugs) * (len(slugs) - 1) // 2
    workers = max(1, min(workers, npairs))
    json.dump(build_roster(slugs), open(_roster_path(), "w"))
    print(f"[{batch}] {len(slugs)} decks, {npairs} pairings x{games} = {npairs*games} games "
          f"across {workers} shards", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_shard, _roster_path(), games, s, workers, batch)
                for s in range(workers)]
        total = sum(f.result() for f in futs)
    dt = time.time() - t0
    print(f"[{batch}] stored {total} games in {dt:.0f}s ({total/dt:.1f} games/s)", flush=True)
    return batch, replays_db.league_matrix(batch)


def _roster_path():
    os.makedirs(f"{ROOT}/.fetch_tmp", exist_ok=True)
    return f"{ROOT}/.fetch_tmp/roster.json"


def print_matrix(matrix):
    """Win rate of deck_a vs deck_b per pairing."""
    print("\n  deck_a            deck_b            a_wr   n")
    for (a, b), c in sorted(matrix.items()):
        wr = (c["a"] + 0.5 * c["d"]) / c["n"] if c["n"] else 0.0
        print(f"  {a:16}  {b:16}  {wr:.3f}  {c['n']}")


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "roundrobin"
    if cmd == "roundrobin":
        games = int(argv[1]) if len(argv) > 1 else 30
        rest = argv[2:]
        workers = 6
        if "--workers" in rest:
            i = rest.index("--workers"); workers = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
        slugs = rest or None
        batch, m = roundrobin(slugs, games, workers=workers)
        print_matrix(m)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
