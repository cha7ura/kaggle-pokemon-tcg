"""Real-field oracle: score a candidate deck by frequency-weighted win-rate vs the REAL ladder
field (decks extracted from replays by extract_field.py), not the old 4-deck gauntlet. The candidate
is driven by its own pilot; field opponents by the generic typhlosion pilot.

  python tools/oracle.py <deck_slug> [pilot_agent.py] [field_min] [games] [workers]

VALIDATION mode (no args): runs the 4 decks with KNOWN ladder scores through the oracle and prints
the ranking next to the ladder ranking. If the oracle order matches the ladder order, it has the
predictive signal every prior local proxy lacked.
"""
import csv, json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS = f"{ROOT}/autoresearch/decks"
FIELD = f"{DECKS}/field"


def field_decks(field_min):
    man = json.load(open(f"{FIELD}/weights.json"))
    return [(m["slug"], m["count"], m["archetype"]) for m in man if m["count"] >= field_min]


def _one(cand_slug, pilot, opp_slug, games):
    s = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
         "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk", "python:3.11-slim",
         "python", "eval.py", "--challenger", pilot, "--champion", "agent_typh.py",
         "--deck", f"decks/{cand_slug}.csv", "--deck-champion", f"decks/field/{opp_slug}.csv",
         "--games", str(games)],
        capture_output=True, text=True, timeout=900).stdout
    for ln in s.splitlines():
        if '"score"' in ln:
            try: return float(ln.split(":")[1].strip().rstrip(","))
            except: pass
    return None


def oracle(cand_slug, pilot, field_min, games, workers):
    fld = field_decks(field_min)
    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {opp: pool.submit(_one, cand_slug, pilot, opp, games) for opp, _, _ in fld}
    num = den = 0.0
    per = {}
    for opp, cnt, arch in fld:
        sc = futs[opp].result()
        if sc is None: continue
        num += cnt * sc; den += cnt
        per.setdefault(arch, [0.0, 0])
        per[arch][0] += cnt * sc; per[arch][1] += cnt
    pool.shutdown()
    weighted = num / den if den else 0.0
    by_arch = {a: round(v[0] / v[1], 3) for a, v in per.items() if v[1]}
    return weighted, by_arch


# decks with KNOWN ladder scores -> (deck_slug, pilot, ladder_score)
KNOWN = [
    ("alakazam_top", "agent_alakazam.py", 1005.5),
    ("trevenant",    "agent_typh.py",      928.8),
    ("keidroid",     "agent_typh.py",      515.7),
    ("pool_empoleonex", "agent_typh.py",   479.4),
]


def main():
    field_min = 20; games = 30; workers = 8
    if len(sys.argv) == 1:
        # VALIDATION: rank known-ladder decks
        print(f"VALIDATION: oracle vs {len(field_decks(field_min))} real field decks "
              f"(>= {field_min}x), {games} games each\n", flush=True)
        res = []
        for slug, pilot, ladder in KNOWN:
            if not os.path.exists(f"{DECKS}/{slug}.csv"):
                print(f"  skip {slug} (missing)"); continue
            w, ba = oracle(slug, pilot, field_min, games, workers)
            res.append((w, slug, ladder, ba))
            print(f"  {slug:14} oracle={w:.3f}  ladder={ladder:6}  {ba}", flush=True)
        print("\n--- oracle rank vs ladder rank ---")
        for w, slug, ladder, _ in sorted(res, reverse=True):
            print(f"  oracle {w:.3f}  {slug:14} (ladder {ladder})")
        print("\nIf the two orderings match, the oracle predicts the ladder.")
        return
    slug = sys.argv[1]
    pilot = sys.argv[2] if len(sys.argv) > 2 else "agent_typh.py"
    field_min = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    w, ba = oracle(slug, pilot, field_min, games, workers)
    print(f"{slug}: oracle={w:.3f}  by-archetype={ba}")


if __name__ == "__main__":
    main()
