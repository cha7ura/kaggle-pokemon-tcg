"""Full-field tournament: each of OUR contender decks (with its real pilot) vs the ENTIRE 157-deck
real ladder field (opponents piloted by typh). Outputs field-weighted score, raw avg, per-archetype
breakdown, ranked. Heavy but thorough. Results -> /tmp/tournament_results.json + printed.

  python tools/tournament.py [games] [workers]
"""
import csv, json, os, sys, subprocess, statistics as st
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = json.load(open(f"{ROOT}/autoresearch/decks/field/weights.json"))
# (deck_slug, pilot, dragdeck_env_or_empty)
CONTENDERS = [
    ("alakazam_top",    "agent_alakazam.py", ""),
    ("dragapult",       "agent_dragapult.py", "decks/dragapult.csv"),
    ("cand_drag284",    "agent_dragapult.py", "decks/cand_drag284.csv"),
    ("trevenant",       "agent_typh.py", ""),
    ("lucario_meta",    "agent_typh.py", ""),
    ("keidroid",        "agent_typh.py", ""),
    ("pool_empoleonex", "agent_typh.py", ""),
    ("crustle",         "agent_typh.py", ""),
]


def run(cand, pilot, dragdeck, opp, games):
    env = ["-e", f"DRAG_DECK={dragdeck}"] if dragdeck else []
    s = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
         "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk"] + env + ["python:3.11-slim",
         "python", "eval.py", "--challenger", pilot, "--champion", "agent_typh.py",
         "--deck", f"decks/{cand}.csv", "--deck-champion", f"decks/field/{opp}.csv", "--games", str(games)],
        capture_output=True, text=True, timeout=900).stdout
    for ln in s.splitlines():
        if '"score"' in ln:
            try: return float(ln.split(":")[1].strip().rstrip(","))
            except: pass
    return None


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pool = ThreadPoolExecutor(max_workers=workers)
    print(f"TOURNAMENT: {len(CONTENDERS)} contenders x {len(FIELD)} field decks x {games} games", flush=True)
    results = {}
    for cand, pilot, dd in CONTENDERS:
        futs = {m["slug"]: pool.submit(run, cand, pilot, dd, m["slug"], games) for m in FIELD}
        num = den = 0.0
        raw = []
        per = {}
        for m in FIELD:
            sc = futs[m["slug"]].result()
            if sc is None: continue
            w, a = m["count"], m["archetype"]
            num += w * sc; den += w; raw.append(sc)
            per.setdefault(a, [0.0, 0]); per[a][0] += w * sc; per[a][1] += w
        fw = num / den if den else 0.0
        by = {a: round(v[0] / v[1], 3) for a, v in per.items() if v[1]}
        results[cand] = {"field_weighted": round(fw, 3), "raw_avg": round(st.mean(raw), 3) if raw else 0,
                         "by_archetype": by, "pilot": pilot}
        print(f"  {cand:16} field_w={fw:.3f} raw={st.mean(raw):.3f}  {by}", flush=True)
    json.dump(results, open("/tmp/tournament_results.json", "w"), indent=2)
    print("\n=== RANKED by field-weighted score ===", flush=True)
    for c, r in sorted(results.items(), key=lambda kv: -kv[1]["field_weighted"]):
        print(f"  {r['field_weighted']:.3f}  {c:16} (raw {r['raw_avg']})  {r['pilot']}", flush=True)


if __name__ == "__main__":
    main()
