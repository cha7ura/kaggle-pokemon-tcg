"""Oracle-rank the harvested limitless decks vs the real ladder field (validated predictor; champion
Alakazam = 0.889). typh pilot baseline — Dragapult/combo will read low (pilot ceiling), Alakazam-type
decks read true. Flags any deck that beats the champion ceiling.

  python tools/oracle_batch.py [games] [field_min]
"""
import sys, os, json, glob
from tools.oracle import oracle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIM = f"{ROOT}/autoresearch/decks/limitless"
CHAMP = 0.889


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    field_min = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    files = sorted(glob.glob(f"{LIM}/*.csv"))
    print(f"oracle {len(files)} harvested decks (typh, {games}g, field_min {field_min}) vs champ {CHAMP}",
          flush=True)
    res = []
    for f in files:
        slug = "limitless/" + os.path.splitext(os.path.basename(f))[0]
        try:
            w, by = oracle(slug, "agent_typh.py", field_min, games, 8)
        except Exception as e:
            print(f"  {slug}: ERR {e}"); continue
        res.append((w, slug, by))
        flag = "  <<< BEATS CHAMP" if w > CHAMP else ""
        print(f"  {w:.3f}  {slug.split('/')[-1]}{flag}", flush=True)
    res.sort(reverse=True)
    print("\n=== RANKED ===", flush=True)
    for w, slug, by in res:
        print(f"  {w:.3f}  {slug.split('/')[-1]}  {by}", flush=True)
    best = res[0] if res else None
    if best and best[0] > CHAMP:
        print(f"\nWINNER: {best[1]} oracles {best[0]:.3f} > champ {CHAMP}", flush=True)
    else:
        print(f"\nnone beat champ {CHAMP} (best {best[0]:.3f} {best[1] if best else ''})", flush=True)


if __name__ == "__main__":
    main()
