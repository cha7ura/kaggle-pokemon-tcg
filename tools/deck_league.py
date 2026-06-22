"""Internal Elo league for decks (co-evolution / AlphaStar-league applied to DECKS).
Population of decks play each other on the engine (typhlosion pilot both sides -> isolates DECK);
Elo emerges. Periodically cull low-Elo, breed top (crossover+mutate), inject pool candidates.
Highest STABLE Elo = most robust deck vs the evolving field. Local = filter; ladder = judge.

  python tools/deck_league.py <rounds> <games_per_pair>
"""
import csv, os, sys, subprocess, random, glob
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS = f"{ROOT}/autoresearch/decks"
BASIC_ENERGY = {2, 3, 4, 5, 6, 7, 8, 9}
RNG = random.Random(7)

# seed population: meta decks + any GA/pool candidates already generated
SEED = ["trevenant", "alakazam_top", "crustle", "lucario_meta", "alakazam", "mist"]
POOL_SLUGS = [os.path.basename(f)[:-4] for f in glob.glob(f"{DECKS}/gen_*.csv")]
POOL_SLUGS += [os.path.basename(f)[:-4] for f in glob.glob(f"{DECKS}/pool_*.csv")]
if os.path.exists(f"{DECKS}/ga_best.csv"): SEED.append("ga_best")


def load(slug): return [int(x) for x in open(f"{DECKS}/{slug}.csv") if x.strip()][:60]
def save(deck, slug): open(f"{DECKS}/{slug}.csv", "w").write("\n".join(map(str, deck)) + "\n")

# card pool for mutation = union of all population decks (coherent)
def build_pool(slugs):
    p = set()
    for s in slugs:
        try: p |= set(load(s))
        except: pass
    return sorted(p)


def legal(deck):
    if len(deck) != 60: return False
    return all(c in BASIC_ENERGY or n <= 4 for c, n in Counter(deck).items())


def mutate(deck, pool, k=3):
    d = deck[:]
    for _ in range(k):
        if RNG.random() < 0.5 and d: d[RNG.randrange(len(d))] = RNG.choice(pool)
        elif d: d.pop(RNG.randrange(len(d))); d.append(RNG.choice(pool))
    cnt = Counter(d); fixed = []
    for c, n in cnt.items(): fixed += [c] * (n if c in BASIC_ENERGY else min(n, 4))
    while len(fixed) < 60: fixed.append(RNG.choice(list(BASIC_ENERGY)))
    fixed = fixed[:60]
    return fixed if legal(fixed) else deck


def crossover(a, b):
    ca, cb = Counter(a), Counter(b); child = []
    for c in set(ca) | set(cb):
        n = ca.get(c, 0) if RNG.random() < 0.5 else cb.get(c, 0)
        child += [c] * n
    cnt = Counter(child[:60]); fixed = []
    for c, n in cnt.items(): fixed += [c] * (n if c in BASIC_ENERGY else min(n, 4))
    while len(fixed) < 60: fixed.append(RNG.choice(list(BASIC_ENERGY)))
    return fixed[:60]


def match(slug_a, slug_b, games):
    """fraction of games slug_a beats slug_b (typhlosion pilot both sides, seat-swapped via eval)."""
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
         "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk", "python:3.11-slim",
         "python", "eval.py", "--challenger", "agent_typh.py", "--champion", "agent_typh.py",
         "--deck", f"decks/{slug_a}.csv", "--deck-champion", f"decks/{slug_b}.csv",
         "--games", str(games)],
        capture_output=True, text=True, timeout=600).stdout
    for ln in out.splitlines():
        if '"score"' in ln:
            try: return float(ln.split(":")[1].strip().rstrip(","))
            except: pass
    return 0.5


def elo_update(ra, rb, sa, k=32):
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    return ra + k * (sa - ea), rb + k * ((1 - sa) - (1 - ea))


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    # materialize population as named decks
    pop = {}
    for s in SEED + POOL_SLUGS:
        try: pop[s] = load(s)
        except: pass
    elo = {s: 1000.0 for s in pop}
    pool = build_pool(list(pop))
    print(f"LEAGUE: {len(pop)} decks, {rounds} rounds x {games} games/pair, pool={len(pool)}", flush=True)
    for rd in range(rounds):
        # Swiss-ish: sort by Elo, pair neighbors (+ a little randomness)
        order = sorted(pop, key=lambda s: -elo[s])
        RNG.shuffle(order) if rd == 0 else None
        pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
        for a, b in pairs:
            sa = match(a, b, games)
            elo[a], elo[b] = elo_update(elo[a], elo[b], sa)
        top = sorted(pop, key=lambda s: -elo[s])
        print(f"round {rd}: " + " | ".join(f"{s}:{elo[s]:.0f}" for s in top[:5]), flush=True)
        # evolve every 3 rounds: cull bottom 2, breed from top 3
        if rd % 3 == 2 and len(pop) >= 6:
            losers = top[-2:]; winners = top[:3]
            for L in losers:
                pa, pb = RNG.sample(winners, 2)
                child = mutate(crossover(pop[pa], pop[pb]), pool, k=RNG.randint(1, 4))
                save(child, L); pop[L] = child; elo[L] = sum(elo[w] for w in winners) / 3  # inherit avg
                print(f"  bred {L} from {pa}+{pb}", flush=True)
    print("\n=== FINAL ELO ===", flush=True)
    for s in sorted(pop, key=lambda s: -elo[s]):
        save(pop[s], f"league_{s}")
        print(f"  {elo[s]:7.0f}  {s}", flush=True)


if __name__ == "__main__":
    main()
