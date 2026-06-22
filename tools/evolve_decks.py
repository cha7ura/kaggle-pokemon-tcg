"""Evolutionary deck search (GA) with self-play fitness. Seed from strong meta decks, mutate
coherently (swap/adjust within a sane card pool, max-4 + 60-card legal), fitness = gauntlet
win-rate vs the meta field (typhlosion pilot both sides -> isolates DECK). Keep top, crossover +
mutate, iterate. Coherence is enforced by fitness (incoherent mutants play badly -> die), not repair.
Local = filter; ladder = judge. Run on host (docker per matchup).

  python tools/evolve_decks.py <generations> <pop> <games>
"""
import csv, os, sys, subprocess, random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS = f"{ROOT}/autoresearch/decks"
rows = list(csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv")))
R = {int(r["cardId"]): r for r in rows}
BASIC_ENERGY = {2, 3, 4, 5, 6, 7, 8, 9}          # unlimited copies allowed
SEED = ["trevenant", "alakazam_top", "crustle", "lucario_meta"]
META = ["alakazam_top", "trevenant", "crustle", "lucario_meta"]
# real meta-share (from replays) -> field-weighted fitness (Crustle is ~1%, Lucario ~29%)
FIELD_W = {"alakazam_top": 0.40, "lucario_meta": 0.33, "trevenant": 0.25, "crustle": 0.02}
FRAGILITY = 0.35   # penalty per point a worst-matchup falls below 0.5 (AlphaStar anti-fragility / PFSP spirit)
# deterministic-ish randomness varied by call (Math.random unavailable note is for workflows; here ok)
RNG = random.Random(1234)


def load(slug):
    return [int(x) for x in open(f"{DECKS}/{slug}.csv") if x.strip()][:60]


def save(deck, slug):
    open(f"{DECKS}/{slug}.csv", "w").write("\n".join(map(str, deck)) + "\n")


# card pool to mutate WITHIN: every card appearing in the seed decks + gen candidates (sane, coherent)
POOL = set()
for s in SEED:
    POOL |= set(load(s))
for f in os.listdir(DECKS):
    if f.startswith("gen_"):
        POOL |= set(load(f[:-4]))
POOL = sorted(POOL)


def legal(deck):
    if len(deck) != 60: return False
    for c, n in Counter(deck).items():
        if c not in BASIC_ENERGY and n > 4: return False
    return True


def mutate(deck, k=3):
    d = deck[:]
    for _ in range(k):
        op = RNG.random()
        if op < 0.45 and d:                       # swap one copy for a pool card
            i = RNG.randrange(len(d)); d[i] = RNG.choice(POOL)
        elif op < 0.75 and d:                     # remove a copy, add a pool card (count shift)
            d.pop(RNG.randrange(len(d))); d.append(RNG.choice(POOL))
        else:                                     # duplicate an existing card (raise a count)
            d[RNG.randrange(len(d))] = RNG.choice(d)
    # enforce 60 + max-4
    cnt = Counter(d); fixed = []
    for c, n in cnt.items():
        fixed += [c] * (n if c in BASIC_ENERGY else min(n, 4))
    while len(fixed) < 60: fixed.append(RNG.choice(list(BASIC_ENERGY)))
    fixed = fixed[:60]
    return fixed if legal(fixed) else deck


def crossover(a, b):
    ca, cb = Counter(a), Counter(b)
    child = []
    for c in set(ca) | set(cb):
        n = ca.get(c, 0) if RNG.random() < 0.5 else cb.get(c, 0)
        child += [c] * n
    # trim/pad to 60 legal
    child = child[:60]
    while len(child) < 60: child.append(RNG.choice(list(BASIC_ENERGY)))
    cnt = Counter(child); fixed = []
    for c, n in cnt.items():
        fixed += [c] * (n if c in BASIC_ENERGY else min(n, 4))
    while len(fixed) < 60: fixed.append(RNG.choice(list(BASIC_ENERGY)))
    return fixed[:60]


def fitness(slug, games):
    scores = []
    for opp in META:
        out = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
             "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk", "python:3.11-slim",
             "python", "eval.py", "--challenger", "agent_typh.py", "--champion", "agent_typh.py",
             "--deck", f"decks/{slug}.csv", "--deck-champion", f"decks/{opp}.csv",
             "--games", str(games)],
            capture_output=True, text=True, timeout=900).stdout
        sc = 0.0
        for ln in out.splitlines():
            if '"score"' in ln:
                try: sc = float(ln.split(":")[1].strip().rstrip(",")); break
                except: pass
        scores.append(sc)
    # field-weighted winrate MINUS fragility penalty (worst matchup below 0.5) = robust objective
    wsum = sum(FIELD_W[o] for o in META)
    fw = sum(FIELD_W[o] * s for o, s in zip(META, scores)) / wsum
    worst = min(scores)
    fit = fw - FRAGILITY * max(0.0, 0.5 - worst)
    return fit, scores


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pop_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    print(f"GA deck search: {gens} gens x {pop_n} pop x {games} games/matchup; pool={len(POOL)} cards", flush=True)
    # init population: seeds + mutations of seeds
    pop = [load(s) for s in SEED]
    while len(pop) < pop_n:
        pop.append(mutate(load(RNG.choice(SEED)), k=RNG.randint(2, 5)))
    best_ever = (0.0, None, None)
    for g in range(gens):
        scored = []
        for i, d in enumerate(pop):
            slug = f"ga_g{g}_i{i}"; save(d, slug)
            avg, sc = fitness(slug, games)
            scored.append((avg, d, sc))
        scored.sort(key=lambda x: -x[0])
        top = scored[: max(2, pop_n // 3)]
        if scored[0][0] > best_ever[0]:
            best_ever = scored[0]; save(best_ever[1], "ga_best")
        print(f"gen {g}: best={scored[0][0]:.3f} {dict(zip(META,[round(s,2) for s in scored[0][2]]))} "
              f"| top3={[round(s[0],3) for s in scored[:3]]}", flush=True)
        # next gen: elites + crossover/mutate
        nxt = [t[1] for t in top]
        while len(nxt) < pop_n:
            a, b = RNG.choice(top)[1], RNG.choice(top)[1]
            child = mutate(crossover(a, b), k=RNG.randint(1, 4))
            nxt.append(child)
        pop = nxt
    print(f"\nBEST: field={best_ever[0]:.3f} -> decks/ga_best.csv {dict(zip(META,[round(s,2) for s in best_ever[2]]))}", flush=True)


if __name__ == "__main__":
    main()
