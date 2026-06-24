"""Evolutionary deck search (GA). Seed from the real meta (most common field decks), mutate
coherently (swap/adjust within a card pool drawn from the field, max-4 + 60-card legal), fitness =
field-weighted win-rate vs the REAL 182-deck field via the imitation-league harness (typh pilot
both sides -> isolates DECK, not pilot), MINUS a fragility penalty on the worst matchup (drives
evolution to fix our structural Dragapult hole). Coherence enforced by fitness (junk mutants lose ->
die), not repair. Local = filter; ladder = judge. Run on host (docker).

  python tools/evolve_decks.py <generations> <pop> <games> [n_field_opponents]
"""
import csv, os, sys, random, sqlite3
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                         # import the tools package when run as a script
from tools.imitation.league import challenge_field, _deck_of
DECKS = f"{ROOT}/autoresearch/decks"
rows = list(csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv")))
R = {int(r["cardId"]): r for r in rows}
BASIC_ENERGY = {2, 3, 4, 5, 6, 7, 8, 9}          # unlimited copies allowed
FRAGILITY = 0.35   # penalty per point a worst-matchup falls below 0.5 (anti-fragility / PFSP spirit)
RNG = random.Random(1234)


def _is_pokemon(c):
    r = R.get(c, {})
    return any(r.get(k) == "True" for k in ("basic", "stage1", "stage2")) and "ENERGY" not in r.get("cardType", "").upper()


def _is_energy(c):
    return "ENERGY" in R.get(c, {}).get("cardType", "").upper()


# real meta from the replay db. COHERENCE: evolve WITHIN one archetype (default Trevenant = us), so
# mutants stay legal real decks instead of cross-archetype junk that only beats the weak typh pilot.
_db = sqlite3.connect(f"{ROOT}/replays.sqlite")
_count = {fn: c for fn, c in _db.execute("SELECT fname,count FROM decks")}
_arch = {fn: a for fn, a in _db.execute("SELECT fname,archetype FROM decks")}
_ranked = sorted(_count, key=lambda s: -_count[s])
ARCHE = sys.argv[5] if len(sys.argv) > 5 else "Trevenant"
N_FIELD = int(sys.argv[4]) if len(sys.argv) > 4 else 30
FIELD_SLUGS = _ranked[:N_FIELD]                   # score vs the whole field (weight in the head)
_arch_slugs = sorted([s for s in _ranked if _arch.get(s) == ARCHE], key=lambda s: -_count[s])
SEED_SLUGS = _arch_slugs[:4]


def load(slug):
    return [int(x) for x in open(f"{DECKS}/{slug}.csv") if x.strip()][:60]


def save(deck, slug):
    open(f"{DECKS}/{slug}.csv", "w").write("\n".join(map(str, deck)) + "\n")


# pool = cards from this archetype's REAL decks only (coherent). Energies likewise restricted to
# what the archetype actually plays (Trevenant runs special energies, never basic).
POOL = set()
for s in _arch_slugs:
    POOL |= set(_deck_of(s))
POOL = sorted(POOL)
ENERGY_POOL = [c for c in POOL if _is_energy(c)] or [11]          # fallback Mist Energy
# core attacker = most common Pokémon across the seed decks; require >=3 copies
_seed_cards = Counter()
for s in SEED_SLUGS:
    _seed_cards.update(_deck_of(s))
CORE = max((c for c in _seed_cards if _is_pokemon(c)), key=lambda c: _seed_cards[c], default=None)
MIN_CORE = 3


def legal(deck):
    if len(deck) != 60: return False
    cnt = Counter(deck)
    for c, n in cnt.items():
        if c not in POOL: return False            # archetype-coherent: no foreign cards
        if c not in BASIC_ENERGY and n > 4: return False
    if CORE is not None and cnt[CORE] < MIN_CORE: return False   # must keep the core attacker
    return True


def _cap(c):
    return 99 if c in BASIC_ENERGY else 4


def _repair(cards):
    """Coerce any card multiset into a legal, coherent 60-card deck of THIS archetype."""
    cnt = Counter(c for c in cards if c in POOL)    # drop foreign cards
    for c in list(cnt):                             # max-4 (special energy included)
        cnt[c] = min(cnt[c], _cap(c))
    if CORE is not None:                            # guarantee the core attacker
        cnt[CORE] = max(cnt.get(CORE, 0), MIN_CORE)
    deck = []
    for c, n in cnt.items():
        deck += [c] * n
    while len(deck) > 60:                           # trim, but never below MIN_CORE of CORE
        i = RNG.randrange(len(deck))
        if deck[i] == CORE and deck.count(CORE) <= MIN_CORE:
            continue
        deck.pop(i)
    while len(deck) < 60:                           # pad with real cards under their cap (energy first)
        opts = ([e for e in ENERGY_POOL if deck.count(e) < 4]
                or [c for c in POOL if deck.count(c) < _cap(c)])
        deck.append(RNG.choice(opts))
    return deck


def mutate(deck, k=3):
    d = deck[:]
    for _ in range(k):
        op = RNG.random()
        if op < 0.45 and d:                       # swap one copy for a pool card
            d[RNG.randrange(len(d))] = RNG.choice(POOL)
        elif op < 0.75 and d:                     # remove a copy, add a pool card
            d.pop(RNG.randrange(len(d))); d.append(RNG.choice(POOL))
        else:                                     # raise a count
            d[RNG.randrange(len(d))] = RNG.choice(d)
    out = _repair(d)
    return out if legal(out) else deck


def crossover(a, b):
    ca, cb = Counter(a), Counter(b)
    child = []
    for c in set(ca) | set(cb):
        n = ca.get(c, 0) if RNG.random() < 0.5 else cb.get(c, 0)
        child += [c] * n
    return _repair(child)


def fitness(deck, games):
    """Field-weighted win rate vs the real field MINUS fragility penalty on the worst matchup."""
    r = challenge_field(deck, opp_slugs=FIELD_SLUGS, games=games, cand_label="ga_cand")
    fit = r["fw"] - FRAGILITY * max(0.0, 0.5 - r["worst"])
    return fit, r


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pop_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    print(f"GA: {gens} gens x {pop_n} pop x {games} games vs {len(FIELD_SLUGS)}-deck field; "
          f"seeds={SEED_SLUGS}; pool={len(POOL)} cards", flush=True)
    pop = [load_field(s) for s in SEED_SLUGS]
    while len(pop) < pop_n:
        pop.append(mutate(load_field(RNG.choice(SEED_SLUGS)), k=RNG.randint(2, 5)))
    best_ever = (0.0, None, None)
    for g in range(gens):
        scored = []
        for i, d in enumerate(pop):
            fit, r = fitness(d, games)
            scored.append((fit, d, r))
        scored.sort(key=lambda x: -x[0])
        top = scored[: max(2, pop_n // 3)]
        if scored[0][0] > best_ever[0]:
            best_ever = scored[0]; save(best_ever[1], "ga_best")
        b = scored[0][2]
        worst_opp = min(b["per"], key=b["per"].get) if b["per"] else "?"
        print(f"gen {g}: fit={scored[0][0]:.3f} fw={b['fw']:.3f} "
              f"worst={b['worst']:.3f} vs {worst_opp}({_arch.get(worst_opp,'?')}) "
              f"| top3={[round(s[0],3) for s in scored[:3]]}", flush=True)
        nxt = [t[1] for t in top]
        while len(nxt) < pop_n:
            a, b2 = RNG.choice(top)[1], RNG.choice(top)[1]
            nxt.append(mutate(crossover(a, b2), k=RNG.randint(1, 4)))
        pop = nxt
    r = best_ever[2]
    print(f"\nBEST: fit={best_ever[0]:.3f} fw={r['fw']:.3f} worst={r['worst']:.3f} -> decks/ga_best.csv",
          flush=True)


def load_field(slug):
    return _deck_of(slug)


if __name__ == "__main__":
    main()
