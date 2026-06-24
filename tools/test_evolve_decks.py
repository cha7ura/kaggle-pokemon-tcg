"""GA operators must always emit legal, archetype-coherent 60-card decks (no fitness-gaming junk)."""
import collections, importlib.util, sys


def _load():
    sys.argv = ["ed", "1", "4", "4", "8"]  # don't trigger main(); just configure module
    spec = importlib.util.spec_from_file_location("ed", "tools/evolve_decks.py")
    ed = importlib.util.module_from_spec(spec); spec.loader.exec_module(ed)
    return ed


def test_mutate_crossover_stay_coherent():
    ed = _load()
    seed = ed._deck_of(ed.SEED_SLUGS[0])
    for _ in range(300):
        m = ed.mutate(seed, k=ed.RNG.randint(1, 5))
        c = ed.crossover(seed, ed._deck_of(ed.SEED_SLUGS[ed.RNG.randrange(len(ed.SEED_SLUGS))]))
        for d in (m, c):
            assert len(d) == 60
            assert ed.legal(d)
            assert all(x in ed.POOL for x in d)                       # no foreign-archetype cards
            assert collections.Counter(d)[ed.CORE] >= ed.MIN_CORE     # keeps the core attacker
            for card, n in collections.Counter(d).items():
                if card not in ed.BASIC_ENERGY:
                    assert n <= 4
