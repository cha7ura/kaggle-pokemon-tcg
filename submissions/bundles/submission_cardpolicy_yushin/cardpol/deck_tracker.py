"""Card-counting over the RAW obs dict. We know our own 60-card decklist, so we can track what's
left and the probability of drawing what we need. Stdlib only; same code offline + in-sim.

  deck_remaining(obs, seat, decklist) -> {id: unknown_count}   # cards not yet seen (deck OR prizes)
  draw_prob(obs, seat, decklist, id, k)                        # P(>=1 copy in next k draws), hypergeom
  needed_in_discard(obs, seat, id)                             # certain fetch available (discard visible)
  deckout_clock(obs, seat)                                     # our deckCount (low => stop drawing)
  prized_likely(obs, seat, decklist, id)                       # expected copies sitting in prizes

Uncertainty model: a copy we haven't seen is in the DECK or the PRIZES (both hidden, both random).
We treat deck+prizes as one shuffled unknown pool, so draw odds are honest about prize uncertainty.
"""
from collections import Counter
from math import comb


def _seen_counts(obs, seat):
    """Count every card id we can currently SEE for our seat: hand + discard + in-play
    (Pokemon ids + their pre-evolution stack + attached energy + tools)."""
    cur = obs.get("current") or {}
    me = (cur.get("players") or [{}, {}])[seat]
    c = Counter()
    for card in (me.get("hand") or []):
        if card:
            c[card.get("id")] += 1
    for card in (me.get("discard") or []):
        if card:
            c[card.get("id")] += 1
    for pk in [a for a in (me.get("active") or []) if a] + [b for b in (me.get("bench") or []) if b]:
        c[pk.get("id")] += 1
        for grp in ("preEvolution", "energyCards", "tools"):
            for card in (pk.get(grp) or []):
                if card:
                    c[card.get("id")] += 1
    return c


def _unknown_pool(obs, seat):
    """Size of the hidden pool = deck + our face-down prizes (both unknown to us)."""
    cur = obs.get("current") or {}
    me = (cur.get("players") or [{}, {}])[seat]
    deck = me.get("deckCount", 0)
    prizes = len(me.get("prize") or [])
    return deck, prizes, deck + prizes


def deck_remaining(obs, seat, decklist):
    """{id: count not yet seen} = decklist - visible. These copies are in deck or prizes."""
    deck = Counter(decklist)
    seen = _seen_counts(obs, seat)
    return {cid: deck[cid] - seen.get(cid, 0) for cid in deck if deck[cid] - seen.get(cid, 0) > 0}


def draw_prob(obs, seat, decklist, card_id, k=1):
    """P(at least one copy of card_id in the next k draws), hypergeometric over the unknown pool.
    Honest about prize uncertainty: unseen copies might be prized, not drawable."""
    _, _, pool = _unknown_pool(obs, seat)
    if pool <= 0 or k <= 0:
        return 0.0
    n = deck_remaining(obs, seat, decklist).get(card_id, 0)
    if n <= 0:
        return 0.0
    k = min(k, pool)
    # P(none in k) = C(pool-n, k) / C(pool, k)
    miss = comb(pool - n, k) / comb(pool, k) if pool - n >= k else 0.0
    return round(1.0 - miss, 4)


def needed_in_discard(obs, seat, card_id):
    cur = obs.get("current") or {}
    me = (cur.get("players") or [{}, {}])[seat]
    return any(c and c.get("id") == card_id for c in (me.get("discard") or []))


def deckout_clock(obs, seat):
    cur = obs.get("current") or {}
    return (cur.get("players") or [{}, {}])[seat].get("deckCount", 0)


def prized_likely(obs, seat, decklist, card_id):
    """Expected copies of card_id sitting in our face-down prizes (>0 => maybe locked away)."""
    deck, prizes, pool = _unknown_pool(obs, seat)
    if pool <= 0:
        return 0.0
    n = deck_remaining(obs, seat, decklist).get(card_id, 0)
    return round(n * prizes / pool, 3)


def _selfcheck():
    ULTRA, ABRA, RARE = 1097, 741, 1079
    decklist = [ULTRA] * 4 + [ABRA] * 4 + [RARE] * 3 + [0] * 49
    # see 1 Ultra in hand, 1 in discard; deck 40, prizes 6 -> unknown pool 46, Ultra unknown=2
    obs = {"current": {"players": [
        {"hand": [{"id": ULTRA}], "discard": [{"id": ULTRA}],
         "active": [{"id": ABRA, "preEvolution": [], "energyCards": [], "tools": []}],
         "bench": [], "deckCount": 40, "prize": [None] * 6}, {}]}}
    rem = deck_remaining(obs, 0, decklist)
    assert rem[ULTRA] == 2, rem                       # 4 - 1 hand - 1 discard
    assert rem[ABRA] == 3, rem                         # 4 - 1 in play
    assert needed_in_discard(obs, 0, ULTRA)
    assert not needed_in_discard(obs, 0, RARE)
    assert deckout_clock(obs, 0) == 40
    p1 = draw_prob(obs, 0, decklist, ULTRA, 1)
    assert abs(p1 - 2 / 46) < 1e-3, p1                 # 1 draw, 2 of 46 unknown (4-dp rounded)
    p7 = draw_prob(obs, 0, decklist, ULTRA, 7)
    assert p1 < p7 < 1.0, (p1, p7)                     # more draws -> higher
    pr = prized_likely(obs, 0, decklist, ULTRA)
    assert abs(pr - 2 * 6 / 46) < 1e-3, pr             # expected copies in 6 prizes
    print("deck_tracker.py self-check PASSED")


if __name__ == "__main__":
    _selfcheck()
