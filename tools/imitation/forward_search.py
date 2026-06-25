"""Forward-search (1-ply lookahead) via the cg engine's search API. The engine hides the opponent's
deck/hand/prizes, so search_begin needs a DETERMINIZATION (a plausible guess of hidden state). This
module reconstructs those inputs from the obs + our decklist + a generic filler, then scores MAIN
options by simulating them one ply and evaluating the resulting state.

DOCKER-ONLY (needs libcg.so) and LIVE-OBS-ONLY (search_begin requires obs.search_begin_input, which
replays lack). Probe first: does search_begin accept our reconstructed inputs and step? See _probe.
"""

BASIC_ENERGY_FILLER = 1   # Basic Grass Energy id (a safe legal filler for unknown cards)


def reconstruct_hidden(obs_dict, seat, decklist):
    """Best-effort hidden-state guess for search_begin. Counts MUST match the real state.
    your_deck/your_prize: our unseen cards (we know our decklist) split deck-vs-prize arbitrarily.
    opponent_*: we don't know their list -> fill with a legal filler (a basic energy) at the right
    counts; opponent_active only needed if face-down."""
    from collections import Counter
    cur = obs_dict["current"]
    me = cur["players"][seat]; opp = cur["players"][1 - seat]
    seen = Counter()
    for c in (me.get("hand") or []):
        if c: seen[c["id"]] += 1
    for c in (me.get("discard") or []):
        if c: seen[c["id"]] += 1
    for grp in ("active", "bench"):
        for p in (me.get(grp) or []):
            if p:
                seen[p["id"]] += 1
                for sub in ("preEvolution", "energyCards", "tools"):
                    for c in (p.get(sub) or []):
                        if c: seen[c["id"]] += 1
    unseen = []
    for cid, n in Counter(decklist).items():
        unseen += [cid] * max(0, n - seen[cid])
    deck_n = me.get("deckCount", 0); prize_n = len(me.get("prize") or [])
    your_deck = unseen[:deck_n]
    your_prize = unseen[deck_n:deck_n + prize_n]
    # pad if the multiset didn't cover (safety): use filler
    your_deck += [BASIC_ENERGY_FILLER] * (deck_n - len(your_deck))
    your_prize += [BASIC_ENERGY_FILLER] * (prize_n - len(your_prize))
    opp_deck = [BASIC_ENERGY_FILLER] * opp.get("deckCount", 0)
    opp_prize = [BASIC_ENERGY_FILLER] * len(opp.get("prize") or [])
    opp_hand = [BASIC_ENERGY_FILLER] * opp.get("handCount", 0)
    oa = (opp.get("active") or [None])[0]
    opp_active = [] if oa else [BASIC_ENERGY_FILLER]   # only needed when face-down
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def _probe(obs, seat, decklist, cg):
    """Return (ok, msg): can we search_begin + search_step from this live obs?"""
    try:
        yd, yp, od, op, oh, oa = reconstruct_hidden(obs if isinstance(obs, dict) else obs.__dict__,
                                                     seat, decklist)
        ss = cg.search_begin(cg.to_observation_class(obs) if isinstance(obs, dict) else obs,
                             yd, yp, od, op, oh, oa)
        return True, f"search_begin ok (searchId={getattr(ss, 'searchId', '?')})"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
