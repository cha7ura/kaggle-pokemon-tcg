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


def _opp_hp_dict(obs_dict, seat):
    pl = obs_dict["current"]["players"][1 - seat]
    tot = 0
    for p in (pl.get("active") or []) + (pl.get("bench") or []):
        if p:
            tot += p.get("hp", 0)
    return tot


def _my_prize_dict(obs_dict, seat):
    return len(obs_dict["current"]["players"][seat].get("prize") or [])


def _leaf_value(obs_after, seat, opp_hp_before, myp_before):
    """Score the post-move state (Observation object): prizes we took dominate, then damage dealt."""
    try:
        pl = obs_after.current.players
        me, opp = pl[seat], pl[1 - seat]
        opp_hp = sum(p.hp for p in (list(me and opp.active or []) + list(opp.bench or [])) if p)
        myp = len(me.prize or [])
        prizes_taken = max(0, myp_before - myp)
        return prizes_taken * 1000 + max(0, opp_hp_before - opp_hp)
    except Exception:
        return -1.0


def _my_prize_after(obs_after, seat):
    try:
        return len(obs_after.current.players[seat].prize or [])
    except Exception:
        return 99


def _opp_hp_after(obs_after, seat):
    try:
        opp = obs_after.current.players[1 - seat]
        return sum(p.hp for p in (list(opp.active or []) + list(opp.bench or [])) if p)
    except Exception:
        return 10 ** 9


def best_option(obs_dict, seat, deck):
    """For an ATTACK selection, override the model ONLY to SECURE A KO: simulate each option and
    return the one that takes a prize (real KO, exact via the engine). If no option KOs, return
    None and defer to the learned policy — 'max damage now' greed hurts control decks (Trevenant),
    but taking a guaranteed KO is always correct. None when not applicable (no engine / not live)."""
    sel = obs_dict.get("select"); cur = obs_dict.get("current")
    if not sel or not cur or not obs_dict.get("search_begin_input"):
        return None
    opts = sel.get("option") or []
    is_attack = sel.get("context") == 35 or any((o or {}).get("type") == 13 for o in opts)
    if not is_attack or len(opts) < 2:
        return None
    try:
        import cg.api as cg
    except Exception:
        return None
    try:
        yd, yp, od, op, oh, oa = reconstruct_hidden(obs_dict, seat, deck)
        obs_obj = cg.to_observation_class(obs_dict)
        opp_before = _opp_hp_dict(obs_dict, seat)
        myp_before = _my_prize_dict(obs_dict, seat)
        best_i, best_prizes, best_dmg = None, 0, 0
        for i in range(len(opts)):
            ss = cg.search_begin(obs_obj, yd, yp, od, op, oh, oa)
            nxt = cg.search_step(ss.searchId, [i])
            prizes = max(0, myp_before - _my_prize_after(nxt.observation, seat))
            dmg = max(0, opp_before - _opp_hp_after(nxt.observation, seat))
            try:
                cg.search_release(ss.searchId)
            except Exception:
                pass
            if (prizes, dmg) > (best_prizes, best_dmg):
                best_prizes, best_dmg, best_i = prizes, dmg, i
        return best_i if best_prizes > 0 else None     # only override to SECURE a KO
    except Exception:
        return None


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
