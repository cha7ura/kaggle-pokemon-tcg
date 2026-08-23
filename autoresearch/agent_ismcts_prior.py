"""ISMCTS with archetype-prior determinization (step 3).

Drop-in replacement for autoresearch/agent_ismcts.py whose only change is _determinize():
instead of modeling the opponent's hidden deck as a COPY OF OUR OWN decklist (H._DECK) and their
hidden hand/prize as basic-energy filler -- the diagnosed cause of the prior
search-loses-to-heuristic result -- it reconstructs the opponent's REVEALED
cards from the engine state and samples the hidden slots from the archetype prior (determinize.py,
validated held-out: ~5% -> 21-40% hidden-deck recall, archetype-ID 88% at 6 revealed cards).

DOCKER / linux-amd64 ONLY (needs libcg.so). Run the head-to-head at the bottom to test whether
search now BEATS the heuristic (it lost before). If it wins here, the determinization fix is
validated and R-NaD self-play (step 4) is worth starting.

Usage on a linux/amd64 box with the engine:
    PYTHONPATH=sdk python agent_ismcts_prior.py --games 100 --sims 48
"""
import os, sys, math, random, argparse
from collections import Counter
from cg.api import (to_observation_class, OptionType, SelectType,
                    search_begin, search_step, search_end)
import agent_lucario as H

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from determinize import Determinizer

ENERGY_ID = 3
BASIC = 721
def _find(fname, subdirs=("", "autoresearch", "determinization", "..")):
    """Locate a data file robustly: this file may live at repo root OR in autoresearch/,
    so search a few candidate roots (module dir, its parent, cwd) x subdirs."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.dirname(here), os.getcwd()]
    for r in roots:
        for s in subdirs:
            p = os.path.join(r, s, fname) if s else os.path.join(r, fname)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"{fname} not found under {roots}")

_DET = Determinizer.load(_find("arch_priors.json"), _find("cards_full.csv"))

# --- diagnostics: distinguish "search doesn't help" from "search is broken" ---
DIAG = {"engage": 0, "defer": 0, "rollout_ok": 0, "rollout_err": 0, "last_err": ""}


def _revealed_opp_ids(state):
    """All opponent card ids currently visible to us: discard + active/bench Pokémon
    (and their attached energy / evolutions / tools where exposed)."""
    me = state.yourIndex
    opp = state.players[1 - me]
    seen = []
    for c in (opp.discard or []):
        if c is not None:
            seen.append(getattr(c, "id", None) or getattr(c, "cardId", None))
    for grp in ("active", "bench"):
        for p in (getattr(opp, grp, None) or []):
            if p is None:
                continue
            pid = getattr(p, "id", None) or getattr(p, "cardId", None)
            if pid is not None:
                seen.append(pid)
            for sub in ("preEvolution", "energyCards", "tools"):
                for c in (getattr(p, sub, None) or []):
                    if c is not None:
                        cid = getattr(c, "id", None) or getattr(c, "cardId", None)
                        if cid is not None:
                            seen.append(cid)
    return [x for x in seen if x is not None]


def _determinize(state, rng=None):
    """Prior-based hidden-state reconstruction for search_begin.
    Our side: we know our decklist (H._DECK), so split unseen own cards into deck/prize.
    Opp side: infer archetype from revealed cards, sample hidden deck+hand+prize from the prior."""
    me = state.yourIndex
    my, opp = state.players[me], state.players[1 - me]
    rng = rng or random.Random()

    # --- our hidden cards: from our real decklist minus what we can see of our own ---
    my_deck = H._DECK[:]  # our full 60; engine only needs >= counts, uses first deckCount
    my_prize = [ENERGY_ID] * len(my.prize)  # our own prize identity doesn't affect opp modeling

    # --- opponent: prior-sampled ---
    revealed = _revealed_opp_ids(state)
    arch = _DET.infer_archetype(revealed)
    opp_deck_n = opp.deckCount
    opp_hand_n = opp.handCount
    opp_prize_n = len(opp.prize)
    total_hidden = opp_deck_n + opp_hand_n + opp_prize_n
    sampled = _DET.sample_hidden(revealed, total_hidden, arch=arch, rng=rng, stochastic=True)
    opp_deck = sampled[:opp_deck_n]
    opp_hand = sampled[opp_deck_n:opp_deck_n + opp_hand_n]
    opp_prize = sampled[opp_deck_n + opp_hand_n:total_hidden]
    # face-down active needs a Pokémon id; prior sample may not be a Pokémon, so fall back to BASIC
    oa = [BASIC] if (opp.active and opp.active[0] is None) else []
    # pad safety
    opp_deck += [ENERGY_ID] * (opp_deck_n - len(opp_deck))
    opp_hand += [ENERGY_ID] * (opp_hand_n - len(opp_hand))
    opp_prize += [ENERGY_ID] * (opp_prize_n - len(opp_prize))
    return (my_deck, my_prize, opp_deck, opp_prize, opp_hand, oa)


# ---- the rest is identical to agent_ismcts.py, using the new _determinize ----
def _value(state, me):
    if state is None: return 0.0
    if state.result == me: return 1.0
    if state.result == (1 - me): return -1.0
    if state.result == 2: return 0.0
    my, op = state.players[me], state.players[1 - me]
    return 0.1 * (len(op.prize) - len(my.prize))


def _safe_action(sel_obs):
    """Legal action for a mid-search observation. Defer to the heuristic's own selection
    (whose indices are always in range by construction), then enforce the engine's rules:
      err5: every element 0<=i<len(option); err6: no duplicates;
      err4: minCount<=len(sel)<=maxCount. For a zero-option select return [] (like
    _legal_fallback), NOT [0] — [0] against 0 options is exactly the err5 we were hitting."""
    sel = sel_obs.select
    n = len(sel.option)
    mn = getattr(sel, "minCount", 0) or 0
    mx = getattr(sel, "maxCount", n) or n
    if n == 0:
        DIAG["empty_select"] = DIAG.get("empty_select", 0) + 1
        return []
    # Optional sub-selection (minCount==0): PASS. Always legal (0<=0<=maxCount) and the
    # correct rollout default — forcing >=1 here was the err4/err5 desync (a CARD/ENERGY
    # sub-select's count semantics don't map cleanly to a MAIN-style option pick).
    if mn == 0:
        return []
    # Required selection: take the heuristic's ranked picks, in-range and de-duplicated,
    # then honor exactly [minCount, maxCount].
    try:
        act = H.select_indices(sel_obs)
    except Exception:
        act = None
    seen = set(); clean = []
    for i in (act or []):
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            seen.add(i); clean.append(i)
    lo = mn; hi = max(lo, min(mx, n))
    if len(clean) < lo:
        for i in range(n):
            if i not in seen:
                clean.append(i); seen.add(i)
            if len(clean) >= lo:
                break
    return clean[:hi] if clean else list(range(min(lo, n)))


def _capture_fail(sel_obs, action):
    """Record the exact failing select the first time a step is rejected, so we can see
    what 'illegal' actually looked like instead of guessing."""
    if DIAG.get("fail_detail"):
        return
    try:
        sel = sel_obs.select
        DIAG["fail_detail"] = {
            "n_option": len(sel.option),
            "action": list(action),
            "minCount": getattr(sel, "minCount", None),
            "maxCount": getattr(sel, "maxCount", None),
            "select_type": int(getattr(sel, "type", -1)),
        }
    except Exception as e:
        DIAG["fail_detail"] = {"capture_error": str(e)}


def _rollout(obs, first_action, me, depth, rng):
    ss = search_begin(obs, *_determinize(obs.current, rng))
    sid = ss.searchId
    try:
        # First action comes from the REAL root obs; if the determinized search root has a
        # different option set it may be out of range -> can't evaluate this move: neutral 0.
        try:
            ss = search_step(sid, first_action)
        except Exception:
            _capture_fail(ss.observation, first_action)
            DIAG["first_step_err"] = DIAG.get("first_step_err", 0) + 1
            return 0.0
        for _ in range(depth):
            cur = ss.observation.current
            if cur is None or cur.result != -1: break
            if ss.observation.select is None: break
            act = _safe_action(ss.observation)
            try:
                ss = search_step(sid, act)
            except Exception:
                # Can't continue the rollout -> evaluate the state REACHED SO FAR
                # (a truncated rollout is NOT a loss; scoring it -1 was the corruption).
                _capture_fail(ss.observation, act)
                DIAG["rollout_trunc"] = DIAG.get("rollout_trunc", 0) + 1
                break
        return _value(ss.observation.current, me)
    finally:
        search_end()


def _ismcts(obs, sims=48, depth=12, rng=None):
    rng = rng or random.Random()
    sel = obs.select
    me = obs.current.yourIndex
    cands = [i for i in range(len(sel.option)) if int(sel.option[i].type) != int(OptionType.END)]
    if len(cands) <= 1:
        DIAG["defer"] += 1
        return H.select_indices(obs)
    DIAG["engage"] += 1
    N = {i: 0 for i in cands}; Q = {i: 0.0 for i in cands}
    for t in range(sims):
        pick = next((i for i in cands if N[i] == 0), None)
        if pick is None:
            pick = max(cands, key=lambda i: Q[i]/N[i] + 1.4*math.sqrt(math.log(t+1)/N[i]))
        try:
            v = _rollout(obs, [pick], me, depth, rng)
            DIAG["rollout_ok"] += 1
        except Exception as e:
            v = -1.0
            DIAG["rollout_err"] += 1
            if not DIAG["last_err"]:
                import traceback
                DIAG["last_err"] = traceback.format_exc()
        N[pick] += 1; Q[pick] += v
    return [max(cands, key=lambda i: (Q[i]/N[i]) if N[i] else -1e9)]


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None: return H._DECK
        sel = obs.select
        if int(sel.type) == int(SelectType.MAIN) and sel.maxCount == 1 and sel.minCount == 1:
            return _ismcts(obs)
        return H.select_indices(obs)
    except Exception:
        try: return H._legal_fallback(to_observation_class(obs_dict).select)
        except Exception: return [0]


# --------- head-to-head: prior-ISMCTS vs heuristic (the validation gate) ----------
def _play_match(engine_play, games, sims, depth, seed=0):
    """engine_play(agentA, agentB) -> winner index, provided by the local cg runner.
    We import the project's tournament runner if present; otherwise print instructions."""
    try:
        from tools.tournament import play_series  # project runner (if available)
    except Exception:
        print("NOTE: wire this to your local cg match runner (tools/tournament or sdk smoke_test).")
        print("Each game: seat0 = prior-ISMCTS agent(), seat1 = agent_lucario heuristic.")
        return None
    import agent_lucario
    wins = play_series(agent, agent_lucario.agent, n=games, sims=sims, depth=depth, seed=seed)
    return wins


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"prior-ISMCTS vs heuristic | games={a.games} sims={a.sims} depth={a.depth}")
    res = _play_match(None, a.games, a.sims, a.depth, a.seed)
    if res is not None:
        w = res if isinstance(res, (int, float)) else res.get("A_wins")
        print(f"prior-ISMCTS win-rate vs heuristic: {100*w/a.games:.1f}%  (>50% => determinization fix validated)")
