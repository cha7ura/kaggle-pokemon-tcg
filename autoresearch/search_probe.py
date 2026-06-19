"""De-risk the search path: confirm search_begin/search_step/search_end work
on a real in-game state, with a naive determinization of the hidden info.

If this prints PROBE SUCCESS, the whole IS-MCTS approach is unlocked.
"""
import random

from cg.api import (to_observation_class, search_begin, search_step, search_end,
                    SelectType)
from cg.game import battle_start, battle_select, battle_finish
from agent import read_deck

DECK = read_deck()
BASIC_POKEMON_ID = 721   # Kyogre — a Basic Pokémon, used to satisfy "deck needs a Basic"
ENERGY_ID = 3            # Basic Water Energy — generic filler


def determinize(obs):
    """Build the 6 prediction lists search_begin needs (counts must be >= actual)."""
    st = obs.current
    me = st.yourIndex
    op = st.players[1 - me]
    my = st.players[me]
    your_deck = DECK[:]                       # we know our own list; len 60 >= deckCount
    your_prize = [ENERGY_ID] * len(my.prize)
    opp_deck = DECK[:]                         # guess: our pool (contains Basic Pokémon)
    opp_prize = [ENERGY_ID] * len(op.prize)
    opp_hand = [ENERGY_ID] * op.handCount
    opp_active = []
    if op.active and op.active[0] is None:     # opponent active is face-down -> must predict it
        opp_active = [BASIC_POKEMON_ID]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def main():
    obs_dict, sd = battle_start(DECK, DECK)
    if obs_dict is None:
        print("battle_start failed"); return
    attempts = 0
    try:
        for step in range(400):
            cur = obs_dict["current"]
            if cur is not None and cur.get("result", -1) != -1:
                print("game ended before a clean probe state"); break
            sel = obs_dict["select"]
            if sel is None:
                obs_dict = battle_select(read_deck()); continue

            # attempt the probe on a normal MAIN decision past setup
            if cur is not None and cur.get("turn", 0) >= 1 \
                    and sel.get("type") == int(SelectType.MAIN) and attempts < 5:
                attempts += 1
                obs = to_observation_class(obs_dict)
                try:
                    args = determinize(obs)
                    ss = search_begin(obs, *args)
                    nopt = len(ss.observation.select.option) if ss.observation.select else None
                    print(f"[try {attempts}] search_begin OK  searchId={ss.searchId}  options={nopt}")
                    ss2 = search_step(ss.searchId, [0])
                    nopt2 = len(ss2.observation.select.option) if ss2.observation.select else None
                    res = ss2.observation.current.result if ss2.observation.current else None
                    print(f"          search_step([0]) OK  next_options={nopt2}  result={res}")
                    search_end()
                    print("PROBE SUCCESS — forward model is usable")
                    return
                except Exception as e:
                    print(f"[try {attempts}] search failed: {e!r}  (advancing)")

            n = len(sel["option"])
            obs_dict = battle_select(random.sample(range(n), sel["maxCount"]))
        print("probe did not reach success within step budget")
    finally:
        battle_finish()


if __name__ == "__main__":
    main()
