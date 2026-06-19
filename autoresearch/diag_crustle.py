"""Diagnose the Lucario-vs-Crustle matchup: decode the Crustle deck and trace
what our agent actually does (does it ever face Crustle active / play Boss / KO?).
"""
from collections import Counter

from cg.api import (to_observation_class, OptionType, all_card_data)
from cg.game import battle_start, battle_select, battle_finish
import agent_lucario as lu
import crustle_agent as cr

CARD = {c.cardId: c for c in all_card_data()}
LU_DECK = lu.read_deck("decks/lucario_meta.csv")
CR_DECK = cr.read_deck("decks/crustle.csv")
REASON = {1: "all-prizes", 2: "deck-out", 3: "no-active", 4: "effect"}


def decode_crustle():
    print("=== CRUSTLE DECK CARDS ===")
    for cid in sorted(set(CR_DECK)):
        c = CARD.get(cid)
        if c:
            tag = "ex" if c.ex else ("megaEx" if c.megaEx else "")
            kind = ("HP" + str(c.hp)) if c.hp else f"type{int(c.cardType)}"
            print(f"  {cid:>4} x{CR_DECK.count(cid):<2} {c.name[:30]:<30} {kind:<8} {tag}")


def trace(n=60):
    stats = Counter()
    reasons = Counter()
    for g in range(n):
        obs, sd = battle_start(LU_DECK, CR_DECK)   # we are player 0
        if obs is None:
            continue
        reason = None
        try:
            for _ in range(4000):
                for lg in obs.get("logs", []):
                    if lg.get("type") == 23:
                        reason = lg.get("reason")
                cur = obs["current"]
                if cur is not None and cur.get("result", -1) != -1:
                    stats["win" if cur["result"] == 0 else "loss" if cur["result"] == 1 else "draw"] += 1
                    reasons[REASON.get(reason, reason)] += 1
                    break
                sel = obs["select"]
                if sel is None:
                    obs = battle_select(lu._DECK if cur is None else cr._DECK); continue
                me = cur["yourIndex"]
                if me == 0:
                    o = to_observation_class(obs)
                    # is opponent active Crustle?
                    opp = o.current.players[1].active
                    if opp and opp[0] and opp[0].id == 345:
                        stats["faces_crustle_decisions"] += 1
                        for opt in o.select.option:
                            if int(opt.type) == int(OptionType.PLAY):
                                c = lu._get(o, 2, opt.index, 0)  # HAND=2
                                if c and c.id == lu.BOSS_ORDERS:
                                    stats["boss_available_vs_crustle"] += 1
                    act = lu.agent(obs)
                else:
                    act = cr.agent(obs)
                obs = battle_select(act)
        finally:
            battle_finish()
    print("\n=== TRACE (Lucario as P0 vs Crustle) ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print("  end reasons:", dict(reasons))


if __name__ == "__main__":
    decode_crustle()
    trace(60)
