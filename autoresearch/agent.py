"""THE EDITABLE POLICY — this is the only file the autoresearch loop changes.

Mirrors Karpathy autoresearch's `train.py`: everything here is fair game
(action priorities, energy logic, attack selection, later: search via
cg.api.search_begin). The harness (eval.py) and engine (cg/) are off-limits.

Contract (same as the Kaggle submission's main.py):
    agent(obs_dict) -> list[int]
    - return indices into obs.select.option
    - length in [minCount, maxCount], no duplicates
    - if obs.select is None (deck-selection phase) return the 60 card IDs

Baseline below = a "develop then attack" greedy. It should comfortably beat the
random opponent. The loop's job is to climb from here.
"""
import os
import random

from cg.api import to_observation_class, OptionType

DECK_PATH = os.environ.get("AGENT_DECK", "decks/champion.csv")

# Action priority: develop the board first, attack last (attacking ends the turn).
# >>> This table is the primary knob the loop should mutate. <<<
PRIORITY = {
    int(OptionType.EVOLVE): 0,
    int(OptionType.ABILITY): 1,
    int(OptionType.ATTACH): 2,
    int(OptionType.ENERGY): 2,
    int(OptionType.ENERGY_CARD): 2,
    int(OptionType.PLAY): 3,
    int(OptionType.TOOL_CARD): 4,
    int(OptionType.CARD): 5,
    int(OptionType.YES): 6,
    int(OptionType.NUMBER): 6,
    int(OptionType.SPECIAL_CONDITION): 6,
    int(OptionType.ATTACK): 7,
    int(OptionType.NO): 8,
    int(OptionType.DISCARD): 9,
    int(OptionType.RETREAT): 10,
    int(OptionType.END): 11,
}


def read_deck(path: str = DECK_PATH) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]


def choose(obs) -> list[int]:
    """Pick option indices for a non-deck selection.

    EXTENSION POINTS for the loop / experiments:
      - rank ATTACK options by expected damage vs the opponent's active HP
        (use cg.api.all_card_data() / all_attack() for damage, weakness x2)
      - prize-trade awareness: avoid trading a Mega-ex (gives up 3 prizes)
        into a single-prize attacker
      - decline coin-flip attacks when a flat-damage line exists
      - retreat logic, status-condition handling, bench development
      - replace this whole function with a determinized search using
        cg.api.search_begin(...) as the forward model
    """
    sel = obs.select
    opts = sel.option
    order = sorted(range(len(opts)), key=lambda i: PRIORITY.get(int(opts[i].type), 6))
    k = min(max(sel.minCount, 1), sel.maxCount) if sel.maxCount >= 1 else sel.minCount
    chosen = order[:k]
    if len(chosen) < sel.minCount:
        chosen = order[:sel.minCount]
    return chosen


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    return choose(obs)
