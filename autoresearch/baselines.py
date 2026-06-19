"""Fixed opponents for the gauntlet. NOT edited by the loop.

`random_agent` is the floor every challenger must clear.
`greedy_agent` is a slightly-smarter reference; promote a challenger past it
once it reliably beats random.
"""
import random

from cg.api import to_observation_class, OptionType

# Lower number = chosen first. The key idea: do board development (evolve, attach
# energy, play, use abilities) BEFORE attacking, because attacking ends the turn.
# END is last resort; RETREAT is deprioritized (usually a tempo loss).
_PRIORITY = {
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
    int(OptionType.ATTACK): 7,        # attack after developing
    int(OptionType.NO): 8,
    int(OptionType.DISCARD): 9,
    int(OptionType.RETREAT): 10,
    int(OptionType.END): 11,          # only if nothing better
}


def random_agent(obs_dict: dict) -> list[int]:
    sel = obs_dict["select"]
    n = len(sel["option"])
    return random.sample(range(n), sel["maxCount"])


def greedy_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    sel = obs.select
    opts = sel.option
    order = sorted(range(len(opts)),
                   key=lambda i: _PRIORITY.get(int(opts[i].type), 6))
    k = max(sel.minCount, 1) if sel.maxCount >= 1 else sel.minCount
    k = min(k, sel.maxCount)
    chosen = order[:k]
    if len(chosen) < sel.minCount:
        chosen = order[:sel.minCount]
    return chosen
