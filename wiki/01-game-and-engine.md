# 01 — The game and the engine

← [index](00-index.md) · next → [02 — the Lucario policy](02-policy-lucario.md)

## What you're actually controlling

Forget the trading-card-game flavour for a second. The dumb version: the engine hands you a
**menu of legal moves** and asks you to **tick some boxes**. Your whole agent is one function:

```python
def agent(obs_dict) -> list[int]: ...
```

It returns a list of **indices into a menu** (`obs.select.option`). That's it. No "play card X
to slot Y" — the engine already enumerated every legal action; you just point at the ones you want.

Two things make this less trivial than it sounds:

1. **It's a card game** — hidden hands, a shuffled deck, coin-flip attacks. You never see the
   opponent's hand or the deck order. (This is why later wiki pages obsess over *imperfect
   information* — see [10 — CFR and Nash](10-cfr-and-nash.md).)
2. **A crash forfeits the game.** If your `agent` throws, the harness scores it as a loss
   (`eval.py:60`). So the real contract is "return *something legal*, always."

## The contract (memorise this, the engine enforces it)

From `autoresearch/program.md:34` and the agent's own code (`autoresearch/agent_lucario.py:290`):

- Return **indices into `obs.select.option`**.
- The count must be in **`[minCount, maxCount]`** (`sdk/cg/api.py:402`), no duplicates.
  `maxCount` never exceeds `len(option)` (`api.py:403`).
- When **`obs.select is None`** you're in the deck phase → return your **60-card deck**
  (`agent_lucario.py:293`).
- The engine **only ever offers legal options.** Do not re-validate legality — it's wasted code
  and the menu is already filtered (`program.md:37`).

Dumb-but-correct baseline agent, in full:

```python
def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    return [0]          # always pick the first legal option
```

That actually runs a whole game without crashing. It just plays badly. Everything in
[02](02-policy-lucario.md) is "replace `[0]` with a good choice."

## What you get to look at

`to_observation_class(obs_dict)` (`sdk/cg/api.py:509`) parses the raw dict into typed objects:

- **`Observation`** (`api.py:439`) — the root. `obs.current` is the game state, `obs.select` is
  the menu (or `None` in the deck phase).
- **`obs.current.yourIndex`** (`api.py:370`) — are you player 0 or 1? You don't get to choose;
  read it every decision. The policy does this constantly (`agent_lucario.py:107`).
- **`SelectData`** (`api.py:399`) — `option` (the menu), `minCount`/`maxCount`, `type`
  (MAIN vs a sub-context like search/discard), and `context` (which sub-context).
- **`Option`** (`api.py:382`) — one menu entry. `type` is an `OptionType` (ATTACK, PLAY, ATTACH,
  EVOLVE, ABILITY, RETREAT, END…); the rest tells you *what* it acts on (area + index).
- **`Pokemon`** (`api.py:339`) — a board Pokémon: `id`, `hp`, `energies`, etc.
- **`CardData`** (`api.py:464`) — static card facts: `weakness`, `resistance`, `ex`, `megaEx`.
  The policy preloads all of them once into a dict (`agent_lucario.py:40`).

The key mental model: **`Option` says what you *could* do; `CardData`/`Pokemon` tell you whether
it's a *good* idea.** Scoring that gap is the policy's entire job.

## Damage is not the number on the card

Effective damage ≠ printed damage. Two adjustments the engine applies and the policy mirrors
(`agent_lucario.py:78`, `_eff_damage`):

- **Weakness** → ×2 (here, anything weak to Fighting takes double).
- **Resistance** → −30.
- **The Crustle wall** (card 345) zeroes *all* damage from an `ex`/`megaEx` attacker. This single
  fact warps the whole strategy — see [02](02-policy-lucario.md).

## Where the engine lives, and why Docker

The engine is a **compiled native library** (`libcg.so` / `cg.dll`) loaded through `ctypes`.
It runs only under `docker run --platform linux/amd64` (see `autoresearch/run.sh`). You cannot
`pip install` it; it's not Python. Practically: agent logic is stdlib-only Python, and you run
the gauntlet inside the container.

## The hidden superpower: a forward model

The engine exposes a **forward model** — you can clone the current state and *simulate* moves:
`search_begin` / `search_step` / `search_end` (`sdk/cg/api.py:517`). This is what makes any kind
of lookahead/MCTS possible (see [17 — imperfect info on the TCG](17-imperfect-info-on-tcg.md)).

The catch is the imperfect-information one: to simulate forward you must **guess the hidden state**
(opponent hand, deck order) — *determinize* it. The rejected `agent_search.py` already does this
(`autoresearch/agent_search.py:49`, `_determinize`), proving the plumbing works even though that
*policy* lost. minizero (Plan 2) builds on exactly this.

> Takeaway: the engine is a legal-move menu over a hidden-information state, plus a forward model
> that needs determinization to use. Hold that and the rest of the wiki is commentary.
