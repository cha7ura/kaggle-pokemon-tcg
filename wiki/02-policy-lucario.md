# 02 — The Lucario policy, annotated

← [01 — game and engine](01-game-and-engine.md) · next → [03 — the autoresearch loop](03-autoresearch-loop.md)

The champion agent is `autoresearch/agent_lucario.py` (≈300 lines, stdlib-only). This page walks it
the way you'd want it explained: the dumb core first, then every piece of "intelligence" that was
*earned* by a logged experiment (see [03](03-autoresearch-loop.md)).

## The dumb core: score every option, take the best

The entire decision procedure is `select_indices` (`agent_lucario.py:280`):

```python
scores = _score_main(obs) if int(sel.type) == 0 else _score_context(obs)
order  = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
return order[:k]                     # k clamped into [minCount, maxCount]
```

That's it — **context-scored greedy**. No search, no learning. Assign every menu option a number,
sort, take the top `k`. All the skill lives in *how the numbers are assigned*. There are two
scorers: `_score_main` for normal turns (`:104`) and `_score_context` for sub-menus like
search/discard/bench-setup (`:206`).

## The one big idea: develop, *then* attack

The weight table `W` (`agent_lucario.py:33`) encodes a strict priority ladder:

```python
W = {"ability": 30000, "play_pokemon": 20000, "evolve": 9000, "attach": 8000,
     "draw": 3000, "boss": 3200, "switch": 6000, "attack": 1000,
     "ko_win": 50000, "crustle_whiff": -10000, ...}
```

Read the magnitudes, not the exact values: **abilities (30k) > playing Pokémon (20k) > evolving
(9k) > attaching energy (8k) > … > attacking (1k).** Attacking is *last*. Why? Experiment 001
(`log/experiments.md`) tried "fire any damaging attack ASAP" and scored **0.168** — attacking
before you've built a board wastes turns. Develop-first beat eager-attack by a mile, so the
ordering is baked into the weights.

Two exceptions punch through the ladder:

- **`ko_win` = 50000** — a knockout that *wins the game* outranks everything. Earned via prize math
  (below).
- **`crustle_whiff` = −10000** — swinging an `ex` into the Crustle wall does 0 damage; make it the
  *worst* option so it never happens.

## Effective damage and the Crustle wall

`_eff_damage` (`agent_lucario.py:78`) is the damage model from [01](01-game-and-engine.md):
weakness ×2, resistance −30, and the killer rule —

```python
if defender.id == CRUSTLE and (a.ex or a.megaEx):   # wall negates ex damage
    dmg = 0
```

Crustle (card 345) is a **stall/mill wall**: it has almost no attackers, blocks your `ex` damage
entirely, and tries to make you *deck out* (run out of cards). The `experiments.md` ledger has a
whole arc on this (`CRUSTLE DECK DECODED`). The structural answer is a **non-`ex` attacker**:
Hariyama (674) is not an `ex`, so its damage lands on Crustle. Most of the policy's "intelligence"
is routing the game through Hariyama when Crustle is across the table.

## Prize-trade awareness (the `ko_win` gate)

Not every KO is worth it. `_prize_count` (`agent_lucario.py:73`) encodes the cost of *being* KO'd:
mega-ex gives the opponent **3** prizes, ex **2**, basics **1**. In the ATTACK branch
(`:137`) a lethal hit only gets the giant `ko_win` weight when it actually closes the game
(`len(op.prize) <= _prize_count(opp_active)`); otherwise it's a strong-but-normal attack. This is
hypothesis #3 from the backlog (`program.md:48`) made concrete.

## The Crustle-routing machinery (earned, v2→v4)

These rules only fire when `crustle` is true (opponent active is the wall, `:111`):

- **Energy routing** (ATTACH branch, `:152`): when walled, `+500` to attaching onto
  Makuhita/Hariyama — "Hariyama is our ONLY answer to the wall, build it fast." Don't starve the
  ex, but prioritise the counter.
- **Boss's Orders drag** (PLAY branch, `:174`): drag a *non-Crustle* benched Pokémon out so your
  charged Lucario can KO it (ex damage works on anything *except* Crustle). Gated on there being a
  bench target and no enemy stadium blocking it (`:177`).
- **Gravity Mountain over Battle Cage** (`:187`): the Crustle deck runs Battle Cage to block Boss;
  overwrite it with our stadium *before* playing Boss (`:191`, weight 9500 > Boss's 9000).
- **Switch / retreat to Hariyama** (`:184`, `:198`): once a charged Hariyama is benched
  (`_hariyama_ready`, `:96`), Switch/Retreat to bring it active and actually swing the counter.

The honest scoreboard (from `experiments.md`): this took Crustle win-rate from 12% → 24% across
v2/v3/v4. Still a losing matchup (the top public bot hits ~70% with deeper play) — but it's a
*stall* deck that's rare on the ladder, and the general-play improvements compounded:
ladder score went **441 → 676 → 820.7** (v1 → v2 → v4).

## The energy-spread rule (v5, task-7 — the one that cleared the gate)

The most recent kept change (`agent_lucario.py:156`):

```python
elif _is_main_attacker(pk.id):
    s += W["energy_need"]
    if len(pk.energies) >= 2:
        s -= 1500          # already attack-ready -> spread energy instead
```

Translation: don't keep dumping energy onto an attacker that can already swing — spread it to the
*next* attacker. Mirror Wilson-LB 0.4967 (≥ 0.48 gate), Dragapult 0.225. Five earlier task-7
attempts were reverted; this is the first to clear both gates. That discipline — only keep a change
that *measurably* helps — is [03](03-autoresearch-loop.md).

## Why it never crashes

`agent` (`agent_lucario.py:290`) wraps everything in try/except → `_legal_fallback` (`:274`),
which just returns the first `minCount` legal indices. A crash forfeits the game
([01](01-game-and-engine.md)), so "always return something legal" is non-negotiable. 1000+ test
games, 0 crashes (`experiments.md`, LUCARIO-v1).

> Takeaway: it's greedy option-scoring. The skill is in the weight ladder (develop-then-attack)
> plus a handful of meta-specific rules, each one paid for by a logged win-rate gain.
