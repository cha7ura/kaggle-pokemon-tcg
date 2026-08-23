# Full game-state information inventory (for a richer featurizer)

The clone failed at 46.9% top-1 because the current featurizer (23 state + 7 option dims) discards
most of this. Here is EVERYTHING an observation exposes, so a new featurizer can use it.

## Game-level (`current`)
- `turn` — turn number (tempo)
- `turnActionCount` — actions taken this turn
- `firstPlayer` — who went first (0/1)
- `yourIndex` — which seat we are
- `stadium` / `stadiumPlayed` — stadium in play + whether one was played this turn
- `supporterPlayed` — supporter used this turn (you get 1) — GATES supporter options
- `energyAttached` — energy attached this turn (you get 1) — GATES energy options
- `retreated` — retreated this turn
- `result` — -1 ongoing, else game result

## Per-player (both seats; opponent partially hidden)
Zones: `active[1]`, `bench[<=5]`, `discard[]`, `hand[]` (SELF only — opp `hand: null`), `prize[6]`
Counts (visible for BOTH): `deckCount`, `handCount`, `benchMax`, prize length
Status conditions: `asleep`, `burned`, `confused`, `paralyzed`, `poisoned`

### Per-Pokémon (active & each bench)
- `id`, `hp`, `maxHp` — current/max HP (KO math, threat)
- `energies` — list of attached energy TYPES/counts
- `energyCards` — the actual energy cards attached (id/serial)
- `tools` — attached tool cards (id) — e.g. Choice Band, Handheld Fan
- `preEvolution` — the evolution stack beneath (basic->stage1->stage2 depth)
- `appearThisTurn` — just played/evolved this turn (summoning-sick for attack? evolve rules)
- `serial` — unique instance id

## The decision (`select`)
- `context` — SelectContext enum (0=MAIN, 7=TO_HAND draw, 35=ATTACK, 37=EVOLVE, 22=ATTACH_TO,
  3=SWITCH, 41=GO_FIRST, 42=MULLIGAN, 46=COIN_HEAD, 30=DISCARD_ENERGY, ... 48 total)
- `type` — option value type (1=Card,9=YesNo,4=Energy,...)
- `minCount`/`maxCount` — how many to pick
- `effect`, `contextCard` — the card/effect driving the choice
- `remainDamageCounter`, `remainEnergyCost` — for damage-placement / energy-cost decisions
- `option[]` — each: `type`, `area`, `index`, `playerIndex` (+ implied card via index into a zone)

## Features the current 23+7 set is MISSING (candidates — your idea)
1. **Action semantics** per option: is it an EVOLVE (basic->stage1->stage2, use `context`+preEvolution
   depth), an ENERGY ATTACH, a SUPPORTER/ITEM/TOOL play, a RETREAT, an ATTACK, a draw/search?
2. **Discard-pile composition & counts** (self + opp): #energy discarded, #supporters used, key cards
   gone — a strong tempo/resource signal.
3. **Hand size** (self exact, opp count), **deck count** (both) — resource race.
4. **Prize race**: prizes remaining both sides (how close to winning/losing).
5. **Board threat**: for each Pokémon, energy-attached vs attack cost -> "can attack / can KO",
   HP-remaining fraction, is-ex (gives 2-3 prizes if KO'd).
6. **Energy-attach / supporter-used flags** (already in `current`, gate legal options).
7. **Status conditions** on active (asleep/paralyzed/confused block attacks).
8. **Evolution readiness**: have the stage-1/stage-2 in hand for a basic in play + Rare Candy.
9. **Tools attached** (Choice Band = +dmg, etc.).
10. **Turn number** + first-player (early-game tempo differs).

## Why this matters
The expert's choice depends on hand contents, KO availability, and prize race — none well-encoded now.
A featurizer using the above turns the 46.9% (< baseline) clone into a potentially learnable policy.
This is a FEATURE-ENGINEERING build (edit tools/imitation/features.py -> re-extract -> retrain ->
re-validate), the single highest-leverage research task for the policy lever.
