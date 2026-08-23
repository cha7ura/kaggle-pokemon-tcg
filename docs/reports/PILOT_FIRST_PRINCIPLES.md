# First-principles pilot analysis: what our agent conditions on vs the full state

## What the game IS (first principles)
At each engine menu, pick the option that best advances a RESOURCE PLAN toward 6 prizes before the
opponent. We FULLY observe our own state (hand, active, bench, discard, deck count, prize count,
energy locations, tools) and PARTIALLY observe the opponent (active/bench/discard/counts visible;
their hand hidden). The whole policy is menu-selection over this state.

## What our shipped pilot (agent_lucario) actually conditions on
GOOD (present):
- KO math: dmg >= opp_active.hp, prize count of target
- Keep main attacker charged; don't over-attach (spread once >=2 energy)
- Boss's Orders targeting; discard-choice protection; damage-counter placement (max prize target)
FIXED PRIORITY (the W table): ability 30000 > play_pokemon 20000 > evolve 9000 > attach 8000 >
  switch 6000 > boss 3200 > draw 3000 > retreat 2000 > attack 1000 > end. Greedy, not planned.

MISSING (from first principles):
- Energy in DISCARD + energy ACCELERATION (trainers that recover/accelerate energy) — not modeled
- Trainers/items that FETCH or REDISTRIBUTE energy — not modeled
- Opponent NEXT-TURN threat (can they KO me? should I deny it?) — not computed
- Resource race / deck-out clock — only a crude low_deck flag
- Hand composition planning (do I have the evolve piece? the energy? sequence them)

## THE CRITICAL FINDING
Every piece of SMART reasoning in agent_lucario is hardcoded to LUCARIO-deck card IDs:
RIOLU, MEGA_LUCARIO, HARIYAMA, MAKUHITA, SOLROCK, LUNATONE, CRUSTLE matchup, HERO_CAPE, GRAVITY_MTN.
- `_is_main_attacker()` returns True ONLY for Lucario cards.
- SETUP_ACTIVE / SETUP_BENCH scoring returns 0 for every non-Lucario card; TO_HAND uses a generic
  score (200 minus dupes) with the only card-specific bonus (+40) hardcoded to RIOLU — so no
  Grimmsnarl card gets any deck-specific search priority.
=> When this pilot plays GRIMMSNARL (what we just shipped), NONE of the smart logic fires. Grimmsnarl
   is played on the bare W fallback: random opener, random bench, no attacker identification, no
   Dark-energy targeting, no Munkidori/Froslass synergy. Legal, won't crash — but not piloted.

## Why heuristic > learning here (settled this session)
Behavioral clone of the 82% expert scored 45-47% (below coin-flip) with base AND rich features — a
static classifier can't represent sequential intent. A hand-written pilot encodes that intent as
rules over the fully-observed state, which is why the heuristic beats every learned pilot (11-26%).
=> The path is NOT more ML. It is extending the heuristic's resource reasoning + making it deck-aware.

## Ranked next moves (all buildable, heuristic not ML)
1. DECK-SPECIFIC PILOT for what we ship (Grimmsnarl now): setup priority (open Impidimp, bench 2nd,
   evolve Morgrem->Grimmsnarl ex ASAP), attach Dark to the evolving attacker, Munkidori damage-move,
   Boss targeting. Directly fixes "we ship a deck the pilot can't drive." HIGHEST VALUE.
2. ENERGY PLANNER: count energy in play+hand+discard+fetchable; commit to one attacker; model accel.
3. THREAT/RESPONSE: estimate opp next-turn KO from visible energy+HP; retreat/deny or race.
4. ARCHETYPE-ADAPTIVE: ID opponent (~88% at 6 cards) + matchup matrix -> shift plan per matchup.

## Caveat
Offline sim != ladder, so validate any pilot change with deck_gate (no-crash + WR vs heuristic) and
confirm live. But a deck-aware pilot for our shipped deck is the clearest, most-defensible lever left.
