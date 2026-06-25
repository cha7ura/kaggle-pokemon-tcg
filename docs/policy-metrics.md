# Policy Metrics & Knowledge Research

What a strong Pokémon TCG policy must *know* and *compute*, mapped to our `cg` obs API.
Sources: card-game AI papers (LOCM, MTG ensemble-determinization, Hearthstone, Grigsby offline-RL
transformers), competitive PTCG guides (JustInBasil, SixPrizes, Limitless), and per-archetype tech.

## TL;DR — three layers we don't fully have

1. **Card-knowledge layer** — every card id → full static properties (HP, attacks, cost, weakness,
   stage, evolvesFrom). We HAVE this: `cards_full.csv` (1267 cards) + in-engine `all_card_data()`.
   Used today only ad-hoc inside the Alakazam pilot; not exposed to the learned policy.
2. **State-tracking / belief layer** — a running memory across the game: opponent's revealed cards,
   our deck-remaining counts → draw probabilities, deck-out risk, prize map. **NOT BUILT.**
3. **Eval/feature layer** — the metrics below, fed to a heuristic score or a learned model.
   Today: rich-but-hand-coded (Alakazam pilot) OR 30 semantic-blind features (learned tree). Both weak.

The single biggest gap is that the *learned* policy sees `opt_type/area/index` only — no card meaning,
no threat, no tracking. Every metric below is invisible to it. That's why the Dragapult cell is inverted.

---

## 0. Key strategic finding from the literature

- **Grigsby et al. 2025 (arXiv:2504.04395), "Human-Level Competitive Pokémon via offline RL with
  Transformers":** top-10% vs humans, **no explicit card-tracking** — a sequence model over the
  trajectory *learns* the hidden-info belief. Beat heuristic search AND LLM agents. → the ceiling is a
  trajectory model, not hand-written tracking. Our learned tree is the impoverished version of this
  (single-step, 30 features, no history).
- **PTCG-Bench (2605.29653) & PokeAgent Challenge (2603.15563):** partial observability is THE
  challenge; specialist RL > generalist LLM, both < elite human; performance "highly sensitive to
  harness design" = **feature engineering is the lever**, exactly our bottleneck.
- **LOCM eval template (2105.01115):** `value = Σ wᵢ·state_featureᵢ + Σ evalCard(mine) − Σ evalCard(opp)`.
  Linear, weights learnable/evolvable. The cleanest thing to port.
- **MTG ensemble-determinization (Cowling 2012):** algorithmic lethal/survival gates + trade-favorability
  (kill higher-cost, lose lower-cost). Directly portable as action gates.

---

## 1. Evolution mechanics (base → stage1 → stage2, Rare Candy skip)

Fully exposed by the engine — we just need to use it consistently.

- `CardData.basic / stage1 / stage2` (bool), `evolvesFrom: str|None`. In `cards_full.csv` already.
- Contexts: `EVOLVE` (37), `EVOLVES_FROM` (18), `EVOLVES_TO` (19), `DEVOLVE` (20). OptionType `EVOLVE` (9).
- **Rare Candy (id 1079)** lets Basic → Stage 2 directly, **skipping Stage 1**. The Alakazam pilot already
  exploits this (Abra→Alakazam, `main.py:383 can_rare_candy_alakazam`). Abilities like Psychic Draw fire
  through Rare Candy.
- `Pokemon.preEvolution` lists the pre-evo cards stacked under a Pokémon — read it to know stage reached.
- **`LogType.EVOLVE` (12) / `DEVOLVE` (13)** appear in `obs.logs` → we *observe* opponent evolutions live.

Computable evolution metrics:
- `evolve_ready(p)` = we hold the evolution of an in-play Pokémon (and it didn't `appearThisTurn`).
- `rare_candy_line_complete` = have Basic in play + Stage-2 + Rare Candy in hand → can skip.
- `stage_exposure` = turns a fragile mid-evolution (e.g. Drakloak, Kadabra) sits un-evolved = gust/KO bait.
- opponent `evolution_threat` = they have a Basic of a known dangerous Stage-2 line in play (Dreepy → Dragapult).

---

## 2. Card-knowledge layer (research ALL cards, identify opponent's)

We can identify any card the moment it's visible because `id` joins to the full DB.

- **What's visible** (see policy visibility table): both players' active+bench Pokémon (id, hp, energies,
  energyCards, tools, preEvolution), both discards, our hand, stadium, `opp_hand_count`.
- **What's hidden:** opponent hand contents, both deck orders, prize contents.
- **On reveal:** `obs.logs` reports every card played/moved with `cardId` → we learn opponent cards as
  they're used, even from hand (they pass through a visible zone). Append to a per-game "seen" set.
- Per-card derived (precompute once from `cards_full.csv`): `best_dmg`, `min_cost_for_best`,
  `dmg_per_energy`, `prize_value` (1 / 2 ex / 3 megaEx, −Legacy/−Lillie adjustments), weakness/resistance.

Computable: `opp_attack_threat(p)` = for opponent Pokémon `p`, its `best_dmg` achievable given its
*current attached* energies (and weakness vs our active doubling). Drives the threat model (§4).

---

## 3. State-tracking & probability layer (card counting toward the end) — TO BUILD

This is the new piece the user asked for. A small running tracker, updated each `agent()` call.

Our deck is fully known (60-card decklist = `deck.csv`, with per-id counts). So:

- `deck_remaining[id]` = `count_in_decklist[id] − seen_in_play_or_discard_or_hand[id]`.
- `deck_size_remaining` = `me.deckCount` (engine gives it directly).
- **Draw probability** of card X on next draw: `deck_remaining[X] / deck_size_remaining`.
- **P(at least one X in next k draws)** (hypergeometric) ≈ `1 − C(D−n, k)/C(D, k)` where
  `D = deck_size_remaining`, `n = deck_remaining[X]`. Drives "do I dig now or hold?" and ACE-SPEC timing.
- **Deck-out clock** = `deck_size_remaining` vs turns left; stop over-drawing when low (rules F33–36).
- **Prize probability**: a needed card not seen and not in deck-remaining → it's likely prized
  (`prized_likely[X] = count − seen − (in_hand)`), revise plan to dig a prize.

Opponent side (weaker, archetype-prior based):
- Maintain `opp_seen[id]` from logs + visible board. Match against known archetype decklists (Limitless)
  to *infer* their list and what they likely still hold (belief, not certainty — cf. Grigsby: a model
  learns this; our heuristic approximates it).
- `opp_supporter_used_this_turn`, `opp_boss_seen_count` (how many gusts spent) → safe-to-bench reads.

`ponytail`: this is one `deck_tracker.py` with a dict + hypergeometric helper. No framework. Stdlib `math.comb`.

---

## 4. The metric catalog (state-value + threat + tempo), mapped to obs

### 4a. State-value features (LOCM-style, both as score terms and learned-tree inputs)
| Feature | Definition (computable) | Source/why |
|---|---|---|
| prize_diff | `opp_prize_left − my_prize_left` (win at 0) | primary objective (LOCM health-diff analog) |
| card_adv | `my_hand + my_board − opp_hand_count − opp_board` | 3-resource model |
| board_presence | `#my_pokemon_in_play − #opp` | board control |
| energy_on_attacker | attached energy on our best attacker vs its attack cost | tempo / attack-ready |
| energy_ceiling | total energy in play + accel sources | ramp advantage |
| deck_remaining | §3 | deck-out + draw odds |
| hand_size | `len(my_hand)` (Alakazam: == damage!) | option count / archetype dmg |
| status flags | asleep/para/poison/burn/confuse both actives | combat math |
| weakness_hit | our attacker type is opp active weakness (2×) | huge swing |

### 4b. Threat / lethal / survival gates (apply BEFORE scoring — MTG-derived)
- `our_lethal` (bool): can we KO enough this turn to take our last prize(s)? → if yes, take it.
- `opp_lethal_next` (bool): `opponent_max_damage(obs) ≥ my_active.hp` (incl. weakness, attached energy,
  Boss reach onto bench) → triggers defensive retreat/heal/switch. **Absent everywhere today.**
- `amax` / `bmin` (how aggressive can I be while surviving) → race-vs-stabilize switch.
- `our_max_damage_vs_each(opp_pokemon)` → attack targeting + Dragapult spread planning.

### 4c. Prize-trade (the competitive core)
- `favorable_trade` = `my_active.prize_value < target.prize_value` → prefer.
- `force_7th_prize` = present only 1-prize attackers when opp map needs few multi-prize KOs.
- `gust_target` = benched opp Pokémon with `hp_remaining ≤ our_attack_damage` AND prize-worthy → Boss it.
- `counter_catcher_window` = we're behind on prizes → item-gust + draw + attack multi-action turn.

### 4d. Tempo / sequencing (mostly heuristic priors)
- never miss energy attachment; attach LAST in turn.
- draw-supporter before search before choice-abilities; Boss/gust last.
- don't bench a multi-prize Pokémon with no role this turn (gust bait).
- energy-denial > chip damage when opp needs a specific type.

---

## 4e. Action model & forward search (the action chain spawns new actions)

A turn is NOT a flat action list — it's a **chain of selection nodes**. Each `agent(obs)` call resolves
ONE node (`obs.select.option` = the legal actions at that node, with a `context`). Choosing an action
that has consequences spawns the next node: e.g. a search trainer "look at deck, add to hand" →
`SelectContext.TO_HAND` over `obs.select.deck`; attach energy → pick energy → pick target. You keep
getting called until the turn resolves (attack/END). So "choose one or a combination, each action gives a
new sequence" IS the native design — you don't pre-enumerate the turn, you walk the chain.

**The engine supports LOOKAHEAD over this chain** (`cg.api`):
- `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand,
  opponent_active, manual_coin)` → root `SearchState`. You must supply the **hidden info as a guess**:
  your deck/prize order, and the opponent's deck/hand/prize/active. (If `obs.select.deck` is set, your
  deck is taken from the real state.)
- `search_step(search_id, select)` → apply a selection, get the next `SearchState`. Walk the whole
  action chain forward, simulate to a leaf, score it with §4a–4d, back up. `search_end` / `search_release` cleanup.
- `manual_coin=True` lets you fix coin flips → evaluate both branches of a flip.

This is the **ensemble-determinization / ISMCTS** architecture (MTG paper) and the engine gives it to us
natively. The catch: `search_begin` needs guessed opponent deck/hand/prize → **the §3 belief/tracking
layer feeds the search.** Better belief → fewer/better determinizations → stronger lookahead. This ties
the thread together:

```
card-knowledge (§1-2)  → identify every visible card
        │
state-tracking (§3)    → guess hidden deck/hand/prize  ─┐
        │                                                ├─→ search_begin/step = forward sim of the action chain
eval features (§4a-d)  → score simulated leaf states  ──┘
```

`ponytail`: don't jump to full MCTS first. The heuristic pilot already walks the chain greedily and wins
(Alakazam 1005). Forward-search is step 4 (the ceiling), valuable once threat+tracking features exist to
score leaves. Build the features first; they're reused whether we stay greedy or add search.

## 5. Per-archetype win-metric (the value function differs per deck)

- **Alakazam:** `damage = 20 × hand_size`. Maximize hand on attack turn. Lethal thresholds:
  60HP→3 cards, 140→7, Dragapult 320→16. Fragile (140 HP, Dark weakness). Battle Cage = anti-spread wall.
- **Hop's Trevenant:** boolean `hop_KO_last_turn` = +100 dmg cliff (30→130). `special_energy_attached`
  (only special energy → energy denial kills it). Mist Energy = spread immunity. Corner = retreat lock.
- **Dragapult:** per-bench `hp_remaining`, `imminent_KO_count` = #bench with `hp_remaining ≤ next_hit`.
  Phantom Dive = 200 active + 60 spread; gate Boss on a softened target. Multi-turn prize-map planner.
  Munkidori/Manaphy/Battle Cage counter the spread.

---

## 6. Build order (smallest → biggest payoff)

1. `threat.py`: `opponent_max_damage(obs)`, `our_max_damage_vs_each(obs)`, `is_lethal_on(active)`.
   Reads visible board + `cards_full.csv`. Unlocks defensive retreat + attack targeting + the inverted
   Dragapult cell. **Start here.**
2. `deck_tracker.py`: `deck_remaining`, hypergeometric draw odds, deck-out clock, prize-likely. (§3)
3. Enrich `option_features` with card-semantic + threat + tracking fields; re-extract + re-train ONE deck
   (Dragapult) and re-run that league cell. If it moves toward real 0.32, roll out.
4. (Bigger) trajectory/sequence policy à la Grigsby instead of single-step tree — the real ceiling.

Decisive experiment for step 3: does adding threat+tracking features fix the Dragapult cell
(sim 0.627 → real 0.32)? If yes, the feature theory is validated and we scale it.
