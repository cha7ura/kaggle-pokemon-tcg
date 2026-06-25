# Feature Inventory — everything we can extract

Exhaustive list of features for the card-policy (`docs/superpowers/specs/2026-06-25-card-policy-design.md`).
Each = name → source. Two halves of every training row: **STATE** (the situation) + **OPTION** (the
candidate action/card). All sourced from the engine obs (`cg.api`), the card KB (`data/cards.json`), or
the deck tracker. Nothing here needs the wiki or anything not observable in-sim.

Legend for source: `S`=State, `P`=PlayerState, `K`=Pokemon, `C`=card KB, `T`=tracker(derived),
`H`=threat(derived), `O`=Option/SelectData.

---

## A. STATE — global / turn (`S`)
- `turn`, `turn_action_count` — S.turn, S.turnActionCount
- `going_second` — S.firstPlayer vs S.yourIndex
- `is_first_turn` — turn ≤ 2 (first-turn restrictions: no attack/supporter/evolve going first)
- `supporter_played`, `stadium_played`, `energy_attached_this_turn`, `retreated_this_turn` — S flags
- `stadium_id`, `stadium_mine` — S.stadium[0] (none / whose / which)

## B. STATE — my side (`P`, `K`)
Prizes / deck / hand:
- `my_prize_left` — len(P.prize)
- `my_deck_count` — P.deckCount ; `my_deckout_clock` = deck_count vs turns
- `my_hand_count` — P.handCount
- `hand_dead_count`, `playable_options_count` — hand cards not currently playable (derived)
- `have_draw_supporter`, `have_search_item`, `have_recovery`, `have_stadium_in_hand`,
  `have_gust`, `have_switch` — scan hand ids vs card KB effect_flags
Active:
- `my_active_hp`, `my_active_maxhp`, `my_active_dmg` (maxhp−hp), `my_active_energy_count`
- `my_active_energy_by_type` — histogram of K.energies
- `my_active_special_energy` — K.energyCards ids ∩ SPECIAL_ENERGY
- `my_active_tool_id`, `my_active_tool_dmg`/`_hp`/`_retreat` — K.tools → card KB effects
- `my_active_status`: asleep/paralyzed/confused/poisoned/burned — P flags
- `my_active_stage`, `my_active_is_ex`, `my_active_prize_value`, `my_active_retreat_cost` — K.id → KB
- `my_active_attack_ready` — energy match vs its best attack (see H)
- `my_active_appeared_this_turn` — K.appearThisTurn (can't evolve / freshly played)
Bench:
- `my_bench_count`, `my_bench_free` = P.benchMax − len(P.bench)
- per-bench: `hp`, `dmg`, `energy_count`, `appearThisTurn`, `can_evolve_now`, `tool_id` (top-K benched)
- `my_bench_damage_total` — sum of damage counters (spread exposure)
- `my_strongest_bench_attacker_ready` — best benched attacker that can attack if promoted

## C. STATE — opponent side (`P`, `K`; hand contents hidden)
- `opp_prize_left` — len(opp.prize) ; `prize_diff` = opp_prize_left − my_prize_left
- `opp_deck_count`, `opp_deckout_clock`
- `opp_hand_count` — opp.handCount (count only; contents hidden)
- `opp_active_hp/maxhp/dmg/energy_count/energy_by_type/special_energy`
- `opp_active_tool_*`, `opp_active_status`, `opp_active_stage/is_ex/prize_value/retreat_cost`
- `opp_bench_count`, `opp_bench_free`
- per opp-bench: `hp`, `dmg`, `energy_count`, `is_ex`, `prize_value`, `tool_id` (the gust-target table)
- `opp_bench_damage_total` — softened targets present

## D. STATE — threat / lethal (derived, `H`) — mechanics-correct (SV: weakness ×2, resistance −30)
- `our_max_damage_vs_active` and `vs_each_bench` — incl. scaling_basis evaluated on live state
  (per_energy_self/both, per_hand_card, on_ko_last_turn, on_opp_prizes/my_prizes, per_bench, per_damaged,
  coinflip-EV), bench-spread attacks, weakness/resistance on Active only
- `our_lethal` — can we KO to take our last prize(s) this turn
- `imminent_ko_count` — opp Pokémon with hp_remaining ≤ our reachable hit (gust/spread targets)
- `opp_max_damage_to_my_active` — incl. their attached energy, weakness, **movable energy** to a benched
  attacker (Energy Switch/abilities), Boss reach onto our bench
- `opp_lethal_next` — opp can KO my active (or take winning prizes) next turn → defensive trigger
- `amax`/`bmin`-style race flags — can I attack and survive / am I dead next turn regardless

## E. STATE — KO / history (state-tracker, `T` from `obs.logs`)
- `active_koed_last_turn`, `ally_ko_last_turn` — feeds revenge attacks (on_ko_last_turn) + reactive cards
- `opp_supporter_seen_count`, `opp_gust_seen_count` — what they've spent (safe-to-bench reads)
- `opp_seen` set — opponent cards revealed via logs/board (belief / archetype-prior)

## F. STATE — deck tracking (derived, `T`; we know our decklist)
- `deck_remaining[id]` — decklist − seen-in-play/discard/hand (stateless recompute → reshuffle-safe)
- `draw_prob_of_needed` — hypergeometric P(draw key piece in next k) via math.comb
- `needed_piece_in_discard` — certain fetch available (discard fully visible)
- `prized_likely[id]` — count − seen − in_hand (likely in face-down prizes → revise plan)
- `discard_energy_count`, `discard_recoverable_pokemon` — recovery fuel

## G. OPTION — the candidate action (`O`, `C`) — one row per legal option
- `context` (MAIN/ATTACK/ATTACH_FROM/DAMAGE_COUNTER/EVOLVE/RETREAT/…) — SelectData.context
- `opt_type` (PLAY/ATTACH/EVOLVE/ABILITY/RETREAT/ATTACK/…), `opt_area`, `opt_index` — Option
- `n_options`, `min_count`, `max_count` — SelectData
- `remain_damage_counter`, `remain_energy_cost` — SelectData (placement/energy sub-selects)
- `context_card_id` — SelectData.contextCard (which card/attack drives this sub-selection)
- the option's **card features** (from KB, by Option.cardId / target K.id):
  - `effect_flags` (21 one-hot: accelerate_energy, draw, search, gust, heal, spread_damage,
    inflict_status, switch, move_energy, disrupt_hand, recover_from_discard, setup_attack, protect,
    damage_boost, retreat_reduce, devolve, item_lock, deck_refresh, prize_manipulate, conditional_activation)
  - `scaling_basis` of the attack (if ATTACK option)
  - `card_static`: card_type, energy_type, hp, retreat_cost, stage, is_ex/mega/tera, prize_value,
    best_dmg, dmg_per_energy, ace_spec
  - `card_id` (raw) + `card_category` bucket — identity (per-card behavior w/o per-card models)
- **threat-relevant option features** (the key ones, derived for the option's target):
  - for DAMAGE_COUNTER/DAMAGE: target `hp_remaining`, `creates_imminent_ko` (bool)
  - for ATTACK: `damage_vs_current_target` (scaling-evaluated), `is_lethal`, `prize_value_of_target`
  - for ATTACH/SWITCH_ENERGY: `attack_readiness_delta` (does it enable an attack/retreat)
  - for EVOLVE: `can_evolve_now`, `rare_candy_skip`, `stage_gain`
  - for RETREAT: `retreat_cost`, `escapes_lethal` (opp_lethal_next on current active)

## H. Per-archetype value hooks (derived, used as features + heuristic gates)
- `alakazam_powerful_hand_dmg` = 20 × hand_size
- `trevenant_revenge_armed` = ally_ko_last_turn (→ +100)
- `dragapult_imminent_ko_after_spread` — bench targets reachable after Phantom Dive
(These are just named combinations of the above; the model can also learn them from the raw features.)

---

## Vector size (rough)
STATE ≈ 60–80 dims (A–F), OPTION ≈ 40 dims (G). Per training row = STATE ⊕ OPTION ≈ ~110 dims.
Far richer than today's 30-dim semantically-blind tree — the whole point.

## Build mapping
- `features.py` — A, B, C (raw obs → numbers); assembles the full vector
- `threat.py` — D, and the threat-relevant parts of G
- `deck_tracker.py` — E, F
- `card_features.py` — C/G card-static + effect_flags + evolution graph (from `data/cards.json`)
- All stdlib/NumPy; identical code offline (replay obs) and in-sim (live obs).
