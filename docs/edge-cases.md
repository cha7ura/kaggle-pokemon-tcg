# Edge Cases — gaps a feature policy misses, and how we encode them

Consolidated from competitive-doc research + rulings + the recurring rule-of-thumb. Each: case →
why it changes the play → observable state → **status** (encoded / backlog). The engine enforces
*legality* (illegal lines never appear as options); we only need features for *value*.

## The three questions (rule of thumb that catches most of these)
1. **damage, counters, or just effects?** — "place damage counters" and "effects" bypass the
   weakness ×2 / resistance / damage-reduction pipeline that "damage" goes through.
2. **retreat, switch, or move?** — "block retreat" does NOT block switch/Boss/move effects.
3. **your / each / that?** — scope: global vs only-mine vs single-target. Changes whether it applies.

## A. Damage pipeline (HIGHEST leverage — pure arithmetic, fires constantly)
- **A1 additive-before-weakness:** order = `(base + tool/ability boosts) × weakness − resistance −
  reduction(last)`. (70+50)×2=240, not 190. → *backlog (needs tool-damage modelling)*.
- **A2 "place/move damage counters" bypasses weakness/resistance/reduction.** → **ENCODED**:
  per-attack `damage_kind ∈ {damage, counters, no_weakness}`; threat applies weakness only to `damage`.
- **A3 defensive HP tools** (Hero's Cape +100, Bravery Charm +50) move the KO threshold. → *backlog
  (model `tools` HP on the defender vs our final damage)*.
- **A4 "not affected by any effects on Active"** (Cornerstone Demolish) = flat through reduction, but
  no weakness bonus. → **ENCODED** as `damage_kind=no_weakness`.
- **A5 ability-lock wall** (Cornerstone: "prevent all damage from Pokémon that have an Ability"):
  most meta attackers have abilities → 0 damage. → *backlog (needs defender wall flag + attacker
  `has_ability`)*. We now expose `has_ability` (below).
- **A6 Rocky Helmet / recoil** on the attacker even on KO. → *backlog*.

## B. Prize parity (huge swings, pure arithmetic) — mostly ENCODED via state
- **B1 at 1 prize, ANY KO wins** → take the safe/easy KO. `my_prize_left`, per-target `prize_value`. ENCODED.
- **B2 force the 7th prize** — single-prize attackers trade up. `prize_diff`, board ex-count. ENCODED (features present).
- **B3 prize-deficit cards** (Counter Catcher/Defiance Band gate on being behind) — stay behind to keep
  them ON. `prize_diff` sign. ENCODED (state) — engine gates availability.
- **B4 coin-flip gated by ahead/behind** — gamble only when the safe line loses. `prize_diff` + base≥HP. ENCODED.

## C. Bench / spread
- **C1 Manaphy blocks "damage" snipe, NOT counters** (Dragapult/Froslass/Munkidori). → tie to A2 + opp archetype. *partial*.
- **C2 counter-MOVERS** (Munkidori/Dusknoir) make lethal invisible on static board. → *backlog (movable-counter sim)*.
- **C3 don't bench what you won't use; lone ex on bench = free 2 prizes to Boss.** `opp_gust_seen`, bench
  prize_value. *partial (bench prize features present; gust-count backlog)*.

## D. Locks / denial (engine masks legality; we need value + threat-awareness)
- **D1 Path to the Peak** shuts Rule-Box abilities **symmetrically** → fire yours first. needs `rule_box`. *backlog*.
- **D2 active-only locks** (Klefki=Basic abilities, Flutter Mane=opp abilities, Iron Thorns) → value gusting the lock. *backlog*.
- **D3 item lock** → Items dead; engine won't offer them, but value pre-loading. *engine-gated*.
- **D4 retreat lock / Corner** → only Switch/Escape escapes (NOT retreat). `retreat ≠ switch`. *backlog (escape-card state)*.

## E. Sequencing (free EV) — partly the model's job to learn
- **E1 free draw/search abilities BEFORE supporter/attach/evolve.** ENCODED via `ability_available` + the model.
- **E2 attach to the BENCH backup once active is charged.** state has bench energy + attack_ready. *partial*.
- **E3 attach/accelerate BEFORE retreat.** model learns from ordering in winner replays. *learned*.
- **E4 own draw/search BEFORE Iono/Judge.** hand-quality + deck_remaining. ENCODED (state).

## F. Special conditions & checkup (game-losing if missed)
- **F1 self-checkup KO:** my poisoned/burned Active can die on MY end step → clear before passing.
  `active poisoned/burned AND hp ≤ checkup_dmg`. → **backlog (high priority — add `self_checkup_lethal`)**.
- **F2 leave opp poisoned Active to die; don't cure it.** opp active poisoned, hp≤10, + alt action. *backlog*.
- **F3 don't KO an asleep/paralyzed attacker** you can leave neutralized. status flags present. *learned*.
- **F4 evolve clears all status + can attack same turn.** evolution-in-hand + status. *partial*.
- **F5 confusion 30 self-damage; never attack into it if hp≤30.** ENCODED-able (status + hp). *partial*.
- **F6 paralysis self-clears 1 turn — don't waste a Switch.** *learned*.

## G. KO timing / retreat-vs-switch / evolution / once-per-turn (the user's list)
- **G1 attack EFFECTS ≠ damage** — reduction/prevention may not stop the effect. → `damage_kind`/effect_flags. *partial*.
- **G2 double-KO resolution order** — next player promotes first; both take prizes. *engine-handled; minor for value*.
- **G3 "switch your Active" ≠ retreat** — retreat-lock doesn't block it. → flag switch vs retreat options. *backlog*.
- **G4 evolution clears effects/status** but keeps damage counters + energy + tools. ENCODED (mechanics in threat/features).
- **G5 once-per-turn is per-COPY** — a 2nd copy of the ability is usable. → engine offers it; *engine-gated*.
- **G6 condition removal scope** (Active-only vs Bench too) — wording. *card-text; mostly engine-handled*.
- **G7 prize-count effects stack oddly** with multi-prize. `prize_value` + Legacy/Lillie's reducers. *partial*.
- **G8 search → must shuffle** (unless card says not) — affects later reveal. *engine-handled*.

## H. Turn-1 / donk
- **H1 first player T1: no Supporter, no attack.** `going_second` + `is_first_turn`. ENCODED (engine also gates).
- **H2 donk** — going-2nd T1 KO of a lone Basic. lone-basic + our T1 damage. *partial (threat present)*.

## Status summary
- **Encoded now:** damage-vs-counters weakness skip (A2/A4), prize parity/diff (B), going-first/T1 (H1),
  status flags, ability_available, hand-quality/tracking (E4), `has_ability`+`rule_box` flags (new).
- **High-priority backlog:** F1 self-checkup-lethal, A1/A3 tool-damage pipeline, D-series lock flags,
  C2 movable-counter sim, G3 switch-vs-retreat option flag.
- **Engine-gated (no feature needed):** illegal-line masking for locks/item-lock/once-per-turn/first-turn.
- **Learned (leave to the model from winner replays):** most sequencing (E1–E4), F3/F6.
