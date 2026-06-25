# Per-Card Data Schema

One record per `card_id` for all 1267 cards. Spine = engine (`all_card_data` + `all_attack`);
trainer effects + set/number + validation = wiki (Bulbapedia/Limitless). Drives `cards_effects.csv`
and the policy feature layer (`docs/superpowers/specs/2026-06-25-card-policy-design.md`).

## Controlled vocabularies

- **card_type:** `POKEMON | ITEM | SUPPORTER | STADIUM | TOOL | BASIC_ENERGY | SPECIAL_ENERGY`
- **stage:** `BASIC | STAGE1 | STAGE2` (Pokémon only)
- **energy_type / weakness / resistance:** `GRASS FIRE WATER LIGHTNING PSYCHIC FIGHTING DARKNESS METAL DRAGON COLORLESS` (+ special: `RAINBOW TEAM_ROCKET`)
- **effect_flags** (multi-label, applies to abilities, attacks, trainers):
  `accelerate_energy draw search gust heal spread_damage inflict_status switch move_energy
   disrupt_hand recover_from_discard setup_attack protect prevent_damage damage_boost
   retreat_reduce devolve item_lock deck_refresh prize_manipulate`
- **scaling_basis** (attack damage formula): `none per_energy_self per_energy_both per_hand_card
   on_ko_last_turn per_bench per_damaged coinflip other`

## Record schema (JSON)

```json
{
  "card_id": 96,
  "name": "Teal Mask Ogerpon ex",
  "card_type": "POKEMON",
  "set": "Twilight Masquerade",        // wiki, nullable
  "number": 25,                         // wiki, nullable
  "wiki_url": "https://bulbapedia.bulbagarden.net/wiki/Teal_Mask_Ogerpon_ex_(Twilight_Masquerade_25)",
  "ace_spec": false,

  // --- POKEMON only (null otherwise) ---
  "hp": 210,
  "energy_type": "GRASS",
  "weakness": "FIRE",
  "resistance": null,
  "retreat_cost": 1,
  "stage": "BASIC",
  "evolves_from": null,
  "is_ex": true,
  "is_mega_ex": false,
  "is_tera": false,
  "prize_value": 2,                     // 1 | 2 (ex/V) | 3 (VMAX/megaEx)
  "abilities": [
    {
      "name": "Teal Dance",
      "text": "Once during your turn, you may attach a Basic Grass Energy from your hand to this Pokémon. If you did, draw a card.",
      "frequency": "once_per_turn",     // once_per_turn | passive | on_evolve | between_turns
      "effect_flags": ["accelerate_energy", "draw"]
    }
  ],
  "attacks": [
    {
      "attack_id": 1234,                // engine attackId
      "name": "Myriad Leaf Shower",
      "energy_cost": ["GRASS", "GRASS", "GRASS"],
      "damage_base": 30,
      "damage_text": "30+ . 30 more damage for each Energy attached to both Active Pokémon.",
      "scaling_basis": "per_energy_both",
      "effect_flags": []
    }
  ],

  // --- TRAINER only (ITEM/SUPPORTER/STADIUM/TOOL; null otherwise) ---
  "trainer_subtype": null,              // = card_type for trainers
  "effect_text": null,
  "effect_flags": [],                   // card-level for trainers
  "conditions": [],                     // e.g. ["if_exactly_6_prizes:draw_8"]

  // --- ENERGY only ---
  "energy_provides": null,              // e.g. ["PSYCHIC"] or ["COLORLESS","COLORLESS"]; special-energy effect in effect_text

  // --- provenance ---
  "source": { "engine": true, "wiki": true, "validated": false }
}
```

## Example 2 — conditional Supporter (Trainer record)

```json
{
  "card_id": 1182,
  "name": "Lillie's Determination",
  "card_type": "SUPPORTER",
  "set": "Mega Evolution", "number": 119,
  "wiki_url": "https://bulbapedia.bulbagarden.net/wiki/Lillie%27s_Determination_(Mega_Evolution_119)",
  "ace_spec": false,
  "hp": null, "energy_type": null, "stage": null, "abilities": null, "attacks": null,
  "trainer_subtype": "SUPPORTER",
  "effect_text": "Shuffle your hand into your deck. Then, draw 6 cards. If you have exactly 6 Prize cards remaining, draw 8 cards instead.",
  "effect_flags": ["draw", "deck_refresh", "prize_manipulate"],
  "conditions": ["if_exactly_6_prizes:draw_8"],
  "source": { "engine": true, "wiki": true, "validated": false }
}
```

## Acquisition pipeline (→ `cards.json`)

1. **Engine spine — COMPLETE, the only required step** (`tools/build_card_text.py`, docker, 1 run, DONE
   2026-06-25 → `autoresearch/data/cards_engine.json`, 1267 cards). Engine `all_card_data()` carries the
   FULL effect text: Pokémon abilities (`CardData.skills`, 218 cards incl. Teal Dance), attacks
   (`Attack.text`), **and all 191 Trainer effects** (Lillie's, Rare Candy …) + special energy. The old
   `cards_full.csv` only looked empty because its extractor read `attacks` and skipped `skills`. So the
   wiki is NOT needed for card effects — the engine is the authoritative, complete source.
2. **Classify** (`tools/build_effect_flags.py`): text → `effect_flags` + `scaling_basis` (keyword rules,
   optional offline LLM for ambiguous). Unparseable → empty flags + `flag_needs_review`. Emit
   `cards.json` (full schema) + flat `cards_effects.csv` (the columns the feature layer consumes).
3. **Wiki enrichment — OPTIONAL, de-scoped** (`tools/fetch_card_wiki.py`): only for `set`/`number`/
   `wiki_url`/`competitive_role`/rulings, not for effects. Entry point:
   `bulbapedia.../Browse:Trading_Card_Game` → expansion pages → per-card URLs
   (`{Name}_({Set}_{Number})`). Batched/rate-limited/resumable; can lag indefinitely without blocking.

Submission ships only the small derived `cards_effects.csv` — never the scrape, never a network call.
