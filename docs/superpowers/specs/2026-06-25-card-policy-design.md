# Card-Policy → Gameplay-Policy — Design Spec

**Date:** 2026-06-25
**Goal:** Learn a per-card play policy ("when to use a card / when not to") from others' gameplay,
then compose those into a gameplay policy that drives BOTH the live Kaggle ladder agent and the
imitation-league opponents. Replaces the hand-coded per-deck pilots and the weak per-deck learned trees.

**Companion reference:** `docs/policy-metrics.md` (metrics catalog, obs visibility, mechanics, build order).

---

## 1. Problem & why now

Two policy systems exist today, both weak:
- **Hand-coded pilots** (e.g. `submission_alakazam_top/main.py`, 814 lines): rich card semantics but
  archetype-specific, zero generalization, hand-labor per deck.
- **Learned per-deck trees** (`tools/imitation/policies/*.json`): generic but **semantically blind** —
  features are only `context/opt_type/opt_area/opt_index` (30 dims, no card meaning, no threat, no
  tracking). This is why the imitation-league **Dragapult cell is inverted** (sim 0.627 vs real 0.32):
  the model can't see lethal lines or spread/snipe threats.

We want one artifact: a **card-aware learned policy**, grounded in correct mechanics, trained on
**winning** human play, that plays any deck and is NumPy/stdlib-inferable for submission.

## 2. Constraints (hard)

- **Inference:** offline, stdlib + NumPy only (Kaggle submission runs `agent(obs)` via `exec()`, no heavy
  ML deps, no network). Training may use sklearn offline; the model must EXPORT to a JSON/array form
  walked in pure Python/NumPy — the **same export+walk path the current `*.json` policies already use**.
- **Engine:** `cg` is linux/amd64, docker-only off-Kaggle. League runs in docker; submission runs on Kaggle.
- **Data latitude (granted 2026-06-25):** may download more replays (once Kaggle 429 throttle clears) and
  train as long as needed. Use this for the winner-filter + sparse-card coverage.

## 3. Success criteria (in order of feedback speed)

1. **Held-out decision accuracy** — given a replay decision from a *winner's* seat, does the model rank
   the actually-chosen option top-1 among the legal set? Target: beat the current tree and a
   most-frequent-action baseline by a clear margin on a held-out replay split.
2. **League recalibration** — re-run the imitation league with the new policy as opponents; the
   **Dragapult cell moves from 0.627 toward the real 0.32** (and aggregate stays calibrated ~0.60).
3. **Ladder A/B** — submit a gameplay-policy agent; compare score to proven champs (Alakazam 1005,
   trev_typh 928). High-variance, slow — judged over 2–3 submissions (see [[our-team-identity]] cadence).

## 3b. Known risk — learned agents historically don't transfer to the ladder

Project history ([[wiki-minizero-build]]) is blunt: **7 ML attempts (net/GBM/ISMCTS/value-search/RL) all
underperformed on the ladder; only copying a real top deck ever moved it; "no predictive offline signal
exists."** This card-policy is a learned-imitation agent — the same class. So we set expectations honestly:

- **Primary, validated payoff = fix the imitation LEAGUE** (the sim oracle). A calibrated league makes
  deck-search trustworthy and is independently valuable, ladder aside. Success criterion #2 is the real bar.
- **Ladder = a long shot**, judged last, over 2–3 submissions, never displacing a proven champ on a single
  noisy score (see [[our-team-identity]] cadence). The winner-seat filter + mechanics-correct features are
  the bet that *this* learned agent transfers better than the prior 7 — but we do not assume it will.
- If decision-accuracy is high AND the league recalibrates BUT the ladder still doesn't move, that is a
  **success for the oracle** and a (known-precedented) null for the ladder — not a failure of the build.

## 4. Architecture

Five units, each one responsibility, each independently testable.

```
replays.sqlite
     │  (1) decision extractor  — winner-seat filter
     ▼
decisions table: (state_feats, option_feats, label)   ── pooled across ALL cards
     │  (3) train RandomForest ranker  [sklearn, offline, long]
     ▼
card_policy.json  (forest of trees → Python/NumPy-walkable)
     │
     │  loaded by ──►  (4) gameplay_policy.py  ◄──  (2) feature layer
     │                       │                       (threat + tracker + card-static + state)
     │                       ├──► live submission (stdlib/NumPy)
     │                       └──► league opponents (docker)
     ▼
(5) validation: decision-accuracy → league cell → ladder A/B
```

### Unit (1) — Decision extractor  `tools/imitation/extract_decisions.py` (extend)
- A **decision point** = any replay `select` with `len(option) >= 2`, excluding the 60-card deck pick.
- Emit **one row per option** at each decision point: `(state_features, option_features, label)`,
  `label = 1` if that option index was in the chosen action list, else `0`. Multi-select (minCount>1)
  → every chosen option labeled 1.
- **Winner-seat filter:** keep only decisions made by the seat whose `reward` indicates the win
  (`reward0>reward1` → seat 0 is positive policy, and vice-versa; draws excluded or down-weighted).
  Optional Elo weight if available. This makes it "imitate winners," not "imitate everyone."
- **Group id** per decision point (for held-out splitting by game and for optional learning-to-rank).
- Output: rows appended to a `decisions` table in `replays.sqlite` (or a parquet/NPZ cache), keyed by
  `(episode_id, step, seat, option_idx)`.

### Unit (2) — Feature layer  (new stdlib modules; shared offline + in-sim)
All operate on the RAW obs dict so identical code runs on replay obs and live obs (as `features.py` does).
- **`tools/imitation/threat.py`**
  - `our_max_damage_vs_each(obs) -> {(area,index): dmg}` — for **every** opponent Pokémon (Active AND
    bench), the best damage we can deal, accounting for:
    - direct attacks vs the Active, applying **weakness ×2 / resistance −30** (SV rules, Active only).
    - **bench-spread attacks** — attacks that place damage counters on the opponent's bench (e.g.
      Dragapult Phantom Dive = 200 Active + 60 distributed; surfaced as `DAMAGE_COUNTER_ANY`). Spread
      damage ignores weakness/resistance. Track per-bench reachable counters → who becomes an "imminent
      KO" (`bench_target_hp_remaining ≤ our_next_hit`).
  - `opponent_max_damage(obs) -> {(area,index): dmg}` — symmetric, the threat TO us, including:
    - direct damage to our Active (weakness applied).
    - **bench-spread onto our bench** (their Phantom-Dive-likes) — our benched Pokémon are damageable.
    - **Boss/gust reach** — they can drag our benched attacker Active and hit it.
    - **energy redistribution** — a benched opponent Pokémon counts as a threat if energy *movable to it*
      this turn (Energy Switch / abilities / `SWITCH_ENERGY`) would let it attack. Use total movable
      energy across their board vs the attacker's cost, not just energy currently attached to it.
  - `is_lethal_on(obs, target) -> bool`, `opp_lethal_next(obs) -> bool` (incl. bench-spread + gust +
    energy-move lines, so defensive retreat fires against a telegraphed snipe).
  - **Conditional / scaling damage** — many attacks aren't a flat number: Myriad Leaf Shower
    (30 + 30×energy on both Actives), Alakazam Powerful Hand (20×hand), Trevenant Horrifying Revenge
    (+100 if a Hop's was KO'd last turn). A static `best_dmg` is wrong for these. `threat.py` must
    EVALUATE the damage given current state, not read a constant. Capture per-attack:
    `damage_base`, `damage_is_conditional`, `scaling_basis ∈ {per_energy_self, per_energy_both,
    per_hand_card, on_ko_last_turn, per_bench, coinflip, none}`, and a small evaluator that plugs in
    live counts. Where the formula is unparseable, fall back to `best_dmg` and flag it.
  - Reads the rebuilt card KB (effect text, `best_dmg`, scaling fields, abilities) + live energies.
    Mechanics from `docs/policy-metrics.md` + the SV rules reference (§ mechanics).
- **`tools/imitation/deck_tracker.py`**
  - `deck_remaining(obs, my_decklist) -> {card_id: count}` = decklist counts − cards currently visible in
    play/discard/hand. **Computed STATELESSLY each call from current zones, not incrementally** — so
    hand/deck-refresh cards (Iono, Lillie's Determination, Professor's Research, anything that shuffles a
    card back via `TO_DECK`) are handled for free: a card shuffled back leaves hand → it re-counts as
    "remaining," no special-casing. (Deck *order* is never tracked, only counts, so a reshuffle of order
    is irrelevant to us.)
  - `draw_prob(card_id, k=1)` via hypergeometric (`math.comb`); `deckout_clock`; `prize_likely(card_id)`.
  - Opponent side: `opp_seen` set from `obs.logs` + visible board (belief, archetype-prior optional v2).
    Note: a reshuffle resets opponent deck-order belief but not the count belief.
- **`tools/imitation/card_features.py`**
  - `card_static(card_id) -> [type, cardType, energy_cost, best_dmg, dmg_per_energy, hp, stage(0/1/2),
    is_ex, prize_value, retreat_cost, ...]` from `cards_full.csv` (precomputed dict).
  - **Effect-category flags (critical — damage is not the only value):** many attacks/abilities/trainers
    do little/no damage but accelerate, draw, search, gust, heal, or spread. Examples in our own DB:
    Teal Dance / Mountain Stroll / Wrathful Hearth / the Kaguras are **0-damage setup attacks** — a
    damage-only feature rates them worthless. So `card_static` includes parsed effect flags:
    `{accelerate_energy, draw, search, gust, heal, spread_damage, inflict_status, switch, move_energy,
    disrupt_hand, recover_from_discard, setup_attack(0dmg+effect)}`. These are derived offline from
    `all_card_data().attacks` + `all_attack().text` and `Skill.text` (the `attacks` column in
    `cards_full.csv` today carries only `name(dmg/cost)` — **no effect text**, the source of this gap).
  - **Per-card knowledge base (`cards_kb/`) — multi-source, built once offline:**
    1. **Engine text (primary, exact):** `tools/build_card_text.py` runs in docker, dumps every card's
       attack text (`all_attack()`) + **ability text (`CardData.skills[].text`)** joined by exact
       `card_id`. Authoritative (it's what the sim enforces). Covers all 1267. NOTE: the current
       `cards_full.csv` **dropped abilities entirely** — e.g. Teal Mask Ogerpon ex's "Teal Dance" (attach
       Grass + draw) is an Ability and is missing; the rebuild must include `skills`, not just `attacks`.
    2. **Wiki — OPTIONAL enrichment (de-scoped; NOT needed for effects).** VERIFIED 2026-06-25: the
       engine already carries ALL effect text incl. all 191 Trainer effects (Lillie's, Rare Candy) and
       218 Pokémon abilities — see `autoresearch/data/cards_engine.json`. The old `cards_full.csv` looked
       empty only because its extractor skipped `skills`. So the wiki is reserved for `set`/`number`/
       `competitive_role`/rulings only, and can lag without blocking. Entry:
       `bulbapedia.../Browse:Trading_Card_Game` → expansion pages → `{Name}_({Set}_{Number})`.
       Trainer effects carry conditions (Lillie's: draw 6, or 8 if exactly 6 prizes) → these come from the
       engine text and are classified into effect flags + magnitude.
    3. **Classify → `cards_effects.csv`:** `tools/build_effect_flags.py` turns the KB text into the
       structured effect flags above (keyword rules; optional offline LLM pass for ambiguous cards).
       Unmatched → all-zero (safe). Adds a coarse `competitive_role` feature from the wiki data.
  - Only steps 1 + 3 are required for the pipeline; step 2 (wiki/web) is enrichment and can lag. Submission
    ships only the derived `cards_effects.csv` (small), never the scrape or any network call.
  - **card-identity** exposed as raw `card_id` + a coarse `card_category` bucket so the forest can split
    on specific cards where data is dense and on category where sparse.
- **`features.py`** (extend) — assemble the full per-(state,option) vector (~55 dims):
  state(~25) + threat(~6) + tracking(~6) + card_static(~10) + identity(2) + context(1).
  - **State must carry bench damage:** per-Pokémon `damage_counters` for both boards (counters persist on
    the bench, per SV rules) so spread/snipe setups are visible. Today's `features.py` only reads Active
    hp — extend to bench hp/damage on both sides.
  - **State must carry hand quality, not just size:** `hand_dead_count` (cards not playable now — wrong
    energy, no target, supporter-already-used) and `playable_options_count`. This is what lets the policy
    learn "play a hand-refresh card (Iono / Lillie's Determination / Research) when the hand is stuck" vs
    "keep a working hand." `hand_size` alone can't distinguish a dead 6-card hand from a live one.
  - **Option features must be threat-relevant for the special contexts:** for `DAMAGE_COUNTER`/
    `DAMAGE_COUNTER_ANY` options, encode the target's `hp_remaining` and whether placement creates an
    imminent KO; for `SWITCH_ENERGY`/`ATTACH`/`DETACH` energy-move options, encode the
    attack-readiness delta it produces (does moving this energy make a Pokémon able to attack / retreat).
    This is what lets the learned policy place spread damage well and redistribute energy correctly,
    rather than treating those options as opaque indices.

### Unit (3) — Card-policy model  `tools/imitation/train_card_policy.py` (new)
- **Model:** sklearn `RandomForestClassifier` (or `ExtraTrees`) predicting `P(chosen | state, option)`.
  Pointwise ranking: at inference, score each option, pick argmax. Chosen over gradient-boosting purely
  because a forest is N independent `DecisionTree`s → **drops into the existing JSON tree-export + walk
  path** (no new NumPy walker to write/verify under the submission constraint).
- **Training:** pooled winner-filtered rows; class-weight or down-sample negatives (each decision has 1
  positive, many negatives). Long-training budget → tree count + depth + light hyperparameter sweep.
- **Export:** serialize each tree to the repo's existing nested `{feature,threshold,left,right,value}`
  JSON; `card_policy.json` = list of trees + the feature-name order. Inference averages tree outputs.
- **Versioned** via the existing `replays_db.store_policy` mechanism (deck_sig replaced by a single
  global `card_policy` id, since it is deck-agnostic).

### Unit (4) — Gameplay policy  `tools/imitation/gameplay_policy.py` (new) + submission `main.py` shim
- `agent(obs_dict)`: if deck phase → return the 60-card deck. Else:
  1. enumerate `obs.select.option` (the legal actions at this node).
  2. featurize each option via Unit (2).
  3. score each via the loaded `card_policy.json` (pure Python/NumPy walk).
  4. **heuristic gates (hard):** if a lethal/winning option exists → take it; drop illegal/suicidal
     options (would expose `opp_lethal_next` for no gain); never return an out-of-range index.
  5. return argmax (or temperature-sampled) surviving option.
- One file; the submission `main.py` is a thin loader (deck.csv + card_policy.json) calling it — mirrors
  the proven `trev_typh` scaffold. The league `make_pilot` loads the same module.

### Unit (5) — Validation  (reuse existing harness)
- **Decision accuracy:** held-out split by `episode_id`; report top-1 accuracy vs current tree + baseline.
- **League:** `tools/imitation/league.py roundrobin` with the new policy; compare archetype matrix to
  real (especially Dragapult) — same method that found the inverted cell.
- **Ladder:** build submission tarball on the proven scaffold; A/B vs champs over 2–3 submissions.

## 5. Data flow summary

`replays.sqlite` → extractor (winner filter) → `decisions` rows → train (sklearn, offline) →
`card_policy.json` → `gameplay_policy.py` (+ feature layer) → {submission tarball, league pilot} →
validation loop → iterate.

## 6. Error handling & edge cases

- **Card missing from `cards_full.csv`** → `card_static` returns a zero/default vector + category fallback;
  never crash. Log once.
- **Hidden info** (opp hand/deck/prizes) → tracker provides counts/belief; features fall back to 0 where
  truly unknown (mirrors current `opp_hand` handling). Never assume hidden card identities.
- **Sparse cards** → shared model + card-category bucket gives a sane prior; gates guarantee legality.
- **Submission safety** → deck phase always returns a valid 60; option scoring failure → fall back to a
  safe heuristic (e.g. the typh/greedy pick) rather than crash (the ERROR-forfeit risk from v16/v18).
- **Mechanics correctness** → threat/lethal use the SV rules (weakness ×2, resistance −30, KO at dmg≥HP,
  W/R only on Active, special-energy semantics); unit-tested against hand-built boards.

## 7. Testing plan

- `test_threat.py` — hand-built obs → known `our_max_damage_vs_each` / `opp_lethal_next` (incl. weakness).
- `test_deck_tracker.py` — `deck_remaining` + hypergeometric `draw_prob` vs closed-form reference.
- `test_extract_decisions.py` — a fixture replay → expected rows + correct winner-seat labeling.
- `test_gameplay_policy.py` — gate guarantees: never returns an illegal index; always takes a provided
  lethal option; deck phase returns 60. Runs without docker (pure obs dicts).
- Calibration check committed as a script: held-out decision accuracy number.

## 8. Build order (smallest → biggest payoff)

1. `threat.py` (+ test) — unlocks gates + threat features; directly attacks the Dragapult cell.
   A 0-damage attack is not a threat, but the attack-choice value comes from its effect flags (below).
2. `deck_tracker.py` (+ test) — card-counting features.
2b. `build_effect_flags.py` (docker, once) → `cards_effects.csv` — effect-category flags so setup
    attacks / effect cards are valued by what they DO, not by damage.
3. Extend extractor: winner filter + per-option rows + rich features (+ test).
4. `train_card_policy.py` — RandomForest, export to JSON; report held-out decision accuracy.
5. `gameplay_policy.py` + gates (+ test); wire into league `make_pilot`.
6. League recalibration run — Dragapult cell check. If it moves toward 0.32 → validated.
7. Submission tarball on proven scaffold; ladder A/B.

## 9. Open questions / v2 (out of scope for v1)

- Opponent belief beyond seen-cards (archetype-prior decklist matching) — v2.
- Forward search via `search_begin/step` (engine supports it) — the ceiling; v2, reuses the same eval.
- Learning-to-rank (group-wise) instead of pointwise — only if pointwise underfits.
- Sequence/trajectory model (à la Grigsby top-10%) — the real ceiling; far-future, not stdlib-trivial.

## 10. Non-goals

- Not changing deck *composition* (that's the GA work; this is the *pilot*).
- Not a full game engine reimplementation — we use `cg` for truth and replays for learning.
- No new runtime dependencies in the submission.
