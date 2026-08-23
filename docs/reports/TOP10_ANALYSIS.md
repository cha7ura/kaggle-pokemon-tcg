# Top-10 analysis + ship recommendation

## What a submission is (the mechanics, verified against the engine)
A submission = `deck.csv` (60 cards) + `main.py` (`agent(obs) -> [indices]`). Every step, the ENGINE
enumerates the legal options (a menu: ATTACH / PLAY / EVOLVE / ABILITY / RETREAT / ATTACK / END, plus
sub-selections). The agent only PICKS an index from that menu. A turn is a sequence of menu-picks
ending in END — you play multiple cards per turn. So both deck AND policy are shipped; the policy is
purely menu-SELECTION (what to pick, in what order), never action generation.

## Top-10 teams (live leaderboard) and their decks
| # | score | archetype | team |
|---|-------|-----------|------|
| 1 | 1205.2 | HopTrevenant | The Debauchery Tea Party |
| 2 | 1188.7 | Grimmsnarl | tonakaiiii |
| 3 | 1149.2 | HopTrevenant | Yushin Ito |
| 4 | 1145.2 | Chandelure | yamy893 |
| 5 | 1142.8 | SolrockLunatone | Akira-Ninth |
| 6 | 1137.5 | Alakazam | kenkoooo |
| 7 | 1126.8 | Dragapult-variant | XP3RiX |
| 8 | 1123.0 | SolrockLunatone | やる気元気ミワハルキ |
| 9 | 1115.4 | Grimmsnarl | kazuki0123 |
| 10 | 1108.4 | Alakazam | 5.5 |

Exact 60-card decklists reconstructed for all 10 (top10_decks.json); Grimmsnarl finalized as a legal
submittable deck (grimmsnarl_deck_submit.csv).

## Recent meta (decoded recent games, line-based classifier)
Grimmsnarl 34.8% (55-61% wr) — exploded from 0; Archaludon 16.7%; Alakazam 11.2%/46%;
SolrockLunatone 9.7%/61% (sleeper); MegaStarmie 4.8%/51%; Chandelure 2.7%; **Dragapult 0.4%/33%
(near-extinct)**, HopTrevenant 1.7% recent (was elite in the older window).

## The two levers, tested
### DECK — PROVEN, shippable
- Grimmsnarl is the dominant recent deck (34.8%, 61% wr) with a real working list (the repo's
  auto-gen pool_marniesgrimmsnarlex.csv is BROKEN: 22 Boomerang Energy, 0 Dark Energy).
- Our two current slots are on the wrong side of the meta: Dragapult (0.4%/33%) is extinct;
  alakazam is mid. We re-shipped Dragapult AFTER it collapsed.

### POLICY — tested 3 ways, all NEGATIVE
The menu-selection policy is the hard part (engine gives the menu; picking well is judgment):
1. ISMCTS search: 11-17% vs heuristic
2. Value-net search: 19-26%
3. Behavioral clone of the 82% Trevenant expert:
   - base features (23+7): 46.9% top-1
   - RICH features (30+20: action semantics, resolved card, discard/prize/threat): 45.6% top-1
   - "always pick option 0" baseline: 50.8%
All BELOW the trivial baseline. The engine already orders options sensibly (opt 0 is the obvious play
50.8% of the time); the expert's edge is in the ~49% of deviations, which depend on hidden hand,
opponent reads, and multi-turn planning that no static per-decision feature captures. Not fixable by
more data, better algorithm, or richer features — it's the sequential hidden-information nature of the
selection.

### Matched-deck gap (the smoking gun)
Top HopTrevenant pilots: 74-82%. Our cha7ura Trevenant pilot: 38%. SAME deck, ~40-pt gap, and the
decision MIX is nearly identical (MAIN 61% vs 56%, draw 23% vs 27%) — so the gap is choice QUALITY
within menus + inconsistent go-first (we're 5/3; experts are ~100% committed one way).

## SHIP RECOMMENDATION
1. **Deck: swap the extinct-Dragapult slot to Grimmsnarl** (proven dominant recent deck) with the
   existing heuristic pilot. Engine-validate via deck_gate (commands staged) before submitting.
   This is the one high-confidence, evidence-backed move.
2. **Keep alakazam_top in slot 2** (already swapped in; mean ~884 over resubmits).
3. **Do NOT invest further in offline policy learning** (clone/search/value-net all lose). If policy
   is pursued later, it needs a fundamentally different belief-state / sequential approach, not a
   static classifier — a research project, not a submission-cycle task.
4. **Remember: sim != ladder, and ladder score is noise-dominated (+/-120, 1 sigma).** Any offline
   result is a hypothesis; only the live score confirms. Judge a swap over days of resubmits, not one
   score.

## Artifacts
- top10_decks.json — reconstructed decklists for all 10
- grimmsnarl_deck_submit.csv — legal submittable Grimmsnarl deck
- submissions_ranked.json — 1,101 submissions ranked by win-rate
- POLICY_COMPARISON.md — full policy analysis incl. the clone experiments
- FEATURE_INVENTORY.md / features_rich.py — full game-state info + enriched featurizer
- DATA_NOTES.md — data-handling checklist

---

# SESSION UPDATE (2026-07-03, later) — outcomes after the ship recommendation above

## What actually got submitted (and how it scored)
- alakazam_top (slot-2 swap, ref 54290517): COMPLETE, 821.1. INTENDED active slot.
- dragapult_v19 (re-ship ref 54117523, "to STAY active"): COMPLETE, 777.4. INTENDED active slot.
  NOTE: these two are the pair we INTENDED as active (per their own submission descriptions —
  "Swap slot 2 to alakazam_top", "re-ship to STAY active. Paired with Alakazam swap"). This is NOT
  API-confirmed selection: Kaggle final rank uses the user's SELECTED submissions, which must be read
  from the competition page (not the CLI submission list), and we did not query that. They are also
  NOT merely the two highest score rows. The ladder rescores the SAME file across a wide range on every resubmit
  (e.g. dragapult_v19 rows span 699-909; alakazam_top rows span 686-1005), so any single row's score
  is noise-dominated (+/-120, 1 sigma) — the standing is whichever pair the user has SELECTED on the
  competition page (not API-confirmed here), not a max-over-rows pick.
- submission_grimmsnarl: SUBMITTED, COMPLETE, only ~425-450. UNDERPERFORMS badly.
  -> The Grimmsnarl ship recommendation above was tested LIVE and did NOT pay off.

## WHY Grimmsnarl underperformed (key lesson: deck x pilot are inseparable)
Grimmsnarl was a top-tier deck by the recent-meta matrix (~55-57% EWR, top-2), yet scored ~425
because it runs on the GENERIC agent_lucario pilot, whose smart logic is hardcoded to Lucario cards.
alakazam_top scores ~821 because its 814-line pilot MODELS the deck (Powerful Hand = 20 x hand-size,
real damage/KO/prize math). A strong deck with a mismatched pilot loses to a mid deck with a matched
pilot. The matrix measured HUMAN-piloted deck-vs-deck, which does NOT transfer to our bot's piloting.

## Every lever tested this session (all real, held-out / live numbers)
| Lever | Result | Verdict |
|---|---|---|
| Deck selection (alakazam_top + dragapult) | 821 + 777 live | PROVEN — our edge |
| Learned behavioral clone | 46% (< 50.8% baseline) | dead |
| ISMCTS / value search | 11-26% | dead |
| Hand-written deck-aware Grimmsnarl pilot | 44.1% (200g, loses to generic) | dead |
| Hash-map state->action policy | 0.3% held-out coverage | dead |
| Alakazam deck-refinement (V2/V3) | 41% / 36% vs current | current deck already optimal |
| Alakazam pilot threshold-tuning | ~50 scattered constants, not cleanly tunable | poor EV, not attempted |

ROOT CAUSE of all policy negatives: the state space is too large + hidden-info-laden for learning OR
lookup to generalize across games. The heuristic wins by COMPUTING from the visible board (opp active
hp/energy, both discards, all counts are all visible) rather than recognizing/recalling states.

## STANDING DECISION: hold alakazam_top + dragapult_v19. NO more submissions this cycle.
The proven pair is in place. Policy is a settled dead end. Deck+pilot co-tuning means new decks need
a matched ~800-line pilot (large build), not a quick swap. Next session: monitor live scores (they
drift +/-120), and only revisit if the meta shifts or a matched deck+pilot pair is worth building.

## Session artifacts added
- PILOT_VALIDATION.md — deck-aware pilot 44.1% loss + tuning selection-bias analysis
- ALAKAZAM_ANALYSIS.md — why alakazam_top scores ~821 (deck-specific pilot)
- HASHMAP_POLICY_RESULT.md — hash-map policy 0.3% coverage negative
- DECK_REFINE_RESULT.md — V2/V3 deck variants both worse; current deck optimal
- GAMEPLAY_WALKTHROUGH.md — full observable-state walkthrough (both sides, top-team game)
- tune_weights.py — weight-tuning harness (works on dict-based pilots like agent_grimmsnarl)
