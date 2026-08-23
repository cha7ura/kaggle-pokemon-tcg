# Hash-map (state->action lookup) policy: tested, does not work

## Idea
Hash a canonical game state -> the semantic action (option_type, card_id) the expert took.
At inference, replay the expert's move on exact-state matches. Lossless on matches, unlike the
behavioral clone (which compressed state into lossy features -> 46% < baseline). Inspired by the
ARC-AGI-3 agents that built state graphs + hash-dedup of experiences.

## Test (SQL-filtered Alakazam games, held-out)
replay_field JOIN to pull 500 Alakazam games directly (no blob scanning). Canonical state =
(turn, active_id, bench_ids, hand_multiset, opp_active_id, prize_count). Semantic action keyed on
(option_type, resolved_card_id), NOT menu index. Train 399 games (16,890 distinct states), test 100.

## Result: 0.3% held-out coverage. IDEA FAILS.
- Overall: 0.3% of test decisions had their state in the book (3 in 1000).
- Early game (steps 0-15): 2.3% coverage — even the most constrained phase barely repeats.
- When covered: only 60% action-match (expert not even self-consistent on the rare hit).
- (Within-training states "repeat" 33%, but held-out to NEW games it collapses to 2.3% — the
  overfitting gap; measuring held-out is what caught it.)

## Why
Canonical state includes the exact hand (card-id multiset). In a 60-card deck with heavy draw/search,
two games are almost never in the same full state -> states don't recur across games. This is the
SAME wall as the clone, from the opposite side: enough detail to pick the right move makes states
unique; enough abstraction to make them recur discards what distinguishes good moves. That tension
IS the policy problem in this hidden-info game.

## The one valid use (doesn't change ranking)
Transposition table WITHIN a single game's search tree (states recur in one position's neighborhood)
-> search-efficiency, not policy. But search already lost to the heuristic (11-26%), so no lever.

## Where this leaves the policy question (four negatives, one root cause)
- learned clone: 46% (< 50.8% baseline)
- 1-ply / value search: 11-26%
- hand-written deck-aware pilot: 44% (tuned)
- hash-map lookup: 0.3% coverage
All fail because the state space is too large + hidden-info-laden for learning OR lookup to
generalize across games. The heuristic wins by COMPUTING from the current state, not recognizing it.
=> Proven lever remains deck selection + matched heuristic pilots.
