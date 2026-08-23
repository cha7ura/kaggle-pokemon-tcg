# Alakazam deck-refinement A/B: current deck is already optimal, don't touch it

## Test (deck_gate, 200 games each, same pilot both seats, seat-alternated)
Candidate variants vs the shipped alakazam_top deck (live score 826.8). Cards chosen were ones the
pilot ALREADY has logic for (Fezandipiti ex 140, Shaymin 343), swapped for marginal 1-ofs.

| Variant | Change | WR vs current | Verdict |
|---|---|---|---|
| V2_Fezandipiti  | +Fezandipiti ex, -Sacred Ash              | 41.0% (82/200) | worse |
| V3_Fez_Shaymin  | +Fezandipiti ex, +Shaymin, -Sacred Ash, -Lana's Aid | 36.0% (72/200) | worse |

Both err 0, both seats consistent. CURRENT DECK WINS decisively both times.

## What it tells us (informative, not just negative)
1. V3 < V2: stacking more "pilot-supported" cards made it WORSE -> more is not better.
2. The CUTS hurt more than the ADDS helped. Sacred Ash (recovery) + Lana's Aid are load-bearing
   for this pilot; the 1-ofs are NOT filler, they are tuned.
3. Deck + pilot are CO-TUNED as a pair. Field-frequency-mined swaps don't transfer because the
   pilot's logic depends on the exact current list. The shipped 60 is a local optimum for this pilot.

## Decision: KEEP the shipped alakazam_top deck unchanged. Deck-refinement lever exhausted.
Next (per user's "both, sequenced"): threshold-tuning of the pilot itself, with a regression guard
(before/after deck_gate on the SAME deck, only pilot thresholds differ).
