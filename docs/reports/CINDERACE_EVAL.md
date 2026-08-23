# Cinderace/Archaludon deck evaluation — verdict: do NOT ship (it's a field-spread effect)

## What we tested
The replay corpus showed a "Cinderace" cohort at **63.6% win-rate (911 games)** — the highest of any
archetype. On inspection the real deck is **Metal/Archaludon ex** (Duraludon → Archaludon ex, 300 HP,
220 dmg; Cinderace is a Fire tech splash). It's broad-based (873 of 911 games are one decklist, 27
pilots at 64%) and the upstream archetype map had buried it inside the "MegaStarmie" bucket.

We put it head-to-head against the two currently-shipped decks on the real cg engine, **same heuristic
pilot on both seats** (deck is the only variable), seat alternating, 100 games each.

## Result — a split, not uniform strength

| matchup | Archaludon win-rate | verdict |
|---|---|---|
| vs **Dragapult** (shipped) | **65.0%** (65/100) | dominates |
| vs **MegaStarmie** (shipped) | **43.0%** (43/100) | loses |

No seat artifact (Dragapult: 33/32 by seat; MegaStarmie: 22/21). Zero engine errors.

## What this explains

The 63.6% replay-field number was **real but a matchup-spread effect, not raw deck strength.**
Archaludon farms decks like Dragapult (+15) but loses the games that matter most for us — it is
**below even against MegaStarmie, the exact deck we already ship.** Adding Archaludon to the field
would not beat our own best deck; it would just be another deck MegaStarmie beats.

This is the same lesson as the Bellibolt/Lightning-counter trap from the meta read: a high win-rate
against *the field* does not survive once you condition on *the decks you actually face*. The field
EWR ranking (MegaStarmie 57%, Dragapult 55.7%) already priced this in; MegaStarmie remains the pick.

## Caveat
The heuristic pilot was tuned on Lucario, so it may underplay Metal/Archaludon — the 43% could be a
few points low from piloting. But the gap to MegaStarmie is 7 points below even, and the deck would
need to *beat* our shipped deck to justify a slot, not merely be close. The verdict is robust to a
modest piloting handicap.

## Bottom line
**Keep the shipped MegaStarmie + Dragapult.** Archaludon is a legitimate deck (beats Dragapult
heavily) but is a losing matchup into MegaStarmie, so it does not improve our slots. Its value is
diagnostic: it confirms MegaStarmie's strength (it even beats the field's highest-raw-WR deck).

## Files
- `deck_gate.py` — deck-vs-deck harness (same pilot both seats, seat-alternating)
- `autoresearch/decks/cinderace_archaludon.csv` — the most-played pure list (873g/64%)
- `cinderace_eval.png`, this writeup
