# Mega Lucario ex + agent_lb950: VALIDATED new deck+policy pair (SHIP)

## Test (pilot_gate, real cg engine, 100 games, same deck both seats)
    agent_lb950 (matched 731-line Lucario pilot) vs agent_lucario (generic)  on lb950.csv

## Result: agent_lb950 80.0% (80/100 decisive). err 0. seat0 43/50, seat1 37/50. CLEAR WIN.
First pilot this session to beat generic — and by a wide margin (80/20), seat-balanced, zero crashes.

## Why this matters (the winning formula, confirmed twice)
- Deck: Mega Lucario ex ("SolrockLunatone" in classifier) = #2 recent EWR (56.7%).
- Pilot: agent_lb950 = 731-line, Lucario-specific (Makuhita/Hariyama/Lunatone/Solrock/Riolu/
  Mega Lucario ex, 17 card constants) — MODELS the deck.
- This is the SAME pattern as alakazam_top (deck-specific pilot -> 821), the INVERSE of Grimmsnarl
  (strong deck + GENERIC pilot -> 425). Matched pilot is the difference.
- Old-meta live proof: submission_lb950 scored 733.5 once.

## Contrast with every other pilot tested this session
| Pilot+deck | pilot_gate WR vs generic | verdict |
|---|---|---|
| agent_lb950 + Mega Lucario | 80.0% | SHIP |
| agent_grimmsnarl + Grimmsnarl | 44.1% | drop |
| (learned clone / search / hashmap) | 46% / 11-26% / 0.3% | dead |

## Decision: SHIP submission_lb950 (bundle already exists: cg/ + deck.csv + matched main.py).
Today's quota fresh (0 used UTC 2026-07-04). Evidence-backed, gate-cleared, err 0.
