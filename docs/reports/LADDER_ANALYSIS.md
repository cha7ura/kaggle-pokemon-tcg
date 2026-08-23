# Live-ladder analysis — variance, gameplay mining, and the action

## Current standing (competition: pokemon-tcg-ai-battle)
- **Rank 1,338 / 4,083** (top ~33%), team `cha7ura`, score **768.0**, deadline 2026-08-16.
- Top of ladder ~1,205. Median ~680.
- Two active slots (Kaggle scores your 2 most-recent submissions):
  **Dragapult v19 → 768** and **alakazam_cp → 476**.

## (a) The ladder is noise-dominated
The SAME identical agent file scores 200–350 pts apart across resubmits:

| agent | resubmits | min→max | range | stdev |
|---|---|---|---|---|
| alakazam_top | 6 | 687→1006 | 319 | 124 |
| dragapult_v19 | 5 | 699→910 | 211 | 81 |
| trev_typh | 2 | 581→929 | 348 | 174 |

**Implication:** score is a high-variance sample (±~120 pts, 1σ), not a stable rating. The gap from
your 768 to your best-ever 1006 is within one agent's own noise band. Chasing a lucky draw by
resubmitting is low-value; only a real win-rate shift moves the whole distribution.

## (c) Gameplay mining — our 54 recorded ladder games (29-25, 54%)
By our deck (which pilot we ran):
| our deck | record | win-rate |
|---|---|---|
| Dragapult | 8-4 | **67%** (best) |
| Alakazam | 14-11 | 56% |
| Trevenant | 6-10 | **38%** (worst; ran it 16×) |

By opponent matchup:
| vs | record | win-rate |
|---|---|---|
| MegaLucario | 12-5 | 71% (most-faced, 17g — strong) |
| Alakazam | 5-9 | 36% (problem matchup) |
| Trevenant / OTHER / Dragapult | ~even | ~50% |

Cross-tab: the Alakazam problem concentrates in **Trevenant-into-Alakazam (0-3)** and the
**Alakazam mirror (3-8)**.

**Honesty on significance:** at n=54 these are *suggestive not proven* (Trevenant-underperformance
Fisher p=0.145; vs-Alakazam p=0.134). But they're directionally consistent with the offline finding
that Trevenant was our weakest deck, which raises confidence modestly.

## The action (combining a + c)
Don't chase noise with exotic/unproven decks. Put your two best-**mean** proven agents in the two
active slots:
- **Slot 1: Dragapult v19** (mean 826, our best deck at 67% in real games) — already active, keep.
- **Slot 2: alakazam_top** (mean **884**, ceiling 1006) — SWAP IN, replacing alakazam_cp (476).

Expected value of the swap: slot 2 goes from ~476 to ~884 mean. Do NOT ship Trevenant (38% real).
Expect ±120 noise on any single submission regardless — judge over multiple submissions, not one.

## Files
- `ladder_variance.png` — same-agent score variance
- `our_ladder_games.json` — the 54 decoded games with archetypes + outcomes
