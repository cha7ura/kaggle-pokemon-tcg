# Design: Diverse-Panel Eval + Deck/Policy A/B (score-driven)

Date: 2026-06-20
Status: Approved (brainstorming) — pending user spec review
Goal: raise the Kaggle ladder score above v5 (664) by fixing the measurement problem first,
then optimizing the deck and generalizing the policy.

## The problem this solves

We have no offline signal that predicts the ladder. The mirror gate is actively **misleading**:
it promoted v4 over v2, but the real ladder ranked them **opposite** (v2=566 > v4=554). The net/RL
track is a confirmed dead end for *score* (plateaus ~0.3 vs the heuristic). So the lever is the
**deck** (least-explored, flagged early as `deck > search > RL`) and **policy generalization**
(v4's Crustle over-fit hurt the diverse ladder) — but neither is optimizable without a trustworthy
gate. Step 0 is therefore a **better measuring stick**, validated against known ladder outcomes.

## Authoritative facts (from kaggle CLI)

Real ladder scores: v1=441.3, v2=565.8, v4=554.6, **v5=664.4 (best, live)**. Same agent (v5)
re-submitted scored 600.0 → ladder variance ≈ ±64. These four labeled points are the validation
set for the panel metric.

## Non-goals

- No neural net / RL (banked as research, wiki/18 — it does not improve score).
- No mirror-only tuning (proven misleading).
- No change to the live submission until a candidate beats v5 on the **ladder** (averaged).

## Architecture

### Unit 1 — `autoresearch/panel_eval.py` (the new gate)

A diverse-opponent gauntlet. Unlike `eval.py`'s same-deck mirror, each opponent plays its **own
deck** (realistic diverse field).

- **Panel** (agent file + deck): `random` and `greedy` (baselines.py), `dragapult_agent` +
  `decks/dragapult.csv`, `crustle_agent` + `decks/crustle.csv`, frozen `champion_lucario` (v4) +
  `decks/lucario_meta.csv`.
- For a candidate (agent file + deck): play a **seat-swapped** gauntlet of N games vs **each**
  panel opponent; record per-opponent wins/draws/losses, win-rate, Wilson LB.
- **Aggregate** = mean win-rate over the *meaningful* opponents (`greedy`, `dragapult`, v4-mirror).
  `random` reported as a floor sanity-check (~1.0 expected). `crustle` reported **separately** as a
  rare stall outlier (not in the aggregate — it distorts toward a matchup the ladder rarely shows).
- Reuses `eval.py`'s `play_one` and `wilson_lb` (import, do not duplicate). Output: JSON with
  per-opponent rows + `panel_score` (the aggregate) + `crustle` + `random_floor`.
- CLI: `panel_eval.py --agent <file> --deck <csv> --games <N>`.

### Unit 2 — Validate the proxy (the critical gate-trust step)

Run `panel_eval.py` on v1/v2/v4/v5 policies (each with its deck) and check the `panel_score`
**ranks them consistently with the ladder** — specifically: v5 highest, and **v2 ≥ v4** (the case
the mirror got wrong). Choose the aggregate weighting that best reproduces the ladder order
(Spearman rank vs [441,566,554,664]; require at least v5-top and v2≥v4).

- **If it matches** → the panel is a trusted gate; proceed to Units 3–4.
- **If it cannot reproduce v2≥v4 after reasonable weighting** → do NOT trust offline; fall back to
  pure-ladder A/B (submit + average), and record that the panel failed.

This step is the integration test. The agent files v1/v2/v4 are reconstructable from the submission
tarballs / git history; v5 = current `agent_lucario.py`.

### Unit 3 — Deck A/B

Generate deck variants of `decks/lucario_meta.csv` (card-count ratios, single-card swaps within the
legal pool from `data/EN_Card_Data.csv`). Score each with the **v5 policy** via `panel_eval`. Keep
variants whose `panel_score` beats the v5 baseline by more than panel noise. Log every variant
(kept + rejected) to `experiments.md`.

### Unit 4 — Policy generalization A/B

Test **simplifying** the Crustle-specific machinery (which correlates with v4's ladder regression)
and adding general-play tech, scored on the panel with a fixed deck. Same keep/reject discipline.

### Unit 5 — Ladder confirmation

The top 1–2 candidates (deck and/or policy) from the panel → build submission tarball → submit to
Kaggle. **Average 2–3 submissions** of a candidate to see past the ±64 ladder noise before
declaring a winner. Promote to live only if it clears v5 (664) on the averaged ladder score.

## Data flow

```
candidate (agent+deck) ──> panel_eval (vs random/greedy/dragapult/crustle/v4, own decks, seat-swapped)
                              └─> per-opponent WR + panel_score (aggregate)
validate: panel_score(v1,v2,v4,v5) ~ranks~ ladder(441,566,554,664)?  yes -> trust ; no -> pure-ladder
trusted gate ──> deck A/B + policy-generalization A/B ──> top 1-2 ──> ladder (avg 2-3) ──> keep if > 664
```

## Testing

- `panel_eval.py`: candidate=v5 vs `random` → win-rate ~1.0 (sanity floor); JSON has all panel rows
  + aggregate; runs in the Docker engine container.
- Validation (Unit 2) is the integration test: panel ranking vs the 4 known ladder scores.
- Deck/policy A/B: each variant logged to `experiments.md` (kept + rejected).
- Ladder: averaged multi-submission, recorded with per-submission scores (noise visible).

## Risks / open items

- **The panel may not predict the ladder** (real opponents unknown). Unit 2 surfaces this early; if
  it fails, fall back to pure-ladder A/B. This is the make-or-break assumption and it's tested first.
- Ladder variance ±64 → single submissions are unreliable; averaging is mandatory (Unit 5).
- Deck legality: variants must stay legal (60 cards, ACE-SPEC ≤1, etc.) — validate against
  `data/EN_Card_Data.csv` rules before scoring.
- `eval.py` must stay untouched (the fixed mirror metric remains available for reference).
