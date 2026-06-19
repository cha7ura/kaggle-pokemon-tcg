# Deepen the Lucario Policy (Real-Meta Benchmarked) — Design Spec

**Date:** 2026-06-19
**Project:** PTCG AI Battle Challenge (Simulation) — `~/Documents/Github/kaggle/pokemon`
**Status:** Approved design, pending implementation plan

## Context

Our champion agent (`autoresearch/agent_lucario.py`, "Lucario v4") scores **820.7** on
the live ladder — a strong climb from the 508 baseline, but mid-tier: the actual
leaderboard top is **~1300**.

Fresh research (2026-06-18/19) reframed the problem:
- The 2nd-place player's 15,000-game matchup study shows **Mega Lucario ex is the #1
  deck (60.4%)**, Dragapult ex #2 (55.6%), with the **Lucario mirror ≈ 50/50**. Crustle
  (which we spent a session fighting) is *not* in the top tier — it was a day-1 spike.
- Therefore: **we are already on the best deck.** The entire gap from 820 → ~1300 is
  **policy depth**, decided largely in the Lucario mirror and the Dragapult matchup.
- A mirror/Dragapult benchmark is now **meta-faithful** (the ladder really is these
  decks), unlike the misleading Crustle bot and pure self-mirror earlier.

## Goal & Success Criteria

Raise the Lucario agent's ladder rating from ~820 toward the top tier by deepening
the policy in the matchups that actually occur.

**Success:**
- A new agent that **holds the mirror (≥50% vs the frozen v4 champion)** and
  **improves or holds vs the Dragapult benchmark**, over N≥400 seat-swapped games.
- **Primary success = a higher real ladder score** than 820.7 after submission.
- Each accepted change is an isolated, logged diff with a measured delta.

## Architecture

Three pieces, reusing the existing `autoresearch/` harness:

### 1. Benchmark harness (build first)
- **Dragapult opponent:** pull the host's Dragapult sample
  (`kiyotah/a-sample-rule-based-agent-dragapult-ex-deck`) → `decks/dragapult.csv`
  + `autoresearch/dragapult_agent.py` (a faithful pilot for that deck).
- **Mirror benchmark:** freeze current v4 as `autoresearch/champion_lucario.py`. Every
  policy change is the *challenger* vs this frozen champion on `decks/lucario_meta.csv`.
- Both run through the existing `eval.py` (seat-swap + Wilson lower bound). The mirror
  is ~50% of the meta; the Dragapult matchup is the other top-tier matchup.

### 2. Policy improvements (one isolated diff each, to `agent_lucario.py`)
Prioritized by expected mirror/Dragapult impact:
1. **Prize-trade math** *(highest priority)* — Mega Lucario ex gives up **3** prizes on
   KO. When we are losing or even in the prize race, prefer the cheaper attacker /
   Hariyama instead of exposing the mega-ex. Generalizes LB960's "penalize the big ex
   at 2–3 prizes remaining" into a prize-differential rule.
2. **Attack sequencing** — choose Mega Brave (270 / 2 energy) vs the 130 / 1-energy
   attack by what actually KOs the target and the current prize state, not a fixed
   priority.
3. **Setup / mulligan quality** — bench the correct basics, reduce brick/whiff turns.
4. **Boss targeting for tempo** — drag the opponent's developing attacker to win the
   prize race (mirror + Dragapult both reward tempo).
5. **Energy / retreat discipline** — stop over-committing energy onto a single ex;
   retreat only when the tempo cost is justified.

Knobs stay in the tunable `W` table where possible so changes are measurable.

### 3. Validation gate + iteration loop
- Implement ONE improvement → run mirror + Dragapult gauntlets (N≥400) → **keep iff it
  holds the mirror AND improves/holds Dragapult** → else revert.
- Append every result (kept or reverted) to `autoresearch/log/experiments.md`.
- Submit periodically (5/day; latest 2 tracked). **The real ladder score is the final
  arbiter** — local benchmarks inform, the ladder decides.

## Data Flow

`agent_lucario.py` (challenger) ⟶ `eval.py` gauntlet vs `champion_lucario.py` (mirror)
and vs `dragapult_agent.py` (Dragapult) ⟶ JSON verdict (score + Wilson LB) ⟶
keep/revert decision ⟶ periodic `kaggle submit` ⟶ ladder score read back.

## Testing / Validation

- Each change measured vs **two** opponents (mirror + Dragapult), seat-swapped, N≥400.
- Crash-safety preserved (the agent must survive the validation mirror — any exception
  forfeits). Self-play smoke (40 games, 0 errors) before every submission.
- Final validation is the ladder score, not the local number.

## Risks & Mitigations

- **Mirror overfitting** — mitigated: the ladder genuinely is Lucario mirrors now
  (ISAKA's data), so mirror performance is ladder-predictive. Dragapult benchmark guards
  against tunnel-visioning the mirror.
- **Noisy 50/50 mirror verdicts** — use N≥400–600 + Wilson LB; only promote on a clear
  signal.
- **Slow ladder feedback** (hours/submission) — iterate locally, submit in batches,
  read scores asynchronously.
- **Dragapult pilot fidelity** — the sample agent may pilot Dragapult sub-optimally;
  treat the Dragapult number as directional, the mirror as primary.

## Out of Scope (YAGNI)

- No deck switch — Lucario is the #1 deck.
- No MCTS / learned value net (direction C) and no behavioral cloning from replays
  (direction B) in this iteration — those are separate, higher-ceiling follow-ups.
- No Crustle-specific work — it is not the current meta.
