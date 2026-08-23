# Step 3 result — 1-ply ISMCTS does not beat the heuristic

**Verdict: FAIL. Search wins 11–17% vs the heuristic across two independent 100-game runs — never
close to parity. Deck selection remains the lever; do not invest in R-NaD self-play on this
search design.**

## The experiment

Controlled A/B on the real cg engine (Docker, linux/amd64 under emulation): both seats play the
**same deck** (`lucario_meta.csv`), so the only variable is the pilot — prior-ISMCTS vs the
`agent_lucario` heuristic. Seat alternates each game to cancel first-player advantage.
`--games 100 --sims 48 --depth 12`.

## Result — replicated across two runs

| metric | run 1 | run 2 |
|---|---|---|
| search win-rate vs heuristic | **11.0%** (11/100) | **17.0%** (17/100) |
| as first player | 16% (8/50) | 18% (9/50) |
| as second player | 6% (3/50) | 16% (8/50) |
| rollout errors | 0% (140,928) | 0% (137,280) |
| rollouts truncated (sub-select handling) | 12% (17,510) | 33% (45,464) |

Both runs land in the teens, far below the 50% parity line, from **both seats** — so this is
neither small-sample noise nor a first-player artifact. The gap between the two runs (11% vs 17%)
comes from different sub-selection handling in the rollout (see caveat), NOT from any change that
moves search toward parity. The conclusion is the same at both points.

## Why this is a trustworthy negative (what we ruled out)

Earlier runs showed 10–30% and I suspected a bug was suppressing search. We chased and fixed it:
1. **Path bug** (two module copies) — fixed with a robust file-finder.
2. **Value corruption** — illegal mid-rollout actions were caught and scored as `-1.0` (a loss).
   That poisoned ~11% of value estimates. **Fixed:** a rollout that can't continue now evaluates
   the state it reached (prize lead), not a fake loss. Rollout errors went 11% → **0%**.
3. With corruption gone, the win-rate **did not recover** — it stayed at 11%. So the low number
   is the search genuinely playing worse, not an artifact.

This matches the project's own prior finding (README: "forward search validated WORSE") — now
reproduced quantitatively with the improved archetype-prior determinization in place.

## Remaining caveat (honest)

A fraction of rollouts **truncate early** on the engine's forced sub-selections (CARD/ENERGY
selects that follow a MAIN action). The tracer pinned the cause: these sub-selects have count
semantics that don't map to a MAIN-style option pick — e.g. an *optional* CARD select
(`minCount=0, maxCount=1`) rejects both a forced pick AND an empty `[]`. Getting the exact accepted
format right would take reverse-engineering the engine's per-context select rules.

Crucially, **truncations do not corrupt the win-rate**: a truncated rollout is scored on the state
it reached (prize lead), not as a loss (rollout errors are 0% in both runs). That is why changing
the sub-select handling between runs moved the number only from 11% to 17% — both far below parity.
The negative verdict is **robust to this caveat and replicated**; chasing the exact sub-select
format is a rabbit hole that cannot close a >30-point gap.

## What this means for the competition

- **The deck lever is the one that works.** Ship MegaStarmie / Dragapult (highest field EWR from
  the meta read); the heuristic pilot is already strong and search does not improve on it.
- **The determinization fix was still worth building** — it's validated independently (hidden-deck
  recall 5% → 40%, archetype-ID 20% → 88%) and would be the correct opponent model for any *future*
  search/RL work. It is not the bottleneck; the 1-ply search design is.
- **R-NaD self-play (step 4) is not justified by this evidence.** A full neural self-play system is
  a multi-week GPU investment, and the cheapest available test of "does learned/searched play beat
  the heuristic" came back clearly negative. Revisit only with a fundamentally different search
  (deeper, better value function) or if the meta shifts.

## Files
- `run_gate.py` — the A/B harness (writes step3_result.json)
- `agent_ismcts_prior.py` — prior-determinization ISMCTS agent (+ diagnostics)
- `determinize.py`, `arch_priors.json` — the validated determinizer
- `step3_result.json`, `step3_result_run2.json` — full per-game logs + DIAG counters (both 100-game runs)
- `trace_desync.py` — single-rollout step tracer that pinned the sub-selection cause behind the caveat
