# Imitation League — Design Spec

Date: 2026-06-23
Status: Approved (design), pending implementation plan

## Problem

Our offline oracle ranks candidate decks against the real 157-deck ladder field, but every
opponent is piloted by the weak generic `agent_typh.py`. This makes the oracle **blind to the
matchups that decide the ladder** — most importantly Trevenant, where the oracle says ~0.80 while
real ladder data says 0.36. We cannot trust any absolute matchup number, only aggregate ranking.

The root cause is opponent *pilot* quality, not deck choice. The fix: pilot each opponent deck with
a **policy learned from the real top players' games**, so the field plays like real humans. This
turns the oracle from "rough deck ranker" into a realistic ladder predictor and, downstream, gives
us a path to imitate the best pilots (the long-standing pilot-ceiling wall).

## Data available

- ~2600 replay JSONs in `json/` (top-Elo ladder games), growing via cookie-free
  `tools/fetch_dataset.py`.
- Each game: `info.TeamNames[seat]` (player identity), `rewards` (W/L), and `steps[]` where each
  step is `[seat0, seat1]`, each seat has `observation.current` (full game state),
  `observation.select` (decision menu: `context`, `option[]`, `minCount`, `maxCount`), and the
  chosen `action` (list of option indices).
- ~301 multi-option (learnable) decisions per game; dominated by context 0 (main action menu) and
  context 7. Forced (1-option) steps are ignored.
- Data census (games / decisions) confirms heavy hitters: The Debauchery Tea Party 351/42628,
  Kadoraba 247/34282, foo_foo 241/36849, keidroid 148/14300, TrustHub hiroingk 143/15919. A long
  tail of ~250 teams has 1–3 games (unlearnable).

## Scope decisions (locked)

- **Coverage:** learn a policy for every deck with **≥30 games**; decks below the floor fall back to
  the generic typh pilot so they still appear in the league. Widest field, mixed fidelity.
- **Granularity:** per individual top player (their deck + their decisions), not pooled by archetype.
- **Primary goal:** upgrade the oracle (realistic imitation field). Ranking deck+policy pairs is a
  downstream bonus, not the gate.
- **Delivery:** staged — insight (accuracy) first, then pilot + league.

## Architecture

Pipeline: replays → decisions table → per-deck policy → pure-Python pilot → league/oracle.

All new code under `tools/imitation/` except the in-sim pilot (`autoresearch/agent_imitation.py`).

### Component 1 — Decision extractor (`tools/imitation/extract_decisions.py`)
Per game: resolve each seat's TeamName and 60-card deck (the 60-int action at deck phase), and the
seat reward. Walk steps; for each seat decision with `len(option) >= 2`, emit a record:
`{deck_sig, player, context, state_features, option_features[], chosen_idx, reward}`.
Group output per deck signature (canonical `tuple(sorted(deck))`, same key as `extract_field.py`).
Output: `tools/imitation/data/<deck_sig>.jsonl` (or a single sharded file) plus a manifest
mapping deck_sig → {player, games, decisions, archetype}.

### Component 2 — Featurizer (`tools/imitation/features.py`) — LOAD-BEARING
Pure-Python (stdlib lists, no numpy in the inference path). Two functions, imported by BOTH the
offline trainer and the in-sim pilot via an identical code path:
- `state_features(current, seat) -> list[float]`: my/opp active HP and damage, my energy count,
  bench count, hand size, prizes mine/opp, turn number, and similar fixed scalars derived only from
  `observation.current`.
- `option_features(option, context) -> list[float]`: option `type`, target `area`/`index`,
  card-id bucket, is-attack / is-ability / is-energy flags, and any cheap derived signal.
Feature order is frozen in this module and shared; training and inference must call the same code.

### Component 3 — Trainer (`tools/imitation/train.py`)
Per deck ≥ floor: build a training set where each decision contributes one positive example (the
chosen option's `state⊕option` vector, label 1) and the other options as negatives (label 0). Train
a sklearn decision tree (or small forest) as an option scorer; at inference pick `argmax` of the
score over the legal options. Export the trained tree to a pure-Python form — serialized arrays
(`feature`, `threshold`, `children_left`, `children_right`, `value`) in JSON the agent walks with
stdlib, no sklearn at runtime. **Report held-out top-1 accuracy per context** (predicted argmax ==
real chosen index) against the baseline `mean(1 / n_options)`. This report is the Stage-1 gate.

### Component 4 — Imitation pilot (`autoresearch/agent_imitation.py`)
Generic stdlib agent. Loads `deck.csv` and a policy JSON (path via env, then `deck.csv`-adjacent,
then `/kaggle_simulations/agent/`). On each obs: if `select is None` (checked on the RAW dict before
any conversion) return the 60-card deck; else featurize each legal option, score via the loaded tree
arrays, return the argmax indices respecting `minCount`/`maxCount`. Wrap the in-play path in
try/except returning a legal fallback `list(range(min(minCount, len(option))))`. NO `__file__`
anywhere (Kaggle runs the agent via `exec()`). Decks without a learned policy use the typh fallback.

### Component 5 — League (`tools/imitation/league.py`)
Assemble the (deck + policy) roster. Run each roster pilot vs our candidate decks and (optionally)
vs each other over N randomized-shuffle games via the existing docker `eval.py` harness, reusing the
field weights from `extract_field.py`. Produce a field-weighted score per candidate. This becomes
the oracle's opponent field, replacing typh.

## Staged delivery & gate

- **Stage 1:** Components 1–3 + accuracy report. **GATE:** proceed only if trees predict real actions
  meaningfully above the n-options baseline (target: top-1 accuracy clearly > `mean(1/n_options)` on
  the dominant contexts). If imitation is near-random, stop; salvage findings as strategy insight and
  do not build the pilot.
- **Stage 2:** Components 4–5. Imitation pilot, league, oracle swap. Validate the upgraded oracle by
  re-checking it ranks the four known-ladder decks correctly AND that its Alakazam-vs-Trevenant cell
  moves toward the real 0.36 (the whole point).

## Risks

1. **Featurizer parity (highest):** state features must be computed identically offline and in-sim.
   Mitigation: one module, no environment branching; a parity self-check that runs the featurizer on
   a saved replay obs and on the live obs of the same state and asserts equal vectors.
2. **Context-0 fidelity:** main-menu options are abstract (`{"type": 1}`); if they don't featurize
   into something predictive, that context falls back to a heuristic rather than a near-random tree.
   The per-context accuracy report makes this visible before it pollutes the league.
3. **Docker load at league time:** ~15–20 policies × round-robin × N games is heavy. Accepted; the
   league is run deliberately, not per-iteration.
4. **Data freshness for thin top players** (e.g. Jaga #2 has 1 game): they fall below the floor and
   use typh fallback until more replays are fetched. Acceptable.

## Out of scope

- Shipping an imitation pilot to the live ladder. That is a separate decision made only after the
  league shows a deck+policy beating the current live agent on the upgraded oracle.
- Reinforcement-learning / self-play improvement on top of imitation (future, not this spec).
