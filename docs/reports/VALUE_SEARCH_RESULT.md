# Value-guided search — result: loses to the heuristic, and better components don't help

## What ran
Colab, native x86 engine. Value-net 1-ply search (scores each candidate move's resulting state with
the learned win-prob net `value_trees.json`, AUC 0.976) vs the heuristic, same deck both seats
(`lucario_meta`), seat alternating, 100 games. Run twice — once with each determinizer.

## Results (four configurations, none near parity)
| pilot | win-rate vs heuristic | seat artifact? |
|---|---|---|
| step-3 heuristic-rollout ISMCTS | 11% / 17% (two runs) | none |
| value-net search + **basic-energy** determinization | **26.3%** (26/99) | none (13/13) |
| value-net search + **archetype-prior** determinization | **19.0%** (19/100) | none (10/9) |

## The surprising finding: "better" determinization made search WORSE
Prediction going in: the archetype-prior determinizer (held-out hidden-deck recall 5%→40%, vs the
basic-energy filler) would *raise* the win-rate by ~6 pts, per the step-3 delta. **It did the
opposite — 26.3% → 19.0%, a 7-pt DROP.** That prediction was wrong, and the direction is the point:

The value net was trained on **real** game states. With basic-energy filler the forward-simmed
opponent is crude but *neutral* — a simplified board the net can still rank consistently. The
archetype-prior determinizer makes the simmed opponent play a **plausible but counterfactual** hidden
deck, producing states that look realistic but are wrong — and the value net confidently mis-scores
them. A sharper-but-wrong opponent model degrades a learned evaluator MORE than a crude-but-neutral
one. Determinization and the value net are not independent knobs; improving one in isolation hurt.

## Verdict — the search question is closed, and this strengthens it
1-ply search loses to the heuristic across all four configurations (11–26%, all ≥24 pts below
parity), AND improving its components does not monotonically help (better determinization made it
worse). That is strong evidence the ceiling is the 1-ply search **design** — one ply of lookahead
with an imperfect opponent model the learned evaluator can't be made robust to — not any single
component.

**R-NaD / deep self-play is not justified by this evidence.** It would inherit the same
1-ply-with-imperfect-opponent-model failure mode, at multi-week GPU cost. The cheapest tests of
"does searched/learned play beat the strong heuristic" have now come back negative four times.

**Deck/meta selection remains the only proven lever** — which is exactly what is shipped
(MegaStarmie + Dragapult, highest field EWR).

## Files
- `value_search_result.json` — basic-energy run (26.3%) + per-game log
- `value_search_result_prior.json` — archetype-prior run (19.0%) + per-game log
- `value_search_gate.png` — the four-configuration comparison
- `value_search_colab.ipynb` — the notebook (determinizer files must be committed for §3 to use prior det.)
