# Pokémon TCG Kaggle AI — Wiki Index

## What this project is

This is a Kaggle Pokémon TCG AI competition project. The task is to ship a single function
`agent(obs) -> list[int]` that selects legal actions from a live game observation. The shipped agent
is **offline, stdlib-only, no network, no heavyweight deps** — it runs inside the competition
sandbox with no external calls. The game engine (`cg`) is a compiled C library accessed via ctypes
(or Docker on linux/amd64). The current champion is a hand-crafted heuristic (`agent_lucario.py`)
scoring ~820 on the ladder. This wiki is the knowledge base that explains the game, the engine, the
policy, the autoresearch method, and the game-AI research lineage that informs how we think about
beating it.

**Nothing here ships an LLM. The only LLM is the developer (Claude) at dev time — it researches,
writes this wiki, and proposes mutations in the evolve loop. The shipped agent is always pure algo:
heuristic or NumPy-net.** No runtime LLM, no torch, no GPU in the submitted agent. Ever.

Two challengers compete to unseat the champion; both are gated by the same Wilson lower-bound
metric in `eval.py`. See [04 — The challenger framework](04-the-challenger-framework.md) for how
that gate works and why "two challengers, one gate" is the right discipline.

All research claims in this wiki cite an entry in [papers/sources.md](papers/sources.md).

---

## Reading order

Pages 01–17 and `perfect-info-ancestors.md` do not exist yet — they arrive in later tasks. This
index is the map; the pages are the territory.

### Foundation (codebase-derived, no external deps)

| # | Page | One-line hook |
|---|------|---------------|
| [00](00-index.md) | [00 — Index](00-index.md) | This page. Project map and reading order. |
| 01 | [01 — Game and engine](01-game-and-engine.md) | TCG rules as the agent sees them; the `obs`/`select` contract; ctypes and Docker. |
| 02 | [02 — Policy: Lucario](02-policy-lucario.md) | Annotated walk of `agent_lucario.py` — `_score_main`, the W matrix, Crustle routing. |
| 03 | [03 — Autoresearch loop](03-autoresearch-loop.md) | Wilson-LB keep/revert discipline; how to read `experiments.md`. |
| [04](04-the-challenger-framework.md) | [04 — The challenger framework](04-the-challenger-framework.md) | Two challengers, one gate: evolved heuristic vs. trained net, both through `eval.py`. |

### Game-AI research lineage

| # | Page | One-line hook |
|---|------|---------------|
| 10 | [10 — CFR and Nash](10-cfr-and-nash.md) | Info sets, regret matching, CFR/CFR+/MCCFR — the dumb version first, then why it's insufficient. |
| 11 | [11 — Player of Games](11-player-of-games.md) | GT-CFR + counterfactual value/policy net; sound search; why it's too expensive at inference. |
| 12 | [12 — DeepNash / R-NaD](12-deepnash-rnad.md) | Model-free, search-free path to Nash; why this is the shippable neural approach. |
| 13 | [13 — AlphaStar league](13-alphastar-league.md) | League training, prioritised fictitious self-play, exploiters — and why we can't afford it. |
| 14 | [14 — AlphaEvolve](14-alphaevolve.md) | Evolutionary code search; our `evolve.py` loop is a manual AlphaEvolve. |
| 15 | [15 — PokeLLMon](15-pokellmon.md) | LLM-as-player; in-context RL + knowledge augmentation; what it teaches us to avoid. |
| 16 | [16 — PTCG-Bench](16-ptcg-bench.md) | LLM-agent benchmark on the TCG; self-evolution instability finding. |
| 17 | [17 — Imperfect info on TCG](17-imperfect-info-on-tcg.md) | The bridge: what ports to our offline agent and what doesn't; why R-NaD-as-NumPy is the answer. |
| — | [Perfect-info ancestors](perfect-info-ancestors.md) | Short: AlphaGo/AlphaZero/MuZero — MCTS assumes perfect info; this is why they don't port. |
| 18 | [18 — Neural net experiments](18-net-experiments.md) | The honest record: BC + RL net runs on Colab/wandb; hit the imperfect-info wall, heuristic stays champion. |

---

## The gate (don't skip this)

Every challenger — whether a mutated heuristic from `tools/evolve.py` or a trained net from
`minizero/rnad/` — must clear the **existing** `eval.py` seat-swapped Wilson lower-bound gate
(mirror LB ≥ 0.48 AND Dragapult holds) before it touches the live submission. The champion stays
live until something beats it. This is not optional discipline; it is the loop. See
[04 — The challenger framework](04-the-challenger-framework.md).

---

## Citation convention

Cite as `[pog2021]`, `[deepnash2022]`, etc. Full bibliography: [papers/sources.md](papers/sources.md).
