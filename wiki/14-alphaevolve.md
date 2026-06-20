# 14 — AlphaEvolve, and why our loop already is one

← [13 — AlphaStar and the league](13-alphastar-league.md) · next → [15 — PokeLLMon](15-pokellmon.md)

AlphaEvolve (DeepMind, 2025) is the odd one out in this wiki: it's not a way to *play* a game, it's
a way to **write better code** [alphaevolve2025]. It belongs here because it is the closest existing
system to what this project already does by hand — the autoresearch loop ([03](03-autoresearch-loop.md)).

## What it is

AlphaEvolve is an evolutionary coding agent. The loop [alphaevolve2025]:

1. **Generate** — an LLM (Gemini Flash for breadth, Pro for depth) proposes code mutations.
2. **Evaluate** — an automated evaluator scores each candidate on objective metrics.
3. **Select** — rank by score; the best candidates feed back into the prompt for the next round.

Mutation → evaluation → selection → feed winners forward. It's a genetic algorithm where the
mutation operator is an LLM and the fitness function is "does this code score better."

Results, to calibrate that this is real [alphaevolve2025]: a faster 4×4 complex matrix-multiply
(improving on Strassen's 1969 algorithm), a ~23% speedup to a matmul kernel that cut Gemini training
time ~1%, up to 32.5% on FlashAttention, a Borg scheduling heuristic recovering ~0.7% of Google's
fleet, and new math bounds (kissing number in 11 dimensions).

## Our autoresearch loop *is* a manual AlphaEvolve

Put them side by side:

| AlphaEvolve | This project ([03](03-autoresearch-loop.md)) |
|---|---|
| LLM proposes a code mutation | Claude (or a human) edits `agent_lucario.py` |
| Automated evaluator scores it | `eval.py` runs the seat-swapped Wilson-LB gauntlet |
| Keep the best, feed winners forward | Keep iff `PROMOTE`; copy to champion; log it |
| Population / island prompts | the `experiments.md` ledger of kept + rejected attempts |

The only missing piece is **automating the propose-step**. Today a human/Claude writes each mutation
in the loop. Automate that and you have AlphaEvolve pointed at our own agent.

## The fast-follow: `tools/evolve.py`

That automation is the Plan-2 fast-follow ([04](04-the-challenger-framework.md)): Claude proposes a
mutation, `eval.py` scores it against the mirror + Dragapult gate, auto-keep on PROMOTE, auto-append
to `experiments.md`. Candidate evaluations parallelize trivially (many gauntlets at once). Crucially
it ships **no runtime LLM** — the LLM is the dev-time mutation generator; the shipped agent stays a
pure algorithm.

This is plausibly the **highest-leverage** thing the project can build: it compounds directly on the
already-strong 820.7 heuristic, rather than betting on a from-scratch net beating it
([12](12-deepnash-rnad.md)).

> Takeaway: AlphaEvolve = LLM-as-mutation-operator + automated evaluator + selection. We already run
> that loop by hand; `tools/evolve.py` just removes the human from the inner cycle.
