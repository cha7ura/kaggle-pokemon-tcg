# 16 — PTCG-Bench (LLM agents on *our* game)

← [15 — PokeLLMon](15-pokellmon.md) · next → [17 — imperfect info on the TCG](17-imperfect-info-on-tcg.md)

PTCG-Bench (Hua et al. 2026) is the paper closest to this project's exact problem: a benchmark for
**LLM agents on the Pokémon Trading Card Game**, focused on whether they can **improve through
experience** [ptcgbench2026]. One finding from it directly justifies the project's caution about its
own neural arm.

## What it measures

Existing agent benchmarks, the authors argue, don't capture *strategic, evolving* scenarios that
mirror how humans learn. PTCG-Bench evaluates two things [ptcgbench2026]:

1. **Decision quality** inside a single intricate game (the TCG).
2. **Self-improvement** — can the agent get better through accumulated gameplay experience?

It uses modular components to separate the agent's *capability* from incidental *implementation/
design* choices.

## The finding that matters to us

> LLM agents play competently but struggle with **sustained and stable self-evolution**, and are
> sensitive to architectural decisions. [ptcgbench2026]

Read that next to our own plan. Plan 2 bets on a from-scratch self-improving net
([12](12-deepnash-rnad.md)). PTCG-Bench is independent evidence that **self-improvement on this
exact game is unstable** — which is exactly why the project:

- keeps the **820.7 hand-tuned heuristic** as the live submission, and
- makes any learned challenger clear the same `eval.py` Wilson-LB gate before it can ship
  ([04](04-the-challenger-framework.md)).

A net that fails to stably improve is the *expected* case here, not a surprise — and the challenger
framework is designed so that outcome costs us nothing on the ladder.

## The contrast that defines our approach

PTCG-Bench (and [PokeLLMon](15-pokellmon.md)) ask "how far does an **LLM agent** get on this game?"
Our project asks a different question: "what's the strongest **pure algorithm** we can ship to an
offline ladder?" — heuristic now, possibly a NumPy R-NaD net later. The papers are the map of the
LLM-agent route; we're driving the algorithmic one. Knowing the LLM route's ceiling (competent but
unstable self-improvement) is useful precisely because it tells us not to over-invest in expecting a
learned agent to dominate.

> Takeaway: the benchmark closest to our game finds LLM agents competent but unstable at
> self-improvement. That validates gating our own learned challenger behind the frozen champion
> rather than trusting it to just get better on its own.
