# 15 — PokeLLMon (LLM as a player)

← [14 — AlphaEvolve](14-alphaevolve.md) · next → [16 — PTCG-Bench](16-ptcg-bench.md)

PokeLLMon (Hu, Huang, Liu 2024) is the "can a large language model just *play* the game?" data
point [pokellmon2024]. It targets Pokémon **battles** (the video-game format, not the TCG), and
reached human parity online. It's context for us, not something we build — the shipped Kaggle agent
runs no LLM ([00](00-index.md)).

## What it did

PokeLLMon drives a battle by prompting an LLM with the game state and asking for the next move.
Three ideas made it work, each patching a specific LLM failure mode:

1. **In-context reinforcement learning** — feed back *text* descriptions of what happened ("your
   move did 0 damage because of immunity") so the model refines its policy within the context window,
   no weight updates.
2. **Knowledge-augmented generation** — retrieve external knowledge (type charts, move effects) to
   curb hallucination and pick type-correct actions. The LLM doesn't reliably *know* the matchup
   table; give it to it.
3. **Consistent action generation** — damp **panic-switching**, the observed tendency to flail
   (switch repeatedly) against a stronger opponent. Commit to a line.

Result: roughly **49% win rate** in ladder play and ~56% in invited matches against human players —
genuinely human-level, human-like play [pokellmon2024].

## Why it's relevant but not our path

- **The deployment model is opposite ours.** PokeLLMon calls an LLM per decision. The Kaggle TCG
  agent is offline, stdlib-only, no network ([01](01-game-and-engine.md)) — an LLM at inference is
  impossible on the ladder. So PokeLLMon's architecture can't ship here.
- **The transferable idea is knowledge-augmentation.** Our heuristic already bakes in domain
  knowledge by hand (weakness ×2, the Crustle wall, prize math — [02](02-policy-lucario.md)). That's
  the same instinct PokeLLMon implements via retrieval: don't make the agent rediscover the rules,
  hand them over.
- **Panic-switching ↔ over-switching.** PokeLLMon's "be consistent" lesson rhymes with our
  energy-spread/Crustle-routing discipline — avoid thrashing, commit to a coherent plan.

The closest LLM-as-player work to *our* exact game (the TCG, and self-improvement) is the companion
benchmark, [16 — PTCG-Bench](16-ptcg-bench.md).

> Takeaway: an LLM can play Pokémon battles at human level with in-context feedback +
> knowledge retrieval + anti-panic consistency — but its per-move-LLM design is unshippable on an
> offline ladder. We keep the knowledge-augmentation instinct, not the architecture.
