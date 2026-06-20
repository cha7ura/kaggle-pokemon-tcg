# Design: Project Wiki + minizero + Training Arm + Auto-update Hook + Game-AI Research

Date: 2026-06-20
Status: Approved (brainstorming) — pending user spec review
Branch: `feat/project-wiki-minizero`

## Summary

Build, around the existing Pokémon TCG Kaggle project:

1. A Karpathy-style **wiki** (knowledge base) covering the game, the engine, the current policy,
   the autoresearch method, and DeepMind's imperfect-information game-AI lineage.
2. **minizero** — a runnable algorithm lab: teaching implementations (CFR, ISMCTS) plus a real
   **R-NaD (DeepNash) training pipeline**, validated tabularly on Leduc poker then scaled to the
   cg sim on a Colab GPU.
3. A **challenger framework**: two independent attempts to beat the current champion — an
   **AlphaEvolve-style heuristic auto-loop** (`tools/evolve.py`) and the **trained R-NaD net** —
   both gated by the existing `eval.py` Wilson-LB metric.
4. A **paper-fetch pipeline** and a lightweight **Stop hook** keeping the wiki current.

### The constraint that shapes everything

The shipped Kaggle agent is **offline, stdlib-only, no network, no heavyweight deps**. This does
NOT forbid neural nets — it forbids *torch at inference on the ladder*. The resolution:

- **Train heavy on Colab** (torch, GPU, self-play R-NaD).
- **Export weights** as plain arrays (`weights.npz`).
- **Ship a pure-NumPy forward pass** (`agent_net.py`) — a small MLP is a few `np.dot` calls.

R-NaD/DeepNash is chosen over Player of Games precisely because it is **search-free at inference**:
the trained policy net emits an action distribution, we mask illegal actions and pick. No MCTS, no
forward model, no Docker per move. That is the only neural approach in this lineage that ships
cleanly under the Kaggle constraint.

**Nothing ships an LLM.** The only LLM is the developer (Claude) at dev time: it researches,
writes the wiki, and proposes mutations in the evolve loop. The shipped agent is always pure algo
(heuristic or NumPy-net).

## Goals

1. Legibility: `wiki/` explains the game, engine contract, `agent_lucario.py`, and the method.
2. Teach + run the genuinely relevant algorithms: CFR/regret minimization, ISMCTS, R-NaD —
   grounded in the real `cg` engine.
3. Two disciplined challengers to the 820.7 champion: evolved heuristic + trained net.
4. Honest, cited research pages: Player of Games, DeepNash, AlphaStar, AlphaEvolve, PokeLLMon,
   PTCG-Bench — each tied to *our* agent and the offline constraint.
5. Keep the wiki current via a Stop hook.

## Non-goals

- No runtime LLM in any shipped or simulated agent.
- No torch / GPU dependency in the **shipped** agent — torch is fine in `colab/` (dev only); the
  ladder agent does NumPy inference from exported weights.
- v1 does not require the net to *win*. The heuristic remains the live submission until a
  challenger clears the gate. A net that loses is a documented, valuable result (cf. PTCG-Bench's
  instability finding), not a failure of the build.
- No reproduction of DeepNash/PoG/AlphaStar at their published scale (we lack that compute). We
  implement the *algorithm* at the scale one Colab GPU allows and report honestly.

## Background: what already exists

- `autoresearch/agent_lucario.py` — champion heuristic (Mega Lucario ex). Develop-then-attack
  context scoring; Crustle routing, prize-trade math, energy-spread, Boss's Orders drag.
- `autoresearch/eval.py` — fixed seat-swapped gauntlet; Wilson lower-bound gate.
- `autoresearch/program.md` — the keep/revert loop methodology.
- `autoresearch/log/experiments.md` — experiment ledger (kept + rejected).
- `sdk/cg/` — engine: `Observation`/`Select` contract; forward model via
  `search_begin`/`search_step`/`search_end`.
- `autoresearch/agent_search.py` — rejected policy that proves the forward-model plumbing: it
  `_determinize`s hidden state and rolls out. Reused by minizero's cg env.

## Architecture

Six loosely-coupled units, each independently understandable and testable.

### Unit A — `wiki/` (markdown knowledge base)

Karpathy voice (dumb-version-first → why-wrong → fix). Pages:

```
wiki/
  00-index.md                  reading order + project map
  01-game-and-engine.md        TCG rules as the agent sees them; obs/select; docker/ctypes
  02-policy-lucario.md         annotated walk of agent_lucario.py (_score_main, W, routing)
  03-autoresearch-loop.md      Wilson-LB keep/revert method; reading experiments.md
  04-the-challenger-framework.md  how evolve + net both challenge the champion via eval.py
  10-cfr-and-nash.md           info sets, regret matching, CFR/CFR+/MCCFR, Nash, exploitability
  11-player-of-games.md        GT-CFR + counterfactual value/policy net; sound search
  12-deepnash-rnad.md          R-NaD: model-free, search-free path to Nash; Stratego scale
  13-alphastar-league.md       league training, prioritized fictitious self-play, exploiters
  14-alphaevolve.md            evolutionary code search; our loop IS a manual AlphaEvolve
  15-pokellmon.md              LLM-as-player; in-context RL + knowledge augmentation
  16-ptcg-bench.md             LLM-agent benchmark on the TCG; self-evolution instability
  17-imperfect-info-on-tcg.md  the bridge: what ports to our offline agent and what doesn't;
                               why R-NaD-net-as-NumPy is the shippable neural path
  perfect-info-ancestors.md    short: AlphaGo/AlphaZero/MuZero; MCTS assumes perfect info
  papers/
    <arxiv-id>.md              raw fetched/transformed paper text
    sources.md                 citation ledger
```

### Unit B — `minizero/` (algorithm lab)

```
minizero/
  cfr/                       TEACHING (pure NumPy, runs anywhere)
    kuhn.py                  Kuhn poker
    cfr.py                   vanilla CFR + regret matching
    mccfr.py                 external-sampling MCCFR (optional multiprocessing)
    exploit.py              best-response / exploitability
    run.py                  train; exploitability -> 0; dump Nash strategy
  cg/                        PRACTICAL bridge (runs in Docker on the real engine)
    cg_env.py               wrapper over sdk/cg search_begin/step/end + _determinize
    ismcts.py               PUCT information-set MCTS; determinize per sim; optional root-parallel
    run.py                  pick a move from a real cg Observation
  rnad/                      THE TRAINING METHOD (DeepNash)
    leduc.py                Leduc poker env (toy validation target)
    rnad_tabular.py         tabular R-NaD on Leduc, pure NumPy — verify exploitability -> 0
    net.py                  torch policy/value net (Colab only)
    train.py                neural R-NaD self-play loop (Colab GPU); exports weights.npz
    export.py               torch state_dict -> weights.npz (plain arrays)
  README.md                 maps every file to its paper/section
```

Progression matches "toy first": `rnad_tabular.py` on Leduc proves the R-NaD update reaches Nash
(NumPy, no GPU), then `train.py` reuses the same update rule with a torch net on the cg env.

### Unit C — GPU training runner (runner-agnostic; choice deferred to step 8)

All training logic lives in versioned `minizero/rnad/*.py` so the GPU runner is interchangeable.
The runner is a thin driver that installs torch, runs `minizero/rnad/train.py` on a GPU, and
emits `weights.npz`. The specific runner is finalized at step 8, not now (steps 1–7 need no GPU).

**Leading candidate: the official `googlecolab/colab-mcp` MCP server** (v1.0.2, Mar 2026; Claude
Code supported). Rationale: this repo's authoritative state — git history, the compiled `cg` SDK,
the `eval.py` Docker gate — stays local, while only the heavy R-NaD training bursts to a Colab GPU
runtime. Colab is linux/amd64, the same platform `libcg.so` targets, so the cg self-play env runs
there too. Claude drives notebook cells via MCP tools (create/edit/execute, pip install) and pulls
`weights.npz` back to the local repo for gating. Caveats: bridges to a browser Colab session
(one-time manual open/auth), early release, GPU provisioning not contractually specified.

Rejected alternatives:
- **claude-colab (Claude Code *inside* an ephemeral Colab VM)** — inverts the architecture; would
  shuffle the whole repo into a disposable box and lose local Docker/eval integration.
- **Kaggle kernels / HF Jobs** — viable fully-headless fallbacks (kaggle CLI already configured;
  HF token for HF Jobs) if colab-mcp proves too rough.
- **Colab via Playwright/Chrome DevTools** — brittle over multi-hour jobs; not used.

Parallel self-play via `n_workers` (multiprocessing) where the cg env is the bottleneck; torch
handles GPU batch parallelism.

### Unit D — shippable net inference: `agent_net.py`

A stdlib+NumPy agent with the same `agent(obs_dict) -> list[int]` contract as `agent_lucario.py`.
Loads `weights.npz`, runs a NumPy forward pass over the encoded observation, masks illegal
actions, picks. No torch. This is what would be bundled into a `submission_net/` if the net ever
clears the gate. Until then it lives in `autoresearch/` as a challenger.

### Unit E — `tools/`

```
tools/
  fetch_papers.py           arxiv id -> PDF -> markdown -> wiki/papers/<id>.md ; append sources.md
  evolve.py                 AlphaEvolve-style auto-loop (see below)
```

`tools/evolve.py`: Claude proposes a mutation to `agent_lucario.py`; `eval.py` scores it against
the mirror + Dragapult Wilson-LB gate; auto-keep if it clears; auto-append to `experiments.md`.
Candidate evaluations run in parallel (multiple `eval.py` gauntlets concurrently — Colab or local
multiprocessing). No runtime LLM in the agent; Claude is the dev-time proposer. (Promoted from
fast-follow into v1 per the "build both arms" decision.)

### Unit F — Stop hook

```
.claude/
  settings.json             registers the Stop hook
  hooks/check-wiki.sh       lightweight, no LLM, idempotent
```

On Stop, inspects the working tree; if code/research/decisions changed but no `wiki/` file was
edited, emits context asking Claude to update the relevant page or note why not. Silent if `wiki/`
was already touched this turn. Never calls an LLM, never edits files — only nudges, so no loop.

## The challenger framework (Unit A page 04 + the gate)

The frozen champion (`champion_lucario.py`) is the benchmark opponent. A challenger ships only if
it beats the champion through the **existing** `eval.py` seat-swapped Wilson-LB gate (current
gate: mirror LB ≥ 0.48 AND Dragapult holds). Two challenger producers:

- `tools/evolve.py` → mutated heuristic challengers.
- `minizero/rnad/` (trained on Colab) → `agent_net.py` + `weights.npz` challenger.

Same gate, same metric, same ledger (`experiments.md`). The live submission only changes when a
challenger wins. This is the project's existing discipline, extended to neural challengers.

## Data flow

```
arxiv ids ──fetch_papers.py──> wiki/papers/*.md ──┐
deep-research workflow ───────────────────────────┼──> wiki/1x-*.md (synthesized, cited)
codebase (mapped) ────────────────────────────────┘──> wiki/0x-*.md

Leduc ──rnad_tabular.py(NumPy)──> verify Nash (exploitability->0)
   │ (same update rule)
cg sim ──minizero/rnad/train.py(torch,GPU runner)──> weights.npz ──> agent_net.py(NumPy) ──┐
agent_lucario.py ──tools/evolve.py(mutate)──> evolved heuristic ────────────────────┤
                                                                                    ├─> eval.py gate
champion_lucario.py (frozen benchmark) ─────────────────────────────────────────────┘
                                                                  winner -> submission + experiments.md

every turn end ──> Stop hook ──> (wiki stale?) ──> nudge Claude ──> edit wiki
```

## Parallelism (self-play)

Stdlib parallelism = `multiprocessing` (real cores; threads useless under the GIL for CPU work).
The `cg` engine loads per-process via ctypes → each worker has an isolated engine instance, no
shared-state collision. Workers seeded by base seed + index for reproducibility.

Ranked by payoff:

1. **`tools/evolve.py` candidate evaluation** — embarrassingly parallel: K `eval.py` gauntlets
   concurrently (each a separate Docker run). Largest practical ladder-climb speedup.
2. **`minizero/rnad/train.py` self-play (GPU runner)** — torch batches on GPU; env stepping fans
   out across `n_workers` processes when the cg env is the bottleneck.
3. **`minizero/cg/ismcts.py`** — root parallelization: N independent trees, merge root visit
   counts. Per-process engine makes it safe. Opt-in `n_workers`, default 1.
4. **`minizero/cfr/mccfr.py`** — distributed external sampling: workers accumulate regret deltas,
   merged per outer iteration. Documented option; Kuhn converges single-core in seconds.

Single-process is always the default/reference; parallelism is opt-in via `n_workers`.

## Testing

- `minizero/cfr/`: exploitability converges below epsilon on Kuhn (known Nash value is the oracle).
- `minizero/rnad/rnad_tabular.py`: exploitability on Leduc decreases and converges — the
  correctness gate for the R-NaD update before any GPU spend.
- `minizero/rnad/export.py`: round-trip test — torch forward and NumPy forward on the same input
  agree within float tolerance (guarantees `agent_net.py` matches the trained net).
- `minizero/cg/`: smoke — ISMCTS returns a legal index for real cg Observations in Docker.
- `agent_net.py`: smoke — legal action for sampled observations; never raises (crash = forfeit).
- `tools/fetch_papers.py`: two known arxiv ids → non-empty markdown + sources.md entry.
- `tools/evolve.py`: dry-run produces a candidate, scores it, and logs without mutating the
  champion when the candidate loses.
- Stop hook: silent when `wiki/` touched; nudges when only code changed.
- Wiki: editorial; the hook is the maintenance mechanism.

## Build order

1. `wiki/` skeleton + `00/01/02/03` (from the mapped codebase — no external deps).
2. `tools/fetch_papers.py`; fetch PoG, DeepNash, AlphaStar, AlphaEvolve, PokeLLMon, PTCG-Bench.
3. Fold deep-research + papers into `wiki/04, 10–17` + ancestors page.
4. Stop hook + `.claude/settings.json`.
5. `minizero/cfr/` (TDD against Kuhn) and `minizero/cg/` ISMCTS on the real sim.
6. `minizero/rnad/` tabular R-NaD on Leduc (TDD against exploitability) — the correctness oracle.
7. `tools/evolve.py` — the low-risk challenger; start climbing the ladder.
8. `minizero/rnad/` neural `net.py`/`train.py`/`export.py` + a runner driver (Kaggle/HF/Colab —
   chosen here); `agent_net.py` NumPy inference; round-trip test; run training on GPU; gate result.

Steps 1–7 need no GPU. Step 8 is the Colab arm and the highest-variance work; it comes last so the
safe ladder gains (step 7) land first.

## Risks / open items

- A from-scratch R-NaD net on one Colab GPU may not beat the 820.7 heuristic. Mitigation: the
  heuristic stays live; the net is a gated challenger; a loss is a documented result.
- ISMCTS and cg-env tests run only inside Docker `linux/amd64`. Tabular CFR/R-NaD run anywhere.
- Action-space encoding for the net is nontrivial (variable, context-dependent `select.option`
  lists). Resolved in step 8 design: encode a fixed action vocabulary + per-state legality mask;
  this is itself a wiki-worthy lesson.
- PDF extraction quality varies; marked WebFetch summaries are an acceptable fallback.
- Deep-research (running at spec-write time) supplies depth for `1x-*.md`; if it surfaces a
  simpler in-family algorithm than R-NaD for the neural arm, we revisit — but R-NaD's search-free
  inference is the property that makes it shippable, so it is the committed default.
