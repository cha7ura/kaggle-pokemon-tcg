# Design: Project Wiki + minizero + Auto-update Hook + Game-AI Research

Date: 2026-06-20
Status: Approved (brainstorming) — pending user spec review
Branch: `feat/project-wiki-minizero`

## Summary

Build a Karpathy-style **knowledge base** for this Pokémon TCG Kaggle project, a small
**runnable algorithm lab** (`minizero/`) implementing imperfect-information game methods in
pure NumPy/stdlib, a **paper-fetch pipeline**, and a **Stop hook** that nudges the wiki to stay
current every turn. Research centers on DeepMind's *imperfect-information* lineage because the
Pokémon TCG is a hidden-information, stochastic game.

**Key constraint that shapes everything:** the shipped Kaggle agent is offline, stdlib-only, and
makes no network calls. Therefore **nothing here ships an LLM**, and **nothing built here ships a
heavyweight ML dependency**. The only LLM in this project is the developer (Claude) at dev time:
it researches, writes the wiki, and runs the keep/revert loop. The competition agent stays a pure
algorithm. minizero is a teaching/experimentation artifact and is never submitted to Kaggle.

## Goals

1. Make the project legible: anyone can read `wiki/` and understand the game, the engine
   contract, the current `agent_lucario.py` policy, and the autoresearch method.
2. Teach (and run) the genuinely relevant algorithms for an imperfect-information card game:
   CFR / regret minimization and Information-Set MCTS, grounded in the real `cg` engine.
3. Synthesize the requested research into honest, cited Karpathy-style pages: Player of Games,
   DeepNash (R-NaD), AlphaStar, AlphaEvolve, PokeLLMon, PTCG-Bench — each tied back to *our*
   agent and the offline-stdlib constraint.
4. Keep the wiki current automatically via a lightweight Stop hook.

## Non-goals

- No runtime LLM in any shipped or simulated agent.
- No PyTorch / no GPU / no training infrastructure. (DeepNash/PoG/AlphaStar are *explained and
  partially mirrored at toy scale*, not reproduced.)
- No change to the existing submission policy in v1. (Improving `agent_lucario.py` is the
  province of the existing autoresearch loop; the AlphaEvolve auto-loop that would automate it is
  a documented fast-follow, not v1.)

## Background: what already exists

- `autoresearch/agent_lucario.py` — current champion heuristic (Mega Lucario ex). Context-scored
  greedy: develop board, then attack; Crustle-wall routing, prize-trade math, energy-spread,
  Boss's Orders drag.
- `autoresearch/eval.py` — fixed seat-swapped gauntlet; Wilson lower-bound gate.
- `autoresearch/program.md` — the Karpathy-style keep/revert loop methodology.
- `autoresearch/log/experiments.md` — experiment ledger (kept + rejected).
- `sdk/cg/` — engine: `Observation`/`Select` contract, and a forward model via
  `search_begin` / `search_step` / `search_end`.
- `autoresearch/agent_search.py` — a *rejected policy* that nonetheless proves the forward-model
  plumbing works: it `_determinize`s hidden state and rolls out inside `search_begin/step`.
  minizero reuses this machinery.

## Architecture

Four loosely-coupled units, each independently understandable and testable:

### Unit A — `wiki/` (markdown knowledge base)

Static markdown, Karpathy voice (dumb-version-first, then why-it's-wrong, then the fix). Pages:

```
wiki/
  00-index.md                  reading order + project map
  01-game-and-engine.md        TCG rules as the agent sees them; obs/select contract; docker/ctypes
  02-policy-lucario.md         annotated walk of agent_lucario.py (_score_main, W, routing)
  03-autoresearch-loop.md      Wilson-LB keep/revert method; how to read experiments.md
  10-cfr-and-nash.md           information sets, regret matching, CFR/CFR+/MCCFR, Nash, exploitability
  11-player-of-games.md        GT-CFR + counterfactual value/policy net; sound search; PoG = "AlphaZero for hidden info"
  12-deepnash-rnad.md          R-NaD, model-free, search-free path to Nash; Stratego scale
  13-alphastar-league.md       league training, prioritized fictitious self-play, exploiters, non-transitivity
  14-alphaevolve.md            evolutionary code search; our autoresearch loop IS a manual AlphaEvolve
  15-pokellmon.md              LLM-as-player (battles); in-context RL + knowledge augmentation
  16-ptcg-bench.md             LLM-agent benchmark on the TCG; self-evolution findings
  17-imperfect-info-on-tcg.md  THE BRIDGE: why PoG/DeepNash/AlphaStar don't port to an offline
                               stdlib Kaggle agent; what genuinely transfers (determinization,
                               opponent uncertainty, league-style robustness, regret intuition)
  perfect-info-ancestors.md    short: AlphaGo/AlphaZero/MuZero, why MCTS assumes perfect info
  papers/
    <arxiv-id>.md              raw fetched/transformed paper text
    sources.md                 citation ledger (title, authors, year, url, retrieved date)
```

Page content is sourced from: (a) the codebase I've mapped, (b) the running deep-research report,
(c) the fetched papers. Every research claim cites `papers/sources.md`.

### Unit B — `minizero/` (pure-NumPy algorithm lab)

Two real, runnable algorithms. Both are pure NumPy/stdlib. Neither ships to Kaggle.

```
minizero/
  cfr/
    kuhn.py        Kuhn poker game definition (canonical hidden-info teaching game)
    cfr.py         vanilla CFR + regret matching; tracks average strategy
    mccfr.py       external-sampling Monte Carlo CFR
    exploit.py     best-response / exploitability computation (the convergence proof you can watch)
    run.py         train; print exploitability -> 0; dump the learned Nash strategy
  cg/
    cg_env.py      thin wrapper over sdk/cg search_begin/step/end + _determinize (from agent_search.py)
    ismcts.py      PUCT information-set MCTS: determinize per simulation, average over worlds
    run.py         run ISMCTS to pick a move from a real cg Observation
  README.md        maps every file to the corresponding paper/section
```

Rationale for the split: CFR-on-Kuhn is the *genuine* member of the imperfect-info family small
enough to verify (exploitability provably → 0). ISMCTS-on-`cg` is the *practical* technique that
actually runs on our engine and connects to `agent_lucario.py`. Full CFR/PoG/DeepNash on the real
TCG is infeasible (astronomical infoset count, no GPU, offline) — `17-imperfect-info-on-tcg.md`
explains exactly why.

### Unit C — `tools/` (paper pipeline)

```
tools/
  fetch_papers.py   arxiv id -> PDF -> clean markdown -> wiki/papers/<id>.md ; appends sources.md
```

stdlib `urllib` for download; light PDF-to-text extraction. If extraction quality is poor for a
given paper, fall back to a WebFetch-sourced summary (clearly marked as summary, not full text).
The deep-research workflow output is folded into the `1x-*.md` pages by hand during implementation.

### Unit D — Stop hook (auto-update nudge)

```
.claude/
  settings.json            registers the Stop hook
  hooks/check-wiki.sh      lightweight, no LLM, idempotent
```

Behavior: on Stop, the hook inspects the working tree (e.g. `git status` / files changed this
session vs. last `wiki/` touch). If code/research/decisions changed but no `wiki/` file was
edited, it emits context asking Claude to update the relevant wiki page or explicitly note why no
update is needed. If `wiki/` was already touched this turn, it stays silent. The hook never calls
an LLM and never edits files itself — it only nudges, so there is no loop risk.

## Data flow

```
arxiv ids ──fetch_papers.py──> wiki/papers/*.md ──┐
deep-research workflow ───────────────────────────┼──> wiki/1x-*.md (synthesized, cited)
codebase (mapped) ────────────────────────────────┘──> wiki/0x-*.md

sdk/cg forward model ──> minizero/cg/cg_env.py ──> ismcts.py ──> move
Kuhn game ────────────> minizero/cfr/*.py ──────> Nash strategy + exploitability curve

every turn end ──> Stop hook ──> (wiki stale?) ──> nudge Claude ──> Claude edits wiki
```

## Parallelism (self-play)

Stdlib only, so parallelism = `multiprocessing` (separate processes for real cores; threads are
useless for this CPU-bound work under the GIL). The `cg` engine is loaded per-process via ctypes,
so each worker holds an isolated engine instance — no shared-state collision. Each worker is
seeded deterministically (e.g. base seed + worker index) for reproducibility.

Ranked by payoff:

1. **(Fast-follow) `tools/evolve.py` candidate evaluation** — embarrassingly parallel: run K
   `eval.py` gauntlets concurrently (each already a separate Docker run). Largest practical win
   for ladder climbing. Designed for now, built in the fast-follow.
2. **`minizero/cg/ismcts.py` — root parallelization** — spawn N independent search trees across
   processes, merge root visit counts at the end. Per-process engine makes this safe and simple.
   Exposed as an optional `n_workers` parameter; default 1 for clarity, >1 for speed.
3. **`minizero/cfr/mccfr.py` — distributed external sampling** — workers run sampled traversals
   independently and accumulate *regret deltas*, merged per outer iteration (never naive shared
   writes). Documented as an option; Kuhn converges in seconds single-core, so default is
   single-process. This keeps the teaching code readable while showing the parallel pattern.

The single-process path is always the default and the reference; parallelism is opt-in via a
`n_workers` argument so the teaching artifact stays readable.

## Testing

- `minizero/cfr/`: assert exploitability decreases monotonically-ish and converges below a small
  epsilon on Kuhn poker (CFR's known Nash value for Kuhn is a checkable target). This is the unit
  test that proves the implementation is correct.
- `minizero/cg/`: smoke test — ISMCTS returns a legal action index for a real `cg` Observation
  inside Docker; no crash across N sampled states.
- `tools/fetch_papers.py`: test against the two known arxiv ids; assert non-empty markdown +
  a sources.md entry.
- Stop hook: test that it stays silent when `wiki/` was touched and nudges when only code changed.
- Wiki: no automated test; correctness is editorial. The hook is the maintenance mechanism.

## Build order

1. `wiki/` skeleton + `00/01/02/03` (from the mapped codebase — no external dependency).
2. `tools/fetch_papers.py`; fetch the four+ papers (PoG, DeepNash, AlphaStar, AlphaEvolve,
   PokeLLMon, PTCG-Bench) into `wiki/papers/`.
3. Fold the deep-research report + fetched papers into `wiki/10–17` + ancestors page.
4. Stop hook + `.claude/settings.json`.
5. `minizero/cfr/` (TDD against Kuhn exploitability), then `minizero/cg/` (ISMCTS on real sim).

## Fast-follow (documented, NOT in v1)

`tools/evolve.py` — an AlphaEvolve-style automation of the autoresearch loop: propose a mutation
to `agent_lucario.py`, score it with `eval.py` against the mirror + Dragapult Wilson-LB gate,
auto-keep if it clears, auto-append to `experiments.md`. Still no runtime LLM in the agent; the
LLM (Claude) is the mutation proposer at dev time. This is the highest-leverage follow-up for
actually climbing the ladder and is the natural capstone the wiki points toward.

## Risks / open items

- ISMCTS on `cg` only runs inside the Docker `linux/amd64` container (engine constraint). Tests
  for `minizero/cg/` must run there; `minizero/cfr/` runs anywhere.
- Deep-research workflow (running in background at spec-write time) supplies the depth for the
  `1x-*.md` pages; if it surfaces a simpler/better in-family algorithm than CFR-on-Kuhn for the
  NumPy artifact, we adopt it (the spec commits to "a genuine imperfect-info family member you can
  watch converge," CFR-on-Kuhn being the default choice).
- PDF extraction quality varies; fallback to marked summaries is acceptable.
