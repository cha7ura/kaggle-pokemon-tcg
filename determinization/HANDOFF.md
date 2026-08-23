# Step 3 handoff — run the ISMCTS determinization gate

**For:** Claude Code (or a human) on a **linux/amd64** machine with the cg engine (`sdk/cg/libcg.so`).
**Goal:** run one experiment that decides whether better *play* (search) can beat the heuristic,
now that opponent determinization is fixed. The assistant (Claude Science) picks up from
`step3_result.json` afterward.

---

## Background (why this run matters)

The project's own history: forward search / self-play **lost** to the heuristic twice. Diagnosed
cause (wiki/17): the search modeled the opponent's hidden **deck as a copy of our own decklist**
(`H._DECK`) and their hidden **hand/prize as basic-energy filler** → implausible states →
strategy fusion. We fixed that with an **archetype-prior
determinizer**, validated held-out (CPU, no engine):

| metric | old (filler) | new (prior) |
|---|---|---|
| archetype-ID @ 6 revealed cards | 20% | **88%** |
| hidden-deck recall @ 10 revealed | 5.1% | **25.3%** |
| hidden-deck recall @ 20 revealed | 5.2% | **39.5%** |

This run tests whether that fix flips the head-to-head. **Gate: search win-rate vs heuristic > 50%.**

---

## Files in this folder

```
determinization/
├── HANDOFF.md              # this file
├── determinize.py          # the fix: archetype inference + prior-based hidden-state sampler (stdlib)
├── arch_priors.json        # per-archetype card priors (precomputed from 59k-covered ladder decks)
├── agent_ismcts_prior.py   # ISMCTS agent; _determinize() now uses determinize.py (was filler)
├── run_gate.py             # the experiment: same-deck, search-seat vs heuristic-seat, alternating
├── step3_gate.ipynb        # Colab notebook version (upload zip → verify engine → run gate → download json)
├── determinize_eval.py     # held-out validation of the sampler (reproduces recall table)
└── classify_eval.py        # held-out validation of the classifier (reproduces the 20%→88% table)
```

## How to run (CLI, on a linux/amd64 box with the repo)

`run_gate.py` and `agent_ismcts_prior.py` compute paths from their own location and expect to sit at
the **repo root** (next to `sdk/` and `autoresearch/`). So:

```bash
cd <repo-root>                       # the dir containing sdk/ and autoresearch/
cp determinization/agent_ismcts_prior.py .
cp determinization/determinize.py .
cp determinization/arch_priors.json .
cp determinization/run_gate.py .

# quick smoke: does the engine load + one game complete?
PYTHONPATH=sdk python sdk/smoke_test.py 5

# the gate (start small to time it, then scale to 100+)
PYTHONPATH=sdk python run_gate.py --games 30 --sims 48 --depth 12
PYTHONPATH=sdk python run_gate.py --games 100 --sims 48 --depth 12
```

Deck defaults to `autoresearch/decks/lucario_meta.csv` (both seats play it — pilot is the only
variable). Override with `--deck path/to/deck.csv`.

## Docker (if not natively linux/amd64, e.g. Apple Silicon)

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/app -w /app python:3.11-slim \
  bash -lc "PYTHONPATH=sdk python run_gate.py --games 100 --sims 48 --depth 12"
```

## Colab

Open `step3_gate.ipynb`. It uploads a repo zip, asserts the runtime is x86_64, smoke-tests the
engine, runs the gate, and downloads `step3_result.json`. Standard CPU runtime is fine (no GPU
for this step).

---

## Output — what to send back

`run_gate.py` writes **`step3_result.json`** at the repo root:

```json
{
  "games": 100, "sims": 48, "depth": 12, "deck": "lucario_meta.csv",
  "search_wins": 0, "heuristic_wins": 0, "draws": 0, "errors": 0,
  "search_winrate_pct": 0.0,
  "gate_passed": false,
  "verdict": "PASS/FAIL ...",
  "per_game": [ {"game":0,"search_seat":0,"winner":"search|heuristic|draw|error"}, ... ]
}
```

**Return `step3_result.json` to the assistant.** It reads the verdict + per-game log and decides
step 4:
- **PASS (>50%)** → determinization fix validated → start R-NaD self-play (needs GPU).
- **FAIL (≤50%)** → search still doesn't beat the heuristic → deck selection stays the lever;
  skip the multi-week self-play effort. (Either outcome is a real result — a FAIL saves weeks.)

## Notes / gotchas

- **Timing:** each ISMCTS decision = `sims × depth` engine calls. 100 games can take many minutes.
  If too slow, drop `--sims` to 24 for a first read; raise for the real run.
- **`errors`/`None`:** engine load failures or >20k-step games are counted as errors, excluded from
  the win-rate denominator. A high error count means something's wrong with the setup — check the
  smoke test first.
- If `run_gate.py`'s `_is_main` check or `agent_lucario` import fails on your engine build, that's
  the one integration point to eyeball — the determinizer itself (`determinize.py`) is engine-free
  and already validated.
