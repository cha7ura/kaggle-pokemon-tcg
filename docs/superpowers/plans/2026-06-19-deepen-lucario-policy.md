# Deepen the Lucario Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the Lucario agent's ladder rating from ~820 toward the top tier by deepening the policy in the matchups that actually occur (Lucario mirror + Dragapult), validated against meta-faithful benchmarks.

**Architecture:** Build two benchmark opponents (a frozen-v4 mirror champion + a Dragapult sample pilot), then apply one isolated policy diff at a time to `autoresearch/agent_lucario.py`, keeping a change only if it holds the mirror AND improves/holds Dragapult over N≥400 seat-swapped games. Periodic Kaggle submissions; the live ladder score is the final arbiter.

**Tech Stack:** Python 3.11 (stdlib only for the agent), the cabt engine via ctypes, Docker `linux/amd64`, the existing `autoresearch/eval.py` gauntlet harness, `kaggle` CLI.

## Global Constraints

- The native engine (`sdk/cg/libcg.so`) runs **only** under `docker run --platform linux/amd64`. All evaluations go through `./autoresearch/run.sh` (sets `PYTHONPATH=/app/sdk`, workdir `/app/autoresearch`).
- Agent contract: `agent(obs_dict) -> list[int]`, indices into `obs.select.option`, length in `[minCount, maxCount]`, no duplicates; if `obs.select is None` return the 60 card IDs.
- **Crash-safety is mandatory** — any exception forfeits the game and can fail the validation mirror. The top-level `try/except → _legal_fallback` wrapper in `agent_lucario.py` must remain intact; new code goes inside it.
- **Validation gate (per policy task):** keep the change only if, over **N≥400** seat-swapped games each: (a) mirror score's Wilson LB ≥ 0.48 (holds the ~50/50 mirror) AND (b) Dragapult score ≥ the pre-change Dragapult score minus noise (improves or holds). Otherwise `git checkout` the diff (revert).
- One isolated diff per task. Append every result (kept OR reverted, with numbers) to `autoresearch/log/experiments.md`.
- Ladder: 5 submissions/day, only latest 2 tracked. Current champion = Lucario v4 @ 820.7.
- Known card IDs: Makuhita 673, Hariyama 674, Lunatone 675, Solrock 676, Riolu 677, Mega Lucario ex 678, Boss's Orders 1182, Switch 1123, Fighting energy 6. Mega Brave attackId 983.
- The project is its own git repo at `~/Documents/Github/kaggle/pokemon`. Commit after each kept change.

---

### Task 1: Dragapult benchmark opponent

**Files:**
- Create: `autoresearch/decks/dragapult.csv`
- Create: `autoresearch/dragapult_agent.py`

**Interfaces:**
- Produces: `dragapult_agent.agent(obs_dict) -> list[int]` and `dragapult_agent.read_deck(path="decks/dragapult.csv")` — used as a fixed `--champion` opponent in later gauntlets.

- [ ] **Step 1: Pull the host Dragapult sample notebook**

```bash
cd ~/Documents/Github/kaggle/pokemon
kaggle kernels pull kiyotah/a-sample-rule-based-agent-dragapult-ex-deck -p refs/dragapult
```

- [ ] **Step 2: Extract the 60-card deck list**

Read `refs/dragapult/*.ipynb`, find the cell that writes `deck.csv` (a list of 60 card IDs). Write those IDs, one per line, to `autoresearch/decks/dragapult.csv`. Verify:

```bash
wc -l autoresearch/decks/dragapult.csv   # expect 60
```
Expected: `60`

- [ ] **Step 3: Create the Dragapult pilot**

Port the notebook's agent into `autoresearch/dragapult_agent.py`. If the notebook's policy is long/Dragapult-specific, port it faithfully; if it only ships a generic rule-based pilot, use this minimal develop-then-attack pilot (sufficient as a directional opponent):

```python
"""Dragapult ex sample pilot — fixed benchmark opponent (not edited by us)."""
import random
from cg.api import to_observation_class, OptionType, SelectContext

_PRI = {int(OptionType.EVOLVE): 900, int(OptionType.ABILITY): 800,
        int(OptionType.ATTACH): 700, int(OptionType.PLAY): 600,
        int(OptionType.ATTACK): 100, int(OptionType.RETREAT): -1}

def read_deck(path="decks/dragapult.csv"):
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]

_DECK = read_deck()

def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _DECK
    sel = obs.select
    if int(sel.type) == 0:
        scores = [_PRI.get(int(o.type), 0) for o in sel.option]
    else:
        scores = [1] * len(sel.option)
    order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
    k = min(sel.maxCount, len(sel.option))
    k = max(k, min(max(1, sel.minCount), len(sel.option)))
    return order[:k]
```

- [ ] **Step 4: Verify the deck is legal and the pilot plays a full game**

```bash
cd ~/Documents/Github/kaggle/pokemon
./autoresearch/run.sh --challenger dragapult_agent.py --champion dragapult_agent.py \
  --deck decks/dragapult.csv --deck-champion decks/dragapult.csv --games 20
```
Expected: JSON with `"games": 20` (NOT 0 — 0 means an illegal deck), no traceback.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/Github/kaggle/pokemon
git add autoresearch/decks/dragapult.csv autoresearch/dragapult_agent.py
git commit -m "test: add Dragapult benchmark opponent (deck + sample pilot)"
```

---

### Task 2: Frozen mirror champion + baseline measurements

**Files:**
- Create: `autoresearch/champion_lucario.py` (frozen copy of current v4)
- Modify: `autoresearch/log/experiments.md` (append baselines)

**Interfaces:**
- Produces: `champion_lucario.py` with `agent(obs_dict)` — the fixed mirror opponent for every later task.

- [ ] **Step 1: Freeze the current champion**

```bash
cd ~/Documents/Github/kaggle/pokemon
cp autoresearch/agent_lucario.py autoresearch/champion_lucario.py
```

- [ ] **Step 2: Measure the mirror baseline (sanity ~50%)**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion champion_lucario.py \
  --deck decks/lucario_meta.csv --deck-champion decks/lucario_meta.csv --games 400
```
Expected: `"score"` ≈ 0.48–0.52 (it's the same agent — a near-tie confirms the mirror harness is unbiased). Record the exact number.

- [ ] **Step 3: Measure the Dragapult baseline**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion dragapult_agent.py \
  --deck decks/lucario_meta.csv --deck-champion decks/dragapult.csv --games 400
```
Record `"score"` and `"wilson_lb"` — this is the **DRAGAPULT_BASELINE** that later tasks must improve or hold.

- [ ] **Step 4: Log the baselines**

Append to `autoresearch/log/experiments.md`:
```
POLICY-DEEPEN baselines (2026-06-19): mirror(self vs frozen v4)=<step2 score>, dragapult=<step3 score, wilson_lb>. Gate: keep change iff mirror LB>=0.48 AND dragapult >= baseline-noise.
```

- [ ] **Step 5: Commit**

```bash
git add autoresearch/champion_lucario.py autoresearch/log/experiments.md
git commit -m "test: freeze v4 mirror champion + record meta baselines"
```

---

### Task 3: Prize-trade math (highest-priority policy diff)

**Files:**
- Modify: `autoresearch/agent_lucario.py` (the `_score_main` ATTACK branch + a new helper)

**Interfaces:**
- Consumes: `_eff_damage`, `_CARD`, the `W` table from `agent_lucario.py`.
- Produces: behavior change only (no new public symbols).

**Hypothesis:** Mega Lucario ex gives up **3** prizes on KO. When an attack with the mega-ex does NOT take a game-winning prize and we are not behind on the prize race, the agent should prefer a cheaper / non-megaEx attacker so it doesn't hand the opponent a 3-prize swing. The gauntlet decides if this helps.

- [ ] **Step 1: Add a prize-aware penalty helper (inline assertion test first)**

Create a scratch check `/tmp/check_prize.py`:
```python
def mega_penalty(is_mega_ex, is_lethal_win, my_prize_left, op_prize_left):
    # penalise exposing the 3-prize mega when it does NOT win the game and
    # we are not behind (we have <= the opponent's remaining prizes)
    if is_mega_ex and not is_lethal_win and my_prize_left <= op_prize_left:
        return 700
    return 0

assert mega_penalty(True, False, 3, 4) == 0      # behind on prizes -> swing freely
assert mega_penalty(True, False, 4, 4) == 700     # even -> protect the mega
assert mega_penalty(True, True, 4, 4) == 0        # lethal win -> always take it
assert mega_penalty(False, False, 4, 4) == 0      # non-mega -> no penalty
print("ok")
```
Run: `python3 /tmp/check_prize.py` → Expected: `ok`

- [ ] **Step 2: Wire the penalty into the ATTACK branch of `_score_main`**

In `autoresearch/agent_lucario.py`, inside the `elif t == int(OptionType.ATTACK):` block, after `dmg` and the KO/Crustle logic compute `s`, subtract the penalty when the attacker is the mega-ex and the attack is not a game-winning KO:
```python
            # prize-trade: don't expose the 3-prize mega-ex unless it wins or we're behind
            if my_active is not None and _CARD.get(my_active.id) and _CARD[my_active.id].megaEx:
                is_lethal_win = (opp_active is not None and dmg >= opp_active.hp
                                 and len(op.prize) <= 1)
                if not is_lethal_win and len(my.prize) <= len(op.prize):
                    s -= 700
```
(Place it after `s` is assigned in that branch; `my_active`, `opp_active`, `my`, `op`, `dmg` are already in scope.)

- [ ] **Step 3: Crash-safety smoke (40-game self-play, 0 errors)**

```bash
cd ~/Documents/Github/kaggle/pokemon
./autoresearch/run.sh --challenger agent_lucario.py --champion agent_lucario.py \
  --deck decks/lucario_meta.csv --games 40
```
Expected: `"games": 40`, no traceback.

- [ ] **Step 4: Gauntlet — mirror vs frozen champion (the test)**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion champion_lucario.py \
  --deck decks/lucario_meta.csv --deck-champion decks/lucario_meta.csv --games 600
```
Record `"score"` and `"wilson_lb"`. PASS condition: `wilson_lb >= 0.48` (ideally score > 0.52, meaning the change beats the frozen v4 in the mirror).

- [ ] **Step 5: Gauntlet — Dragapult**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion dragapult_agent.py \
  --deck decks/lucario_meta.csv --deck-champion decks/dragapult.csv --games 400
```
PASS condition: `score >= DRAGAPULT_BASELINE - 0.03` (improves or holds).

- [ ] **Step 6: Keep or revert + log**

If BOTH gates pass: keep. Append to `experiments.md`:
```
PRIZE-TRADE: mega-ex penalty when not lethal-win and not behind. mirror=<s4>, dragapult=<s5>. KEPT.
```
then commit:
```bash
git add autoresearch/agent_lucario.py autoresearch/log/experiments.md
git commit -m "feat(agent): prize-trade penalty on exposing the mega-ex"
```
If EITHER gate fails: revert and log the rejection:
```bash
git checkout autoresearch/agent_lucario.py
# append a REJECT line with the numbers to experiments.md, then commit just the log
```

---

### Task 4: Attack sequencing (Mega Brave vs the cheap attack)

**Files:**
- Modify: `autoresearch/agent_lucario.py` (ATTACK branch attack-value comparison)

**Interfaces:**
- Consumes: `_eff_damage`, `_ATK`, `_CARD`.
- Produces: behavior change only.

**Hypothesis:** Mega Lucario ex has a cheap 1-energy attack (~130) and Mega Brave (983, ~270 / 2 energy). The agent should choose the attack that KOs the target for the fewest resources, and reach for Mega Brave only when the cheaper attack does NOT KO. The engine offers each attack as a separate ATTACK option with `attackId`; we currently score them only by raw effective damage, which over-uses Mega Brave.

- [ ] **Step 1: Add a "cheapest lethal" preference in the ATTACK branch**

In the `elif t == int(OptionType.ATTACK):` block, after computing `dmg`, bias toward a KO that uses fewer energy: add a small bonus when this attack KOs AND its energy cost (len of the attack's energies) is the minimum among the offered KOing attacks. Concretely, compute the attack's cost and reward low-cost KOs:
```python
            if opp_active is not None and dmg >= opp_active.hp:
                cost = len(_ATK_COST.get(o.attackId, []))   # see Step 2 for _ATK_COST
                s += max(0, 300 - 100 * cost)                # cheaper KO preferred
```

- [ ] **Step 2: Add the attack-cost lookup next to `_ATK`**

Near the top of `agent_lucario.py` where `_ATK = {a.attackId: a.damage ...}` is defined, add:
```python
from cg.api import all_attack as _all_attack_for_cost   # noqa
_ATK_COST = {a.attackId: a.energies for a in _all_attack_for_cost()}
```
(Or reuse the existing `all_attack` import and add `_ATK_COST = {a.attackId: a.energies for a in all_attack()}` beside `_ATK`.)

- [ ] **Step 3: Crash-safety smoke**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion agent_lucario.py \
  --deck decks/lucario_meta.csv --games 40
```
Expected: `"games": 40`, no traceback.

- [ ] **Step 4: Mirror gauntlet (test)**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion champion_lucario.py \
  --deck decks/lucario_meta.csv --deck-champion decks/lucario_meta.csv --games 600
```
PASS: `wilson_lb >= 0.48`.

- [ ] **Step 5: Dragapult gauntlet**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion dragapult_agent.py \
  --deck decks/lucario_meta.csv --deck-champion decks/dragapult.csv --games 400
```
PASS: `score >= DRAGAPULT_BASELINE - 0.03`.

- [ ] **Step 6: Keep or revert + log + commit** (same procedure as Task 3 Step 6, message `feat(agent): cheapest-lethal attack sequencing`).

---

### Task 5: Setup / mulligan quality

**Files:**
- Modify: `autoresearch/agent_lucario.py` (`_score_context` SETUP/TO_BENCH branches)

**Interfaces:**
- Consumes: `_CARD`, `field` counts in `_score_context`.
- Produces: behavior change only.

**Hypothesis:** Better opening setup (always secure a Riolu line + a draw engine, avoid dead benches) reduces brick turns that lose the mirror. Refine the existing SETUP_ACTIVE / SETUP_BENCH / TO_BENCH scoring so the Lucario line and one draw-support are prioritized before redundant copies.

- [ ] **Step 1: Strengthen the bench-setup ordering**

In `_score_context`, in the `SETUP_BENCH_POKEMON`/`TO_BENCH` branch, ensure the first Riolu and first draw-support (Solrock/Lunatone) outrank duplicates. Replace the existing scores with:
```python
                    if card.id == RIOLU:
                        s = 130 - 30 * field.get(RIOLU, 0)
                    elif card.id == SOLROCK:
                        s = 100 if field.get(SOLROCK, 0) == 0 else -5
                    elif card.id == LUNATONE:
                        s = 95 if field.get(LUNATONE, 0) == 0 else -5
                    elif card.id == MAKUHITA:
                        s = 70 if field.get(MAKUHITA, 0) == 0 else 15
```

- [ ] **Step 2: Crash-safety smoke**

```bash
./autoresearch/run.sh --challenger agent_lucario.py --champion agent_lucario.py \
  --deck decks/lucario_meta.csv --games 40
```
Expected: `"games": 40`, no traceback.

- [ ] **Step 3: Mirror gauntlet (test)** — same command as Task 4 Step 4. PASS: `wilson_lb >= 0.48`.

- [ ] **Step 4: Dragapult gauntlet** — same command as Task 4 Step 5. PASS: `score >= DRAGAPULT_BASELINE - 0.03`.

- [ ] **Step 5: Keep or revert + log + commit** (message `feat(agent): tighter opening bench setup`).

---

### Task 6: Boss targeting for tempo

**Files:**
- Modify: `autoresearch/agent_lucario.py` (`PLAY` branch, Boss's Orders scoring — non-Crustle path)

**Interfaces:**
- Consumes: `op` (opponent state), `_CARD`.
- Produces: behavior change only.

**Hypothesis:** In the mirror/Dragapult, dragging the opponent's most-developed benched attacker (highest energy count) and KOing it wins the prize race. Currently Boss is played whenever `op.bench` exists; make it target-aware so it only fires when there is a worthwhile (energized) bench target.

- [ ] **Step 1: Make the non-Crustle Boss path target-aware**

In the `elif card.id == BOSS_ORDERS:` block, replace the `elif op.bench: s = W["boss"]` arm with:
```python
                elif any(b is not None and len(b.energies) >= 1 for b in op.bench):
                    s = W["boss"]
                else:
                    s = -1
```
(Keep the existing Crustle-specific arm above it unchanged.)

- [ ] **Step 2: Crash-safety smoke** — same as Task 5 Step 2. Expected: `"games": 40`, no traceback.

- [ ] **Step 3: Mirror gauntlet (test)** — Task 4 Step 4 command. PASS: `wilson_lb >= 0.48`.

- [ ] **Step 4: Dragapult gauntlet** — Task 4 Step 5 command. PASS: `score >= DRAGAPULT_BASELINE - 0.03`.

- [ ] **Step 5: Keep or revert + log + commit** (message `feat(agent): target-aware Boss for prize tempo`).

---

### Task 7: Energy / retreat discipline

**Files:**
- Modify: `autoresearch/agent_lucario.py` (`ATTACH` branch — avoid over-loading a charged attacker)

**Interfaces:**
- Consumes: `_is_main_attacker`, the active/bench Pokémon energy counts.
- Produces: behavior change only.

**Hypothesis:** Pouring a 3rd+ energy onto an already-attack-ready attacker wastes tempo; energy is better spread to a backup. Lower the attach score for a Pokémon that already has enough energy for its main attack.

- [ ] **Step 1: De-prioritize over-attaching**

In the `elif t == int(OptionType.ATTACH):` block (non-Crustle path), after the existing main-attacker bonus, reduce the score when the target already has ≥2 energy:
```python
                elif _is_main_attacker(pk.id):
                    s += W["energy_need"]
                    if len(pk.energies) >= 2:
                        s -= 1500          # already attack-ready -> spread energy instead
```

- [ ] **Step 2: Crash-safety smoke** — Task 5 Step 2. Expected: `"games": 40`, no traceback.

- [ ] **Step 3: Mirror gauntlet (test)** — Task 4 Step 4 command. PASS: `wilson_lb >= 0.48`.

- [ ] **Step 4: Dragapult gauntlet** — Task 4 Step 5 command. PASS: `score >= DRAGAPULT_BASELINE - 0.03`.

- [ ] **Step 5: Keep or revert + log + commit** (message `feat(agent): energy-spread discipline`).

---

### Task 8: Submit the deepened agent + read the ladder

**Files:**
- Modify: `submission_lucario/main.py` (sync from the improved `agent_lucario.py`)

**Interfaces:**
- Consumes: the final kept `agent_lucario.py`.

- [ ] **Step 1: Sync the agent into the submission bundle**

```bash
cd ~/Documents/Github/kaggle/pokemon
cp autoresearch/agent_lucario.py submission_lucario/main.py
```

- [ ] **Step 2: Validate the bundle self-plays cleanly (0 crashes)**

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/app -w /app/autoresearch \
  -e PYTHONPATH=/app/sdk python:3.11-slim \
  python eval.py --challenger ../submission_lucario/main.py --champion ../submission_lucario/main.py \
  --deck ../submission_lucario/deck.csv --games 40
```
Expected: `"games": 40`, no traceback.

- [ ] **Step 3: Build the tarball (main.py at top level) and submit**

```bash
cd ~/Documents/Github/kaggle/pokemon/submission_lucario
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true
rm -f ../submission_lucario_v5.tar.gz
tar -czf ../submission_lucario_v5.tar.gz *
tar -tzf ../submission_lucario_v5.tar.gz | grep -vE '^cg/' | grep -v '/$'   # expect: deck.csv, main.py
kaggle competitions submit -c pokemon-tcg-ai-battle -f ../submission_lucario_v5.tar.gz \
  -m "Lucario v5: prize-trade + sequencing + setup + Boss tempo + energy discipline (real-meta tuned)"
```
Expected: `Successfully submitted`.

- [ ] **Step 4: Read the ladder score (async; check after it processes)**

```bash
cd ~/Documents/Github/kaggle/pokemon
kaggle competitions submissions pokemon-tcg-ai-battle 2>&1 | grep -viE "^Warning" | head -3
```
Record the v5 `publicScore`. SUCCESS = score > 820.7. Append the final result to `experiments.md` and commit.

---

## Self-Review

**Spec coverage:**
- Dragapult opponent → Task 1. ✓
- Frozen mirror champion → Task 2. ✓
- Prize-trade math → Task 3. ✓
- Attack sequencing → Task 4. ✓
- Setup/mulligan → Task 5. ✓
- Boss tempo → Task 6. ✓
- Energy/retreat discipline → Task 7. ✓
- Validation gate (mirror LB≥0.48 AND Dragapult hold) → applied in Tasks 3–7. ✓
- Crash-safety preserved → smoke step in Tasks 3–7 + Task 8. ✓
- Ladder is final arbiter → Task 8. ✓
- Out of scope (MCTS/BC/Crustle/deck switch) → not present. ✓

**Placeholder scan:** No "TBD/TODO"; each policy diff shows concrete code; the only "extract from notebook" step (Task 1 Step 2) is a genuine action with a verification command, not a placeholder. ✓

**Type/name consistency:** `_score_main`, `_score_context`, `_eff_damage`, `_ATK`, `_CARD`, `_is_main_attacker`, `W`, `BOSS_ORDERS`, `RIOLU/SOLROCK/LUNATONE/MAKUHITA`, `read_deck`, `agent` — all match the existing `agent_lucario.py` symbols and the `eval.py` CLI (`--challenger/--champion/--deck/--deck-champion/--games`). ✓
