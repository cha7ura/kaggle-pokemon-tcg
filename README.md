# Kaggle Pokémon TCG AI Battle — START HERE (for Claude)

Orientation for a fresh session (e.g. on another PC). Read this first, then check live state.

## Goal
Maximize **ladder score** in Kaggle comp `pokemon-tcg-ai-battle`. Agent contract:
`agent(obs_dict) -> list[int]`; deck phase returns 60 card IDs. Offline, stdlib-or-NumPy at
submission. The compiled engine `sdk/cg/libcg.so` (linux/amd64) is the SAME engine the competition
runs — **mechanics are ground-truth, no mechanics gap**. Only opponents/matchmaking differ offline.

## First commands in a new session
```bash
kaggle competitions submissions -c pokemon-tcg-ai-battle | head -5   # what's live now
python3 -c "import sqlite3;print(sqlite3.connect('replays.sqlite').execute('select count(*) from replays').fetchone()[0])"
```
Only the **2 most-recent** submissions are ACTIVE on the ladder; older ones freeze. To keep a
proven agent live you must RESUBMIT it.

## The ONE proven lever: meta-counter (rock-paper-scissors)
Dragapult > Trevenant > Alakazam > Dragapult. Read replays → find the rising archetype → ship its
**counter**, do NOT mirror it. This is the only thing that has moved our score.
- Best result this work: new #1 was Trevenant (Yushin Ito) → shipped **Dragapult counter = ~900–978**.
- Mirroring the meta (CardPolicy on the #1 Trev deck) **flopped at 460**.

## What's EXHAUSTED (don't re-grind — all empirically flat/worse)
- **Gameplay policy**: feature-limited plateau. top-1 0.488→0.507→0.518→0.524(+40% data)→0.527→0.529.
  More data ≈ +½pp and shrinking. Bottleneck is FEATURES, not rows. RF≈GBM≫linear.
- **More data**: marginal for the policy. DOES help the deck/meta read (worth fresh full-day dumps).
- **Forward search**: validated WORSE (bad opponent determinization). Gate off, code kept.
- **Self-play**: gate failed, no gain (no policy-improvement operator without working search).
- **Deck-harvest / sim oracle**: pilot wall — sim does NOT predict ladder (trev_ga1 +0.057 offline → −86 live).

## Pipeline (all tools/imitation/, stdlib+numpy, self-checked)
- Card KB: `build_card_text.py`→`cards_engine.json`; `build_effect_flags.py`→`cards.json`; `fetch_card_wiki.py`.
- Features: `threat.py`, `deck_tracker.py`, `card_features.py`, `featurize.py` (131-dim row).
- Train: `build_decisions.py` (winner-filtered)→decisions.npz; `train_card_policy.py` (XGBoost)→card_policy.npz.
- Agent: `gameplay_policy.py` `CardPolicy(deck, model).act(obs)` (lethal gate + never-crash fallback).
- Data: `fetch_dataset.py N` (BFS top subs, keep episode if max(Elo)≥1000, pull+ingest-dedup);
  `fetch_day.py DATE CAP` (per-file daily, but ~96% miss — files often unpublished).

## Running the engine (docker only — libcg.so is linux/amd64)
```
docker run --rm --platform linux/amd64 -v $PWD:/app -w /app/autoresearch \
  -e PYTHONPATH=/app/sdk python:3.11-slim python <script>
```

## Submitting
Bundle = `cardpol/` package + cards.json + card_policy.npz + cg/ + deck.csv + robust `main.py`
(must detect AGENT_DIR WITHOUT `__file__` — Kaggle runs via exec(); that was the v16/v18 crash).
Docker-verify deck phase returns 60 + no crash, then `kaggle competitions submit`.
Examples: `submission_cardpolicy_yushin/`, banked `submission_dragapult_v19.tar.gz`.

## Data note (other PC)
git has CODE ONLY. `replays.sqlite` (~745M, Elo≥1000 games), `tools/imitation/data/*.npz`,
`cards_engine.json`, and `sdk/cg/libcg.so` are gitignored — they ride along only if you COPY the
folder (not `git clone`).

## Likely next steps
1. Check live slots; if the meta shifted, re-run replay analysis → ship the counter (the playbook).
2. To improve the policy you must add **new features** (the plateau is features, not data/model) —
   or build a working search/determinization operator (the missing piece for search + self-play).
3. Fresh full-day data dumps help the **deck/meta read** (not the policy ceiling).
