# Data notes & checklist — replays.sqlite and the meta pipeline

Purpose: a repeatable procedure so we don't repeat the mistakes this project already hit
(stale table trusted as current; compression format changed silently; classifier keyed on the
wrong card; win-rate tables quoted on partial coverage).

## What the data IS (as of last check)

- **`replays.sqlite`** — the whole corpus. Grew 6,084 → 88,353 games in one data drop; expect it to
  keep growing. Table `replays(episode_id TEXT PK, team0, team1, reward0, reward1, blob, source)`.
  - `source`: `leader` (real ladder games, 88,042) + `ours` (311 — NOTE: these are **scouting games of
    The Debauchery Tea Party, the #1 team**, not our own games. Our own games are in `leader` under
    team name `cha7ura`).
  - `blob`: **LZMA/XZ-compressed** JSON (magic bytes `\xfd7zXZ`). It was **zlib** in the 6k-era dump —
    the format CHANGED. Always sniff magic bytes; try lzma then zlib. Decoded JSON has
    `steps[]`, each `step[seat]['observation']['current']` = game state with `players[seat]` zones
    (`active/bench/hand/discard/deck/prize/lostZone`), each card a dict with `id`/`cardId`.
  - `reward0/reward1`: +1 win / -1 loss / 0 draw. **~45 games (0.1%) have a NULL reward** = the agent
    crashed (ERROR) or hung (TIMEOUT); the DONE side won. Handle NULLs (winner = the non-null side).

- **`replay_field(episode_id, player, archetype, deck_sig, team)`** — a PRECOMPUTED table.
  - `deck_sig` = underscore-joined sorted card IDs = an **EXACT full 60-card decklist** per player-game.
    Use it instead of decoding blobs when the episode is covered.
  - **CRITICAL: it is STALE / partial.** Covers only episodes 80.1M–82.1M (59,440 of 88,353 = 67%),
    and **0% of the recent 25%**. The current meta (e.g. Grimmsnarl at ~35% share) is NOT in it.
  - `archetype` column uses an **old map** — 0% agreement with a correct classifier on
    SolrockLunatone / HopTrevenant / Grimmsnarl / ToolboxEx / Archaludon / Chandelure. Do NOT trust it.

- **`autoresearch/cards_full.csv`** — card DB: `cardId,name,cardType,hp,type,weakness,resistance,retreat,basic,stage1,stage2,ex`.
- **`/tmp/lb/…zip`** — leaderboard download (`kaggle competitions leaderboard -c pokemon-tcg-ai-battle --download`).
  Header has a BOM — read with `utf-8-sig`.

## CHECKLIST when new data is added

1. **Re-measure size & range.** `COUNT(*)`, `MIN/MAX(CAST(episode_id AS INTEGER))` per source.
   Did the corpus grow? What's the new max episode?
2. **Sniff the blob compression** on a fresh row (magic bytes). Don't assume last drop's codec.
3. **Check precomputed tables for staleness.** For `replay_field` (or any cached table): what episode
   range does it cover vs the full `replays` range? If it misses the recent tail, it represents the
   OLD meta — do not quote it as current.
4. **State the coverage denominator in every result.** "N games / which episode window". A win-rate on
   the 67% old-era join is a different claim than on the full/recent corpus. Never say "full 88k" when
   you computed on 59k.
5. **Rebuild the recent-era deck registry** for episodes NOT in `replay_field`: decode blobs, take the
   UNION of each player's cards across ALL steps (a single early snapshot is NOT a full 60 — recent
   games only reveal the deck gradually). Cap non-basic-energy at 4 copies; basic energy uncapped.
6. **Validate every reconstructed deck against construction rules:** exactly 60 cards; ≤4 copies of any
   non-basic-energy card; sane Pokémon/Trainer/Energy split (typical ~13-20 / 27-39 / 7-12).
   Under-60 means rare 1-ofs didn't surface → sample more games.
7. **Re-derive the archetype classifier from data, not the old map.** Key on the whole evolution LINE
   (the BASIC, e.g. Marnie's Impidimp 646), not just the final ex (648) — the ex often isn't in the
   first N steps, so line-blind classifiers dump decks into OTHER. Keep OTHER% low; if OTHER is large,
   inspect its most-common Pokémon and add signatures.
8. **Recompute prevalence AND win-rate on a TIME-SPLIT** (older vs recent episode windows). The meta
   turns over fast (Dragapult 12.6%→0.4%; Grimmsnarl 0→~35%). A single pooled number hides the trend.
9. **Refresh live standings** via Kaggle API (`competitions leaderboard/submissions`). Note: uploads go
   to `storage.googleapis.com` (denylisted in sandbox) — pushes/submits run from the user's Mac, but
   read-only API calls work here.
10. **Re-sync the shipped decks against the current meta.** Are our two active slots still on decks with
    ≥50% recent win-rate and non-trivial share? (This project shipped Dragapult AFTER it fell to 0.4%.)

## Known gotchas (already paid for)
- Sim ≠ ladder: offline `deck_gate` win-rate does NOT predict live score (MegaStarmie 58% EWR → 541 live).
  Offline tells you WHAT to try; only a live submission confirms.
- Ladder score is noise-dominated: same agent file scored 600–1006 across resubmits (±~120, 1σ).
- `ProcessPoolExecutor` is blocked in the sandbox (semaphore syscall denied) — decode single-process or
  in background cells; use SQL wherever possible instead of mass blob decode.
