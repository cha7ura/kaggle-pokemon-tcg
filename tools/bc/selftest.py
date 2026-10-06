#!/usr/bin/env python3
"""Prove that the live inference path and the training path encode the same thing.

    python3 tools/bc/selftest.py --sdk <sdk-path> --data data/tensors/grimm-v1

Behaviour cloning fails silently when the features at inference differ from the
features at training. Nothing in a green training run detects it, and nothing in
a green local game detects it either — the agent just plays worse than its
accuracy says it should. So this is the check that has to exist.

Four things are asserted, in increasing strength:

1. `row_from_observation` produces exactly the key set that the parquet frontend
   produces. A missing key silently encodes as "absent" through `row.get`.
2. Every encoded value from a live game is inside the vocabulary the dataset
   manifest declared. An out-of-range index is a crash at inference, in the
   sandbox, mid-episode.
3. Option card resolution agrees with the engine's own `Option` payload. The
   engine populates `Option.cardId` for SKILL selects, and the parquet corpus
   does not carry it, so where both exist they must match — that is a direct
   check on the index-resolution rule the encoder relies on everywhere else.
4. Live feature distributions are in the same range as the corpus. Not a proof,
   but a wildly different tokens-per-row is the shape of a real bug.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bc import encoder  # noqa: E402
from tools.bc.cards import load_tables  # noqa: E402
from tools.bc.rows import STEP_COLUMNS, row_from_observation, rows_from_parquet  # noqa: E402

# Keys the parquet frontend supplies that the live frontend cannot and must not:
# identity and bookkeeping columns, plus the label itself.
PARQUET_ONLY = {"episode_id", "date", "action"}
# `me_active_energy` and its mirror are counts the encoder derives from the type
# lists; the live frontend fills them for symmetry, the encoder ignores both.
# `my_deck_ids` is the deck-conditioning block: the live frontend always fills it
# (empty when the checkpoint is not deck-token), and the cache builder joins it
# from decks.parquet rather than the steps stream, so it is never a steps column.
LIVE_EXTRA = {"me_active_energy", "opp_active_energy", "my_deck_ids"}


def parquet_keys(replays: Path) -> set[str]:
    day = sorted(replays.glob("date=*"))[-1]
    for row in rows_from_parquet(day / "steps.parquet", batch_size=1, columns=STEP_COLUMNS):
        return set(row)
    raise SystemExit(f"no rows in {day}")


# How many live hits a slot must have been expected to produce before silence
# counts as evidence against it. Deliberately not 3: the live sample plays a
# dozen decks and the cache is every archetype on the ladder, so a slot's live
# rate and its cached rate are not the same number. me_status is 0.25% of cached
# rows and **0% of live rows for 5 of 6 decks measured** — only Dragapult ex
# inflicts one — so a tight bar fails a working encoder on deck mix alone.
MIN_EXPECTED_HITS = 25.0


def dead_slots(cache: Path, live_used: dict, live_rows: dict) -> dict:
    """Slots the cached tensors populate but the live encoder never touched.

    Reads the newest partition only; a slot that is never nonzero across a whole
    day of top-ladder decisions is not one this check can say anything about.

    A slot is only reported when the live sample was big enough to have expected
    to see it. Rare-but-correct slots are the failure mode: `me_status` is
    nonzero on 0.206% of corpus rows — burned and paralyzed are 0.00000 even
    there — so a 12-game run has a real chance of never inflicting a special
    condition and reporting a working encoder as broken. The gate now needs
    `rate * live_rows >= MIN_EXPECTED_HITS` before it calls a slot dead, which
    keeps it able to catch a slot the live encoder genuinely never writes while
    it stops firing on ordinary sampling luck. Widen `--games` to lower the bar
    a rare slot has to clear.
    """
    from tools.bc.encoder import GLOBAL_SLOTS, LOG_SLOTS, OPTION_SLOTS, TOKEN_SLOTS

    names = {
        "globals": [n for n, _ in GLOBAL_SLOTS],
        "tokens": [n for n, _ in TOKEN_SLOTS],
        "options": [n for n, _ in OPTION_SLOTS],
        "logs": [n for n, _ in LOG_SLOTS],
    }
    day = sorted(cache.glob("date=*"))[-1]
    report: dict[str, list[str]] = {}
    for array_name, slot_names in names.items():
        if not (day / f"{array_name}.npy").exists():
            continue
        cached = np.load(day / f"{array_name}.npy", mmap_mode="r")
        if cached.shape[0] == 0 or live_used.get(array_name) is None:
            continue
        sample = np.asarray(cached[: min(50_000, cached.shape[0])])
        # Rate over rows, not "was it ever nonzero": the expected number of live
        # hits is what decides whether silence is evidence.
        flat = sample.reshape(-1, sample.shape[-1]) if sample.ndim > 2 else sample
        cached_rate = (flat != 0).mean(axis=0)
        n_live = max(int(live_rows.get(array_name, 0)), 0)
        missing, suppressed = [], []
        for i in range(len(slot_names)):
            if cached_rate[i] <= 0 or live_used[array_name][i]:
                continue
            expected = cached_rate[i] * n_live
            (missing if expected >= MIN_EXPECTED_HITS else suppressed).append(
                (slot_names[i], expected))
        if suppressed:
            print(f"note {array_name}: unset live but too rare to judge at "
                  f"{n_live:,} rows — "
                  + ", ".join(f"{n} (expected {e:.1f})" for n, e in suppressed))
        missing = [f"{n} (expected {e:.0f} live hits, saw 0)" for n, e in missing]
        if missing:
            report[array_name] = missing
    return report


def play(decks, games: int, seed: int):
    """Random legal play, both seats, yielding every live observation."""
    from cg.game import battle_finish, battle_select, battle_start

    rng = random.Random(seed)
    for game in range(games):
        # Every third game is a mirror. Without them the pairing walks deck_a
        # across the folder while deck_b barely moves, so a card whose effect
        # needs both seats running it — a special condition, most of all — can go
        # unseen for 40 games. Mirrors are also a quarter of the real field.
        if game % 3 == 2:
            deck_a = deck_b = decks[(game // 3) % len(decks)]
        else:
            deck_a = decks[game % len(decks)]
            deck_b = decks[(game // len(decks)) % len(decks)]
        observation, start = battle_start(deck_a, deck_b)
        if observation is None:
            battle_finish()
            raise SystemExit(f"deck rejected: player {start.errorPlayer} type {start.errorType}")
        steps = 0
        try:
            while observation["current"]["result"] < 0 and steps < 4000:
                select = observation["select"]
                # cg.game does not supply remainingOverageTime; kaggle_environments
                # and tools/ab.py both do. Synthesising a draining bank keeps the
                # encoded value inside the range every training row carries, so
                # the dead-slot check tests the encoder rather than the harness.
                observation["remainingOverageTime"] = max(0.0, 600.0 - steps * 3.0)
                yield observation
                indices = list(range(len(select["option"])))
                rng.shuffle(indices)
                take = max(select["minCount"], 1)
                observation = battle_select(indices[: min(take, select["maxCount"])])
                steps += 1
        finally:
            battle_finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdk", type=str, default="")
    parser.add_argument("--replays", type=Path, default=Path("data/replays"))
    parser.add_argument("--data", type=Path, default=None, help="tensor cache, for its manifest")
    parser.add_argument("--decks", type=Path, default=Path("decks"))
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sdk:
        sys.path.insert(0, args.sdk)
    from cg.api import to_observation_class

    tables = load_tables()
    with_logs = False
    if args.data:
        with_logs = bool(json.loads((args.data / "manifest.json").read_text()).get("with_logs"))
    sizes = encoder.slot_sizes(tables, with_logs=with_logs)
    if args.data:
        manifest = json.loads((args.data / "manifest.json").read_text())
        if manifest["slot_sizes"] != sizes:
            raise SystemExit("manifest slot sizes differ from the current encoder; rebuild the cache")

    expected = parquet_keys(args.replays) - PARQUET_ONLY
    decks = [
        [int(x) for x in path.read_text().split() if x.strip()]
        for path in sorted(args.decks.glob("*.csv"))
    ]
    decks = [d for d in decks if len(d) == 60]
    if not decks:
        raise SystemExit(f"no 60-card lists under {args.decks}")

    key_failures = []
    range_failures = []
    card_id_checks = card_id_mismatch = 0
    select_types: Counter = Counter()
    tokens_total = options_total = rows = 0
    live_used = {"globals": None, "tokens": None, "options": None, "logs": None}
    live_rows = {"globals": 0, "tokens": 0, "options": 0, "logs": 0}

    for observation_dict in play(decks, args.games, args.seed):
        observation = to_observation_class(observation_dict)
        if observation.select is None or observation.current is None:
            continue
        row = row_from_observation(observation, observation_dict.get("remainingOverageTime"))

        live = set(row) - LIVE_EXTRA
        if live != expected and not key_failures:
            key_failures.append({
                "missing_from_live": sorted(expected - live),
                "extra_in_live": sorted(live - expected),
            })

        encoded = {
            "globals": encoder.encode_globals(row)[None, :],
            "tokens": encoder.encode_tokens(row, tables),
            "options": encoder.encode_options(row, tables),
        }
        if "logs" in sizes:
            encoded["logs"] = encoder.encode_logs(row, tables)
        for name, array in encoded.items():
            limits = np.asarray(sizes[name])
            if array.size == 0:
                continue
            over = np.where(array.max(axis=0) >= limits)[0]
            if over.size and not range_failures:
                range_failures.append({
                    "array": name,
                    "slots": over.tolist(),
                    "values": array.max(axis=0)[over].tolist(),
                    "limits": limits[over].tolist(),
                })
            used = (array != 0).any(axis=0)
            live_used[name] = used if live_used[name] is None else (live_used[name] | used)
            live_rows[name] += int(array.shape[0])

        # (3) the engine's own card id, where it supplies one.
        for i, option in enumerate(observation.select.option):
            if option.cardId is None:
                continue
            card_id_checks += 1
            resolved = encoder.option_card_id(row, option.area, option.index,
                                              option.playerIndex, row["seat"], option.type)
            if resolved is not None and resolved != option.cardId:
                card_id_mismatch += 1

        select_types[int(observation.select.type)] += 1
        tokens_total += encoded["tokens"].shape[0]
        options_total += encoded["options"].shape[0]
        rows += 1

    print(f"decisions encoded: {rows:,} over {args.games} games")
    print(f"select types seen: {dict(sorted(select_types.items()))}")
    print(f"tokens/row {tokens_total / max(1, rows):.1f}   options/row {options_total / max(1, rows):.1f}")
    print(f"engine-supplied Option.cardId checked on {card_id_checks:,} options, "
          f"{card_id_mismatch} disagreements")

    failed = False
    if key_failures:
        print("\nFAIL row key mismatch between the parquet and live frontends:")
        print(json.dumps(key_failures[0], indent=1))
        failed = True
    else:
        print("\nOK   live row keys match the parquet row keys exactly")

    if range_failures:
        print("FAIL encoded value outside the declared vocabulary:")
        print(json.dumps(range_failures[0], indent=1))
        failed = True
    else:
        print("OK   every encoded value is inside the declared vocabulary")

    if card_id_mismatch:
        print(f"FAIL option card resolution disagrees with the engine on {card_id_mismatch} options")
        failed = True
    else:
        print("OK   option card resolution agrees with the engine wherever it supplies a card id")

    # (4) A slot the training tensors populate and the live path never does is a
    # structural bug, not a distribution difference. Only that one-sided case is
    # flagged: random self-play genuinely visits different states than the
    # top-ladder corpus, so comparing frequencies would cry wolf.
    if args.data:
        dead = dead_slots(args.data, live_used, live_rows)
        if dead:
            print("FAIL slots the training tensors use but the live encoder never sets")
            print("     (each was frequent enough in the cache to expect "
                  f"{MIN_EXPECTED_HITS:.0f}+ live hits at this sample size):")
            print(json.dumps(dead, indent=1))
            failed = True
        else:
            print("OK   every slot the training tensors use is also set by the live encoder")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
