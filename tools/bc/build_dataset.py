#!/usr/bin/env python3
"""Parquet -> packed int16 tensors for behaviour cloning.

    python3 tools/bc/build_dataset.py --sdk <sdk-path> --out data/tensors/grimm-v1

Runs the encoder once per decision row and caches the result, so training reads
memmapped arrays instead of re-parsing parquet every epoch. Output goes to a
git-ignored directory: it is derived from replay data and there is no reason to
carry it in the repo.

Storage is ragged. Padding every row to the corpus maximum of 130 options would
cost about 25x the space for a mean of 7 options per row, and the model masks
per batch anyway.

Filtering controls used by the production pipeline:

* Keep steps whose acting seat played the pilot deck (`--deck-card`, default 648
  = Marnie's Grimmsnarl ex).
* Keep only seats that won, unless `--all-seats`. With `--all-seats`,
  `--loss-weight` scales the loss on seats that lost: 0.0 reproduces
  winners-only, 1.0 is the unweighted extreme that measured as a regression
  during the campaign, and the useful range is in between.
* `--with-logs` adds `obs.logs` as a second token stream. Off by default so the
  delta is measurable on its own.
* Drop forced rows — the engine left no choice, the guard shell answers them
  without inference, and counting them inflates accuracy by ~8 points.
* Temporal holdout by date, never a random row split: adjacent decisions in one
  game are near-duplicates and the meta drifts week to week.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bc import encoder  # noqa: E402
from tools.bc.cards import load_tables  # noqa: E402
from tools.bc.rows import STEP_COLUMNS, rows_from_parquet  # noqa: E402

ARRAY_NAMES = (
    "globals", "tokens", "tokens_off", "options", "options_off",
    "targets", "targets_off", "meta", "episode_id", "weights",
)


def encoder_fingerprint() -> str:
    """Hash the code that produced the tensors.

    Training refuses a cache whose fingerprint differs from the current source.
    Silently training on features from an older encoder is the expensive kind of
    mistake this repo has already paid for once in packaging.
    """
    digest = hashlib.sha256()
    for name in ("encoder.py", "cards.py", "rows.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def seat_filter(day_dir: Path, deck_card: int, winners_only: bool,
                min_avg_score: float = 0.0) -> dict[int, bool]:
    """Kept `episode_id * 2 + seat` keys, mapped to whether that seat won.

    The win flag is what `--loss-weight` needs. Winners-only training throws it
    away by construction; keeping every seat and *weighting* it is the delta
    The retained training recipe always specified this control; the unweighted extreme
    was not used by the final agents.

    `min_avg_score` drops episodes whose `games.avg_score` is below the floor.
    That column is the mean ladder score of the two seats, so the filter keeps
    games where **both** players are strong — it cannot select one strong seat
    out of a lopsided pairing, because the corpus records no per-seat score.
    Being explicit about that matters: the effect is "high-level games", not
    "games by high-level players".
    """
    import pyarrow.parquet as pq

    decks = pq.read_table(day_dir / "decks.parquet", columns=["episode_id", "seat", "deck"])
    games = pq.read_table(day_dir / "games.parquet",
                          columns=["episode_id", "winner_seat", "avg_score"])
    winner = dict(zip(games["episode_id"].to_pylist(), games["winner_seat"].to_pylist()))
    score = dict(zip(games["episode_id"].to_pylist(), games["avg_score"].to_pylist()))

    keep: dict[int, bool] = {}
    for episode_id, seat, deck in zip(
        decks["episode_id"].to_pylist(), decks["seat"].to_pylist(), decks["deck"].to_pylist()
    ):
        if deck_card >= 0 and deck_card not in deck:
            continue
        if min_avg_score > 0.0:
            value = score.get(episode_id)
            # A null score is unknown, not high. Dropping it keeps the filter
            # honest at the cost of a few rows.
            if value is None or value < min_avg_score:
                continue
        won = winner.get(episode_id) == seat
        if winners_only and not won:
            continue
        keep[episode_id * 2 + seat] = won
    return keep


def recency_weight(date: str, newest: str, halflife_days: float) -> float:
    """Exponential decay on a row's age, in days, relative to the newest partition.

    The newest day weighs 1.0 and every `halflife_days` back halves it. The
    corpus spans 48 days and the field turned over on 2026-07-31
    (`agents/README.md`), so a cache built flat over all of it is majority
    Grimmsnarl-mirror data from a meta that no longer exists. This is the cheap
    correction: same rows, less say for the dead ones.

    `0.0` disables it and returns 1.0, which is what every run before
    2026-08-05 did implicitly.
    """
    if halflife_days <= 0.0:
        return 1.0
    from datetime import date as _date
    age = (_date.fromisoformat(newest) - _date.fromisoformat(date)).days
    if age <= 0:
        return 1.0
    return float(0.5 ** (age / halflife_days))


def _survey(day_dir: Path, keep_keys) -> tuple[int, int]:
    """Rows the filter admits, and how many of them are forced.

    Counted on four narrow columns rather than while iterating the encoded rows,
    because the encoder never sees a filtered-out row: filtering happens on the
    Arrow batch, before any Python dict exists.
    """
    import pyarrow.parquet as pq

    seen = forced = 0
    reader = pq.ParquetFile(day_dir / "steps.parquet")
    columns = ["episode_id", "seat", "n_options", "min_count", "max_count"]
    for batch in reader.iter_batches(batch_size=65536, columns=columns):
        n = batch.column("n_options").to_numpy()
        lo = batch.column("min_count").to_numpy()
        hi = batch.column("max_count").to_numpy()
        mask = np.ones(len(n), dtype=bool)
        if keep_keys is not None:
            key = (batch.column("episode_id").to_numpy().astype(np.int64) * 2
                   + batch.column("seat").to_numpy().astype(np.int64))
            mask = np.isin(key, keep_keys)
        seen += int(mask.sum())
        forced += int((mask & ((n <= 1) | ((lo == hi) & (hi == n)))).sum())
    return seen, forced


def seat_decks(day_dir: Path, keep_keys) -> dict[int, list[int]]:
    """`episode_id * 2 + seat` -> that seat's own 60-card list.

    **The acting seat's, and only the acting seat's.** The agent knows its own
    list at inference because it ships `deck.csv`; it never sees the opponent's,
    and joining `decks.deck` for the other seat would leak hidden information.
    The key is built from this table's own
    `seat` column and the encoder reads the result under `my_deck_ids`, so no
    path here can reach the opponent's row.
    """
    import pyarrow.parquet as pq

    wanted = None if keep_keys is None else set(keep_keys.tolist())
    decks = pq.read_table(day_dir / "decks.parquet", columns=["episode_id", "seat", "deck"])
    out: dict[int, list[int]] = {}
    for episode_id, seat, deck in zip(
        decks["episode_id"].to_pylist(), decks["seat"].to_pylist(), decks["deck"].to_pylist()
    ):
        key = episode_id * 2 + seat
        if wanted is None or key in wanted:
            out[key] = [int(c) for c in deck]
    return out


def build_day(job: dict) -> dict:
    # A dict, not a tuple. This used to be eight positional fields and every new
    # flag was a chance to transpose two of them silently — the kind of bug that
    # produces a cache that trains fine and means something else.
    day_dir, out_dir = Path(job["day_dir"]), Path(job["out_dir"])
    deck_card = job["deck_card"]
    winners_only = job["winners_only"]
    keep_forced = job["keep_forced"]
    sdk = job["sdk"]
    loss_weight = job["loss_weight"]
    with_logs = job["with_logs"]
    deck_tokens = job["deck_tokens"]
    value = job["value"]
    min_avg_score = job["min_avg_score"]
    decay = recency_weight(day_dir.name.replace("date=", ""), job["newest_date"],
                           job["recency_halflife"])
    if sdk:
        sys.path.insert(0, sdk)
    tables = load_tables()

    # An empty key set is a real answer — a day before the archetype existed —
    # not "no filter". Testing `if keys` here would silently keep every seat on
    # exactly the days the filter was supposed to empty.
    # `loss_weight` belongs in this condition even though it drops no seats: the
    # win flag lives in `keys`, and without it every seat is weighted 1.0. The
    # combination that exposed it is `--deck-card -1 --all-seats
    # --loss-weight 0.2`, where nothing else forces a filter — it built a cache
    # that looked right, trained fine, and applied no weighting at all.
    # `value` is here for the same reason: the critic target IS the win flag, and
    # `--value --deck-card -1 --loss-weight 1.0` would otherwise leave every seat
    # `won=True` and the target a constant 1.0 — a value head that learns nothing.
    filtering = (deck_card >= 0 or winners_only or min_avg_score > 0.0
                 or loss_weight != 1.0 or value)
    keys = seat_filter(day_dir, deck_card, winners_only, min_avg_score) if filtering else None
    keep_keys = np.sort(np.fromiter(keys, dtype=np.int64, count=len(keys))) if keys is not None else None
    started = time.perf_counter()

    seen, forced_dropped = _survey(day_dir, keep_keys)
    if keep_keys is not None and keep_keys.size == 0:
        seen, forced_dropped = 0, 0

    # The acting seat's own 60 for the deck block, joined only when the cache is
    # built with --deck-tokens; otherwise no deck tokens are emitted at all.
    decks = seat_decks(day_dir, keep_keys) if deck_tokens else None
    deck_truncated = 0

    globals_buf: list[np.ndarray] = []
    token_buf: list[np.ndarray] = []
    option_buf: list[np.ndarray] = []
    target_buf: list[np.ndarray] = []
    meta_buf: list[tuple[int, int, int, int, int, int]] = []
    episode_buf: list[int] = []
    weight_buf: list[float] = []
    value_buf: list[float] = []
    log_buf: list[np.ndarray] = []
    token_off = [0]
    option_off = [0]
    target_off = [0]
    log_off = [0]
    kept = 0

    rows = rows_from_parquet(
        day_dir / "steps.parquet",
        columns=STEP_COLUMNS,
        keep_keys=keep_keys,
        drop_forced=not keep_forced,
    )
    for row in rows:
        if decks is not None:
            # Keyed on the row's OWN seat. A row whose seat is missing from the
            # join gets no deck block rather than someone else's — the deck
            # tokens would otherwise be silently wrong instead of absent, and
            # absent is a state the encoder already handles. Never the opponent's
            # list: that would leak hidden information into the features.
            row["my_deck_ids"] = decks.get(row["episode_id"] * 2 + row["seat"]) or []
            if row["my_deck_ids"] and len(set(row["my_deck_ids"])) > encoder.DECK_TOKEN_CAP:
                deck_truncated += 1
        forced = encoder.is_forced(row)
        options = encoder.encode_options(row, tables)
        tokens = encoder.encode_tokens(row, tables)
        target = np.asarray(row["action"], dtype=np.int16)
        # The corpus has zero contract violations, but a regenerated corpus is
        # not this corpus. A bad label would train the model to emit an illegal
        # action, which scores -1 exactly like a loss.
        if (target.size == 0 or target.max() >= options.shape[0] or target.min() < 0
                or np.unique(target).size != target.size):
            continue

        logs = encoder.encode_logs(row, tables) if with_logs else None
        globals_buf.append(encoder.encode_globals(row))
        token_buf.append(tokens)
        option_buf.append(options)
        target_buf.append(target)
        token_off.append(token_off[-1] + tokens.shape[0])
        option_off.append(option_off[-1] + options.shape[0])
        target_off.append(target_off[-1] + target.shape[0])
        # min/max count ride along from 2026-08-08: the autoregressive decoder
        # needs to know, at train time, whether stopping was even legal on this
        # row. Recomputing it from the option count is not possible — minCount 0
        # and minCount 1 look identical once the row is encoded.
        meta_buf.append((row["select_type"], row["context"], int(forced),
                         options.shape[0], int(row["min_count"] or 0),
                         int(row["max_count"] or 0)))
        episode_buf.append(row["episode_id"])
        # Seats absent from `keys` cannot occur — the Arrow filter dropped them —
        # so a missing key means no filter is active, i.e. every seat is kept and
        # the win flag is unknown. Weight those at 1.0.
        won = True if keys is None else keys.get(row["episode_id"] * 2 + row["seat"], True)
        weight_buf.append((1.0 if won else loss_weight) * decay)
        # The critic's Monte-Carlo target: this seat's terminal outcome. DeNA's
        # rule "redistribute the Action-completion value to every AtomicAction"
        # degenerates, for a behaviour-cloning corpus walked to the terminal, to
        # the same z on every row of the seat's game. Needs both winner and loser
        # seats present or z is constant and the head learns nothing — enforced by
        # requiring --all-seats.
        value_buf.append(1.0 if won else 0.0)
        if logs is not None:
            log_buf.append(logs)
            log_off.append(log_off[-1] + logs.shape[0])
        kept += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    empty_token = np.zeros((0, encoder.N_TOKEN), dtype=np.int16)
    empty_option = np.zeros((0, encoder.N_OPT), dtype=np.int16)
    arrays = {
        "globals": np.stack(globals_buf) if globals_buf else np.zeros((0, encoder.N_GLOBAL), np.int16),
        "tokens": np.concatenate(token_buf) if token_buf else empty_token,
        "tokens_off": np.asarray(token_off, dtype=np.int64),
        "options": np.concatenate(option_buf) if option_buf else empty_option,
        "options_off": np.asarray(option_off, dtype=np.int64),
        "targets": np.concatenate(target_buf) if target_buf else np.zeros(0, np.int16),
        "targets_off": np.asarray(target_off, dtype=np.int64),
        "meta": np.asarray(meta_buf, dtype=np.int16).reshape(-1, 6),
        "episode_id": np.asarray(episode_buf, dtype=np.int64),
        "weights": np.asarray(weight_buf, dtype=np.float32),
    }
    if value:
        arrays["values"] = np.asarray(value_buf, dtype=np.float32)
    if with_logs:
        arrays["logs"] = (np.concatenate(log_buf) if log_buf
                          else np.zeros((0, encoder.N_LOG), dtype=np.int16))
        arrays["logs_off"] = np.asarray(log_off, dtype=np.int64)
    for name, array in arrays.items():
        np.save(out_dir / f"{name}.npy", array)

    return {
        "date": day_dir.name.replace("date=", ""),
        "rows_seen": seen,
        "rows_kept": kept,
        "forced_dropped": forced_dropped,
        "options_total": int(option_off[-1]),
        "tokens_total": int(token_off[-1]),
        "logs_total": int(log_off[-1]),
        "loss_weighted_rows": int(sum(1 for w in weight_buf if w != 1.0)),
        "recency_decay": round(decay, 4),
        "weight_sum": round(float(sum(weight_buf)), 1),
        "deck_truncated": deck_truncated,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, default=Path("data/replays"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sdk", type=str, default="", help="directory containing the cg package")
    parser.add_argument("--deck-card", type=int, default=648,
                        help="keep seats whose deck contains this card id; -1 keeps every seat")
    parser.add_argument("--all-seats", action="store_true", help="keep losing seats too")
    parser.add_argument("--keep-forced", action="store_true")
    parser.add_argument("--loss-weight", type=float, default=1.0,
                        help="loss weight for seats that lost; only bites with --all-seats. "
                             "1.0 is a measured regression, 0.0 is winners-only")
    parser.add_argument("--with-logs", action="store_true",
                        help="encode obs.logs as a second token stream")
    parser.add_argument("--value", action="store_true",
                        help="emit the critic target: each row's seat terminal outcome (win=1, "
                             "loss=0). Trains the value head for value-backed PUCT search. "
                             "Requires --all-seats so both outcomes are present; winners-only "
                             "makes the target constant and the head learns nothing")
    parser.add_argument("--deck-tokens", action="store_true",
                        help="emit the acting seat's own 60-card list as deck tokens "
                             "(Architecture_A deck conditioning). Joined from decks.parquet on "
                             "the acting seat; the agent supplies deck.csv at inference. Lets "
                             "one checkpoint serve many decks — the measured win for low-seat "
                             "archetypes")
    parser.add_argument("--recency-halflife", type=float, default=0.0,
                        help="halve a row's loss weight for every N days it is older than the "
                             "newest partition included. 0 disables. The field turned over on "
                             "2026-07-31, so a flat cache over 48 days is mostly dead meta")
    parser.add_argument("--min-avg-score", type=float, default=0.0,
                        help="drop episodes whose games.avg_score is below this. The column is "
                             "the mean of both seats, so this keeps high-level GAMES, not "
                             "high-level players. Corpus median is 1090")
    parser.add_argument("--since", type=str, default="", help="first date to include, YYYY-MM-DD")
    parser.add_argument("--until", type=str, default="", help="last date to include, YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if args.sdk:
        sys.path.insert(0, args.sdk)
    tables = load_tables()  # fails fast here rather than inside a worker
    if not 0.0 <= args.loss_weight <= 1.0:
        raise SystemExit("--loss-weight must be in [0, 1]")
    if args.loss_weight != 1.0 and not args.all_seats:
        raise SystemExit("--loss-weight only does anything with --all-seats; "
                         "winners-only has no losing seats to weight")
    if args.value and not args.all_seats:
        raise SystemExit("--value needs --all-seats: winners-only makes every critic target 1.0, "
                         "and a head that only ever sees wins learns the constant 1")

    days = sorted(p for p in args.replays.glob("date=*") if p.is_dir())
    if args.since:
        days = [d for d in days if d.name.replace("date=", "") >= args.since]
    if args.until:
        days = [d for d in days if d.name.replace("date=", "") <= args.until]
    if not days:
        raise SystemExit("no partitions selected")

    if args.recency_halflife < 0.0:
        raise SystemExit("--recency-halflife must be >= 0")
    # Relative to the newest partition actually built, not to today: the corpus
    # is a fixed snapshot, and anchoring on the wall clock would silently change
    # every weight the next time the same command is run.
    newest_date = days[-1].name.replace("date=", "")

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "day_dir": str(d),
            "out_dir": str(args.out / d.name),
            "deck_card": args.deck_card,
            "winners_only": not args.all_seats,
            "keep_forced": args.keep_forced,
            "sdk": args.sdk,
            "loss_weight": args.loss_weight,
            "with_logs": args.with_logs,
            "deck_tokens": args.deck_tokens,
            "value": args.value,
            "min_avg_score": args.min_avg_score,
            "recency_halflife": args.recency_halflife,
            "newest_date": newest_date,
        }
        for d in days
    ]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(build_day, jobs):
            reports.append(report)
            print(f"  {report['date']}  kept {report['rows_kept']:>8,} of {report['rows_seen']:>8,}"
                  f"  forced dropped {report['forced_dropped']:>7,}  {report['seconds']:>6.1f}s", flush=True)

    if args.deck_tokens:
        truncated = sum(r.get("deck_truncated", 0) for r in reports)
        if truncated:
            print(f"WARNING: {truncated:,} rows held more than encoder.DECK_TOKEN_CAP "
                  f"({encoder.DECK_TOKEN_CAP}) distinct deck cards; their deck block lost its "
                  f"tail. Raise the cap — the feature is quietly wrong, not absent.")

    manifest = {
        "encoder_fingerprint": encoder_fingerprint(),
        "slot_sizes": encoder.slot_sizes(tables, with_logs=args.with_logs),
        "with_logs": args.with_logs,
        "deck_tokens": args.deck_tokens,
        "value": args.value,
        "n_global": encoder.N_GLOBAL,
        "n_token": encoder.N_TOKEN,
        "n_option": encoder.N_OPT,
        "card_rows": tables.card_rows,
        "attack_rows": tables.attack_rows,
        "filter": {
            "deck_card": args.deck_card,
            "winners_only": not args.all_seats,
            "forced_kept": args.keep_forced,
            "loss_weight": args.loss_weight,
            "min_avg_score": args.min_avg_score,
            "recency_halflife": args.recency_halflife,
            "newest_date": newest_date,
        },
        "days": reports,
        "rows_kept": sum(r["rows_kept"] for r in reports),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    rows = manifest["rows_kept"]
    if rows == 0:
        raise SystemExit(
            f"\nthe filter kept 0 rows. Check --min-avg-score against the corpus median "
            f"(1090) and --deck-card. Refusing to write a cache that would train on nothing."
        )
    print(f"\ntotal rows kept: {rows:,}  ->  {args.out}")
    if args.recency_halflife > 0:
        weighted = sum(r["weight_sum"] for r in reports)
        print(f"recency halflife {args.recency_halflife:g}d: effective rows "
              f"{weighted:,.0f} ({weighted / rows:.2f}x per row), newest partition {newest_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
