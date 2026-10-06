#!/usr/bin/env python3
"""replays.sqlite -> the day-partitioned parquet layout tools/bc/build_dataset.py reads.

    python3 tools/bc/sqlite_to_parquet.py --sdk data/playground/sample_submission/sample_submission/sample_submission \
        --out data/bc_parquet [--limit N] [--min-id ID] [--workers 8]

Writes data/bc_parquet/date=YYYY-MM-DD/{steps,decks,games}.parquet.

Decision rows go through Cleo's own `row_from_observation`, the same frontend the live agent
uses, so training and inference see one row shape by construction.

Pairing: in a Kaggle replay, the observation at step t for the seat whose status is ACTIVE is
answered by that seat's `action` at step t+1. This was checked on our DB on 2026-10-05.

Our DB stores no per-episode date. Kaggle episode ids increase over time, so the date is
interpolated linearly from the id across the first comp's window (2026-06-16 .. 2026-08-31).
It is approximate. It is only used to partition by date, for the temporal holdout and for
recency weighting. `games.avg_score` is null: we hold no per-episode ladder score, so do
not pass --min-avg-score.
"""

from __future__ import annotations

import argparse
import json
import lzma
import sqlite3
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D0, D1 = date(2026, 6, 16), date(2026, 8, 31)
ID0, ID1 = 80164996, 97184824


def id_to_date(eid: int) -> str:
    frac = min(1.0, max(0.0, (eid - ID0) / (ID1 - ID0)))
    return (D0 + timedelta(days=round(frac * (D1 - D0).days))).isoformat()


def decode(blob: bytes) -> dict:
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        raw = zlib.decompress(blob)
    return json.loads(raw)


def episode_rows(eid: int, d: dict, decks: dict[int, list[int]]):
    """Yield (step_row, ...) for every decision either seat made in one replay."""
    from cg.api import to_observation_class
    from tools.bc.rows import row_from_observation

    steps = d["steps"]
    day = id_to_date(eid)
    for t in range(len(steps) - 1):
        for seat in (0, 1):
            cur = steps[t][seat]
            if cur.get("status") != "ACTIVE":
                continue
            obs = cur.get("observation") or {}
            if obs.get("select") is None or obs.get("current") is None:
                continue  # deck phase: nothing to encode
            action = steps[t + 1][seat].get("action")
            if not isinstance(action, list) or not action:
                continue
            row = row_from_observation(to_observation_class(obs),
                                       remaining_overage=obs.get("remainingOverageTime"))
            if row["seat"] != seat or any(a >= row["n_options"] for a in action):
                continue  # misaligned step; never train on it
            row.pop("my_deck_ids", None)  # builder joins decks.parquet itself
            row.update(episode_id=eid, date=day, action=[int(a) for a in action])
            yield row


def convert(job):
    db, sdk, ids = job
    sys.path[:0] = [sdk, str(ROOT)]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    steps, decks, games = [], [], []
    for eid_s in ids:
        eid_s, r0, r1, blob = con.execute(
            "select episode_id, reward0, reward1, blob from replays where episode_id=?", (eid_s,)).fetchone()
        eid = int(eid_s)
        d = decode(blob)
        seat_deck = {}
        for seat in (0, 1):
            first = d["steps"][1][seat].get("action") if len(d["steps"]) > 1 else None
            if isinstance(first, list) and len(first) == 60:
                seat_deck[seat] = [int(c) for c in first]
        # The step-1 action is the deck only for the seat that submitted it; prefer the
        # replay_field join, which was built per player.
        for seat, cards in con.execute(
                "select f.player, k.cards_json from replay_field f join decks k on k.sig=f.deck_sig "
                "where f.episode_id=?", (eid_s,)):
            seat_deck[int(seat)] = [int(c) for c in json.loads(cards)]
        if len(seat_deck) < 2:
            continue
        winner = 0 if (r0 or 0) > (r1 or 0) else 1 if (r1 or 0) > (r0 or 0) else None
        try:
            rows = list(episode_rows(eid, d, seat_deck))
        except Exception as e:  # one bad replay must not kill a worker's whole chunk
            print(f"skip {eid}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        steps += rows
        for seat in (0, 1):
            decks.append({"episode_id": eid, "seat": seat, "deck": seat_deck[seat]})
        games.append({"episode_id": eid, "winner_seat": winner, "avg_score": None})
    return to_tables(steps, decks, games)


def to_tables(steps, decks, games) -> dict[str, list]:
    """Group one chunk by date as Arrow tables: ~10x smaller than the dicts, cheap to pickle.

    No `date` column in games/decks: the builder reads them with pq.read_table, which infers
    the hive `date=` partition and refuses a same-named column of a different type.
    """
    import pyarrow as pa

    day_of = {g["episode_id"]: id_to_date(g["episode_id"]) for g in games}
    by_day: dict[str, tuple[list, list, list]] = {}
    for i, rows in enumerate((steps, decks, games)):
        for r in rows:
            by_day.setdefault(day_of[r["episode_id"]], ([], [], []))[i].append(r)
    return {day: [pa.Table.from_pylist(x) for x in parts] for day, parts in by_day.items()}


def flush(out: Path, day: str, parts: list[list]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = out / f"date={day}"
    p.mkdir(parents=True, exist_ok=True)  # overwrites: always convert into a fresh --out
    for name, tables in zip(("steps", "decks", "games"), parts):
        # a column that is all-null in one chunk infers as `null`; promotion fixes the type
        pq.write_table(pa.concat_tables(tables, promote_options="default"), p / f"{name}.parquet")
    print(f"{day}: {sum(t.num_rows for t in parts[2])} games, "
          f"{sum(t.num_rows for t in parts[0])} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "replays.sqlite"))
    ap.add_argument("--sdk", required=True, help="directory containing the cg package")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="newest N episodes only (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    ids = [r[0] for r in con.execute(
        "select episode_id from replays where source='leader' order by cast(episode_id as int) desc"
        + (f" limit {a.limit}" if a.limit else ""))]
    chunks = [(a.db, a.sdk, ids[i:i + a.chunk]) for i in range(0, len(ids), a.chunk)]
    # Ids are descending and ex.map yields in order, so once a chunk's oldest date is below a
    # buffered date, that date can receive no more rows: write it and drop it from memory.
    buf: dict[str, list[list]] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, got in enumerate(ex.map(convert, chunks), 1):
            for day, tables in got.items():
                for slot, t in zip(buf.setdefault(day, [[], [], []]), tables):
                    slot.append(t)
            if got:
                oldest = min(got)
                for day in [d for d in buf if d > oldest]:
                    flush(a.out, day, buf.pop(day))
            if i % 50 == 0:
                print(f"{i}/{len(chunks)} chunks", flush=True)
    for day in sorted(buf):
        flush(a.out, day, buf[day])


if __name__ == "__main__":
    main()
