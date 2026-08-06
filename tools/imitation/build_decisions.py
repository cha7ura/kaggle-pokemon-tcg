"""Replays -> a pointwise ranking dataset for the card policy. WINNER-FILTERED: we only learn from
the decisions made by the seat that WON the game (imitate winners, not all play). One row per legal
option: label=1 if the winner chose it, else 0. Features via featurize.row (state+threat+tracker+card).

  python -m tools.imitation.build_decisions [--limit N] [--out data/decisions.npz]

Output NPZ: X (rows x dims float32), y (0/1), groups (decision id, for held-out split), names, contexts.
Needs numpy (offline only; nothing here ships in the submission).
"""
import os, sys, time
import numpy as np
from tools.replays_db import iter_replays
from tools.imitation import featurize
from tools.imitation.extract_decisions import _deck_of

DATA = os.path.join(os.path.dirname(__file__), "data")


def _winner_seat(rewards):
    if not rewards or len(rewards) < 2:
        return None
    r0, r1 = rewards[0], rewards[1]
    if r0 is None or r1 is None or r0 == r1:   # some bulk replays have null rewards -> skip
        return None
    return 0 if r0 > r1 else 1


def iter_rows(game, gid):
    steps = game.get("steps") or []
    rewards = game.get("rewards") or [0, 0]
    seat = _winner_seat(rewards)
    if seat is None:
        return
    deck = _deck_of(steps, seat)
    if not deck:
        return
    for si, s in enumerate(steps):
        obs = s[seat].get("observation", {})
        sel = obs.get("select"); cur = obs.get("current")
        act = s[seat].get("action")
        if not (sel and cur and isinstance(act, list) and act):
            continue
        opts = sel.get("option") or []
        if len(opts) < 2 or len(opts) == 60:        # skip trivial + the 60-card deck pick
            continue
        if not all(isinstance(x, int) and 0 <= x < len(opts) for x in act):
            continue
        chosen = set(act)
        # State is identical for every option in this decision -> compute ONCE (the threat/tracker
        # work is the expensive part). Then only the cheap option_row varies per option.
        try:
            sv, sn = featurize.state_row(obs, seat, deck)
        except Exception:
            continue
        for oi, opt in enumerate(opts):
            try:
                ov, on = featurize.option_row(obs, seat, opt, deck)
            except Exception:
                continue
            yield sv + ov, (1 if oi in chosen else 0), f"{gid}:{si}", sn + on


def _save_shard(path, X, y, groups, names):
    np.savez_compressed(path, X=np.asarray(X, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int8), groups=np.asarray(groups),
                        names=np.asarray(names))


def main():
    limit = None
    out = os.path.join(DATA, "decisions.npz")
    archetype = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if "--archetype" in sys.argv:
        archetype = sys.argv[sys.argv.index("--archetype") + 1]
    # Sharded (streaming) mode: for the FULL 88k corpus (~38M rows) a single in-memory
    # accumulation OOMs. --shard-rows N flushes a compressed shard to {prefix}_NNNN.npz every
    # >=N rows and clears the live lists, bounding peak memory to one shard. Writes a manifest.
    shard_rows = int(sys.argv[sys.argv.index("--shard-rows") + 1]) if "--shard-rows" in sys.argv else 0
    prefix = sys.argv[sys.argv.index("--out-prefix") + 1] if "--out-prefix" in sys.argv else None
    if shard_rows and not prefix:
        prefix = os.path.splitext(out)[0]
    os.makedirs(DATA, exist_ok=True)

    # Deck-matched policy: restrict to games the WINNER played with `archetype` (via the
    # replay_field/v_episode_arch SQL pipeline). SQLite filters before any blob is decompressed.
    where = params = None
    if archetype:
        where = ("episode_id IN (SELECT episode_id FROM v_episode_arch "
                 "WHERE (reward0>reward1 AND arch0=?) OR (reward1>reward0 AND arch1=?))")
        params = (archetype, archetype)
        print(f"deck-matched: winner-archetype = {archetype}", flush=True)

    X, y, groups, names = [], [], [], None
    gi, used = 0, 0
    shards, total_rows, total_pos = [], 0, 0

    def flush():
        nonlocal X, y, groups, total_rows, total_pos
        if not X:
            return
        path = f"{prefix}_{len(shards):04d}.npz"
        _save_shard(path, X, y, groups, names)
        total_rows += len(X); total_pos += int(np.asarray(y, dtype=np.int8).sum())
        shards.append({"file": os.path.basename(path), "rows": len(X)})
        print(f"  flushed shard {path} ({len(X)} rows, {total_rows} total)", flush=True)
        X, y, groups = [], [], []

    t0 = time.time()
    for ep, g in (iter_replays(where, params) if where else iter_replays()):
        gi += 1
        if limit and gi > limit:
            break
        had = False
        for vec, label, grp, nm in iter_rows(g, ep):
            X.append(vec); y.append(label); groups.append(grp); names = nm; had = True
        used += 1 if had else 0
        if gi % 200 == 0:
            print(f"  {gi} games, {total_rows + len(X)} rows, {used} usable, {time.time()-t0:.0f}s", flush=True)
        if shard_rows and len(X) >= shard_rows:
            flush()
    if not X and not shards:
        print("no rows produced"); return

    if shard_rows:                                  # streaming mode: write final shard + manifest
        flush()
        import json as _json
        man = {"prefix": os.path.basename(prefix), "shards": shards, "games": gi, "usable": used,
               "rows": total_rows, "positives": total_pos, "dims": len(names),
               "names": list(names)}
        with open(f"{prefix}_manifest.json", "w", encoding="utf-8") as fh:
            _json.dump(man, fh)
        pct = total_pos / total_rows * 100 if total_rows else 0.0
        print(f"DONE games={gi} usable={used} rows={total_rows} dims={len(names)} "
              f"positives={total_pos} ({pct:.1f}%) shards={len(shards)} -> {prefix}_manifest.json")
        return

    Xa = np.asarray(X, dtype=np.float32)
    ya = np.asarray(y, dtype=np.int8)
    ga = np.asarray(groups)
    np.savez_compressed(out, X=Xa, y=ya, groups=ga, names=np.asarray(names))
    pos = int(ya.sum())
    print(f"DONE games={gi} usable={used} rows={len(X)} dims={Xa.shape[1]} "
          f"positives={pos} ({pos/len(X)*100:.1f}%) decisions={len(set(groups))} -> {out}")


if __name__ == "__main__":
    main()
