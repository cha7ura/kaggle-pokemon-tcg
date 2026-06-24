# tools/imitation/extract_decisions.py
"""Replays -> per-deck decision tables. A decision is learnable when select has >=2 options and the
action is a list of small indices into option (the 60-card deck action is naturally excluded since
its entries are card ids >> n_options). STDLIB ONLY."""
import json, glob, os, sys, collections, hashlib
from tools.imitation.features import state_features, option_features
from tools.replays_db import iter_replays

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(os.path.dirname(__file__), "data")
import csv as _csv
_names = {int(r["cardId"]): r["name"]
          for r in _csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}


def deck_sig_of(deck):
    return "_".join(str(x) for x in sorted(deck))


def sig_to_fname(sig):
    """Short, filesystem-safe filename stem for a (long) deck signature."""
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def _deck_of(steps, seat):
    for s in steps:
        a = s[seat].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None


def _archetype(deck):
    ids = set(deck)
    for cid in ids:
        nm = _names.get(cid, "")
        if nm == "Alakazam Powerful Hand" or cid == 743: return "Alakazam"
        if nm == "Hop's Trevenant": return "Trevenant"
        if nm == "Dragapult ex": return "Dragapult"
        if cid == 678: return "MegaLucario"
        if cid == 1031: return "MegaStarmie"
    return "OTHER"


def iter_decisions(game):
    steps = game.get("steps") or []
    info = game.get("info") or {}
    names = info.get("TeamNames") or [None, None]
    rewards = game.get("rewards") or [0, 0]
    decks = {seat: _deck_of(steps, seat) for seat in (0, 1)}
    for s in steps:
        for seat in (0, 1):
            obs = s[seat].get("observation", {})
            sel = obs.get("select"); cur = obs.get("current")
            act = s[seat].get("action")
            if not (sel and cur and isinstance(act, list)):
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                continue
            if not act or not all(isinstance(x, int) and 0 <= x < len(opts) for x in act):
                continue
            if decks[seat] is None:
                continue
            ctx = sel.get("context")
            yield {
                "deck_sig": deck_sig_of(decks[seat]),
                "player": names[seat],
                "reward": rewards[seat] if seat < len(rewards) else 0,
                "context": ctx,
                "state": state_features(cur, seat),
                "options": [option_features(o, ctx, sel) for o in opts],
                "chosen": act[0],
            }


def main():
    os.makedirs(DATA, exist_ok=True)
    files = {}
    meta = collections.defaultdict(lambda: {"player": None, "games": set(), "decisions": 0,
                                            "deck": None})
    for ep, g in iter_replays():
        gid = g.get("info", {}).get("EpisodeId", ep)
        seen_sig = set()
        for r in iter_decisions(g):
            sig = r["deck_sig"]
            if sig not in files:
                # Use a short hash for the filename — full sigs can exceed OS filename limits.
                fname = sig_to_fname(sig)
                files[sig] = open(os.path.join(DATA, f"{fname}.jsonl"), "a")
            files[sig].write(json.dumps(r) + "\n")
            m = meta[sig]; m["player"] = r["player"]; m["decisions"] += 1
            m["games"].add(gid); seen_sig.add(sig)
    for fh in files.values():
        fh.close()
    manifest = []
    for sig, m in meta.items():
        deck = [int(x) for x in sig.split("_")]
        manifest.append({"deck_sig": sig, "file": sig_to_fname(sig),
                         "player": m["player"],
                         "archetype": _archetype(deck),
                         "games": len(m["games"]), "decisions": m["decisions"]})
    manifest.sort(key=lambda x: -x["games"])
    json.dump(manifest, open(os.path.join(DATA, "manifest.json"), "w"), indent=2)
    print(f"decks={len(manifest)} "
          f">=30g={sum(1 for x in manifest if x['games']>=30)}")
    for x in manifest[:12]:
        print(f"  {x['games']:4}g {x['decisions']:6}d  {x['archetype']:12} {x['player']}")


if __name__ == "__main__":
    main()
