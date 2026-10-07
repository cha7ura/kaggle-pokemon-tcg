#!/usr/bin/env python3
"""Playground meta read: which decks the top teams run, and how they fare.

    python3 tools/pg_meta.py --top 40 --episodes 6        # appends data/pg_meta.jsonl
    python3 tools/pg_meta.py --report                     # archetype table from the jsonl

Downloads each replay via `kaggle competitions replay`, keeps only the two submitted decks,
team names and rewards, and deletes the ~10 MB file at once (the Mac disk is near full).
Re-runs skip episodes already in the jsonl.
"""
import argparse, csv, glob, io, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
OUT = os.path.join(ROOT, "data", "pg_meta.jsonl")
CARDS = os.path.join(ROOT, "data", "playground", "EN_Card_Data_R2_full.csv")


def kcsv(*args):
    out = subprocess.run(["kaggle", "competitions", *args, "-v"], capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(out.lstrip("﻿"))))


def fetch(ep):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "competitions", "replay", str(ep), "-p", tmp, "-q"], capture_output=True)
        files = glob.glob(os.path.join(tmp, "*.json"))
        if not files:
            return None
        d = json.load(open(files[0]))
    decks = [d["steps"][1][s].get("action") for s in (0, 1)]
    if not all(isinstance(x, list) and len(x) == 60 for x in decks):
        return None
    return {"episode": ep, "teams": d["info"]["TeamNames"], "rewards": d["rewards"], "decks": decks}


def collect(top, per):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["episode"] for l in open(OUT)}
    lb = sorted(glob.glob("/private/tmp/claude-501/lb/*.csv"))
    rows = list(csv.DictReader(open(lb[-1], encoding="utf-8-sig")))[:top]
    eps = []
    for r in rows:
        subs = kcsv("team-submissions", r["TeamId"])
        for s in subs[:2]:  # the two active submissions
            for e in kcsv("episodes", s["id"])[:per]:
                if int(e["id"]) not in seen and "COMPLETED" in e["state"]:
                    seen.add(int(e["id"])); eps.append(int(e["id"]))
    print(f"{len(eps)} new episodes from top {top}", flush=True)
    with ThreadPoolExecutor(4) as ex, open(OUT, "a") as f:
        for i, rec in enumerate(ex.map(fetch, eps), 1):
            if rec:
                f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(eps)}", flush=True)


def archetype(deck, names, stage):
    """Name a deck by its top two rule-box / evolved Pokemon by copy count."""
    c = Counter(deck)
    keys = [cid for cid, _ in c.most_common()
            if stage.get(cid) in ("Stage 2", "Stage 1") or " ex" in names.get(cid, "")]
    return " / ".join(sorted({names[k] for k in keys[:2]})) or "?"


def report():
    names, stage = {}, {}
    for r in csv.DictReader(open(CARDS, encoding="utf-8-sig")):
        if r["Card ID"].isdigit():
            names[int(r["Card ID"])] = r["Card Name"]
            stage[int(r["Card ID"])] = r["Stage (Pokémon)/Type (Energy and Trainer)"]
    lb = sorted(glob.glob("/private/tmp/claude-501/lb/*.csv"))
    rank = {r["TeamName"]: int(r["Rank"]) for r in csv.DictReader(open(lb[-1], encoding="utf-8-sig"))}
    use, win, games, teams = Counter(), Counter(), Counter(), defaultdict(set)
    for l in open(OUT):
        rec = json.loads(l)
        for s in (0, 1):
            a = archetype(rec["decks"][s], names, stage)
            use[a] += 1; games[a] += 1; win[a] += rec["rewards"][s] == 1
            teams[a].add(rank.get(rec["teams"][s], 999))
    print(f"{'archetype':45s} seats  win%  best-rank  teams")
    for a, n in use.most_common(25):
        print(f"{a[:45]:45s} {n:5d}  {win[a]/games[a]:.2f}  {min(teams[a]):9d}  {len(teams[a]):5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else collect(a.top, a.episodes)
