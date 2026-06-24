"""Cookie-free replay downloader via Kaggle's official daily episode DATASETS.
API ListEpisodes gives ep id + Elo + createTime; the replay JSON is pulled per-file from
kaggle/pokemon-tcg-ai-battle-episodes-<date> (no cookie, no rate-limit). Each replay is
compressed straight into replays.sqlite (source 'leader' or 'ours'); no json/ files kept.
Re-runnable; dedups against the db.
"""
import os, time, subprocess, datetime, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # sibling fetch_replays
sys.path.insert(0, os.path.dirname(_HERE))      # tools package (replays_db)
import fetch_replays as fr
from tools import replays_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".fetch_tmp")  # scratch for the kaggle CLI; emptied per file
OUR_SUBS = [53880887, 53892473]
TOP_SEEDS = [53802029, 53880887, 53878567]
MIN_SCORE = 1100
DS = "kaggle/pokemon-tcg-ai-battle-episodes-"


def enum_safe(sid):
    for a in range(4):
        try:
            return fr.list_episodes(sid)
        except Exception as e:
            if "429" in str(e): time.sleep(5 * (a + 1)); continue
            return {"episodes": []}
    return {"episodes": []}


def day_of(iso):
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()


def pull(ep, day, source):
    """try the ep's day-dataset +/-1 (the daily dump straddles midnight); store into db."""
    if replays_db.has(ep):
        return False
    p = f"{TMP}/{ep}.json"
    for delta in (0, -1, 1):
        d = day + datetime.timedelta(days=delta)
        try:
            subprocess.run(["kaggle", "datasets", "download", f"{DS}{d}", "-f", f"{ep}.json", "-p", TMP],
                           capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue  # one slow file must not abort the whole crawl
        z = f"{p}.zip"
        if os.path.exists(z):  # CLI sometimes leaves a .zip
            subprocess.run(["unzip", "-o", z, "-d", TMP], capture_output=True); os.remove(z)
        if os.path.exists(p):
            replays_db.ingest_file(p, source=source, delete=True)  # compress into db, drop temp
            return True
    return None  # not found in any adjacent day


def _count(source):
    db = replays_db._connect()
    n = db.execute("SELECT COUNT(*) FROM replays WHERE source=?", (source,)).fetchone()[0]
    db.close()
    return n


def main():
    os.makedirs(TMP, exist_ok=True)
    # OUR games (all)
    got = 0
    for sid in OUR_SUBS:
        for e in enum_safe(sid).get("episodes", []):
            if pull(e["id"], day_of(e["createTime"]), "ours"): got += 1
    print(f"ours +{got} (total {_count('ours')})", flush=True)
    # LEADER games (top-Elo). Storage is cheap (db), so crawl wide: override the
    # submission-seed cap via argv, e.g. `python tools/fetch_dataset.py 400`.
    max_subs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    ep_day, seen, q = {}, set(), list(TOP_SEEDS)
    while q and len(seen) < max_subs:
        s = q.pop(0)
        if s in seen: continue
        seen.add(s)
        for e in enum_safe(s).get("episodes", []):
            scs = [a.get("updatedScore", 0) for a in e.get("agents", [])]
            if (min(scs) if scs else 0) >= MIN_SCORE:
                ep_day[e["id"]] = day_of(e["createTime"])
            for a in e.get("agents", []):
                sid = a.get("submissionId")
                if sid and sid not in seen and sid not in q: q.append(sid)
    print(f"top-Elo (>= {MIN_SCORE}) episodes to consider: {len(ep_day)}", flush=True)
    ldl = 0
    for ep, day in ep_day.items():
        if pull(ep, day, "leader"):
            ldl += 1
            if ldl % 25 == 0: print(f"  ...leaders +{ldl}", flush=True)
    print(f"leaders +{ldl} (total {_count('leader')})", flush=True)


if __name__ == "__main__":
    main()
