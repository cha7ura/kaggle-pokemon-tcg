"""Cookie-free replay downloader via Kaggle's official daily episode DATASETS.
API ListEpisodes gives ep id + Elo + createTime; the replay JSON is pulled per-file from
kaggle/pokemon-tcg-ai-battle-episodes-<date> (no cookie, no rate-limit). Top-Elo leaders -> json/,
our games -> json_ours/. Re-runnable; dedups.
"""
import os, glob, time, subprocess, datetime
import fetch_replays as fr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_DIR = os.path.join(ROOT, "json_ours"); LEAD_DIR = fr.OUT
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


def pull(ep, day, outdir):
    """try the ep's day-dataset +/-1 (the daily dump straddles midnight)."""
    if os.path.exists(f"{outdir}/{ep}.json"):
        return False
    for delta in (0, -1, 1):
        d = day + datetime.timedelta(days=delta)
        r = subprocess.run(["kaggle", "datasets", "download", f"{DS}{d}", "-f", f"{ep}.json", "-p", outdir],
                           capture_output=True, text=True, timeout=120)
        if os.path.exists(f"{outdir}/{ep}.json"):
            return True
        # CLI sometimes leaves a .zip
        z = f"{outdir}/{ep}.json.zip"
        if os.path.exists(z):
            subprocess.run(["unzip", "-o", z, "-d", outdir], capture_output=True); os.remove(z)
            if os.path.exists(f"{outdir}/{ep}.json"): return True
    return None  # not found in any adjacent day


def main():
    os.makedirs(OUR_DIR, exist_ok=True)
    # OUR games (all)
    got = 0
    for sid in OUR_SUBS:
        for e in enum_safe(sid).get("episodes", []):
            if pull(e["id"], day_of(e["createTime"]), OUR_DIR): got += 1
    print(f"ours +{got} (total {len(glob.glob(OUR_DIR+'/*.json'))})", flush=True)
    # LEADER games (top-Elo)
    ep_day, seen, q = {}, set(), list(TOP_SEEDS)
    while q and len(seen) < 40:
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
        if pull(ep, day, LEAD_DIR):
            ldl += 1
            if ldl % 25 == 0: print(f"  ...leaders +{ldl}", flush=True)
    print(f"leaders +{ldl} (total {len(glob.glob(LEAD_DIR+'/*.json'))})", flush=True)


if __name__ == "__main__":
    main()
