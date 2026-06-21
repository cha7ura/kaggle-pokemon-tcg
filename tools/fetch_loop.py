"""Continuous throttled downloader: OUR games (json_ours/) + top-leader games (json/).
Loops with sleeps to avoid 429. Leaders discovered by BFS (catches a changed top). Our games
have no score floor (our Elo < 1000); leader games require weaker-agent Elo >= MIN_SCORE.
Run: KAGGLE_COOKIE=... python tools/fetch_loop.py
"""
import os, time, glob, json
import fetch_replays as fr   # reuse list_episodes / download_replay / REPLAY_DEFAULT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_DIR = os.path.join(ROOT, "json_ours"); LEAD_DIR = fr.OUT
OUR_SUBS = [53880887, 53892473]
TOP_SEEDS = [53802029, 53880887, 53878567]
MIN_SCORE = 1000
SLEEP_DL = 0.8        # between downloads (throttle)
SLEEP_ROUND = 45      # between rounds
COOKIE = os.environ["KAGGLE_COOKIE"]; URL = fr.REPLAY_DEFAULT


def have(d):
    return {os.path.basename(f).split(".")[0] for f in glob.glob(f"{d}/*.json")}


def grab(ep, outdir):
    try:
        data = fr.download_replay(ep, COOKIE, URL)
        if len(data) > 10000:
            open(f"{outdir}/{ep}.json", "wb").write(data); return True
        print(f"  {ep}: tiny ({len(data)}b) — cookie expired?", flush=True)
        return None  # signal stop
    except Exception as e:
        print(f"  {ep}: {repr(e)[:70]} (skip)", flush=True); return False  # transient -> skip, don't stop


def enum_safe(sid):
    for a in range(4):
        try:
            return fr.list_episodes(sid)
        except Exception as e:
            if "429" in str(e): time.sleep(5 * (a + 1)); continue
            return {"episodes": []}
    return {"episodes": []}


def main():
    os.makedirs(OUR_DIR, exist_ok=True)
    rnd = 0
    while True:
        rnd += 1
        # ---- OUR games (all, no score floor) ----
        h = have(OUR_DIR); got = 0
        for sid in OUR_SUBS:
            for e in enum_safe(sid).get("episodes", []):
                ep = e["id"]
                if str(ep) in h: continue
                r = grab(ep, OUR_DIR)
                if r is None: print("stop: cookie expired", flush=True); return
                if r: got += 1; h.add(str(ep))
                time.sleep(SLEEP_DL)
            time.sleep(0.4)
        # ---- LEADER games (BFS, Elo >= MIN_SCORE) ----
        ep_score, seen, q = {}, set(), list(TOP_SEEDS)
        while q and len(seen) < 40:
            s = q.pop(0)
            if s in seen: continue
            seen.add(s)
            for e in enum_safe(s).get("episodes", []):
                scs = [a.get("updatedScore", 0) for a in e.get("agents", [])]
                ep_score[e["id"]] = min(scs) if scs else 0
                for a in e.get("agents", []):
                    sid = a.get("submissionId")
                    if sid and sid not in seen and sid not in q: q.append(sid)
            time.sleep(0.3)
        ranked = [ep for ep, sc in sorted(ep_score.items(), key=lambda kv: -kv[1]) if sc >= MIN_SCORE]
        hl = have(LEAD_DIR); ldl = 0
        for ep in ranked:
            if str(ep) in hl: continue
            r = grab(ep, LEAD_DIR)
            if r is None: print("stop: cookie expired", flush=True); return
            if r:
                ldl += 1; hl.add(str(ep))
                if ldl % 20 == 0: print(f"  ...leaders +{ldl}", flush=True)
            time.sleep(SLEEP_DL)
        print(f"round {rnd}: ours +{got} (total {len(have(OUR_DIR))}) | "
              f"leaders +{ldl} (total {len(have(LEAD_DIR))})", flush=True)
        time.sleep(SLEEP_ROUND)


if __name__ == "__main__":
    main()
