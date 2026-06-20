"""Auto-download top-team episode replays.

Enumeration uses the Kaggle API key (ListEpisodes works). The replay fetch needs your BROWSER
SESSION COOKIE (the API key is blocked for replays). Grab it once from DevTools -> Network when
you download a replay: the request URL + the full `cookie:` header.

Set:
  export KAGGLE_COOKIE='<the full cookie header string>'
  # REPLAY_URL_TEMPLATE: the request URL with the episode id replaced by {ep}
  export REPLAY_URL='https://www.kaggle.com/...{ep}...'   # paste yours, put {ep} where the id is

Usage:
  python tools/fetch_replays.py --submissions 53802029 53878567   # enumerate these subs' episodes -> download all
  python tools/fetch_replays.py --teams 16376775                  # (if you have submissionIds; teams need a submissionId)
"""
import argparse, json, os, base64, urllib.request, urllib.error, time, glob

KJ = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH = base64.b64encode(f"{KJ['username']}:{KJ['key']}".encode()).decode()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json")


def list_episodes(submission_id):
    req = urllib.request.Request(
        "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps({"submissionId": submission_id}).encode(),
        headers={"Authorization": f"Basic {AUTH}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read().decode())


REPLAY_DEFAULT = "https://www.kaggle.com/competitions/episodes/{ep}/replay.json"


def download_replay(ep_id, cookie, url_tmpl):
    url = url_tmpl.replace("{ep}", str(ep_id))
    req = urllib.request.Request(url, headers={"cookie": cookie, "User-Agent": "Mozilla/5.0",
                                               "Accept": "application/json",
                                               "Accept-Encoding": "identity"})  # avoid brotli
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", type=int, nargs="*", default=[53802029])
    ap.add_argument("--max-subs", type=int, default=40, help="how many submissions to enumerate (BFS)")
    ap.add_argument("--max-dl", type=int, default=150, help="cap new replay downloads")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    have = {os.path.basename(f).split(".")[0] for f in glob.glob(f"{OUT}/*.json")}
    cookie = os.environ.get("KAGGLE_COOKIE"); url_tmpl = os.environ.get("REPLAY_URL") or REPLAY_DEFAULT

    # 1) BFS-enumerate episodes across discovered top-tier submissions (API key)
    ep_ids, seen_subs, queue = set(), set(), list(args.submissions)
    while queue and len(seen_subs) < args.max_subs:
        s = queue.pop(0)
        if s in seen_subs:
            continue
        seen_subs.add(s)
        try:
            d = list_episodes(s)
        except Exception as e:
            print(f"  list {s}: {repr(e)[:60]}"); continue
        for e in d.get("episodes", []):
            ep_ids.add(e["id"])
            for a in e.get("agents", []):
                sid = a.get("submissionId")
                if sid and sid not in seen_subs and sid not in queue:
                    queue.append(sid)
    print(f"enumerated {len(ep_ids)} episodes across {len(seen_subs)} top-tier submissions")

    if not cookie:
        print("\nSet KAGGLE_COOKIE to download. Enumeration-only.")
        return
    # 2) download replays via browser cookie
    got = 0
    for ep in sorted(ep_ids):
        if str(ep) in have or got >= args.max_dl:
            continue
        try:
            data = download_replay(ep, cookie, url_tmpl)
            if len(data) > 10000:
                open(f"{OUT}/{ep}.json", "wb").write(data); got += 1
                if got % 10 == 0:
                    print(f"  ...{got} downloaded")
            else:
                print(f"  {ep}: tiny ({len(data)}b) — cookie expired?"); break
            time.sleep(0.7)
        except urllib.error.HTTPError as e:
            print(f"  {ep}: HTTP {e.code} (cookie expired?)"); break
    print(f"downloaded {got} new replays -> {OUT} (total now {len(have)+got})")


if __name__ == "__main__":
    main()
