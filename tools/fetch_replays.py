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


def download_replay(ep_id, cookie, url_tmpl):
    url = url_tmpl.replace("{ep}", str(ep_id))
    req = urllib.request.Request(url, headers={"cookie": cookie, "User-Agent": "Mozilla/5.0",
                                               "Accept": "application/json"})
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", type=int, nargs="*", default=[])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    have = {os.path.basename(f).split(".")[0] for f in glob.glob(f"{OUT}/*.json")}
    cookie = os.environ.get("KAGGLE_COOKIE"); url_tmpl = os.environ.get("REPLAY_URL")

    # 1) enumerate episode ids (works with API key) — also harvests opponent submissionIds to expand
    ep_ids, subs = set(), set(args.submissions)
    for s in args.submissions:
        d = list_episodes(s)
        for e in d.get("episodes", []):
            ep_ids.add(e["id"])
            for a in e.get("agents", []):
                if a.get("submissionId"): subs.add(a["submissionId"])
    print(f"enumerated {len(ep_ids)} episodes from {len(args.submissions)} submissions "
          f"(+{len(subs)-len(args.submissions)} opponent submissions discovered)")

    if not cookie or not url_tmpl:
        print("\nSet KAGGLE_COOKIE + REPLAY_URL (with {ep}) to download. Enumeration-only for now.")
        print("episode ids:", sorted(ep_ids)[:20], "..." if len(ep_ids) > 20 else "")
        return

    # 2) download replays via browser cookie
    got = 0
    for ep in sorted(ep_ids):
        if str(ep) in have:
            continue
        try:
            data = download_replay(ep, cookie, url_tmpl)
            if len(data) > 10000:
                open(f"{OUT}/{ep}.json", "wb").write(data); got += 1
                print(f"  saved {ep}.json ({len(data)//1024}KB)")
            else:
                print(f"  {ep}: tiny response ({len(data)}b) — cookie expired or wrong URL?"); break
            time.sleep(1.0)
        except urllib.error.HTTPError as e:
            print(f"  {ep}: HTTP {e.code} (cookie expired?)"); break
    print(f"downloaded {got} new replays -> {OUT}")


if __name__ == "__main__":
    main()
