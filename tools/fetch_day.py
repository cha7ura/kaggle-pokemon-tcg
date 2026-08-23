"""Stream a day's episodes into replays.sqlite WITHOUT the 429-blocked ListEpisodes API.
Enumerate episode ids via `dataset_list_files` (paginated, not rate-limited), then per-file
pull()+ingest-with-delete (peak disk ~one 4MB file). Disk-guarded for the tight free space.

  python tools/fetch_day.py 2026-06-24 [cap]      # default cap 1500

The whole day is ~21GB raw; we stream a capped sample (the day's avg Elo is high, so games are
strong). Winner-filtering happens later at decision extraction.
"""
import sys, os, time, datetime, shutil, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_dataset as fd
from tools import replays_db
from kaggle.api.kaggle_api_extended import KaggleApi

MIN_FREE_GB = 1.0
_LOCK = threading.Lock()           # sqlite single-writer: serialize ingests across download workers


def _download(ep, day):
    """Download one episode json (parallel-safe; no db). Returns path or None."""
    if replays_db.has(ep):
        return "dup"
    p = f"{fd.TMP}/{ep}.json"
    for delta in (0, -1, 1):
        d = day + datetime.timedelta(days=delta)
        try:
            subprocess.run([sys.executable, "-m", "kaggle.cli", "datasets", "download",
                            f"{fd.DS}{d}", "-f", f"{ep}.json", "-p", fd.TMP],
                           capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue
        z = f"{p}.zip"
        if os.path.exists(z):
            subprocess.run(["unzip", "-o", z, "-d", fd.TMP], capture_output=True); os.remove(z)
        if os.path.exists(p):
            return p
    return None


def free_gb():
    return shutil.disk_usage(".").free / 1e9


def _list_files(api, slug, token):
    """dataset_list_files with 429 backoff. 404 (no dataset) -> None (empty day, legit stop);
    other errors re-raise so a real failure isn't mistaken for an empty day."""
    for a in range(8):
        try:
            return api.dataset_list_files(slug, page_token=token, page_size=200)
        except Exception as e:
            s = str(e)
            if "404" in s:
                return None
            if "429" in s:
                time.sleep(15 * (a + 1)); continue  # ponytail: linear backoff 15..120s (~9min total) rides out 429 waves
            raise
    raise RuntimeError(f"429 persisted after retries: {slug}")


def enum_ids(day, cap):
    import socket
    socket.setdefaulttimeout(60)                      # ponytail: enum HTTP has no timeout -> hangs under throttle; 60s cap
    api = KaggleApi(); api.authenticate()
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    ids, token = [], None
    while len(ids) < cap:
        r = _list_files(api, slug, token)
        if r is None:
            break
        files = getattr(r, "files", None) or []
        if not files:
            break
        for f in files:
            nm = getattr(f, "name", "")
            if nm.endswith(".json"):
                ids.append(int(nm[:-5]))
        token = getattr(r, "next_page_token", None) or getattr(r, "nextPageToken", None)
        if not token:
            break
    return ids[:cap]


def main():
    day_s = sys.argv[1] if len(sys.argv) > 1 else "2026-06-24"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    y, m, d = map(int, day_s.split("-"))
    day = datetime.date(y, m, d)
    os.makedirs(fd.TMP, exist_ok=True)
    print(f"enumerating {day_s} (cap {cap})...", flush=True)
    ids = enum_ids(day_s, cap)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print(f"got {len(ids)} episode ids; {workers} workers; free disk {free_gb():.1f}GB", flush=True)
    ctr = {"new": 0, "dup": 0, "miss": 0, "done": 0}
    stop = threading.Event()

    def work(ep):
        if stop.is_set():
            return
        if free_gb() < MIN_FREE_GB:
            stop.set(); return
        r = _download(ep, day)
        with _LOCK:
            ctr["done"] += 1
            if r == "dup":
                ctr["dup"] += 1
            elif r:
                replays_db.ingest_file(r, source="leader", delete=True)   # locked: single sqlite writer
                ctr["new"] += 1
            else:
                ctr["miss"] += 1
            if ctr["done"] % 100 == 0:
                print(f"  {ctr['done']}/{len(ids)}: +{ctr['new']} new, {ctr['dup']} dup, "
                      f"{ctr['miss']} miss, free {free_gb():.1f}GB", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, ids))
    print(f"DONE {day_s}: +{ctr['new']} new, {ctr['dup']} dup, {ctr['miss']} miss, "
          f"free {free_gb():.1f}GB", flush=True)


if __name__ == "__main__":
    main()
