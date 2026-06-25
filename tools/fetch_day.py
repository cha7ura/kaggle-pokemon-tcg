"""Stream a day's episodes into replays.sqlite WITHOUT the 429-blocked ListEpisodes API.
Enumerate episode ids via `dataset_list_files` (paginated, not rate-limited), then per-file
pull()+ingest-with-delete (peak disk ~one 4MB file). Disk-guarded for the tight free space.

  python tools/fetch_day.py 2026-06-24 [cap]      # default cap 1500

The whole day is ~21GB raw; we stream a capped sample (the day's avg Elo is high, so games are
strong). Winner-filtering happens later at decision extraction.
"""
import sys, os, datetime, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_dataset as fd
from kaggle.api.kaggle_api_extended import KaggleApi

MIN_FREE_GB = 0.8


def free_gb():
    return shutil.disk_usage(".").free / 1e9


def enum_ids(day, cap):
    api = KaggleApi(); api.authenticate()
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    ids, token = [], None
    while len(ids) < cap:
        r = api.dataset_list_files(slug, page_token=token, page_size=200)
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
    print(f"got {len(ids)} episode ids; free disk {free_gb():.1f}GB", flush=True)
    got = skip = 0
    for i, ep in enumerate(ids, 1):
        if free_gb() < MIN_FREE_GB:
            print(f"STOP: low disk ({free_gb():.1f}GB)", flush=True); break
        r = fd.pull(ep, day, "leader")
        if r is True:
            got += 1
        elif r is False:
            skip += 1
        if i % 100 == 0:
            print(f"  {i}/{len(ids)}: +{got} new, {skip} dup, free {free_gb():.1f}GB", flush=True)
    print(f"DONE {day_s}: +{got} new games, {skip} dup, free {free_gb():.1f}GB", flush=True)


if __name__ == "__main__":
    main()
