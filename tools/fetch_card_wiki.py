"""Enrich the engine card data (data/cards_engine.json) with Bulbapedia per-card info via the
MediaWiki API: resolves each card's page URL + set/number + Rulings. Resumable, rate-limited, polite.

  python tools/fetch_card_wiki.py            # crawl all cards (resumes from cache), then merge
  python tools/fetch_card_wiki.py --merge    # merge cache -> cards.json + cards_text.csv only
  python tools/fetch_card_wiki.py --limit 20 # crawl only first N uncached (smoke test)

Cache: data/wiki_cache.json  (card_id -> {title,url,set,number,rulings})  — never re-fetches a hit.
Outputs: data/cards.json (engine + wiki), data/cards_text.csv (adds wiki_url + set columns).
Network only here; nothing network-bound ships in the submission.
"""
import urllib.request, urllib.parse, json, os, sys, time, re, csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "autoresearch", "data")
API = "https://bulbapedia.bulbagarden.net/w/api.php"
WIKI = "https://bulbapedia.bulbagarden.net/wiki/"
UA = "pokemon-tcg-research/1.0 (educational card-data enrichment; polite, rate-limited)"
CACHE = os.path.join(DATA, "wiki_cache.json")


def _get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.loads(urllib.request.urlopen(req, timeout=25).read())


ESYM = {"{G}": "Grass", "{R}": "Fire", "{W}": "Water", "{L}": "Lightning", "{P}": "Psychic",
        "{F}": "Fighting", "{D}": "Darkness", "{M}": "Metal", "{C}": "Colorless", "{N}": "Dragon",
        "{Y}": "Fairy"}


def norm_name(name):
    """Engine names use energy symbols like 'Basic {G} Energy' -> 'Basic Grass Energy' for wiki match."""
    for k, v in ESYM.items():
        name = name.replace(k, v)
    return re.sub(r"\s+", " ", name).strip()


def _search(srsearch, **extra):
    p = {"action": "query", "list": "search", "format": "json", "srlimit": "15", "srsearch": srsearch}
    p.update(extra)
    try:
        r = _get(p)
        return [h["title"] for h in r.get("query", {}).get("search", [])]
    except Exception:
        return []


def resolve_title(name):
    """Best Bulbapedia card page title for an engine card name. Prefers a print page
    'Name (Set Number)', then any 'Name (...)' (incl. '(TCG)' for energies), then first hit."""
    nm = norm_name(name)
    hits = _search(f'intitle:"{nm}"', srnamespace="0") or _search(nm)
    pref = nm.lower() + " ("
    prints = [t for t in hits if t.lower().startswith(pref) and re.search(r"\(.+\s[0-9A-Za-z]+\)$", t)]
    if prints:
        return prints[0]
    starts = [t for t in hits if t.lower().startswith(pref) or t.lower() == nm.lower()]
    if starts:
        return starts[0]
    return hits[0] if hits else None


def parse_page(title):
    """Return {url, set, number, rulings} for a resolved card title."""
    out = {"title": title, "url": WIKI + urllib.parse.quote(title.replace(" ", "_")),
           "set": None, "number": None, "rulings": []}
    try:
        d = _get({"action": "parse", "format": "json", "prop": "wikitext", "page": title})
        wt = d.get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception:
        return out
    m = re.search(r"\|expansion=\{\{TCG\|([^}]+)\}\}", wt)
    if m:
        out["set"] = m.group(1).strip()
    m = re.search(r"\|cardno=([0-9A-Za-z]+)/", wt)
    if m:
        out["number"] = m.group(1).strip()
    rm = re.search(r"==\s*Rulings\s*==(.*?)(?:\n==|\Z)", wt, re.S)
    if rm:
        bullets = re.findall(r"^\*\s*(.+)$", rm.group(1), re.M)
        clean = []
        for b in bullets:
            b = re.sub(r"<ref[^>]*>.*?</ref>", "", b, flags=re.S)
            b = re.sub(r"<ref[^>]*/?>", "", b)
            b = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", b)   # [[a|b]] -> b
            b = re.sub(r"\{\{[^}]*\}\}", "", b)
            b = re.sub(r"'''?", "", b).strip()
            if b:
                clean.append(b)
        out["rulings"] = clean[:8]
    return out


def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def save_cache(c):
    json.dump(c, open(CACHE, "w"), ensure_ascii=False)


def merge():
    cards = json.load(open(os.path.join(DATA, "cards_engine.json")))
    # prefer cards.json if effect_flags already added, so we don't clobber them
    cj = os.path.join(DATA, "cards.json")
    if os.path.exists(cj):
        flagged = {r["card_id"]: r for r in json.load(open(cj))}
        for r in cards:
            r.update({k: flagged[r["card_id"]][k] for k in
                      ("effect_flags", "has_setup_attack", "flag_needs_review")
                      if r["card_id"] in flagged and k in flagged[r["card_id"]]})
    cache = load_cache()
    for r in cards:
        w = cache.get(str(r["card_id"]))
        if w:
            r["wiki_url"] = w.get("url")
            r["set"] = w.get("set")
            r["number"] = w.get("number")
            r["rulings"] = w.get("rulings", [])
        else:
            r.setdefault("wiki_url", None)
    json.dump(cards, open(cj, "w"), ensure_ascii=False, indent=0)

    def clean(s):
        return (s or "").replace("\xa0", " ").replace("\n", " ").strip()
    with open(os.path.join(DATA, "cards_text.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "name", "card_type", "set", "number", "wiki_url", "hp", "stage",
                    "energy_type", "weakness", "retreat", "prize", "ability_text", "attacks",
                    "effect_text", "rulings"])
        for r in sorted(cards, key=lambda x: x["card_id"]):
            ab = " || ".join(f"{a['name']}: {clean(a['text'])}" for a in (r.get("abilities") or []))
            at = " || ".join(f"{a['name']}({a['damage_base']}/{'+'.join(a['energy_cost'])}): "
                             f"{clean(a['damage_text'])}" for a in (r.get("attacks") or []))
            w.writerow([r["card_id"], r["name"], r["card_type"], r.get("set"), r.get("number"),
                        r.get("wiki_url"), r["hp"], r["stage"], r["energy_type"], r["weakness"],
                        r["retreat_cost"], r["prize_value"], ab, at, clean(r.get("effect_text")),
                        " ; ".join(r.get("rulings", []))])
    have = sum(1 for r in cards if r.get("wiki_url"))
    print(f"merged: {have}/{len(cards)} cards have a wiki_url")


def crawl(limit=None, sleep=0.6):
    cards = json.load(open(os.path.join(DATA, "cards_engine.json")))
    cache = load_cache()
    todo = [r for r in cards if str(r["card_id"]) not in cache]
    if limit:
        todo = todo[:limit]
    print(f"crawl: {len(todo)} cards to fetch ({len(cache)} cached); ~{len(todo)*sleep*2/60:.0f} min",
          flush=True)
    for i, r in enumerate(todo, 1):
        cid, name = str(r["card_id"]), r["name"]
        try:
            title = resolve_title(name)
            cache[cid] = parse_page(title) if title else {"title": None, "url": None,
                                                          "set": None, "number": None, "rulings": []}
        except Exception as e:
            cache[cid] = {"title": None, "url": None, "set": None, "number": None,
                          "rulings": [], "error": str(e)[:80]}
        time.sleep(sleep)
        if i % 25 == 0:
            save_cache(cache)
            hits = sum(1 for v in cache.values() if v.get("url"))
            print(f"  {i}/{len(todo)} done; {hits} urls total", flush=True)
    save_cache(cache)
    merge()


def main():
    os.makedirs(DATA, exist_ok=True)
    if "--merge" in sys.argv:
        merge(); return
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    crawl(limit=limit)


if __name__ == "__main__":
    main()
