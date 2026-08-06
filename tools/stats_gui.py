"""Tiny stdlib stats GUI over replays.sqlite. No deps.

  python tools/stats_gui.py        # serve at http://localhost:8765
  python tools/stats_gui.py 9000   # custom port

Endpoints: / (page), /api/summary, /api/matchups, /api/field, /api/policies.
ponytail: http.server + sqlite3, one file. Swap for a real framework only if this
needs auth, write paths, or many concurrent users.
"""
import json, os, sqlite3, sys, collections, csv
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/replays.sqlite"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replays_db import _decompress                    # shared blob codec (lzma/xz new, zlib legacy)
US = "The Debauchery Tea Party"

# archetype signature (same mapping as extract_field)
_names = {int(r["cardId"]): r["name"] for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}
SIG = {743: "Alakazam", 678: "MegaLucario", 1031: "MegaStarmie", 345: "Crustle", 121: "Dragapult"}
for cid, nm in _names.items():
    if nm == "Hop's Trevenant": SIG.setdefault(cid, "Trevenant")
    if "Bellibolt" in nm: SIG.setdefault(cid, "Bellibolt")


def _arch(deck):
    s = set(deck)
    for cid, nm in SIG.items():
        if cid in s: return nm
    return "OTHER"


def _db():
    return sqlite3.connect(DB)


_MATCH_CACHE = {}


def summary():
    db = _db()
    out = {}
    for t in ("replays", "cards", "decks", "policies"):
        out[t] = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    out["by_source"] = dict(db.execute("SELECT source, COUNT(*) FROM replays GROUP BY source"))
    out["db_mb"] = round(os.path.getsize(DB) / 1e6, 1)
    db.close()
    return out


def matchups():
    """Our Trevenant win rate by opponent archetype (decompresses ours games once, cached)."""
    if _MATCH_CACHE:
        return _MATCH_CACHE
    db = _db()
    W = collections.Counter(); N = collections.Counter()
    for r0, r1, blob in db.execute("SELECT reward0,reward1,blob FROM replays WHERE source='ours'"):
        g = json.loads(_decompress(blob)); tn = g.get("info", {}).get("TeamNames", ["", ""])
        if US not in tn: continue
        me = 0 if tn[0] == US else 1; opp = 1 - me
        odeck = next((s[opp]["action"] for s in g["steps"]
                      if isinstance(s[opp].get("action"), list) and len(s[opp]["action"]) == 60), None)
        if odeck is None: continue
        a = _arch(odeck); mine = (r0, r1)[me]; his = (r0, r1)[opp]
        N[a] += 1
        if mine is not None and his is not None and mine > his: W[a] += 1
    db.close()
    rows = [{"archetype": a, "wins": W[a], "games": n, "winrate": round(W[a] / n, 3)}
            for a, n in N.most_common()]
    _MATCH_CACHE["rows"] = rows
    _MATCH_CACHE["overall"] = {"games": sum(N.values()), "wins": sum(W.values()),
                               "winrate": round(sum(W.values()) / max(sum(N.values()), 1), 3)}
    return _MATCH_CACHE


def field():
    db = _db()
    rows = [{"archetype": a, "count": c} for a, c in
            db.execute("SELECT archetype, SUM(count) FROM decks GROUP BY archetype ORDER BY 2 DESC")]
    db.close()
    return {"rows": rows}


def policies():
    db = _db()
    rows = [{"deck_sig": s[:24] + "…", "versions": v, "accuracy": acc}
            for s, v, acc in db.execute(
                "SELECT deck_sig, MAX(version), AVG(accuracy) FROM policies GROUP BY deck_sig")]
    db.close()
    return {"rows": rows, "count": len(rows)}


_LEAGUE_CACHE = {}


def league(batch="full182-n30"):
    """Archetype-vs-archetype win-rate matrix from the latest full league batch (cached)."""
    if batch in _LEAGUE_CACHE:
        return _LEAGUE_CACHE[batch]
    db = _db()
    arch = {fn: a for fn, a in db.execute("SELECT fname,archetype FROM decks")}
    W = collections.defaultdict(float); N = collections.defaultdict(int)
    for a, b, w in db.execute("SELECT deck_a,deck_b,winner FROM league_games WHERE batch=?", (batch,)):
        aa, ab = arch.get(a), arch.get(b)
        if not aa or not ab:
            continue
        W[(aa, ab)] += 1.0 if w == 0 else 0.5 if w == 2 else 0.0; N[(aa, ab)] += 1
        W[(ab, aa)] += 1.0 if w == 1 else 0.5 if w == 2 else 0.0; N[(ab, aa)] += 1
    db.close()
    archs = sorted(set(arch.values()))
    # rank archetypes by mean win rate vs all others
    rank = sorted(archs, key=lambda x: -(sum(W[(x, y)] for y in archs if N[(x, y)]) /
                                         max(sum(N[(x, y)] for y in archs), 1)))
    matrix = [{"a": x, "cells": [{"b": y, "wr": round(W[(x, y)] / N[(x, y)], 3) if N[(x, y)] else None}
                                 for y in rank]} for x in rank]
    out = {"archs": rank, "matrix": matrix, "batch": batch}
    _LEAGUE_CACHE[batch] = out
    return out


PAGE = """<!doctype html><html><head><meta charset=utf8><title>pokemon stats</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}
h1{font-size:20px}h2{font-size:15px;color:#9ad;margin-top:28px}
table{border-collapse:collapse;margin:6px 0}td,th{padding:4px 12px;text-align:right;border-bottom:1px solid #2a2d34}
th:first-child,td:first-child{text-align:left}
.bar{height:10px;background:#3a6;border-radius:2px;display:inline-block;vertical-align:middle}
.lose{background:#c54}.win{background:#3a6}.mid{background:#ca5}
.pill{background:#1b1e26;padding:2px 8px;border-radius:10px;margin-right:6px}
small{color:#789}
</style></head><body>
<h1>Pokémon TCG — ladder stats <small id=db></small></h1>
<div id=sum></div>
<h2>Our Trevenant — win rate by opponent archetype</h2><div id=match></div>
<h2>League: archetype vs archetype (row beats column)</h2><div id=league></div>
<h2>Field composition (weighted)</h2><div id=field></div>
<h2>Policy coverage</h2><div id=pol></div>
<script>
const j=u=>fetch(u).then(r=>r.json());
const bar=(p,cls)=>`<span class=bar style="width:${Math.round(p*120)}px" class="${cls}"></span>`;
j('/api/summary').then(s=>{
  document.getElementById('db').textContent=`· ${s.db_mb} MB · ${s.replays} replays`;
  document.getElementById('sum').innerHTML=Object.entries({replays:s.replays,cards:s.cards,
    decks:s.decks,policies:s.policies,...s.by_source}).map(([k,v])=>
    `<span class=pill>${k}: <b>${v}</b></span>`).join('');
});
j('/api/matchups').then(m=>{
  let h=`<p>overall <b>${m.overall.winrate}</b> (${m.overall.wins}/${m.overall.games})</p>`;
  h+='<table><tr><th>vs</th><th>winrate</th><th></th><th>n</th></tr>';
  for(const r of m.rows){let c=r.winrate>=0.55?'win':r.winrate<=0.45?'lose':'mid';
    h+=`<tr><td>${r.archetype}</td><td>${r.winrate}</td><td><span class=bar style="width:${Math.round(r.winrate*120)}px;background:${c=='win'?'#3a6':c=='lose'?'#c54':'#ca5'}"></span></td><td>${r.games}</td></tr>`;}
  document.getElementById('match').innerHTML=h+'</table>';
});
j('/api/league').then(L=>{
  let h=`<small>batch ${L.batch}</small><table><tr><th></th>`;
  for(const a of L.archs)h+=`<th>${a.slice(0,4)}</th>`;
  h+='</tr>';
  for(const row of L.matrix){h+=`<tr><td>${row.a}</td>`;
    for(const c of row.cells){if(c.wr==null){h+='<td>·</td>';continue;}
      let g=Math.round(c.wr*180+40),r=Math.round((1-c.wr)*180+40);
      h+=`<td style="background:rgb(${r},${g},60);color:#000">${c.wr.toFixed(2)}</td>`;}
    h+='</tr>';}
  document.getElementById('league').innerHTML=h+'</table>';
});
j('/api/field').then(f=>{let mx=Math.max(...f.rows.map(r=>r.count));
  let h='<table>';for(const r of f.rows)h+=`<tr><td>${r.archetype}</td><td>${r.count}</td><td><span class=bar style="width:${Math.round(r.count/mx*160)}px;background:#46a"></span></td></tr>`;
  document.getElementById('field').innerHTML=h+'</table>';
});
j('/api/policies').then(p=>{
  let h=`<p><b>${p.count}</b> decks with a learned policy</p><table><tr><th>deck</th><th>versions</th><th>acc</th></tr>`;
  for(const r of p.rows)h+=`<tr><td>${r.deck_sig}</td><td>${r.versions}</td><td>${r.accuracy?r.accuracy.toFixed(3):'—'}</td></tr>`;
  document.getElementById('pol').innerHTML=h+'</table>';
});
</script></body></html>"""

ROUTES = {"/api/summary": summary, "/api/matchups": matchups,
          "/api/field": field, "/api/policies": policies, "/api/league": league}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/" or self.path == "":
            body = PAGE.encode(); ctype = "text/html"
        elif self.path in ROUTES:
            body = json.dumps(ROUTES[self.path]()).encode(); ctype = "application/json"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"stats GUI → http://localhost:{port}  (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
