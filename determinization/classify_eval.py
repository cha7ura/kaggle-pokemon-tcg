"""Step 1: frequency-fingerprint archetype classifier vs single-signature-card baseline.
Held-out, CPU-only. Score revealed cards against each archetype's prior (log-likelihood)
instead of keying off one signature card."""
import sqlite3, csv, random, math
from collections import Counter, defaultdict
random.seed(0)
c=sqlite3.connect('replays.sqlite')
names={int(r['cardId']):r['name'] for r in csv.DictReader(open('autoresearch/cards_full.csv'))}

seen=set(); arch_decks=defaultdict(list)
for sig,arch in c.execute('select distinct deck_sig,archetype from replay_field'):
    if sig in seen: continue
    seen.add(sig); ids=[int(x) for x in sig.split('_')]
    if len(ids)==60: arch_decks[arch].append(ids)

# 80/20 split
train=defaultdict(list); test=[]
for a,decks in arch_decks.items():
    random.shuffle(decks); k=int(len(decks)*0.8)
    train[a]=decks[:k]
    for d in decks[k:]: test.append((a,d))
ARCHES=[a for a in train if train[a]]

# prior: P(card present | archetype), smoothed; also base rate of archetype
present={a:Counter() for a in ARCHES}
for a in ARCHES:
    for d in train[a]:
        for cid in set(d): present[a][cid]+=1
n_train={a:len(train[a]) for a in ARCHES}
base={a:n_train[a]/sum(n_train.values()) for a in ARCHES}
# vocabulary
vocab=set()
for a in ARCHES: vocab|=set(present[a])
def p_present(a,cid):  # Laplace-smoothed
    return (present[a].get(cid,0)+1)/(n_train[a]+2)

# --- baseline: single signature card (our old infer_archetype) ---
def infer_signature(revealed):
    s=set(revealed)
    for cid in s:
        if 'Bellibolt' in names.get(cid,''): return 'Bellibolt'
    for cid in s:
        nm=names.get(cid,'')
        if nm=='Alakazam Powerful Hand' or cid==743: return 'Alakazam'
        if nm=="Hop's Trevenant": return 'Trevenant'
        if nm=='Dragapult ex': return 'Dragapult'
        if cid==678: return 'MegaLucario'
        if cid==1031: return 'MegaStarmie'
        if cid==345: return 'Crustle'
    return None

# --- new: fingerprint log-likelihood over PRESENT revealed cards ---
def infer_fingerprint(revealed):
    rs=set(revealed)
    best=None; bestlp=-1e18
    for a in ARCHES:
        lp=math.log(base[a])
        for cid in rs:
            if cid in vocab:
                lp+=math.log(p_present(a,cid))
        if lp>bestlp: bestlp=lp; best=a
    return best

# hybrid: signature card wins if present (unambiguous), else fingerprint
def infer_hybrid(revealed):
    s=infer_signature(revealed)
    return s if s is not None else infer_fingerprint(revealed)

res=defaultdict(lambda: defaultdict(list))
for R in (4,6,10,15,20):
    for true_arch,deck in test:
        idx=list(range(60)); random.shuffle(idx); rev=[deck[i] for i in idx[:R]]
        res[R]['sig'].append(1 if infer_signature(rev)==true_arch else 0)
        res[R]['fp'].append(1 if infer_fingerprint(rev)==true_arch else 0)
        res[R]['hyb'].append(1 if infer_hybrid(rev)==true_arch else 0)
        res[R]['sig_none'].append(1 if infer_signature(rev) is None else 0)

print(f"Archetype-ID accuracy ({len(test)} held-out decks)\n")
print(f"{'revealed':>9}{'signature':>11}{'(unknown%)':>11}{'fingerprint':>13}{'hybrid':>9}")
for R in (4,6,10,15,20):
    sig=sum(res[R]['sig'])/len(res[R]['sig'])
    non=sum(res[R]['sig_none'])/len(res[R]['sig_none'])
    fp=sum(res[R]['fp'])/len(res[R]['fp'])
    hyb=sum(res[R]['hyb'])/len(res[R]['hyb'])
    print(f"{R:>9}{sig*100:10.0f}%{non*100:10.0f}%{fp*100:12.0f}%{hyb*100:8.0f}%")
