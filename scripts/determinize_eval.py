"""Validate archetype-prior determinization vs the old basic-energy-filler baseline.

Test protocol (held-out, CPU-only, no engine):
  For each held-out deck: reveal the first R cards a player would typically expose early
  (we simulate 'revealed' by sampling R cards from the deck), then predict the remaining
  60-R hidden cards. Score = how many of the true hidden cards the method recovers.
    - filler baseline: predict all-basic-energy for the hidden slots (the old approach)
    - archetype prior: infer archetype from revealed cards, fill hidden slots with the
      archetype's most-likely remaining cards (by mean_copies, consistent with revealed)
"""
import sqlite3, csv, json, random
from collections import Counter, defaultdict
random.seed(0)
c=sqlite3.connect('replays.sqlite')
names={int(r['cardId']):r['name'] for r in csv.DictReader(open('autoresearch/cards_full.csv'))}
BASIC_ENERGY={int(r['cardId']) for r in csv.DictReader(open('autoresearch/cards_full.csv')) if r.get('cardType')=='BASIC_ENERGY'}
FILLER=1  # basic grass energy (old approach)

# load unique decks per archetype
seen=set(); arch_decks=defaultdict(list); all_decks=[]
for sig,arch in c.execute('select distinct deck_sig,archetype from replay_field'):
    if sig in seen: continue
    seen.add(sig); ids=[int(x) for x in sig.split('_')]
    if len(ids)==60: arch_decks[arch].append((sig,ids)); all_decks.append((arch,ids))

# split train/test 80/20 per archetype
train=defaultdict(list); test=[]
for a,decks in arch_decks.items():
    random.shuffle(decks); k=int(len(decks)*0.8)
    train[a]=[d for _,d in decks[:k]]
    for _,d in decks[k:]: test.append((a,d))

# build priors from TRAIN only
def build_prior(decks):
    n=len(decks); pres=Counter(); cop=Counter()
    for d in decks:
        cc=Counter(d)
        for cid,k in cc.items(): pres[cid]+=1; cop[cid]+=k
    return {'n':n,'mean_copies':{cid:cop[cid]/n for cid in cop},'p_present':{cid:pres[cid]/n for cid in pres}}
priors={a:build_prior(train[a]) for a in train if train[a]}
# archetype signature cards for classification (name-based, same as our archetype())
def infer_archetype(revealed_ids):
    s=set(revealed_ids)
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
    return None  # unknown yet

def predict_prior(revealed, hidden_count, arch_guess):
    """Return a multiset (list) of hidden_count predicted card ids using the archetype prior,
    subtracting what's already revealed."""
    if arch_guess not in priors:
        return [FILLER]*hidden_count
    mc=priors[arch_guess]['mean_copies']
    revealed_c=Counter(revealed)
    # expected full-deck copies = round(mean_copies); remaining = expected - revealed
    cand=[]
    for cid,m in sorted(mc.items(), key=lambda kv:-kv[1]):
        exp=round(m)
        rem=exp - revealed_c.get(cid,0)
        cand += [cid]*max(0,rem)
    pred=cand[:hidden_count]
    if len(pred)<hidden_count: pred += [FILLER]*(hidden_count-len(pred))
    return pred

def recall(pred, true_hidden):
    """multiset overlap / total hidden"""
    p=Counter(pred); t=Counter(true_hidden)
    inter=sum(min(p[k],t[k]) for k in t)
    return inter/sum(t.values())

# run test: reveal R cards, predict rest
results=defaultdict(lambda:{'filler':[], 'prior':[], 'arch_ok':[]})
for R in (6,10,15,20):
    for true_arch,deck in test:
        idx=list(range(60)); random.shuffle(idx)
        rev=[deck[i] for i in idx[:R]]; hid=[deck[i] for i in idx[R:]]
        guess=infer_archetype(rev)
        results[R]['arch_ok'].append(1 if guess==true_arch else 0)
        results[R]['filler'].append(recall([FILLER]*len(hid), hid))
        results[R]['prior'].append(recall(predict_prior(rev,len(hid),guess), hid))

print(f"Held-out determinization test ({len(test)} test decks)\n")
print(f"{'revealed':>9}{'arch-id acc':>13}{'filler recall':>15}{'prior recall':>14}{'lift':>8}")
for R in (6,10,15,20):
    a=sum(results[R]['arch_ok'])/len(results[R]['arch_ok'])
    f=sum(results[R]['filler'])/len(results[R]['filler'])
    p=sum(results[R]['prior'])/len(results[R]['prior'])
    print(f"{R:>9}{a*100:12.0f}%{f*100:14.1f}%{p*100:13.1f}%{(p-f)*100:+7.1f}")
