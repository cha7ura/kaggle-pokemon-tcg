"""Train a board VALUE model: state-features -> P(win), from top replays. One row per decision
state, label = did that player win the game. Simpler target than move-imitation (which failed)."""
import json,glob,numpy as np
from cg.api import to_observation_class, all_card_data, all_attack
CD={c.cardId:c for c in all_card_data()}; ATK={a.attackId:a for a in all_attack()}
NE=12; SF=6+NE+2+4
def oh(i,n):
    v=np.zeros(n,np.float32); 
    if i is not None and 0<=int(i)<n: v[int(i)]=1
    return v
def cf(cid):
    c=CD.get(cid)
    if not c: return np.zeros(SF,np.float32)
    bd=max([ATK[a].damage for a in c.attacks if a in ATK],default=0)
    return np.concatenate([oh(int(c.cardType),6),oh(int(c.energyType),NE),
      [1.0 if c.ex else 0,1.0 if c.megaEx else 0],[c.hp/300,c.retreatCost/4,len(c.attacks)/4,bd/300]]).astype(np.float32)
CF0=np.zeros(SF,np.float32)
def mons(p): return (([p.active[0]] if p.active and p.active[0] else [])+[b for b in p.bench if b])[:6]
def ctx(o):
    st=o.current; me=st.yourIndex; my=st.players[me]; op=st.players[1-me]
    def pool(p):
        ms=mons(p)
        if not ms: return np.concatenate([CF0,[0,0]])
        return np.concatenate([np.mean([cf(m.id) for m in ms],0),[np.mean([m.hp/300 for m in ms]),np.mean([len(m.energies)/4 for m in ms])]])
    state=np.array([len(my.prize)/6,len(op.prize)/6,len(my.bench)/5,len(op.bench)/5,len(my.hand or [])/10,st.turn/40,
      my.poisoned,my.burned,my.asleep,my.paralyzed,my.confused],np.float32)
    return np.concatenate([pool(my),pool(op),state])
X,y=[],[]
for f in glob.glob("/app/json/*.json"):
    try: d=json.load(open(f))
    except: continue
    rw=d.get("rewards",[0,0])
    for p in (0,1):
        won=1.0 if rw[p]>rw[1-p] else 0.0
        for s in d["steps"]:
            ob=s[p].get("observation")
            if not isinstance(ob,dict) or not ob.get("select"): continue
            try: o=to_observation_class(ob)
            except: continue
            X.append(ctx(o)); y.append(won)
X=np.array(X,np.float32); y=np.array(y,np.float32)
print(f"value rows={len(y)} winrate={y.mean():.2f} dim={X.shape[1]}",flush=True)
import xgboost as xgb
n=len(y); idx=np.random.RandomState(0).permutation(n); tr,te=idx[:int(n*.85)],idx[int(n*.85):]
bst=xgb.train({"objective":"binary:logistic","max_depth":5,"eta":0.2,"subsample":0.8,"eval_metric":"auc"},
  xgb.DMatrix(X[tr],label=y[tr]),num_boost_round=150)
pred=bst.predict(xgb.DMatrix(X[te]))
from math import isnan
acc=((pred>0.5)==(y[te]>0.5)).mean()
# AUC
import numpy as _np
order=_np.argsort(pred); ranks=_np.empty(len(pred)); ranks[order]=_np.arange(len(pred))
pos=y[te]==1; auc=(ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum())
bst.save_model("/app/weights/value_alakazam.json")
print(f"VALUE net: held-out acc={acc:.3f} auc={auc:.3f} -> saved",flush=True)
