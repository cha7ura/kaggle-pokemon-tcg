"""Train a board VALUE model: state-features -> P(win), from top replays. One row per decision
state, label = did that player win the game. Simpler target than move-imitation (which failed).

PARALLELIZED: the per-state cg.api parse (to_observation_class) is the bottleneck, so decode
across processes (rowid-chunked, like build_field_table) then train xgboost in the parent.
    PYTHONPATH=/app:/app/sdk VALUE_LIMIT=25000 python tools/build_value.py   (docker, linux/amd64)
"""
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

import os, sqlite3, time
from concurrent.futures import ProcessPoolExecutor
from tools.replays_db import iter_replays, DB   # 59k-replay sqlite, NOT the emptied json/ dir

WORKERS=max(2,(os.cpu_count() or 4)-1)


def process_chunk(bounds):
    """Decode a rowid range in a worker process: parse every decision state, label = did-this-seat-win.
    Returns (X_chunk, y_chunk) numpy arrays, or None if empty. cg.api re-imports per process."""
    lo,hi=bounds
    Xc=[]; yc=[]
    for ep,d in iter_replays("rowid BETWEEN ? AND ?",(lo,hi)):
        rw=d.get("rewards",[0,0])
        if not rw or len(rw)<2 or rw[0] is None or rw[1] is None or rw[0]==rw[1]:
            continue                       # null/draw rewards (some bulk replays) -> skip game
        for p in (0,1):
            won=1.0 if rw[p]>rw[1-p] else 0.0
            for s in d.get("steps",[]):
                ob=s[p].get("observation")
                if not isinstance(ob,dict) or not ob.get("select"): continue
                try: o=to_observation_class(ob)
                except Exception: continue
                try: Xc.append(ctx(o)); yc.append(won)
                except Exception: continue
    if not yc: return None
    return np.asarray(Xc,np.float32), np.asarray(yc,np.float32)


def main():
    lim=int(os.environ.get("VALUE_LIMIT","25000"))
    con=sqlite3.connect(DB)
    hi=min(con.execute("SELECT MAX(rowid) FROM replays").fetchone()[0] or 0, lim)
    con.close()
    CHUNKS=WORKERS*4
    step=max(1,hi//CHUNKS+1)
    bounds=[(lo,min(lo+step-1,hi)) for lo in range(1,hi+1,step)]
    t0=time.time(); Xs=[]; ys=[]; rows=0
    print(f"value-net PARALLEL: ~{hi} games, {len(bounds)} chunks x {WORKERS} procs",flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i,res in enumerate(ex.map(process_chunk,bounds),1):
            if res is not None:
                xc,yc=res; Xs.append(xc); ys.append(yc); rows+=len(yc)
            if i%5==0 or i==len(bounds):
                print(f"  chunk {i}/{len(bounds)} | rows {rows} | {time.time()-t0:.0f}s",flush=True)
    X=np.vstack(Xs); y=np.concatenate(ys)
    print(f"value rows={len(y)} winrate={y.mean():.2f} dim={X.shape[1]} in {time.time()-t0:.0f}s",flush=True)

    import xgboost as xgb
    n=len(y); idx=np.random.RandomState(0).permutation(n); tr,te=idx[:int(n*.85)],idx[int(n*.85):]
    bst=xgb.train({"objective":"binary:logistic","max_depth":5,"eta":0.2,"subsample":0.8,"eval_metric":"auc"},
      xgb.DMatrix(X[tr],label=y[tr]),num_boost_round=150)
    pred=bst.predict(xgb.DMatrix(X[te]))
    acc=((pred>0.5)==(y[te]>0.5)).mean()
    order=np.argsort(pred); ranks=np.empty(len(pred)); ranks[order]=np.arange(len(pred))
    pos=y[te]==1; auc=(ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum())
    os.makedirs("/app/weights",exist_ok=True)
    bst.save_model("/app/weights/value_alakazam.json")
    print(f"VALUE net: held-out acc={acc:.3f} auc={auc:.3f} -> saved",flush=True)


if __name__=="__main__":
    main()
