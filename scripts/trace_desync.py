"""Capture the exact step sequence of ONE failing rollout to root-cause the error-5 desync.
Runs games until a rollout raises error 5, logs every step's (n_option, action, select_type,
context, minCount, maxCount, yourIndex) up to the failure, then stops. Prints the trace."""
import os, sys, random
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(ROOT,"sdk")); sys.path.insert(0,os.path.join(ROOT,"autoresearch"))
from cg.game import battle_start, battle_select, battle_finish
from cg.api import (to_observation_class, OptionType, SelectType, SelectContext,
                    search_begin, search_step, search_end)
import agent_lucario as H
import agent_ismcts_prior as S

TRACE=[]; CAUGHT=[False]

def traced_rollout(obs, first_action, me, depth, rng):
    ss = search_begin(obs, *S._determinize(obs.current, rng)); sid=ss.searchId
    local=[]
    try:
        for stepi in range(depth+1):
            sel = ss.observation.select
            if stepi==0:
                act=first_action
            else:
                cur=ss.observation.current
                if cur is None or cur.result!=-1 or sel is None: break
                act=S._safe_action(ss.observation)
            # record what we're about to send
            rec=dict(step=stepi,
                     n_option=(len(sel.option) if sel else None),
                     action=list(act),
                     stype=(int(sel.type) if sel else None),
                     ctx=(int(getattr(sel,'context',-1)) if sel else None),
                     minC=(getattr(sel,'minCount',None) if sel else None),
                     maxC=(getattr(sel,'maxCount',None) if sel else None),
                     yourIdx=(ss.observation.current.yourIndex if ss.observation.current else None))
            try:
                ss = search_step(sid, act); rec['ok']=True; local.append(rec)
            except Exception as e:
                rec['ok']=False; rec['err']=str(e); local.append(rec)
                if not CAUGHT[0]:
                    CAUGHT[0]=True; TRACE.extend(local)
                break
        return 0.0
    finally:
        search_end()

S._rollout = traced_rollout  # monkeypatch

def play(seed):
    random.seed(seed)
    deck=[int(l) for l in open('autoresearch/decks/lucario_meta.csv') if l.strip()][:60]
    obs,sd=battle_start(deck,deck)
    steps=0
    try:
        while not CAUGHT[0]:
            cur=obs['current']
            if cur is not None and cur.get('result',-1)!=-1: return
            sel=obs['select']
            if sel is None: return
            seat=cur['yourIndex']
            o=to_observation_class(obs)
            is_main = (int(sel['type'])==int(SelectType.MAIN) and sel.get('maxCount')==1 and sel.get('minCount')==1)
            if seat==0 and is_main:
                action=S._ismcts(o, sims=24, depth=12)
            else:
                action=H.agent(obs)
            obs=battle_select(action); steps+=1
            if steps>20000: return
    finally:
        battle_finish()

for s in range(30):
    play(s)
    if CAUGHT[0]: break

print(f"caught={CAUGHT[0]}, trace of failing rollout ({len(TRACE)} steps):")
for r in TRACE:
    print(f"  step{r['step']}: n_opt={r['n_option']} act={r['action']} type={r['stype']} ctx={r['ctx']} "
          f"min={r['minC']} max={r['maxC']} yourIdx={r['yourIdx']} ok={r['ok']}"
          + (f"  ERR={r.get('err')}" if not r['ok'] else ""))
