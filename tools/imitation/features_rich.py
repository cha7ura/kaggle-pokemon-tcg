"""Enriched featurizer — option SEMANTICS (action type + resolved card) + resource/threat state.
STDLIB ONLY. option_features now takes (option, context, select, current, seat) to resolve cards.
"""
import csv, os
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CARD={int(r["cardId"]):r for r in csv.DictReader(open(f"{_ROOT}/autoresearch/cards_full.csv"))}
def _b(r,k): return str(r.get(k,"")).lower() in ("true","1")
def _ctype(cid): return (_CARD.get(cid,{}).get("cardType","") or "").lower()

# OptionType: PLAY=7 ATTACH=8 EVOLVE=9 ABILITY=10 DISCARD=11 RETREAT=12 ATTACK=13 END=14
# AreaType:  DECK=1 HAND=2 DISCARD=3 ACTIVE=4 BENCH=5 PRIZE=6 STADIUM=7 ENERGY=8 TOOL=9
STATE_NAMES=(
 "turn","turn_action_count","supporter_played","stadium_played","energy_attached","retreated",
 "my_active_hp_frac","my_active_energy","my_active_dmg","my_active_is_ex","my_active_tools","my_active_evo_depth",
 "my_bench","my_hand","my_deck","my_discard","my_prize_left","my_energy_in_discard",
 "opp_active_hp_frac","opp_active_energy","opp_active_dmg","opp_active_is_ex",
 "opp_bench","opp_hand","opp_deck","opp_discard","opp_prize_left",
 "my_status_any","opp_status_any","prize_diff",
)
OPTION_NAMES=(
 "context","opt_type","min_count","max_count","n_options",
 "is_play","is_attach","is_evolve","is_ability","is_discard","is_retreat","is_attack","is_end",
 "card_is_pokemon","card_is_trainer","card_is_energy","card_is_basic","card_is_stage2","card_is_ex","card_hp",
)
STATE_DIM=len(STATE_NAMES); OPTION_DIM=len(OPTION_NAMES)
def feature_names(): return list(STATE_NAMES+OPTION_NAMES)
def _active(p):
    a=p.get("active") or []; return a[0] if a else None
def _plen(p): return len(p.get("prize") or [])

def state_features(current, seat):
    P=current.get("players") or [{},{}]
    me=P[seat] if seat<len(P) else {}; opp=P[1-seat] if (1-seat)<len(P) else {}
    ma,oa=_active(me),_active(opp)
    hpfrac=lambda a: (float(a.get("hp",0))/float(a.get("maxHp",1) or 1)) if a else 0.0
    energy=lambda a: float(len(a.get("energies") or [])) if a else 0.0
    dmg=lambda a: float((a.get("maxHp",0) or 0)-(a.get("hp",0) or 0)) if a else 0.0
    isex=lambda a: 1.0 if a and _b(_CARD.get(a.get("id"),{}),"ex") else 0.0
    edis=lambda p: float(sum(1 for c in (p.get("discard") or []) if "energy" in _ctype(c.get("id")) or "Energy" in (_CARD.get(c.get("id"),{}).get("name",""))))
    status=lambda p: 1.0 if any(p.get(k) for k in ("asleep","confused","paralyzed","poisoned","burned")) else 0.0
    myp,oppp=_plen(me),_plen(opp)
    return [float(current.get("turn",0)),float(current.get("turnActionCount",0)),
       1.0 if current.get("supporterPlayed") else 0.0,1.0 if current.get("stadiumPlayed") else 0.0,
       1.0 if current.get("energyAttached") else 0.0,1.0 if current.get("retreated") else 0.0,
       hpfrac(ma),energy(ma),dmg(ma),isex(ma),float(len(ma.get("tools") or []) if ma else 0),float(len(ma.get("preEvolution") or []) if ma else 0),
       float(len(me.get("bench") or [])),float(me.get("handCount",len(me.get("hand") or []))),
       float(me.get("deckCount",0)),float(len(me.get("discard") or [])),float(myp),edis(me),
       hpfrac(oa),energy(oa),dmg(oa),isex(oa),
       float(len(opp.get("bench") or [])),float(opp.get("handCount",0)),
       float(opp.get("deckCount",0)),float(len(opp.get("discard") or [])),float(oppp),
       status(me),status(opp),float(myp-oppp)]

def _resolve_card(o, current, seat):
    """Resolve the option's referenced card id via area+index into the player's zones."""
    area=o.get("area"); idx=o.get("index")
    if area is None or idx is None: return None
    P=current.get("players") or [{},{}]
    pidx=o.get("playerIndex", seat)
    p=P[pidx] if pidx is not None and pidx<len(P) else P[seat]
    zone={2:"hand",3:"discard",4:"active",5:"bench",6:"prize",1:"deck"}.get(area)
    if not zone: return None
    z=p.get(zone) or []
    if 0<=idx<len(z) and isinstance(z[idx],dict): return z[idx].get("id")
    return None

def option_features(option, context, select, current=None, seat=0):
    o=option or {}; t=o.get("type",-1)
    cid=_resolve_card(o,current,seat) if current is not None else None
    r=_CARD.get(cid,{}) if cid else {}; ct=_ctype(cid) if cid else ""; nm=(r.get("name","") or "").lower()
    is_poke="pokemon" in ct; is_energy=("energy" in ct) or ("energy" in nm)
    is_trainer=(not is_poke and not is_energy and cid is not None)
    return [float(context if context is not None else -1),float(t),
        float(select.get("minCount",0)),float(select.get("maxCount",0)),float(len(select.get("option") or [])),
        1.0 if t==7 else 0.0,1.0 if t==8 else 0.0,1.0 if t==9 else 0.0,1.0 if t==10 else 0.0,
        1.0 if t==11 else 0.0,1.0 if t==12 else 0.0,1.0 if t==13 else 0.0,1.0 if t==14 else 0.0,
        1.0 if is_poke else 0.0,1.0 if is_trainer else 0.0,1.0 if is_energy else 0.0,
        1.0 if _b(r,"basic") and is_poke else 0.0,1.0 if _b(r,"stage2") else 0.0,1.0 if _b(r,"ex") else 0.0,float(r.get("hp",0) or 0)]
