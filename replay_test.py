"""Historical replay + diagnostics for the MT5-native Gold Sniper engine.

Safe research mode: no AI, Telegram, or orders. Uses only data available at each
closed M5 timestamp. Reports gate funnel, SL geometry policy, trigger classes,
MFE/MAE, TP sensitivity, expiry sensitivity, and management simulations.
"""
from __future__ import annotations
import os
from datetime import timedelta
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mt5_sniper_engine as engine
from data_manager import MT5DataManager
from mt5_sniper_engine import build_candidate, diagnose_gates, MIN_SCORE, MIN_SL_DISTANCE, MIN_SL_ATR_MULT, MIN_SL_SPREAD_MULT, PIVOT_LEFT, PIVOT_RIGHT, MIN_STRUCTURE_BARS

load_dotenv()
BARS = int(os.getenv("REPLAY_BARS", "500"))
MIN_HISTORY = 40
MAX_FORWARD_BARS = int(os.getenv("REPLAY_FORWARD_BARS", "60"))
COOLDOWN_BARS = int(os.getenv("REPLAY_COOLDOWN_BARS", "6"))
PROGRESS_EVERY = int(os.getenv("REPLAY_PROGRESS_EVERY", "500"))


def _history(manager, tf, count):
    df = manager.client.bars(tf, count)
    if df.empty:
        raise RuntimeError(f"No {tf} history returned")
    return df.sort_values("Time").reset_index(drop=True)


def _fast_recent_pivots(df):
    """Replay-only pivot implementation.

    The production engine needs only the latest two confirmed highs/lows for
    structure and sweep decisions. The old replay implementation recomputed
    every historical pivot on every M5 bar, turning a 10k replay into an
    unnecessarily large O(N^2) pandas workload. Scan backwards and stop once
    two confirmed highs and two confirmed lows are found. This preserves the
    values used by structure()/sweep() while avoiding look-ahead: a pivot is
    considered only when its PIVOT_RIGHT confirmation bars are already present.
    """
    highs, lows = [], []
    if df is None or len(df) < MIN_STRUCTURE_BARS:
        return highs, lows
    h = np.asarray(df["High"], dtype=float)
    l = np.asarray(df["Low"], dtype=float)
    last_confirmed = len(df) - PIVOT_RIGHT - 1
    for i in range(last_confirmed, PIVOT_LEFT - 1, -1):
        if len(highs) < 2:
            hi = h[i]
            if hi > np.max(h[i-PIVOT_LEFT:i]) and hi > np.max(h[i+1:i+PIVOT_RIGHT+1]):
                highs.append((i, float(hi)))
        if len(lows) < 2:
            lo = l[i]
            if lo < np.min(l[i-PIVOT_LEFT:i]) and lo < np.min(l[i+1:i+PIVOT_RIGHT+1]):
                lows.append((i, float(lo)))
        if len(highs) >= 2 and len(lows) >= 2:
            break
    highs.reverse(); lows.reverse()
    return highs, lows


# Monkey-patch only the replay process. Production/live engine behavior is not
# changed. structure() and sweep() consume only the latest two pivots, so this
# is equivalent for the current v2 hybrid logic and dramatically faster.
engine.pivots = _fast_recent_pivots


def forward_stats(m5, idx, c, tp_r=None, max_forward_bars=None):
    entry, sl = float(c["entry"]), float(c["sl"])
    risk = abs(entry - sl)
    if risk <= 0:
        return "INVALID", 0, 0.0, 0.0
    if tp_r is None:
        tp = float(c["tp"])
    elif c["direction"] == "LONG":
        tp = entry + risk * float(tp_r)
    else:
        tp = entry - risk * float(tp_r)
    window = MAX_FORWARD_BARS if max_forward_bars is None else int(max_forward_bars)
    end = min(len(m5), idx + 1 + window)
    max_fav = max_adv = 0.0
    outcome = "EXPIRED"
    outcome_bars = end - idx - 1
    for j in range(idx + 1, end):
        row = m5.iloc[j]
        hi, lo = float(row.High), float(row.Low)
        if c["direction"] == "LONG":
            fav = (hi - entry) / risk; adv = (entry - lo) / risk
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            fav = (entry - lo) / risk; adv = (hi - entry) / risk
            hit_sl, hit_tp = hi >= sl, lo <= tp
        max_fav = max(max_fav, fav); max_adv = max(max_adv, adv)
        if hit_sl and hit_tp:
            outcome, outcome_bars = "SL", j - idx; break
        if hit_sl:
            outcome, outcome_bars = "SL", j - idx; break
        if hit_tp:
            outcome, outcome_bars = "TP", j - idx; break
    return outcome, outcome_bars, max_fav, max_adv


def management_stats(m5, idx, c, tp_r=2.0, expiry=45, be_r=None, trail_start_r=None, trail_dist_r=0.5):
    entry = float(c["entry"]); initial_sl = float(c["sl"]); risk = abs(entry - initial_sl)
    if risk <= 0: return "INVALID", 0, 0.0
    direction = c["direction"]
    tp = entry + risk * tp_r if direction == "LONG" else entry - risk * tp_r
    stop = initial_sl; best = entry
    end = min(len(m5), idx + 1 + int(expiry))
    for j in range(idx + 1, end):
        row = m5.iloc[j]; hi, lo = float(row.High), float(row.Low)
        if direction == "LONG":
            fav = (hi - entry) / risk; hit_sl, hit_tp = lo <= stop, hi >= tp
        else:
            fav = (entry - lo) / risk; hit_sl, hit_tp = hi >= stop, lo <= tp
        if hit_sl and hit_tp: return "SL", j - idx, max(fav, 0.0)
        if hit_sl: return "SL", j - idx, max(fav, 0.0)
        if hit_tp: return "TP", j - idx, max(fav, 0.0)
        best = max(best, hi) if direction == "LONG" else min(best, lo)
        best_r = (best - entry) / risk if direction == "LONG" else (entry - best) / risk
        if be_r is not None and best_r >= be_r:
            stop = max(stop, entry) if direction == "LONG" else min(stop, entry)
        if trail_start_r is not None and best_r >= trail_start_r:
            trail_price = best - trail_dist_r*risk if direction == "LONG" else best + trail_dist_r*risk
            stop = max(stop, trail_price) if direction == "LONG" else min(stop, trail_price)
    return "EXPIRED", end - idx - 1, max((best-entry)/risk if direction == "LONG" else (entry-best)/risk, 0.0)


def _sim_stats(results, tp_r=1.0):
    tp = sum(r == "TP" for r in results); sl = sum(r == "SL" for r in results); expired = sum(r == "EXPIRED" for r in results)
    decided = tp + sl; wr = tp / decided * 100 if decided else 0.0
    ev = ((tp * tp_r) - sl) / len(results) if results else 0.0
    return tp, sl, expired, wr, ev


def trigger_class(c):
    reasons = set(c.get("reasons", [])); has_mss = "M5_MSS" in reasons; has_fvg = "M5_FVG" in reasons; has_sweep = "M5_SWEEP" in reasons
    if has_sweep and has_mss and has_fvg: return "M5_SWEEP_MSS_FVG"
    if has_sweep and has_mss: return "M5_SWEEP_MSS"
    if has_mss and has_fvg: return "M5_MSS_FVG"
    if has_sweep: return "M5_SWEEP"
    if has_mss: return "M5_MSS"
    if has_fvg: return "M5_FVG"
    return "OTHER"


def _wr(bucket):
    tp = sum(c["outcome"] == "TP" for c in bucket); sl = sum(c["outcome"] == "SL" for c in bucket); decided = tp + sl
    return tp, sl, len(bucket)-decided, (tp/decided*100 if decided else 0.0)


def print_trigger_diagnostics(unique):
    print("\n[REPLAY] ===== TRIGGER ANALYSIS =====")
    for cls in sorted({trigger_class(c) for c in unique}):
        bucket=[c for c in unique if trigger_class(c)==cls]; tp,sl,expired,wr=_wr(bucket)
        mfe=pd.Series([c["mfe_r"] for c in bucket])
        print(f"[REPLAY] {cls}: n={len(bucket)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% MFE_med={mfe.median():.2f}R MFE_mean={mfe.mean():.2f}R")


def print_mfe_diagnostics(unique):
    print("\n[REPLAY] ===== MFE BUCKETS =====")
    buckets=[(float("-inf"),.5,"<0.5R"),(.5,1,"0.5-1R"),(1,1.5,"1-1.5R"),(1.5,2,"1.5-2R"),(2,3,"2-3R"),(3,float("inf"),">3R")]
    for lo,hi,label in buckets:
        bucket=[c for c in unique if lo<=c["mfe_r"]<hi]; tp,sl,expired,wr=_wr(bucket)
        print(f"[REPLAY] {label}: n={len(bucket)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}%")


def print_tp_simulation(m5, unique):
    print("\n[REPLAY] ===== TP SIMULATION =====")
    for tp_r in (1.,1.5,2.,2.5,3.):
        results=[forward_stats(m5,c["replay_index"],c,tp_r=tp_r)[0] for c in unique]; tp,sl,expired,wr,ev=_sim_stats(results,tp_r)
        print(f"[REPLAY] TP={tp_r:.1f}R: n={len(results)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% approx_EV={ev:+.2f}R")


def print_expiry_simulation(m5, unique):
    print("\n[REPLAY] ===== EXPIRY SIMULATION =====")
    for bars in (15,30,45,60):
        results=[forward_stats(m5,c["replay_index"],c,max_forward_bars=bars)[0] for c in unique]; tp,sl,expired,wr,ev=_sim_stats(results,1.)
        print(f"[REPLAY] EXPIRY={bars} bars ({bars*5}m): n={len(results)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% EV@1R={ev:+.2f}R")


def print_management_simulation(m5, unique):
    print("\n[REPLAY] ===== MANAGEMENT SIMULATION =====")
    configs=[("BASE_TP2_EXP45",2.,45,None,None,.5),("BE0.5_TP2_EXP45",2.,45,.5,None,.5),("BE1.0_TP2_EXP45",2.,45,1.,None,.5),("TRAIL1.0x0.5_TP2_EXP45",2.,45,None,1.,.5),("BE1.0_TRAIL1.5x0.5_TP2_EXP45",2.,45,1.,1.5,.5),("BE1.0_TRAIL1.0x0.5_TP3_EXP45",3.,45,1.,1.,.5)]
    rows=[]
    for name,tp_r,expiry,be_r,trail_start_r,trail_dist_r in configs:
        results=[management_stats(m5,c["replay_index"],c,tp_r,expiry,be_r,trail_start_r,trail_dist_r) for c in unique]; outcomes=[r[0] for r in results]
        tp,sl,expired,wr,ev=_sim_stats(outcomes,tp_r); avg=sum(r[2] for r in results)/len(results) if results else 0.
        rows.append((ev,name,tp,sl,expired,wr,avg))
    for ev,name,tp,sl,expired,wr,avg in sorted(rows,reverse=True):
        print(f"[REPLAY] {name}: n={len(unique)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% approx_EV={ev:+.2f}R avg_exit_MFE={avg:.2f}R")


def main():
    manager=MT5DataManager(); manager.start()
    try:
        symbol=manager.client.symbol; m5=_history(manager,"M5",BARS); m15=_history(manager,"M15",max(BARS//3+100,150)); m30=_history(manager,"M30",max(BARS//6+100,120))
        print(f"[REPLAY] symbol={symbol} M5={len(m5)} M15={len(m15)} M30={len(m30)} threshold={MIN_SCORE}")
        print(f"[REPLAY] SL GUARD: min_distance={MIN_SL_DISTANCE:.3f} ATRx={MIN_SL_ATR_MULT:.2f} spreadx={MIN_SL_SPREAD_MULT:.1f}")
        print(f"[REPLAY] PERFORMANCE: recent-pivot cache active; progress every {PROGRESS_EVERY} bars")
        print("[REPLAY] SAFE MODE: no AI, Telegram, or orders.")
        unique=[]; scanned=0; funnel={"valid":0,"score_pass":0,"location":0,"trigger":0,"final":0}; side_funnel={"LONG":{k:0 for k in funnel},"SHORT":{k:0 for k in funnel}}; last_unique_idx=-10**9
        total=len(m5)-1
        for i in range(MIN_HISTORY,total):
            candle_time=pd.Timestamp(m5.iloc[i]["Time"]); scan_time=candle_time+timedelta(minutes=5); m15_hist=m15[m15["Time"]+timedelta(minutes=15)<=scan_time]; m30_hist=m30[m30["Time"]+timedelta(minutes=30)<=scan_time]
            if len(m15_hist)<MIN_HISTORY or len(m30_hist)<MIN_HISTORY: continue
            data={"M5":m5.iloc[:i+1],"M15":m15_hist.reset_index(drop=True),"M30":m30_hist.reset_index(drop=True)}; d=diagnose_gates(data)
            if not d.get("valid"): continue
            scanned+=1; funnel["valid"]+=1; side=d["direction"]; sf=side_funnel[side]; sf["valid"]+=1
            for key in ("score_pass","location","trigger","final"):
                gate_value=d.get(key, d.get("final_gate", False) if key=="final" else False)
                if gate_value: funnel[key]+=1; sf[key]+=1
            c=build_candidate(data,tick=None)
            if not c: continue
            c["replay_index"]=i; c["replay_scan_time"]=scan_time.isoformat()
            if i-last_unique_idx<COOLDOWN_BARS: continue
            c["outcome"],c["outcome_bars"],c["mfe_r"],c["mae_r"]=forward_stats(m5,i,c); unique.append(c); last_unique_idx=i
            print(f"[REPLAY] UNIQUE #{len(unique)} {c['direction']} score={c['score']:.0f} RR={c['rr']:.2f} entry={c['entry']:.3f} sl={c['sl']:.3f} tp={c['tp']:.3f} outcome={c['outcome']} bars={c['outcome_bars']} MFE={c['mfe_r']:.2f}R MAE={c['mae_r']:.2f}R candle={c['candle_time']}")
            if PROGRESS_EVERY > 0 and (i-MIN_HISTORY+1) % PROGRESS_EVERY == 0:
                print(f"[REPLAY] progress={i-MIN_HISTORY+1}/{total-MIN_HISTORY} ({(i-MIN_HISTORY+1)/(total-MIN_HISTORY)*100:.1f}%) unique={len(unique)}")
        print("\n[REPLAY] ===== GATE DIAGNOSTICS =====")
        print(f"[REPLAY] valid={funnel['valid']} score_pass={funnel['score_pass']} location={funnel['location']} trigger={funnel['trigger']} final={funnel['final']}")
        for direction in ("LONG","SHORT"):
            f=side_funnel[direction]; print(f"[REPLAY] {direction} gates: valid={f['valid']} score_pass={f['score_pass']} location={f['location']} trigger={f['trigger']} final={f['final']}")
        print_trigger_diagnostics(unique); print_mfe_diagnostics(unique); print_tp_simulation(m5,unique); print_expiry_simulation(m5,unique); print_management_simulation(m5,unique)
        print("[REPLAY] top setups:")
        for rank,c in enumerate(sorted(unique,key=lambda x:(x["score"],x["mfe_r"]),reverse=True)[:10],1):
            print(f"  {rank}. {c['direction']} score={c['score']:.0f} outcome={c['outcome']} MFE={c['mfe_r']:.2f}R MAE={c['mae_r']:.2f}R candle={c['candle_time']} reasons={','.join(c['reasons'])}")
    finally:
        manager.stop()


if __name__ == "__main__": main()
