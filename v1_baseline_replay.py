"""MT5 replay of the original Gold-Sniper v1 planner.

Purpose: establish a V1 baseline before adding V2 filters/AI.
- Uses MT5 data only; no TradingView, AI, Telegram or orders.
- Reproduces the original V1 concepts: EQH/EQL sweep, OB+POC, FVG,
  confluence direction, mean entry, structure-based SL and 1:3 TP.
- Evaluates completed M30 candles using only data available at that candle.
- Limit orders have expiry; Market entries never expire.
"""
from __future__ import annotations

import argparse
from collections import Counter
import numpy as np
import pandas as pd

from mt5_data import MT5Data

PIVOT_LEFT = 8
PIVOT_RIGHT = 3
THRESHOLD_PCT = 0.03
AMOUNT_OF_BOXES = 10
RR = 3.0
SL_BUFFER = 1.5
LIMIT_EXPIRY_M5 = 60


def pivots(df):
    highs, lows = df.High.values, df.Low.values
    n = len(df)
    ph, pl = [], []
    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        if all(highs[i] > highs[i-PIVOT_LEFT:i]) and all(highs[i] > highs[i+1:i+PIVOT_RIGHT+1]):
            ph.append((i, float(highs[i])))
        if all(lows[i] < lows[i-PIVOT_LEFT:i]) and all(lows[i] < lows[i+1:i+PIVOT_RIGHT+1]):
            pl.append((i, float(lows[i])))
    return ph, pl


def swept_liquidity(df):
    ph, pl = pivots(df)
    h, l, c = float(df.High.iloc[-1]), float(df.Low.iloc[-1]), float(df.Close.iloc[-1])
    if len(ph) >= 2:
        _, p2 = ph[-1]; _, p1 = ph[-2]
        if abs(p2-p1) / max(p1, 1e-9) * 100 <= THRESHOLD_PCT:
            level = max(p1, p2)
            if h > level and c < level:
                return {"direction":"SHORT", "level":level}
    if len(pl) >= 2:
        _, p2 = pl[-1]; _, p1 = pl[-2]
        if abs(p2-p1) / max(p1, 1e-9) * 100 <= THRESHOLD_PCT:
            level = min(p1, p2)
            if l < level and c > level:
                return {"direction":"LONG", "level":level}
    return None


def fvg(df):
    if len(df) < 3:
        return None
    c1, c3 = df.iloc[-3], df.iloc[-1]
    if c3.Low > c1.High:
        return {"direction":"LONG", "level":float((c3.Low+c1.High)/2), "top":float(c3.Low), "bot":float(c1.High)}
    if c3.High < c1.Low:
        return {"direction":"SHORT", "level":float((c3.High+c1.Low)/2), "top":float(c1.Low), "bot":float(c3.High)}
    return None


def ob_poc(df):
    if len(df) < 25:
        return None
    n = len(df)
    avg_body = abs(df.Close-df.Open).iloc[-20:].mean()
    found = None
    for i in range(2, 7):
        base, nxt = df.iloc[-i], df.iloc[-i+1]
        body = abs(nxt.Close-nxt.Open)
        if base.Close < base.Open and nxt.Close > nxt.Open and body > avg_body*1.2:
            found = ("LONG", base, df.iloc[n-i:n-i+2]); break
        if base.Close > base.Open and nxt.Close < nxt.Open and body > avg_body*1.2:
            found = ("SHORT", base, df.iloc[n-i:n-i+2]); break
    if found is None:
        return None
    direction, base, sub = found
    top, bot = float(base.High), float(base.Low)
    inc = (top-bot)/AMOUNT_OF_BOXES
    if inc <= 0:
        return None
    vols = [0.0]*AMOUNT_OF_BOXES
    for _, row in sub.iterrows():
        ch, cl, cv = float(row.High), float(row.Low), float(row.Volume)
        span = ch-cl if ch != cl else 0.0001
        for b in range(AMOUNT_OF_BOXES):
            t = top-inc*b; bo = top-inc*(b+1)
            if cl <= t and ch >= bo:
                overlap = min(ch,t)-max(cl,bo)
                vols[b] += max(overlap,0)/span*cv
    idx = int(np.argmax(vols))
    poc_top = top-inc*idx; poc_bot = top-inc*(idx+1)
    return {"direction":direction, "ob_top":top, "ob_bot":bot, "poc_midpoint":float((poc_top+poc_bot)/2)}


def plan(df):
    price = float(df.Close.iloc[-1])
    sweep = swept_liquidity(df)
    ob = ob_poc(df)
    gap = fvg(df)
    signals = [x for x in (sweep, ob, gap) if x]
    if not signals:
        return None
    dirs = [x["direction"] for x in signals]
    if dirs.count("LONG") == dirs.count("SHORT"):
        return None
    direction = "LONG" if dirs.count("LONG") > dirs.count("SHORT") else "SHORT"
    entries=[]; reasons=[]
    if ob and ob["direction"] == direction:
        entries.append(ob["poc_midpoint"]); reasons.append("OB_POC")
    if gap and gap["direction"] == direction:
        entries.append(gap["level"]); reasons.append("FVG")
    if sweep and sweep["direction"] == direction:
        entries.append(sweep["level"]); reasons.append("LIQUIDITY_SWEEP")
    entry=float(np.mean(entries)) if entries else price
    if direction == "SHORT":
        bounds=[x for x in [ob["ob_top"] if ob and ob["direction"]==direction else None,
                            sweep["level"] if sweep and sweep["direction"]==direction else None,
                            gap["top"] if gap and gap["direction"]==direction else None] if x is not None]
        sl=(max(bounds) if bounds else price)+SL_BUFFER
        tp=entry-(sl-entry)*RR
        if price > sl: return None
        order_type="MARKET" if entry >= price else "LIMIT"
    else:
        bounds=[x for x in [ob["ob_bot"] if ob and ob["direction"]==direction else None,
                            sweep["level"] if sweep and sweep["direction"]==direction else None,
                            gap["bot"] if gap and gap["direction"]==direction else None] if x is not None]
        sl=(min(bounds) if bounds else price)-SL_BUFFER
        tp=entry+(entry-sl)*RR
        if price < sl: return None
        order_type="MARKET" if entry <= price else "LIMIT"
    return {"direction":direction,"entry":entry,"sl":float(sl),"tp":float(tp),
            "rr":RR,"type":order_type,"reasons":reasons,"time":df.Time.iloc[-1]}


def replay(m5, m30, expiry=LIMIT_EXPIRY_M5):
    results=[]
    for i in range(max(PIVOT_LEFT+PIVOT_RIGHT+5,25), len(m30)):
        hist=m30.iloc[:i+1].copy()
        p=plan(hist)
        if not p: continue
        t0=p["time"]
        future=m5[m5.Time > t0]
        if p["type"] == "LIMIT":
            future=future.iloc[:expiry]
            fill_idx=None
            for j,row in future.iterrows():
                if row.Low <= p["entry"] <= row.High:
                    fill_idx=j; break
            if fill_idx is None:
                results.append({**p,"outcome":"EXPIRED","fill":False}); continue
            future=future.loc[fill_idx:]
        else:
            future=future.iloc[:expiry]
        outcome="EXPIRED"
        for _,row in future.iterrows():
            if p["direction"] == "LONG":
                hit_sl=row.Low <= p["sl"]; hit_tp=row.High >= p["tp"]
            else:
                hit_sl=row.High >= p["sl"]; hit_tp=row.Low <= p["tp"]
            if hit_sl and hit_tp:
                outcome="SL_FIRST"  # conservative same-bar assumption
                break
            if hit_sl:
                outcome="SL"; break
            if hit_tp:
                outcome="TP"; break
        results.append({**p,"outcome":outcome,"fill":True})
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--m5",type=int,default=1000)
    ap.add_argument("--m30",type=int,default=400)
    args=ap.parse_args()
    client=MT5Data()
    try:
        client.connect()
        m5=client.bars("M5",args.m5)
        m30=client.bars("M30",args.m30)
    finally:
        client.close()
    results=replay(m5,m30)
    print(f"[V1] symbol={client.symbol} M5={len(m5)} M30={len(m30)} plans={len(results)}")
    print("[V1] BASELINE: original planner concepts; no AI, Telegram, or orders.")
    for n,r in enumerate(results,1):
        print(f"[V1] #{n} {r['direction']} {r['type']} entry={r['entry']:.3f} sl={r['sl']:.3f} tp={r['tp']:.3f} outcome={r['outcome']} reasons={','.join(r['reasons'])} candle={r['time'].isoformat()}")
    c=Counter(r["outcome"] for r in results)
    print("[V1] SUMMARY", dict(c))
    for typ in ("LIMIT","MARKET"):
        x=[r for r in results if r["type"]==typ]
        if x:
            cc=Counter(r["outcome"] for r in x)
            print(f"[V1] {typ}: n={len(x)} {dict(cc)}")

if __name__ == "__main__":
    main()
