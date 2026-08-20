"""MT5-native Gold Sniper engine. No TradingView dependency."""
from __future__ import annotations
import json, os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from mt5_data import MT5Data

SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSDc")
MIN_SCORE = float(os.getenv("MIN_TRADE_SCORE", "50"))
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "70"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "2"))
EQUAL_TOLERANCE = float(os.getenv("EQUAL_TOLERANCE", "0.0015"))

@dataclass
class Zone:
    direction: str
    kind: str
    low: float
    high: float
    strength: float = 0.0
    source_tf: str = ""

def atr(df, period=ATR_PERIOD):
    if len(df) < period + 1: return float("nan")
    prev=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-prev).abs(),(df.Low-prev).abs()],axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def pivots(df):
    highs,lows=[],[]
    for i in range(PIVOT_LEFT,len(df)-PIVOT_RIGHT):
        h,l=float(df.High.iloc[i]),float(df.Low.iloc[i])
        if h>df.High.iloc[i-PIVOT_LEFT:i].max() and h>df.High.iloc[i+1:i+PIVOT_RIGHT+1].max(): highs.append((i,h))
        if l<df.Low.iloc[i-PIVOT_LEFT:i].min() and l<df.Low.iloc[i+1:i+PIVOT_RIGHT+1].min(): lows.append((i,l))
    return highs,lows

def structure(df):
    highs,lows=pivots(df); close=float(df.Close.iloc[-1])
    if len(highs)<2 or len(lows)<2: return {"bias":"NEUTRAL","mss":"NONE","bos":"NONE"}
    hh=highs[-1][1]>highs[-2][1]; hl=lows[-1][1]>lows[-2][1]
    lh=highs[-1][1]<highs[-2][1]; ll=lows[-1][1]<lows[-2][1]
    bias="BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    bos="BULLISH" if close>highs[-1][1] else "BEARISH" if close<lows[-1][1] else "NONE"
    return {"bias":bias,"mss":bos,"bos":bos,"last_high":highs[-1][1],"last_low":lows[-1][1]}

def sweep(df):
    highs,lows=pivots(df)
    if len(highs)>=2:
        a,b=highs[-2][1],highs[-1][1]; level=max(a,b)
        if abs(a-b)/max(level,1e-9)<=EQUAL_TOLERANCE and df.High.iloc[-1]>level and df.Close.iloc[-1]<level:
            return {"direction":"SHORT","level":level,"type":"EQH_SWEEP"}
    if len(lows)>=2:
        a,b=lows[-2][1],lows[-1][1]; level=min(a,b)
        if abs(a-b)/max(level,1e-9)<=EQUAL_TOLERANCE and df.Low.iloc[-1]<level and df.Close.iloc[-1]>level:
            return {"direction":"LONG","level":level,"type":"EQL_SWEEP"}
    return None

def fvg(df, tf):
    if len(df)<3:return None
    a,c=df.iloc[-3],df.iloc[-1]; av=atr(df)
    if c.Low>a.High:
        gap=float(c.Low-a.High); return Zone("LONG","FVG",float(a.High),float(c.Low),min(gap/max(av,1e-9),3)/3,tf)
    if c.High<a.Low:
        gap=float(a.Low-c.High); return Zone("SHORT","FVG",float(c.High),float(a.Low),min(gap/max(av,1e-9),3)/3,tf)
    return None

def order_block(df,tf):
    if len(df)<25:return None
    avg=(df.Close-df.Open).abs().rolling(20).mean().iloc[-1]
    for i in range(2,8):
        base,imp=df.iloc[-i],df.iloc[-i+1]; body=abs(float(imp.Close-imp.Open))
        if body<avg*1.2:continue
        if base.Close<base.Open and imp.Close>imp.Open:return Zone("LONG","OB",float(base.Low),float(base.High),min(body/max(avg,1e-9)/2,1),tf)
        if base.Close>base.Open and imp.Close<imp.Open:return Zone("SHORT","OB",float(base.Low),float(base.High),min(body/max(avg,1e-9)/2,1),tf)
    return None

def session(dt):
    h=dt.hour
    if 14<=h<19:return "LONDON"
    if 19<=h<22:return "LONDON_NY_OVERLAP"
    if h>=22 or h<4:return "NEW_YORK"
    if 6<=h<13:return "ASIA"
    return "DEAD_ZONE"

def build_candidate(data:Dict[str,pd.DataFrame],tick=None)->Optional[Dict[str,Any]]:
    m30,m15,m5=data["M30"],data["M15"],data["M5"]
    ss={tf:structure(data[tf]) for tf in ("M30","M15","M5")}
    sw={tf:sweep(data[tf]) for tf in ("M30","M15","M5")}
    zones={tf:[z for z in (fvg(data[tf],tf),order_block(data[tf],tf)) if z] for tf in ("M30","M15","M5")}
    best=None
    for direction in ("LONG","SHORT"):
        want="BULLISH" if direction=="LONG" else "BEARISH"; score=0; reasons=[]
        if ss["M30"]["bias"]==want:score+=20;reasons.append("M30_BIAS")
        if sw["M30"] and sw["M30"]["direction"]==direction:score+=15;reasons.append("M30_SWEEP")
        if any(z.direction==direction for z in zones["M30"]):score+=15;reasons.append("M30_ZONE")
        if ss["M15"]["mss"]==want:score+=15;reasons.append("M15_MSS")
        if sw["M15"] and sw["M15"]["direction"]==direction:score+=10;reasons.append("M15_SWEEP")
        if any(z.direction==direction and z.kind=="FVG" for z in zones["M15"]):score+=10;reasons.append("M15_FVG")
        if sw["M5"] and sw["M5"]["direction"]==direction:score+=10;reasons.append("M5_SWEEP")
        if ss["M5"]["mss"]==want:score+=5;reasons.append("M5_MSS")
        if any(z.direction==direction and z.kind=="FVG" for z in zones["M5"]):score+=5;reasons.append("M5_FVG")
        item=(score,direction,reasons)
        if best is None or item[0]>best[0]:best=item
    score,direction,reasons=best
    if score<MIN_SCORE:return None
    price=float(tick.last if tick else m5.Close.iloc[-1]); av=atr(m5)
    z=[x for tf in ("M5","M15","M30") for x in zones[tf] if x.direction==direction]
    if direction=="LONG":
        sl=min([x.low for x in z] or [price-av])-0.15*av; tp=max(price+MIN_RR*(price-sl),max([x.high for x in z if x.high>price] or [price+3*(price-sl)]))
    else:
        sl=max([x.high for x in z] or [price+av])+0.15*av; tp=min(price-MIN_RR*(sl-price),min([x.low for x in z if x.low<price] or [price-3*(sl-price)]))
    risk=abs(price-sl); rr=abs(tp-price)/max(risk,1e-9)
    if rr<MIN_RR:return None
    spread=float(tick.spread) if tick else None
    return {"symbol":SYMBOL,"direction":direction,"score":round(score,1),"reasons":reasons,"price":price,"entry":price,"sl":float(sl),"tp":float(tp),"rr":round(rr,2),"spread":spread,"session":session(m5.Time.iloc[-1].to_pydatetime()),"structure":ss,"sweeps":sw,"zones":{tf:[asdict(x) for x in zones[tf]] for tf in zones},"candle_time":m5.Time.iloc[-1].isoformat()}

class MT5SniperEngine:
    def __init__(self, manager): self.manager=manager; self.last_scanned=None
    def scan_if_new(self):
        snap=self.manager.snapshot(); latest=snap["M5"].Time.iloc[-1]
        if self.last_scanned==latest:return None
        self.last_scanned=latest
        return build_candidate(snap,self.manager.tick())

def format_alert(c):
    return (f"🚨 Gold Sniper Candidate\nSymbol: {c['symbol']}\nDirection: {c['direction']}\nScore: {c['score']}\n"
            f"Entry: {c['entry']:.3f}\nSL: {c['sl']:.3f}\nTP: {c['tp']:.3f}\nRR: {c['rr']}\nSpread: {c['spread']:.3f}\n"
            f"Session: {c['session']}\nReasons: {', '.join(c['reasons'])}\nCandle: {c['candle_time']}")
