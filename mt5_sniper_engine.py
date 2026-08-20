"""MT5-native Gold Sniper engine. No TradingView dependency."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSDc")
MIN_SCORE = float(os.getenv("MIN_TRADE_SCORE", "50"))
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "70"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "2"))
EQUAL_TOLERANCE = float(os.getenv("EQUAL_TOLERANCE", "0.0015"))
LOCATION_ATR_BUFFER = float(os.getenv("LOCATION_ATR_BUFFER", "0.35"))
REQUIRE_TRIGGER = os.getenv("REQUIRE_TRIGGER", "1") != "0"
MIN_STRUCTURE_BARS = max(ATR_PERIOD + 5, PIVOT_LEFT + PIVOT_RIGHT + 5)


def atr(df, period=ATR_PERIOD):
    if df is None or len(df) < period + 1:
        return float("nan")
    prev = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.Low-prev).abs(), (df.High-prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def pivots(df):
    highs, lows = [], []
    if df is None or len(df) < MIN_STRUCTURE_BARS:
        return highs, lows
    for i in range(PIVOT_LEFT, len(df)-PIVOT_RIGHT):
        h, l = float(df.High.iloc[i]), float(df.Low.iloc[i])
        if h > df.High.iloc[i-PIVOT_LEFT:i].max() and h > df.High.iloc[i+1:i+PIVOT_RIGHT+1].max():
            highs.append((i, h))
        if l < df.Low.iloc[i-PIVOT_LEFT:i].min() and l < df.Low.iloc[i+1:i+PIVOT_RIGHT+1].min():
            lows.append((i, l))
    return highs, lows


def structure(df):
    if df is None or len(df) < MIN_STRUCTURE_BARS:
        return {"bias":"NEUTRAL", "mss":"NONE", "bos":"NONE"}
    highs, lows = pivots(df)
    close = float(df.Close.iloc[-1])
    if len(highs) < 2 or len(lows) < 2:
        return {"bias":"NEUTRAL", "mss":"NONE", "bos":"NONE"}
    hh = highs[-1][1] > highs[-2][1]; hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]; ll = lows[-1][1] < lows[-2][1]
    bias = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    bos = "BULLISH" if close > highs[-1][1] else "BEARISH" if close < lows[-1][1] else "NONE"
    return {"bias":bias, "mss":bos, "bos":bos, "last_high":highs[-1][1], "last_low":lows[-1][1]}


def sweep(df):
    if df is None or len(df) < MIN_STRUCTURE_BARS:
        return None
    highs, lows = pivots(df)
    if len(highs) >= 2:
        a, b = highs[-2][1], highs[-1][1]; level = max(a, b)
        if abs(a-b)/max(level,1e-9) <= EQUAL_TOLERANCE and df.High.iloc[-1] > level and df.Close.iloc[-1] < level:
            return {"direction":"SHORT", "level":level, "type":"EQH_SWEEP"}
    if len(lows) >= 2:
        a, b = lows[-2][1], lows[-1][1]; level = min(a, b)
        if abs(a-b)/max(level,1e-9) <= EQUAL_TOLERANCE and df.Low.iloc[-1] < level and df.Close.iloc[-1] > level:
            return {"direction":"LONG", "level":level, "type":"EQL_SWEEP"}
    return None


def fvg(df, tf):
    if df is None or len(df) < 3:
        return None
    a, c = df.iloc[-3], df.iloc[-1]; av = atr(df)
    if c.Low > a.High:
        gap = float(c.Low-a.High)
        return {"direction":"LONG","kind":"FVG","low":float(a.High),"high":float(c.Low),"strength":min(gap/max(av,1e-9),3)/3,"source_tf":tf}
    if c.High < a.Low:
        gap = float(a.Low-c.High)
        return {"direction":"SHORT","kind":"FVG","low":float(c.High),"high":float(a.Low),"strength":min(gap/max(av,1e-9),3)/3,"source_tf":tf}
    return None


def order_block(df, tf):
    if df is None or len(df) < 25:
        return None
    avg = (df.Close-df.Open).abs().rolling(20).mean().iloc[-1]
    for i in range(2,8):
        base, imp = df.iloc[-i], df.iloc[-i+1]
        body = abs(float(imp.Close-imp.Open))
        if body < avg*1.2:
            continue
        if base.Close < base.Open and imp.Close > imp.Open:
            return {"direction":"LONG","kind":"OB","low":float(base.Low),"high":float(base.High),"strength":min(body/max(avg,1e-9)/2,1),"source_tf":tf}
        if base.Close > base.Open and imp.Close < imp.Open:
            return {"direction":"SHORT","kind":"OB","low":float(base.Low),"high":float(base.High),"strength":min(body/max(avg,1e-9)/2,1),"source_tf":tf}
    return None


def session(dt):
    h = dt.hour
    if 14 <= h < 19: return "LONDON"
    if 19 <= h < 22: return "LONDON_NY_OVERLAP"
    if h >= 22 or h < 4: return "NEW_YORK"
    if 6 <= h < 13: return "ASIA"
    return "DEAD_ZONE"


def in_or_near_zone(price, zone, buffer):
    return zone["low"]-buffer <= price <= zone["high"]+buffer


def _base_analysis(data):
    ss = {tf:structure(data[tf]) for tf in ("M30","M15","M5")}
    sw = {tf:sweep(data[tf]) for tf in ("M30","M15","M5")}
    zones = {tf:[z for z in (fvg(data[tf],tf), order_block(data[tf],tf)) if z] for tf in ("M30","M15","M5")}
    price = float(data["M5"].Close.iloc[-1])
    av = atr(data["M5"])
    best = None
    for direction in ("LONG","SHORT"):
        want = "BULLISH" if direction == "LONG" else "BEARISH"
        score = 0; reasons = []
        if ss["M30"]["bias"] == want: score += 20; reasons.append("M30_BIAS")
        if sw["M30"] and sw["M30"]["direction"] == direction: score += 15; reasons.append("M30_SWEEP")
        if any(z["direction"] == direction for z in zones["M30"]): score += 15; reasons.append("M30_ZONE")
        if ss["M15"]["mss"] == want: score += 15; reasons.append("M15_MSS")
        if sw["M15"] and sw["M15"]["direction"] == direction: score += 10; reasons.append("M15_SWEEP")
        if any(z["direction"] == direction and z["kind"] == "FVG" for z in zones["M15"]): score += 10; reasons.append("M15_FVG")
        if sw["M5"] and sw["M5"]["direction"] == direction: score += 10; reasons.append("M5_SWEEP")
        if ss["M5"]["mss"] == want: score += 5; reasons.append("M5_MSS")
        if any(z["direction"] == direction and z["kind"] == "FVG" for z in zones["M5"]): score += 5; reasons.append("M5_FVG")
        item = (score, direction, reasons)
        if best is None or item[0] > best[0]: best = item
    score, direction, reasons = best
    directional_zones = [z for tf in ("M5","M15","M30") for z in zones[tf] if z["direction"] == direction]
    location_zone = next((z for z in directional_zones if np.isfinite(av) and in_or_near_zone(price, z, LOCATION_ATR_BUFFER*av)), None)
    want = "BULLISH" if direction == "LONG" else "BEARISH"
    trigger = bool((sw["M5"] and sw["M5"]["direction"] == direction) or (ss["M5"]["mss"] == want and any(z["direction"] == direction and z["kind"] == "FVG" for z in zones["M5"])))
    return {"score":score,"direction":direction,"reasons":reasons,"price":price,"atr":av,"zones":zones,"structure":ss,"sweeps":sw,"location_zone":location_zone,"trigger":trigger}


def diagnose_gates(data: Dict[str,pd.DataFrame]) -> Dict[str,Any]:
    """Research-only gate diagnostics. No orders and no look-ahead.
    Reports the selected direction's score/location/trigger plus raw LONG/SHORT scores.
    """
    required = ("M30","M15","M5")
    if any(tf not in data or data[tf] is None or len(data[tf]) < MIN_STRUCTURE_BARS for tf in required):
        return {"valid":False}
    a = _base_analysis(data)
    return {
        "valid":True,
        "direction":a["direction"],
        "score":a["score"],
        "score_pass":a["score"] >= MIN_SCORE,
        "location":a["location_zone"] is not None,
        "trigger":a["trigger"],
        "final_gate":a["score"] >= MIN_SCORE and a["location_zone"] is not None and (a["trigger"] or not REQUIRE_TRIGGER),
        "location_zone":a["location_zone"],
        "reasons":a["reasons"],
    }


def build_candidate(data: Dict[str,pd.DataFrame], tick=None) -> Optional[Dict[str,Any]]:
    required = ("M30","M15","M5")
    if any(tf not in data or data[tf] is None or len(data[tf]) < MIN_STRUCTURE_BARS for tf in required): return None
    a = _base_analysis(data)
    score, direction, reasons = a["score"], a["direction"], list(a["reasons"])
    ss, sw, zones, price, av = a["structure"], a["sweeps"], a["zones"], a["price"], a["atr"]
    if not np.isfinite(av) or av <= 0 or score < MIN_SCORE: return None
    location_zone = a["location_zone"]
    if location_zone is None: return None
    reasons.append("LOCATION")
    trigger = a["trigger"]
    if trigger:
        if sw["M5"] and sw["M5"]["direction"] == direction: reasons.append("M5_TRIGGER_SWEEP")
        elif ss["M5"]["mss"] == ("BULLISH" if direction == "LONG" else "BEARISH"): reasons.append("M5_TRIGGER_MSS_FVG")
    if REQUIRE_TRIGGER and not trigger: return None
    directional_zones = [z for tf in ("M5","M15","M30") for z in zones[tf] if z["direction"] == direction]
    if direction == "LONG":
        sl = min([z["low"] for z in directional_zones] or [price-av]) - 0.15*av
        tp = max(price + MIN_RR*(price-sl), max([z["high"] for z in directional_zones if z["high"] > price] or [price+3*(price-sl)]))
    else:
        sl = max([z["high"] for z in directional_zones] or [price+av]) + 0.15*av
        tp = min(price - MIN_RR*(sl-price), min([z["low"] for z in directional_zones if z["low"] < price] or [price-3*(sl-price)]))
    risk = abs(price-sl); rr = abs(tp-price)/max(risk,1e-9)
    if not np.isfinite(risk) or risk <= 0 or rr < MIN_RR: return None
    spread = float(tick.spread) if tick else None
    return {"symbol":SYMBOL,"direction":direction,"score":round(score,1),"reasons":reasons,"price":price,"entry":price,"sl":float(sl),"tp":float(tp),"rr":round(rr,2),"spread":spread,"session":session(data["M5"].Time.iloc[-1].to_pydatetime()),"location":location_zone,"trigger":trigger,"structure":ss,"sweeps":sw,"zones":zones,"candle_time":data["M5"].Time.iloc[-1].isoformat()}


class MT5SniperEngine:
    def __init__(self, manager): self.manager=manager; self.last_scanned=None
    def scan_if_new(self):
        snap=self.manager.snapshot(); latest=snap["M5"].Time.iloc[-1]
        if self.last_scanned == latest: return None
        self.last_scanned = latest
        return build_candidate(snap, self.manager.tick())


def format_alert(c):
    return (f"🚨 Gold Sniper Candidate\nSymbol: {c['symbol']}\nDirection: {c['direction']}\nScore: {c['score']}\n"
            f"Entry: {c['entry']:.3f}\nSL: {c['sl']:.3f}\nTP: {c['tp']:.3f}\nRR: {c['rr']}\n"
            f"Spread: {c['spread'] if c['spread'] is not None else 0:.3f}\nSession: {c['session']}\n"
            f"Reasons: {', '.join(c['reasons'])}\nCandle: {c['candle_time']}")
