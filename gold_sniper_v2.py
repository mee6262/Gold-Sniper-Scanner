import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import pytz
from tvDatafeed import Interval, TvDatafeed

SYMBOL = os.getenv("GOLD_SYMBOL", "XAUUSD")
EXCHANGE = os.getenv("GOLD_EXCHANGE", "OANDA")
BARS = int(os.getenv("GOLD_BARS", "300"))
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "70"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
SPREAD_MAX = float(os.getenv("MAX_SPREAD", "999999"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "2"))
EQUAL_TOLERANCE = float(os.getenv("EQUAL_TOLERANCE", "0.0015"))
LOG_FILE = os.getenv("GOLD_LOG_FILE", "gold_signals.csv")
TZ = pytz.timezone("Asia/Bangkok")

TF_MAP = {"M30": Interval.in_30_minute, "M15": Interval.in_15_minute, "M5": Interval.in_5_minute}

@dataclass
class Zone:
    direction: str
    kind: str
    low: float
    high: float
    strength: float = 0.0
    source_tf: str = ""


def fetch_data(tv: TvDatafeed, tf: str) -> Optional[pd.DataFrame]:
    try:
        df = tv.get_hist(symbol=SYMBOL, exchange=EXCHANGE, interval=TF_MAP[tf], n_bars=BARS)
        if df is None or df.empty:
            return None
        df = df.copy()
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df.index = idx.tz_convert(TZ)
        df = df.rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        print(f"[DATA] {tf}: {exc}")
        return None


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if len(df) < period + 1:
        return float("nan")
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def pivots(df: pd.DataFrame) -> Tuple[list, list]:
    highs, lows = [], []
    for i in range(PIVOT_LEFT, len(df)-PIVOT_RIGHT):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        if h > df["High"].iloc[i-PIVOT_LEFT:i].max() and h > df["High"].iloc[i+1:i+PIVOT_RIGHT+1].max():
            highs.append((i, float(h)))
        if l < df["Low"].iloc[i-PIVOT_LEFT:i].min() and l < df["Low"].iloc[i+1:i+PIVOT_RIGHT+1].min():
            lows.append((i, float(l)))
    return highs, lows


def liquidity_sweep(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    highs, lows = pivots(df)
    if len(highs) >= 2:
        a, b = highs[-2][1], highs[-1][1]
        level = max(a, b)
        if abs(a-b) / max(level, 1e-9) <= EQUAL_TOLERANCE:
            if df["High"].iloc[-1] > level and df["Close"].iloc[-1] < level:
                return {"direction":"SHORT", "level":level, "type":"EQH_SWEEP"}
    if len(lows) >= 2:
        a, b = lows[-2][1], lows[-1][1]
        level = min(a, b)
        if abs(a-b) / max(level, 1e-9) <= EQUAL_TOLERANCE:
            if df["Low"].iloc[-1] < level and df["Close"].iloc[-1] > level:
                return {"direction":"LONG", "level":level, "type":"EQL_SWEEP"}
    return None


def structure(df: pd.DataFrame) -> Dict[str, Any]:
    highs, lows = pivots(df)
    close = float(df["Close"].iloc[-1])
    if len(highs) < 2 or len(lows) < 2:
        return {"bias":"NEUTRAL", "mss":"NONE", "bos":"NONE"}
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]
    bias = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    bos = "BULLISH" if close > highs[-1][1] else "BEARISH" if close < lows[-1][1] else "NONE"
    return {"bias":bias, "mss":bos, "bos":bos, "last_high":highs[-1][1], "last_low":lows[-1][1]}


def fvg(df: pd.DataFrame) -> Optional[Zone]:
    if len(df) < 3:
        return None
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    atrv = atr(df)
    if c["Low"] > a["High"]:
        gap = float(c["Low"] - a["High"])
        strength = min(gap / max(atrv, 1e-9), 3.0) / 3.0
        if gap > 0:
            return Zone("LONG", "FVG", float(a["High"]), float(c["Low"]), strength, "")
    if c["High"] < a["Low"]:
        gap = float(a["Low"] - c["High"])
        strength = min(gap / max(atrv, 1e-9), 3.0) / 3.0
        if gap > 0:
            return Zone("SHORT", "FVG", float(c["High"]), float(a["Low"]), strength, "")
    return None


def order_block(df: pd.DataFrame) -> Optional[Zone]:
    if len(df) < 25:
        return None
    avg_body = (df["Close"]-df["Open"]).abs().rolling(20).mean().iloc[-1]
    for i in range(2, 8):
        base = df.iloc[-i]
        impulse = df.iloc[-i+1]
        body = abs(float(impulse["Close"]-impulse["Open"]))
        if body < avg_body * 1.2:
            continue
        if base["Close"] < base["Open"] and impulse["Close"] > impulse["Open"]:
            return Zone("LONG", "OB", float(base["Low"]), float(base["High"]), min(body/max(avg_body,1e-9)/2,1), "")
        if base["Close"] > base["Open"] and impulse["Close"] < impulse["Open"]:
            return Zone("SHORT", "OB", float(base["Low"]), float(base["High"]), min(body/max(avg_body,1e-9)/2,1), "")
    return None


def session(dt: pd.Timestamp) -> str:
    h = dt.hour
    if 14 <= h < 19: return "LONDON"
    if 19 <= h < 22: return "LONDON_NY_OVERLAP"
    if h >= 22 or h < 4: return "NEW_YORK"
    if 6 <= h < 13: return "ASIA"
    return "DEAD_ZONE"


def build_candidate(data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    m30, m15, m5 = data["M30"], data["M15"], data["M5"]
    s30, s15, s5 = structure(m30), structure(m15), structure(m5)
    sw30, sw15, sw5 = liquidity_sweep(m30), liquidity_sweep(m15), liquidity_sweep(m5)
    f30, f15, f5 = fvg(m30), fvg(m15), fvg(m5)
    ob30, ob15, ob5 = order_block(m30), order_block(m15), order_block(m5)

    candidates = []
    for direction in ("LONG", "SHORT"):
        score = 0.0
        reasons = []
        if s30["bias"] == ("BULLISH" if direction=="LONG" else "BEARISH"):
            score += 20; reasons.append("M30_BIAS")
        if sw30 and sw30["direction"] == direction:
            score += 15; reasons.append("M30_SWEEP")
        if (f30 and f30.direction == direction) or (ob30 and ob30.direction == direction):
            score += 15; reasons.append("M30_ZONE")
        if s15["mss"] == direction.replace("LONG","BULLISH").replace("SHORT","BEARISH"):
            score += 15; reasons.append("M15_MSS")
        if sw15 and sw15["direction"] == direction:
            score += 10; reasons.append("M15_SWEEP")
        if f15 and f15.direction == direction:
            score += 10; reasons.append("M15_FVG")
        if sw5 and sw5["direction"] == direction:
            score += 10; reasons.append("M5_SWEEP")
        if s5["mss"] == direction.replace("LONG","BULLISH").replace("SHORT","BEARISH"):
            score += 5; reasons.append("M5_MSS")
        if f5 and f5.direction == direction:
            score += 5; reasons.append("M5_FVG")
        candidates.append((score, direction, reasons))

    score, direction, reasons = max(candidates, key=lambda x: x[0])
    if score < 50:
        return None

    price = float(m5["Close"].iloc[-1])
    atr5 = atr(m5)
    zones = [z for z in (f5, ob5, f15, ob15, f30, ob30) if z and z.direction == direction]
    if direction == "LONG":
        invalidation = min([z.low for z in zones] or [price - atr5])
        sl = invalidation - 0.15 * atr5
        target = max([z.high for z in zones if z.high > price] or [price + 3*(price-sl)])
        tp = max(target, price + MIN_RR*(price-sl))
    else:
        invalidation = max([z.high for z in zones] or [price + atr5])
        sl = invalidation + 0.15 * atr5
        target = min([z.low for z in zones if z.low < price] or [price - 3*(sl-price)])
        tp = min(target, price - MIN_RR*(sl-price))
    risk = abs(price-sl)
    rr = abs(tp-price)/max(risk,1e-9)
    if rr < MIN_RR:
        return None
    return {
        "symbol":SYMBOL, "direction":direction, "score":round(score,1), "reasons":reasons,
        "price":price, "entry":price, "sl":float(sl), "tp":float(tp), "rr":round(rr,2),
        "session":session(m5.index[-1]), "m30":s30, "m15":s15, "m5":s5,
        "sweeps":{"M30":sw30,"M15":sw15,"M5":sw5},
        "zones":{"M30":list(filter(None,[asdict(f30) if f30 else None, asdict(ob30) if ob30 else None])),
                  "M15":list(filter(None,[asdict(f15) if f15 else None, asdict(ob15) if ob15 else None])),
                  "M5":list(filter(None,[asdict(f5) if f5 else None, asdict(ob5) if ob5 else None]))},
        "candle_time":m5.index[-1].isoformat()
    }


def ai_review(candidate: Dict[str, Any]) -> Dict[str, Any]:
    key = os.getenv("AI_API_KEY")
    endpoint = os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    if not key:
        return {"enabled":False,"decision":"SKIP","quality":None,"reason":"AI_API_KEY not configured"}
    system = ("You are a strict XAUUSD setup quality reviewer. "
              "Do not invent market data and do not create a new trade direction. "
              "Review only the supplied deterministic candidate. Return JSON only: "
              "{decision: PASS|REJECT, quality: 0-100, risk: LOW|MEDIUM|HIGH, "
              "flags: [string], reason: string}. Reject contradictions, weak structure, "
              "poor alignment, stale zones, or unrealistic execution.")
    payload = {"model":model,"temperature":0,"response_format":{"type":"json_object"},
               "messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(candidate,default=str)}]}
    try:
        r = requests.post(endpoint, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        result = json.loads(text)
        result["enabled"] = True
        return result
    except Exception as exc:
        return {"enabled":True,"decision":"ERROR","quality":None,"risk":"HIGH","flags":[str(exc)],"reason":"AI review failed"}


def log_result(candidate: Dict[str, Any], ai: Dict[str, Any]) -> None:
    row = {"timestamp":datetime.now(timezone.utc).isoformat(), **candidate,
           "ai_decision":ai.get("decision"), "ai_quality":ai.get("quality"),
           "ai_risk":ai.get("risk"), "ai_reason":ai.get("reason")}
    row = {k:(json.dumps(v,default=str) if isinstance(v,(dict,list)) else v) for k,v in row.items()}
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def telegram(message: str) -> None:
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(message); return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat,"text":message}, timeout=10)
    except Exception as exc:
        print(f"[TELEGRAM] {exc}")


def run_once() -> None:
    tv = TvDatafeed()
    data = {tf:fetch_data(tv,tf) for tf in TF_MAP}
    if any(v is None for v in data.values()):
        print("[SCAN] Missing timeframe data; skip")
        return
    candidate = build_candidate(data)
    if not candidate:
        print(f"[SCAN] {datetime.now(TZ):%Y-%m-%d %H:%M:%S} no candidate")
        return
    print(f"[SCAN] Candidate {candidate['direction']} score={candidate['score']} RR={candidate['rr']}")
    if candidate["score"] < AI_MIN_SCORE:
        log_result(candidate,{"enabled":False,"decision":"BELOW_AI_THRESHOLD"})
        return
    ai = ai_review(candidate)
    log_result(candidate,ai)
    if ai.get("decision") == "PASS" and float(ai.get("quality") or 0) >= AI_MIN_SCORE and ai.get("risk") != "HIGH":
        telegram(f"🔱 GOLD SNIPER V2\n{candidate['symbol']} {candidate['direction']}\nScore: {candidate['score']}\nAI Quality: {ai.get('quality')}\nEntry: {candidate['entry']:.2f}\nSL: {candidate['sl']:.2f}\nTP: {candidate['tp']:.2f}\nRR: 1:{candidate['rr']}\nSession: {candidate['session']}\nReason: {ai.get('reason','')}")
    else:
        print(f"[AI] {ai.get('decision')} quality={ai.get('quality')} risk={ai.get('risk')}")


if __name__ == "__main__":
    run_once()
