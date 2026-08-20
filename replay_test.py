"""Historical replay harness for the MT5-native Gold Sniper engine.

Safe research mode: no AI, Telegram, or orders. It replays closed M5 candles,
checks only information available at each timestamp, groups repeated signals,
and evaluates forward TP/SL outcomes from later M5 candles.
"""
from __future__ import annotations
import os
from datetime import timedelta
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
from data_manager import MT5DataManager
from mt5_sniper_engine import build_candidate, MIN_SCORE

load_dotenv()
BARS = int(os.getenv("REPLAY_BARS", "500"))
MIN_HISTORY = 40
MAX_FORWARD_BARS = int(os.getenv("REPLAY_FORWARD_BARS", "60"))
COOLDOWN_BARS = int(os.getenv("REPLAY_COOLDOWN_BARS", "6"))


def _history(manager, tf, count):
    df = manager.client.bars(tf, count)
    if df.empty: raise RuntimeError(f"No {tf} history returned")
    return df.sort_values("Time").reset_index(drop=True)


def outcome(m5, idx, c):
    entry, sl, tp = float(c["entry"]), float(c["sl"]), float(c["tp"])
    end = min(len(m5), idx + 1 + MAX_FORWARD_BARS)
    for j in range(idx + 1, end):
        row = m5.iloc[j]
        hi, lo = float(row.High), float(row.Low)
        if c["direction"] == "LONG":
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
        # Conservative same-bar handling: if both levels are touched, count SL first.
        if hit_sl and hit_tp: return "SL", j - idx
        if hit_sl: return "SL", j - idx
        if hit_tp: return "TP", j - idx
    return "EXPIRED", end - idx - 1


def main():
    manager = MT5DataManager(); manager.start()
    try:
        symbol = manager.client.symbol
        m5 = _history(manager, "M5", BARS)
        m15 = _history(manager, "M15", max(BARS // 3 + 100, 150))
        m30 = _history(manager, "M30", max(BARS // 6 + 100, 120))
        print(f"[REPLAY] symbol={symbol} M5={len(m5)} M15={len(m15)} M30={len(m30)} threshold={MIN_SCORE}")
        print("[REPLAY] SAFE MODE: no AI, Telegram, or orders.")
        raw, unique, scanned = [], [], 0
        last_unique_idx = -10**9
        for i in range(MIN_HISTORY, len(m5) - 1):
            candle_time = pd.Timestamp(m5.iloc[i]["Time"])
            scan_time = candle_time + timedelta(minutes=5)
            m15_hist = m15[m15["Time"] + timedelta(minutes=15) <= scan_time]
            m30_hist = m30[m30["Time"] + timedelta(minutes=30) <= scan_time]
            if len(m15_hist) < MIN_HISTORY or len(m30_hist) < MIN_HISTORY: continue
            data = {"M5":m5.iloc[:i+1], "M15":m15_hist.reset_index(drop=True), "M30":m30_hist.reset_index(drop=True)}
            c = build_candidate(data, tick=None); scanned += 1
            if not c: continue
            c["replay_index"] = i; c["replay_scan_time"] = scan_time.isoformat(); raw.append(c)
            if i - last_unique_idx < COOLDOWN_BARS: continue
            c["outcome"], c["outcome_bars"] = outcome(m5, i, c)
            unique.append(c); last_unique_idx = i
            print(f"[REPLAY] UNIQUE #{len(unique)} {c['direction']} score={c['score']:.0f} RR={c['rr']:.2f} entry={c['entry']:.3f} sl={c['sl']:.3f} tp={c['tp']:.3f} outcome={c['outcome']} bars={c['outcome_bars']} candle={c['candle_time']}")
        print("\n[REPLAY] ===== SUMMARY =====")
        print(f"[REPLAY] scanned={scanned} raw_candidates={len(raw)} unique_setups={len(unique)}")
        if not unique: print("[REPLAY] No unique candidates."); return
        for lo, hi, label in [(50,59,"50-59"),(60,69,"60-69"),(70,79,"70-79"),(80,999,"80+")]:
            bucket=[c for c in unique if lo<=c['score']<=hi]
            wins=sum(c['outcome']=='TP' for c in bucket); losses=sum(c['outcome']=='SL' for c in bucket)
            decided=wins+losses; wr=(wins/decided*100) if decided else 0
            print(f"[REPLAY] score {label}: n={len(bucket)} TP={wins} SL={losses} EXPIRED={len(bucket)-decided} WR={wr:.1f}%")
        for direction in ("LONG","SHORT"):
            bucket=[c for c in unique if c['direction']==direction]; wins=sum(c['outcome']=='TP' for c in bucket); losses=sum(c['outcome']=='SL' for c in bucket); decided=wins+losses
            print(f"[REPLAY] {direction}: n={len(bucket)} TP={wins} SL={losses} EXPIRED={len(bucket)-decided} WR={(wins/decided*100 if decided else 0):.1f}%")
        print("[REPLAY] top setups:")
        for n,c in enumerate(sorted(unique,key=lambda x:x['score'],reverse=True)[:10],1):
            print(f"  {n}. {c['direction']} score={c['score']:.0f} outcome={c['outcome']} candle={c['candle_time']} reasons={','.join(c['reasons'])}")
    finally: manager.stop()

if __name__ == "__main__": main()
