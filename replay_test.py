"""Historical replay harness for the MT5-native Gold Sniper engine.

Safe test only:
- reads historical M5/M15/M30 data from the connected MT5 terminal
- replays CLOSED M5 candles from oldest to newest
- aligns M15/M30 strictly to information that was available at that time
- reports candidates
- never calls AI, Telegram, or OrderSend
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


def _history(manager: MT5DataManager, tf: str, count: int) -> pd.DataFrame:
    """Fetch history through the manager's MT5 client without relying on manager.symbol."""
    df = manager.client.bars(tf, count)
    if df.empty:
        raise RuntimeError(f"No {tf} history returned")
    return df.sort_values("Time").reset_index(drop=True)


def main() -> None:
    manager = MT5DataManager()
    manager.start()
    try:
        symbol = manager.client.symbol
        m5 = _history(manager, "M5", BARS)
        m15 = _history(manager, "M15", max(BARS // 3 + 100, 150))
        m30 = _history(manager, "M30", max(BARS // 6 + 100, 120))

        print(f"[REPLAY] symbol={symbol} M5={len(m5)} M15={len(m15)} M30={len(m30)} threshold={MIN_SCORE}")
        print("[REPLAY] SAFE MODE: historical replay only; no AI, Telegram, or orders.")

        candidates = []
        scanned = 0

        # Each M5 row represents a candle that is CLOSED at row.Time + 5 minutes.
        # Higher-TF bars must also be closed before that scan time, otherwise the
        # replay would leak future information into the signal.
        for i in range(MIN_HISTORY, len(m5)):
            m5_row = m5.iloc[i]
            candle_time = pd.Timestamp(m5_row["Time"])
            scan_time = candle_time + timedelta(minutes=5)

            m5_hist = m5.iloc[: i + 1]
            m15_hist = m15[m15["Time"] + timedelta(minutes=15) <= scan_time]
            m30_hist = m30[m30["Time"] + timedelta(minutes=30) <= scan_time]

            if len(m15_hist) < MIN_HISTORY or len(m30_hist) < MIN_HISTORY:
                continue

            data = {
                "M5": m5_hist,
                "M15": m15_hist.reset_index(drop=True),
                "M30": m30_hist.reset_index(drop=True),
            }
            candidate = build_candidate(data, tick=None)
            scanned += 1

            if candidate:
                candidate["replay_scan_time"] = scan_time.isoformat()
                candidates.append(candidate)
                print(
                    f"[REPLAY] #{len(candidates)} {candidate['direction']} "
                    f"score={candidate['score']:.1f} RR={candidate['rr']:.2f} "
                    f"entry={candidate['entry']:.3f} sl={candidate['sl']:.3f} "
                    f"tp={candidate['tp']:.3f} candle={candidate['candle_time']}"
                )

        print("\n[REPLAY] ===== SUMMARY =====")
        print(f"[REPLAY] scanned={scanned}")
        print(f"[REPLAY] candidates={len(candidates)}")
        if candidates:
            top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:10]
            print("[REPLAY] top candidates:")
            for n, c in enumerate(top, 1):
                print(
                    f"  {n}. {c['direction']} score={c['score']:.1f} "
                    f"RR={c['rr']:.2f} candle={c['candle_time']} "
                    f"reasons={','.join(c['reasons'])}"
                )
        else:
            print("[REPLAY] No historical candidates met the current engine threshold.")

    finally:
        manager.stop()


if __name__ == "__main__":
    main()
