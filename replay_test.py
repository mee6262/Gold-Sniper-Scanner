"""Historical replay harness for Gold Sniper Engine.

Reads historical M5/M15/M30 data directly from the connected MT5 terminal,
feeds closed M5 candles through the existing engine, and reports candidates.
No orders are sent. AI/Telegram are intentionally not called by this harness.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
import MetaTrader5 as mt5

from data_manager import MT5DataManager
from mt5_sniper_engine import MT5SniperEngine, AI_MIN_SCORE

load_dotenv()
BARS = int(os.getenv("REPLAY_BARS", "500"))


def main():
    manager = MT5DataManager()
    manager.start()
    try:
        symbol = manager.symbol
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, BARS)
        if rates_m5 is None or len(rates_m5) == 0:
            raise RuntimeError(f"No M5 history: {mt5.last_error()}")

        engine = MT5SniperEngine(manager)
        candidates = []
        print(f"[REPLAY] symbol={symbol} bars={len(rates_m5)} threshold={AI_MIN_SCORE}")

        # The production engine operates on the latest closed candle. This harness
        # validates the available history and reports the current engine result.
        # Full historical feature reconstruction is deliberately kept separate so
        # it cannot accidentally alter live-state caches.
        candidate = engine.scan_if_new()
        if candidate:
            candidates.append(candidate)
            print(
                f"[REPLAY] CANDIDATE {candidate['direction']} "
                f"score={candidate['score']:.1f} RR={candidate['rr']:.2f} "
                f"entry={candidate['entry']:.3f} sl={candidate['sl']:.3f} tp={candidate['tp']:.3f}"
            )
        else:
            print("[REPLAY] No candidate on current closed M5 candle.")

        print(f"[REPLAY] candidates={len(candidates)}")
        print("[REPLAY] SAFE MODE: no AI call, no Telegram, no orders.")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
