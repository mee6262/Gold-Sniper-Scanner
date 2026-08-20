"""MT5 market-data adapter for Gold Sniper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSDc")
BARS = int(os.getenv("MT5_BARS", "500"))
MT5_PATH = os.getenv("MT5_PATH", "").strip()
MT5_LOGIN = os.getenv("MT5_LOGIN", "").strip()
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "").strip()

TF_MAP = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30}

@dataclass
class TickSnapshot:
    time: datetime
    bid: float
    ask: float
    spread: float
    last: float

class MT5Data:
    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self.connected = False

    def connect(self) -> None:
        kwargs = {}
        if MT5_PATH:
            kwargs["path"] = MT5_PATH
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            kwargs.update(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
        if not mt5.initialize(**kwargs):
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {code} {msg}")
        if not mt5.symbol_select(self.symbol, True):
            code, msg = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 symbol_select({self.symbol}) failed: {code} {msg}")
        self.connected = True

    def close(self) -> None:
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def ensure_connection(self) -> None:
        if not self.connected:
            self.connect()

    def tick(self) -> TickSnapshot:
        self.ensure_connection()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 tick failed: {code} {msg}")
        bid, ask = float(tick.bid), float(tick.ask)
        last = float(tick.last or (bid + ask) / 2.0)
        return TickSnapshot(datetime.fromtimestamp(int(tick.time), tz=timezone.utc), bid, ask, ask - bid, last)

    def bars(self, timeframe: str, count: int = BARS) -> pd.DataFrame:
        self.ensure_connection()
        if timeframe not in TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(self.symbol, TF_MAP[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 bars {timeframe} failed: {code} {msg}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.rename(columns={"time":"Time", "open":"Open", "high":"High", "low":"Low", "close":"Close", "tick_volume":"Volume", "spread":"BarSpread", "real_volume":"RealVolume"}).sort_values("Time").reset_index(drop=True)

    def snapshot(self, count: int = BARS) -> Dict[str, pd.DataFrame]:
        return {tf: self.bars(tf, count) for tf in TF_MAP}

    def health(self) -> dict:
        self.ensure_connection()
        info, account = mt5.terminal_info(), mt5.account_info()
        return {"connected": bool(info), "symbol": self.symbol, "terminal": info.name if info else None, "trade_allowed": bool(info.trade_allowed) if info else False, "account_login": int(account.login) if account else None, "server": account.server if account else None}

def smoke_test() -> None:
    client = MT5Data()
    try:
        client.connect()
        print("[MT5] HEALTH", client.health())
        tick = client.tick()
        print(f"[MT5] TICK bid={tick.bid} ask={tick.ask} spread={tick.spread}")
        for tf in ("M30", "M15", "M5"):
            df = client.bars(tf, 5)
            print(f"[MT5] {tf}: {len(df)} bars | last={df.iloc[-1]['Close']} | {df.iloc[-1]['Time']}")
    finally:
        client.close()

if __name__ == "__main__":
    smoke_test()
