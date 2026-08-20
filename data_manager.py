"""Stateful MT5 data manager for continuous Gold Sniper scanning."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from mt5_data import MT5Data, TickSnapshot


class MT5DataManager:
    def __init__(self, symbol: Optional[str] = None, bars: int = 500):
        self.client = MT5Data(symbol=symbol) if symbol else MT5Data()
        self.bars_count = bars
        self.cache: Dict[str, pd.DataFrame] = {}
        self.last_bar_time: Dict[str, datetime] = {}
        self.last_ok: Optional[datetime] = None
        self.last_error: Optional[str] = None

    def start(self) -> None:
        self.client.connect()
        self.refresh_all(force=True)

    def stop(self) -> None:
        self.client.close()

    def _refresh(self, tf: str, force: bool = False) -> bool:
        df = self.client.bars(tf, self.bars_count)
        if df.empty:
            return False
        latest = df.iloc[-1]["Time"]
        latest_dt = latest.to_pydatetime() if hasattr(latest, "to_pydatetime") else latest
        changed = force or self.last_bar_time.get(tf) != latest_dt
        self.cache[tf] = df
        self.last_bar_time[tf] = latest_dt
        self.last_ok = datetime.now(timezone.utc)
        self.last_error = None
        return changed

    def refresh_all(self, force: bool = False) -> Dict[str, bool]:
        result = {}
        for tf in ("M5", "M15", "M30"):
            result[tf] = self._refresh(tf, force=force)
        return result

    def refresh_changed(self) -> Dict[str, bool]:
        return self.refresh_all(force=False)

    def tick(self) -> TickSnapshot:
        return self.client.tick()

    def snapshot(self) -> Dict[str, pd.DataFrame]:
        if not self.cache:
            self.refresh_all(force=True)
        return self.cache.copy()

    def health(self) -> dict:
        h = self.client.health()
        h.update({
            "cache_ready": bool(self.cache),
            "last_ok": self.last_ok.isoformat() if self.last_ok else None,
            "last_error": self.last_error,
            "last_bars": {tf: str(ts) for tf, ts in self.last_bar_time.items()},
        })
        return h

    def poll(self, seconds: int = 60) -> None:
        """Run a resilient polling loop. Strategy/Telegram is intentionally outside this class."""
        try:
            self.start()
            while True:
                try:
                    self.refresh_changed()
                except Exception as exc:
                    self.last_error = str(exc)
                    try:
                        self.client.close()
                        time.sleep(2)
                        self.client.connect()
                    except Exception as reconnect_exc:
                        self.last_error = str(reconnect_exc)
                        time.sleep(min(seconds, 30))
                time.sleep(seconds)
        finally:
            self.stop()
