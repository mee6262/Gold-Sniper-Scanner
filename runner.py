"""Persistent Gold Sniper runtime using MT5 as the market-data source."""
from __future__ import annotations

import logging
import time

from data_manager import MT5DataManager

POLL_SECONDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gold-sniper")


def run() -> None:
    manager = MT5DataManager()
    manager.start()
    log.info("MT5 READY: %s", manager.health())
    last_m5 = None

    try:
        while True:
            try:
                tick = manager.tick()
                changes = manager.refresh_changed()
                m5_time = manager.last_bar_time.get("M5")

                if m5_time != last_m5:
                    last_m5 = m5_time
                    snap = manager.snapshot()
                    log.info(
                        "NEW M5 | %s | bid=%.3f ask=%.3f spread=%.3f | M15=%s M30=%s",
                        m5_time,
                        tick.bid,
                        tick.ask,
                        tick.spread,
                        snap["M15"].iloc[-1]["Time"],
                        snap["M30"].iloc[-1]["Time"],
                    )
                elif any(changes.values()):
                    log.info("TF UPDATE %s", changes)

            except Exception as exc:
                log.exception("runtime error: %s", exc)
                try:
                    manager.stop()
                    time.sleep(2)
                    manager.start()
                except Exception as reconnect_exc:
                    log.error("MT5 reconnect failed: %s", reconnect_exc)
                    time.sleep(10)

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log.info("Stopping Gold Sniper...")
    finally:
        manager.stop()


if __name__ == "__main__":
    run()
