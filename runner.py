"""Persistent MT5-native Gold Sniper runtime. No TradingView dependency."""
from __future__ import annotations
import json, logging, os, time
from datetime import datetime, timezone
import requests
from data_manager import MT5DataManager
from mt5_sniper_engine import MT5SniperEngine, format_alert, AI_MIN_SCORE
from ai_reviewer import review

POLL_SECONDS=int(os.getenv("POLL_SECONDS","5"))
JOURNAL_FILE=os.getenv("SIGNAL_JOURNAL","signals.jsonl")
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
log=logging.getLogger("gold-sniper")

class ClosedCandleView:
    """Engine view that excludes the currently-forming candle on every timeframe."""
    def __init__(self, manager): self.manager=manager
    def snapshot(self):
        return {tf:df.iloc[:-1].copy() for tf,df in self.manager.snapshot().items() if len(df)>2}
    def tick(self): return self.manager.tick()

def telegram(message):
    token,chat=os.getenv("TELEGRAM_TOKEN"),os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.info("TELEGRAM disabled; message:\n%s",message); return
    try:
        r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":message},timeout=10); r.raise_for_status()
    except Exception as exc: log.error("Telegram error: %s",exc)

def journal(candidate, ai):
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"candidate":candidate,"ai":ai}
    with open(JOURNAL_FILE,"a",encoding="utf-8") as f: f.write(json.dumps(row,default=str)+"\n")

def run():
    manager=MT5DataManager(); manager.start(); engine=MT5SniperEngine(ClosedCandleView(manager))
    log.info("MT5 READY: %s",manager.health())
    try:
        while True:
            try:
                changes=manager.refresh_changed()
                candidate=engine.scan_if_new()
                if candidate:
                    log.info("CANDIDATE %s score=%.1f RR=%.2f",candidate["direction"],candidate["score"],candidate["rr"])
                    if candidate["score"]>=AI_MIN_SCORE:
                        ai=review(candidate); journal(candidate,ai)
                        log.info("AI %s quality=%s risk=%s",ai.get("decision"),ai.get("quality"),ai.get("risk"))
                        if ai.get("decision")=="PASS":
                            telegram(format_alert(candidate)+f"\nAI: PASS | Quality: {ai.get('quality')} | Risk: {ai.get('risk')}\nReason: {ai.get('reason','')}")
                    else:
                        journal(candidate,{"decision":"NOT_SENT","reason":"below AI threshold"})
                        log.info("Candidate below AI threshold: %.1f < %.1f",candidate["score"],AI_MIN_SCORE)
                elif any(changes.values()):
                    log.info("TF UPDATE %s",changes)
            except Exception as exc:
                log.exception("runtime error: %s",exc)
                try: manager.stop(); time.sleep(2); manager.start()
                except Exception as reconnect_exc: log.error("MT5 reconnect failed: %s",reconnect_exc); time.sleep(10)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt: log.info("Stopping Gold Sniper...")
    finally: manager.stop()

if __name__=="__main__": run()
