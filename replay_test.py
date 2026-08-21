"""Historical replay + diagnostics for the MT5-native Gold Sniper engine.

Safe research mode: no AI, Telegram, or orders. Uses only data available at each
closed M5 timestamp. Reports gate funnel, trigger classes, MFE/MAE, TP sensitivity,
and expiry sensitivity for final unique candidates.
"""
from __future__ import annotations
import os
from datetime import timedelta
from dotenv import load_dotenv
import pandas as pd
from data_manager import MT5DataManager
from mt5_sniper_engine import build_candidate, diagnose_gates, MIN_SCORE

load_dotenv()
BARS = int(os.getenv("REPLAY_BARS", "500"))
MIN_HISTORY = 40
MAX_FORWARD_BARS = int(os.getenv("REPLAY_FORWARD_BARS", "60"))
COOLDOWN_BARS = int(os.getenv("REPLAY_COOLDOWN_BARS", "6"))


def _history(manager, tf, count):
    df = manager.client.bars(tf, count)
    if df.empty:
        raise RuntimeError(f"No {tf} history returned")
    return df.sort_values("Time").reset_index(drop=True)


def forward_stats(m5, idx, c, tp_r=None, max_forward_bars=None):
    """Evaluate outcome plus MFE/MAE in R over the forward window.

    If tp_r is supplied, simulate a TP at tp_r * initial risk without changing
    the candidate's original SL. Same-bar SL+TP is handled conservatively as SL.
    """
    entry, sl = float(c["entry"]), float(c["sl"])
    risk = abs(entry - sl)
    if risk <= 0:
        return "INVALID", 0, 0.0, 0.0
    if tp_r is None:
        tp = float(c["tp"])
    elif c["direction"] == "LONG":
        tp = entry + risk * float(tp_r)
    else:
        tp = entry - risk * float(tp_r)

    window = MAX_FORWARD_BARS if max_forward_bars is None else int(max_forward_bars)
    end = min(len(m5), idx + 1 + window)
    max_fav = 0.0
    max_adv = 0.0
    outcome = "EXPIRED"
    outcome_bars = end - idx - 1
    for j in range(idx + 1, end):
        row = m5.iloc[j]
        hi, lo = float(row.High), float(row.Low)
        if c["direction"] == "LONG":
            fav = (hi - entry) / risk
            adv = (entry - lo) / risk
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            fav = (entry - lo) / risk
            adv = (hi - entry) / risk
            hit_sl, hit_tp = hi >= sl, lo <= tp
        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)
        # Conservative same-bar handling: if both levels are touched, count SL first.
        if hit_sl and hit_tp:
            outcome, outcome_bars = "SL", j - idx
            break
        if hit_sl:
            outcome, outcome_bars = "SL", j - idx
            break
        if hit_tp:
            outcome, outcome_bars = "TP", j - idx
            break
    return outcome, outcome_bars, max_fav, max_adv


def trigger_class(c):
    """Classify the M5 trigger using only recorded engine reasons."""
    reasons = set(c.get("reasons", []))
    has_mss = "M5_MSS" in reasons
    has_fvg = "M5_FVG" in reasons
    has_sweep = "M5_SWEEP" in reasons
    if has_sweep and has_mss and has_fvg:
        return "M5_SWEEP_MSS_FVG"
    if has_sweep and has_mss:
        return "M5_SWEEP_MSS"
    if has_mss and has_fvg:
        return "M5_MSS_FVG"
    if has_sweep:
        return "M5_SWEEP"
    if has_mss:
        return "M5_MSS"
    if has_fvg:
        return "M5_FVG"
    return "OTHER"


def _wr(bucket):
    tp = sum(c["outcome"] == "TP" for c in bucket)
    sl = sum(c["outcome"] == "SL" for c in bucket)
    decided = tp + sl
    return tp, sl, len(bucket) - decided, (tp / decided * 100 if decided else 0.0)


def _sim_stats(results, tp_r=1.0):
    tp = results.count("TP")
    sl = results.count("SL")
    expired = results.count("EXPIRED")
    decided = tp + sl
    wr = tp / decided * 100 if decided else 0.0
    ev = ((tp * tp_r) - sl) / len(results) if results else 0.0
    return tp, sl, expired, wr, ev


def print_trigger_diagnostics(unique):
    print("\n[REPLAY] ===== TRIGGER ANALYSIS =====")
    classes = sorted({trigger_class(c) for c in unique})
    for cls in classes:
        bucket = [c for c in unique if trigger_class(c) == cls]
        tp, sl, expired, wr = _wr(bucket)
        mfe = pd.Series([c["mfe_r"] for c in bucket])
        print(f"[REPLAY] {cls}: n={len(bucket)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% MFE_med={mfe.median():.2f}R MFE_mean={mfe.mean():.2f}R")


def print_mfe_diagnostics(unique):
    print("\n[REPLAY] ===== MFE BUCKETS =====")
    buckets = [
        (float("-inf"), 0.5, "<0.5R"),
        (0.5, 1.0, "0.5-1R"),
        (1.0, 1.5, "1-1.5R"),
        (1.5, 2.0, "1.5-2R"),
        (2.0, 3.0, "2-3R"),
        (3.0, float("inf"), ">3R"),
    ]
    for lo, hi, label in buckets:
        bucket = [c for c in unique if lo <= c["mfe_r"] < hi]
        tp, sl, expired, wr = _wr(bucket)
        print(f"[REPLAY] {label}: n={len(bucket)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}%")


def print_tp_simulation(m5, unique):
    print("\n[REPLAY] ===== TP SIMULATION =====")
    for tp_r in (1.0, 1.5, 2.0, 2.5, 3.0):
        results = [forward_stats(m5, c["replay_index"], c, tp_r=tp_r)[0] for c in unique]
        tp, sl, expired, wr, expectancy = _sim_stats(results, tp_r)
        print(f"[REPLAY] TP={tp_r:.1f}R: n={len(results)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% approx_EV={expectancy:+.2f}R")


def print_expiry_simulation(m5, unique):
    """Compare expiry windows without changing the candidate or entry logic."""
    print("\n[REPLAY] ===== EXPIRY SIMULATION =====")
    for bars in (15, 30, 45, 60):
        results = [forward_stats(m5, c["replay_index"], c, max_forward_bars=bars)[0] for c in unique]
        tp, sl, expired, wr, expectancy = _sim_stats(results, 1.0)
        minutes = bars * 5
        print(f"[REPLAY] EXPIRY={bars} bars ({minutes}m): n={len(results)} TP={tp} SL={sl} EXPIRED={expired} WR={wr:.1f}% EV@1R={expectancy:+.2f}R")


def main():
    manager = MT5DataManager()
    manager.start()
    try:
        symbol = manager.client.symbol
        m5 = _history(manager, "M5", BARS)
        m15 = _history(manager, "M15", max(BARS // 3 + 100, 150))
        m30 = _history(manager, "M30", max(BARS // 6 + 100, 120))
        print(f"[REPLAY] symbol={symbol} M5={len(m5)} M15={len(m15)} M30={len(m30)} threshold={MIN_SCORE}")
        print("[REPLAY] SAFE MODE: no AI, Telegram, or orders.")
        raw, unique, scanned = [], [], 0
        funnel = {"valid": 0, "score_pass": 0, "location": 0, "trigger": 0, "final": 0}
        side_funnel = {"LONG": {"valid": 0, "score_pass": 0, "location": 0, "trigger": 0, "final": 0}, "SHORT": {"valid": 0, "score_pass": 0, "location": 0, "trigger": 0, "final": 0}}
        last_unique_idx = -10**9
        for i in range(MIN_HISTORY, len(m5) - 1):
            candle_time = pd.Timestamp(m5.iloc[i]["Time"])
            scan_time = candle_time + timedelta(minutes=5)
            m15_hist = m15[m15["Time"] + timedelta(minutes=15) <= scan_time]
            m30_hist = m30[m30["Time"] + timedelta(minutes=30) <= scan_time]
            if len(m15_hist) < MIN_HISTORY or len(m30_hist) < MIN_HISTORY:
                continue
            data = {"M5": m5.iloc[:i + 1], "M15": m15_hist.reset_index(drop=True), "M30": m30_hist.reset_index(drop=True)}
            d = diagnose_gates(data)
            if not d.get("valid"):
                continue
            scanned += 1
            funnel["valid"] += 1
            side = d["direction"]
            sf = side_funnel[side]
            sf["valid"] += 1
            if d["score_pass"]:
                funnel["score_pass"] += 1; sf["score_pass"] += 1
            if d["location"]:
                funnel["location"] += 1; sf["location"] += 1
            if d["trigger"]:
                funnel["trigger"] += 1; sf["trigger"] += 1
            if d["final_gate"]:
                funnel["final"] += 1; sf["final"] += 1
            c = build_candidate(data, tick=None)
            if not c:
                continue
            c["replay_index"] = i
            c["replay_scan_time"] = scan_time.isoformat()
            raw.append(c)
            if i - last_unique_idx < COOLDOWN_BARS:
                continue
            c["outcome"], c["outcome_bars"], c["mfe_r"], c["mae_r"] = forward_stats(m5, i, c)
            unique.append(c)
            last_unique_idx = i
            print(f"[REPLAY] UNIQUE #{len(unique)} {c['direction']} score={c['score']:.0f} RR={c['rr']:.2f} entry={c['entry']:.3f} sl={c['sl']:.3f} tp={c['tp']:.3f} outcome={c['outcome']} bars={c['outcome_bars']} MFE={c['mfe_r']:.2f}R MAE={c['mae_r']:.2f}R candle={c['candle_time']}")

        print("\n[REPLAY] ===== GATE DIAGNOSTICS =====")
        print(f"[REPLAY] valid={funnel['valid']} score_pass={funnel['score_pass']} location={funnel['location']} trigger={funnel['trigger']} final={funnel['final']}")
        for direction in ("LONG", "SHORT"):
            f = side_funnel[direction]
            print(f"[REPLAY] {direction} gates: valid={f['valid']} score_pass={f['score_pass']} location={f['location']} trigger={f['trigger']} final={f['final']}")

        print("\n[REPLAY] ===== SUMMARY =====")
        print(f"[REPLAY] scanned={scanned} raw_candidates={len(raw)} unique_setups={len(unique)}")
        if not unique:
            print("[REPLAY] No unique candidates.")
            return
        for lo, hi, label in [(50, 59, "50-59"), (60, 69, "60-69"), (70, 79, "70-79"), (80, 999, "80+")]:
            bucket = [c for c in unique if lo <= c["score"] <= hi]
            wins, losses, expired, wr = _wr(bucket)
            print(f"[REPLAY] score {label}: n={len(bucket)} TP={wins} SL={losses} EXPIRED={expired} WR={wr:.1f}%")
        for direction in ("LONG", "SHORT"):
            bucket = [c for c in unique if c["direction"] == direction]
            wins, losses, expired, wr = _wr(bucket)
            print(f"[REPLAY] {direction}: n={len(bucket)} TP={wins} SL={losses} EXPIRED={expired} WR={wr:.1f}%")
        print(f"[REPLAY] MFE median={pd.Series([c['mfe_r'] for c in unique]).median():.2f}R mean={pd.Series([c['mfe_r'] for c in unique]).mean():.2f}R")
        print(f"[REPLAY] MAE median={pd.Series([c['mae_r'] for c in unique]).median():.2f}R mean={pd.Series([c['mae_r'] for c in unique]).mean():.2f}R")

        print_trigger_diagnostics(unique)
        print_mfe_diagnostics(unique)
        print_tp_simulation(m5, unique)
        print_expiry_simulation(m5, unique)

        print("[REPLAY] top setups:")
        for n, c in enumerate(sorted(unique, key=lambda x: x["score"], reverse=True)[:10], 1):
            print(f"  {n}. {c['direction']} score={c['score']:.0f} outcome={c['outcome']} MFE={c['mfe_r']:.2f}R MAE={c['mae_r']:.2f}R candle={c['candle_time']} reasons={','.join(c['reasons'])}")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
