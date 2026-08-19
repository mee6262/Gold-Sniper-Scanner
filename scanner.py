import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tvDatafeed import TvDatafeed, Interval

# ==========================================
# 1. PARAMETERS CONFIGURATION
# ==========================================
PIVOT_LEFT = 8            # Liquidity Pivot Left
PIVOT_RIGHT = 3           # Liquidity Pivot Right
THRESHOLD_PCT = 0.03      # Equality Threshold (%)
AMOUNT_OF_BOXES = 10      # ซอย OB เป็น 10 ช่องย่อยเพื่อหา POC
LAST_ALERT_FILE = ".last_alert_time"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==========================================
# 2. DATA FETCHING & TELEGRAM HELPER
# ==========================================
def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ ไม่พบ Token/ChatID - แสดงผลบน Console:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")

def fetch_gold_data_tv():
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol='XAUUSD', exchange='OANDA', interval=Interval.in_30_minute, n_bars=100)
        if df is None or df.empty:
            send_telegram_alert("⚠️ *Scanner System Alert:* ไม่สามารถดึงข้อมูลจาก TradingView ได้ (Rate Limit/IP Block)")
            return None
            
        df.index = pd.to_datetime(df.index)
        tz_th = pytz.timezone('Asia/Bangkok')
        df.index = df.index.tz_localize('UTC').tz_convert(tz_th)
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        send_telegram_alert(f"❌ *Scanner System Error:* {e}")
        return None


# ==========================================
# 3. MXWLL FILTERS (VOLUME & SESSION)
# ==========================================
def calculate_volume_activity(df, lookback=48):
    if len(df) < lookback:
        return "Normal"
    recent_vol = df['Volume'].iloc[-lookback:]
    current_vol = df['Volume'].iloc[-1]
    
    p66 = np.percentile(recent_vol, 66)
    p90 = np.percentile(recent_vol, 90)
    
    if current_vol >= p90:
        return "Very High 🔥"
    elif current_vol >= p66:
        return "High ⚡"
    else:
        return "Normal / Low 💤"

def get_market_session(dt_th):
    hour = dt_th.hour
    # เวลาไทย (UTC+7)
    if 14 <= hour < 19:
        return "London Session 🇬🇧"
    elif 19 <= hour < 22:
        return "London/NY Overlap 🇺🇸🇬🇧"
    elif 22 <= hour or hour < 4:
        return "New York Session 🇺🇸"
    elif 6 <= hour < 13:
        return "Asia Session 🇯🇵"
    else:
        return "Dead Zone 😴"


# ==========================================
# 4. CORE ANALYTICS ENGINE (DYNAMIC SMC)
# ==========================================
def find_pivots(df):
    highs, lows, volumes = df['High'].values, df['Low'].values, df['Volume'].values
    n = len(df)
    p_highs, p_lows = [], []
    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        if all(highs[i] > highs[i - PIVOT_LEFT:i]) and all(highs[i] > highs[i + 1:i + PIVOT_RIGHT + 1]):
            p_highs.append((i, highs[i], volumes[i]))
        if all(lows[i] < lows[i - PIVOT_LEFT:i]) and all(lows[i] < lows[i + 1:i + PIVOT_RIGHT + 1]):
            p_lows.append((i, lows[i], volumes[i]))
    return p_highs, p_lows

def check_swept_liquidity(df, p_highs, p_lows):
    latest_high = df['High'].iloc[-1]
    latest_low = df['Low'].iloc[-1]
    latest_close = df['Close'].iloc[-1]
    
    # Check Bearish Swept EQH
    if len(p_highs) >= 2:
        for i in range(len(p_highs) - 1, 0, -1):
            _, price2, _ = p_highs[i]
            _, price1, _ = p_highs[i - 1]
            if abs(price2 - price1) / price1 * 100 <= THRESHOLD_PCT:
                eqh_top = max(price1, price2)
                if latest_high > eqh_top and latest_close < eqh_top:
                    return {'direction': 'SHORT', 'level': eqh_top}
                    
    # Check Bullish Swept EQL
    if len(p_lows) >= 2:
        for i in range(len(p_lows) - 1, 0, -1):
            _, price2, _ = p_lows[i]
            _, price1, _ = p_lows[i - 1]
            if abs(price2 - price1) / price1 * 100 <= THRESHOLD_PCT:
                eql_bot = min(price1, price2)
                if latest_low < eql_bot and latest_close > eql_bot:
                    return {'direction': 'LONG', 'level': eql_bot}
                    
    return None

def check_fvg(df):
    if len(df) < 3: return None
    c1_high, c1_low = df['High'].iloc[-3], df['Low'].iloc[-3]
    c3_high, c3_low = df['High'].iloc[-1], df['Low'].iloc[-1]
    
    # Bullish FVG: Low ของแท่งปัจจุบัน อยู่สูงกว่า High ของแท่งเมื่อ 2 แท่งก่อน
    if c3_low > c1_high:
        gap_mid = (c3_low + c1_high) / 2
        return {'direction': 'LONG', 'level': gap_mid, 'top': c3_low, 'bot': c1_high}
        
    # Bearish FVG: High ของแท่งปัจจุบัน อยู่ต่ำกว่า Low ของแท่งเมื่อ 2 แท่งก่อน
    if c3_high < c1_low:
        gap_mid = (c3_high + c1_low) / 2
        return {'direction': 'SHORT', 'level': gap_mid, 'top': c1_low, 'bot': c3_high}
        
    return None

def calculate_ob_poc(df):
    """
    Dynamic Order Block (SMC Standard):
    หาแท่งสีตรงข้ามย้อนหลังในระยะ 6 แท่งล่าสุด ที่มีแท่ง Impulse พุ่งออกไป
    """
    if len(df) < 10: return None
    
    # สแกนย้อนหลังจากแท่งล่าสุดกลับไป 6 แท่ง
    for i in range(2, 7):
        target_bar = df.iloc[-i]
        next_bar = df.iloc[-i+1]
        
        is_red = target_bar['Close'] < target_bar['Open']
        is_green = target_bar['Close'] > target_bar['Open']
        
        avg_body = abs(df['Close'] - df['Open']).iloc[-20:].mean()
        next_body = abs(next_bar['Close'] - next_bar['Open'])
        
        # Bullish OB: แท่งแดง ที่ถูกแท่งเขียวถัดมาพุ่งกลืน (Engulfing/Impulse)
        if is_red and (next_bar['Close'] > next_bar['Open']) and (next_body > avg_body * 1.2):
            direction = 'LONG'
            ob_top, ob_bot = target_bar['High'], target_bar['Low']
            sub_df = df.iloc[-i:-i+2]
            break
        # Bearish OB: แท่งเขียว ที่ถูกแท่งแดงถัดมาพุ่งกลืน
        elif is_green and (next_bar['Close'] < next_bar['Open']) and (next_body > avg_body * 1.2):
            direction = 'SHORT'
            ob_top, ob_bot = target_bar['High'], target_bar['Low']
            sub_df = df.iloc[-i:-i+2]
            break
    else:
        return None

    # คำนวณ POC 10 ช่องย่อยภายในกรอบ OB
    increment = (ob_top - ob_bot) / AMOUNT_OF_BOXES
    if increment <= 0: return None
    box_volumes = [0.0] * AMOUNT_OF_BOXES
    
    for _, row in sub_df.iterrows():
        c_high, c_low, c_vol = row['High'], row['Low'], row['Volume']
        ltf_diff = c_high - c_low if c_high != c_low else 0.0001
        for b_idx in range(AMOUNT_OF_BOXES):
            top_grid = ob_top - (increment * b_idx)
            bot_grid = ob_top - (increment * (b_idx + 1))
            if c_low <= top_grid and c_high >= bot_grid:
                reg_diff = min(c_high, top_grid) - max(c_low, bot_grid)
                box_volumes[b_idx] += (reg_diff / ltf_diff) * c_vol
                
    max_vol_idx = int(np.argmax(box_volumes))
    poc_top = ob_top - (increment * max_vol_idx)
    poc_bot = ob_top - (increment * (max_vol_idx + 1))
    poc_midpoint = (poc_top + poc_bot) / 2
    
    return {
        'direction': direction,
        'ob_top': ob_top,
        'ob_bot': ob_bot,
        'poc_midpoint': poc_midpoint
    }


# ==========================================
# 5. EXECUTION & DEDUPLICATION
# ==========================================
def run_scanner():
    df = fetch_gold_data_tv()
    if df is None: return
    
    last_candle_dt = df.index[-1]
    last_candle_time = last_candle_dt.strftime('%Y-%m-%d %H:%M')
    
    # Deduplication Guard
    if os.path.exists(LAST_ALERT_FILE):
        with open(LAST_ALERT_FILE, 'r') as f:
            if f.read().strip() == last_candle_time:
                print(f"ℹ️ [{last_candle_time}] แท่งนี้เคยส่ง Alert ไปแล้ว - ข้ามการทำงาน")
                return

    current_price = df['Close'].iloc[-1]
    p_highs, p_lows = find_pivots(df)
    
    # ดึงค่าจาก Analytics Engine ต่างๆ
    swept_info = check_swept_liquidity(df, p_highs, p_lows)
    ob_info = calculate_ob_poc(df)
    fvg_info = check_fvg(df)
    
    vol_activity = calculate_volume_activity(df)
    market_session = get_market_session(last_candle_dt)

    # รวมสัญญาณและประเมินทิศทางหลัก
    signals = [s for s in [swept_info, ob_info, fvg_info] if s is not None]
    if not signals:
        print(f"ℹ️ [{last_candle_time}] ไม่พบ Setup ตามเงื่อนไขในแท่งปัจจุบัน")
        return

    # ตรวจสอบ Confluence Direction
    directions = [s['direction'] for s in signals]
    long_count = directions.count('LONG')
    short_count = directions.count('SHORT')
    
    if long_count > short_count:
        setup_direction = 'LONG'
    elif short_count > long_count:
        setup_direction = 'SHORT'
    else:
        print(f"ℹ️ [{last_candle_time}] สัญญาณขัดแย้งกัน (Long vs Short) - ข้ามการส่ง Alert")
        return

    # คำนวณจุด Entry, SL, TP
    confluence_reasons = []
    entry_candidates = []
    
    if ob_info and ob_info['direction'] == setup_direction:
        entry_candidates.append(ob_info['poc_midpoint'])
        confluence_reasons.append("Order Block (POC)")
    if fvg_info and fvg_info['direction'] == setup_direction:
        entry_candidates.append(fvg_info['level'])
        confluence_reasons.append("Fair Value Gap (FVG)")
    if swept_info and swept_info['direction'] == setup_direction:
        entry_candidates.append(swept_info['level'])
        confluence_reasons.append("Liquidity Swept (EQH/EQL)")

    entry_price = np.mean(entry_candidates) if entry_candidates else current_price
    
    if setup_direction == 'SHORT':
        high_bounds = [p for p in [
            ob_info['ob_top'] if ob_info and ob_info['direction'] == 'SHORT' else None,
            swept_info['level'] if swept_info and swept_info['direction'] == 'SHORT' else None,
            fvg_info['top'] if fvg_info and fvg_info['direction'] == 'SHORT' else None
        ] if p is not None]
        sl_price = (max(high_bounds) if high_bounds else current_price) + 1.5
        tp_price = entry_price - ((sl_price - entry_price) * 3)
        
        if current_price > sl_price:
            print(f"⚠️ พบ Short Setup แต่ราคาปิดทะลุ SL ({sl_price:.2f}) ขึ้นไปแล้ว - ข้าม")
            return
    else: # LONG
        low_bounds = [p for p in [
            ob_info['ob_bot'] if ob_info and ob_info['direction'] == 'LONG' else None,
            swept_info['level'] if swept_info and swept_info['direction'] == 'LONG' else None,
            fvg_info['bot'] if fvg_info and fvg_info['direction'] == 'LONG' else None
        ] if p is not None]
        sl_price = (min(low_bounds) if low_bounds else current_price) - 1.5
        tp_price = entry_price + ((entry_price - sl_price) * 3)
        
        if current_price < sl_price:
            print(f"⚠️ พบ Long Setup แต่ราคาปิดทะลุ SL ({sl_price:.2f}) ลงไปแล้ว - ข้าม")
            return

    # สร้างข้อความแจ้งเตือน Telegram
    reasons_str = " + ".join(confluence_reasons)
    message = (
        f"🚨 *พบ Trade Setup: XAUUSD (M30)*\n"
        f"📅 *Candle Time:* `{last_candle_time} (TH)`\n"
        f"🌐 *Session:* {market_session}\n"
        f"📊 *Volume Activity:* {vol_activity}\n"
        f"----------------------------------\n"
        f"🎯 *Direction:* `{setup_direction}`\n"
        f"🧩 *Confluence:* {reasons_str}\n"
        f"📍 *Current Price:* `{current_price:.2f}`\n"
        f"🎯 *Suggested Entry:* `{entry_price:.2f}`\n"
        f"🛑 *Suggested SL:* `{sl_price:.2f}`\n"
        f"🟢 *Suggested TP (1:3):* `{tp_price:.2f}`\n"
        f"----------------------------------\n"
        f"💡 *Note:* พิจารณาตั้ง Limit Order หรือรอยืนยันตามแผนการเทรด"
    )
    
    send_telegram_alert(message)
    
    with open(LAST_ALERT_FILE, 'w') as f:
        f.write(last_candle_time)
    print("✅ ส่งแจ้งเตือนเรียบร้อย!")

if __name__ == "__main__":
    run_scanner()
