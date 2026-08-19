import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tvDatafeed import TvDatafeed, Interval

# ==========================================
# 1. PARAMETERS CONFIGURATION (ตั้งค่าตามจริงของมี่)
# ==========================================
TUNING = 5                # Order Block Candle Count
AMOUNT_OF_BOXES = 10      # ซอย OB เป็น 10 ช่องย่อยเพื่อหา POC
PIVOT_LEFT = 8            # Liquidity Pivot Left
PIVOT_RIGHT = 3           # Liquidity Pivot Right
THRESHOLD_PCT = 0.03      # Equality Threshold (%)
MAX_ENTRY_DISTANCE = 2.0  # ถ้าราคาปัจจุบันห่างจาก POC เกิน $2.0 (200 จุด) ไม่ส่ง Alert

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==========================================
# 2. DATA FETCHING & TELEGRAM HELPER
# ==========================================
def fetch_gold_data_tv():
    """ดึงข้อมูลราคาและ Volume ตรงจาก TradingView (OANDA:XAUUSD)"""
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol='XAUUSD', exchange='OANDA', interval=Interval.in_30_minute, n_bars=100)
        if df is None or df.empty:
            print("⚠️ ไม่สามารถดึงข้อมูลจาก TradingView ได้")
            return None
            
        df.index = pd.to_datetime(df.index)
        tz_th = pytz.timezone('Asia/Bangkok')
        df.index = df.index.tz_localize('UTC').tz_convert(tz_th)
        
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"❌ Error fetching TradingView data: {e}")
        return None

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ ไม่พบ Token/ChatID - แสดงผลบน Console:")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")


# ==========================================
# 3. CORE ANALYTICS ENGINE
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

def check_recent_swept_eqh(df, p_highs):
    if len(p_highs) < 2: return None
    for i in range(len(p_highs) - 1, 0, -1):
        idx2, price2, vol2 = p_highs[i]
        idx1, price1, vol1 = p_highs[i - 1]
        diff = abs(price2 - price1) / price1 * 100
        if diff <= THRESHOLD_PCT:
            eqh_top = max(price1, price2)
            latest_high = df['High'].iloc[-1]
            latest_close = df['Close'].iloc[-1]
            if latest_high > eqh_top and latest_close < eqh_top:
                return {'swept_level': eqh_top, 'total_vol': vol1 + vol2}
    return None

def calculate_ob_poc(df):
    if len(df) < TUNING: return None
    sub_df = df.iloc[-TUNING:]
    first_bar = sub_df.iloc[0]
    
    # เงื่อนไข Bearish OB: แท่งแรกเขียว + อีก 4 แท่งปิดแดง
    if first_bar['Close'] > first_bar['Open'] and all(sub_df.iloc[1:]['Close'] <= sub_df.iloc[1:]['Open']):
        ob_top, ob_bot = first_bar['High'], first_bar['Low']
        increment = (ob_top - ob_bot) / AMOUNT_OF_BOXES
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
        poc_midpoint = (poc_top + poc_bot) / 2  # 50% Midpoint of POC
        return {'ob_top': ob_top, 'ob_bot': ob_bot, 'poc_midpoint': poc_midpoint}
    return None


# ==========================================
# 4. EXECUTION & ALERT GENERATOR
# ==========================================
def run_scanner():
    print("🔎 กำลังดึงข้อมูล OANDA:XAUUSD (M30)...")
    df = fetch_gold_data_tv()
    if df is None: return
    
    last_candle_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    current_price = df['Close'].iloc[-1]
    
    p_highs, p_lows = find_pivots(df)
    swept_info = check_recent_swept_eqh(df, p_highs)
    ob_info = calculate_ob_poc(df)
    
    if swept_info or ob_info:
        entry_price = ob_info['poc_midpoint'] if ob_info else swept_info['swept_level']
        
        # Filter: ราคาวิ่งออกห่างเกิน $2.0 ไม่ส่ง Alert
        if abs(current_price - entry_price) > MAX_ENTRY_DISTANCE:
            print(f"⚠️ สแกนพบ Setup แต่ราคาวิ่งเลยจุด Entry ({entry_price:.2f}) ไปแล้ว - ข้ามการส่ง Alert")
            return
            
        sl_price = max(swept_info['swept_level'] if swept_info else 0, ob_info['ob_top'] if ob_info else 0) + 1.5
        tp_price = entry_price - ((sl_price - entry_price) * 3)  # Risk:Reward 1:3
        
        message = (
            f"🔔 *พบ Setup เทรด: XAUUSD (M30)*\n"
            f"📅 *Candle Time:* `{last_candle_time} (TH)`\n"
            f"----------------------------------\n"
            f"🎯 *Direction:* SHORT (Bearish Setup)\n"
            f"📍 *Current Price:* {current_price:.2f}\n"
            f"📊 *50% POC Entry:* `{entry_price:.2f}`\n"
            f"🛑 *Suggested SL:* `{sl_price:.2f}`\n"
            f"🟢 *Suggested TP (1:3):* `{tp_price:.2f}`\n"
            f"----------------------------------\n"
            f"⚠️ *Note:* ตั้ง Sell Limit ดักไว้ และลบออเดอร์ทันทีหากราคาวิ่งลงไปชน TP ก่อนเกี่ยว"
        )
        send_telegram_alert(message)
        print("✅ สแกนพบ Setup และส่งแจ้งเตือน Telegram เรียบร้อย!")
    else:
        print(f"ℹ️ [{last_candle_time}] ไม่พบ Setup ตามเงื่อนไขในแท่งปัจจุบัน")

if __name__ == "__main__":
    run_scanner()
