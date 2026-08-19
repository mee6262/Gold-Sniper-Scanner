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
TUNING = 5                # Order Block Candle Count
AMOUNT_OF_BOXES = 10      # ซอย OB เป็น 10 ช่องย่อยเพื่อหา POC
PIVOT_LEFT = 8            # Liquidity Pivot Left
PIVOT_RIGHT = 3           # Liquidity Pivot Right
THRESHOLD_PCT = 0.03      # Equality Threshold (%)
MAX_ENTRY_DISTANCE = 2.0  # ถ้าราคาห่างจาก Entry เกิน $2.0 (200 จุด) ไม่ส่ง Alert
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
# 3. CORE ANALYTICS ENGINE (DYNAMIC)
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

def calculate_ob_poc(df):
    if len(df) < TUNING: return None
    sub_df = df.iloc[-TUNING:]
    first_bar = sub_df.iloc[0]
    
    is_bear_ob = (first_bar['Close'] > first_bar['Open']) and all(sub_df.iloc[1:]['Close'] <= sub_df.iloc[1:]['Open'])
    is_bull_ob = (first_bar['Close'] < first_bar['Open']) and all(sub_df.iloc[1:]['Close'] >= sub_df.iloc[1:]['Open'])
    
    if not (is_bear_ob or is_bull_ob):
        return None
        
    direction = 'SHORT' if is_bear_ob else 'LONG'
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
    poc_midpoint = (poc_top + poc_bot) / 2
    
    return {
        'direction': direction,
        'ob_top': ob_top,
        'ob_bot': ob_bot,
        'poc_midpoint': poc_midpoint
    }


# ==========================================
# 4. EXECUTION & DEDUPLICATION
# ==========================================
def run_scanner():
    df = fetch_gold_data_tv()
    if df is None: return
    
    last_candle_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    
    # Deduplication Guard
    if os.path.exists(LAST_ALERT_FILE):
        with open(LAST_ALERT_FILE, 'r') as f:
            if f.read().strip() == last_candle_time:
                print(f"ℹ️ [{last_candle_time}] แท่งนี้เคยส่ง Alert ไปแล้ว - ข้ามการทำงาน")
                return

    current_price = df['Close'].iloc[-1]
    p_highs, p_lows = find_pivots(df)
    swept_info = check_swept_liquidity(df, p_highs, p_lows)
    ob_info = calculate_ob_poc(df)

    # --- DEBUG: ดูว่าใกล้เกณฑ์แค่ไหน ---
    print(f"🔍 Debug | Pivot Highs: {len(p_highs)} | Pivot Lows: {len(p_lows)}")
    if len(p_highs) >= 2:
        _, ph2, _ = p_highs[-1]; _, ph1, _ = p_highs[-2]
        diff_h = abs(ph2 - ph1) / ph1 * 100
        print(f"   ล่าสุด Pivot High คู่ท้าย: {ph1:.2f} vs {ph2:.2f} (ห่างกัน {diff_h:.4f}% | เกณฑ์ {THRESHOLD_PCT}%)")
    if len(p_lows) >= 2:
        _, pl2, _ = p_lows[-1]; _, pl1, _ = p_lows[-2]
        diff_l = abs(pl2 - pl1) / pl1 * 100
        print(f"   ล่าสุด Pivot Low คู่ท้าย: {pl1:.2f} vs {pl2:.2f} (ห่างกัน {diff_l:.4f}% | เกณฑ์ {THRESHOLD_PCT}%)")
    last5 = df.iloc[-TUNING:][['Open', 'Close']]
    colors = ['เขียว' if r.Close > r.Open else 'แดง' for r in last5.itertuples()]
    print(f"   5 แท่งล่าสุด (สำหรับ OB check): {colors}")
    # --- END DEBUG ---

    # Resolve direction & filter out conflicting swept_info
    setup_direction = None
    if swept_info and ob_info:
        if swept_info['direction'] == ob_info['direction']:
            setup_direction = swept_info['direction']
        else:
            # หากทิศทางขัดกัน ตัด swept_info ทิ้ง ไม่นำ level มาคำนวณ SL ข้ามฝั่ง
            swept_info = None 
            setup_direction = ob_info['direction']
    elif ob_info:
        setup_direction = ob_info['direction']
    elif swept_info:
        setup_direction = swept_info['direction']
        
    if setup_direction:
        entry_price = ob_info['poc_midpoint'] if ob_info else swept_info['level']
        
        if setup_direction == 'SHORT':
            high_bounds = [p for p in [ob_info['ob_top'] if ob_info else None, swept_info['level'] if swept_info else None] if p is not None]
            sl_price = max(high_bounds) + 1.5
            tp_price = entry_price - ((sl_price - entry_price) * 3)
        else: # LONG
            low_bounds = [p for p in [ob_info['ob_bot'] if ob_info else None, swept_info['level'] if swept_info else None] if p is not None]
            sl_price = min(low_bounds) - 1.5
            tp_price = entry_price + ((entry_price - sl_price) * 3)

        if abs(current_price - entry_price) > MAX_ENTRY_DISTANCE:
            print(f"⚠️ สแกนพบ Setup {setup_direction} แต่ราคาวิ่งเลยจุด Entry ไปแล้ว - ข้ามการส่ง Alert")
            return
            
        message = (
            f"🔔 *พบ Setup เทรด: XAUUSD (M30)*\n"
            f"📅 *Candle Time:* `{last_candle_time} (TH)`\n"
            f"----------------------------------\n"
            f"🎯 *Direction:* {setup_direction}\n"
            f"📍 *Current Price:* {current_price:.2f}\n"
            f"📊 *50% POC Entry:* `{entry_price:.2f}`\n"
            f"🛑 *Suggested SL:* `{sl_price:.2f}`\n"
            f"🟢 *Suggested TP (1:3):* `{tp_price:.2f}`\n"
            f"----------------------------------\n"
            f"⚠️ *Note:* ตั้งวาง Limit Order ดักไว้ และลบออเดอร์ทันทีหากราคาวิ่งชน TP ก่อนเกี่ยว"
        )
        send_telegram_alert(message)
        
        with open(LAST_ALERT_FILE, 'w') as f:
            f.write(last_candle_time)
        print("✅ ส่งแจ้งเตือนเรียบร้อย!")
    else:
        print(f"ℹ️ [{last_candle_time}] ไม่พบ Setup ตามเงื่อนไขในแท่งปัจจุบัน")

if __name__ == "__main__":
    run_scanner()
