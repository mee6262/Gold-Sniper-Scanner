# 🔱 XAUUSD Smart Money Concepts (SMC) Scanner

ระบบสแกนสัญญาณเทรดทองคำ (XAUUSD) บน Timeframe M30 แบบอัตโนมัติ พัฒนาด้วย Python และรันบน GitHub Actions พร้อมแจ้งเตือนผ่าน Telegram Bot โดยไม่มีค่าใช้จ่ายรายเดือน

---

## 🌟 Key Features

* **Dynamic Order Block & POC Engine:** ตรวจหา Order Block ตามโครงสร้าง SMC พร้อมประมวลผล Volume Profile แบบ 10 ช่องย่อยเพื่อหาจุด **POC (Point of Control)**
* **Fair Value Gap (FVG) Detection:** คำนวณหาจุด Imbalance บนโครงสร้างแท่งเทียนย้อนหลัง 3 แท่ง
* **Liquidity Sweeps (EQH / EQL):** สแกนหาจุด Fakeout บริเวณ Equal Highs และ Equal Lows เพื่อจับจังหวะกวาด Liquidity
* **Mxwll Volume & Session Filters:**
  * **Volume Activity:** วัดเปอร์เซ็นต์ความหนาแน่นของ Volume ย้อนหลัง 24 ชั่วโมง (`Very High` / `High` / `Normal`)
  * **Market Session Filter:** ระบุช่วงเวลาตลาด (London, NY, London/NY Overlap, Asia) เพื่อหลีกเลี่ยงช่วงไร้สภาพคล่อง
* **Automated Risk Management:** คำนวณจุด Entry, Stop Loss (Buffer +1.5$) และ Take Profit (R:R 1:3) พร้อมระบบยกเลิกแจ้งเตือนหากราคาปิดทะลุ SL ไปก่อน
* **Serverless & Zero Cost:** รันผ่าน GitHub Actions ทุกๆ 30 นาที (ตรงเวลาปิดแท่ง M30) พร้อมระบบ **Deduplication Guard** ป้องกันการยิงแจ้งเตือนซ้ำในแท่งเดิม

---

## 📁 Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── scanner.yml      # GitHub Actions Cron Configuration
├── .last_alert_time         # State file ป้องกันการส่ง Alert ซ้ำ
├── scanner.py               # Main Analytics Engine
├── requirements.txt         # Python Dependencies
└── README.md                # Project Documentation
