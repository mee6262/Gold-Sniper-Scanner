# Gold Sniper Scanner 🎯

ระบบสแกนสัญญาณเทรดทองคำ (XAUUSD) อัตโนมัติ รันผ่าน GitHub Actions ส่งแจ้งเตือนเข้า Telegram

## 📌 Features & Strategies
- **Data Feed:** ดึงราคาเรียลไทม์ตรงจาก `OANDA:XAUUSD` (M30)
- **Logic:** ตรวจจับ Liquidity Swept (EQH/EQL) ร่วมกับ Volume Profile Order Block 10 Grids (หาจุด 50% POC)
- **Filters:** ป้องกันการส่ง Alert ราคาวิ่งเลยจุดเข้าเกิน $2.0 (200 จุด) และมีระบบ Deduplication กันสแปมข้อความซ้ำ
- **Automation:** รันอัตโนมัติทุก 15 นาทีผ่าน GitHub Actions (Serverless)

## ⚙️ Required Repository Secrets
ตั้งค่าใน **Settings > Secrets and variables > Actions**:
1. `TELEGRAM_TOKEN` : Bot Token จาก @BotFather
2. `TELEGRAM_CHAT_ID` : Telegram Chat ID ตัวเลขของคุณ
