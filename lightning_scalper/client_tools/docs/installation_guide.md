# 🚀 Lightning Scalper - Client Installation Guide

**คู่มือติดตั้งระบบ Lightning Scalper สำหรับลูกค้า**

📍 **ไฟล์นี้**: `client_tools/docs/installation_guide.md`

---

## 📋 สิ่งที่ต้องเตรียม

### 1. ข้อมูลจาก IB (Introducing Broker)
- 🆔 **Client ID** - รหัสลูกค้า Lightning Scalper
- 🔑 **API Key** - คีย์สำหรับเชื่อมต่อ API
- 🔐 **API Secret** - รหัสลับสำหรับยืนยันตัวตน

### 2. บัญชี MetaTrader 5
- 📊 **หมายเลขบัญชี** - เลขบัญชี MT5
- 🔒 **รหัสผ่าน** - รหัสผ่าน MT5
- 🌐 **ชื่อเซิร์ฟเวอร์** - เช่น "YourBroker-Demo" หรือ "YourBroker-Live"

### 3. คอมพิวเตอร์/VPS
- 💻 **Windows 10/11** หรือ **Windows Server 2019+**
- 🐍 **Python 3.8+** 
- 📊 **MetaTrader 5** Terminal
- 🌐 **อินเทอร์เน็ตเสถียร**

---

## 🔧 ขั้นตอนการติดตั้ง

### Step 1: ติดตั้ง Python

1. **ดาวน์โหลด Python**
   ```
   เข้าไปที่: https://www.python.org/downloads/
   ดาวน์โหลด Python 3.8 หรือใหม่กว่า
   ```

2. **ติดตั้ง Python**
   ```
   ✅ เลือก "Add Python to PATH"
   ✅ เลือก "Install for all users"
   คลิก "Install Now"
   ```

3. **ตรวจสอบการติดตั้ง**
   ```bash
   # เปิด Command Prompt (cmd) และพิมพ์:
   python --version
   pip --version
   ```

### Step 2: ติดตั้ง MetaTrader 5

1. **ดาวน์โหลด MT5**
   ```
   เข้าไปที่เว็บไซต์ Broker ของคุณ
   ดาวน์โหลด MetaTrader 5
   ```

2. **ติดตั้งและล็อกอิน**
   ```
   - ติดตั้ง MetaTrader 5
   - ล็อกอินด้วยบัญชี Demo/Live
   - ตรวจสอบว่าเชื่อมต่อเซิร์ฟเวอร์ได้
   ```

3. **เปิดใช้งาน Expert Advisors**
   ```
   Tools → Options → Expert Advisors
   ✅ Allow automated trading
   ✅ Allow DLL imports
   ✅ Allow imports of external experts
   ```

### Step 3: ดาวน์โหลด Lightning Scalper Client

1. **สร้างโฟลเดอร์**
   ```bash
   # สร้างโฟลเดอร์บนเดสก์ท็อป
   mkdir C:\LightningScalper
   cd C:\LightningScalper
   ```

2. **ดาวน์โหลดไฟล์**
   ```
   ได้รับไฟล์จาก IB หรือดาวน์โหลดจากลิงค์ที่ได้รับ:
   - lightning_scalper_sdk.py
   - lightning_scalper_mt5_ea.py  
   - simple_client_example.py
   - client_config.json
   ```

### Step 4: ติดตั้ง Python Packages

```bash
# เปิด Command Prompt ในโฟลเดอร์ LightningScalper
cd C:\LightningScalper

# ติดตั้ง packages ที่จำเป็น
pip install MetaTrader5
pip install websocket-client
pip install requests
pip install pandas
pip install numpy
```

### Step 5: ตั้งค่า Configuration

1. **แก้ไขไฟล์ simple_client_example.py**
   ```python
   CONFIG = {
       'client_id': 'LS_CLIENT_001',        # ใส่ Client ID ที่ได้รับ
       'api_key': 'your_api_key_here',      # ใส่ API Key ที่ได้รับ
       'api_secret': 'your_api_secret_here', # ใส่ API Secret ที่ได้รับ
       
       'mt5_login': 12345678,               # ใส่เลขบัญชี MT5 จริง
       'mt5_password': 'your_password',     # ใส่รหัสผ่าน MT5 จริง
       'mt5_server': 'YourBroker-Demo',     # ใส่ชื่อเซิร์ฟเวอร์จริง
       
       'risk_per_trade': 2.0,               # ความเสี่ยง 2% ต่อเทรด
       'max_positions': 3,                  # เปิดได้สูงสุด 3 ออเดอร์
       'max_daily_trades': 10,              # เทรดได้สูงสุด 10 ครั้ง/วัน
       'daily_loss_limit': 200.0,           # ขาดทุนสูงสุด $200/วัน
       
       'use_demo': True,                    # True=Demo, False=Live
   }
   ```

### Step 6: ทดสอบการเชื่อมต่อ

1. **เปิด MetaTrader 5**
   ```
   - เปิด MT5 และล็อกอินให้เรียบร้อย
   - ตรวจสอบว่ามีสัญญาณอินเทอร์เน็ต (มุมขวาล่าง)
   ```

2. **รันระบบทดสอบ**
   ```bash
   # เปิด Command Prompt ในโฟลเดอร์
   cd C:\LightningScalper
   python simple_client_example.py
   ```

3. **ตรวจสอบผลลัพธ์**
   ```
   ✅ ต้องเห็นข้อความ "Lightning Scalper started successfully!"
   ✅ ต้องเห็นสถานะ "MT5 Connected: True"
   ✅ ต้องเห็นสถานะ "Server Connected: True"
   ```

---

## 🎯 การใช้งานระบบ

### การเริ่มต้นระบบ

```bash
# 1. เปิด MetaTrader 5 และล็อกอิน
# 2. เปิด Command Prompt
cd C:\LightningScalper
python simple_client_example.py
```

### คำสั่งขณะใช้งาน

```
s + Enter    = แสดงสถานะปัจจุบัน
q + Enter    = ออกจากระบบ
e + Enter    = เปิด/ปิด Emergency Stop
a + Enter    = เปิด/ปิด Auto Trading
Ctrl+C       = หยุดระบบ
```

### การดูผลลัพธ์

```
📊 ดูใน MT5: Tab "Trade" และ "History"
📋 ดูใน Log File: lightning_scalper_simple.log
🌐 ดูใน Dashboard: เข้าผ่านลิงค์ที่ได้รับจาก IB
```

---

## 🔒 ความปลอดภัย

### การป้องกันข้อมูล
- 🔐 **ไม่เปิดเผย API Key/Secret** ให้ใครฟัง
- 💾 **สำรองข้อมูล config** ไว้ในที่ปลอดภัย
- 🔄 **เปลี่ยนรหัสผ่าน** เป็นระยะ

### การจัดการความเสี่ยง
- 💰 **ตั้งขีดจำกัดขาดทุน** ที่เหมาะสม
- 📊 **ใช้บัญชี Demo** ก่อนใช้ Live
- 👀 **ติดตามผลการเทรด** สม่ำเสมอ

---

## 🆘 แก้ไขปัญหา

### ปัญหาที่พบบ่อย

#### 1. "Failed to connect to MT5"
```
✅ ตรวจสอบ MT5 เปิดอยู่และล็อกอินแล้ว
✅ ตรวจสอบ Expert Advisors เปิดใช้งาน
✅ ตรวจสอบ username/password/server ถูกต้อง
```

#### 2. "Authentication failed"
```
✅ ตรวจสอบ Client ID, API Key, API Secret
✅ ตรวจสอบใช้ server environment ที่ถูกต้อง (demo/live)
✅ ติดต่อ IB เพื่อยืนยันข้อมูล
```

#### 3. "No signals received"
```
✅ ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
✅ ตรวจสอบ trading sessions เปิดอยู่
✅ รอสักครู่ signals อาจมาไม่ต่อเนื่อง
```

#### 4. "Orders not executing"
```
✅ ตรวจสอบ auto_trading = True
✅ ตรวจสอบไม่ติด emergency_stop
✅ ตรวจสอบ spread ไม่สูงเกินไป
✅ ตรวจสอบยอดเงินในบัญชีพอ
```

### การติดต่อสำหรับความช่วยเหลือ

```
📧 Email: support@your-ib-company.com
📱 Line: @your-ib-line
📞 Phone: +66-xxx-xxx-xxxx
🌐 Dashboard: https://your-dashboard-url.com
```

---

## 📈 การปรับแต่งขั้นสูง

### ปรับพารามิเตอร์การเทรด

```python
# แก้ไขใน simple_client_example.py
CONFIG = {
    'risk_per_trade': 1.5,      # ลดเป็น 1.5% ถ้าต้องการรักษาทุน
    'max_positions': 5,         # เพิ่มเป็น 5 ถ้าต้องการเทรดมากขึ้น
    'max_daily_trades': 15,     # เพิ่มเป็น 15 ถ้าต้องการเทรดบ่อยขึ้น
    'daily_loss_limit': 100.0,  # ลดเป็น $100 ถ้าต้องการรักษาทุน
}
```

### การใช้งาน Advanced Configuration

```python
# ใช้ client_config.json สำหรับการตั้งค่าละเอียด
# อ่านเพิ่มเติมใน advanced_configuration.md
```

---

## ✅ Checklist ก่อนใช้งาน Live

```
□ ทดสอบระบบกับบัญชี Demo แล้ว
□ ตรวจสอบผลการเทรดใน Demo อย่างน้อย 1 สัปดาห์
□ เข้าใจการทำงานของระบบแล้ว
□ ตั้งค่า risk management ที่เหมาะสม
□ เตรียมทุนที่พร้อมจะเสี่ยงได้
□ ตรวจสอบการเชื่อมต่อเสถียร
□ มี VPS หรือคอมพิวเตอร์ที่เปิดตลอด 24/5
□ ติดต่อ IB เพื่อยืนยันการใช้งาน Live
```

---

## 🎓 คำแนะนำสำหรับมือใหม่

### สัปดาห์แรก
1. **ใช้บัญชี Demo เท่านั้น**
2. **ดูการทำงานของระบบ**
3. **เรียนรู้การอ่าน log file**

### สัปดาห์ที่ 2-4
1. **วิเคราะห์ผลการเทรด**
2. **ปรับแต่งพารามิเตอร์**
3. **ทดสอบในสถานการณ์ต่าง ๆ**

### เมื่อพร้อมใช้ Live
1. **เริ่มด้วยทุนน้อย**
2. **ใช้ risk ต่ำ (1-2%)**
3. **ติดตามผลอย่างใกล้ชิด**

---

**🎯 ขอให้การเทรดของคุณประสบความสำเร็จ!**

*Lightning Scalper - Professional FVG Trading System*