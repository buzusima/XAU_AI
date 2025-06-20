import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
import logging

class RawCandlestickCollector:
    """
    ระบบดึงข้อมูลแท่งเทียนดิบสำหรับ AI Training
    เฉพาะ OHLCV + Time context
    ไม่มี Feature Engineering - ให้ AI เรียนรู้เอง
    """
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        self.original_symbol = symbol
        self.symbol = None
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MT5 and find working symbol
        if self._initialize_mt5():
            self.symbol = self._find_working_symbol()
    
    def _initialize_mt5(self) -> bool:
        """เชื่อมต่อ MT5"""
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        
        self.logger.info("MT5 initialized successfully")
        return True
    
    def _find_working_symbol(self) -> str:
        """หาสัญลักษณ์ที่ใช้งานได้จริง"""
        gold_symbols = [
            self.original_symbol, "XAUUSD", "XAUUSD.c", "XAUUSDm", 
            "GOLD", "GOLDm", "GOLD.c", "#XAUUSD"
        ]
        
        self.logger.info("🔍 ค้นหาสัญลักษณ์ที่ใช้งานได้...")
        
        for symbol in gold_symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                continue
                
            # ทดสอบดึงข้อมูล
            test_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)
            if test_rates is not None and len(test_rates) > 0:
                self.logger.info(f"✅ ใช้สัญลักษณ์: {symbol}")
                return symbol
        
        self.logger.error("❌ ไม่พบสัญลักษณ์ที่ใช้งานได้")
        return None
    
    def get_raw_candlestick_data(self, timeframe: str, bars: int = 10000) -> pd.DataFrame:
        """
        ดึงข้อมูลแท่งเทียนดิบ - เฉพาะ OHLCV
        
        Args:
            timeframe: M1, M5, M30, H1, H4, D1
            bars: จำนวนแท่งที่ต้องการ
            
        Returns:
            DataFrame: Pure OHLCV data with datetime index
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe. Use: {list(self.timeframes.keys())}")
        
        mt5_timeframe = self.timeframes[timeframe]
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, bars)
        
        if rates is None:
            self.logger.error(f"Failed to get rates for {self.symbol} {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # เก็บเฉพาะ Raw OHLCV
        raw_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        df = df[raw_columns]
        
        # Rename columns เป็นชื่อมาตรฐาน
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # เพิ่มเฉพาะ Time Context พื้นฐาน (ไม่ใช่ Technical Analysis)
        df = self._add_time_context(df)
        
        self.logger.info(f"📊 {timeframe}: ดึงได้ {len(df):,} แท่ง (Raw OHLCV)")
        return df
    
    def _add_time_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่มเฉพาะ Time Context พื้นฐาน
        ไม่ใช่ Technical Analysis
        """
        df = df.copy()
        
        # === Basic Time Information ===
        df['Hour'] = df.index.hour
        df['Day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['Day_of_month'] = df.index.day
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        
        # === Market Session (GMT+0) ===
        # ข้อมูลพื้นฐานช่วงเวลาตลาด - ไม่ใช่ analysis
        df['Is_asian_hours'] = ((df['Hour'] >= 0) & (df['Hour'] < 9)).astype(int)
        df['Is_european_hours'] = ((df['Hour'] >= 8) & (df['Hour'] < 17)).astype(int)
        df['Is_us_hours'] = ((df['Hour'] >= 13) & (df['Hour'] < 22)).astype(int)
        
        # === Weekend Markers ===
        df['Is_monday'] = (df['Day_of_week'] == 0).astype(int)
        df['Is_friday'] = (df['Day_of_week'] == 4).astype(int)
        df['Is_weekend'] = (df['Day_of_week'] >= 5).astype(int)
        
        return df
    
    def get_multi_timeframe_raw_data(self, bars_per_tf: Dict[str, int] = None) -> Dict[str, pd.DataFrame]:
        """
        ดึงข้อมูล Raw หลายไทม์เฟรม
        
        Args:
            bars_per_tf: จำนวนแท่งต่อไทม์เฟรม
        """
        if bars_per_tf is None:
            # Default bars - ปรับให้เหมาะสมกับการเทรน AI
            bars_per_tf = {
                'D1': 2000,    # ~5-6 ปี
                'H4': 10000,   # ~4-5 ปี  
                'H1': 20000,   # ~2-3 ปี
                'M30': 30000,  # ~1-2 ปี
                'M5': 50000,   # ~6 เดือน
                'M1': 60000    # ~6 สัปดาห์
            }
        
        self.logger.info("📦 เริ่มดึงข้อมูล Raw Candlestick หลายไทม์เฟรม")
        
        raw_data = {}
        
        # เรียงจากไทม์เฟรมใหญ่ไปเล็ก (stable)
        for tf in ['D1', 'H4', 'H1', 'M30', 'M5', 'M1']:
            if tf in bars_per_tf:
                bars = bars_per_tf[tf]
                
                try:
                    df = self.get_raw_candlestick_data(tf, bars)
                    
                    if not df.empty:
                        raw_data[tf] = df
                        coverage_days = (df.index.max() - df.index.min()).days
                        self.logger.info(f"✅ {tf}: {len(df):,} แท่ง | {coverage_days} วัน")
                    else:
                        self.logger.warning(f"❌ {tf}: ไม่มีข้อมูล")
                        
                except Exception as e:
                    self.logger.error(f"❌ {tf}: Error - {str(e)}")
                    continue
                
                time.sleep(0.3)  # พักเล็กน้อย
        
        return raw_data
    
    def prepare_ai_raw_dataset(self, save_to_file: bool = True) -> Dict[str, pd.DataFrame]:
        """
        เตรียมชุดข้อมูล Raw สำหรับ AI Training
        เฉพาะ OHLCV + Time Context
        """
        if not self.symbol:
            self.logger.error("ไม่มีสัญลักษณ์ที่ใช้งานได้")
            return {}
        
        self.logger.info("🚀 เตรียม Raw Dataset สำหรับ AI Training")
        
        # ดึงข้อมูล Raw
        raw_data = self.get_multi_timeframe_raw_data()
        
        if not raw_data:
            self.logger.error("ไม่สามารถดึงข้อมูลได้")
            return {}
        
        # ทำความสะอาดข้อมูล
        cleaned_data = {}
        for tf, df in raw_data.items():
            # ลบข้อมูลที่ผิดปกติ
            df_clean = self._clean_raw_data(df, tf)
            cleaned_data[tf] = df_clean
        
        # บันทึกไฟล์
        if save_to_file:
            folder_name = f"raw_ai_data_{self.symbol.replace('.', '_').replace('/', '_')}"
            self._save_raw_dataset(cleaned_data, folder_name)
        
        self.logger.info("✅ เตรียม Raw Dataset เสร็จสมบูรณ์")
        return cleaned_data
    
    def _clean_raw_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        ทำความสะอาดข้อมูล Raw
        """
        df_clean = df.copy()
        
        # ลบแถวที่มีค่า 0 หรือ NaN
        df_clean = df_clean.dropna()
        df_clean = df_clean[(df_clean[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]
        
        # ตรวจสอบความถูกต้องของแท่งเทียน
        # High >= max(Open, Close) และ Low <= min(Open, Close)
        valid_candles = (
            (df_clean['High'] >= df_clean[['Open', 'Close']].max(axis=1)) &
            (df_clean['Low'] <= df_clean[['Open', 'Close']].min(axis=1))
        )
        df_clean = df_clean[valid_candles]
        
        # ลบ outliers (ราคาเปลี่ยนแปลงมากกว่า 10% ในแท่งเดียว)
        price_change = abs(df_clean['Close'].pct_change())
        df_clean = df_clean[price_change < 0.1]
        
        self.logger.info(f"🧹 {timeframe}: ทำความสะอาด {len(df):,} → {len(df_clean):,} แท่ง")
        return df_clean
    
    def _save_raw_dataset(self, data: Dict[str, pd.DataFrame], folder: str):
        """บันทึกข้อมูล Raw Dataset"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # บันทึก CSV แต่ละไทม์เฟรม
        for tf, df in data.items():
            filename = f"{folder}/{self.symbol}_{tf}_raw.csv"
            df.to_csv(filename)
            self.logger.info(f"💾 บันทึก: {filename}")
        
        # สร้าง Dataset Summary
        summary = {
            'dataset_type': 'Raw Candlestick Data for AI Training',
            'symbol': self.symbol,
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Pure OHLCV data without feature engineering',
            'data_range': {
                'start': min([df.index.min() for df in data.values()]).strftime('%Y-%m-%d'),
                'end': max([df.index.max() for df in data.values()]).strftime('%Y-%m-%d')
            },
            'timeframes': {},
            'total_bars': sum([len(df) for df in data.values()]),
            'columns': list(data[list(data.keys())[0]].columns.tolist())
        }
        
        for tf, df in data.items():
            coverage_days = (df.index.max() - df.index.min()).days
            summary['timeframes'][tf] = {
                'bars': len(df),
                'columns': df.shape[1],
                'start': df.index.min().strftime('%Y-%m-%d %H:%M'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M'),
                'coverage_days': coverage_days,
                'file': f"{self.symbol}_{tf}_raw.csv"
            }
        
        # บันทึก Summary JSON
        import json
        with open(f'{folder}/raw_dataset_info.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # สร้าง README
        readme = f"""# Raw Candlestick Dataset - {self.symbol}

## 🎯 จุดประสงค์
ข้อมูลแท่งเทียนดิบสำหรับการเทรน AI Trading
**ไม่มี Feature Engineering** - ให้ AI เรียนรู้และหา Pattern เอง

## 📊 ข้อมูลที่มี
- **Symbol**: {self.symbol}
- **Total Bars**: {summary['total_bars']:,}
- **Date Range**: {summary['data_range']['start']} ถึง {summary['data_range']['end']}
- **Timeframes**: {', '.join(data.keys())}

## 📋 Columns
```
{', '.join(summary['columns'])}
```

## 🎯 Philosophy
> "ให้ AI หา Pattern เอง แทนที่จะบอกมันล่วงหน้า"

### ❌ ไม่มี:
- Technical Indicators (RSI, MACD, etc.)
- Chart Patterns (Head & Shoulders, etc.)
- Support/Resistance levels
- Trend analysis

### ✅ มีเฉพาะ:
- Pure OHLCV data
- Time context (Hour, Day, Session)
- Multiple timeframes

## 🧠 Next Steps
1. **Train Candlestick Recognition Model**
   - เรียนรู้ความหมายของแต่ละแท่งเทียน
   - เข้าใจ Market Psychology จากแท่งเทียน

2. **Multi-Timeframe Analysis**
   - รวมข้อมูลหลายไทม์เฟรม
   - หา Context จากไทม์เฟรมใหญ่

3. **Pattern Discovery**
   - ให้ AI หา Pattern ที่มนุษย์อาจมองไม่เห็น
   - เรียนรู้จากข้อมูลจริง ไม่ใช่ Theory

4. **Decision Making**
   - เทรน AI ให้ตัดสินใจ Buy/Sell/Hold
   - ใช้ Reinforcement Learning

Generated: {summary['creation_date']}
"""
        
        with open(f'{folder}/README.md', 'w') as f:
            f.write(readme)
        
        self.logger.info(f"📋 สร้าง Documentation เรียบร้อย: {folder}/")

# === Usage Example ===
if __name__ == "__main__":
    print("🎯 Raw Candlestick Data Collector for AI Training")
    print("=" * 50)
    
    # สร้าง collector
    collector = RawCandlestickCollector("XAUUSD.c")
    
    if not collector.symbol:
        print("❌ ไม่พบสัญลักษณ์ที่ใช้งานได้")
        exit()
    
    print(f"✅ ใช้สัญลักษณ์: {collector.symbol}")
    
    # เตรียมข้อมูล Raw
    print("\n🚀 เริ่มเตรียมข้อมูล Raw Dataset...")
    raw_dataset = collector.prepare_ai_raw_dataset()
    
    if raw_dataset:
        print(f"\n🎉 สำเร็จ! ได้ข้อมูล Raw {collector.symbol}")
        print("=" * 60)
        
        total_bars = 0
        for tf, df in raw_dataset.items():
            days = (df.index.max() - df.index.min()).days
            total_bars += len(df)
            
            print(f"📊 {tf:>3}: {len(df):>8,} แท่ง | {days:>4} วัน | {df.shape[1]} คอลัมน์")
            print(f"     📅 {df.index.min().strftime('%Y-%m-%d')} ถึง {df.index.max().strftime('%Y-%m-%d')}")
        
        print("=" * 60)
        print(f"📊 รวมทั้งหมด: {total_bars:,} แท่ง")
        print(f"📁 บันทึกใน: raw_ai_data_{collector.symbol.replace('.', '_')}/")
        
        # แสดง sample data
        sample_df = list(raw_dataset.values())[0]
        print(f"\n📋 Columns: {list(sample_df.columns)}")
        print(f"📈 Sample Data (latest 3 rows):")
        print(sample_df.tail(3)[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        print(f"\n✅ Raw Dataset พร้อมสำหรับ AI Training!")
        print(f"🧠 ขั้นตอนต่อไป: สอน AI ให้รู้จักแท่งเทียน")
        
    else:
        print("❌ ไม่สามารถเตรียมข้อมูลได้")