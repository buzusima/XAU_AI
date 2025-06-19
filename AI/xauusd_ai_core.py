# ================================
# XAUUSD AI TRADING SYSTEM - CORE FOUNDATION
# ส่วนที่ 1: Data Pipeline & MT5 Connection
# ================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import pickle
import json
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging (Windows compatible)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xauusd_ai.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XAUUSDDataManager:
    """
    หัวใจของระบบ - จัดการข้อมูลทั้งหมด
    ความสำคัญ: ถ้าข้อมูลผิด AI จะเรียนรู้ผิด!
    """
    
    def __init__(self, symbol=None, db_path="xauusd_data.db"):
        # Auto-detect symbol ถ้าไม่ระบุ
        self.symbol = symbol
        self.detected_symbol = None
        self.db_path = db_path
        self.is_connected = False
        self.is_streaming = False
        self.tick_buffer = []
        self.buffer_size = 1000
        
        # ตัวแปรสำคัญสำหรับ Real-time
        self.current_tick = None
        self.current_candle = None
        self.last_candle_time = None
        
        self._setup_database()
        self._connect_mt5()
    
    def _setup_database(self):
        """สร้าง Database สำหรับเก็บข้อมูล"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ตาราง Tick Data (สำหรับ Scalping)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TIMESTAMP,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    spread REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง OHLC Data (M1, M5)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT,
                    time TIMESTAMP,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง AI Decisions (สำหรับเรียนรู้)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_time TIMESTAMP,
                    signal_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    confidence_score REAL,
                    features TEXT,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def _connect_mt5(self):
        """เชื่อมต่อ MT5 และค้นหา Symbol อัตโนมัติ"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            # แสดงข้อมูล MT5
            logger.info(f"MT5 version: {mt5.version()}")
            logger.info(f"MT5 terminal info: {mt5.terminal_info()}")
            
            # ถ้าไม่ระบุ symbol ให้ auto-detect
            if self.symbol is None:
                self.symbol = self._detect_xau_symbol()
            
            if self.symbol is None:
                logger.error("Cannot find any GOLD symbol in broker")
                return False
            
            # ตรวจสอบ Symbol
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                # ลองหา symbol อื่น
                alternative = self._detect_xau_symbol()
                if alternative and alternative != self.symbol:
                    self.symbol = alternative
                    symbol_info = mt5.symbol_info(self.symbol)
                
                if symbol_info is None:
                    logger.error("No valid GOLD symbol found")
                    return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    return False
            
            self.detected_symbol = self.symbol
            self.is_connected = True
            logger.info(f"MT5 connected successfully - Using symbol: {self.symbol}")
            logger.info(f"Spread: {symbol_info.spread} points")
            logger.info(f"Digits: {symbol_info.digits}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            return False
    
    def _detect_xau_symbol(self):
        """ค้นหา Symbol ทองคำอัตโนมัติ"""
        try:
            # รายชื่อ Symbol ทองคำที่เป็นไปได้
            possible_symbols = [
                "XAUUSD", "XAUUSDm", "XAUUSD.m", "XAUUSD-", "XAUUSD.a",
                "GOLD", "GOLDm", "GOLD.m", "GOLD-", "GOLD.a",
                "Gold", "Goldm", "Gold.m", "Gold-", "Gold.a",
                "XAUUSD.c", "XAUUSDc", "XAUUSD_", "XAU-USD",
                "#XAUUSD", "XAU/USD", "XAUUSD#", "XAUUSDf",
                "XAUUSD.", "XAUUSDi", "XAUUSDe", "GOLD#"
            ]
            
            # ดึงรายชื่อ symbols ทั้งหมด
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.error("Cannot get symbols list from MT5")
                return None
            
            available_symbols = [s.name for s in symbols]
            logger.info(f"Found {len(available_symbols)} symbols in broker")
            
            # แสดง symbols ตัวอย่าง
            sample_symbols = available_symbols[:10]
            logger.info(f"Sample symbols: {sample_symbols}")
            
            # หา symbol ที่ตรงกับรายการ
            for symbol in possible_symbols:
                if symbol in available_symbols:
                    logger.info(f"Found GOLD symbol: {symbol}")
                    return symbol
            
            # ถ้าไม่เจอ ลองหาจากคำที่มี XAU หรือ GOLD
            gold_symbols = []
            for symbol_name in available_symbols:
                upper_name = symbol_name.upper()
                if "XAU" in upper_name or "GOLD" in upper_name:
                    gold_symbols.append(symbol_name)
            
            if gold_symbols:
                logger.info(f"Found potential GOLD symbols: {gold_symbols}")
                return gold_symbols[0]  # ใช้ตัวแรกที่เจอ
            
            # ถ้ายังไม่เจอ แสดงรายการ symbols ทั้งหมดเพื่อช่วยดู
            logger.info(f"All available symbols: {available_symbols}")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting symbol: {e}")
            return None
    
    def get_historical_data(self, timeframe=mt5.TIMEFRAME_M1, count=10000):
        """
        ดึงข้อมูลย้อนหลังสำหรับฝึก AI
        นี่คือข้อมูลที่ AI จะเรียนรู้!
        """
        if not self.is_connected:
            logger.error("MT5 not connected")
            return None
        
        try:
            # ดึงข้อมูล OHLC - ใช้ copy_rates_from_pos
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            if rates is None:
                logger.error("No historical data received")
                logger.error(f"MT5 Error: {mt5.last_error()}")
                return None
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # คำนวณ Features สำคัญสำหรับ AI
            df = self._calculate_features(df)
            
            # บันทึกลง Database
            self._save_to_database(df, 'M1' if timeframe == mt5.TIMEFRAME_M1 else 'M5')
            
            logger.info(f"Historical data loaded: {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def _calculate_features(self, df):
        """
        คำนวณ Features สำหรับ AI - ส่วนนี้สำคัญมาก!
        Features ที่ดี = AI ที่ฉลาด
        """
        try:
            # Basic Price Features
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            # Price Action Features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Trend Features
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
            
            # Volatility Features
            df['atr_14'] = self._calculate_atr(df, 14)
            # ป้องกัน division by zero
            df['volatility_ratio'] = np.where(
                df['atr_14'] > 0, 
                df['total_range'] / df['atr_14'], 
                0
            )
            
            # Volume Features (ถ้ามี)
            if 'tick_volume' in df.columns:
                df['volume_sma_10'] = df['tick_volume'].rolling(10).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma_10']
            
            # Momentum Features
            df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Support/Resistance Levels
            df['local_high'] = df['high'].rolling(5, center=True).max() == df['high']
            df['local_low'] = df['low'].rolling(5, center=True).min() == df['low']
            
            # Time-based Features (สำคัญสำหรับ XAUUSD)
            df['hour'] = df.index.hour
            df['is_london_session'] = (df['hour'] >= 8) & (df['hour'] <= 16)
            df['is_ny_session'] = (df['hour'] >= 13) & (df['hour'] <= 21)
            df['is_overlap'] = (df['hour'] >= 13) & (df['hour'] <= 16)
            
            logger.info("Features calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df
    
    def _calculate_atr(self, df, period=14):
        """คำนวณ Average True Range"""
        try:
            if len(df) < period + 1:
                return pd.Series(index=df.index, dtype=float)
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            # สร้าง DataFrame และหา max
            ranges_df = pd.DataFrame({
                'hl': high_low,
                'hc': high_close,
                'lc': low_close
            }, index=df.index)
            
            true_ranges = ranges_df.max(axis=1)
            atr = true_ranges.rolling(window=period, min_periods=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def start_real_time_stream(self):
        """
        เริ่ม Real-time Data Stream
        นี่คือหัวใจของ Scalping System!
        """
        if not self.is_connected:
            logger.error("Cannot start stream - MT5 not connected")
            return
        
        self.is_streaming = True
        stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        stream_thread.start()
        logger.info("Real-time streaming started")
    
    def _stream_worker(self):
        """Worker สำหรับ Real-time streaming"""
        while self.is_streaming:
            try:
                # ดึง Tick ล่าสุด
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is not None:
                    # คำนวณ spread ที่ถูกต้องตาม digits ของ symbol
                    symbol_info = mt5.symbol_info(self.symbol)
                    if symbol_info:
                        # สำหรับ XAUUSD.c ที่มี digits=2, spread จะเป็น points แล้ว
                        if symbol_info.digits == 2:
                            spread_value = (tick.ask - tick.bid) * 100  # แปลงเป็น points
                        else:
                            spread_value = (tick.ask - tick.bid) * 100000  # สำหรับ 5 digits
                    else:
                        spread_value = (tick.ask - tick.bid) * 100
                    
                    self.current_tick = {
                        'time': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'volume': tick.volume,
                        'spread': spread_value
                    }
                    
                    # เพิ่มเข้า Buffer
                    self.tick_buffer.append(self.current_tick)
                    
                    # ล้าง Buffer ถ้าเต็ม
                    if len(self.tick_buffer) > self.buffer_size:
                        self.tick_buffer.pop(0)
                    
                    # สร้าง Candle แบบ Real-time
                    self._update_current_candle()
                
                time.sleep(0.1)  # Update ทุก 100ms
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                time.sleep(1)
    
    def _update_current_candle(self):
        """อัพเดท Candle ปัจจุบันแบบ Real-time"""
        if not self.current_tick:
            return
        
        current_minute = self.current_tick['time'].replace(second=0, microsecond=0)
        
        # ถ้าเป็นนาทีใหม่ บันทึก Candle เก่า
        if self.last_candle_time != current_minute:
            if self.current_candle:
                self._save_completed_candle()
            
            # เริ่ม Candle ใหม่
            self.current_candle = {
                'time': current_minute,
                'open': (self.current_tick['bid'] + self.current_tick['ask']) / 2,
                'high': (self.current_tick['bid'] + self.current_tick['ask']) / 2,
                'low': (self.current_tick['bid'] + self.current_tick['ask']) / 2,
                'close': (self.current_tick['bid'] + self.current_tick['ask']) / 2,
                'volume': 1
            }
            self.last_candle_time = current_minute
        else:
            # อัพเดท Candle ปัจจุบัน
            if self.current_candle:
                mid_price = (self.current_tick['bid'] + self.current_tick['ask']) / 2
                self.current_candle['high'] = max(self.current_candle['high'], mid_price)
                self.current_candle['low'] = min(self.current_candle['low'], mid_price)
                self.current_candle['close'] = mid_price
                self.current_candle['volume'] += 1
    
    def _save_completed_candle(self):
        """บันทึก Candle ที่เสร็จแล้วลง Database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ohlc_data (timeframe, time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'M1_LIVE',
                self.current_candle['time'],
                self.current_candle['open'],
                self.current_candle['high'],
                self.current_candle['low'],
                self.current_candle['close'],
                self.current_candle['volume']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving candle: {e}")
    
    def _save_to_database(self, df, timeframe):
        """บันทึกข้อมูลลง Database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for index, row in df.iterrows():
                # แปลง timestamp เป็น string เพื่อป้องกัน binding error
                time_str = index.strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlc_data 
                    (timeframe, time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timeframe,
                    time_str,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row.get('tick_volume', 0))
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(df)} records to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            # ไม่ให้ error นี้หยุดระบบ - ข้อมูลยังใช้ได้
    
    def get_latest_features(self, lookback=100):
        """
        ดึง Features ล่าสุดสำหรับ AI ตัดสินใจ
        นี่คือข้อมูลที่ AI จะใช้ตัดสินใจ Entry/Exit!
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT DISTINCT timeframe, time, open, high, low, close, volume 
                FROM ohlc_data 
                WHERE timeframe = 'M1' OR timeframe = 'M1_LIVE'
                ORDER BY time DESC 
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=[lookback])
            conn.close()
            
            if len(df) < 20:  # ข้อมูลไม่พอ
                return None
            
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            # ลบ duplicate index
            df = df[~df.index.duplicated(keep='last')]
            
            # คำนวณ Features
            df = self._calculate_features(df)
            
            # ส่งกลับ Features ล่าสุด - ใช้ค่าล่าสุดที่มีข้อมูลครบ
            latest_idx = -1
            for i in range(-1, -min(5, len(df)), -1):
                if not pd.isna(df['atr_14'].iloc[i]):
                    latest_idx = i
                    break
            
            latest_features = {
                'price': df['close'].iloc[latest_idx],
                'atr': df['atr_14'].iloc[latest_idx] if not pd.isna(df['atr_14'].iloc[latest_idx]) else 1.0,
                'trend_5': df['price_vs_sma_5'].iloc[latest_idx] if not pd.isna(df['price_vs_sma_5'].iloc[latest_idx]) else 0.0,
                'trend_20': df['price_vs_sma_20'].iloc[latest_idx] if not pd.isna(df['price_vs_sma_20'].iloc[latest_idx]) else 0.0,
                'volatility': df['volatility_ratio'].iloc[latest_idx] if not pd.isna(df['volatility_ratio'].iloc[latest_idx]) else 1.0,
                'momentum_5': df['roc_5'].iloc[latest_idx] if not pd.isna(df['roc_5'].iloc[latest_idx]) else 0.0,
                'is_london': df['is_london_session'].iloc[latest_idx],
                'is_ny': df['is_ny_session'].iloc[latest_idx],
                'spread': self.current_tick['spread'] if self.current_tick else 0,
                'bid': self.current_tick['bid'] if self.current_tick else 0,
                'ask': self.current_tick['ask'] if self.current_tick else 0
            }
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {e}")
            return None
    
    def cleanup(self):
        """ปิดการเชื่อมต่อ"""
        self.is_streaming = False
        if self.is_connected:
            mt5.shutdown()
        logger.info("System cleanup completed")

# ================================
# USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    # เริ่มระบบ - จะ auto-detect symbol
    print("Initializing XAUUSD AI Trading System...")
    data_manager = XAUUSDDataManager()  # ไม่ระบุ symbol ให้ auto-detect
    
    # แสดง symbol ที่เจอ
    if data_manager.is_connected:
        print(f"Connected successfully using symbol: {data_manager.detected_symbol}")
        
        # ดึงข้อมูลย้อนหลัง
        print("Loading historical data...")
        historical_data = data_manager.get_historical_data(count=5000)
        
        if historical_data is not None:
            print(f"Loaded {len(historical_data)} historical candles")
            print("\nSample data:")
            print(historical_data[['open', 'high', 'low', 'close', 'atr_14']].tail())
            
            # เริ่ม Real-time streaming
            print("\nStarting real-time stream...")
            data_manager.start_real_time_stream()
            
            # ทดสอบ Features
            try:
                while True:
                    features = data_manager.get_latest_features()
                    if features:
                        print(f"\nLatest Features:")
                        for key, value in features.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.5f}")
                            else:
                                print(f"  {key}: {value}")
                    
                    time.sleep(5)  # แสดงทุก 5 วินาที
                    
            except KeyboardInterrupt:
                print("\nStopping system...")
                data_manager.cleanup()
        else:
            print("Failed to load historical data")
    else:
        print("Failed to connect to MT5 or find GOLD symbol")
        print("Please make sure:")
        print("1. MT5 is running and logged in")
        print("2. GOLD symbol is available in Market Watch")
        print("3. Internet connection is stable")