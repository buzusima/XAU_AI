import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Chart visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import os

# Set style for better charts
plt.style.use('dark_background')
sns.set_palette("husl")

class SMCMarketStructure:
    """
    Smart Money Concepts Market Structure Analysis Engine
    เฉพาะสำหรับ XAUUSD ตามสไตล์การเทรดแบบ Multi-timeframe
    """
    
    def __init__(self):
        self.symbol = "XAUUSD.c"
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1, 
            'M30': mt5.TIMEFRAME_M30,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        
        # Configuration ตามสไตล์ของคุณ
        self.config = {
            'bos_method': 'body_close',  # ใช้ body close ไม่นับไส้เทียน
            'confirmation_candles': 1,   # ยืนยัน 1 เทียน
            'premium_discount_level': 50,  # Fibo 50%
            'min_displacement': 20,      # pips สำหรับ valid structure
        }
        
        # เก็บข้อมูลของแต่ละไทม์เฟรม
        self.market_data = {}
        self.structure_data = {}
        
    def connect_mt5(self) -> bool:
        """เชื่อมต่อ MetaTrader 5"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # ตรวจสอบ symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found")
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"Failed to select {self.symbol}")
                return False
                
        print(f"✅ Connected to MT5 - {self.symbol} ready")
        return True
    
    def get_market_data(self, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """ดึงข้อมูลตลาดตามไทม์เฟรม"""
        tf = self.timeframes.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
        if rates is None:
            raise Exception(f"Failed to get rates for {timeframe}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # เพิ่มคอลัมน์ที่จำเป็น
        df['body_high'] = np.maximum(df['open'], df['close'])
        df['body_low'] = np.minimum(df['open'], df['close'])
        df['is_bullish'] = df['close'] > df['open']
        df['body_size'] = abs(df['close'] - df['open'])
        
        return df
    
    def identify_swing_points(self, df: pd.DataFrame, timeframe: str = 'H4') -> pd.DataFrame:
        """
        ระบุ Swing High และ Swing Low แบบ Smart Money Concepts
        ใช้ dynamic lookback และ significance filtering
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        # Dynamic lookback ตาม timeframe
        lookback_config = {
            'H4': 5, 'H1': 4, 'M30': 3, 'M15': 3, 'M5': 2, 'M1': 2
        }
        lookback = lookback_config.get(timeframe, 3)
        
        # คำนวณ ATR สำหรับ significance filter
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Minimum move size สำหรับ significant swing
        min_move_multiplier = {
            'H4': 2.0, 'H1': 1.5, 'M30': 1.2, 'M15': 1.0, 'M5': 0.8, 'M1': 0.5
        }
        min_move = min_move_multiplier.get(timeframe, 1.0)
        
        for i in range(lookback, len(df) - lookback):
            current_atr = df.iloc[i]['atr']
            if pd.isna(current_atr):
                continue
            
            # Swing High Detection
            if df.iloc[i]['body_high'] == df.iloc[i-lookback:i+lookback+1]['body_high'].max():
                # ตรวจสอบ significance - ต้องมี move ขึ้นอย่างน้อย ATR * multiplier
                left_low = df.iloc[i-lookback:i+1]['body_low'].min()
                move_size = df.iloc[i]['body_high'] - left_low
                
                if move_size >= current_atr * min_move:
                    df.iloc[i, df.columns.get_loc('swing_high')] = True
                    df.iloc[i, df.columns.get_loc('swing_high_price')] = df.iloc[i]['body_high']
            
            # Swing Low Detection  
            if df.iloc[i]['body_low'] == df.iloc[i-lookback:i+lookback+1]['body_low'].min():
                # ตรวจสอบ significance - ต้องมี move ลงอย่างน้อย ATR * multiplier
                left_high = df.iloc[i-lookback:i+1]['body_high'].max()
                move_size = left_high - df.iloc[i]['body_low']
                
                if move_size >= current_atr * min_move:
                    df.iloc[i, df.columns.get_loc('swing_low')] = True
                    df.iloc[i, df.columns.get_loc('swing_low_price')] = df.iloc[i]['body_low']
        
        return df
    
    def detect_bos_choch(self, df: pd.DataFrame, timeframe: str = 'H4') -> pd.DataFrame:
        """
        ตรวจจับ Break of Structure (BOS) และ Change of Character (CHoCH)
        ด้วย Smart Money Concepts logic ที่ปรับปรุงแล้ว
        """
        df = df.copy()
        df['structure_type'] = 'RANGING'
        df['bos'] = False
        df['choch'] = False
        df['structure_break_type'] = ''
        df['structure_strength'] = 0  # 0=neutral, 1-3=strength level
        
        # หา swing points ก่อน (ใช้ timeframe-aware method)
        df = self.identify_swing_points(df, timeframe)
        
        swing_highs = df[df['swing_high']].copy()
        swing_lows = df[df['swing_low']].copy()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return df
        
        # สร้าง structure points ที่ significant
        structure_points = []
        
        # รวม swing points และเรียงตามเวลา
        for idx, row in swing_highs.iterrows():
            structure_points.append({
                'time': idx,
                'price': row['swing_high_price'],
                'type': 'high'
            })
        
        for idx, row in swing_lows.iterrows():
            structure_points.append({
                'time': idx,
                'price': row['swing_low_price'],
                'type': 'low'
            })
        
        structure_points = sorted(structure_points, key=lambda x: x['time'])
        
        if len(structure_points) < 4:  # ต้องมีอย่างน้อย 4 points
            return df
        
        # วิเคราะห์ trend structure
        current_trend = self._analyze_trend_structure(structure_points)
        last_structure_high = None
        last_structure_low = None
        
        # หา last significant highs/lows
        for point in reversed(structure_points):
            if point['type'] == 'high' and last_structure_high is None:
                last_structure_high = point
            if point['type'] == 'low' and last_structure_low is None:
                last_structure_low = point
            if last_structure_high and last_structure_low:
                break
        
        # วิเคราะห์ structure breaks
        for i in range(len(df)):
            current_time = df.index[i]
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # อัพเดท trend classification
            df.iloc[i, df.columns.get_loc('structure_type')] = current_trend
            
            if last_structure_high is None or last_structure_low is None:
                continue
            
            # Calculate displacement strength
            body_size = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
            atr = df.iloc[i]['atr'] if 'atr' in df.columns else 20
            displacement_strength = min(3, int(body_size / (atr * 0.5))) if atr > 0 else 1
            
            # BOS/CHoCH Detection Logic
            if current_trend == 'UPTREND':
                # BOS Bullish - strong break above last high
                if current_close > last_structure_high['price']:
                    # ต้องมี strong displacement
                    if displacement_strength >= 1:
                        df.iloc[i, df.columns.get_loc('bos')] = True
                        df.iloc[i, df.columns.get_loc('structure_break_type')] = 'BOS_BULLISH'
                        df.iloc[i, df.columns.get_loc('structure_strength')] = displacement_strength
                
                # CHoCH - strong break below last low
                elif current_close < last_structure_low['price']:
                    if displacement_strength >= 1:
                        df.iloc[i, df.columns.get_loc('choch')] = True
                        df.iloc[i, df.columns.get_loc('structure_break_type')] = 'CHOCH_BEARISH'
                        df.iloc[i, df.columns.get_loc('structure_strength')] = displacement_strength
                        current_trend = 'DOWNTREND'
                        
            elif current_trend == 'DOWNTREND':
                # BOS Bearish - strong break below last low
                if current_close < last_structure_low['price']:
                    if displacement_strength >= 1:
                        df.iloc[i, df.columns.get_loc('bos')] = True
                        df.iloc[i, df.columns.get_loc('structure_break_type')] = 'BOS_BEARISH'
                        df.iloc[i, df.columns.get_loc('structure_strength')] = displacement_strength
                
                # CHoCH - strong break above last high
                elif current_close > last_structure_high['price']:
                    if displacement_strength >= 1:
                        df.iloc[i, df.columns.get_loc('choch')] = True
                        df.iloc[i, df.columns.get_loc('structure_break_type')] = 'CHOCH_BULLISH'
                        df.iloc[i, df.columns.get_loc('structure_strength')] = displacement_strength
                        current_trend = 'UPTREND'
        
        return df
    
    def _analyze_trend_structure(self, structure_points: List[Dict]) -> str:
        """
        วิเคราะห์ trend structure จาก significant swing points
        ตาม Smart Money Concepts
        """
        if len(structure_points) < 4:
            return 'RANGING'
        
        # เอาแค่ 6 points ล่าสุดสำหรับ analysis
        recent_points = structure_points[-6:]
        
        highs = [p for p in recent_points if p['type'] == 'high']
        lows = [p for p in recent_points if p['type'] == 'low']
        
        if len(highs) < 2 or len(lows) < 2:
            return 'RANGING'
        
        # เรียงตามเวลา
        highs = sorted(highs, key=lambda x: x['time'])
        lows = sorted(lows, key=lambda x: x['time'])
        
        # เปรียบเทียบ recent highs/lows
        if len(highs) >= 2 and len(lows) >= 2:
            latest_high = highs[-1]['price']
            prev_high = highs[-2]['price']
            latest_low = lows[-1]['price']
            prev_low = lows[-2]['price']
            
            # Higher Highs และ Higher Lows = Uptrend
            if latest_high > prev_high and latest_low > prev_low:
                return 'UPTREND'
            # Lower Highs และ Lower Lows = Downtrend
            elif latest_high < prev_high and latest_low < prev_low:
                return 'DOWNTREND'
        
        return 'RANGING'
    
    def detect_order_blocks(self, df: pd.DataFrame, timeframe: str = 'M15') -> pd.DataFrame:
        """
        ตรวจจับ Order Blocks ตามหลัก Smart Money Concepts
        """
        df = df.copy()
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        df['ob_top'] = np.nan
        df['ob_bottom'] = np.nan
        df['ob_strength'] = 0
        
        # หา displacement moves (strong impulse moves)
        df['displacement'] = False
        atr_period = 14
        displacement_threshold = {
            'H4': 3.0, 'H1': 2.5, 'M30': 2.0, 'M15': 1.5, 'M5': 1.2, 'M1': 1.0
        }
        threshold = displacement_threshold.get(timeframe, 1.5)
        
        for i in range(atr_period, len(df)):
            if 'atr' not in df.columns:
                continue
                
            current_atr = df.iloc[i]['atr']
            if pd.isna(current_atr):
                continue
                
            # ตรวจสอบ displacement (strong move)
            body_size = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
            if body_size >= current_atr * threshold:
                df.iloc[i, df.columns.get_loc('displacement')] = True
        
        # ตรวจหา Order Blocks
        displacement_candles = df[df['displacement']].index
        
        for disp_idx in displacement_candles:
            disp_pos = df.index.get_loc(disp_idx)
            
            # Skip ถ้าอยู่ใกล้ขอบ
            if disp_pos < 5 or disp_pos >= len(df) - 2:
                continue
                
            displacement_candle = df.iloc[disp_pos]
            is_bullish_displacement = displacement_candle['close'] > displacement_candle['open']
            
            if is_bullish_displacement:
                # หา Bullish Order Block (last bearish candle before bullish displacement)
                for lookback in range(1, 6):  # มองย้อนหลัง 5 candles
                    if disp_pos - lookback < 0:
                        break
                        
                    potential_ob = df.iloc[disp_pos - lookback]
                    
                    # ต้องเป็น bearish candle
                    if potential_ob['close'] < potential_ob['open']:
                        # ตรวจสอบว่าไม่มี candle ขงไป overlap กับ OB
                        ob_top = potential_ob['body_high']
                        ob_bottom = potential_ob['body_low']
                        
                        # Strength based on displacement และ position
                        strength = min(3, int(displacement_candle['body_size'] / displacement_candle['atr'])) if displacement_candle['atr'] > 0 else 1
                        
                        df.iloc[disp_pos - lookback, df.columns.get_loc('bullish_ob')] = True
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_top')] = ob_top
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_bottom')] = ob_bottom
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_strength')] = strength
                        break
                        
            else:
                # หา Bearish Order Block (last bullish candle before bearish displacement)
                for lookback in range(1, 6):
                    if disp_pos - lookback < 0:
                        break
                        
                    potential_ob = df.iloc[disp_pos - lookback]
                    
                    # ต้องเป็น bullish candle
                    if potential_ob['close'] > potential_ob['open']:
                        ob_top = potential_ob['body_high']
                        ob_bottom = potential_ob['body_low']
                        
                        strength = min(3, int(displacement_candle['body_size'] / displacement_candle['atr'])) if displacement_candle['atr'] > 0 else 1
                        
                        df.iloc[disp_pos - lookback, df.columns.get_loc('bearish_ob')] = True
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_top')] = ob_top
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_bottom')] = ob_bottom
                        df.iloc[disp_pos - lookback, df.columns.get_loc('ob_strength')] = strength
                        break
        
        return df
    
    def detect_fair_value_gaps(self, df: pd.DataFrame, timeframe: str = 'M15') -> pd.DataFrame:
        """
        ตรวจจับ Fair Value Gaps (FVG)
        """
        df = df.copy()
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan
        
        # Minimum gap size ตาม timeframe
        min_gap_size = {
            'H4': 20, 'H1': 15, 'M30': 12, 'M15': 8, 'M5': 5, 'M1': 3
        }
        min_size = min_gap_size.get(timeframe, 8)
        
        for i in range(2, len(df)):
            # FVG = gap ระหว่าง candle[i-2] และ candle[i] ที่ไม่ได้ fill โดย candle[i-1]
            
            candle_1 = df.iloc[i-2]  # First candle
            candle_2 = df.iloc[i-1]  # Middle candle  
            candle_3 = df.iloc[i]    # Third candle
            
            # Bullish FVG
            # Gap between candle_1 high และ candle_3 low, ไม่ถูก fill โดย candle_2
            if candle_1['high'] < candle_3['low']:
                gap_size = candle_3['low'] - candle_1['high']
                
                # ตรวจสอบว่า candle_2 ไม่ fill gap
                if (candle_2['low'] > candle_1['high'] and 
                    candle_2['high'] < candle_3['low'] and 
                    gap_size >= min_size):
                    
                    df.iloc[i, df.columns.get_loc('fvg_bullish')] = True
                    df.iloc[i, df.columns.get_loc('fvg_top')] = candle_3['low']
                    df.iloc[i, df.columns.get_loc('fvg_bottom')] = candle_1['high']
            
            # Bearish FVG
            # Gap between candle_1 low และ candle_3 high, ไม่ถูก fill โดย candle_2
            elif candle_1['low'] > candle_3['high']:
                gap_size = candle_1['low'] - candle_3['high']
                
                if (candle_2['high'] < candle_1['low'] and 
                    candle_2['low'] > candle_3['high'] and 
                    gap_size >= min_size):
                    
                    df.iloc[i, df.columns.get_loc('fvg_bearish')] = True
                    df.iloc[i, df.columns.get_loc('fvg_top')] = candle_1['low']
                    df.iloc[i, df.columns.get_loc('fvg_bottom')] = candle_3['high']
        
        return df
        """กำหนดเทรนด์เริ่มต้นจาก swing points"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'RANGING'
        
        recent_highs = swing_highs.tail(2)['swing_high_price'].values
        recent_lows = swing_lows.tail(2)['swing_low_price'].values
        
        # Higher Highs และ Higher Lows = Uptrend
        if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
            return 'UPTREND'
        # Lower Highs และ Lower Lows = Downtrend  
        elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
            return 'DOWNTREND'
        else:
            return 'RANGING'
    
    def calculate_premium_discount(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        คำนวณ Premium/Discount zones ด้วย Fibonacci 50%
        ตามสไตล์ของคุณ
        """
        df = df.copy()
        
        # คำนวณ range ของ period ที่กำหนด
        df['period_high'] = df['high'].rolling(window=period).max()
        df['period_low'] = df['low'].rolling(window=period).min()
        df['range_size'] = df['period_high'] - df['period_low']
        
        # Fibonacci 50% level
        df['fib_50'] = df['period_low'] + (df['range_size'] * 0.5)
        
        # Premium/Discount classification
        df['premium_discount'] = np.where(
            df['close'] > df['fib_50'], 
            'PREMIUM', 
            'DISCOUNT'
        )
        
        # Distance from 50% level (for zone analysis)
        df['distance_from_50'] = abs(df['close'] - df['fib_50'])
        df['distance_pct'] = (df['distance_from_50'] / df['range_size']) * 100
        
        return df
    
    def analyze_multi_timeframe(self) -> Dict:
        """
        วิเคราะห์หลายไทม์เฟรมตามสไตล์ของคุณ
        H4→H1→M30→M15→M5→M1
        """
        analysis_result = {}
        
        print("🔄 Starting Multi-Timeframe Analysis...")
        
        for tf_name in ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']:
            print(f"📊 Analyzing {tf_name}...")
            
            try:
                # ดึงข้อมูล
                bars = 500 if tf_name in ['H4', 'H1'] else 1000
                df = self.get_market_data(tf_name, bars)
                
                # วิเคราะห์โครงสร้าง
                df = self.detect_bos_choch(df, tf_name)
                df = self.calculate_premium_discount(df)
                df = self.detect_order_blocks(df, tf_name)
                df = self.detect_fair_value_gaps(df, tf_name)
                
                # บันทึกข้อมูล
                self.market_data[tf_name] = df
                
                # สรุปผลการวิเคราะห์
                latest = df.iloc[-1]
                analysis_result[tf_name] = {
                    'current_price': latest['close'],
                    'structure_type': latest['structure_type'],
                    'premium_discount': latest['premium_discount'],
                    'recent_bos': df['bos'].tail(10).any(),
                    'recent_choch': df['choch'].tail(10).any(),
                    'last_structure_break': latest['structure_break_type'] if latest['structure_break_type'] else 'NONE'
                }
                
                print(f"   ✅ {tf_name}: {latest['structure_type']} | {latest['premium_discount']}")
                
            except Exception as e:
                print(f"   ❌ {tf_name}: Error - {str(e)}")
                analysis_result[tf_name] = {'error': str(e)}
        
        return analysis_result
    
    def get_trading_bias(self, analysis: Dict) -> Dict:
        """
        กำหนด Trading Bias ตามหลัก SMC Multi-timeframe ของคุณ
        """
        bias_result = {
            'h4_bias': 'NEUTRAL',
            'h1_bias': 'NEUTRAL', 
            'counter_trend_opportunity': False,
            'current_phase': 'ANALYSIS',
            'recommended_action': 'WAIT'
        }
        
        try:
            # H4 Bias (เทรนด์หลัก)
            if 'H4' in analysis and 'structure_type' in analysis['H4']:
                bias_result['h4_bias'] = analysis['H4']['structure_type']
            
            # H1 Bias (เทรนด์ย่อย)  
            if 'H1' in analysis and 'structure_type' in analysis['H1']:
                bias_result['h1_bias'] = analysis['H1']['structure_type']
            
            # Counter-trend opportunity (M30 analysis)
            if 'M30' in analysis and 'structure_type' in analysis['M30']:
                h4_trend = bias_result['h4_bias']
                m30_trend = analysis['M30']['structure_type']
                
                # ถ้า H4 uptrend แต่ M30 มี bearish structure = counter opportunity
                if h4_trend == 'UPTREND' and m30_trend == 'DOWNTREND':
                    bias_result['counter_trend_opportunity'] = True
                elif h4_trend == 'DOWNTREND' and m30_trend == 'UPTREND':
                    bias_result['counter_trend_opportunity'] = True
            
            # กำหนด Phase ปัจจุบัน
            if bias_result['counter_trend_opportunity']:
                bias_result['current_phase'] = 'COUNTER_TREND_SETUP'
                bias_result['recommended_action'] = 'MONITOR_M15_M5'
            elif bias_result['h4_bias'] != 'RANGING':
                bias_result['current_phase'] = 'TREND_FOLLOWING'
                bias_result['recommended_action'] = 'MONITOR_PULLBACKS'
            
        except Exception as e:
            bias_result['error'] = str(e)
        
        return bias_result
    
    def save_chart_analysis(self, timeframe: str, save_dir: str = "smc_charts") -> str:
        """
        สร้างและบันทึกกราฟวิเคราะห์ SMC พร้อม annotations
        """
        if timeframe not in self.market_data:
            raise ValueError(f"No data available for {timeframe}")
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.market_data[timeframe].tail(200)  # แสดง 200 candles ล่าสุด
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # === Main Price Chart ===
        # Candlestick chart
        for i, (idx, row) in enumerate(df.iterrows()):
            color = '#00ff88' if row['is_bullish'] else '#ff4444'
            
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            ax1.add_patch(Rectangle((i, body_bottom), 0.8, body_height, 
                                  facecolor=color, alpha=0.8))
            
            # Wicks
            ax1.plot([i+0.4, i+0.4], [row['low'], row['high']], 
                    color=color, linewidth=1, alpha=0.6)
        
        # === SMC Annotations ===
        
        # 1. Swing Points (reduced และ significant only)
        swing_highs = df[df['swing_high'] == True]
        swing_lows = df[df['swing_low'] == True]
        
        # แสดงเฉพาะ swing points ที่ significant
        for idx, row in swing_highs.iterrows():
            pos = df.index.get_loc(idx)
            ax1.scatter(pos, row['swing_high_price'], color='#FFD700', 
                       s=120, marker='^', edgecolors='white', linewidths=1,
                       label='Swing High' if pos == df.index.get_loc(swing_highs.index[0]) else "")
        
        for idx, row in swing_lows.iterrows():
            pos = df.index.get_loc(idx)
            ax1.scatter(pos, row['swing_low_price'], color='#00FFFF', 
                       s=120, marker='v', edgecolors='white', linewidths=1,
                       label='Swing Low' if pos == df.index.get_loc(swing_lows.index[0]) else "")
        
        # 2. Order Blocks
        bullish_obs = df[df['bullish_ob'] == True]
        bearish_obs = df[df['bearish_ob'] == True]
        
        for idx, row in bullish_obs.iterrows():
            pos = df.index.get_loc(idx)
            if not pd.isna(row['ob_top']) and not pd.isna(row['ob_bottom']):
                # Draw OB rectangle
                ob_height = row['ob_top'] - row['ob_bottom']
                rect = Rectangle((pos-0.4, row['ob_bottom']), 3, ob_height,
                               facecolor='green', alpha=0.3, edgecolor='green', linewidth=2)
                ax1.add_patch(rect)
                
                # Label with strength
                ax1.text(pos, row['ob_top'] + 5, f'Bull OB\n{int(row["ob_strength"])}★',
                        ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
        
        for idx, row in bearish_obs.iterrows():
            pos = df.index.get_loc(idx)
            if not pd.isna(row['ob_top']) and not pd.isna(row['ob_bottom']):
                ob_height = row['ob_top'] - row['ob_bottom']
                rect = Rectangle((pos-0.4, row['ob_bottom']), 3, ob_height,
                               facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)
                
                ax1.text(pos, row['ob_bottom'] - 5, f'Bear OB\n{int(row["ob_strength"])}★',
                        ha='center', va='top', fontsize=8, color='red', fontweight='bold')
        
        # 3. Fair Value Gaps
        fvg_bullish = df[df['fvg_bullish'] == True]
        fvg_bearish = df[df['fvg_bearish'] == True]
        
        for idx, row in fvg_bullish.iterrows():
            pos = df.index.get_loc(idx)
            if not pd.isna(row['fvg_top']) and not pd.isna(row['fvg_bottom']):
                fvg_height = row['fvg_top'] - row['fvg_bottom']
                rect = Rectangle((pos-0.2, row['fvg_bottom']), 1, fvg_height,
                               facecolor='cyan', alpha=0.2, edgecolor='cyan', linestyle='--')
                ax1.add_patch(rect)
        
        for idx, row in fvg_bearish.iterrows():
            pos = df.index.get_loc(idx)
            if not pd.isna(row['fvg_top']) and not pd.isna(row['fvg_bottom']):
                fvg_height = row['fvg_top'] - row['fvg_bottom']
                rect = Rectangle((pos-0.2, row['fvg_bottom']), 1, fvg_height,
                               facecolor='magenta', alpha=0.2, edgecolor='magenta', linestyle='--')
                ax1.add_patch(rect)
        
        # 4. BOS/CHoCH Signals (improved)
        bos_signals = df[df['bos'] == True]
        choch_signals = df[df['choch'] == True]
        
        for idx, row in bos_signals.iterrows():
            pos = df.index.get_loc(idx)
            color = '#00FF00' if 'BULLISH' in row['structure_break_type'] else '#FF0000'
            strength = row['structure_strength'] if 'structure_strength' in row else 1
            
            ax1.annotate(f'BOS\n{int(strength)}★', xy=(pos, row['close']), 
                        xytext=(pos, row['close'] + 15),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        color=color, fontsize=9, fontweight='bold', ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
        
        for idx, row in choch_signals.iterrows():
            pos = df.index.get_loc(idx)
            color = '#90EE90' if 'BULLISH' in row['structure_break_type'] else '#FFB6C1'
            strength = row['structure_strength'] if 'structure_strength' in row else 1
            
            ax1.annotate(f'CHoCH\n{int(strength)}★', xy=(pos, row['close']), 
                        xytext=(pos, row['close'] - 15),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        color=color, fontsize=9, fontweight='bold', ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
        
        # 5. Premium/Discount Zones (improved)
        latest_fib50 = df['fib_50'].iloc[-1]
        ax1.axhline(y=latest_fib50, color='#FFA500', linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(len(df)-1, latest_fib50, f' 50% EQ ({latest_fib50:.2f})', 
                color='#FFA500', fontsize=11, va='center', fontweight='bold')
        
        # Premium zone (above 50%)
        premium_y = df['period_high'].iloc[-1]
        ax1.fill_between(range(len(df)), latest_fib50, premium_y, 
                        alpha=0.15, color='red', label='Premium Zone')
        
        # Discount zone (below 50%)
        discount_y = df['period_low'].iloc[-1]
        ax1.fill_between(range(len(df)), discount_y, latest_fib50, 
                        alpha=0.15, color='green', label='Discount Zone')
        
        # 6. Current Structure & Statistics
        current_structure = df['structure_type'].iloc[-1]
        current_pd = df['premium_discount'].iloc[-1]
        
        # Count significant elements
        total_obs = len(df[df['bullish_ob'] | df['bearish_ob']])
        total_fvgs = len(df[df['fvg_bullish'] | df['fvg_bearish']])
        total_swings = len(df[df['swing_high'] | df['swing_low']])
        
        structure_color = {'UPTREND': '#00ff00', 'DOWNTREND': '#ff0000', 'RANGING': '#ffff00'}
        
        # Status box with enhanced info
        status_text = f'Structure: {current_structure}\nZone: {current_pd}\nOBs: {total_obs} | FVGs: {total_fvgs}\nSwings: {total_swings}'
        
        ax1.text(0.02, 0.95, status_text, transform=ax1.transAxes, fontsize=12, fontweight='bold',
                color=structure_color.get(current_structure, 'white'),
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        
        # Chart formatting
        ax1.set_title(f'{self.symbol} - {timeframe} Enhanced SMC Analysis', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#FFD700', markersize=12, label='Swing High'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#00FFFF', markersize=12, label='Swing Low'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.3, label='Bullish OB'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.3, label='Bearish OB'),
            plt.Rectangle((0,0),1,1, facecolor='cyan', alpha=0.2, label='Bull FVG'),
            plt.Rectangle((0,0),1,1, facecolor='magenta', alpha=0.2, label='Bear FVG'),
            plt.Line2D([0], [0], color='#FFA500', linestyle='--', linewidth=2, label='50% EQ'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.75), fontsize=10)
        
        # === Enhanced Structure Strength Indicator ===
        structure_strength = []
        for idx, row in df.iterrows():
            strength = 0
            if row['bos'] and 'structure_strength' in row:
                strength = row['structure_strength'] if 'BULLISH' in row['structure_break_type'] else -row['structure_strength']
            elif row['choch'] and 'structure_strength' in row:
                strength = row['structure_strength'] * 0.5 if 'BULLISH' in row['structure_break_type'] else -row['structure_strength'] * 0.5
            elif row['bos']:
                strength = 2 if 'BULLISH' in row['structure_break_type'] else -2
            elif row['choch']:
                strength = 1 if 'BULLISH' in row['structure_break_type'] else -1
            structure_strength.append(strength)
        
        colors = ['#ff4444' if x < 0 else '#00ff88' if x > 0 else '#666666' for x in structure_strength]
        bars = ax2.bar(range(len(structure_strength)), structure_strength, color=colors, alpha=0.8)
        
        # Highlight significant strength
        for i, (bar, strength) in enumerate(zip(bars, structure_strength)):
            if abs(strength) >= 2:
                bar.set_alpha(1.0)
                bar.set_edgecolor('white')
                bar.set_linewidth(1)
        
        ax2.set_ylabel('Structure\nStrength', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Candles', fontsize=12)
        ax2.axhline(y=0, color='white', linestyle='-', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-3.5, 3.5)
        
        # Enhanced legend for structure strength
        strength_info = 'BOS: ±2-3★ | CHoCH: ±1-1.5★\nHigher ★ = Stronger Move'
        ax2.text(0.02, 0.85, strength_info, transform=ax2.transAxes, 
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # บันทึกไฟล์
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/SMC_{self.symbol}_{timeframe}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return filename
    
    def create_multi_timeframe_dashboard(self, save_dir: str = "smc_charts") -> str:
        """
        สร้าง Dashboard แสดงหลายไทม์เฟรมในภาพเดียว
        """
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        timeframes = ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']
        
        for i, tf in enumerate(timeframes):
            if tf not in self.market_data:
                continue
                
            ax = axes[i]
            df = self.market_data[tf].tail(100)  # 100 candles สำหรับ overview
            
            # Simple line chart สำหรับ overview
            ax.plot(range(len(df)), df['close'], color='white', linewidth=1.5, alpha=0.8)
            
            # Swing points
            swing_highs = df[df['swing_high'] == True]
            swing_lows = df[df['swing_low'] == True]
            
            for idx, row in swing_highs.iterrows():
                pos = df.index.get_loc(idx)
                ax.scatter(pos, row['swing_high_price'], color='yellow', s=50, marker='^')
            
            for idx, row in swing_lows.iterrows():
                pos = df.index.get_loc(idx)
                ax.scatter(pos, row['swing_low_price'], color='cyan', s=50, marker='v')
            
            # BOS/CHoCH
            bos_signals = df[df['bos'] == True]
            choch_signals = df[df['choch'] == True]
            
            for idx, row in bos_signals.iterrows():
                pos = df.index.get_loc(idx)
                ax.scatter(pos, row['close'], color='lime', s=80, marker='o', alpha=0.8)
            
            for idx, row in choch_signals.iterrows():
                pos = df.index.get_loc(idx)
                ax.scatter(pos, row['close'], color='orange', s=80, marker='s', alpha=0.8)
            
            # Premium/Discount
            latest_fib50 = df['fib_50'].iloc[-1]
            ax.axhline(y=latest_fib50, color='orange', linestyle='--', alpha=0.5)
            
            # Status
            current_structure = df['structure_type'].iloc[-1]
            current_pd = df['premium_discount'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            structure_color = {'UPTREND': 'green', 'DOWNTREND': 'red', 'RANGING': 'yellow'}
            
            ax.set_title(f'{tf}\n{current_structure} | {current_pd}\n{current_price:.2f}', 
                        fontsize=12, fontweight='bold',
                        color=structure_color.get(current_structure, 'white'))
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Price', fontsize=10)
            
        plt.suptitle(f'{self.symbol} - Multi-Timeframe SMC Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', markersize=10, label='Swing High'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='cyan', markersize=10, label='Swing Low'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='BOS'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='CHoCH'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='50% Fib'),
        ]
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        
        # บันทึกไฟล์
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/SMC_Dashboard_{self.symbol}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return filename

    def display_analysis_summary(self, analysis: Dict, bias: Dict):
        """แสดงสรุปผลการวิเคราะห์"""
        print("\n" + "="*60)
        print("📈 SMC MARKET STRUCTURE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Symbol: {self.symbol}")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("🏗️  STRUCTURE ANALYSIS:")
        for tf in ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']:
            if tf in analysis and 'structure_type' in analysis[tf]:
                data = analysis[tf]
                print(f"   {tf:3}: {data['structure_type']:9} | {data['premium_discount']:8} | {data['last_structure_break']}")
        
        print()
        print("🎯 TRADING BIAS:")
        print(f"   H4 Bias: {bias['h4_bias']}")
        print(f"   H1 Bias: {bias['h1_bias']}")
        print(f"   Counter-trend Opportunity: {bias['counter_trend_opportunity']}")
        print(f"   Current Phase: {bias['current_phase']}")
        print(f"   Recommended Action: {bias['recommended_action']}")
        
        print()
        if 'M1' in analysis and 'current_price' in analysis['M1']:
            print(f"💰 Current Price: {analysis['M1']['current_price']:.2f}")

    def run_full_analysis_with_charts(self, save_charts: bool = True) -> Dict:
        """
        รันการวิเคราะห์แบบเต็มพร้อมสร้างกราฟ
        """
        print("🚀 SMC Trading AI - Full Analysis with Chart Generation")
        print("=" * 60)
        
        # วิเคราะห์หลายไทม์เฟรม
        analysis = self.analyze_multi_timeframe()
        bias = self.get_trading_bias(analysis)
        
        # แสดงผลสรุป
        self.display_analysis_summary(analysis, bias)
        
        chart_files = []
        if save_charts:
            print("\n📊 Generating Charts...")
            
            try:
                # สร้างกราฟแต่ละไทม์เฟรม
                for tf in ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']:
                    if tf in self.market_data:
                        print(f"   📈 Creating {tf} chart...")
                        chart_file = self.save_chart_analysis(tf)
                        chart_files.append(chart_file)
                        print(f"   ✅ Saved: {chart_file}")
                
                # สร้าง Dashboard
                print("   📊 Creating multi-timeframe dashboard...")
                dashboard_file = self.create_multi_timeframe_dashboard()
                chart_files.append(dashboard_file)
                print(f"   ✅ Dashboard saved: {dashboard_file}")
                
                print(f"\n🎨 Total {len(chart_files)} charts generated!")
                
            except Exception as e:
                print(f"   ❌ Chart generation error: {str(e)}")
                print("   ℹ️  Install: pip install matplotlib seaborn")
        
        return {
            'analysis': analysis,
            'bias': bias,
            'chart_files': chart_files,
            'timestamp': datetime.now().isoformat()
        }

# ตัวอย่างการใช้งาน
def main():
    """ฟังก์ชันหลักสำหรับทดสอบระบบ"""
    
    print("🚀 SMC Trading AI - Market Structure Engine with Chart Visualization")
    print("=" * 70)
    
    # สร้าง instance
    smc = SMCMarketStructure()
    
    # เชื่อมต่อ MT5
    if not smc.connect_mt5():
        print("❌ Cannot connect to MT5. Please check your MT5 terminal.")
        return
    
    try:
        # รันการวิเคราะห์พร้อมสร้างกราฟ
        result = smc.run_full_analysis_with_charts(save_charts=True)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Charts saved in: ./smc_charts/")
        
        print("\n📋 Next Steps:")
        print("1. Review generated charts for visual confirmation")
        print("2. Monitor M15/M5 for Order Block formations")
        print("3. Wait for M1 CHoCH + BOS confirmation")
        print("4. Enter on second zone (after LQ zone)")
        print("5. SL below M1 OB + spread buffer")
        
        # แสดงรายชื่อไฟล์ที่สร้าง
        if result['chart_files']:
            print(f"\n🎨 Generated Charts:")
            for chart_file in result['chart_files']:
                print(f"   📊 {chart_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()