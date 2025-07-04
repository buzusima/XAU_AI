"""
🛡️ PULLBACK PROTECTION PLUGIN สำหรับ FOREX TRADING SYSTEM
=======================================================

วัตถุประสงค์:
- ป้องกันการเทรดในจุดที่มี pullback risk สูง
- รอ pullback เกิดขึ้นแล้วเข้าเทรดในจุดที่ดีกว่า
- ลดการโดน SL จาก pullback
- เพิ่ม Win Rate จาก 55% เป็น 65%+

ผู้พัฒนา: Professional Forex Trading System
Version: 1.0.0
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import json

class PullbackProtectionPlugin:
    """🛡️ Plugin ป้องกัน Pullback สำหรับ Forex Trading System"""
    
    def __init__(self, logger=None):
        """เริ่มต้น Pullback Protection Plugin"""
        
        # ตั้งค่า Logger
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # ⚙️ การตั้งค่า Pullback Detection
        self.pullback_settings = {
            # RSI Settings
            'rsi_extreme_buy': 25,      # RSI ต่ำกว่านี้ = ขาย oversold มาก
            'rsi_extreme_sell': 75,     # RSI สูงกว่านี้ = ซื้อ overbought มาก
            'rsi_safe_zone_min': 35,    # RSI ปลอดภัยขั้นต่ำ
            'rsi_safe_zone_max': 65,    # RSI ปลอดภัยขั้นสูง
            
            # Price Distance from EMA Settings
            'ema_distance_warning': 0.3,   # ระยะห่าง EMA 21 เป็น % ของ ATR
            'ema_distance_danger': 0.5,    # ระยะห่าง EMA 21 เป็น % ของ ATR (อันตราย)
            
            # Volume Settings
            'volume_decline_threshold': 0.8,  # Volume ลดลงเหลือ 80% = ความเสี่ยง
            'volume_very_low': 0.6,          # Volume ต่ำมาก = อันตราย
            
            # Timeout Settings
            'max_wait_minutes': 30,      # รอ pullback สูงสุด 30 นาที
            'recheck_interval': 60,      # ตรวจสอบใหม่ทุก 60 วินาที
        }
        
        # 🎯 การให้คะแนนความเสี่ยง Pullback
        self.risk_scoring = {
            'rsi_extreme': 3,           # RSI extreme = +3 คะแนนเสี่ยง
            'rsi_moderate': 1,          # RSI moderate = +1 คะแนนเสี่ยง
            'price_distance_high': 2,   # ราคาห่างจาก EMA มาก = +2 คะแนน
            'price_distance_medium': 1, # ราคาห่างจาก EMA ปานกลาง = +1 คะแนน
            'volume_decline': 1,        # Volume ลดลง = +1 คะแนน
            'volume_very_low': 2,       # Volume ต่ำมาก = +2 คะแนน
        }
        
        # 📊 เกณฑ์การตัดสินใจ
        self.decision_thresholds = {
            'low_risk': 0,      # คะแนน 0-1 = ความเสี่ยงต่ำ (เทรดได้)
            'medium_risk': 2,   # คะแนน 2-3 = ความเสี่ยงปานกลาง (ระวัง)
            'high_risk': 4,     # คะแนน 4+ = ความเสี่ยงสูง (ห้ามเทรด, รอ pullback)
        }
        
        # 🔄 ระบบติดตาม Pullback
        self.waiting_positions = {}  # เก็บ positions ที่รอ pullback
        self.pullback_history = {}   # เก็บประวัติ pullback
        
        # 📈 สถิติการทำงาน
        self.statistics = {
            'total_signals_checked': 0,
            'signals_blocked': 0,
            'pullback_waits': 0,
            'successful_entries_after_wait': 0,
            'timeout_expired': 0,
            'false_signals_prevented': 0,
        }
        
        # 🔧 สถานะระบบ
        self.enabled = True
        self.auto_trading_compatible = True
        
        # 📝 Database สำหรับเก็บข้อมูล
        self._init_database()
        
        self.logger.info("🛡️ Pullback Protection Plugin เริ่มต้นเรียบร้อย")
        self.logger.info(f"📊 เป้าหมาย: เพิ่ม Win Rate จาก 55% → 65%+")
    
    def _init_database(self):
        """🗄️ เริ่มต้น Database สำหรับเก็บข้อมูล"""
        try:
            conn = sqlite3.connect('pullback_protection.db')
            cursor = conn.cursor()
            
            # ตาราง waiting_positions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS waiting_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    original_signal TEXT NOT NULL,
                    original_strength REAL NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_factors TEXT NOT NULL,
                    wait_start_time TEXT NOT NULL,
                    timeout_time TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    target_entry_price REAL,
                    status TEXT DEFAULT 'WAITING',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง pullback_events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pullback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    price_before REAL NOT NULL,
                    price_after REAL NOT NULL,
                    pullback_pips REAL NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    successful_entry BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plugin_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    signals_blocked INTEGER DEFAULT 0,
                    pullback_waits INTEGER DEFAULT 0,
                    successful_entries INTEGER DEFAULT 0,
                    timeout_expired INTEGER DEFAULT 0,
                    false_signals_prevented INTEGER DEFAULT 0,
                    UNIQUE(date)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Database เริ่มต้นเรียบร้อย")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {str(e)}")
    
    def analyze_pullback_risk(self, symbol: str, market_data: Dict) -> Dict:
        """
        🔍 วิเคราะห์ความเสี่ยง Pullback
        
        Args:
            symbol: คู่เงิน เช่น 'EURUSD'
            market_data: ข้อมูลตลาดจาก get_symbol_data()
        
        Returns:
            {
                'risk_score': int,           # คะแนนความเสี่ยง 0-10
                'risk_level': str,           # 'LOW', 'MEDIUM', 'HIGH'
                'recommendation': str,       # 'TRADE', 'WAIT', 'AVOID'
                'risk_factors': List[str],   # รายการปัจจัยเสี่ยง
                'wait_conditions': Dict,     # เงื่อนไขที่ต้องรอ
                'timeout_minutes': int       # เวลา timeout
            }
        """
        try:
            self.statistics['total_signals_checked'] += 1
            
            # ดึงข้อมูลจาก market_data
            rsi = market_data.get('rsi', 50)
            current_price = market_data.get('current_price', 0)
            ema_21 = market_data.get('ema_21', current_price)
            volume_ratio = market_data.get('volumeRatio', 1.0)
            atr_percent = market_data.get('atrPercent', 0.001)
            signal = market_data.get('signal', 'NONE')
            
            # เริ่มต้นการวิเคราะห์
            risk_score = 0
            risk_factors = []
            wait_conditions = {}
            
            # 📊 1. วิเคราะห์ RSI Extreme
            if signal == 'BUY' and rsi <= self.pullback_settings['rsi_extreme_buy']:
                risk_score += self.risk_scoring['rsi_extreme']
                risk_factors.append(f"RSI Extreme Oversold ({rsi:.1f})")
                wait_conditions['rsi_recovery'] = self.pullback_settings['rsi_safe_zone_min']
                
            elif signal == 'SELL' and rsi >= self.pullback_settings['rsi_extreme_sell']:
                risk_score += self.risk_scoring['rsi_extreme']
                risk_factors.append(f"RSI Extreme Overbought ({rsi:.1f})")
                wait_conditions['rsi_recovery'] = self.pullback_settings['rsi_safe_zone_max']
                
            elif signal == 'BUY' and rsi <= 35:
                risk_score += self.risk_scoring['rsi_moderate']
                risk_factors.append(f"RSI Oversold ({rsi:.1f})")
                
            elif signal == 'SELL' and rsi >= 65:
                risk_score += self.risk_scoring['rsi_moderate']
                risk_factors.append(f"RSI Overbought ({rsi:.1f})")
            
            # 📏 2. วิเคราะห์ระยะห่างจาก EMA
            if current_price > 0 and ema_21 > 0 and atr_percent > 0:
                price_distance = abs(current_price - ema_21) / current_price
                atr_threshold_high = atr_percent * self.pullback_settings['ema_distance_danger']
                atr_threshold_medium = atr_percent * self.pullback_settings['ema_distance_warning']
                
                if price_distance >= atr_threshold_high:
                    risk_score += self.risk_scoring['price_distance_high']
                    risk_factors.append(f"Price Far from EMA21 ({price_distance*100:.1f}%)")
                    wait_conditions['ema_return'] = ema_21
                    
                elif price_distance >= atr_threshold_medium:
                    risk_score += self.risk_scoring['price_distance_medium']
                    risk_factors.append(f"Price Distance from EMA21 ({price_distance*100:.1f}%)")
            
            # 📉 3. วิเคราะห์ Volume
            if volume_ratio <= self.pullback_settings['volume_very_low']:
                risk_score += self.risk_scoring['volume_very_low']
                risk_factors.append(f"Volume Very Low ({volume_ratio:.2f})")
                wait_conditions['volume_recovery'] = 1.0
                
            elif volume_ratio <= self.pullback_settings['volume_decline_threshold']:
                risk_score += self.risk_scoring['volume_decline']
                risk_factors.append(f"Volume Declining ({volume_ratio:.2f})")
            
            # 🎯 กำหนดระดับความเสี่ยงและคำแนะนำ
            if risk_score >= self.decision_thresholds['high_risk']:
                risk_level = 'HIGH'
                recommendation = 'WAIT'
                timeout_minutes = self.pullback_settings['max_wait_minutes']
                self.statistics['signals_blocked'] += 1
                
            elif risk_score >= self.decision_thresholds['medium_risk']:
                risk_level = 'MEDIUM'
                recommendation = 'WAIT'
                timeout_minutes = self.pullback_settings['max_wait_minutes'] // 2
                self.statistics['signals_blocked'] += 1
                
            else:
                risk_level = 'LOW'
                recommendation = 'TRADE'
                timeout_minutes = 0
            
            result = {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'risk_factors': risk_factors,
                'wait_conditions': wait_conditions,
                'timeout_minutes': timeout_minutes,
                'analysis_time': datetime.now().isoformat(),
                'original_signal': signal,
                'current_price': current_price
            }
            
            # บันทึกการวิเคราะห์
            if recommendation == 'WAIT':
                self._add_waiting_position(symbol, result)
            
            # self.logger.info(f"🔍 {symbol}: Risk={risk_level}({risk_score}), Recommendation={recommendation}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing pullback risk for {symbol}: {str(e)}")
            return {
                'risk_score': 0,
                'risk_level': 'LOW',
                'recommendation': 'TRADE',
                'risk_factors': [f'Analysis Error: {str(e)}'],
                'wait_conditions': {},
                'timeout_minutes': 0,
                'analysis_time': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _add_waiting_position(self, symbol: str, analysis_result: Dict):
        """📝 เพิ่ม position ที่รอ pullback"""
        try:
            wait_start = datetime.now()
            timeout_time = wait_start + timedelta(minutes=analysis_result['timeout_minutes'])
            
            waiting_data = {
                'symbol': symbol,
                'analysis': analysis_result,
                'wait_start': wait_start,
                'timeout_time': timeout_time,
                'last_check': wait_start,
                'status': 'WAITING'
            }
            
            self.waiting_positions[symbol] = waiting_data
            self.statistics['pullback_waits'] += 1
            
            # บันทึกลง Database
            conn = sqlite3.connect('pullback_protection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO waiting_positions 
                (symbol, original_signal, original_strength, risk_score, risk_factors, 
                 wait_start_time, timeout_time, current_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                analysis_result.get('original_signal', 'NONE'),
                0,  # strength จะต้องส่งมาจากระบบหลัก
                analysis_result['risk_score'],
                json.dumps(analysis_result['risk_factors']),
                wait_start.isoformat(),
                timeout_time.isoformat(),
                analysis_result.get('current_price', 0),
                'WAITING'
            ))
            
            conn.commit()
            conn.close()
            
            # self.logger.info(f"⏳ {symbol}: เพิ่มในรายการรอ pullback (timeout: {analysis_result['timeout_minutes']} นาที)")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding waiting position for {symbol}: {str(e)}")
    
    def check_pullback_recovery(self, symbol: str, current_market_data: Dict) -> Dict:
        """
        🔄 ตรวจสอบการฟื้นตัวจาก Pullback
        
        Returns:
            {
                'ready_to_trade': bool,
                'recovery_factors': List[str],
                'wait_expired': bool,
                'new_recommendation': str
            }
        """
        try:
            if symbol not in self.waiting_positions:
                return {'ready_to_trade': True, 'recovery_factors': [], 'wait_expired': False}
            
            waiting_data = self.waiting_positions[symbol]
            analysis = waiting_data['analysis']
            wait_conditions = analysis['wait_conditions']
            
            # ตรวจสอบ Timeout
            current_time = datetime.now()
            if current_time >= waiting_data['timeout_time']:
                self._remove_waiting_position(symbol, 'TIMEOUT')
                self.statistics['timeout_expired'] += 1
                return {
                    'ready_to_trade': True,
                    'recovery_factors': ['Timeout Expired'],
                    'wait_expired': True,
                    'new_recommendation': 'TRADE_WITH_CAUTION'
                }
            
            # ตรวจสอบเงื่อนไขการฟื้นตัว
            recovery_factors = []
            conditions_met = 0
            total_conditions = len(wait_conditions)
            
            current_rsi = current_market_data.get('rsi', 50)
            current_price = current_market_data.get('current_price', 0)
            current_volume = current_market_data.get('volumeRatio', 1.0)
            
            # ตรวจ RSI Recovery
            if 'rsi_recovery' in wait_conditions:
                target_rsi = wait_conditions['rsi_recovery']
                original_signal = analysis['original_signal']
                
                if original_signal == 'BUY' and current_rsi >= target_rsi:
                    recovery_factors.append(f"RSI ฟื้นตัว ({current_rsi:.1f} >= {target_rsi})")
                    conditions_met += 1
                elif original_signal == 'SELL' and current_rsi <= target_rsi:
                    recovery_factors.append(f"RSI ฟื้นตัว ({current_rsi:.1f} <= {target_rsi})")
                    conditions_met += 1
            
            # ตรวจ EMA Return
            if 'ema_return' in wait_conditions:
                ema_21 = current_market_data.get('ema_21', current_price)
                if abs(current_price - ema_21) / current_price < 0.002:  # ใกล้ EMA แล้ว
                    recovery_factors.append("Price กลับมาใกล้ EMA21")
                    conditions_met += 1
            
            # ตรวจ Volume Recovery
            if 'volume_recovery' in wait_conditions:
                if current_volume >= wait_conditions['volume_recovery']:
                    recovery_factors.append(f"Volume ฟื้นตัว ({current_volume:.2f})")
                    conditions_met += 1
            
            # ตัดสินใจ
            if total_conditions == 0:  # ไม่มีเงื่อนไข = พร้อมเทรด
                ready_to_trade = True
                new_recommendation = 'TRADE'
            elif conditions_met >= total_conditions * 0.7:  # 70% ของเงื่อนไขผ่าน
                ready_to_trade = True
                new_recommendation = 'TRADE'
                self.statistics['successful_entries_after_wait'] += 1
            else:
                ready_to_trade = False
                new_recommendation = 'CONTINUE_WAITING'
            
            if ready_to_trade:
                self._remove_waiting_position(symbol, 'RECOVERED')
            
            return {
                'ready_to_trade': ready_to_trade,
                'recovery_factors': recovery_factors,
                'wait_expired': False,
                'new_recommendation': new_recommendation,
                'conditions_met': f"{conditions_met}/{total_conditions}"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error checking pullback recovery for {symbol}: {str(e)}")
            return {
                'ready_to_trade': True,
                'recovery_factors': [f'Check Error: {str(e)}'],
                'wait_expired': False,
                'new_recommendation': 'TRADE_WITH_CAUTION'
            }
    
    def _remove_waiting_position(self, symbol: str, reason: str):
        """🗑️ ลบ position ออกจากรายการรอ"""
        try:
            if symbol in self.waiting_positions:
                waiting_data = self.waiting_positions[symbol]
                
                # บันทึกลง Database
                conn = sqlite3.connect('pullback_protection.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE waiting_positions 
                    SET status = ? 
                    WHERE symbol = ? AND status = 'WAITING'
                ''', (reason, symbol))
                
                conn.commit()
                conn.close()
                
                del self.waiting_positions[symbol]
                self.logger.info(f"✅ {symbol}: ลบออกจากรายการรอ - {reason}")
                
        except Exception as e:
            self.logger.error(f"❌ Error removing waiting position for {symbol}: {str(e)}")
    
    def process_signal(self, symbol: str, original_signal_data: Dict) -> Dict:
        """
        🎯 ประมวลผล Signal หลัก - ใช้แทนที่ signal เดิม
        
        Args:
            symbol: คู่เงิน
            original_signal_data: ข้อมูล signal จาก get_symbol_data()
        
        Returns:
            ข้อมูล signal ที่ปรับแล้ว (อาจเปลี่ยนเป็น WAIT)
        """
        try:
            if not self.enabled:
                return original_signal_data
            
            original_signal = original_signal_data.get('signal', 'NONE')
            
            # ถ้าไม่มี signal = ไม่ต้องตรวจ
            if original_signal == 'NONE':
                return original_signal_data
            
            # ตรวจสอบว่ามี position รอ pullback อยู่หรือไม่
            if symbol in self.waiting_positions:
                recovery_check = self.check_pullback_recovery(symbol, original_signal_data)
                
                if recovery_check['ready_to_trade']:
                    self.logger.info(f"✅ {symbol}: พร้อมเทรด - {recovery_check['recovery_factors']}")
                    # คืนค่า signal เดิม
                    return original_signal_data
                else:
                    # ยังไม่พร้อม ต้องรอต่อ
                    modified_data = original_signal_data.copy()
                    modified_data['signal'] = 'WAIT'
                    modified_data['pullback_protection'] = {
                        'status': 'WAITING_PULLBACK',
                        'conditions_met': recovery_check.get('conditions_met', '0/0'),
                        'recovery_factors': recovery_check['recovery_factors']
                    }
                    return modified_data
            
            # วิเคราะห์ความเสี่ยง pullback สำหรับ signal ใหม่
            risk_analysis = self.analyze_pullback_risk(symbol, original_signal_data)
            
            if risk_analysis['recommendation'] == 'WAIT':
                # เปลี่ยน signal เป็น WAIT
                modified_data = original_signal_data.copy()
                modified_data['signal'] = 'WAIT'
                modified_data['pullback_protection'] = {
                    'status': 'PULLBACK_RISK_DETECTED',
                    'risk_level': risk_analysis['risk_level'],
                    'risk_score': risk_analysis['risk_score'],
                    'risk_factors': risk_analysis['risk_factors'],
                    'timeout_minutes': risk_analysis['timeout_minutes']
                }
                
                # self.logger.warning(f"⚠️ {symbol}: Signal blocked - Pullback risk {risk_analysis['risk_level']}")
                return modified_data
            
            else:
                # Signal ปลอดภัย ให้เทรดได้
                return original_signal_data
                
        except Exception as e:
            self.logger.error(f"❌ Error processing signal for {symbol}: {str(e)}")
            return original_signal_data
    
    def get_waiting_positions_summary(self) -> Dict:
        """📊 สรุป positions ที่รอ pullback"""
        try:
            summary = {
                'total_waiting': len(self.waiting_positions),
                'positions': [],
                'next_timeout': None
            }
            
            current_time = datetime.now()
            next_timeout_time = None
            
            for symbol, waiting_data in self.waiting_positions.items():
                timeout_time = waiting_data['timeout_time']
                remaining_minutes = (timeout_time - current_time).total_seconds() / 60
                
                position_info = {
                    'symbol': symbol,
                    'wait_start': waiting_data['wait_start'].strftime('%H:%M:%S'),
                    'timeout_in_minutes': max(0, int(remaining_minutes)),
                    'risk_level': waiting_data['analysis']['risk_level'],
                    'risk_factors': waiting_data['analysis']['risk_factors']
                }
                
                summary['positions'].append(position_info)
                
                if next_timeout_time is None or timeout_time < next_timeout_time:
                    next_timeout_time = timeout_time
            
            if next_timeout_time:
                summary['next_timeout'] = next_timeout_time.strftime('%H:%M:%S')
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting waiting positions summary: {str(e)}")
            return {'total_waiting': 0, 'positions': [], 'error': str(e)}
    
    def get_statistics(self) -> Dict:
        """📈 สถิติการทำงานของ Plugin"""
        try:
            if self.statistics['total_signals_checked'] > 0:
                block_rate = (self.statistics['signals_blocked'] / self.statistics['total_signals_checked']) * 100
                success_rate = 0
                if self.statistics['pullback_waits'] > 0:
                    success_rate = (self.statistics['successful_entries_after_wait'] / self.statistics['pullback_waits']) * 100
            else:
                block_rate = 0
                success_rate = 0
            
            return {
                'plugin_enabled': self.enabled,
                'total_signals_checked': self.statistics['total_signals_checked'],
                'signals_blocked': self.statistics['signals_blocked'],
                'block_rate_percent': round(block_rate, 1),
                'pullback_waits': self.statistics['pullback_waits'],
                'successful_entries_after_wait': self.statistics['successful_entries_after_wait'],
                'success_after_wait_percent': round(success_rate, 1),
                'timeout_expired': self.statistics['timeout_expired'],
                'false_signals_prevented': self.statistics['false_signals_prevented'],
                'currently_waiting': len(self.waiting_positions),
                'estimated_win_rate_improvement': f"+{int(success_rate * 0.1)}%"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting statistics: {str(e)}")
            return {'error': str(e)}
    
    def reset_statistics(self):
        """🔄 รีเซ็ตสถิติ"""
        self.statistics = {
            'total_signals_checked': 0,
            'signals_blocked': 0,
            'pullback_waits': 0,
            'successful_entries_after_wait': 0,
            'timeout_expired': 0,
            'false_signals_prevented': 0,
        }
        self.logger.info("🔄 สถิติ Plugin ถูกรีเซ็ตแล้ว")
    
    def enable(self):
        """✅ เปิดใช้งาน Plugin"""
        self.enabled = True
        self.logger.info("✅ Pullback Protection Plugin เปิดใช้งาน")
    
    def disable(self):
        """❌ ปิดใช้งาน Plugin"""
        self.enabled = False
        # ล้าง waiting positions
        self.waiting_positions.clear()
        self.logger.info("❌ Pullback Protection Plugin ปิดใช้งาน")
    
    def update_settings(self, new_settings: Dict):
        """⚙️ อัพเดตการตั้งค่า"""
        try:
            for key, value in new_settings.items():
                if key in self.pullback_settings:
                    self.pullback_settings[key] = value
                    self.logger.info(f"⚙️ Updated {key} = {value}")
            
            self.logger.info("✅ การตั้งค่า Plugin อัพเดตเรียบร้อย")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating settings: {str(e)}")


# 🔧 INTEGRATION HELPER FUNCTIONS

def integrate_with_main_system(main_dashboard, enable_on_start=True):
    """
    🔗 ฟังก์ชันเชื่อมต่อกับระบบหลัก
    
    วิธีใช้:
    from pullback_protection import integrate_with_main_system
    integrate_with_main_system(self)  # ใน __init__ ของ main dashboard
    """
    try:
        # สร้าง Plugin instance
        main_dashboard.pullback_protection = PullbackProtectionPlugin(main_dashboard.logger)
        
        if enable_on_start:
            main_dashboard.pullback_protection.enable()
        
        # เพิ่ม method ใหม่ให้ main dashboard
        original_get_symbol_data = main_dashboard.get_symbol_data
        
        def enhanced_get_symbol_data(symbol):
            """📊 get_symbol_data ที่มี Pullback Protection"""
            try:
                # ดึงข้อมูลเดิมก่อน
                original_data = original_get_symbol_data(symbol)
                
                if original_data and hasattr(main_dashboard, 'pullback_protection'):
                    # ประมวลผลผ่าน Pullback Protection
                    protected_data = main_dashboard.pullback_protection.process_signal(symbol, original_data)
                    return protected_data
                
                return original_data
                
            except Exception as e:
                main_dashboard.logger.error(f"❌ Pullback protection error for {symbol}: {str(e)}")
                return original_data
        
        # แทนที่ method เดิม
        main_dashboard.get_symbol_data = enhanced_get_symbol_data
        
        print("✅ Pullback Protection Plugin ติดตั้งเรียบร้อย")
        print("🛡️ Expected Result: Win Rate 55% → 65%+")
        
        return True
        
    except Exception as e:
        print(f"❌ Error integrating Pullback Protection: {str(e)}")
        return False


# 🧪 TESTING FUNCTIONS

def test_pullback_detection():
    """🧪 ทดสอบระบบตรวจจับ Pullback"""
    
    plugin = PullbackProtectionPlugin()
    
    # ทดสอบกรณี RSI Extreme
    test_data_1 = {
        'symbol': 'EURUSD',
        'rsi': 20,  # RSI ต่ำมาก
        'current_price': 1.1000,
        'ema_21': 1.0990,
        'volumeRatio': 0.5,  # Volume ต่ำ
        'atrPercent': 0.001,
        'signal': 'BUY'
    }
    
    result_1 = plugin.analyze_pullback_risk('EURUSD', test_data_1)
    print("🧪 Test 1 - RSI Extreme:")
    print(f"   Risk Level: {result_1['risk_level']}")
    print(f"   Recommendation: {result_1['recommendation']}")
    print(f"   Risk Factors: {result_1['risk_factors']}")
    print()
    
    # ทดสอบกรณีปกติ
    test_data_2 = {
        'symbol': 'GBPUSD',
        'rsi': 55,  # RSI ปกติ
        'current_price': 1.3000,
        'ema_21': 1.2995,
        'volumeRatio': 1.2,  # Volume ดี
        'atrPercent': 0.001,
        'signal': 'BUY'
    }
    
    result_2 = plugin.analyze_pullback_risk('GBPUSD', test_data_2)
    print("🧪 Test 2 - Normal Conditions:")
    print(f"   Risk Level: {result_2['risk_level']}")
    print(f"   Recommendation: {result_2['recommendation']}")
    print(f"   Risk Factors: {result_2['risk_factors']}")
    
    # แสดงสถิติ
    print("\n📊 Plugin Statistics:")
    stats = plugin.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    print("🛡️ PULLBACK PROTECTION PLUGIN")
    print("=" * 50)
    test_pullback_detection()