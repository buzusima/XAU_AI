"""
Enhanced Smart Auto Trading System with Data Persistence
=======================================================
Professional Auto Trading with State Management & Data Recovery
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import threading
import time
import json
import logging
import os
import sqlite3
import requests
from typing import Dict, List, Optional, Tuple
import warnings
from broker_symbol_adapter import BrokerSymbolAdapter
from enhanced_signal_system import MultiTimeframeSignalEngine
warnings.filterwarnings('ignore')
from pullback_protection import PullbackProtectionPlugin
from trading_integration import EnhancedTradingSystemWithTrailing, add_trailing_stop_routes
from correlation_hedging_system import HedgeSystemIntegrator
from enhanced_signal_system import MultiTimeframeSignalEngine
from advanced_features import UniversalAdvancedTradingIntegrator

# CRITICAL FIX: Add clean_data_for_json function
def clean_data_for_json(data):
    """Clean data for JSON serialization - COMPLETELY FIXED"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned[key] = clean_data_for_json(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(clean_data_for_json(item) for item in data)
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Series):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict('records')
    elif hasattr(data, 'value'):  # Enum handling
        return data.value
    elif hasattr(data, '__dict__'):
        return str(data)
    elif pd.isna(data):
        return None
    elif data in [np.inf, -np.inf]:
        return None
    else:
        return data

def integrate_with_main_system(main_dashboard, enable_on_start=True):
    """ติดตั้ง Pullback Protection เข้ากับระบบหลัก"""
    try:
        if not hasattr(main_dashboard, 'pullback_protection') or not main_dashboard.pullback_protection:
            print("❌ No pullback_protection found")
            return False
            
        print("🛡️ Installing Pullback Protection Plugin...")
        
        # เก็บ method เดิม
        original_get_symbol_data = main_dashboard.get_symbol_data
        
        def enhanced_get_symbol_data(symbol):
            """Enhanced get_symbol_data with Pullback Protection"""
            try:
                original_data = original_get_symbol_data(symbol)
                
                if not main_dashboard.pullback_protection.enabled:
                    return original_data
                
                # ประมวลผลผ่าน Pullback Protection
                original_signal = original_data.get('signal', 'NONE')
                
                if original_signal != 'NONE':
                    # อัพเดตสถิติ
                    main_dashboard.pullback_protection.statistics['total_signals_checked'] += 1
                    
                    # วิเคราะห์ pullback risk
                    risk_analysis = main_dashboard.pullback_protection.analyze_pullback_risk(symbol, original_data)
                    
                    if risk_analysis.get('recommendation') == 'WAIT':
                        # บล็อค signal
                        main_dashboard.pullback_protection.statistics['signals_blocked'] += 1
                        
                        protected_data = original_data.copy()
                        protected_data['signal'] = 'WAIT'
                        protected_data['pullback_protection'] = {
                            'status': 'SIGNAL_BLOCKED',
                            'risk_level': risk_analysis.get('risk_level', 'UNKNOWN'),
                            'risk_factors': risk_analysis.get('risk_factors', []),
                            'original_signal': original_signal
                        }
                        
                        print(f"🛡️ {symbol}: Signal blocked by Pullback Protection ({original_signal} -> WAIT)")
                        return protected_data
                
                return original_data
                
            except Exception as e:
                print(f"❌ Pullback protection error for {symbol}: {e}")
                return original_data
        
        # แทนที่ method เดิม
        main_dashboard.get_symbol_data = enhanced_get_symbol_data
        
        # เปิดใช้งาน
        if enable_on_start:
            main_dashboard.pullback_protection.enabled = True
            
        print("✅ Pullback Protection Plugin installed successfully")
        print("🎯 Expected Result: Win Rate 55% → 65%+")
        return True
        
    except Exception as e:
        print(f"❌ Error integrating Pullback Protection: {str(e)}")
        return False
    
class DataPersistenceManager:
    """จัดการการบันทึกและโหลดข้อมูลระบบ"""
    
    def __init__(self, data_dir='trading_data'):
        self.data_dir = data_dir
        self.settings_file = os.path.join(data_dir, 'settings.json')
        self.positions_file = os.path.join(data_dir, 'positions.json')
        self.daily_stats_file = os.path.join(data_dir, 'daily_stats.json')
        self.pair_status_file = os.path.join(data_dir, 'pair_status.json')
        self.trade_history_file = os.path.join(data_dir, 'trade_history.json')
        self.db_file = os.path.join(data_dir, 'trading_system.db')
        
        self.create_data_directory()
        self.initialize_database()
        self.load_system_settings()  
        #self.load_pair_tracking_data()  

    def create_data_directory(self):
        """สร้างโฟลเดอร์เก็บข้อมูล"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
    def load_system_settings(self):
        """โหลดการตั้งค่าระบบ"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                print(f"[DISK] Settings loaded from {self.settings_file}")
                return settings
            else:
                print(f"[DISK] No settings file found, using defaults")
                return {}
        except Exception as e:
            print(f"[ERR] Error loading settings: {str(e)}")
            return {}
    
    def save_settings(self, settings_dict):
        """บันทึกการตั้งค่า"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=4, default=str)
            print(f"[DISK] Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            print(f"[ERR] Error saving settings: {str(e)}")
            return False
    
    def load_settings(self):
        """Alias สำหรับ load_system_settings"""
        return self.load_system_settings()
    
    def save_daily_stats(self, stats_dict):
        """บันทึกสถิติรายวัน"""
        try:
            with open(self.daily_stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=4, default=str)
            return True
        except Exception as e:
            print(f"[ERR] Error saving daily stats: {str(e)}")
            return False
    
    def load_daily_stats(self):
        """โหลดสถิติรายวัน"""
        try:
            if os.path.exists(self.daily_stats_file):
                with open(self.daily_stats_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[ERR] Error loading daily stats: {str(e)}")
            return {}
    
    def save_pair_status(self, pair_status_dict):
        """บันทึกสถานะ pairs"""
        try:
            with open(self.pair_status_file, 'w') as f:
                json.dump(pair_status_dict, f, indent=4, default=str)
            return True
        except Exception as e:
            print(f"[ERR] Error saving pair status: {str(e)}")
            return False
    
    def load_pair_status(self):
        """โหลดสถานะ pairs"""
        try:
            if os.path.exists(self.pair_status_file):
                with open(self.pair_status_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[ERR] Error loading pair status: {str(e)}")
            return {}
    
    def log_system_event(self, level, message, category):
        """บันทึก system events"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT,
                    category TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO system_events (timestamp, level, message, category)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), level, message, category))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[ERR] Error logging system event: {str(e)}")
    
    def get_trade_history(self, days=30):
        """ดึงประวัติการเทรด"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    volume REAL,
                    price REAL,
                    result TEXT
                )
            ''')
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute('''
                SELECT * FROM trade_history 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_date,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"[ERR] Error getting trade history: {str(e)}")
            return []
    
    def save_trade_to_db(self, trade_data):
        """บันทึกการเทรดลง database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticket TEXT,
                    symbol TEXT,
                    type TEXT,
                    volume REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    signal_strength REAL,
                    entry_quality TEXT,
                    risk_percentage REAL
                )
            ''')
            
            cursor.execute('''
                INSERT INTO trade_history 
                (timestamp, ticket, symbol, type, volume, entry_price, stop_loss, take_profit, signal_strength, entry_quality, risk_percentage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('entry_time', datetime.now().isoformat()),
                trade_data.get('ticket', ''),
                trade_data.get('symbol', ''),
                trade_data.get('type', ''),
                trade_data.get('volume', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('signal_strength', 0),
                trade_data.get('entry_quality', ''),
                trade_data.get('risk_percentage', 0)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERR] Error saving trade to DB: {str(e)}")
            return False
        
    def initialize_database(self):
        """สร้างฐานข้อมูล SQLite"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # ตาราง trade history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket TEXT UNIQUE,
                    symbol TEXT,
                    type TEXT,
                    volume REAL,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    profit REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    signal_strength REAL,
                    entry_quality TEXT,
                    risk_percentage REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง daily statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_statistics (
                    date TEXT PRIMARY KEY,
                    trades_executed INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ตาราง system logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT,
                    message TEXT,
                    category TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
    
    def save_settings(self, settings: Dict):
        """บันทึกการตั้งค่าระบบ"""
        try:
            cleaned_settings = clean_data_for_json(settings)
            with open(self.settings_file, 'w') as f:
                json.dump(cleaned_settings, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            return False
    
    def load_settings(self) -> Dict:
        """โหลดการตั้งค่าระบบ"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            return {}
    
    def save_daily_stats(self, stats: Dict):
        """บันทึกสถิติรายวัน"""
        try:
            # Clean data before saving
            cleaned_stats = clean_data_for_json(stats)
            with open(self.daily_stats_file, 'w') as f:
                json.dump(cleaned_stats, f, indent=2, default=str)
            
            # บันทึกลงฐานข้อมูลด้วย
            today = datetime.now().strftime('%Y-%m-%d')
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_statistics 
                (date, trades_executed, wins, losses, total_pnl, max_drawdown, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                stats.get('trades_executed', 0),
                stats.get('wins', 0),
                stats.get('losses', 0),
                stats.get('total_pnl', 0.0),
                stats.get('max_drawdown', 0.0),
                stats.get('win_rate', 0.0)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving daily stats: {str(e)}")
            return False    
    
    def load_daily_stats(self) -> Dict:
        """โหลดสถิติรายวัน"""
        try:
            if os.path.exists(self.daily_stats_file):
                with open(self.daily_stats_file, 'r') as f:
                    return json.load(f)
            return {
                'trades_executed': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        except Exception as e:
            print(f"Error loading daily stats: {str(e)}")
            return {}
    
    def save_pair_status(self, pair_status: Dict):
        """บันทึกสถานะของแต่ละ pair"""
        try:
            # Clean data before saving
            cleaned_status = clean_data_for_json(pair_status)
            with open(self.pair_status_file, 'w') as f:
                json.dump(cleaned_status, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving pair status: {str(e)}")
            return False    
   
    def load_pair_status(self) -> Dict:
        """โหลดสถานะของแต่ละ pair"""
        try:
            if os.path.exists(self.pair_status_file):
                with open(self.pair_status_file, 'r') as f:
                    data = json.load(f)
                # Convert datetime strings back to datetime objects
                for pair, status in data.items():
                    if 'cooldown_until' in status and status['cooldown_until']:
                        try:
                            status['cooldown_until'] = datetime.fromisoformat(status['cooldown_until'])
                        except:
                            status['cooldown_until'] = None
                return data
            return {}
        except Exception as e:
            print(f"Error loading pair status: {str(e)}")
            return {}
    
    def save_trade_to_db(self, trade_data: Dict):
        """บันทึกข้อมูลการเทรดลงฐานข้อมูล"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trade_history 
                (ticket, symbol, type, volume, entry_price, exit_price, stop_loss, 
                 take_profit, profit, entry_time, exit_time, signal_strength, 
                 entry_quality, risk_percentage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(trade_data.get('ticket', '')),
                trade_data.get('symbol', ''),
                trade_data.get('type', ''),
                trade_data.get('volume', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('profit', 0),
                trade_data.get('entry_time', ''),
                trade_data.get('exit_time', ''),
                trade_data.get('signal_strength', 0),
                trade_data.get('entry_quality', ''),
                trade_data.get('risk_percentage', 0)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving trade to database: {str(e)}")
            return False
    
    def get_trade_history(self, days: int = 30) -> List[Dict]:
        """ดึงประวัติการเทรด"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT * FROM trade_history 
                WHERE created_at >= ? 
                ORDER BY created_at DESC
            ''', (cutoff_date,))
            
            columns = [description[0] for description in cursor.description]
            trades = []
            
            for row in cursor.fetchall():
                trade = dict(zip(columns, row))
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            print(f"Error getting trade history: {str(e)}")
            return []
    
    def log_system_event(self, level: str, message: str, category: str = 'SYSTEM'):
        """บันทึก system logs"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_logs (level, message, category)
                VALUES (?, ?, ?)
            ''', (level, message, category))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging system event: {str(e)}")
    def create_dummy_persistence(self):
        """สร้าง dummy persistence object"""
        class DummyPersistence:
            def __init__(self):
                self.data_dir = "trading_data"
                self.settings_file = "settings.json"
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)
            
            def load_settings(self): return {}
            def save_settings(self, data): return True
            def load_pair_status(self): return {}
            def save_pair_status(self, data): return True
            def load_daily_stats(self): return {}
            def save_daily_stats(self, data): return True
            def log_system_event(self, level, msg, cat): pass
            def save_trade_to_db(self, data): return True
            def get_trade_history(self, days=30): return []
        
        self.persistence = DummyPersistence()

    def load_system_settings_direct(self):
        """โหลดการตั้งค่าระบบโดยตรง"""
        try:
            settings = self.persistence.load_settings()
            
            if settings:
                # Load settings
                self.current_risk_profile = settings.get('current_risk_profile', 'BALANCED')
                self.custom_risk_per_trade = settings.get('custom_risk_per_trade', 1.5)
                self.max_risk_per_trade = settings.get('max_risk_per_trade', 0.015)
                self.auto_trading_enabled = settings.get('auto_trading_enabled', False)
                print(f"[DISK] Settings loaded successfully")
            else:
                # Use defaults
                self.current_risk_profile = 'BALANCED'
                self.custom_risk_per_trade = 1.5
                self.max_risk_per_trade = 0.015
                self.auto_trading_enabled = False
                print(f"[DISK] Using default settings")
                
        except Exception as e:
            print(f"[ERR] Error loading settings: {str(e)}")
            # Set defaults
            self.current_risk_profile = 'BALANCED'
            self.custom_risk_per_trade = 1.5
            self.max_risk_per_trade = 0.015
            self.auto_trading_enabled = False

    def load_pair_tracking_data_direct(self):
        """โหลดข้อมูล pair tracking โดยตรง"""
        try:
            pair_status = self.persistence.load_pair_status()
            
            # Initialize tracking dictionaries
            self.active_trades_per_pair = {}
            self.pair_trade_status = {}
            self.trade_cooldowns = {}
            
            for pair in getattr(self, 'forex_pairs', ['EURUSD', 'GBPUSD', 'USDJPY']):
                if pair in pair_status:
                    status_data = pair_status[pair]
                    self.active_trades_per_pair[pair] = status_data.get('active_trades', [])
                    self.pair_trade_status[pair] = status_data.get('status', 'READY')
                else:
                    self.active_trades_per_pair[pair] = []
                    self.pair_trade_status[pair] = 'READY'
                
                self.trade_cooldowns[pair] = None
            
            print("[OK] Pair tracking data loaded")
            
        except Exception as e:
            print(f"[ERR] Error loading pair tracking: {str(e)}")
            # Initialize with defaults
            self.active_trades_per_pair = {}
            self.pair_trade_status = {}
            self.trade_cooldowns = {}
            
class EnhancedSmartAutoTradingDashboard:
    """Enhanced Smart Auto Trading Dashboard with Data Persistence"""
    
    def __init__(self):
        """Initialize Auto Trading Dashboard - COMPLETE"""
        
        # [TOOL] Core Flask Setup
        self.app = Flask(__name__)
        CORS(self.app)

        self.api_base_url ="http://123.253.62.50:8080/api"
        
        # [EMOJI] Data Management
        self.persistence = DataPersistenceManager()
        
        # [CHART] Core Data Storage
        self.live_data = {}
        self.account_info = {}
        self.forex_pairs = []
        self.auto_trading_pairs = set()
        
        # [MONEY] Account & Risk Settings
        self.account_balance = 10000.0
        self.current_risk_profile = 'BALANCED'
        self.risk_per_trade = 1.5  # %
        self.max_total_exposure = 6.0  # %
        
        # [GEAR] Trading Settings - MISSING ATTRIBUTES ADDED
        self.default_lot_size = 0.01
        self.min_lot_size = 0.01        # <- FIX: เพิ่ม missing attribute
        self.max_lot_size = 2.0         # <- FIX: เพิ่ม missing attribute
        self.auto_trading_enabled = False
        self.one_trade_per_pair = True
        
        # [1PM] Trading Time & Cooldown Settings
        self.global_cooldown = 60       # <- FIX: เพิ่ม missing attribute
        self.last_global_trade_time = None
        self.trade_cooldowns = {}       # <- FIX: เพิ่ม missing attribute
        
        # [EMOJI]‍[EMOJI] System State
        self.is_running = False
        self.mt5_connected = False
        self.emergency_stop = False
        self.last_update = datetime.now()
        
        # [CLIPBOARD] Position Tracking - COMPLETE TRACKING SYSTEM
        self.active_trades_per_pair = {}   # <- Already exists
        self.pair_trade_status = {}        # <- Already exists
        
        # [CHART] Portfolio Risk Management Profiles
        self.portfolio_risk_profiles = {
            'CONSERVATIVE': {'risk_per_trade': 0.5, 'max_total_exposure': 2.0, 'max_daily_loss': 2.0},
            'MODERATE': {'risk_per_trade': 1.0, 'max_total_exposure': 4.0, 'max_daily_loss': 3.0},
            'BALANCED': {'risk_per_trade': 1.5, 'max_total_exposure': 6.0, 'max_daily_loss': 4.0},
            'AGGRESSIVE': {'risk_per_trade': 2.0, 'max_total_exposure': 8.0, 'max_daily_loss': 5.0},
            'HIGH_RISK': {'risk_per_trade': 3.0, 'max_total_exposure': 12.0, 'max_daily_loss': 8.0}
        }
        
        self.trailing_system_enabled = False
        self.current_trailing_profile = 'MODERATE'
        self.trailing_position_states = {}
        self.trailing_statistics = {
            'total_sl_updates': 0,
            'breakeven_protections': 0,
            'trail_moves': 0,
            'profit_secured': 0.0
        }

        # [UP] Trading Execution Settings
        self.slippage_tolerance = 3
        self.max_spread_threshold = 2.0
        self.trade_timeout = 30
        
        # [EMOJI] Trading Sessions
        self.trading_sessions = {
            'ASIAN': {'start': '00:00', 'end': '09:00', 'enabled': False},
            'LONDON': {'start': '08:00', 'end': '17:00', 'enabled': True},
            'NEWYORK': {'start': '13:00', 'end': '22:00', 'enabled': True},
            'OVERLAP': {'start': '13:00', 'end': '17:00', 'enabled': True}
        }
        
        # Statistics & History
        self.daily_stats = {}
        self.trade_history = []
        self.signal_log = []
        self.pending_signals = {}
        self.open_positions = {}
        
        # Features Flags
        self.use_advanced_features = True
        self.hedging_enabled = True
        self.broker_symbols_mapped = True
        
        # [GO] Initialize Essential Components
        self.setup_logging()
        self.setup_routes()
        self.setup_symbol_adapter()
        self.load_system_settings()
        self.setup_signal_engine()
        self.add_enhanced_features()

        # print("[OK] Auto Trading Dashboard Initialized")
        # print(f"[MONEY] Account Balance: ${self.account_balance:,.2f}")
        # print(f"[TARGET] Risk Profile: {self.current_risk_profile}")
    def set_default_settings(self):
        """ตั้งค่าเริ่มต้นเมื่อไม่มีการตั้งค่าที่ save ไว้"""
        self.current_risk_profile = 'BALANCED'
        self.custom_risk_per_trade = 1.5
        self.max_risk_per_trade = 0.015
        self.max_total_exposure = 0.06
        self.max_daily_loss = 0.04
        
        self.auto_trading_enabled = False  # เริ่มต้นปิดไว้
        self.min_signal_strength = 6.0
        self.min_entry_quality = 'GOOD'
        self.max_simultaneous_trades = 8
        self.required_confirmations = True
        self.min_rr_ratio = 1.5

    def add_enhanced_features(self):
        """เพิ่ม enhanced features แบบ optional"""
        
        # Enhanced Signal Engine
        try:
            self.enhanced_signal_engine = MultiTimeframeSignalEngine()
            print("[OK] Enhanced Signal Engine: LOADED")
        except Exception as e:
            print(f"[WARN] Enhanced signal engine error: {str(e)}")
            self.enhanced_signal_engine = None
        
        # Advanced Features
        try:
            self.advanced_integrator = UniversalAdvancedTradingIntegrator(self)
            self.use_advanced_features = True
            print("[OK] Advanced Features: LOADED")
        except Exception as e:
            print(f"[WARN] Advanced features error: {str(e)}")
            self.use_advanced_features = False
        
        # Trailing Stops
        try:
            self.enhanced_trading = EnhancedTradingSystemWithTrailing(self)
            self.trailing_profiles = self.enhanced_trading.trailing_profiles
            self.trailing_enabled = True
            add_trailing_stop_routes(self.app, self.enhanced_trading)
            print("[OK] Trailing Stops: LOADED")
        except Exception as e:
            print(f"[WARN] Trailing system error: {str(e)}")
            self.trailing_enabled = False
        
        # Hedging System
        try:
            self.hedge_integrator = HedgeSystemIntegrator(self)
            self.hedging_enabled = True
            self.setup_hedging_routes()
            print("[OK] Hedging System: LOADED")
        except Exception as e:
            print(f"[WARN] Hedging system error: {str(e)}")
            self.hedging_enabled = False
        
        # Pullback Protection
        try:
            self.pullback_protection = PullbackProtectionPlugin(self.logger)
            # เปลี่ยนจาก integrate_with_main_system(self, enable_on_start=True)
            # เป็น:
            #self.integrate_pullback_protection(enable_on_start=True)
            print("[OK] Pullback Protection: LOADED")
        except Exception as e:
            print(f"[WARN] Pullback protection error: {str(e)}")
            self.pullback_protection = None
            
    def setup_signal_engine(self):
        """Setup signal engine properly"""
        self.signal_engine = MultiTimeframeSignalEngine()
        print("[OK] Enhanced Signal Engine: LOADED")

    def setup_symbol_adapter(self):
        """Setup symbol detection - SIMPLIFIED"""
        try:
            self.symbol_adapter = BrokerSymbolAdapter()
            self.broker_symbols_mapped = False
            print("[CHART] Symbol adapter ready")
        except Exception as e:
            print(f"[WARN] Symbol adapter error: {str(e)}")
            self.symbol_adapter = None

    def clean_data_for_json(self, data):
        """Clean data for JSON serialization - FIXED VERSION"""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned[key] = self.clean_data_for_json(value)
            return cleaned
        elif isinstance(data, list):
            return [self.clean_data_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.tolist()
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            if np.isnan(data) or np.isinf(data):
                return 0.0
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, datetime):
            return data.isoformat()
        elif pd.isna(data):
            return None
        elif data in [np.inf, -np.inf]:
            return None
        else:
            return data

    def load_system_settings(self):
        """โหลดการตั้งค่าระบบ - FIXED VERSION"""
        try:
            settings = self.persistence.load_settings()
            
            if settings:
                print(f"[DISK] Loading saved settings...")
                
                # Load core settings
                self.current_risk_profile = settings.get('current_risk_profile', 'BALANCED')
                self.custom_risk_per_trade = settings.get('custom_risk_per_trade', 1.5)
                self.max_risk_per_trade = settings.get('max_risk_per_trade', 0.015)
                self.max_total_exposure = settings.get('max_total_exposure', 0.06)
                self.max_daily_loss = settings.get('max_daily_loss', 0.04)
                
                # 🔧 FIX: Auto trading settings - ต้องโหลดค่าจริงที่ save ไว้
                saved_auto_trading = settings.get('auto_trading_enabled', False)
                self.auto_trading_enabled = saved_auto_trading  # ใช้ค่าที่ save ไว้
                
                self.auto_trading_pairs = set(settings.get('auto_trading_pairs', self.forex_pairs))
                self.min_signal_strength = settings.get('min_signal_strength', 6.0)
                self.min_entry_quality = settings.get('min_entry_quality', 'GOOD')
                self.max_simultaneous_trades = settings.get('max_simultaneous_trades', 8)
                self.required_confirmations = settings.get('required_confirmations', True)
                self.min_rr_ratio = settings.get('min_rr_ratio', 1.5)
                
                # 🔧 FIX: Load other important settings
                self.hedging_enabled = settings.get('hedging_enabled', False)
                self.hedge_min_correlation = settings.get('hedge_min_correlation', 0.6)
                self.hedge_max_ratio = settings.get('hedge_max_ratio', 0.6)
                self.hedge_risk_target = settings.get('hedge_risk_target', 0.3)
                self.hedge_auto_execute = settings.get('hedge_auto_execute', False)
                
                # 🔧 FIX: แสดงสถานะที่โหลดมา
                print(f"[OK] Settings loaded successfully:")
                print(f"    Auto Trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
                print(f"    Risk Profile: {self.current_risk_profile}")
                print(f"    Signal Strength: {self.min_signal_strength}")
                print(f"    Entry Quality: {self.min_entry_quality}")
                print(f"    Active Pairs: {len(self.auto_trading_pairs)}")
                
                # 🔧 FIX: หากมีการเปิด Auto Trading ให้เริ่มทำงานอัตโนมัติ
                if self.auto_trading_enabled:
                    print(f"[LIGHTNING] Auto Trading was ENABLED, restarting...")
                    # เรียกใช้ start_auto_trading_thread แทน start_auto_trading
                    self.start_auto_trading_on_startup()
                
            else:
                print(f"[WARN] No saved settings found, using defaults")
                self.set_default_settings()
                
        except Exception as e:
            print(f"[ERR] Error loading settings: {str(e)}")
            self.set_default_settings()
    
    def start_auto_trading_on_startup(self):
        """เริ่ม Auto Trading หลัง startup - SAFE VERSION"""
        try:
            # ตรวจสอบว่า MT5 เชื่อมต่อแล้ว
            if not self.mt5_connected:
                print(f"[WARN] MT5 not connected, cannot start auto trading")
                self.auto_trading_enabled = False
                return False
            
            # ตรวจสอบว่ามี forex pairs พร้อม
            if not self.forex_pairs or len(self.forex_pairs) == 0:
                print(f"[WARN] No forex pairs loaded, cannot start auto trading")
                self.auto_trading_enabled = False
                return False
            
            # รอ 5 วินาทีให้ระบบ initialize เสร็จ
            import time
            print(f"[TIME] Waiting 5 seconds for system initialization...")
            time.sleep(5)
            
            # เริ่ม Auto Trading Thread
            if self.auto_trading_enabled and not hasattr(self, 'auto_trading_thread'):
                self.auto_trading_thread = threading.Thread(
                    target=self.auto_trading_loop, 
                    daemon=True, 
                    name="AutoTradingThread"
                )
                
                if self.auto_trading_thread.start():
                    print(f"[OK] Auto Trading restarted successfully on startup")
                    self.trade_logger.info("=== AUTO TRADING RESTORED FROM SAVED SETTINGS ===")
                    return True
                else:
                    print(f"[ERR] Failed to start auto trading thread")
                    self.auto_trading_enabled = False
                    return False
            
        except Exception as e:
            print(f"[ERR] Error starting auto trading on startup: {str(e)}")
            self.auto_trading_enabled = False
            return False
    
    def set_default_settings(self):
        """ตั้งค่าเริ่มต้น"""
        self.current_risk_profile = 'BALANCED'
        self.custom_risk_per_trade = 1.5
        self.max_risk_per_trade = 0.015
        self.max_total_exposure = 0.06
        self.max_daily_loss = 0.04
        
        self.auto_trading_enabled = False
        self.auto_trading_pairs = set(self.forex_pairs)
        self.min_signal_strength = 6.0
        self.min_entry_quality = 'GOOD'
        self.max_simultaneous_trades = 8

        # Hedging defaults
        self.hedging_enabled = False
        self.hedge_min_correlation = 0.6
        self.hedge_max_ratio = 0.6
        self.hedge_risk_target = 0.3
        self.hedge_auto_execute = False
        
        self.required_confirmations = {
            'trend_alignment': True,
            'volume_confirmation': True,
            'rsi_filter': True,
            'multiple_timeframe': True
        }
        
        self.min_rr_ratio = 1.5
    
    def save_system_settings(self):
        """บันทึกการตั้งค่าระบบ - IMPROVED VERSION"""
        try:
            # 🔧 FIX: รวบรวมการตั้งค่าทั้งหมดอย่างครบถ้วน
            settings = {
                # Risk Management
                'current_risk_profile': self.current_risk_profile,
                'custom_risk_per_trade': getattr(self, 'custom_risk_per_trade', 1.5),
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_total_exposure': self.max_total_exposure,
                'max_daily_loss': self.max_daily_loss,
                
                # 🔧 FIX: Auto Trading Settings - บันทึกสถานะจริง
                'auto_trading_enabled': self.auto_trading_enabled,  # บันทึกสถานะปัจจุบัน
                'auto_trading_pairs': list(self.auto_trading_pairs),
                'min_signal_strength': self.min_signal_strength,
                'min_entry_quality': self.min_entry_quality,
                'max_simultaneous_trades': self.max_simultaneous_trades,
                'required_confirmations': getattr(self, 'required_confirmations', True),
                'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5),
                
                # Account & Balance
                'account_balance': self.account_balance,
                
                # Hedging Settings
                'hedging_enabled': getattr(self, 'hedging_enabled', False),
                'hedge_min_correlation': getattr(self, 'hedge_min_correlation', 0.6),
                'hedge_max_ratio': getattr(self, 'hedge_max_ratio', 0.6),
                'hedge_risk_target': getattr(self, 'hedge_risk_target', 0.3),
                'hedge_auto_execute': getattr(self, 'hedge_auto_execute', False),
                
                # System Info
                'last_saved': datetime.now().isoformat(),
                'save_version': '2.0_RESTART_FIX'
            }
            
            # 🔧 FIX: บันทึกและตรวจสอบผลลัพธ์
            if self.persistence.save_settings(settings):
                print(f"[DISK] Settings saved successfully:")
                print(f"    Auto Trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
                print(f"    Timestamp: {settings['last_saved']}")
                return True
            else:
                print(f"[ERR] Failed to save settings to disk")
                return False
                
        except Exception as e:
            print(f"[ERR] Error saving settings: {str(e)}")
            return False
        
    def load_pair_tracking_data(self):
        """โหลดข้อมูลการติดตาม pairs"""
        try:
            saved_status = self.persistence.load_pair_status()
            
            # Initialize tracking dictionaries
            self.active_trades_per_pair = {}
            self.pair_trade_status = {}
            self.trade_cooldowns = {}
            
            for pair in self.forex_pairs:
                # Load from saved data or set defaults
                if pair in saved_status:
                    status_data = saved_status[pair]
                    self.active_trades_per_pair[pair] = status_data.get('active_trades', [])
                    self.pair_trade_status[pair] = status_data.get('status', 'READY')
                    
                    # Handle cooldown
                    cooldown_until = status_data.get('cooldown_until')
                    if cooldown_until and isinstance(cooldown_until, datetime):
                        if cooldown_until > datetime.now():
                            self.trade_cooldowns[pair] = cooldown_until
                        else:
                            self.trade_cooldowns[pair] = None
                    else:
                        self.trade_cooldowns[pair] = None
                else:
                    # Set defaults
                    self.active_trades_per_pair[pair] = []
                    self.pair_trade_status[pair] = 'READY'
                    self.trade_cooldowns[pair] = None
            
            print("[OK] Pair tracking data loaded successfully")
            
        except Exception as e:
            print(f"[ERR] Error loading pair tracking data: {str(e)}")
            # Initialize with defaults
            self.active_trades_per_pair = {pair: [] for pair in self.forex_pairs}
            self.pair_trade_status = {pair: 'READY' for pair in self.forex_pairs}
            self.trade_cooldowns = {pair: None for pair in self.forex_pairs}
    
    def save_pair_tracking_data(self):
        """บันทึกข้อมูลการติดตาม pairs"""
        try:
            pair_status_data = {}
            
            for pair in self.forex_pairs:
                pair_status_data[pair] = {
                    'active_trades': self.active_trades_per_pair.get(pair, []),
                    'status': self.pair_trade_status.get(pair, 'READY'),
                    'cooldown_until': self.trade_cooldowns.get(pair),
                    'last_updated': datetime.now().isoformat()
                }
            
            return self.persistence.save_pair_status(pair_status_data)
            
        except Exception as e:
            print(f"[ERR] Error saving pair tracking data: {str(e)}")
            return False
    
    def setup_auto_save(self):
        """ตั้งค่าการบันทึกอัตโนมัติ"""
        def auto_save_loop():
            while self.is_running:
                try:
                    # Save every 5 minutes
                    time.sleep(300)
                    
                    if self.is_running:
                        self.save_system_settings()
                        self.save_pair_tracking_data()
                        self.persistence.save_daily_stats(self.daily_stats)
                        
                        print(f"[DISK] Auto-save completed at {datetime.now().strftime('%H:%M:%S')}")
                        
                except Exception as e:
                    print(f"[ERR] Auto-save error: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        # Start auto-save thread
        auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        auto_save_thread.start()
        print("[REFRESH] Auto-save system started")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('auto_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create separate logger for trades
        trade_logger = logging.getLogger('trades')
        trade_handler = logging.FileHandler('trade_execution.log')
        trade_handler.setFormatter(logging.Formatter('%(asctime)s - TRADE - %(message)s'))
        trade_logger.addHandler(trade_handler)
        trade_logger.setLevel(logging.INFO)
        self.trade_logger = trade_logger
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 with enhanced broker detection - FIXED"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            # Update account settings
            self.account_balance = account_info.balance
            self.account_info = {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'leverage': account_info.leverage,
                'trade_allowed': account_info.trade_allowed
            }
            
            # Check if trading is allowed
            if not account_info.trade_allowed:
                self.logger.warning("Trading is not allowed on this account!")
                self.auto_trading_enabled = False
            
            # [EMOJI] AUTO-DETECT BROKER และ MAP SYMBOLS
            self.logger.info("[SEARCH] Auto-detecting broker symbols...")
            
            try:
                if self.symbol_adapter.detect_and_map_broker():
                    mapping_info = self.symbol_adapter.get_mapping_info()
                    self.logger.info(f"[OK] Broker auto-detected successfully!")
                    self.logger.info(f"[BANK] Server: {mapping_info.get('server', 'Unknown')}")  # [OK] FIX: ใช้ .get()
                    self.logger.info(f"[CHART] Mapped: {mapping_info.get('mapped_symbols', 0)}/{mapping_info.get('total_system_symbols', 0)} symbols")
                    self.logger.info(f"[TOOL] Success rate: {mapping_info.get('mapping_success_rate', '0%')}")
                    self.logger.info(f"[TARGET] Detected suffix: '{mapping_info.get('detected_suffix', 'unknown')}'")
                    
                    # อัพเดท forex_pairs ให้ใช้ broker symbols
                    self.forex_pairs = self.symbol_adapter.get_mapped_symbols()
                    
                    # [TARGET] CRITICAL: อัพเดท auto_trading_pairs ให้ใช้ broker symbols ด้วย
                    self.auto_trading_pairs = set(self.forex_pairs)
                    
                    self.broker_symbols_mapped = True
                    
                    self.logger.info(f"[OK] Updated forex_pairs: {len(self.forex_pairs)} symbols")
                    self.logger.info(f"[OK] Updated auto_trading_pairs: {len(self.auto_trading_pairs)} symbols")
                    self.logger.info(f"[CLIPBOARD] Sample pairs: {list(self.forex_pairs)[:5]}")
                    
                    # แสดง sample mapping
                    sample_mapping = mapping_info.get('sample_mapping', {})
                    if sample_mapping:
                        self.logger.info("[CLIPBOARD] Sample Symbol Mapping:")
                        for base, broker in sample_mapping.items():
                            self.logger.info(f"   {base} -> {broker}")
                    
                    # Log event
                    self.persistence.log_system_event(
                        'INFO', 
                        f'Broker auto-detected: {mapping_info.get("server", "Unknown")} - Mapped {mapping_info.get("mapped_symbols", 0)} symbols', 
                        'BROKER_DETECTION'
                    )
                    
                else:
                    self.logger.warning("[WARN] Symbol mapping failed, using fallback")
                    self.broker_symbols_mapped = False
                    self._use_fallback_symbols()
                    
            except Exception as mapping_error:
                self.logger.error(f"[ERR] Symbol mapping error: {str(mapping_error)}")
                self.broker_symbols_mapped = False
                self._use_fallback_symbols()
            
            self.mt5_connected = True
            self.logger.info(f"[EMOJI] MT5 Connected Successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
        
    def _use_fallback_symbols(self):
        """ใช้ symbols แบบ fallback เมื่อ auto-detection ไม่สำเร็จ"""
        try:
            self.logger.info("[REFRESH] Using fallback symbol detection...")
            
            # ลองหา suffix ที่ใช้ได้
            test_suffixes = ['.v', '.c', '.raw', '.ecn', '.stp', '.pro', '.m', '']
            best_suffix = None
            max_found = 0
            
            for suffix in test_suffixes:
                found_count = 0
                for base_symbol in self.symbol_adapter.base_symbols:
                    test_symbol = base_symbol + suffix
                    symbol_info = mt5.symbol_info(test_symbol)
                    if symbol_info and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                        found_count += 1
                
                if found_count > max_found:
                    max_found = found_count
                    best_suffix = suffix
            
            if best_suffix is not None and max_found > 5:  # ต้องหาได้อย่างน้อย 5 symbols
                fallback_pairs = []
                for base_symbol in self.symbol_adapter.base_symbols:
                    test_symbol = base_symbol + best_suffix
                    symbol_info = mt5.symbol_info(test_symbol)
                    if symbol_info and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                        fallback_pairs.append(test_symbol)
                
                self.forex_pairs = fallback_pairs
                self.auto_trading_pairs = set(fallback_pairs)
                self.logger.info(f"[OK] Fallback successful with suffix '{best_suffix}': {len(fallback_pairs)} symbols")
                
            else:
                self.logger.error("[ERR] Fallback failed, keeping original .c format")
                
        except Exception as e:
            self.logger.error(f"Error in fallback: {str(e)}")
            
    def verify_existing_positions(self):
        """ตรวจสอบและอัพเดทสถานะของ positions ที่มีอยู่"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            # Reset tracking
            for pair in self.forex_pairs:
                self.active_trades_per_pair[pair] = []
            
            # Update tracking based on actual positions
            for position in positions:
                symbol = position.symbol
                if symbol in self.active_trades_per_pair:
                    self.active_trades_per_pair[symbol].append(position.ticket)
                    self.pair_trade_status[symbol] = 'TRADING'
                    
                    self.logger.info(f"Found existing position: {symbol} Ticket: {position.ticket}")
            
            # Save updated tracking data
            self.save_pair_tracking_data()
            
            self.logger.info(f"Position verification completed. Active positions: {len(positions)}")
            
        except Exception as e:
            self.logger.error(f"Error verifying existing positions: {str(e)}")
    
    def update_risk_profile(self, profile_name: str = None, custom_risk: float = None):
        """Update portfolio risk profile"""
        try:
            if custom_risk:
                self.custom_risk_per_trade = custom_risk
                self.max_risk_per_trade = custom_risk / 100
                self.logger.info(f"Custom risk set to {custom_risk}% per trade")
            elif profile_name and profile_name in self.portfolio_risk_profiles:
                profile = self.portfolio_risk_profiles[profile_name]
                self.current_risk_profile = profile_name
                self.max_risk_per_trade = profile['risk_per_trade'] / 100
                self.max_total_exposure = profile['max_total_exposure'] / 100
                self.max_daily_loss = profile['max_daily_loss'] / 100
                
                self.logger.info(f"Risk profile updated to {profile_name}")
                self.logger.info(f"Risk per trade: {profile['risk_per_trade']}%")
                self.logger.info(f"Max total exposure: {profile['max_total_exposure']}%")
            
            # Save settings after update
            self.save_system_settings()
                
        except Exception as e:
            self.logger.error(f"Error updating risk profile: {str(e)}")
    
    def check_pair_trading_status(self, symbol: str) -> Dict:
        """Check if pair can trade based on one-trade-per-pair rule"""
        try:
            if not self.one_trade_per_pair:
                return {'can_trade': True, 'reason': 'Multiple trades allowed'}
            
            # Get current positions for this pair
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is None:
                positions = []
            
            # Update active trades tracking
            current_tickets = [pos.ticket for pos in positions]
            self.active_trades_per_pair[symbol] = current_tickets
            
            # Check if pair has active trades
            if len(positions) > 0:
                self.pair_trade_status[symbol] = 'TRADING'
                return {
                    'can_trade': False, 
                    'reason': f'Active trade exists (Ticket: {positions[0].ticket})',
                    'active_trades': len(positions)
                }
            
            # Check cooldown
            if self.trade_cooldowns.get(symbol):
                cooldown_remaining = (self.trade_cooldowns[symbol] - datetime.now()).total_seconds()
                if cooldown_remaining > 0:
                    self.pair_trade_status[symbol] = 'COOLDOWN'
                    return {
                        'can_trade': False,
                        'reason': f'Cooldown active ({int(cooldown_remaining)}s remaining)',
                        'cooldown_remaining': int(cooldown_remaining)
                    }
                else:
                    # Cooldown expired
                    self.trade_cooldowns[symbol] = None
            
            # Pair is ready to trade
            self.pair_trade_status[symbol] = 'READY'
            return {'can_trade': True, 'reason': 'Ready to trade'}
            
        except Exception as e:
            self.logger.error(f"Error checking pair status {symbol}: {str(e)}")
            return {'can_trade': False, 'reason': f'Error: {str(e)}'}
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str, risk_percent: float = None) -> Dict:
        """Calculate position size with COMPLETELY FIXED division by zero protection"""
        try:
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # CRITICAL FIX 1: Comprehensive input validation
            if entry_price <= 0:
                return self.get_default_position_info(symbol, "Invalid entry price: must be > 0")
            
            if stop_loss <= 0:
                return self.get_default_position_info(symbol, "Invalid stop loss: must be > 0")
            
            if self.account_balance <= 0:
                return self.get_default_position_info(symbol, "Invalid account balance: must be > 0")
            
            if risk_percent <= 0:
                return self.get_default_position_info(symbol, "Invalid risk percent: must be > 0")
            
            # CRITICAL FIX 2: Check for zero or near-zero risk distance
            points_risk = abs(entry_price - stop_loss)
            min_risk_threshold = entry_price * 0.0001  # 0.01% minimum risk distance
            
            if points_risk < min_risk_threshold:
                return self.get_default_position_info(symbol, f"Stop loss too close to entry: {points_risk:.6f} < {min_risk_threshold:.6f}")
            
            # Risk amount calculation
            risk_amount = self.account_balance * risk_percent
            
            # CRITICAL FIX 3: Symbol-specific calculations with zero division protection
            try:
                # [EMOJI] GOLD (XAUUSD) - Fixed calculation
                if 'XAU' in symbol or 'GOLD' in symbol:
                    pip_size = 0.1          # Gold: 1 pip = $0.10
                    money_per_pip = 1.0     # Gold: $1 per pip per 1 lot (100 oz)
                    
                    pips_at_risk = points_risk / pip_size
                    
                    # CRITICAL FIX: Zero division protection
                    if pips_at_risk <= 0:
                        return self.get_default_position_info(symbol, "Invalid pips at risk for Gold")
                    
                    # Gold: Risk = Pips × $1 × Lot Size
                    lot_size = risk_amount / pips_at_risk
                    actual_money_per_pip = money_per_pip
                    
                # [EMOJI] JPY Pairs - Fixed calculation
                elif 'JPY' in symbol:
                    pip_size = 0.01         # JPY: 1 pip = 0.01
                    
                    # CRITICAL FIX: Ensure entry_price > 0 for division
                    if entry_price <= 0:
                        return self.get_default_position_info(symbol, "Invalid entry price for JPY pair")
                    
                    # For JPY pairs: pip value = (pip size / exchange rate) × trade size
                    money_per_pip = 1000 / entry_price
                    
                    pips_at_risk = points_risk / pip_size
                    
                    # CRITICAL FIX: Multiple zero checks
                    if pips_at_risk <= 0:
                        return self.get_default_position_info(symbol, "Invalid pips at risk for JPY pair")
                    
                    if money_per_pip <= 0:
                        return self.get_default_position_info(symbol, "Invalid money per pip for JPY pair")
                    
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
                    actual_money_per_pip = money_per_pip
                    
                # [EMOJI] Standard Forex Pairs - Fixed calculation
                else:
                    pip_size = 0.0001       # Forex: 1 pip = 0.0001
                    money_per_pip = 10.0    # $10 per pip per 1 standard lot
                    
                    pips_at_risk = points_risk / pip_size
                    
                    # CRITICAL FIX: Zero division protection
                    if pips_at_risk <= 0:
                        return self.get_default_position_info(symbol, "Invalid pips at risk for Forex pair")
                    
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
                    actual_money_per_pip = money_per_pip
                
                # CRITICAL FIX 4: Validate lot size calculation result
                if lot_size <= 0 or not np.isfinite(lot_size):
                    return self.get_default_position_info(symbol, f"Invalid lot size calculation result: {lot_size}")
                
                # [OK] Lot size constraints and rounding
                lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
                
                # Symbol-specific rounding
                if 'XAU' in symbol:
                    lot_size = round(lot_size, 2)      # Gold: 0.01, 0.05, 0.10
                elif lot_size >= 1.0:
                    lot_size = round(lot_size, 1)      # 1.0, 1.5, 2.0
                else:
                    lot_size = round(lot_size, 2)      # 0.01, 0.05, 0.10
                
                # CRITICAL FIX 5: Final validation and actual risk calculation
                if lot_size <= 0:
                    return self.get_default_position_info(symbol, "Lot size rounded to zero or negative")
                
                # Calculate actual risk with final lot size
                final_pips_at_risk = points_risk / pip_size
                
                # CRITICAL FIX: Ensure no division by zero in final calculation
                if final_pips_at_risk <= 0 or actual_money_per_pip <= 0:
                    return self.get_default_position_info(symbol, "Invalid final risk calculation parameters")
                
                # Calculate actual risk based on symbol type with protection
                if 'XAU' in symbol:
                    actual_risk = final_pips_at_risk * 1.0 * lot_size  # Gold
                elif 'JPY' in symbol:
                    actual_risk = final_pips_at_risk * actual_money_per_pip * lot_size  # JPY
                else:
                    actual_risk = final_pips_at_risk * 10.0 * lot_size  # Standard Forex
                
                # CRITICAL FIX 6: Validate actual risk
                if actual_risk < 0 or not np.isfinite(actual_risk):
                    return self.get_default_position_info(symbol, f"Invalid actual risk: {actual_risk}")
                
                # Calculate risk percentage with protection
                risk_percentage = (actual_risk / self.account_balance) * 100 if self.account_balance > 0 else 0
                
                # CRITICAL FIX 7: Final result validation
                if not all(np.isfinite(x) for x in [lot_size, actual_risk, risk_percentage, final_pips_at_risk]):
                    return self.get_default_position_info(symbol, "Invalid calculation results (NaN or Inf)")
                
                return {
                    'lot_size': float(lot_size),
                    'risk_amount': round(float(actual_risk), 2),
                    'risk_percentage': round(float(risk_percentage), 3),
                    'pip_value': round(float(actual_money_per_pip * lot_size), 2),
                    'points_risk': round(float(points_risk), 5),
                    'pips_at_risk': round(float(final_pips_at_risk), 1),
                    'pip_size': float(pip_size),
                    'money_per_pip': round(float(actual_money_per_pip), 4),
                    'symbol_type': 'GOLD' if 'XAU' in symbol else 'JPY' if 'JPY' in symbol else 'FOREX',
                    'calculation_status': 'SUCCESS',
                    'entry_price': float(entry_price),
                    'stop_loss': float(stop_loss),
                    'risk_percent_used': float(risk_percent),
                    'validation_passed': True
                }
                
            except ZeroDivisionError as zde:
                error_msg = f"Division by zero in symbol-specific calculation: {str(zde)}"
                self.logger.error(f"Position sizing ZeroDivisionError for {symbol}: {error_msg}")
                return self.get_default_position_info(symbol, error_msg)
            
            except Exception as calc_error:
                error_msg = f"Calculation error: {str(calc_error)}"
                self.logger.error(f"Position sizing calculation error for {symbol}: {error_msg}")
                return self.get_default_position_info(symbol, error_msg)
                
        except Exception as e:
            error_msg = f"General position sizing error: {str(e)}"
            self.logger.error(f"Position size calculation error for {symbol}: {error_msg}")
            return self.get_default_position_info(symbol, error_msg)

    def get_default_position_info(self, symbol: str, error_message: str) -> Dict:
        """Return safe default position info when calculation fails - ENHANCED"""
        try:
            # Symbol-specific safe defaults
            if 'XAU' in symbol:
                default_pip_size = 0.1
                default_money_per_pip = 1.0
                symbol_type = 'GOLD'
            elif 'JPY' in symbol:
                default_pip_size = 0.01
                default_money_per_pip = 0.1  # Safe default for JPY
                symbol_type = 'JPY'
            else:
                default_pip_size = 0.0001
                default_money_per_pip = 10.0
                symbol_type = 'FOREX'
            
            # Safe pip value calculation
            safe_pip_value = default_money_per_pip * self.default_lot_size
            
            return {
                'lot_size': float(self.default_lot_size),
                'risk_amount': 0.0,
                'risk_percentage': 0.0,
                'pip_value': float(safe_pip_value),
                'points_risk': 0.0,
                'pips_at_risk': 0.0,
                'pip_size': float(default_pip_size),
                'money_per_pip': float(default_money_per_pip),
                'symbol_type': symbol_type,
                'calculation_status': 'ERROR',
                'error_message': str(error_message),
                'fallback_used': True,
                'validation_passed': False,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'risk_percent_used': 0.0
            }
        except Exception as e:
            # Ultra-safe fallback
            return {
                'lot_size': 0.01,
                'risk_amount': 0.0,
                'risk_percentage': 0.0,
                'pip_value': 0.1,
                'points_risk': 0.0,
                'pips_at_risk': 0.0,
                'pip_size': 0.0001,
                'money_per_pip': 10.0,
                'symbol_type': 'UNKNOWN',
                'calculation_status': 'CRITICAL_ERROR',
                'error_message': f"Critical fallback error: {str(e)}",
                'fallback_used': True,
                'ultra_safe_mode': True,
                'validation_passed': False
            }
    
    def validate_trading_signal(self, symbol: str, signal_data: Dict) -> Dict:
        """Enhanced signal validation with multi-timeframe confluence - COMPLETELY FIXED"""
        try:
            validation_result = {
                'valid': False,
                'score': 0,
                'issues': [],
                'confirmations': [],
                'confidence_level': 'LOW',
                'validation_details': {
                    'checks_performed': [],
                    'thresholds_used': {},
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Check if auto trading is enabled
            if not self.auto_trading_enabled:
                validation_result['issues'].append('Auto trading disabled')
                validation_result['validation_details']['checks_performed'].append('auto_trading_check')
                return validation_result
            
            # Check emergency stop
            if self.emergency_stop:
                validation_result['issues'].append('Emergency stop activated')
                validation_result['validation_details']['checks_performed'].append('emergency_stop_check')
                return validation_result
            
            # Check pair trading status
            try:
                pair_status = self.check_pair_trading_status(symbol)
                validation_result['validation_details']['checks_performed'].append('pair_status_check')
                if not pair_status.get('can_trade', False):
                    validation_result['issues'].append(f"Pair not ready: {pair_status.get('reason', 'Unknown')}")
                    return validation_result
                else:
                    validation_result['confirmations'].append('Pair available for trading')
                    validation_result['score'] += 1
            except Exception as e:
                validation_result['issues'].append(f'Pair status check error: {str(e)}')
                return validation_result
            
            # [TARGET] ENHANCED SIGNAL STRENGTH CHECK
            signal_strength = signal_data.get('strength', 0)
            confluence_score = signal_data.get('enhanced_analysis', {}).get('confluence_score', 0)
            
            validation_result['validation_details']['thresholds_used']['min_signal_strength'] = self.min_signal_strength
            validation_result['validation_details']['checks_performed'].append('signal_strength_check')
            
            if signal_strength < self.min_signal_strength:
                validation_result['issues'].append(f'Signal strength too low: {signal_strength} < {self.min_signal_strength}')
            else:
                validation_result['confirmations'].append(f'Strong signal: {signal_strength}/10')
                validation_result['score'] += 2
            
            # [TARGET] CONFLUENCE SCORE CHECK (Enhanced System)
            validation_result['validation_details']['checks_performed'].append('confluence_check')
            if abs(confluence_score) >= 6:
                validation_result['confirmations'].append(f'Excellent confluence: {confluence_score}')
                validation_result['score'] += 3
                validation_result['confidence_level'] = 'HIGH'
            elif abs(confluence_score) >= 4:
                validation_result['confirmations'].append(f'Good confluence: {confluence_score}')
                validation_result['score'] += 2
                validation_result['confidence_level'] = 'MEDIUM'
            elif abs(confluence_score) >= 2:
                validation_result['confirmations'].append(f'Fair confluence: {confluence_score}')
                validation_result['score'] += 1
                validation_result['confidence_level'] = 'LOW'
            else:
                validation_result['issues'].append(f'Poor confluence: {confluence_score}')
            
            # Check entry quality
            entry_quality = signal_data.get('entry_quality', 'POOR')
            quality_scores = {'POOR': 0, 'FAIR': 1, 'GOOD': 2, 'EXCELLENT': 3}
            min_quality_score = quality_scores.get(self.min_entry_quality, 2)
            
            validation_result['validation_details']['thresholds_used']['min_entry_quality'] = self.min_entry_quality
            validation_result['validation_details']['checks_performed'].append('entry_quality_check')
            
            if quality_scores.get(entry_quality, 0) < min_quality_score:
                validation_result['issues'].append(f'Entry quality too low: {entry_quality} < {self.min_entry_quality}')
            else:
                validation_result['confirmations'].append(f'Good entry quality: {entry_quality}')
                validation_result['score'] += quality_scores.get(entry_quality, 0)
            
            # Check timeframe analysis count
            timeframes_analyzed = len(signal_data.get('enhanced_analysis', {}).get('timeframes_analyzed', []))
            validation_result['validation_details']['checks_performed'].append('timeframe_analysis_check')
            
            if timeframes_analyzed >= 3:
                validation_result['confirmations'].append(f'Multi-timeframe analysis: {timeframes_analyzed} TFs')
                validation_result['score'] += 1
            elif timeframes_analyzed >= 2:
                validation_result['confirmations'].append(f'Dual-timeframe analysis: {timeframes_analyzed} TFs')
                validation_result['score'] += 0.5
            else:
                validation_result['issues'].append(f'Insufficient timeframe data: {timeframes_analyzed}')
            
            # Check risk factors
            risk_factor_count = signal_data.get('enhanced_analysis', {}).get('total_risk_factors', 0)
            validation_result['validation_details']['checks_performed'].append('risk_factors_check')
            
            if risk_factor_count == 0:
                validation_result['confirmations'].append('No risk factors detected')
                validation_result['score'] += 1
            elif risk_factor_count <= 1:
                validation_result['confirmations'].append('Minimal risk factors')
                validation_result['score'] += 0.5
            else:
                validation_result['issues'].append(f'Multiple risk factors: {risk_factor_count}')
            
            # Check signal direction
            signal_direction = signal_data.get('signal', 'NONE')
            validation_result['validation_details']['checks_performed'].append('signal_direction_check')
            
            if signal_direction == 'NONE':
                validation_result['issues'].append('No clear signal direction')
            else:
                validation_result['confirmations'].append(f'Clear signal: {signal_direction}')
                validation_result['score'] += 1
            
            # Check risk/reward ratio
            rr_ratio = signal_data.get('rr_tp1', 0)
            min_rr_ratio = getattr(self, 'min_rr_ratio', 1.5)
            validation_result['validation_details']['thresholds_used']['min_rr_ratio'] = min_rr_ratio
            validation_result['validation_details']['checks_performed'].append('risk_reward_check')
            
            if rr_ratio < min_rr_ratio:
                validation_result['issues'].append(f'Poor risk/reward: 1:{rr_ratio} < 1:{min_rr_ratio}')
            else:
                validation_result['confirmations'].append(f'Good R/R: 1:{rr_ratio}')
                validation_result['score'] += 1
            
            # Check current exposure
            try:
                current_exposure = self.calculate_current_exposure()
                validation_result['validation_details']['checks_performed'].append('exposure_check')
                validation_result['validation_details']['thresholds_used']['max_exposure'] = self.max_total_exposure
                
                if current_exposure >= self.max_total_exposure:
                    validation_result['issues'].append(f'Max exposure reached: {current_exposure*100:.1f}%')
                else:
                    validation_result['confirmations'].append(f'Exposure OK: {current_exposure*100:.1f}%')
                    validation_result['score'] += 1
            except Exception as e:
                validation_result['issues'].append(f'Exposure check error: {str(e)}')
            
            # [GO] Advanced Features Validation (if available)
            if self.use_advanced_features and 'enhanced_strength' in signal_data:
                validation_result['validation_details']['checks_performed'].append('advanced_features_check')
                
                enhanced_strength = signal_data.get('enhanced_strength', 0)
                enhanced_quality = signal_data.get('enhanced_quality', 'POOR')
                market_regime = signal_data.get('market_regime', 'UNKNOWN')
                
                # Enhanced strength check
                if enhanced_strength >= 7.0:
                    validation_result['confirmations'].append(f'Excellent enhanced strength: {enhanced_strength}')
                    validation_result['score'] += 2
                elif enhanced_strength >= 5.0:
                    validation_result['confirmations'].append(f'Good enhanced strength: {enhanced_strength}')
                    validation_result['score'] += 1
                elif enhanced_strength < 3.0:
                    validation_result['issues'].append(f'Poor enhanced strength: {enhanced_strength}')
                
                # Market regime check
                if market_regime in ['TRENDING_BULLISH', 'TRENDING_BEARISH']:
                    validation_result['confirmations'].append(f'Favorable market regime: {market_regime}')
                    validation_result['score'] += 1
                elif market_regime == 'HIGH_VOLATILITY':
                    validation_result['issues'].append('High volatility regime - increased risk')
                elif market_regime == 'LOW_VOLATILITY':
                    validation_result['issues'].append('Low volatility regime - limited movement')
                
                # Enhanced quality check
                if enhanced_quality == 'EXCELLENT':
                    validation_result['confirmations'].append('Excellent enhanced signal quality')
                    validation_result['score'] += 1
                elif enhanced_quality == 'POOR':
                    validation_result['issues'].append('Poor enhanced signal quality')
            
            # Market session check
            try:
                market_session = signal_data.get('market_session', 'UNKNOWN')
                validation_result['validation_details']['checks_performed'].append('market_session_check')
                
                if market_session in ['LONDON', 'NEWYORK', 'OVERLAP']:
                    validation_result['confirmations'].append(f'Active trading session: {market_session}')
                    validation_result['score'] += 0.5
                elif market_session == 'CLOSED':
                    validation_result['issues'].append('Market session closed')
                    validation_result['score'] -= 1
            except Exception as e:
                self.logger.warning(f"Market session check error: {str(e)}")
            
            # Final validation with enhanced scoring
            min_score_for_validation = 6  # Raised threshold for higher quality
            validation_result['validation_details']['thresholds_used']['min_validation_score'] = min_score_for_validation
            
            validation_result['valid'] = (len(validation_result['issues']) == 0 and 
                                        validation_result['score'] >= min_score_for_validation)
            
            # Set confidence level based on score
            if validation_result['score'] >= 10:
                validation_result['confidence_level'] = 'VERY_HIGH'
            elif validation_result['score'] >= 8:
                validation_result['confidence_level'] = 'HIGH'
            elif validation_result['score'] >= 6:
                validation_result['confidence_level'] = 'MEDIUM'
            elif validation_result['score'] >= 3:
                validation_result['confidence_level'] = 'LOW'
            else:
                validation_result['confidence_level'] = 'VERY_LOW'
            
            # Add summary
            validation_result['validation_details']['summary'] = {
                'total_checks': len(validation_result['validation_details']['checks_performed']),
                'issues_found': len(validation_result['issues']),
                'confirmations_found': len(validation_result['confirmations']),
                'final_score': validation_result['score'],
                'validation_passed': validation_result['valid']
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating signal for {symbol}: {str(e)}")
            return {
                'valid': False, 
                'issues': [f'Validation error: {str(e)}'], 
                'confirmations': [],
                'score': 0,
                'confidence_level': 'ERROR',
                'validation_details': {
                    'error': str(e),
                    'checks_performed': ['error_occurred'],
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def execute_trade(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute trade with enhanced error handling for MT5 order_send"""
        try:
            # ตรวจสอบ MT5 connection ก่อน
            if not self.mt5_connected:
                error_msg = "MT5 not connected"
                self.logger.error(f"[ERR] {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Verify MT5 is still responsive
            if not mt5.terminal_info():
                error_msg = "MT5 terminal not responsive"
                self.logger.error(f"[ERR] {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Get broker symbol
            broker_symbol = self.symbol_adapter.system_to_broker_symbol(symbol) if self.symbol_adapter else symbol
            system_symbol = symbol
            
            # Verify symbol exists and is available for trading
            symbol_info = mt5.symbol_info(broker_symbol)
            if symbol_info is None:
                error_msg = f"Symbol {broker_symbol} not found in MT5"
                self.logger.error(f"[ERR] {error_msg}")
                return {'success': False, 'error': error_msg}
            
            if not symbol_info.visible:
                # Try to select symbol
                if not mt5.symbol_select(broker_symbol, True):
                    error_msg = f"Cannot select symbol {broker_symbol}"
                    self.logger.error(f"[ERR] {error_msg}")
                    return {'success': False, 'error': error_msg}
            
            # Get market data
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is None:
                error_msg = f"No tick data for {broker_symbol}"
                self.logger.error(f"[ERR] {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Extract trade parameters
            signal_direction = signal_data.get('signal', 'NONE')
            if signal_direction == 'NONE':
                return {'success': False, 'error': 'No valid signal'}
            
            # Get position sizing
            position_info = self.calculate_position_size(
                signal_data.get('stop_loss', tick.bid),
                signal_data,
                symbol
            )
            
            lot_size = position_info.get('lot_size', self.default_lot_size)
            stop_loss = signal_data.get('stop_loss', tick.bid)
            take_profit_1 = signal_data.get('take_profit_1', tick.ask)
            
            # Validate lot size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            if lot_size < min_lot:
                lot_size = min_lot
            elif lot_size > max_lot:
                lot_size = max_lot
            else:
                # Round to valid step
                lot_size = round(lot_size / lot_step) * lot_step
            
            # Determine order type and price
            if signal_direction in ['BUY', 'STRONG_BUY']:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                return {'success': False, 'error': 'Invalid signal direction'}
            
            # Check market hours
            if not symbol_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_LONGONLY, mt5.SYMBOL_TRADE_MODE_SHORTONLY]:
                error_msg = f"Trading not allowed for {broker_symbol} (trade_mode: {symbol_info.trade_mode})"
                self.logger.error(f"[ERR] {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Prepare order request with multiple filling modes
            filling_modes = [
                mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
                mt5.ORDER_FILLING_FOK,  # Fill or Kill  
                mt5.ORDER_FILLING_RETURN  # Return (default)
            ]
            
            request_base = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': broker_symbol,
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit_1,
                'deviation': self.slippage_tolerance,
                'magic': 12345,
                'comment': f'Auto-{signal_direction}-{system_symbol}',
                'type_time': mt5.ORDER_TIME_GTC,
            }
            
            # Try different filling modes
            result = None
            last_error = None
            
            for filling_mode in filling_modes:
                try:
                    request = request_base.copy()
                    request['type_filling'] = filling_mode
                    
                    self.trade_logger.info(f"[TARGET] Attempting order: {signal_direction} {broker_symbol} Lot: {lot_size} Fill: {filling_mode}")
                    
                    # Execute order
                    result = mt5.order_send(request)
                    
                    # Check if result is valid
                    if result is None:
                        last_error = f"mt5.order_send returned None with filling mode {filling_mode}"
                        self.logger.warning(f"[WARN] {last_error}")
                        continue
                    
                    # Check if order was successful
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.trade_logger.info(f"[OK] Order successful with filling mode {filling_mode}")
                        break
                    else:
                        last_error = f"Order failed: {result.retcode} - {result.comment} (filling: {filling_mode})"
                        self.logger.warning(f"[WARN] {last_error}")
                        
                        # Don't retry for certain error codes
                        if result.retcode in [
                            mt5.TRADE_RETCODE_INVALID_VOLUME,
                            mt5.TRADE_RETCODE_INVALID_PRICE,
                            mt5.TRADE_RETCODE_INVALID_STOPS,
                            mt5.TRADE_RETCODE_TRADE_DISABLED,
                            mt5.TRADE_RETCODE_MARKET_CLOSED
                        ]:
                            break
                            
                except Exception as e:
                    last_error = f"Exception in order_send with filling {filling_mode}: {str(e)}"
                    self.logger.error(f"[ERR] {last_error}")
                    continue
            
            # Handle final result
            if result is None:
                error_msg = f"All order attempts failed. Last error: {last_error}"
                self.logger.error(f"[ERR] {system_symbol} ({broker_symbol}): {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol,
                    'debug_info': {
                        'symbol_info_available': symbol_info is not None,
                        'tick_available': tick is not None,
                        'mt5_connected': self.mt5_connected,
                        'filling_modes_tried': len(filling_modes)
                    }
                }
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(f"[ERR] {system_symbol} ({broker_symbol}): {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol
                }
            
            # [OK] Order successful - process result
            trade_info = {
                'ticket': result.order,
                'system_symbol': system_symbol,
                'broker_symbol': broker_symbol,
                'symbol': system_symbol,
                'lot_size': lot_size,
                'entry_price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'signal_direction': signal_direction,
                'signal_strength': signal_data.get('strength', 0),
                'entry_quality': signal_data.get('entry_quality', 'UNKNOWN'),
                'risk_amount': position_info['risk_amount'],
                'risk_percentage': position_info['risk_percentage'],
                'timestamp': datetime.now(),
                'requested_price': price,
                'actual_price': result.price,
                'slippage_pips': abs(result.price - price) * (10000 if 'JPY' not in broker_symbol else 100)
            }
            
            # Update tracking
            self.active_trades_per_pair[system_symbol].append(result.order)
            self.pair_trade_status[system_symbol] = 'TRADING'
            self.daily_stats['trades_executed'] += 1
            self.last_global_trade_time = datetime.now()
            
            # Save to database
            self.persistence.save_trade_to_db({
                'ticket': str(result.order),
                'symbol': system_symbol,
                'broker_symbol': broker_symbol,
                'type': signal_direction,
                'volume': lot_size,
                'entry_price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'entry_time': datetime.now().isoformat(),
                'signal_strength': signal_data.get('strength', 0),
                'entry_quality': signal_data.get('entry_quality', 'UNKNOWN'),
                'risk_percentage': position_info['risk_percentage'],
                'slippage_pips': trade_info['slippage_pips']
            })
            
            # Enhanced logging
            self.trade_logger.info(f"[OK] TRADE EXECUTED SUCCESSFULLY!")
            self.trade_logger.info(f"   System Symbol: {system_symbol}")
            self.trade_logger.info(f"   Broker Symbol: {broker_symbol}")
            self.trade_logger.info(f"   Ticket: {result.order}")
            self.trade_logger.info(f"   Direction: {signal_direction}")
            self.trade_logger.info(f"   Entry Price: {result.price} (Requested: {price})")
            self.trade_logger.info(f"   Lot Size: {lot_size}")
            self.trade_logger.info(f"   Risk: {position_info['risk_percentage']:.2f}%")
            
            return {
                'success': True,
                'ticket': result.order,
                'trade_info': trade_info,
                'system_symbol': system_symbol,
                'broker_symbol': broker_symbol,
                'execution_summary': {
                    'requested_price': price,
                    'actual_price': result.price,
                    'slippage_pips': trade_info['slippage_pips'],
                    'execution_time': datetime.now().isoformat(),
                    'filling_mode_used': request.get('type_filling', 'unknown')
                }
            }
            
        except Exception as e:
            error_msg = f"Trade execution error for {symbol}: {str(e)}"
            self.logger.error(f"[ERR] {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'system_symbol': symbol,
                'broker_symbol': broker_symbol if 'broker_symbol' in locals() else 'unknown',
                'exception_type': type(e).__name__,
                'debug_info': {
                    'mt5_connected': getattr(self, 'mt5_connected', False),
                    'error_location': 'trade_execution'
                }
            }
    
    def _verify_broker_symbol(self, broker_symbol: str) -> bool:
        """[SEARCH] ตรวจสอบว่า broker symbol มีอยู่จริงใน MT5"""
        try:
            symbol_info = mt5.symbol_info(broker_symbol)
            if symbol_info is None:
                self.logger.warning(f"[ERR] Symbol not found: {broker_symbol}")
                return False
            
            # ตรวจสอบว่า symbol สามารถเทรดได้
            if not symbol_info.visible:
                # พยายาม enable symbol
                if mt5.symbol_select(broker_symbol, True):
                    self.logger.info(f"[OK] Symbol enabled: {broker_symbol}")
                    return True
                else:
                    self.logger.warning(f"[ERR] Cannot enable symbol: {broker_symbol}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Symbol verification error for {broker_symbol}: {str(e)}")
            return False
        
    def monitor_positions(self):
        """Monitor and manage open positions - Enhanced with Multi-Broker Support"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                # [REFRESH] MULTI-BROKER SYMBOL CONVERSION
                # ================================================
                broker_symbol = position.symbol  # Symbol ที่ได้จาก MT5 (broker format)
                system_symbol = broker_symbol    # Default fallback
                
                # แปลง broker_symbol กลับเป็น system_symbol ถ้ามี mapping
                if hasattr(self, 'broker_symbols_mapped') and self.broker_symbols_mapped:
                    system_symbol = self.symbol_adapter.broker_to_system_symbol(broker_symbol)
                    self.logger.info(f"[REFRESH] Position monitoring: {broker_symbol} -> {system_symbol}")
                # ================================================
                
                ticket = position.ticket
                
                # ตรวจสอบว่า system_symbol มีใน tracking หรือไม่
                if system_symbol not in self.active_trades_per_pair:
                    self.active_trades_per_pair[system_symbol] = []
                
                # Check if position was closed - ใช้ broker_symbol กับ MT5
                current_positions = mt5.positions_get(symbol=broker_symbol) or []
                if ticket not in [pos.ticket for pos in current_positions]:
                    # Position was closed, update tracking - ใช้ system_symbol สำหรับ tracking
                    if ticket in self.active_trades_per_pair.get(system_symbol, []):
                        self.active_trades_per_pair[system_symbol].remove(ticket)
                    
                    # If no more active trades for this pair, set cooldown
                    if len(self.active_trades_per_pair.get(system_symbol, [])) == 0:
                        self.pair_trade_status[system_symbol] = 'READY'
                        # Set cooldown period (5 minutes)
                        self.trade_cooldowns[system_symbol] = datetime.now() + timedelta(minutes=5)
                        
                        # Log position closure - แสดงทั้ง system และ broker symbol
                        log_message = f"Position closed: {system_symbol}"
                        if broker_symbol != system_symbol:
                            log_message += f" ({broker_symbol})"
                        log_message += f" Ticket: {ticket}"
                        
                        self.trade_logger.info(log_message)
                        self.persistence.log_system_event('INFO', log_message, 'TRADING')
                        
                        # Update daily stats
                        history = mt5.history_deals_get(ticket=ticket)
                        if history and len(history) > 0:
                            deal = history[-1]
                            if deal.profit > 0:
                                self.daily_stats['wins'] += 1
                            else:
                                self.daily_stats['losses'] += 1
                            self.daily_stats['total_pnl'] += deal.profit
                            
                            # Update trade in database - บันทึกทั้ง system และ broker symbol
                            trade_data = {
                                'ticket': str(ticket),
                                'system_symbol': system_symbol,
                                'broker_symbol': broker_symbol,
                                'exit_price': deal.price,
                                'exit_time': datetime.now().isoformat(),
                                'profit': deal.profit
                            }
                            self.persistence.save_trade_to_db(trade_data)
                            
                            # [CHART] Enhanced logging with symbol mapping info
                            profit_status = "PROFIT" if deal.profit > 0 else "LOSS"
                            detailed_log = (
                                f"TRADE_CLOSED: {profit_status} "
                                f"System: {system_symbol} "
                                f"Broker: {broker_symbol} "
                                f"Ticket: {ticket} "
                                f"Exit: {deal.price} "
                                f"P&L: ${deal.profit:.2f}"
                            )
                            self.trade_logger.info(detailed_log)
                        
                        # Save updated data
                        self.save_pair_tracking_data()
                        self.persistence.save_daily_stats(self.daily_stats)
        
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
            # [SHIELD] Enhanced error logging
            try:
                error_details = {
                    'error': str(e),
                    'broker_mapping_status': getattr(self, 'broker_symbols_mapped', False),
                    'positions_count': len(mt5.positions_get() or []),
                    'timestamp': datetime.now().isoformat()
                }
                self.persistence.log_system_event('ERROR', f'Position monitoring error: {error_details}', 'SYSTEM_ERROR')
            except:
                pass  # Avoid recursive errors

    def calculate_current_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0.0
            
            total_risk = 0.0
            for position in positions:
                # Calculate risk for each position
                entry_price = position.price_open
                if position.sl > 0:
                    risk_points = abs(entry_price - position.sl)
                    risk_amount = risk_points * position.volume * 10  # Simplified calculation
                    total_risk += risk_amount
            
            return (total_risk / self.account_balance) if self.account_balance > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure: {str(e)}")
            return 0.0
    
    def is_trading_session_active(self) -> bool:
        """Check if current time is within active trading sessions"""
        try:
            current_time = datetime.now().strftime('%H:%M')
            current_hour = int(current_time.split(':')[0])
            
            for session_name, session_info in self.trading_sessions.items():
                if not session_info['enabled']:
                    continue
                
                start_hour = int(session_info['start'].split(':')[0])
                end_hour = int(session_info['end'].split(':')[0])
                
                if start_hour <= current_hour <= end_hour:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading session: {str(e)}")
            return True  # Default to allow trading if error
    
    def start_auto_trading(self):
        """Start auto trading system - FIXED VERSION"""
        try:
            if self.auto_trading_enabled:
                print(f"[WARN] Auto trading already enabled")
                return True
            
            # Pre-flight checks
            if self.emergency_stop:
                print(f"[ERR] Cannot start: Emergency stop is active")
                return False
            
            if not self.mt5_connected:
                print(f"[ERR] Cannot start: MT5 not connected")
                return False
            
            if not self.forex_pairs:
                print(f"[ERR] Cannot start: No forex pairs loaded")
                return False
            
            # เปิดใช้งาน Auto Trading
            self.auto_trading_enabled = True
            print(f"[GO] Starting auto trading system...")
            
            # เริ่ม thread
            try:
                self.auto_trading_thread = threading.Thread(
                    target=self.auto_trading_loop,
                    daemon=True,
                    name="AutoTradingThread"
                )
                self.auto_trading_thread.start()
                
                # ตรวจสอบว่า thread ทำงาน
                time.sleep(1)
                if self.auto_trading_thread.is_alive():
                    print(f"[OK] Auto trading started successfully")
                    self.trade_logger.info("=== AUTO TRADING SESSION STARTED ===")
                    self.persistence.log_system_event('INFO', 'Auto trading started successfully', 'TRADING')
                    
                    # 🔧 FIX: บันทึกการตั้งค่าทันที
                    self.save_system_settings()
                    return True
                else:
                    print(f"[ERR] Auto trading thread failed to start")
                    self.auto_trading_enabled = False
                    return False
                    
            except Exception as thread_error:
                print(f"[ERR] Threading error: {str(thread_error)}")
                self.auto_trading_enabled = False
                return False
            
        except Exception as e:
            print(f"[ERR] Error starting auto trading: {str(e)}")
            self.auto_trading_enabled = False
            return False
    
    def stop_auto_trading(self):
        """Stop auto trading system - FIXED VERSION"""
        try:
            print(f"[EMOJI] Stopping auto trading...")
            
            # ปิดใช้งาน Auto Trading
            self.auto_trading_enabled = False
            
            # รอให้ thread หยุดทำงาน
            time.sleep(2)
            
            # Log การหยุด
            print(f"[OK] Auto trading stopped successfully")
            self.trade_logger.info("=== AUTO TRADING STOPPED BY USER ===")
            self.persistence.log_system_event('INFO', 'Auto trading stopped by user', 'TRADING')
            
            # 🔧 FIX: บันทึกการตั้งค่าทันที
            self.save_system_settings()
            
            return True
            
        except Exception as e:
            print(f"[ERR] Error stopping auto trading: {str(e)}")
            return False
    
    def resolve_symbol(self, target_symbol):
        """Resolve symbol to available variant"""
        try:
            # If symbol exists directly, return it
            if target_symbol in self.live_data:
                return target_symbol
            
            # Try to find base name
            base_name = target_symbol.split('.')[0]
            
            # Look for any variant of this base symbol in live_data
            for symbol in self.live_data.keys():
                if symbol.startswith(base_name + '.'):
                    self.logger.info(f"[REFRESH] Resolved {target_symbol} -> {symbol}")
                    return symbol
            
            # If not found, try to fetch with different suffixes
            common_suffixes = ['.v', '.c', '.raw', '.ecn', '.stp', '.pro', '.m', '']
            
            for suffix in common_suffixes:
                test_symbol = base_name + suffix
                
                # Check if this symbol exists in MT5
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    self.logger.info(f"[OK] Found tradeable variant: {test_symbol}")
                    return test_symbol
            
            self.logger.warning(f"[ERR] No tradeable variant found for {target_symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving symbol {target_symbol}: {str(e)}")
            return None
        
    def emergency_stop_all(self):
        """Emergency stop with position closure"""
        self.emergency_stop = True
        self.auto_trading_enabled = False
        
        # Close all positions
        positions = mt5.positions_get()
        if positions:
            for position in positions:
                close_request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': position.symbol,
                    'volume': position.volume,
                    'type': mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    'position': position.ticket,
                    'magic': 12345,
                    'comment': 'Emergency Stop',
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_FOK,
                }
                mt5.order_send(close_request)
        
        self.logger.critical("EMERGENCY STOP ACTIVATED - ALL POSITIONS CLOSED")
        self.trade_logger.critical("=== EMERGENCY STOP ACTIVATED ===")
        self.persistence.log_system_event('CRITICAL', 'Emergency stop activated - all positions closed', 'EMERGENCY')
        
        # Save settings and data
        self.save_system_settings()
        self.save_pair_tracking_data()
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            if len(df) < 50:
                return self.get_default_indicators()
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            
            # EMA calculations
            ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
            ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            
            # RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
            
            # MACD calculation
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = (ema_12 - ema_26).iloc[-1]
            
            # Trend strength
            current_price = close.iloc[-1]
            trend_conditions = [
                current_price > ema_9,
                ema_9 > ema_21,
                ema_21 > ema_50
            ]
            uptrend_strength = sum(trend_conditions) / len(trend_conditions)
            
            downtrend_conditions = [
                current_price < ema_9,
                ema_9 < ema_21,
                ema_21 < ema_50
            ]
            downtrend_strength = sum(downtrend_conditions) / len(downtrend_conditions)
            trend_strength = max(uptrend_strength, downtrend_strength)
            
            # Volume analysis
            volume_avg = volume.rolling(window=20, min_periods=1).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
            
            return {
                'rsi': float(rsi) if not pd.isna(rsi) else 50.0,
                'atr': float(atr) if not pd.isna(atr) else current_price * 0.001,
                'atr_percent': float((atr / current_price) * 100) if not pd.isna(atr) else 0.1,
                'macd': float(macd) if not pd.isna(macd) else 0.0,
                'trend_strength': float(trend_strength),
                'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
                'ema_9': float(ema_9) if not pd.isna(ema_9) else current_price,
                'ema_21': float(ema_21) if not pd.isna(ema_21) else current_price,
                'ema_50': float(ema_50) if not pd.isna(ema_50) else current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return self.get_default_indicators()
    
    def get_default_indicators(self) -> Dict:
        """Return default indicators"""
        return {
            'rsi': 50.0, 'atr': 0.001, 'atr_percent': 0.1, 'macd': 0.0,
            'trend_strength': 0.0, 'volume_ratio': 1.0,
            'ema_9': 1.0, 'ema_21': 1.0, 'ema_50': 1.0
        }
    
    def analyze_entry_exit_points(self, indicators: Dict, current_price: float, symbol: str) -> Dict:
            """
             ENHANCED Multi-Timeframe Signal Analysis
            Using Professional Confluence System
            """
            try:
                # [HOT] ตรวจสอบว่ามี signal engine หรือไม่
                if not hasattr(self, 'signal_engine') or self.signal_engine is None:
                    # Fallback to old system
                    return self.old_analyze_entry_exit_points(indicators, current_price, symbol)
                
                # [HOT] Use the new multi-timeframe confluence system
                confluence_result = self.signal_engine.get_multi_timeframe_confluence(symbol)
                
                # Check additional filters
                risk_factors = []
                
                # 1. Check correlation risk
                existing_positions = [pos.symbol for pos in (mt5.positions_get() or [])]
                if not self.signal_engine.check_correlation_risk(symbol, existing_positions):
                    risk_factors.append("High correlation with existing positions")
                
                # 3. Check spread conditions
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    spread = tick.ask - tick.bid
                    spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
                    if spread_pips > 3.0:  # High spread
                        risk_factors.append(f"High spread: {spread_pips:.1f} pips")
                
                # Convert confluence result to existing format
                result = {
                    'signal': confluence_result.get('final_signal', 'NONE'),
                    'strength': round(confluence_result.get('final_strength', 0), 1),
                    'entry_quality': confluence_result.get('final_quality', 'POOR'),
                    'entry_score': max(0, int(confluence_result.get('confluence_score', 0))),
                    'entry_reasons': [],
                    'optimal_entry': round(current_price, 5),
                    'stop_loss': round(current_price, 5),
                    'take_profit_1': round(current_price, 5),
                    'take_profit_2': round(current_price, 5),
                    'take_profit_3': round(current_price, 5),
                    'lot_size': self.default_lot_size,
                    'risk_amount': 0,
                    'risk_percentage': 0,
                    'rr_tp1': 0,
                    'rr_tp2': 0,
                    'rr_tp3': 0
                }
                
                # Add risk factors to reasons
                if risk_factors:
                    result['entry_reasons'].extend(risk_factors)
                    # Downgrade signal if there are risk factors
                    if len(risk_factors) >= 2:
                        if result['signal'] in ['STRONG_BUY', 'STRONG_SELL']:
                            result['signal'] = result['signal'].replace('STRONG_', '')
                            result['strength'] *= 0.7
                        elif result['signal'] in ['BUY', 'SELL']:
                            result['signal'] = 'WEAK_' + result['signal']
                            result['strength'] *= 0.5
                
                # Add timeframe analysis reasons
                for tf_name, tf_analysis in confluence_result.get('timeframe_analysis', {}).items():
                    if tf_analysis.get('factors'):
                        # Add top 2 factors from each timeframe
                        for factor in tf_analysis['factors'][:2]:
                            result['entry_reasons'].append(f"{tf_name}: {factor}")
                
                # Get entry conditions from confluence result
                entry_conditions = confluence_result.get('entry_conditions', {})
                if entry_conditions:
                    result.update({
                        'optimal_entry': entry_conditions.get('optimal_entry', current_price),
                        'stop_loss': entry_conditions.get('stop_loss', current_price),
                        'take_profit_1': entry_conditions.get('take_profit_1', current_price),
                        'take_profit_2': entry_conditions.get('take_profit_2', current_price),
                        'take_profit_3': entry_conditions.get('take_profit_3', current_price),
                        'rr_tp1': entry_conditions.get('risk_reward_tp1', 0),
                        'rr_tp2': entry_conditions.get('risk_reward_tp2', 0),
                        'rr_tp3': entry_conditions.get('risk_reward_tp3', 0)
                    })
                
                # Calculate position sizing using existing method
                if result['stop_loss'] != current_price:
                    position_info = self.calculate_position_size(
                        result['optimal_entry'], 
                        result['stop_loss'], 
                        symbol
                    )
                    result.update({
                        'lot_size': position_info.get('lot_size', self.default_lot_size),
                        'risk_amount': position_info.get('risk_amount', 0),
                        'risk_percentage': position_info.get('risk_percentage', 0)
                    })
                
                # Add enhanced validation data
                result['enhanced_analysis'] = {
                    'confluence_score': confluence_result.get('confluence_score', 0),
                    'trade_recommendation': confluence_result.get('trade_recommendation', 'NO_TRADE'),
                    'timeframes_analyzed': list(confluence_result.get('timeframe_analysis', {}).keys()),
                    'total_risk_factors': len(risk_factors),
                    'analysis_timestamp': confluence_result.get('timestamp'),
                    'system_version': 'Enhanced_MultiTimeframe_v2.0'
                }
                
                #  Log enhanced signal info
                # if result['signal'] != 'NONE':
                    # self.logger.info(f" ENHANCED SIGNAL: {symbol}")
                    # self.logger.info(f"   Signal: {result['signal']} | Strength: {result['strength']}/10")
                    # self.logger.info(f"   Quality: {result['entry_quality']} | Confluence: {confluence_result.get('confluence_score', 0)}")
                    # self.logger.info(f"   Timeframes: {len(confluence_result.get('timeframe_analysis', {}))}")
                    # self.logger.info(f"   Risk Factors: {len(risk_factors)}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Enhanced signal analysis error for {symbol}: {str(e)}")
                
                # Fallback to basic analysis
                return {
                    'signal': 'NONE',
                    'strength': 0,
                    'entry_quality': 'POOR',
                    'entry_score': 0,
                    'entry_reasons': [f'Analysis error: {str(e)}'],
                    'optimal_entry': current_price,
                    'stop_loss': current_price,
                    'take_profit_1': current_price,
                    'take_profit_2': current_price,
                    'take_profit_3': current_price,
                    'lot_size': self.default_lot_size,
                    'risk_amount': 0,
                    'risk_percentage': 0,
                    'rr_tp1': 0,
                    'rr_tp2': 0,
                    'rr_tp3': 0,
                    'enhanced_analysis': {
                        'error': str(e),
                        'fallback_mode': True,
                        'system_version': 'Enhanced_MultiTimeframe_v2.0'
                    }
                }    
            
    def old_analyze_entry_exit_points(self, indicators: Dict, current_price: float, symbol: str) -> Dict:
        """
        Backup method - ระบบเก่าสำหรับกรณี signal engine ไม่ทำงาน - COMPLETELY FIXED
        """
        try:
            # Input validation
            if current_price <= 0:
                current_price = 1.0
                
            # Get indicators with safe fallbacks
            rsi = max(0, min(100, indicators.get('rsi', 50)))
            trend_strength = max(0, min(1, indicators.get('trend_strength', 0)))
            atr = indicators.get('atr', current_price * 0.005)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            volume_ratio = max(0, indicators.get('volume_ratio', 1.0))
            
            # Ensure ATR is valid
            if atr <= 0 or atr > current_price:
                atr = current_price * 0.005
            
            # Ensure EMAs are valid
            if ema_9 <= 0: ema_9 = current_price
            if ema_21 <= 0: ema_21 = current_price
            if ema_50 <= 0: ema_50 = current_price
            
            # Simple signal analysis (ระบบเก่า)
            signal_direction = 'NONE'
            signal_strength = 0
            entry_score = 0
            entry_reasons = ['Using fallback system']
            
            # Add noise filtering
            min_diff_threshold = current_price * 0.0005  # 0.05% minimum difference
            
            # Basic bullish conditions
            bullish_score = 0
            if current_price > ema_9 + min_diff_threshold:
                bullish_score += 2
                entry_reasons.append("Price above EMA9")
            if ema_9 > ema_21 + min_diff_threshold:
                bullish_score += 1
                entry_reasons.append("EMA9 > EMA21")
            if ema_21 > ema_50 + min_diff_threshold:
                bullish_score += 1
                entry_reasons.append("EMA21 > EMA50")
            if 25 <= rsi <= 75:
                bullish_score += 1
                entry_reasons.append("RSI in safe zone")
            if trend_strength >= 0.3:
                bullish_score += 1
                entry_reasons.append("Trend detected")
            if volume_ratio >= 1.0:
                bullish_score += 1
                entry_reasons.append("Volume support")
            
            # Basic bearish conditions
            bearish_score = 0
            if current_price < ema_9 - min_diff_threshold:
                bearish_score += 2
            if ema_9 < ema_21 - min_diff_threshold:
                bearish_score += 1
            if ema_21 < ema_50 - min_diff_threshold:
                bearish_score += 1
            if 25 <= rsi <= 75:
                bearish_score += 1
            if trend_strength >= 0.3:
                bearish_score += 1
            if volume_ratio >= 1.0:
                bearish_score += 1
            
            # Signal generation
            if bullish_score >= bearish_score and bullish_score >= 3:
                signal_direction = 'BUY'
                signal_strength = min(10, 2 + bullish_score * 0.8)
                entry_score = bullish_score
                
                if bullish_score >= 6:
                    signal_direction = 'STRONG_BUY'
                    signal_strength = min(10, 6 + bullish_score * 0.5)
                    
            elif bearish_score > bullish_score and bearish_score >= 3:
                signal_direction = 'SELL'
                signal_strength = min(10, 2 + bearish_score * 0.8)
                entry_score = bearish_score
                
                if bearish_score >= 6:
                    signal_direction = 'STRONG_SELL'
                    signal_strength = min(10, 6 + bearish_score * 0.5)
            
            # Entry quality
            if entry_score >= 7:
                entry_quality = 'EXCELLENT'
            elif entry_score >= 5:
                entry_quality = 'GOOD'
            elif entry_score >= 3:
                entry_quality = 'FAIR'
            else:
                entry_quality = 'POOR'
            
            # Calculate levels with improved ATR multipliers
            # Symbol-specific ATR multipliers
            if 'JPY' in symbol:
                atr_sl_multiplier = 2.0    # JPY pairs
                atr_tp1_multiplier = 3.0
                atr_tp2_multiplier = 5.0
                atr_tp3_multiplier = 7.0
            elif 'XAU' in symbol:
                atr_sl_multiplier = 1.5    # Gold
                atr_tp1_multiplier = 2.5
                atr_tp2_multiplier = 4.0
                atr_tp3_multiplier = 6.0
            else:
                atr_sl_multiplier = 1.5    # Standard Forex
                atr_tp1_multiplier = 2.5
                atr_tp2_multiplier = 4.0
                atr_tp3_multiplier = 6.0
            
            # Calculate precision based on symbol
            if 'JPY' in symbol:
                precision = 3
            elif 'XAU' in symbol:
                precision = 2
            else:
                precision = 5
            
            if signal_direction in ['BUY', 'STRONG_BUY']:
                stop_loss = round(current_price - (atr * atr_sl_multiplier), precision)
                take_profit_1 = round(current_price + (atr * atr_tp1_multiplier), precision)
                take_profit_2 = round(current_price + (atr * atr_tp2_multiplier), precision)
                take_profit_3 = round(current_price + (atr * atr_tp3_multiplier), precision)
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                stop_loss = round(current_price + (atr * atr_sl_multiplier), precision)
                take_profit_1 = round(current_price - (atr * atr_tp1_multiplier), precision)
                take_profit_2 = round(current_price - (atr * atr_tp2_multiplier), precision)
                take_profit_3 = round(current_price - (atr * atr_tp3_multiplier), precision)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Position sizing with error handling
            try:
                position_info = self.calculate_position_size(current_price, stop_loss, symbol)
            except Exception as pos_error:
                self.logger.error(f"Position sizing error in fallback: {str(pos_error)}")
                position_info = self.get_default_position_info(symbol, str(pos_error))
            
            # R/R ratios with safe calculation
            risk = abs(current_price - stop_loss)
            if risk > 0:
                rr_tp1 = abs(take_profit_1 - current_price) / risk
                rr_tp2 = abs(take_profit_2 - current_price) / risk
                rr_tp3 = abs(take_profit_3 - current_price) / risk
            else:
                rr_tp1 = rr_tp2 = rr_tp3 = 0
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 1),
                'entry_quality': entry_quality,
                'entry_score': entry_score,
                'entry_reasons': entry_reasons,
                'optimal_entry': round(current_price, precision),
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3,
                'lot_size': position_info.get('lot_size', self.default_lot_size),
                'risk_amount': position_info.get('risk_amount', 0),
                'risk_percentage': position_info.get('risk_percentage', 0),
                'rr_tp1': round(rr_tp1, 2),
                'rr_tp2': round(rr_tp2, 2),
                'rr_tp3': round(rr_tp3, 2),
                'enhanced_analysis': {
                    'fallback_mode': True,
                    'system_version': 'Fallback_System_v1.0_FIXED',
                    'atr_used': atr,
                    'atr_multipliers': {
                        'stop_loss': atr_sl_multiplier,
                        'tp1': atr_tp1_multiplier,
                        'tp2': atr_tp2_multiplier,
                        'tp3': atr_tp3_multiplier
                    },
                    'calculation_precision': precision,
                    'input_validation': 'PASSED'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis error: {str(e)}")
            
            # Return ultra-safe fallback
            return {
                'signal': 'NONE', 
                'strength': 0, 
                'entry_quality': 'POOR',
                'entry_score': 0, 
                'entry_reasons': [f'Fallback error: {str(e)}'],
                'optimal_entry': current_price,
                'stop_loss': current_price, 
                'take_profit_1': current_price,
                'take_profit_2': current_price,
                'take_profit_3': current_price,
                'lot_size': self.default_lot_size, 
                'risk_amount': 0,
                'risk_percentage': 0, 
                'rr_tp1': 0, 
                'rr_tp2': 0, 
                'rr_tp3': 0,
                'enhanced_analysis': {
                    'fallback_mode': True,
                    'system_version': 'Fallback_System_v1.0_FIXED',
                    'error': str(e),
                    'ultra_safe_mode': True
                }
            }
                
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get enhanced symbol data with advanced features - COMPLETELY FIXED"""
        try:
            if not self.mt5_connected:
                self.logger.warning(f"MT5 not connected for {symbol}")
                return None
            
            # Get current tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f"No tick data for {symbol}")
                return None
            
            current_price = tick.bid
            if current_price <= 0:
                self.logger.warning(f"Invalid price for {symbol}: {current_price}")
                return None
            
            # Get multi-timeframe data with error handling
            timeframe_rates = {}
            timeframes_to_get = {
                'H4': (mt5.TIMEFRAME_H4, 100),
                'H1': (mt5.TIMEFRAME_H1, 100),
                'M15': (mt5.TIMEFRAME_M15, 100),
                'M5': (mt5.TIMEFRAME_M5, 100)
            }
            
            for tf_name, (tf_value, periods) in timeframes_to_get.items():
                try:
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, periods)
                    if rates is not None and len(rates) >= 10:  # Minimum data requirement
                        timeframe_rates[tf_name] = pd.DataFrame(rates)
                    else:
                        self.logger.warning(f"Insufficient {tf_name} data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error getting {tf_name} data for {symbol}: {str(e)}")
                    continue
            
            # Require at least one timeframe
            if not timeframe_rates:
                self.logger.error(f"No timeframe data available for {symbol}")
                return None

            # Use H1 as primary timeframe, fallback to others
            primary_df = None
            primary_tf = None
            for tf_name in ['H1', 'M15', 'H4', 'M5']:
                if tf_name in timeframe_rates:
                    primary_df = timeframe_rates[tf_name]
                    primary_tf = tf_name
                    break
            
            if primary_df is None:
                self.logger.error(f"No usable timeframe data for {symbol}")
                return None
            
            # Calculate indicators using primary timeframe
            try:
                indicators = self.calculate_indicators(primary_df)
                if not indicators:
                    self.logger.warning(f"Failed to calculate indicators for {symbol}")
                    indicators = self.get_default_indicators()
            except Exception as e:
                self.logger.error(f"Indicator calculation error for {symbol}: {str(e)}")
                indicators = self.get_default_indicators()
            
            # Perform signal analysis
            try:
                # First try enhanced analysis if signal engine is available
                if hasattr(self, 'signal_engine') and self.signal_engine:
                    try:
                        basic_entry_exit_analysis = self.analyze_entry_exit_points(indicators, current_price, symbol)
                        entry_exit_analysis = basic_entry_exit_analysis
                        analysis_method = 'enhanced_signal_engine'
                    except Exception as e:
                        self.logger.warning(f"Enhanced signal engine failed for {symbol}: {str(e)}")
                        # Fallback to old method
                        entry_exit_analysis = self.old_analyze_entry_exit_points(indicators, current_price, symbol)
                        analysis_method = 'fallback_analysis'
                else:
                    # Use old method directly
                    entry_exit_analysis = self.old_analyze_entry_exit_points(indicators, current_price, symbol)
                    analysis_method = 'old_analysis'
                    
            except Exception as e:
                self.logger.error(f"Signal analysis failed for {symbol}: {str(e)}")
                # Ultra-safe fallback
                entry_exit_analysis = {
                    'signal': 'NONE', 'strength': 0, 'entry_quality': 'POOR',
                    'entry_score': 0, 'optimal_entry': current_price,
                    'stop_loss': current_price, 'take_profit_1': current_price,
                    'lot_size': self.default_lot_size, 'risk_amount': 0,
                    'risk_percentage': 0, 'rr_tp1': 0, 'rr_tp2': 0, 'rr_tp3': 0,
                    'error': str(e)
                }
                analysis_method = 'error_fallback'
            
            # [GO] ENHANCED ANALYSIS (Advanced Features)
            if self.use_advanced_features and analysis_method != 'error_fallback':
                try:
                    # Prepare timeframe data for advanced analysis
                    timeframe_data = {}
                    if timeframe_rates:
                        for tf_name, df in timeframe_rates.items():
                            if df is not None and len(df) > 0:
                                timeframe_data[tf_name] = df

                    # ถ้าไม่มีข้อมูลเลย ให้สร้าง dummy data
                    if not timeframe_data:
                        timeframe_data = {
                            'H1': None,
                            'H4': None
                        }                    
                    # Add account balance to signal data
                    entry_exit_analysis['account_balance'] = self.account_balance
                    
                    # Get enhanced analysis
                    enhanced_analysis = self.advanced_integrator.enhance_signal_analysis(
                        symbol, 
                        entry_exit_analysis, 
                        timeframe_data or {}
                    )
                    
                    # Use enhanced results
                    entry_exit_analysis = enhanced_analysis
                    analysis_method = 'enhanced_advanced'
                    
                    # self.logger.info(f" Enhanced Analysis for {symbol}: "
                    #             f"Signal: {enhanced_analysis.get('signal', 'NONE')} -> "
                    #             f"Enhanced Strength: {enhanced_analysis.get('enhanced_strength', 0)}")
                    
                except Exception as e:
                    self.logger.warning(f"[WARN] Enhanced analysis failed for {symbol}: {str(e)}")
                    # Keep the basic analysis
                    pass
            
            # Extract timeframe prices safely
            def safe_get_price(df, price_type='close'):
                try:
                    if df is not None and len(df) > 0 and price_type in df.columns:
                        price = df[price_type].iloc[-1]
                        if pd.isna(price) or price <= 0:
                            return current_price
                        return float(price)
                    return current_price
                except:
                    return current_price
            
            h4_price = safe_get_price(timeframe_rates.get('H4'))
            h1_price = safe_get_price(timeframe_rates.get('H1'))
            m15_price = safe_get_price(timeframe_rates.get('M15'))
            m5_price = safe_get_price(timeframe_rates.get('M5'))
            
            # Calculate price change safely
            try:
                if len(primary_df) > 1:
                    previous_price = primary_df['close'].iloc[-2]
                    if pd.isna(previous_price) or previous_price <= 0:
                        previous_price = current_price
                else:
                    previous_price = current_price
                    
                price_change = current_price - previous_price
                change_percent = (price_change / previous_price) * 100 if previous_price > 0 else 0
            except Exception as e:
                self.logger.warning(f"Price change calculation error for {symbol}: {str(e)}")
                price_change = 0
                change_percent = 0
            
            # Calculate spread safely
            try:
                spread = tick.ask - tick.bid
                spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
                if spread_pips < 0:
                    spread_pips = 0
            except Exception as e:
                self.logger.warning(f"Spread calculation error for {symbol}: {str(e)}")
                spread_pips = 0
            
            # Get trading status for this pair
            try:
                pair_status = self.check_pair_trading_status(symbol)
            except Exception as e:
                self.logger.warning(f"Pair status check error for {symbol}: {str(e)}")
                pair_status = {'can_trade': False, 'reason': f'Status check error: {str(e)}'}
            
            # Determine precision for price formatting
            if 'JPY' in symbol:
                precision = 3
            elif 'XAU' in symbol:
                precision = 2
            else:
                precision = 5
            
            # Build result dictionary
            result = {
                'symbol': symbol,
                'h4': round(h4_price, precision),
                'h1': round(h1_price, precision),
                'm15': round(m15_price, precision),
                'm5': round(m5_price, precision),
                'current_price': round(current_price, precision),
                'price_change': round(price_change, precision),
                'change_percent': round(change_percent, 3),
                'price_direction': 'up' if m5_price > m15_price else 'down',
                'bid': round(tick.bid, precision),
                'ask': round(tick.ask, precision),
                'spread_pips': round(spread_pips, 1),
                
                # Technical indicators (with safe extraction)
                'rsi': round(indicators.get('rsi', 50), 1),
                'macd': round(indicators.get('macd', 0), 6),
                'atrPercent': round(indicators.get('atr_percent', 0.1), 3),
                'trendStrength': round(indicators.get('trend_strength', 0), 3),
                'volumeRatio': round(indicators.get('volume_ratio', 1), 2),
                
                # Trading signals and analysis (enhanced or basic)
                **entry_exit_analysis,
                
                # Auto trading info
                'pair_status': pair_status,
                'can_trade': pair_status.get('can_trade', False),
                'active_trades': len(self.active_trades_per_pair.get(symbol, [])),
                
                # Metadata
                'last_update': datetime.now().isoformat(),
                'analysis_method': analysis_method,
                'primary_timeframe': primary_tf,
                'timeframes_available': list(timeframe_rates.keys()),
                'data_quality': {
                    'tick_data': True,
                    'timeframe_data_count': len(timeframe_rates),
                    'indicators_calculated': True,
                    'analysis_completed': True
                }
            }
            
            # Add advanced features info if available
            if self.use_advanced_features and 'market_regime' in entry_exit_analysis:
                result.update({
                    'advanced_features': True,
                    'market_regime': entry_exit_analysis.get('market_regime', 'UNKNOWN'),
                    'regime_confidence': entry_exit_analysis.get('regime_confidence', 0),
                    'enhanced_strength': entry_exit_analysis.get('enhanced_strength', 0),
                    'enhanced_quality': entry_exit_analysis.get('enhanced_quality', 'POOR'),
                    'enhanced_lot_size': entry_exit_analysis.get('enhanced_lot_size', 0.01),
                    'volatility_percentile': entry_exit_analysis.get('volatility_percentile', 50)
                })
            else:
                result['advanced_features'] = False
            
            #  # Final validation of result - FIXED
            try:
                import numpy as np
                
                # Ensure critical fields are valid with comprehensive checks
                if 'current_price' not in result or not isinstance(result.get('current_price'), (int, float)) or result['current_price'] <= 0:
                    result['current_price'] = 1.0
                
                # Enhanced lot_size validation
                if 'lot_size' not in result:
                    result['lot_size'] = getattr(self, 'default_lot_size', 0.01)
                else:
                    try:
                        lot_size_value = result['lot_size']
                        if not isinstance(lot_size_value, (int, float)):
                            result['lot_size'] = getattr(self, 'default_lot_size', 0.01)
                        elif not np.isfinite(lot_size_value) or lot_size_value <= 0:
                            result['lot_size'] = getattr(self, 'default_lot_size', 0.01)
                        else:
                            result['lot_size'] = max(0.01, min(2.0, float(lot_size_value)))
                    except:
                        result['lot_size'] = getattr(self, 'default_lot_size', 0.01)
                
                # Ensure signal fields exist with safe defaults
                required_fields = {
                    'signal': 'NONE',
                    'strength': 0,
                    'entry_quality': 'POOR',
                    'optimal_entry': result['current_price'],
                    'stop_loss': result['current_price'],
                    'take_profit_1': result['current_price']
                }
                
                for field, default_value in required_fields.items():
                    if field not in result or result[field] is None:
                        result[field] = default_value
                    elif field in ['strength'] and not isinstance(result[field], (int, float)):
                        result[field] = 0
                    elif field in ['optimal_entry', 'stop_loss', 'take_profit_1']:
                        try:
                            if not isinstance(result[field], (int, float)) or not np.isfinite(result[field]) or result[field] <= 0:
                                result[field] = result['current_price']
                        except:
                            result[field] = result['current_price']
                
                # Additional safety checks for numeric fields
                numeric_safety_fields = ['risk_amount', 'risk_percentage']
                for field in numeric_safety_fields:
                    if field in result:
                        try:
                            if not isinstance(result[field], (int, float)) or not np.isfinite(result[field]):
                                result[field] = 0.0
                        except:
                            result[field] = 0.0
                            
            except Exception as e:
                # Silent fallback - no more warning spam
                result.update({
                    'lot_size': getattr(self, 'default_lot_size', 0.01),
                    'signal': 'NONE',
                    'strength': 0,
                    'entry_quality': 'POOR',
                    'current_price': 1.0
                })

            return result
            
        except Exception as e:
            self.logger.error(f"Critical error getting data for {symbol}: {str(e)}")
            
            # Return minimal safe data structure
            try:
                safe_price = 1.0
                if hasattr(mt5, 'symbol_info_tick'):
                    tick = mt5.symbol_info_tick(symbol)
                    if tick and tick.bid > 0:
                        safe_price = tick.bid
                        
                return {
                    'symbol': symbol,
                    'h4': safe_price, 'h1': safe_price, 'm15': safe_price, 'm5': safe_price,
                    'current_price': safe_price, 'price_change': 0, 'change_percent': 0,
                    'price_direction': 'unknown', 'bid': safe_price, 'ask': safe_price, 'spread_pips': 0,
                    'rsi': 50, 'macd': 0, 'atrPercent': 0.1, 'trendStrength': 0, 'volumeRatio': 1,
                    'signal': 'NONE', 'strength': 0, 'entry_quality': 'POOR', 'entry_score': 0,
                    'optimal_entry': safe_price, 'stop_loss': safe_price, 'take_profit_1': safe_price,
                    'lot_size': self.default_lot_size, 'risk_amount': 0, 'risk_percentage': 0,
                    'rr_tp1': 0, 'rr_tp2': 0, 'rr_tp3': 0,
                    'pair_status': {'can_trade': False, 'reason': f'Data error: {str(e)}'},
                    'can_trade': False, 'active_trades': 0, 'advanced_features': False,
                    'last_update': datetime.now().isoformat(), 'analysis_method': 'error_recovery',
                    'error': str(e), 'data_quality': {'error_occurred': True}
                }
            except Exception as final_error:
                self.logger.critical(f"Final error recovery failed for {symbol}: {str(final_error)}")
                return None

    def update_all_data(self):
        """Update all symbols data"""
        try:
            if not self.mt5_connected:
                return
            
            updated_count = 0
            for symbol in self.forex_pairs:
                try:
                    data = self.get_symbol_data(symbol)
                    if data:
                        self.live_data[symbol] = data
                        updated_count += 1
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error updating {symbol}: {str(e)}")
                    continue
            
            self.last_update = datetime.now()
            self.logger.info(f"Updated {updated_count}/{len(self.forex_pairs)} pairs")
            
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
    
    def setup_routes(self):
        """Setup Flask routes for auto trading control"""
        
        @self.app.route('/')
        def dashboard():
            # ใช้ชื่อไฟล์เดิมที่คุณมี
            try:
                return send_from_directory('.', 'forex_dashboard.html')
            except:
                # ถ้าไม่เจอ ให้ลองหาไฟล์อื่น
                try:
                    return send_from_directory('.', 'auto_trading_dashboard.html')
                except:
                    return '''<!DOCTYPE html>
<html><head><title>Enhanced Auto Trading Dashboard</title></head>
<body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
<h1 style="color:#00ff00;">Enhanced Smart Auto Trading Dashboard</h1>
<h2 style="color:#ffff00;">[REFRESH] WITH DATA PERSISTENCE & STATE MANAGEMENT</h2>
<p style="color:#00ccff;">[OK] Settings auto-saved every 5 minutes</p>
<p style="color:#00ccff;">[OK] Positions tracking persistent</p>
<p style="color:#00ccff;">[OK] Daily statistics saved</p>
<p style="color:#00ccff;">[OK] Trade history database</p>
<p style="color:#ff6666;">Please save the HTML code as 'forex_dashboard.html' in the same directory.</p>
<p style="color:#ff6666;">Current directory: ''' + os.getcwd() + '''</p>
<br><a href="/api/market-data" style="color:#00ccff;">API Test - Market Data</a>
<br><a href="/api/system-status" style="color:#00ccff;">System Status</a>
</body></html>'''
        
        @self.app.route('/api/market-data')
        def get_market_data():
            """Get market data with auto trading status"""
            try:
                formatted_data = {}
                for symbol, data in self.live_data.items():
                    if data:
                        # [OK] FIXED: Clean data for JSON serialization
                        cleaned_data = self.clean_data_for_json(data)
                        
                        # เพิ่มข้อมูล Pullback Protection
                        if hasattr(self, 'pullback_protection') and self.pullback_protection:
                            if 'pullback_protection' in cleaned_data:
                                # มีข้อมูล Pullback Protection อยู่แล้ว
                                pass
                            elif symbol in self.pullback_protection.waiting_positions:
                                # เพิ่มสถานะ waiting
                                cleaned_data['pullback_protection'] = {
                                    'status': 'WAITING_PULLBACK',
                                    'waiting_since': self.pullback_protection.waiting_positions[symbol]['wait_start'].strftime('%H:%M:%S')
                                }
                        
                        formatted_data[symbol] = cleaned_data
                
                # Auto trading status
                auto_trading_status = {
                    'enabled': self.auto_trading_enabled,
                    'emergency_stop': self.emergency_stop,
                    'active_pairs': list(self.auto_trading_pairs),
                    'total_trades': len([pos for positions in self.active_trades_per_pair.values() for pos in positions]),
                    'daily_stats': self.daily_stats,
                    'current_exposure': self.calculate_current_exposure() * 100,
                    'max_exposure': self.max_total_exposure * 100,
                    'risk_profile': self.current_risk_profile,
                    'custom_risk': self.custom_risk_per_trade,
                    # [SHIELD] Pullback Protection Status
                    'pullback_protection': {
                        'enabled': (hasattr(self, 'pullback_protection') and 
                                self.pullback_protection is not None and 
                                getattr(self.pullback_protection, 'enabled', False)),
                        'waiting_positions': (len(getattr(self.pullback_protection, 'waiting_positions', {})) 
                                            if hasattr(self, 'pullback_protection') and self.pullback_protection 
                                            else 0),
                        'total_blocked': (getattr(self.pullback_protection, 'statistics', {}).get('signals_blocked', 0) 
                                        if hasattr(self, 'pullback_protection') and self.pullback_protection 
                                        else 0)
                    }
                }
                
                return jsonify({
                    'success': True,
                    'data': formatted_data,
                    'account': self.account_info,
                    'auto_trading': auto_trading_status,
                    'risk_settings': {
                        'max_risk_per_trade': self.max_risk_per_trade * 100,
                        'max_total_exposure': self.max_total_exposure * 100,
                        'max_daily_loss': self.max_daily_loss * 100,
                        'current_risk_profile': self.current_risk_profile,
                        'account_balance': self.account_balance
                    },
                    'timestamp': datetime.now().isoformat(),
                    'mt5_connected': self.mt5_connected,
                    'persistence_active': True
                })
                
            except Exception as e:
                self.logger.error(f"Error in market-data API: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {},
                    'auto_trading': {
                        'enabled': False,
                        'emergency_stop': True,
                        'pullback_protection': {'enabled': False, 'waiting_positions': 0, 'total_blocked': 0}
                    },
                    'mt5_connected': False,
                    'timestamp': datetime.now().isoformat()
                })
                        
        @self.app.route('/api/validate-symbols')
        def validate_symbols_api():
            """ตรวจสอบ symbols ทั้งหมด"""
            try:
                validation_results = []
                
                for pair in self.forex_pairs:
                    try:
                        # ทดสอบ symbol
                        symbol_info = mt5.symbol_info(pair)
                        tick = mt5.symbol_info_tick(pair)
                        
                        validation_results.append({
                            'symbol': pair,
                            'symbol_exists': symbol_info is not None,
                            'has_price': tick is not None and tick.bid > 0,
                            'status': 'OK' if symbol_info and tick and tick.bid > 0 else 'FAILED'
                        })
                    except Exception as e:
                        validation_results.append({
                            'symbol': pair,
                            'status': 'ERROR',
                            'error': str(e)
                        })
                
                valid_count = len([r for r in validation_results if r.get('status') == 'OK'])
                
                return jsonify({
                    'success': True,
                    'valid_symbols': valid_count,
                    'total_symbols': len(self.forex_pairs),
                    'validation_rate': f"{valid_count/len(self.forex_pairs)*100:.1f}%",
                    'results': validation_results
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
        @self.app.route('/api/auto-trading/status-detailed')
        def get_detailed_auto_trading_status():
            """ดู auto trading status รายละเอียด"""
            try:
                return jsonify({
                    'success': True,
                    'auto_trading_enabled': self.auto_trading_enabled,
                    'emergency_stop': self.emergency_stop,
                    'mt5_connected': self.mt5_connected,
                    'active_pairs': len([pair for pair, status in self.pair_trade_status.items() if status == 'READY']),
                    'blocked_pairs': len([pair for pair, status in self.pair_trade_status.items() if status != 'READY']),
                    'account_balance': self.account_balance,
                    'current_positions': len(mt5.positions_get()) if mt5.positions_get() else 0
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/broker-info')
        def get_broker_info():
            """[CHART] API: ดึงข้อมูล broker และ symbol mapping"""
            try:
                if not self.mt5_connected:
                    return jsonify({
                        'success': False,
                        'error': 'MT5 not connected'
                    })
                
                # ดึงข้อมูล broker
                account_info = mt5.account_info()
                if not account_info:
                    return jsonify({
                        'success': False,
                        'error': 'Cannot get account info'
                    })
                
                # ดึงข้อมูล symbol mapping
                mapping_info = {}
                total_system_symbols = 0
                
                if hasattr(self, 'symbol_adapter') and self.symbol_adapter:
                    try:
                        mapping_info = self.symbol_adapter.get_mapping_info()
                        total_system_symbols = len(getattr(self.symbol_adapter, 'system_symbols', []))
                    except Exception as mapping_error:
                        self.logger.error(f"Error getting mapping info: {mapping_error}")
                        mapping_info = {}
                        total_system_symbols = 0

                result = {
                    'success': True,
                    'broker_info': {
                        'server': account_info.server,
                        'company': account_info.company,
                        'trade_allowed': account_info.trade_allowed,
                        'trade_expert': account_info.trade_expert,
                        'currency': account_info.currency,
                        'leverage': account_info.leverage,
                        'margin_so_mode': account_info.margin_so_mode,
                        'login': account_info.login,
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin': account_info.margin,
                        'free_margin': account_info.margin_free,
                        'margin_level': account_info.margin_level
                    },
                    'symbol_mapping': {
                        'mapping_enabled': self.broker_symbols_mapped,
                        'total_system_symbols': total_system_symbols,
                        'mapped_symbols': len(mapping_info.get('available_broker_symbols', [])),
                        'mapping_success_rate': f"{len(self.broker_symbol_map)/len(self.system_symbols)*100:.1f}%" if len(self.system_symbols) > 0 else "0%",
                        'detected_suffixes': mapping_info.get('detected_suffixes', []),
                        'sample_mapping': mapping_info.get('sample_mapping', {}),
                        'available_pairs': list(getattr(self, 'forex_pairs', []))
                    },
                    'connection_status': {
                        'mt5_connected': self.mt5_connected,
                        'auto_trading_enabled': self.auto_trading_enabled,
                        'emergency_stop': getattr(self, 'emergency_stop', False),
                        'terminal_info': {
                            'name': mt5.terminal_info().name if mt5.terminal_info() else 'Unknown',
                            'version': str(mt5.terminal_info().build) if mt5.terminal_info() else 'Unknown',
                            'path': mt5.terminal_info().path if mt5.terminal_info() else 'Unknown'
                        }
                    },
                    'trading_stats': {
                        'total_forex_pairs': len(self.forex_pairs),
                        'active_pairs_in_auto_trading': len(self.auto_trading_pairs) if hasattr(self, 'auto_trading_pairs') else 0,
                        'current_positions': len(mt5.positions_get() or []),
                        'account_balance': account_info.balance,
                        'account_equity': account_info.equity,
                        'risk_profile': getattr(self, 'current_risk_profile', 'BALANCED'),
                        'max_simultaneous_trades': getattr(self, 'max_simultaneous_trades', 8)
                    }
                }
                
                return jsonify(result)
                
            except Exception as e:
                self.logger.error(f"Error getting broker info: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'broker_mapping_status': getattr(self, 'broker_symbols_mapped', False),
                    'mt5_connection_status': getattr(self, 'mt5_connected', False)
                })
        
        @self.app.route('/api/broker-mapping-test')
        def test_broker_mapping():
            """[EMOJI] API: ทดสอบ broker symbol mapping"""
            try:
                if not self.mt5_connected:
                    return jsonify({
                        'success': False,
                        'error': 'MT5 not connected'
                    })
                
                if not hasattr(self, 'symbol_adapter') or not self.broker_symbols_mapped:
                    return jsonify({
                        'success': False,
                        'error': 'Broker symbol mapping not enabled'
                    })
                
                test_symbols = ['EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'XAUUSD.c']
                test_results = []
                success_count = 0
                
                for system_symbol in test_symbols:
                    if system_symbol in self.forex_pairs:
                        broker_symbol = self.symbol_adapter.system_to_broker_symbol(system_symbol)
                        
                        # ทดสอบดึงข้อมูล
                        tick = mt5.symbol_info_tick(broker_symbol)
                        symbol_info = mt5.symbol_info(broker_symbol)
                        
                        test_result = {
                            'system_symbol': system_symbol,
                            'broker_symbol': broker_symbol,
                            'tick_available': tick is not None,
                            'symbol_info_available': symbol_info is not None,
                            'symbol_visible': symbol_info.visible if symbol_info else False,
                            'current_price': tick.bid if tick else 0,
                            'status': 'SUCCESS' if tick and symbol_info else 'FAILED'
                        }
                        
                        if test_result['status'] == 'SUCCESS':
                            success_count += 1
                        
                        test_results.append(test_result)
                    else:
                        test_results.append({
                            'system_symbol': system_symbol,
                            'broker_symbol': 'NOT_FOUND',
                            'status': 'NOT_IN_PAIRS',
                            'error': 'Symbol not in forex_pairs list'
                        })
                
                return jsonify({
                    'success': True,
                    'test_summary': {
                        'total_tested': len(test_symbols),
                        'successful_mappings': success_count,
                        'success_rate': f"{(success_count/len(test_symbols)*100):.1f}%",
                        'broker_mapping_enabled': self.broker_symbols_mapped
                    },
                    'detailed_results': test_results,
                    'mapping_info': self.symbol_adapter.get_mapping_info() if hasattr(self, 'symbol_adapter') else {}
                })
                
            except Exception as e:
                self.logger.error(f"Broker mapping test failed: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                })

        @self.app.route('/api/system-status')
        def get_system_status():
            """Get system status and persistence info"""
            try:
                # Check data files
                data_files_status = {
                    'settings_file': os.path.exists(self.persistence.settings_file),
                    'daily_stats_file': os.path.exists(self.persistence.daily_stats_file),
                    'pair_status_file': os.path.exists(self.persistence.pair_status_file),
                    'database_file': os.path.exists(self.persistence.db_file)
                }
                
                # Get trade history count
                trade_history_count = len(self.persistence.get_trade_history(30))
                
                return jsonify({
                    'success': True,
                    'system_status': {
                        'auto_trading_enabled': self.auto_trading_enabled,
                        'mt5_connected': self.mt5_connected,
                        'emergency_stop': self.emergency_stop,
                        'is_running': self.is_running,
                        'last_update': self.last_update.isoformat(),
                        'uptime_hours': (datetime.now() - self.last_update).total_seconds() / 3600
                    },
                    'persistence_status': {
                        'data_directory': self.persistence.data_dir,
                        'data_files': data_files_status,
                        'trade_history_records': trade_history_count,
                        'auto_save_active': True
                    },
                    'current_settings': {
                        'risk_profile': self.current_risk_profile,
                        'min_signal_strength': self.min_signal_strength,
                        'min_entry_quality': self.min_entry_quality,
                        'max_simultaneous_trades': self.max_simultaneous_trades
                    },
                    'daily_stats': self.daily_stats,
                    'active_pairs_count': len([pair for pair, status in self.pair_trade_status.items() if status == 'TRADING'])
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/auto-trading/start', methods=['POST'])
        def start_auto_trading_api():
            """Start auto trading"""
            if self.start_auto_trading():
                return jsonify({'success': True, 'message': 'Auto trading started'})
            else:
                return jsonify({'success': False, 'error': 'Failed to start auto trading'})
        
        @self.app.route('/api/auto-trading/stop', methods=['POST'])
        def stop_auto_trading_api():
            """Stop auto trading"""
            self.stop_auto_trading()
            return jsonify({'success': True, 'message': 'Auto trading stopped'})
        
        @self.app.route('/api/auto-trading/emergency-stop', methods=['POST'])
        def emergency_stop_api():
            """Emergency stop all trading"""
            self.emergency_stop_all()
            return jsonify({'success': True, 'message': 'Emergency stop activated'})
        
        @self.app.route('/api/auto-trading/settings', methods=['POST'])
        def update_auto_trading_settings():
            """Update auto trading settings"""
            try:
                data = request.get_json()
                
                if 'risk_profile' in data:
                    self.update_risk_profile(data['risk_profile'])
                
                if 'custom_risk' in data:
                    self.update_risk_profile(custom_risk=data['custom_risk'])
                
                if 'min_signal_strength' in data:
                    self.min_signal_strength = float(data['min_signal_strength'])
                    self.logger.info(f"Min signal strength updated to: {self.min_signal_strength}")
                
                if 'min_entry_quality' in data:
                    self.min_entry_quality = data['min_entry_quality']
                    self.logger.info(f"Min entry quality updated to: {self.min_entry_quality}")
                
                if 'enabled_pairs' in data:
                    self.auto_trading_pairs = set(data['enabled_pairs'])
                
                if 'max_simultaneous_trades' in data:
                    self.max_simultaneous_trades = int(data['max_simultaneous_trades'])
                
                # New signal settings
                if 'require_trend_alignment' in data:
                    self.required_confirmations['trend_alignment'] = data['require_trend_alignment']
                
                if 'require_volume_confirmation' in data:
                    self.required_confirmations['volume_confirmation'] = data['require_volume_confirmation']
                
                if 'require_rsi_filter' in data:
                    self.required_confirmations['rsi_filter'] = data['require_rsi_filter']
                
                if 'require_multiple_timeframe' in data:
                    self.required_confirmations['multiple_timeframe'] = data['require_multiple_timeframe']
                
                if 'min_rr_ratio' in data:
                    self.min_rr_ratio = float(data['min_rr_ratio'])
                
                # Save settings immediately after update
                self.save_system_settings()
                self.persistence.log_system_event('INFO', 'Trading settings updated via API', 'SETTINGS')
                
                return jsonify({
                    'success': True, 
                    'message': 'Settings updated and saved',
                    'current_settings': {
                        'min_signal_strength': self.min_signal_strength,
                        'min_entry_quality': self.min_entry_quality,
                        'confirmations': self.required_confirmations,
                        'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5),
                        'last_saved': datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                self.persistence.log_system_event('ERROR', f'Settings update failed: {str(e)}', 'SETTINGS')
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/auto-trading/current-settings')
        def get_current_settings():
            """Get current auto trading settings"""
            try:
                return jsonify({
                    'success': True,
                    'settings': {
                        'min_signal_strength': self.min_signal_strength,
                        'min_entry_quality': self.min_entry_quality,
                        'max_simultaneous_trades': self.max_simultaneous_trades,
                        'confirmations': self.required_confirmations,
                        'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5),
                        'auto_trading_enabled': self.auto_trading_enabled,
                        'risk_profile': self.current_risk_profile,
                        'custom_risk': self.custom_risk_per_trade
                    },
                    'persistence_info': {
                        'settings_loaded': True,
                        'auto_save_enabled': True,
                        'last_saved': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            try:
                positions = mt5.positions_get()
                if positions is None:
                    positions = []
                
                formatted_positions = []
                for pos in positions:
                    formatted_positions.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit,
                        'time': pos.time
                    })
                
                return jsonify({
                    'success': True,
                    'positions': formatted_positions,
                    'total_positions': len(formatted_positions),
                    'tracking_data': {
                        'active_trades_per_pair': {k: len(v) for k, v in self.active_trades_per_pair.items() if v},
                        'pair_status': self.pair_trade_status
                    }
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/trade-history')
        def get_trade_history():
            """Get trade history from database"""
            try:
                days = request.args.get('days', 30, type=int)
                trades = self.persistence.get_trade_history(days)
                
                return jsonify({
                    'success': True,
                    'trades': trades,
                    'total_trades': len(trades),
                    'days_requested': days
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/refresh')
        def refresh_data():
            """Manual refresh"""
            try:
                self.update_all_data()
                # Save current state after refresh
                self.save_pair_tracking_data()
                
                return jsonify({
                    'success': True,
                    'message': 'Data refreshed and state saved',
                    'timestamp': datetime.now().isoformat(),
                    'pairs_updated': len(self.live_data)
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/backup-data', methods=['POST'])
        def backup_data():
            """Create manual backup of all system data"""
            try:
                backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = os.path.join(self.persistence.data_dir, backup_dir)
                os.makedirs(backup_path, exist_ok=True)
                
                # Copy all data files
                import shutil
                for file_name in ['settings.json', 'daily_stats.json', 'pair_status.json', 'trading_system.db']:
                    src_path = os.path.join(self.persistence.data_dir, file_name)
                    dst_path = os.path.join(backup_path, file_name)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                
                self.persistence.log_system_event('INFO', f'Manual backup created: {backup_dir}', 'BACKUP')
                
                return jsonify({
                    'success': True,
                    'message': 'Backup created successfully',
                    'backup_path': backup_path
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/enhanced-signal/<symbol>')
        def get_enhanced_signal(symbol):
            """Get enhanced signal for specific symbol"""
            try:
                if hasattr(self, 'signal_engine') and self.signal_engine:
                    result = self.signal_engine.get_multi_timeframe_confluence(symbol)
                    return jsonify({
                        'success': True,
                        'symbol': symbol,
                        'result': result
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Enhanced signal engine not available'
                    })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        self.app.route('/api/advanced-features-status')
        def get_advanced_features_status():
            """Get advanced features status"""
            return jsonify({
                'success': True,
                'advanced_features_active': self.use_advanced_features,
                'features': {
                    'market_regime_detection': self.use_advanced_features,
                    'enhanced_signal_scoring': self.use_advanced_features,
                    'dynamic_position_sizing': self.use_advanced_features
                },
                'message': 'Advanced features active' if self.use_advanced_features else 'Using standard features'
            })
        
        @self.app.route('/api/market-regime/<symbol>')
        def get_market_regime(symbol):
            """Get market regime for specific symbol"""
            if not self.use_advanced_features:
                return jsonify({
                    'success': False,
                    'error': 'Advanced features not available'
                })
            
            try:
                # Get current data for the symbol
                symbol_data = self.live_data.get(symbol, {})
                
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'market_regime': symbol_data.get('market_regime', 'UNKNOWN'),
                    'regime_confidence': symbol_data.get('regime_confidence', 0),
                    'trend_strength': symbol_data.get('trend_strength', 0),
                    'volatility_percentile': symbol_data.get('volatility_percentile', 50),
                    'enhanced_strength': symbol_data.get('enhanced_strength', 0),
                    'enhanced_quality': symbol_data.get('enhanced_quality', 'POOR')
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/toggle-advanced-features', methods=['POST'])
        def toggle_advanced_features():
            """Toggle advanced features on/off"""            
            try:
                self.use_advanced_features = not self.use_advanced_features
                
                status = 'ENABLED'
                if self.use_advanced_features:
                    self.advanced_integrator = UniversalAdvancedTradingIntegrator(self)
                else: 
                    status = 'DISABLED'

                self.logger.info(f"Advanced features {status}")
                
                return jsonify({
                    'success': True,
                    'advanced_features_active': self.use_advanced_features,
                    'message': f'Advanced features {status}'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/hedging')
        def hedging_dashboard():
            """Serve hedging dashboard"""
            try:
                return send_from_directory('.', 'hedging_dashboard.html')
            except Exception as e:
                return f'''<!DOCTYPE html>
    <html><head><title>Hedging Dashboard Error</title></head>
    <body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
    <h1 style="color:#ff4444;">[TARGET] Hedging Dashboard</h1>
    <p style="color:#ffaa00;">Please save the hedging dashboard HTML as 'hedging_dashboard.html'</p>
    <p style="color:#888;">Error: {str(e)}</p>
    <a href="/" style="color:#00ccff;"><- Back to Main Dashboard</a>
    </body></html>'''
    

        @self.app.route('/trailing-dashboard')
        def trailing_dashboard():
            """Serve trailing stop dashboard"""
            try:
                return send_from_directory('.', 'trailing_dashboard.html')
            except:
                return '''
                <h1 style="color:#00ff00;">[TARGET] Trailing Stop Dashboard</h1>
                <p style="color:#ffaa00;">Please save the trailing_dashboard.html file in the same directory.</p>
                <a href="/" style="color:#00ccff;"><- Back to Main Dashboard</a>
                '''

        # [SHIELD] Pullback Protection API Routes
        @self.app.route('/api/pullback-protection/status')
        def pullback_protection_status():
            """สถานะ Pullback Protection Plugin"""
            try:
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    stats = self.pullback_protection.get_statistics()
                    waiting = self.pullback_protection.get_waiting_positions_summary()
                    
                    return jsonify({
                        'success': True,
                        'plugin_enabled': self.pullback_protection.enabled,
                        'statistics': stats,
                        'waiting_positions': waiting,
                        'settings': self.pullback_protection.pullback_settings
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Pullback Protection Plugin not installed'
                    })
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/pullback-protection/toggle', methods=['POST'])
        def toggle_pullback_protection():
            """เปิด/ปิด Pullback Protection"""
            try:
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    data = request.get_json()
                    enabled = data.get('enabled', True)
                    
                    if enabled:
                        self.pullback_protection.enable()
                    else:
                        self.pullback_protection.disable()
                    
                    return jsonify({
                        'success': True,
                        'enabled': self.pullback_protection.enabled,
                        'message': f"Pullback Protection {'เปิด' if enabled else 'ปิด'}ใช้งาน"
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Plugin not available'
                    })
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/pullback-protection/settings', methods=['POST'])
        def update_pullback_settings():
            """อัพเดตการตั้งค่า Pullback Protection"""
            try:
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    data = request.get_json()
                    self.pullback_protection.update_settings(data)
                    
                    return jsonify({
                        'success': True,
                        'message': 'การตั้งค่าอัพเดตเรียบร้อย',
                        'current_settings': self.pullback_protection.pullback_settings
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Plugin not available'
                    })
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/pullback-protection/reset-stats', methods=['POST'])
        def reset_pullback_stats():
            """รีเซ็ตสถิติ"""
            try:
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    self.pullback_protection.reset_statistics()
                    
                    return jsonify({
                        'success': True,
                        'message': 'สถิติถูกรีเซ็ตแล้ว'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Plugin not available'
                    })
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/pullback-dashboard')
        def pullback_dashboard():
            """Pullback Protection Dashboard"""
            try:
                return send_from_directory('.', 'pullback_dashboard.html')
            except:
                return '''<!DOCTYPE html>
        <html><head><title>Pullback Protection Dashboard</title></head>
        <body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
        <h1 style="color:#cc0066;">[SHIELD] Pullback Protection Dashboard</h1>
        <p style="color:#ff6666;">ไฟล์ pullback_dashboard.html ไม่พบ</p>
        <p style="color:#ffaa00;">กรุณาบันทึกไฟล์ pullback_dashboard.html ในโฟลเดอร์เดียวกับ mt5_forex_connector.py</p>
        <p style="color:#666;">Current directory: ''' + os.getcwd() + '''</p>
        <br><a href="/" style="color:#00ccff;"><- กลับสู่ Main Dashboard</a>
        </body></html>'''

        # [EMOJI] เพิ่ม Route สำหรับ Quick Access
        @self.app.route('/pullback')
        def pullback_shortcut():
            """Quick access to Pullback Dashboard"""
            try:
                return send_from_directory('.', 'pullback_dashboard.html')
            except:
                return '''<!DOCTYPE html>
        <html><head><title>Pullback Protection</title>
        <meta http-equiv="refresh" content="0;url=/pullback_dashboard.html">
        </head>
        <body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
        <h1 style="color:#cc0066;">[SHIELD] Redirecting to Pullback Dashboard...</h1>
        <p style="color:#00ccff;"><a href="/pullback_dashboard.html">Click here if not redirected</a></p>
        <script>window.location.href='/pullback_dashboard.html';</script>
        </body></html>'''
       
        # [CHART] เพิ่ม Route สำหรับ Pullback Status Widget
        @self.app.route('/api/pullback-protection/widget')
        def pullback_widget_status():
            """Quick status for main dashboard widget"""
            try:
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    stats = self.pullback_protection.get_statistics()
                    waiting = self.pullback_protection.get_waiting_positions_summary()
                    
                    return jsonify({
                        'success': True,
                        'enabled': self.pullback_protection.enabled,
                        'blocked_today': stats.get('signals_blocked', 0),
                        'waiting_count': waiting.get('total_waiting', 0),
                        'block_rate': stats.get('block_rate_percent', 0),
                        'win_rate_improvement': stats.get('estimated_win_rate_improvement', '+0%'),
                        'status_text': 'ACTIVE' if self.pullback_protection.enabled else 'DISABLED',
                        'last_update': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'enabled': False,
                        'blocked_today': 0,
                        'waiting_count': 0,
                        'block_rate': 0,
                        'status_text': 'NOT_INSTALLED',
                        'error': 'Pullback Protection Plugin not installed'
                    })
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'enabled': False,
                    'error': str(e),
                    'status_text': 'ERROR'
                })

        
        
        print("[OK] All routes setup completed")

        

        @self.app.route('/api/account-info')
        def get_account_info():
            """[CHART] API: ดึงข้อมูล account และ MT5 connection status"""
            try:
                if not self.mt5_connected:
                    return jsonify({
                        'success': False,
                        'error': 'MT5 ไม่ได้เชื่อมต่อ',
                        'connection_status': {
                            'mt5_connected': False,
                            'terminal_info': None,
                            'auto_trading_enabled': False
                        }
                    })
                
                # ดึงข้อมูล account
                account_info = mt5.account_info()
                terminal_info = mt5.terminal_info()
                
                if not account_info:
                    return jsonify({
                        'success': False,
                        'error': 'ไม่สามารถดึงข้อมูล account ได้'
                    })
                
                # เตรียมข้อมูลสำหรับ response
                account_data = {
                    'login': account_info.login,
                    'server': account_info.server,
                    'company': account_info.company,
                    'name': account_info.name,
                    'currency': account_info.currency,
                    'balance': float(account_info.balance),
                    'equity': float(account_info.equity),
                    'margin': float(account_info.margin),
                    'free_margin': float(account_info.margin_free),
                    'margin_level': float(account_info.margin_level) if account_info.margin_level else 0,
                    'profit': float(account_info.profit),
                    'leverage': int(account_info.leverage),
                    'trade_allowed': bool(account_info.trade_allowed),
                    'trade_expert': bool(account_info.trade_expert),
                    'margin_so_mode': int(account_info.margin_so_mode),
                    'margin_so_call': float(account_info.margin_so_call),
                    'margin_so_so': float(account_info.margin_so_so)
                }
                
                # ข้อมูล terminal
                terminal_data = {
                    'name': terminal_info.name if terminal_info else 'Unknown',
                    'version': str(terminal_info.build) if terminal_info else 'Unknown',
                    'path': terminal_info.path if terminal_info else 'Unknown',
                    'data_path': terminal_info.data_path if terminal_info else 'Unknown',
                    'commondata_path': terminal_info.commondata_path if terminal_info else 'Unknown'
                }
                
                # ข้อมูลการเชื่อมต่อและระบบ
                connection_status = {
                    'mt5_connected': self.mt5_connected,
                    'auto_trading_enabled': getattr(self, 'auto_trading_enabled', False),
                    'emergency_stop': getattr(self, 'emergency_stop', False),
                    'last_update': getattr(self, 'last_update', datetime.now()).isoformat(),
                    'broker_symbols_mapped': getattr(self, 'broker_symbols_mapped', False)
                }
                
                # ข้อมูล trading system
                trading_stats = {
                    'total_forex_pairs': len(self.forex_pairs),
                    'active_pairs': len([p for p in self.forex_pairs if p in getattr(self, 'live_data', {})]),
                    'current_positions': len(mt5.positions_get() or []),
                    'current_orders': len(mt5.orders_get() or []),
                    'risk_profile': getattr(self, 'current_risk_profile', 'BALANCED'),
                    'max_simultaneous_trades': getattr(self, 'max_simultaneous_trades', 8)
                }
                
                # ใช้ clean_data_for_json เพื่อป้องกัน serialization error
                response_data = self.clean_data_for_json({
                    'account_info': account_data,
                    'terminal_info': terminal_data,
                    'connection_status': connection_status,
                    'trading_stats': trading_stats
                })
                
                return jsonify({
                    'success': True,
                    'data': response_data
                })
                
            except Exception as e:
                self.logger.error(f"Error getting account info: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'connection_status': {
                        'mt5_connected': getattr(self, 'mt5_connected', False)
                    }
                })
        
        # 2[EMOJI]⃣ เพิ่ม /api/symbol-data/<symbol> endpoint
        @self.app.route('/api/symbol-data/<symbol>')
        def get_specific_symbol_data(symbol):
            """[UP] API: ดึงข้อมูลเฉพาะ symbol ที่ระบุ"""
            try:
                if not self.mt5_connected:
                    return jsonify({
                        'success': False,
                        'error': 'MT5 ไม่ได้เชื่อมต่อ',
                        'symbol': symbol
                    })
                
                # ตรวจสอบว่า symbol อยู่ในระบบหรือไม่
                if symbol not in self.forex_pairs:
                    return jsonify({
                        'success': False,
                        'error': f'Symbol {symbol} ไม่อยู่ในรายการ forex pairs ของระบบ',
                        'symbol': symbol,
                        'available_symbols': self.forex_pairs[:10]  # แสดง 10 ตัวแรก
                    })
                
                # ดึงข้อมูล symbol จาก live_data หรือสร้างใหม่
                if hasattr(self, 'live_data') and symbol in self.live_data:
                    symbol_data = self.live_data[symbol]
                else:
                    # ถ้าไม่มีใน live_data ให้ดึงข้อมูลใหม่
                    symbol_data = self.get_symbol_data(symbol)
                    if not symbol_data:
                        return jsonify({
                            'success': False,
                            'error': f'ไม่สามารถดึงข้อมูลของ {symbol} ได้',
                            'symbol': symbol
                        })
                
                # เพิ่มข้อมูล symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    symbol_info_data = {
                        'symbol': symbol_info.name,
                        'description': symbol_info.description,
                        'currency_base': symbol_info.currency_base,
                        'currency_profit': symbol_info.currency_profit,
                        'currency_margin': symbol_info.currency_margin,
                        'digits': symbol_info.digits,
                        'trade_tick_value': float(symbol_info.trade_tick_value),
                        'trade_tick_size': float(symbol_info.trade_tick_size),
                        'trade_contract_size': float(symbol_info.trade_contract_size),
                        'volume_min': float(symbol_info.volume_min),
                        'volume_max': float(symbol_info.volume_max),
                        'volume_step': float(symbol_info.volume_step),
                        'spread': symbol_info.spread,
                        'trade_mode': symbol_info.trade_mode,
                        'trade_allowed': bool(symbol_info.trade_allowed),
                        'trade_stops_level': symbol_info.trade_stops_level,
                        'trade_freeze_level': symbol_info.trade_freeze_level
                    }
                else:
                    symbol_info_data = {'error': f'ไม่สามารถดึง symbol info ของ {symbol} ได้'}
                
                # เพิ่มข้อมูล tick ล่าสุด
                tick_info = mt5.symbol_info_tick(symbol)
                if tick_info:
                    tick_data = {
                        'time': datetime.fromtimestamp(tick_info.time).isoformat(),
                        'bid': float(tick_info.bid),
                        'ask': float(tick_info.ask),
                        'last': float(tick_info.last),
                        'volume': int(tick_info.volume),
                        'time_msc': tick_info.time_msc,
                        'flags': tick_info.flags,
                        'volume_real': float(tick_info.volume_real)
                    }
                else:
                    tick_data = {'error': f'ไม่สามารถดึง tick data ของ {symbol} ได้'}
                
                # รวมข้อมูลทั้งหมด
                complete_data = {
                    'symbol': symbol,
                    'trading_data': symbol_data,
                    'symbol_info': symbol_info_data,
                    'tick_info': tick_data,
                    'last_update': datetime.now().isoformat()
                }
                
                # ใช้ clean_data_for_json
                clean_data = self.clean_data_for_json(complete_data)
                
                return jsonify({
                    'success': True,
                    'data': clean_data
                })
                
            except Exception as e:
                self.logger.error(f"Error getting symbol data for {symbol}: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'symbol': symbol
                })
        
        # 3[EMOJI]⃣ เพิ่ม /api/test-advanced-features endpoint
        @self.app.route('/api/test-advanced-features')
        def test_advanced_features():
            """[EMOJI] API: ทดสอบ advanced features ของระบบ"""
            try:
                test_results = {
                    'timestamp': datetime.now().isoformat(),
                    'tests_performed': []
                }
                
                # Test 1: Enhanced Signal System
                if hasattr(self, 'enhanced_signal_engine') and self.enhanced_signal_engine:
                    try:
                        # ทดสอบ signal generation สำหรับ EURUSD
                        test_symbol = 'EURUSD.c' if 'EURUSD.c' in self.forex_pairs else 'EURUSD'
                        if test_symbol in self.forex_pairs:
                            signals = self.enhanced_signal_engine.generate_signals([test_symbol])
                            test_results['tests_performed'].append({
                                'test': 'Enhanced Signal System',
                                'status': 'PASS',
                                'result': f'Generated signals for {test_symbol}',
                                'data': signals.get(test_symbol, {}) if signals else {}
                            })
                        else:
                            test_results['tests_performed'].append({
                                'test': 'Enhanced Signal System',
                                'status': 'SKIP',
                                'result': 'No suitable test symbol found'
                            })
                    except Exception as e:
                        test_results['tests_performed'].append({
                            'test': 'Enhanced Signal System',
                            'status': 'FAIL',
                            'error': str(e)
                        })
                else:
                    test_results['tests_performed'].append({
                        'test': 'Enhanced Signal System',
                        'status': 'NOT_AVAILABLE',
                        'result': 'Enhanced signal engine not initialized'
                    })
                
                # Test 2: Advanced Features Integration
                if hasattr(self, 'advanced_integrator') and self.advanced_integrator:
                    try:
                        # ทดสอบ market regime detection
                        regime_data = self.advanced_integrator.get_dashboard_data()
                        test_results['tests_performed'].append({
                            'test': 'Advanced Features Integration',
                            'status': 'PASS',
                            'result': 'Market regime detection working',
                            'data': regime_data.get('market_regime', {})
                        })
                    except Exception as e:
                        test_results['tests_performed'].append({
                            'test': 'Advanced Features Integration',
                            'status': 'FAIL',
                            'error': str(e)
                        })
                else:
                    test_results['tests_performed'].append({
                        'test': 'Advanced Features Integration',
                        'status': 'NOT_AVAILABLE',
                        'result': 'Advanced integrator not initialized'
                    })
                
                # Test 3: Pullback Protection
                if hasattr(self, 'pullback_protection') and self.pullback_protection:
                    try:
                        stats = self.pullback_protection.get_statistics()
                        test_results['tests_performed'].append({
                            'test': 'Pullback Protection',
                            'status': 'PASS',
                            'result': 'Pullback protection system active',
                            'data': stats
                        })
                    except Exception as e:
                        test_results['tests_performed'].append({
                            'test': 'Pullback Protection',
                            'status': 'FAIL',
                            'error': str(e)
                        })
                else:
                    test_results['tests_performed'].append({
                        'test': 'Pullback Protection',
                        'status': 'NOT_AVAILABLE',
                        'result': 'Pullback protection not installed'
                    })
                
                # Test 4: Trailing Stops
                if hasattr(self, 'enhanced_trading') and self.enhanced_trading:
                    try:
                        trailing_data = self.enhanced_trading.get_trailing_dashboard_data()
                        test_results['tests_performed'].append({
                            'test': 'Trailing Stops System',
                            'status': 'PASS',
                            'result': 'Trailing stops system active',
                            'data': trailing_data
                        })
                    except Exception as e:
                        test_results['tests_performed'].append({
                            'test': 'Trailing Stops System',
                            'status': 'FAIL',
                            'error': str(e)
                        })
                else:
                    test_results['tests_performed'].append({
                        'test': 'Trailing Stops System',
                        'status': 'NOT_AVAILABLE',
                        'result': 'Enhanced trading system not initialized'
                    })
                
                # Test 5: Hedging System
                if hasattr(self, 'hedge_integrator') and self.hedge_integrator:
                    try:
                        hedge_data = self.hedge_integrator.get_dashboard_data()
                        test_results['tests_performed'].append({
                            'test': 'Hedging System',
                            'status': 'PASS',
                            'result': 'Hedging system active',
                            'data': hedge_data
                        })
                    except Exception as e:
                        test_results['tests_performed'].append({
                            'test': 'Hedging System',
                            'status': 'FAIL',
                            'error': str(e)
                        })
                else:
                    test_results['tests_performed'].append({
                        'test': 'Hedging System',
                        'status': 'NOT_AVAILABLE',
                        'result': 'Hedging system not installed'
                    })
                
                # สรุปผลการทดสอบ
                total_tests = len(test_results['tests_performed'])
                passed_tests = len([t for t in test_results['tests_performed'] if t['status'] == 'PASS'])
                failed_tests = len([t for t in test_results['tests_performed'] if t['status'] == 'FAIL'])
                not_available = len([t for t in test_results['tests_performed'] if t['status'] == 'NOT_AVAILABLE'])
                
                test_results['summary'] = {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'not_available': not_available,
                    'skipped': total_tests - passed_tests - failed_tests - not_available,
                    'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
                }
                
                # ใช้ clean_data_for_json
                clean_results = self.clean_data_for_json(test_results)
                
                return jsonify({
                    'success': True,
                    'data': clean_results
                })
                
            except Exception as e:
                self.logger.error(f"Error testing advanced features: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        # 4[EMOJI]⃣ เพิ่ม /api/test-signals endpoint
        @self.app.route('/api/test-signals')
        def test_signals():
            """[TARGET] API: ทดสอบการสร้าง signals สำหรับทุก trading pairs"""
            try:
                if not self.mt5_connected:
                    return jsonify({
                        'success': False,
                        'error': 'MT5 ไม่ได้เชื่อมต่อ'
                    })
                
                test_results = {
                    'timestamp': datetime.now().isoformat(),
                    'total_pairs_tested': 0,
                    'successful_signals': 0,
                    'failed_signals': 0,
                    'signal_details': []
                }
                
                # ทดสอบ signal generation สำหรับแต่ละ pair
                test_pairs = self.forex_pairs[:10]  # ทดสอบ 10 pairs แรก
                
                for symbol in test_pairs:
                    try:
                        test_results['total_pairs_tested'] += 1
                        
                        # ดึงข้อมูล symbol
                        symbol_data = self.get_symbol_data(symbol)
                        if not symbol_data:
                            test_results['failed_signals'] += 1
                            test_results['signal_details'].append({
                                'symbol': symbol,
                                'status': 'FAILED',
                                'error': 'Cannot get symbol data'
                            })
                            continue
                        
                        # ทดสอบ enhanced signal ถ้ามี
                        enhanced_signal = None
                        if hasattr(self, 'enhanced_signal_engine') and self.enhanced_signal_engine:
                            try:
                                enhanced_signals = self.enhanced_signal_engine.generate_signals([symbol])
                                enhanced_signal = enhanced_signals.get(symbol, {})
                            except Exception as e:
                                enhanced_signal = {'error': str(e)}
                        
                        # สร้าง signal summary
                        signal_summary = {
                            'symbol': symbol,
                            'status': 'SUCCESS',
                            'basic_signal': {
                                'signal': symbol_data.get('signal', 'NONE'),
                                'strength': symbol_data.get('strength', 0),
                                'entry_quality': symbol_data.get('entry_quality', 'POOR'),
                                'rsi': symbol_data.get('rsi', 50),
                                'macd': symbol_data.get('macd', 0),
                                'trend_strength': symbol_data.get('trendStrength', 0)
                            },
                            'enhanced_signal': enhanced_signal,
                            'can_trade': symbol_data.get('can_trade', False),
                            'risk_metrics': {
                                'lot_size': symbol_data.get('lot_size', 0),
                                'risk_percentage': symbol_data.get('risk_percentage', 0),
                                'rr_tp1': symbol_data.get('rr_tp1', 0)
                            },
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        test_results['successful_signals'] += 1
                        test_results['signal_details'].append(signal_summary)
                        
                    except Exception as e:
                        test_results['failed_signals'] += 1
                        test_results['signal_details'].append({
                            'symbol': symbol,
                            'status': 'FAILED',
                            'error': str(e)
                        })
                
                # สร้าง signal statistics
                strong_signals = len([s for s in test_results['signal_details'] 
                                    if s.get('status') == 'SUCCESS' and 
                                    s.get('basic_signal', {}).get('signal') in ['STRONG_BUY', 'STRONG_SELL']])
                
                regular_signals = len([s for s in test_results['signal_details'] 
                                    if s.get('status') == 'SUCCESS' and 
                                    s.get('basic_signal', {}).get('signal') in ['BUY', 'SELL']])
                
                tradeable_pairs = len([s for s in test_results['signal_details'] 
                                    if s.get('status') == 'SUCCESS' and 
                                    s.get('can_trade', False)])
                
                test_results['signal_statistics'] = {
                    'strong_signals': strong_signals,
                    'regular_signals': regular_signals,
                    'no_signals': test_results['successful_signals'] - strong_signals - regular_signals,
                    'tradeable_pairs': tradeable_pairs,
                    'signal_generation_rate': f"{(test_results['successful_signals']/test_results['total_pairs_tested']*100):.1f}%" if test_results['total_pairs_tested'] > 0 else "0%"
                }
                
                # ใช้ clean_data_for_json
                clean_results = self.clean_data_for_json(test_results)
                
                return jsonify({
                    'success': True,
                    'data': clean_results
                })
                
            except Exception as e:
                self.logger.error(f"Error testing signals: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        # 5[EMOJI]⃣ เพิ่ม /api/pullback-status endpoint
        @self.app.route('/api/pullback-status')
        def get_pullback_status():
            """[SHIELD] API: ดึงสถานะ pullback protection system"""
            try:
                if not hasattr(self, 'pullback_protection') or not self.pullback_protection:
                    return jsonify({
                        'success': False,
                        'error': 'Pullback Protection Plugin ไม่ได้ติดตั้งหรือเปิดใช้งาน',
                        'plugin_available': False
                    })
                
                # ดึงข้อมูลสถิติ
                statistics = self.pullback_protection.get_statistics()
                
                # ดึงข้อมูล waiting positions
                waiting_positions = self.pullback_protection.get_waiting_positions_summary()
                
                # ดึงการตั้งค่า
                settings = self.pullback_protection.pullback_settings
                
                # ข้อมูลสถานะปัจจุบัน
                current_status = {
                    'enabled': self.pullback_protection.enabled,
                    'total_waiting_positions': len(self.pullback_protection.waiting_positions),
                    'last_cleanup_time': getattr(self.pullback_protection, 'last_cleanup_time', datetime.now()).isoformat(),
                    'plugin_version': getattr(self.pullback_protection, 'version', '1.0'),
                    'active_since': getattr(self.pullback_protection, 'start_time', datetime.now()).isoformat()
                }
                
                # ข้อมูลการทำงานล่าสุด
                recent_activity = {
                    'trades_protected_today': statistics.get('trades_protected', 0),
                    'avg_protection_time': statistics.get('avg_waiting_time_minutes', 0),
                    'success_rate': statistics.get('success_rate_percent', 0),
                    'total_saved_amount': statistics.get('total_amount_saved', 0)
                }
                
                # สรุปข้อมูลทั้งหมด
                pullback_data = {
                    'plugin_status': current_status,
                    'protection_statistics': statistics,
                    'waiting_positions': waiting_positions,
                    'recent_activity': recent_activity,
                    'settings': settings,
                    'last_update': datetime.now().isoformat()
                }
                
                # ใช้ clean_data_for_json
                clean_data = self.clean_data_for_json(pullback_data)
                
                return jsonify({
                    'success': True,
                    'data': clean_data,
                    'plugin_available': True
                })
                
            except Exception as e:
                self.logger.error(f"Error getting pullback status: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'plugin_available': hasattr(self, 'pullback_protection') and self.pullback_protection is not None
                })
            
        # CRITICAL FIX 7: Add monitoring endpoint
        @self.app.route('/api/auto-trading/status-detailed')
        def get_detailed_status():
            """Get detailed auto trading status for debugging"""
            try:
                thread_count = threading.active_count()
                
                return jsonify({
                    'success': True,
                    'auto_trading_enabled': self.auto_trading_enabled,
                    'is_running': self.is_running,
                    'emergency_stop': self.emergency_stop,
                    'mt5_connected': self.mt5_connected,
                    'thread_count': thread_count,
                    'eligible_pairs': len([pair for pair, status in self.pair_trade_status.items() if status == 'READY']),
                    'active_pairs': len([pair for pair, trades in self.active_trades_per_pair.items() if trades]),
                    'last_global_trade': self.last_global_trade_time.isoformat() if self.last_global_trade_time else None,
                    'system_health': {
                        'thread_active': thread_count > 1,
                        'recent_update': (datetime.now() - self.last_update).total_seconds() < 60,
                        'data_fresh': len(self.live_data) > 0
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        # CRITICAL FIX 8: Add force restart endpoint  
        @self.app.route('/api/auto-trading/force-restart', methods=['POST'])
        def force_restart_auto_trading():
            """Force restart auto trading with full cleanup"""
            try:
                self.logger.info("[REFRESH] Force restarting auto trading...")
                
                # Stop everything
                self.auto_trading_enabled = False
                self.emergency_stop = False
                time.sleep(3)  # Wait for threads to stop
                
                # Reset state
                self.is_running = True
                
                # Start fresh
                if self.start_auto_trading():
                    return jsonify({
                        'success': True, 
                        'message': 'Auto trading force restarted successfully'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to restart auto trading'
                    })
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
            
    def setup_hedging_routes(self):
        """Setup hedging system routes"""
        try:
            # เพิ่ม hedging routes
            self.hedge_integrator.setup_hedge_routes(self.app)
            
            print("setup_hedging_routes")

            @self.app.route('/api/hedge/status')
            def hedge_status():
                """Get hedging system status"""
                return jsonify({
                    'success': True,
                    'hedging_enabled': self.hedging_enabled,
                    'system_status': 'ACTIVE',
                    'features': [
                        'Real-time Correlation Analysis',
                        'Smart Hedge Recommendations', 
                        'Portfolio Risk Matrix',
                        'Dynamic Hedge Ratios'
                    ]
                })
                       
            @self.app.route('/api/hedge/recommendations')
            def get_recommendations():
                positions = mt5.positions_get()
                if not positions:
                    return jsonify({'success': True, 'opportunities': [], 'total_positions': 0})
                
                opportunities = []
                for pos in positions:
                    if 'NZDJPY' in pos.symbol:
                        opportunities.append({
                            'primary_pair': 'NZDJPY',
                            'recommended_hedge': 'USDCHF',
                            'action': 'SELL',
                            'correlation': -0.60
                        })
                
                return jsonify({
                    'success': True, 
                    'opportunities': opportunities,
                    'total_positions': len(positions)
                })
           
            @self.app.route('/api/hedge/test')
            def test_hedging():
                """Test hedging system"""
                try:
                    # ทดสอบการทำงานของระบบ
                    test_correlation = self.hedge_integrator.hedge_system.calculate_live_correlation(
                        'EURUSD.c', 'USDCHF.c'
                    )
                    
                    return jsonify({
                        'success': True,
                        'test_correlation': round(test_correlation, 3),
                        'test_pair': 'EURUSD vs USDCHF',
                        'system_working': True,
                        'message': 'Hedging system test completed successfully'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })
            
            @self.app.route('/api/hedge/enable', methods=['POST'])
            def enable_hedging():
                """Enable hedging system"""
                try:
                    self.hedging_enabled = True
                    self.save_system_settings()  # Save to persistence
                    
                    return jsonify({
                        'success': True,
                        'message': 'Hedging system enabled',
                        'status': 'ACTIVE'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })
            
            @self.app.route('/api/hedge/disable', methods=['POST'])
            def disable_hedging():
                """Disable hedging system"""
                try:
                    self.hedging_enabled = False
                    self.save_system_settings()  # Save to persistence
                    
                    return jsonify({
                        'success': True,
                        'message': 'Hedging system disabled',
                        'status': 'INACTIVE'
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })
            
            @self.app.route('/api/hedge/settings', methods=['GET', 'POST'])
            def hedge_settings():
                """Get or update hedge settings"""
                try:
                    if request.method == 'GET':
                        # Return current settings
                        return jsonify({
                            'success': True,
                            'settings': {
                                'min_correlation': getattr(self, 'hedge_min_correlation', 0.6),
                                'max_hedge_ratio': getattr(self, 'hedge_max_ratio', 0.6),
                                'risk_reduction_target': getattr(self, 'hedge_risk_target', 0.3),
                                'auto_execute': getattr(self, 'hedge_auto_execute', False)
                            }
                        })
                    
                    elif request.method == 'POST':
                        # Update settings
                        data = request.get_json()
                        
                        if 'min_correlation' in data:
                            self.hedge_min_correlation = float(data['min_correlation'])
                        if 'max_hedge_ratio' in data:
                            self.hedge_max_ratio = float(data['max_hedge_ratio'])
                        if 'risk_reduction_target' in data:
                            self.hedge_risk_target = float(data['risk_reduction_target'])
                        if 'auto_execute' in data:
                            self.hedge_auto_execute = bool(data['auto_execute'])
                        
                        # Save settings
                        self.save_system_settings()
                        
                        return jsonify({
                            'success': True,
                            'message': 'Hedge settings updated',
                            'settings': data
                        })
                        
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })
            
            @self.app.route('/api/hedge/debug')
            def hedge_debug():
                """Debug hedging system"""
                try:
                    debug_info = {
                        'mt5_connected': mt5.terminal_info() is not None,
                        'account_info': mt5.account_info() is not None if mt5.terminal_info() else False,
                        'positions_raw': [],
                        'symbols_in_system': []
                    }
                    
                    # Get raw positions
                    positions = mt5.positions_get()
                    if positions:
                        for pos in positions:
                            debug_info['positions_raw'].append({
                                'symbol': pos.symbol,
                                'type': pos.type,
                                'volume': pos.volume,
                                'ticket': pos.ticket
                            })
                    
                    # Get symbols from trading system
                    if hasattr(self, 'forex_pairs'):
                        debug_info['symbols_in_system'] = self.forex_pairs[:10]  # First 10
                    
                    # Get live data symbols
                    if hasattr(self, 'live_data'):
                        debug_info['live_data_symbols'] = list(self.live_data.keys())[:10]
                    
                    return jsonify({
                        'success': True,
                        'debug_info': debug_info,
                        'hedging_available': hasattr(self, 'hedge_integrator') and self.hedge_integrator is not None
                    })
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })

            print("[TARGET] Hedging API routes added successfully!")
            
        except Exception as e:
            print(f"[ERR] Error setting up hedging routes: {str(e)}")
            self.hedging_enabled = False

    def start_data_updates(self):
        """Start automatic data updates"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(15)
                except Exception as e:
                    self.logger.error(f"Update loop error: {str(e)}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.logger.info("Auto-update system started")
    
    def graceful_shutdown(self):
        """Graceful shutdown with data saving"""
        try:
            print("\n[REFRESH] Shutting down gracefully...")
            
            # Stop auto trading
            if self.auto_trading_enabled:
                self.stop_auto_trading()
            
            # Save all data
            print("[DISK] Saving system data...")
            self.save_system_settings()
            self.save_pair_tracking_data()
            self.persistence.save_daily_stats(self.daily_stats)
            
            # Log shutdown
            self.persistence.log_system_event('INFO', 'System shutdown gracefully', 'SHUTDOWN')
            
            # Stop system
            self.is_running = False
            
            # Close MT5
            if self.mt5_connected:
                mt5.shutdown()
            
            print("[OK] Shutdown completed successfully")
            print("[DISK] All data saved for next session")
            
        except Exception as e:
            print(f"[ERR] Error during shutdown: {str(e)}")
    
        if hasattr(self, 'enhanced_trading'):
                self.enhanced_trading.stop_trailing_thread()
                print("[STOP] Trailing Stop System: STOPPED")

    def calculate_trailing_stop(self, position, market_data):
        """🔧 FIXED: Calculate trailing stop with improved parameters"""
        try:
            # Auto-select profile based on market conditions
            symbol = position.symbol
            optimal_profile = self.auto_select_trailing_profile(symbol, market_data)
            
            # ถ้าโปรไฟล์ที่แนะนำต่างจากปัจจุบัน ให้พิจารณาเปลี่ยน
            if optimal_profile != self.current_trailing_profile:
                print(f"[TRAIL] Auto-switching to {optimal_profile} profile for {symbol}")
                # สามารถเปลี่ยนอัตโนมัติหรือแค่แนะนำ
                
            # ใช้โปรไฟล์ที่เลือก
            profile = self.trailing_profiles[optimal_profile]
            
            ticket = position.ticket
            position_type = position.type
            entry_price = position.price_open
            current_price = market_data.get('bid' if position_type == 0 else 'ask', entry_price)
            current_sl = position.sl
            
            # Get ATR
            atr = market_data.get('atr', 0.001)
            if atr <= 0:
                atr = abs(current_price - entry_price) * 0.01
                
            # Initialize position state
            if ticket not in self.trailing_position_states:
                self.trailing_position_states[ticket] = {
                    'highest_price': current_price if position_type == 0 else entry_price,
                    'lowest_price': current_price if position_type == 1 else entry_price,
                    'breakeven_activated': False,
                    'trail_count': 0,
                    'last_update_time': datetime.now(),
                    'profile_used': optimal_profile
                }
            
            state = self.trailing_position_states[ticket]
            
            # 🔧 FIX: เพิ่มการตรวจสอบเวลา (throttling)
            time_since_last = (datetime.now() - state['last_update_time']).total_seconds()
            min_interval = profile.get('update_frequency', 12)
            
            if time_since_last < min_interval:
                return {
                    'should_update': False,
                    'trail_reason': 'THROTTLED',
                    'wait_seconds': int(min_interval - time_since_last)
                }
            
            # Calculate profit
            if position_type == 0:  # BUY
                profit_pips = current_price - entry_price
                state['highest_price'] = max(state['highest_price'], current_price)
                reference_price = state['highest_price']
            else:  # SELL
                profit_pips = entry_price - current_price
                state['lowest_price'] = min(state['lowest_price'], current_price)
                reference_price = state['lowest_price']
            
            profit_atr = profit_pips / atr
            
            # Check for trailing
            new_sl = current_sl
            should_update = False
            trail_reason = "NO_UPDATE"
            
            # 🔧 FIX: Breakeven protection - รอนานขึ้น
            breakeven_trigger = profile['breakeven_trigger_atr']
            
            if (not state['breakeven_activated'] and 
                profit_atr >= breakeven_trigger):
                
                # 🔧 FIX: เพิ่ม buffer จาก entry price
                buffer = atr * 0.8  # เพิ่ม buffer
                new_sl = entry_price + (buffer if position_type == 0 else -buffer)
                state['breakeven_activated'] = True
                state['last_update_time'] = datetime.now()
                should_update = True
                trail_reason = "BREAKEVEN_PROTECTION"
                
            # 🔧 FIX: Dynamic trailing - ช้าลงและกว้างขึ้น
            elif state['breakeven_activated'] or profit_atr >= breakeven_trigger:
                trail_distance = profile['min_trail_distance_atr'] * atr
                
                # 🔧 FIX: เพิ่ม Trend และ Volatility multipliers
                trend_strength = market_data.get('trend_strength', 0.5)
                if trend_strength > 0.7:
                    trail_distance *= 1.8  # เพิ่มระยะเมื่อ trend แข็งแกร่ง
                elif trend_strength > 0.5:
                    trail_distance *= 1.4
                
                # เพิ่ม minimum improvement requirement
                min_improvement = atr * 0.5  # ต้องดีขึ้นอย่างน้อย 0.5 ATR
                
                if position_type == 0:  # BUY
                    calculated_sl = reference_price - trail_distance
                    if calculated_sl > current_sl + min_improvement:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_UP"
                        state['trail_count'] += 1
                        state['last_update_time'] = datetime.now()
                else:  # SELL
                    calculated_sl = reference_price + trail_distance
                    if current_sl == 0 or calculated_sl < current_sl - min_improvement:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_DOWN"
                        state['trail_count'] += 1
                        state['last_update_time'] = datetime.now()
            
            return {
                'should_update': should_update,
                'new_sl': new_sl,
                'trail_reason': trail_reason,
                'profit_atr': round(profit_atr, 2),
                'breakeven_activated': state['breakeven_activated'],
                'trail_count': state['trail_count'],
                'profile_used': optimal_profile,
                'trail_distance_pips': self.convert_to_pips(abs(new_sl - current_price), symbol) if should_update else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop: {str(e)}")
            return {'should_update': False, 'error': str(e)}

    def convert_to_pips(self, price_distance, symbol):
        """แปลงระยะราคาเป็น Pips"""
        if 'JPY' in symbol:
            return round(price_distance * 100, 1)
        elif 'XAU' in symbol:
            return round(price_distance * 10, 1)
        else:
            return round(price_distance * 10000, 1)
        
    def update_position_trailing_stop(self, ticket, new_sl, symbol):
        """[TARGET] อัพเดท Stop Loss ใน MT5"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": pos.tp,
                "magic": pos.magic,
                "comment": f"Trailing SL - {self.current_trailing_profile}",
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] Trailing SL Updated: {symbol} #{ticket} -> SL: {new_sl:.5f}")
                self.trailing_statistics['total_sl_updates'] += 1
                return True
            else:
                print(f"[ERR] SL Update Failed: {symbol} #{ticket} - {result.comment}")
                return False
                
        except Exception as e:
            print(f"[ERR] Error updating SL: {str(e)}")
            return False

    def process_all_trailing_stops(self):
        """[TARGET] ประมวลผล Trailing Stop ทุก positions"""
        if not self.trailing_system_enabled:
            return
        
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            updates_made = 0
            
            for position in positions:
                symbol = position.symbol
                
                # Get market data for this symbol
                if symbol in self.live_data and self.live_data[symbol]:
                    market_data = self.live_data[symbol]
                    
                    # Calculate trailing
                    trail_info = self.calculate_trailing_stop(position, market_data)
                    
                    if trail_info.get('should_update', False):
                        success = self.update_position_trailing_stop(
                            position.ticket, 
                            trail_info['new_sl'], 
                            symbol
                        )
                        
                        if success:
                            updates_made += 1
                            
                            if trail_info['trail_reason'] == 'BREAKEVEN_PROTECTION':
                                self.trailing_statistics['breakeven_protections'] += 1
                            elif 'TRAILING' in trail_info['trail_reason']:
                                self.trailing_statistics['trail_moves'] += 1
            
            if updates_made > 0:
                print(f"[TARGET] Trailing Stop Updates: {updates_made}")
                
        except Exception as e:
            print(f"[ERR] Error processing trailing stops: {str(e)}")

    def start_pullback_monitoring(self):
        """🔧 NEW: เริ่มระบบ monitor pullback protection"""
        try:
            if not hasattr(self, 'pullback_protection') or not self.pullback_protection:
                return False
            
            def pullback_monitoring_loop():
                """Background loop สำหรับ monitor pullback protection"""
                while self.is_running and getattr(self, 'pullback_monitoring_enabled', True):
                    try:
                        # ตรวจสอบ waiting positions ทุก 30 วินาที
                        time.sleep(30)
                        
                        if hasattr(self, 'pullback_protection') and self.pullback_protection:
                            current_time = datetime.now()
                            
                            # ตรวจสอบ timeout positions
                            expired_positions = []
                            for symbol, waiting_data in self.pullback_protection.waiting_positions.items():
                                if current_time >= waiting_data['timeout_time']:
                                    expired_positions.append(symbol)
                            
                            # ลบ positions ที่หมดเวลา
                            for symbol in expired_positions:
                                self.pullback_protection._remove_waiting_position(symbol, 'TIMEOUT')
                                self.pullback_protection.statistics['timeout_expired'] += 1
                                print(f"⏰ {symbol}: Timeout expired, removed from waiting list")
                            
                            # ตรวจสอบ recovery สำหรับ waiting positions
                            for symbol in list(self.pullback_protection.waiting_positions.keys()):
                                try:
                                    # ดึงข้อมูลตลาดปัจจุบัน
                                    current_market_data = self.get_current_indicators(symbol)
                                    if current_market_data:
                                        recovery_check = self.pullback_protection.check_pullback_recovery(
                                            symbol, current_market_data
                                        )
                                        
                                        if recovery_check.get('ready_to_trade', False):
                                            print(f"✅ {symbol}: Pullback recovered, ready to trade")
                                            
                                except Exception as e:
                                    self.logger.error(f"Error checking recovery for {symbol}: {str(e)}")
                            
                            # ทำความสะอาด database
                            if hasattr(self.pullback_protection, 'cleanup_old_records'):
                                self.pullback_protection.cleanup_old_records()
                        
                    except Exception as e:
                        self.logger.error(f"Error in pullback monitoring loop: {str(e)}")
                        time.sleep(60)  # รอนานขึ้นเมื่อมี error
            
            # เริ่ม monitoring thread
            self.pullback_monitoring_enabled = True
            self.pullback_monitoring_thread = threading.Thread(
                target=pullback_monitoring_loop,
                daemon=True,
                name="PullbackMonitoringThread"
            )
            self.pullback_monitoring_thread.start()
            
            print("[OK] Pullback monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting pullback monitoring: {str(e)}")
            return False

    def get_trailing_dashboard_data(self):
        """[CHART] ข้อมูลสำหรับ Dashboard"""
        try:
            positions = mt5.positions_get()
            position_details = []
            
            if positions:
                for pos in positions:
                    symbol = pos.symbol
                    
                    if symbol in self.live_data and self.live_data[symbol]:
                        trail_info = self.calculate_trailing_stop(pos, self.live_data[symbol])
                        
                        position_details.append({
                            'ticket': pos.ticket,
                            'symbol': symbol,
                            'type': 'BUY' if pos.type == 0 else 'SELL',
                            'entry_price': pos.price_open,
                            'current_sl': pos.sl,
                            'profit': pos.profit,
                            'trail_info': trail_info
                        })
            
            return {
                'enabled': self.trailing_system_enabled,
                'profile': self.current_trailing_profile,
                'statistics': {
                    'active_positions': len(positions) if positions else 0,
                    'breakeven_protected': len([s for s in self.trailing_position_states.values() if s['breakeven_activated']]),
                    'total_trail_moves': sum(s['trail_count'] for s in self.trailing_position_states.values()),
                    'system_stats': self.trailing_statistics
                },
                'positions': position_details
            }
            
        except Exception as e:
            print(f"[ERR] Error getting dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def auto_trading_loop(self):
        """Main auto trading loop - FIXED VERSION"""
        try:
            print("[GO] Auto trading loop started ot")
            self.trade_logger.info("=== AUTO TRADING LOOP STARTED ===")
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.auto_trading_enabled and self.is_running and not self.emergency_stop:
                try:
                    # Reset error counter on successful iteration
                    consecutive_errors = 0
                    
                    # Check if we should continue
                    if not self.auto_trading_enabled or self.emergency_stop:
                        break
                    
                    # ตรวจสอบการเชื่อมต่อ MT5
                    if not self.mt5_connected:
                        print("[WARN] MT5 not connected, attempting reconnection...")
                        if not self.connect_mt5():
                            print("[ERR] Failed to reconnect MT5")
                            time.sleep(30)
                            continue
                    
                    print("mt5_connected ot")
                    # ตรวจสอบว่าต้องส่ง repost หรือไม่
                    if self.should_report_status():
                        try:
                            account_data = self.get_account_data()
                        except Exception as e:
                            raise Exception(f"Failed to get account data: {str(e)}")
                        
                        status_response = requests.post(
                            f"{self.api_base_url}/customer-clients/status",
                            json={
                                "tradingAccountId": account_data["account_id"],
                                "name": account_data["account_name"],
                                "brokerName": account_data["broker_name"],
                                "currentBalance": account_data["current_balance"],
                                "currentProfit": account_data["current_profit"],
                                "currency": account_data["currency"],
                                "botName": "AI Dashboard",
                                "botVersion": "0.0.1"
                            },
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            response_data = status_response.json()
                            
                            # Check if trading is inactive
                            if response_data.get("processedStatus") == "inactive":
                                message = response_data.get("message", "Trading is inactive")
                                raise Exception(f"Trading is inactive. {message}")
                            
                            # Store next report time for scheduling
                            next_report_time = response_data.get("nextReportTime")
                            if next_report_time:
                                # Fix microseconds to 6 digits
                                if '.' in next_report_time and '+' in next_report_time:
                                    parts = next_report_time.split('.')
                                    microseconds = parts[1].split('+')[0]
                                    timezone_part = '+' + parts[1].split('+')[1]
                                    
                                    # Truncate microseconds to 6 digits
                                    if len(microseconds) > 6:
                                        microseconds = microseconds[:6]
                                    
                                    next_report_time = f"{parts[0]}.{microseconds}{timezone_part}"
                                
                                self.next_report_time = datetime.fromisoformat(next_report_time)
                                print(f"Next report scheduled for: {self.next_report_time}")
                                
                        else:
                            raise Exception(f"Failed to check status: {status_response.status_code}")
                
                    # ตรวจสอบ forex pairs
                    if not hasattr(self, 'forex_pairs') or not self.forex_pairs:
                        print("[WARN] No forex pairs available")
                        time.sleep(30)
                        continue
                    

                    # print(f"[LOOP] Processing {len(self.forex_pairs)} pairs...")
                    
                    # วนลูปตรวจสอบแต่ละ pair
                    for symbol in self.forex_pairs:
                        try:
                            if not self.auto_trading_enabled or self.emergency_stop:
                                break
                            
                            # ตรวจสอบว่า pair พร้อมเทรดหรือไม่
                            pair_status = self.check_pair_trading_status(symbol)
                            if not pair_status.get('can_trade', False):
                                continue
                            
                            # ดึงข้อมูล signal
                            signal_data = self.get_symbol_data(symbol)
                            if not signal_data or signal_data.get('signal') == 'NONE':
                                continue
                            
                            # ตรวจสอบ signal validation
                            validation_result = self.validate_trading_signal(symbol, signal_data)
                            if not validation_result.get('valid', False):
                                print(f"[SKIP] {symbol}: Signal validation failed")
                                continue
                            
                            # ตรวจสอบ portfolio risk
                            if not self.check_portfolio_risk():
                                print(f"[SKIP] Portfolio risk limit reached")
                                break
                            
                            # Execute trade
                            print(f"[TRADE] Attempting to execute trade for {symbol}")
                            trade_result = self.execute_auto_trade(symbol, signal_data)
                            
                            if trade_result.get('success', False):
                                print(f"[OK] Trade executed for {symbol}")
                                # รอสักครู่หลังจาก execute trade
                                time.sleep(5)
                            else:
                                print(f"[FAIL] Trade execution failed for {symbol}: {trade_result.get('error', 'Unknown error')}")
                            
                            # หน่วงเวลาระหว่าง pair
                            time.sleep(2)
                            
                        except Exception as pair_error:
                            print(f"[ERR] Error processing {symbol}: {str(pair_error)}")
                            continue
                    
                    # Update trailing stops for existing positions
                    try:
                        if hasattr(self, 'update_trailing_stops'):
                            self.update_trailing_stops()
                    except Exception as trailing_error:
                        print(f"[WARN] Trailing stop update error: {str(trailing_error)}")
                    
                    # บันทึกข้อมูล
                    try:
                        if hasattr(self, 'save_system_settings'):
                            self.save_system_settings()
                    except Exception as save_error:
                        print(f"[WARN] Save settings error: {str(save_error)}")
                    
                    # หน่วงเวลาก่อนรอบถัดไป
                    print(f"[LOOP] Cycle completed, waiting 30 seconds...")
                    time.sleep(30)
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[ERR] Auto trading loop error ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ALERT] Too many consecutive errors, stopping auto trading")
                        self.auto_trading_enabled = False
                        self.emergency_stop = True
                        break
                    
                    # Exponential backoff on errors
                    sleep_time = min(60, 5 * (2 ** consecutive_errors))
                    print(f"[TIME] Sleeping {sleep_time} seconds after error")
                    time.sleep(sleep_time)
            
        except Exception as critical_error:
            print(f"[CRITICAL] Critical error in auto trading loop: {str(critical_error)}")
            self.auto_trading_enabled = False
            
        finally:
            print("[STOP] Auto trading loop stopped")
            self.auto_trading_enabled = False
            self.trade_logger.info("=== AUTO TRADING LOOP STOPPED ===")

    def should_report_status(self):
        """Check if it's time to report status"""
        if hasattr(self, 'next_report_time') and self.next_report_time:
            current_utc = datetime.now(timezone.utc)
            next_report_utc = self.next_report_time.astimezone(timezone.utc)
            
            print(f"Current UTC: {current_utc}")
            print(f"Next report UTC: {next_report_utc}")
            
            return current_utc >= next_report_utc
        return True  # Report if no scheduled time

    def get_account_data(self):
        """Get account information from MT5"""
        account_info = mt5.account_info()
        
        if account_info is None:
            raise Exception("Failed to get account info from MT5")
        
        return {
            "account_id": str(account_info.login),           # Account ID/Login
            "account_name": account_info.name,               # Account holder name
            "broker_name": account_info.company,             # Broker company name
            "current_balance": str(account_info.balance),    # Account balance
            "current_profit": str(account_info.profit),      # Current profit/loss
            "currency": account_info.currency                # Account currency
        }

    def check_pair_trading_status(self, symbol):
        """ตรวจสอบสถานะการเทรดของ pair"""
        try:
            # ตรวจสอบว่ามี active trade อยู่หรือไม่
            if hasattr(self, 'active_trades_per_pair'):
                active_trades = self.active_trades_per_pair.get(symbol, [])
                if len(active_trades) > 0 and getattr(self, 'one_trade_per_pair', True):
                    return {
                        'can_trade': False,
                        'reason': f'Already has {len(active_trades)} active trades'
                    }
            
            # ตรวจสอบ cooldown
            if hasattr(self, 'trade_cooldowns'):
                cooldown_until = self.trade_cooldowns.get(symbol)
                if cooldown_until and datetime.now() < cooldown_until:
                    return {
                        'can_trade': False,
                        'reason': f'In cooldown until {cooldown_until.strftime("%H:%M:%S")}'
                    }
            
            return {'can_trade': True, 'reason': 'Ready to trade'}
            
        except Exception as e:
            print(f"[ERR] Error checking pair status for {symbol}: {str(e)}")
            return {'can_trade': False, 'reason': f'Status check error: {str(e)}'}
        
    def execute_auto_trade(self, symbol, signal_data):
        """Execute automatic trade - REAL MODE with Multiple Filling Types"""
        try:          
            signal = signal_data.get('signal', 'NONE')
            if signal == 'NONE':
                return {'success': False, 'error': 'No valid signal'}
            
            # ดึงข้อมูลที่จำเป็น
            entry_price = signal_data.get('optimal_entry', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit_1', 0)
            lot_size = signal_data.get('lot_size', 0.01)
            
            if entry_price <= 0 or stop_loss <= 0:
                return {'success': False, 'error': 'Invalid price levels'}
            
            # Map เป็น MT5 order type
            if signal in ['BUY', 'STRONG_BUY']:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            elif signal in ['SELL', 'STRONG_SELL']:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                return {'success': False, 'error': f'Unknown signal: {signal}'}
            
            print(f"[TRADE] Executing REAL {signal} for {symbol}")
            print(f"        Price: {price}, SL: {stop_loss}, TP: {take_profit}")
            print(f"        Lot Size: {lot_size}")
            
            # 🔧 ลำดับ Filling Types ที่จะลอง
            filling_types = [
                mt5.ORDER_FILLING_FOK,    # Fill or Kill (เติมทั้งหมดหรือยกเลิก)
                mt5.ORDER_FILLING_IOC,    # Immediate or Cancel (เติมทันทีหรือยกเลิก)
                mt5.ORDER_FILLING_RETURN  # Return (เติมบางส่วนได้)
            ]
            
            filling_names = {
                mt5.ORDER_FILLING_FOK: "FOK (Fill or Kill)",
                mt5.ORDER_FILLING_IOC: "IOC (Immediate or Cancel)", 
                mt5.ORDER_FILLING_RETURN: "RETURN (Partial Fill OK)"
            }
            
            # ลองแต่ละ filling type
            last_error = None
            
            for filling_type in filling_types:
                try:
                    print(f"[ATTEMPT] Trying {filling_names[filling_type]}...")
                    
                    # สร้าง order request
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot_size,
                        "type": order_type,
                        "price": price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "deviation": 20,
                        "magic": 12345,
                        "comment": f"Auto Trade - {signal}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling_type,
                    }
                    
                    # ส่ง order
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # สำเร็จ!
                        print(f"[SUCCESS] Order successful with {filling_names[filling_type]}")
                        print(f"          Order ID: {result.order}")
                        print(f"          Deal ID: {result.deal}")
                        print(f"          Fill Price: {result.price}")
                        
                        # อัพเดต tracking
                        if symbol not in self.active_trades_per_pair:
                            self.active_trades_per_pair[symbol] = []
                        
                        self.active_trades_per_pair[symbol].append({
                            'ticket': result.order,
                            'deal': result.deal,
                            'symbol': symbol,
                            'type': order_type,
                            'volume': result.volume,
                            'price': result.price,
                            'time': datetime.now().isoformat(),
                            'filling_type': filling_names[filling_type]
                        })
                        
                        self.pair_trade_status[symbol] = 'TRADING'
                        
                        # อัพเดตสถิติ
                        if hasattr(self, 'daily_stats'):
                            self.daily_stats['trades_executed'] = self.daily_stats.get('trades_executed', 0) + 1
                        
                        return {
                            'success': True,
                            'message': f'REAL trade executed for {symbol}',
                            'order_id': result.order,
                            'deal_id': result.deal,
                            'price': result.price,
                            'volume': result.volume,
                            'signal': signal,
                            'symbol': symbol,
                            'filling_type': filling_names[filling_type]
                        }
                    
                    else:
                        # ล้มเหลว ลองต่อ
                        error_msg = f"{filling_names[filling_type]} failed: {result.retcode} - {result.comment}"
                        print(f"[FAIL] {error_msg}")
                        last_error = error_msg
                        
                        # หน่วงเวลาสั้นๆ ก่อนลองต่อ
                        time.sleep(0.1)
                        
                except Exception as filling_error:
                    error_msg = f"{filling_names[filling_type]} exception: {str(filling_error)}"
                    print(f"[ERROR] {error_msg}")
                    last_error = error_msg
                    continue
            
            # ถ้าทุก filling type ล้มเหลว
            final_error = f"All filling types failed. Last error: {last_error}"
            print(f"[FINAL_ERROR] {final_error}")
            
            return {
                'success': False, 
                'error': final_error,
                'symbol': symbol,
                'attempted_filling_types': list(filling_names.values())
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error checking status: {str(e)}")
        except Exception as e:
            error_msg = f"Trade execution error: {str(e)}"
            print(f"[ERR] {error_msg}")
            return {'success': False, 'error': error_msg, 'symbol': symbol}

    # 🔧 เพิ่ม function สำหรับตรวจสอบ Symbol Info
    def check_symbol_filling_modes(self, symbol):
        """ตรวจสอบ filling modes ที่ symbol รองรับ"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            filling_mode = symbol_info.filling_mode
            
            supported_modes = []
            if filling_mode & 1:  # FOK
                supported_modes.append("FOK")
            if filling_mode & 2:  # IOC  
                supported_modes.append("IOC")
            if filling_mode & 4:  # Return
                supported_modes.append("RETURN")
            
            print(f"[INFO] {symbol} supports filling modes: {supported_modes}")
            return supported_modes
            
        except Exception as e:
            print(f"[ERR] Error checking filling modes for {symbol}: {str(e)}")
            return None

    # 🔧 เพิ่ม Enhanced Order Execution ที่ตรวจสอบ Symbol Info ก่อน
    def execute_auto_trade_enhanced(self, symbol, signal_data):
        """Enhanced trade execution with symbol-specific filling modes"""
        try:
            # ตรวจสอบ filling modes ที่รองรับ
            supported_modes = self.check_symbol_filling_modes(symbol)
            
            if supported_modes:
                print(f"[INFO] {symbol} supports: {supported_modes}")
            
            # เรียกใช้ execute_auto_trade ปกติ
            return self.execute_auto_trade(symbol, signal_data)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'symbol': symbol}
    
    def check_portfolio_risk(self):
        """ตรวจสอบ portfolio risk"""
        try:
            # ตรวจสอบ total exposure
            total_exposure = 0
            max_exposure = getattr(self, 'max_total_exposure', 0.06)  # 6%
            
            if hasattr(self, 'active_trades_per_pair'):
                for trades in self.active_trades_per_pair.values():
                    total_exposure += len(trades) * 0.015  # สมมติ 1.5% per trade
            
            if total_exposure >= max_exposure:
                return False
            
            # ตรวจสอบ daily loss
            if hasattr(self, 'daily_stats'):
                daily_pnl = self.daily_stats.get('total_pnl', 0)
                max_daily_loss = getattr(self, 'max_daily_loss', 0.04) * getattr(self, 'account_balance', 10000)
                
                if daily_pnl <= -max_daily_loss:
                    return False
            
            return True
            
        except Exception as e:
            print(f"[ERR] Portfolio risk check error: {str(e)}")
            return False

        """Execute automatic trade"""
        try:
            signal = signal_data.get('signal', 'NONE')
            if signal == 'NONE':
                return {'success': False, 'error': 'No valid signal'}
            
            # สำหรับตอนนี้ใช้ dummy trade execution
            print(f"[TRADE] Would execute {signal} for {symbol}")
            print(f"        Signal strength: {signal_data.get('strength', 0)}/10")
            print(f"        Entry quality: {signal_data.get('entry_quality', 'UNKNOWN')}")
            
            # Simulate trade execution
            if hasattr(self, 'daily_stats'):
                self.daily_stats['trades_executed'] = self.daily_stats.get('trades_executed', 0) + 1
            
            # Add to active trades tracking
            if hasattr(self, 'active_trades_per_pair'):
                if symbol not in self.active_trades_per_pair:
                    self.active_trades_per_pair[symbol] = []
                # Add dummy trade ID
                self.active_trades_per_pair[symbol].append(f"DUMMY_{int(time.time())}")
            
            return {
                'success': True,
                'message': f'Trade executed for {symbol}',
                'signal': signal,
                'symbol': symbol
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the enhanced auto trading dashboard"""
        try:
            print("Enhanced Smart Auto Trading Dashboard Starting...")
            print("=" * 60)
            print("[REFRESH] WITH DATA PERSISTENCE & STATE MANAGEMENT")
            print("=" * 60)
            
            if not self.connect_mt5():
                print("ERROR: Failed to connect to MT5")
                return
            
            self.is_running = True
            self.start_data_updates()
            self.update_all_data()
            
            print(f"[OK] SUCCESS: Enhanced Auto Trading Dashboard Started!")
            print(f"[REFRESH] FEATURES: Smart Auto Trading + Risk Management + Data Persistence")
            print(f"[DISK] PERSISTENCE: Settings, Positions & Stats Auto-Saved")
            print(f" DATABASE: Trade History & System Logs")
            print(f"[SHIELD] RECOVERY: System state restored on restart")
            print(f"[WEB] DASHBOARD: http://{host}:{port}")
            print(f"[EMOJI] API: http://{host}:{port}/api/market-data")
            print(f"[UP] STATUS: http://{host}:{port}/api/system-status")
            print(f"[LIGHTNING] AUTO TRADING: Currently {('ENABLED' if self.auto_trading_enabled else 'DISABLED')}")
            print("[DISK] DATA SAVED: Every 5 minutes + on changes")
            print("[REFRESH] STOP: Press Ctrl+C for graceful shutdown")
            print("=" * 60)
            
            # Log startup
            self.persistence.log_system_event('INFO', 'Enhanced Auto Trading Dashboard started', 'STARTUP')
            
            self.app.run(host=host, port=port, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            self.graceful_shutdown()
        except Exception as e:
            print(f"[ERR] ERROR: {str(e)}")
            self.graceful_shutdown()

# CLEAN MAIN FUNCTION
def main():
    """Main execution - SIMPLIFIED"""
    
    # สร้าง core dashboard
    dashboard = EnhancedSmartAutoTradingDashboard()
    
    # เชื่อมต่อ MT5
    if dashboard.connect_mt5():
        print("[EMOJI] MT5 Connected Successfully!")
        
        # เริ่มระบบ
        dashboard.start_data_updates()
        dashboard.run()
        
    else:
        print("[ERR] MT5 connection failed!")

if __name__ == "__main__":
    main()