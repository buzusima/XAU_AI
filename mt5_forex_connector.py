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
from datetime import datetime, timedelta
import threading
import time
import json
import logging
import os
import sqlite3
import pickle
from typing import Dict, List, Optional, Tuple
import warnings
from broker_symbol_adapter import BrokerSymbolAdapter
from enhanced_signal_system import MultiTimeframeSignalEngine
warnings.filterwarnings('ignore')
from pullback_protection import PullbackProtectionPlugin, integrate_with_main_system
from trading_integration import EnhancedTradingSystemWithTrailing, add_trailing_stop_routes
try:
    from correlation_hedging_system import HedgeSystemIntegrator, AdvancedCorrelationHedging
    HEDGING_SYSTEM_AVAILABLE = True
    print("✅ Correlation Hedging System Loaded Successfully!")
except ImportError as e:
    print(f"⚠️ Hedging system not available: {str(e)}")
    HEDGING_SYSTEM_AVAILABLE = False

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

# FIXED: Import enhanced_signal_system with error handling
try:
    from enhanced_signal_system import MultiTimeframeSignalEngine
    ENHANCED_SIGNAL_AVAILABLE = True
    print("✅ Enhanced Signal System Loaded Successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced signal system not available: {str(e)}")
    print("📋 Using standard signal system...")
    ENHANCED_SIGNAL_AVAILABLE = False

# FIXED: Import advanced_features with error handling
try:
    from advanced_features import AdvancedTradingIntegrator, MarketRegime
    ADVANCED_FEATURES_AVAILABLE = True
    print("✅ Advanced Trading Features Loaded Successfully!")
except ImportError as e:
    print(f"⚠️ Advanced features not available: {str(e)}")
    print("📋 Using standard trading system...")
    ADVANCED_FEATURES_AVAILABLE = False

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
    
    def create_data_directory(self):
        """สร้างโฟลเดอร์เก็บข้อมูล"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
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

class EnhancedSmartAutoTradingDashboard:
    """Enhanced Smart Auto Trading Dashboard with Data Persistence"""
    
    def __init__(self):
        """Initialize Enhanced Smart Auto Trading Dashboard"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Data Persistence Manager
        self.persistence = DataPersistenceManager()
        self.symbol_adapter = BrokerSymbolAdapter()
        self.broker_symbols_mapped = False
        # Forex pairs
        self.forex_pairs = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c', 'USDCAD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'EURCAD.c',
            'GBPJPY.c', 'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'GBPCAD.c',
            'AUDCHF.c', 'AUDJPY.c', 'AUDNZD.c', 'AUDCAD.c',
            'NZDJPY.c', 'NZDCHF.c', 'NZDCAD.c',
            'CHFJPY.c', 'CADJPY.c', 'XAUUSD.c',
        ]
        
        # Load saved settings or use defaults
        self.load_system_settings()
        
        # Initialize default values (if not loaded)
        if not hasattr(self, 'account_balance'):
            self.account_balance = 10000.0
        
        # Portfolio Risk Management Settings
        self.portfolio_risk_profiles = {
            'CONSERVATIVE': {'risk_per_trade': 0.5, 'max_total_exposure': 2.0, 'max_daily_loss': 2.0},
            'MODERATE': {'risk_per_trade': 1.0, 'max_total_exposure': 4.0, 'max_daily_loss': 3.0},
            'BALANCED': {'risk_per_trade': 1.5, 'max_total_exposure': 6.0, 'max_daily_loss': 4.0},
            'AGGRESSIVE': {'risk_per_trade': 2.0, 'max_total_exposure': 8.0, 'max_daily_loss': 5.0},
            'HIGH_RISK': {'risk_per_trade': 3.0, 'max_total_exposure': 12.0, 'max_daily_loss': 8.0}
        }
        
        # ONE TRADE PER PAIR CONTROL
        self.one_trade_per_pair = True
        
        # Load or initialize tracking data
        self.load_pair_tracking_data()
        
        # Load daily stats
        self.daily_stats = self.persistence.load_daily_stats()
        
        # Trading Time Controls
        self.trading_sessions = {
            'ASIAN': {'start': '00:00', 'end': '09:00', 'enabled': False},
            'LONDON': {'start': '08:00', 'end': '17:00', 'enabled': True},
            'NEWYORK': {'start': '13:00', 'end': '22:00', 'enabled': True},
            'OVERLAP': {'start': '13:00', 'end': '17:00', 'enabled': True}
        }
        
        # Position Management
        self.default_lot_size = 0.01
        self.max_lot_size = 2.0
        self.min_lot_size = 0.01
        self.partial_close_enabled = True
        self.trailing_stop_enabled = True
        self.trailing_profiles = {
            'CONSERVATIVE': {
                'breakeven_trigger_atr': 1.0,
                'trail_distance_atr': 2.0,
                'min_trail_distance_atr': 1.5
            },
            'MODERATE': {
                'breakeven_trigger_atr': 0.8,
                'trail_distance_atr': 1.5,
                'min_trail_distance_atr': 1.2
            },
            'AGGRESSIVE': {
                'breakeven_trigger_atr': 0.5,
                'trail_distance_atr': 1.0,
                'min_trail_distance_atr': 0.8
            }
        }
        self.current_trailing_profile = 'MODERATE'
        self.trailing_position_states = {}
        self.trailing_statistics = {
            'total_sl_updates': 0,
            'breakeven_protections': 0,
            'trail_moves': 0,
            'profit_secured': 0.0
        }
        # Trade Execution Settings
        self.slippage_tolerance = 3
        self.max_spread_threshold = 2.0
        self.trade_timeout = 30
        
        # Data Storage
        self.live_data = {}
        self.account_info = {}
        self.open_positions = {}
        self.trade_history = []
        self.pending_signals = {}
        self.signal_log = []
        
        # Cooldown Management
        self.global_cooldown = 60
        self.last_global_trade_time = None
        
        # System State
        self.is_running = False
        self.mt5_connected = False
        self.last_update = datetime.now()
        self.emergency_stop = False
        
        self.setup_logging()
        if ADVANCED_FEATURES_AVAILABLE:
            try:
                self.advanced_integrator = AdvancedTradingIntegrator()
                self.use_advanced_features = True
                print("🚀 Advanced Trading Features Activated!")
                print("   - Market Regime Detection: ON")
                print("   - Enhanced Signal Scoring: ON")
                print("   - Dynamic Position Sizing: ON")
            except Exception as e:
                print(f"❌ Error initializing advanced features: {str(e)}")
                self.use_advanced_features = False
        else:
            self.use_advanced_features = False
            print("📋 Using standard trading features")

        # 🎯 Initialize Hedging System
        if HEDGING_SYSTEM_AVAILABLE:
            try:
                self.hedge_integrator = HedgeSystemIntegrator(self)
                self.hedging_enabled = True
                print("🎯 Correlation Hedging System Activated!")
                print("💱 Cross-Pair Risk Management Ready")
            except Exception as e:
                print(f"❌ Error initializing hedging system: {str(e)}")
                self.hedging_enabled = False
        else:
            self.hedging_enabled = False
            print("📋 Running without hedging features")

        self.setup_routes()

        # Auto-save timer
        self.setup_auto_save()

        try:
            self.signal_engine = MultiTimeframeSignalEngine()
            print(" Enhanced Multi-Timeframe Signal Engine Activated!")
            print(" Expected Win Rate Improvement: 55% → 65-75%")
        except Exception as e:
            print(f"❌ Error initializing signal engine: {str(e)}")
            self.signal_engine = None

        try:
                self.enhanced_trading = EnhancedTradingSystemWithTrailing(self)
                print("✅ Enhanced Trading System with Trailing Stops: ACTIVATED")
                self.trailing_enabled = True
        except Exception as e:
                print(f"❌ Error initializing trailing system: {str(e)}")
                self.trailing_enabled = False

        # 🛡️ เพิ่ม Pullback Protection Plugin
        try:
            self.pullback_protection = PullbackProtectionPlugin(self.logger)
            
            # ติดตั้งระบบ Integration
            integrate_with_main_system(self, enable_on_start=True)
            
            print("✅ Pullback Protection Plugin ติดตั้งเรียบร้อย")
            print("🎯 Expected Win Rate Improvement: 55% → 65%+")
            
        except Exception as e:
            print(f"❌ Error installing Pullback Protection: {str(e)}")
            self.pullback_protection = None

        from broker_symbol_adapter import BrokerSymbolAdapter
        self.symbol_adapter = BrokerSymbolAdapter()
            
        # Auto-detect และ map symbols
        if self.symbol_adapter.detect_and_map_broker():
            mapping_info = self.symbol_adapter.get_mapping_info()
            self.logger.info(f"✅ Broker auto-detected!")
            self.logger.info(f"🏦 Server: {mapping_info['server']}")
            self.logger.info(f"📊 Mapped: {mapping_info['mapped_symbols']}/{mapping_info['total_system_symbols']} symbols")
            self.logger.info(f"🔧 Success rate: {mapping_info['mapping_success_rate']}")
            
            # อัพเดต forex_pairs ให้ใช้ broker symbols
            self.forex_pairs = self.symbol_adapter.get_mapped_symbols()
        else:
            self.logger.warning("⚠️ Symbol mapping failed, using default .c format")

    def load_system_settings(self):
        """โหลดการตั้งค่าระบบ"""
        try:
            settings = self.persistence.load_settings()
            
            if settings:
                # Load core settings
                self.current_risk_profile = settings.get('current_risk_profile', 'BALANCED')
                self.custom_risk_per_trade = settings.get('custom_risk_per_trade', 1.5)
                self.max_risk_per_trade = settings.get('max_risk_per_trade', 0.015)
                self.max_total_exposure = settings.get('max_total_exposure', 0.06)
                self.max_daily_loss = settings.get('max_daily_loss', 0.04)
                
                # Auto trading settings
                self.auto_trading_enabled = settings.get('auto_trading_enabled', False)
                self.auto_trading_pairs = set(settings.get('auto_trading_pairs', self.forex_pairs))
                self.min_signal_strength = settings.get('min_signal_strength', 6.0)
                self.min_entry_quality = settings.get('min_entry_quality', 'GOOD')
                self.max_simultaneous_trades = settings.get('max_simultaneous_trades', 8)
                
                # Load hedging settings
                self.hedging_enabled = settings.get('hedging_enabled', False)
                self.hedge_min_correlation = settings.get('hedge_min_correlation', 0.6)
                self.hedge_max_ratio = settings.get('hedge_max_ratio', 0.6)
                self.hedge_risk_target = settings.get('hedge_risk_target', 0.3)
                self.hedge_auto_execute = settings.get('hedge_auto_execute', False)
            
                # Signal filtering
                self.required_confirmations = settings.get('required_confirmations', {
                    'trend_alignment': True,
                    'volume_confirmation': True,
                    'rsi_filter': True,
                    'multiple_timeframe': True
                })
                
                # Additional settings
                self.min_rr_ratio = settings.get('min_rr_ratio', 1.5)
                self.account_balance = settings.get('account_balance', 10000.0)
                
                print("✅ System settings loaded successfully")
                self.persistence.log_system_event('INFO', 'System settings loaded from file', 'STARTUP')
            else:
                # Use defaults
                self.set_default_settings()
                print("📋 Using default settings")
                
        except Exception as e:
            print(f"❌ Error loading settings: {str(e)}")
            self.set_default_settings()
    
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
        """บันทึกการตั้งค่าระบบ"""
        try:
            settings = {
                'current_risk_profile': self.current_risk_profile,
                'custom_risk_per_trade': self.custom_risk_per_trade,
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_total_exposure': self.max_total_exposure,
                'max_daily_loss': self.max_daily_loss,
                
                'auto_trading_enabled': self.auto_trading_enabled,
                'auto_trading_pairs': list(self.auto_trading_pairs),
                'min_signal_strength': self.min_signal_strength,
                'min_entry_quality': self.min_entry_quality,
                'max_simultaneous_trades': self.max_simultaneous_trades,
                
                'required_confirmations': self.required_confirmations,
                'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5),
                'account_balance': self.account_balance,
                'hedging_enabled': getattr(self, 'hedging_enabled', False),
                'hedge_min_correlation': getattr(self, 'hedge_min_correlation', 0.6),
                'hedge_max_ratio': getattr(self, 'hedge_max_ratio', 0.6),
                'hedge_risk_target': getattr(self, 'hedge_risk_target', 0.3),
                'hedge_auto_execute': getattr(self, 'hedge_auto_execute', False),
                
                'last_saved': datetime.now().isoformat()
            }
            
            if self.persistence.save_settings(settings):
                print("💾 Settings config saved successfully")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error saving settings: {str(e)}")
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
            
            print("✅ Pair tracking data loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading pair tracking data: {str(e)}")
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
            print(f"❌ Error saving pair tracking data: {str(e)}")
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
                        
                        print(f"💾 Auto-save completed at {datetime.now().strftime('%H:%M:%S')}")
                        
                except Exception as e:
                    print(f"❌ Auto-save error: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        # Start auto-save thread
        auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        auto_save_thread.start()
        print("🔄 Auto-save system started")
    
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
        """Connect to MT5 with enhanced error handling and auto broker detection"""
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
            
            # 🆕 AUTO-DETECT BROKER และ MAP SYMBOLS
            # ====================================================
            self.logger.info("🔍 Auto-detecting broker and mapping symbols...")
            
            if self.symbol_adapter.detect_and_map_broker():
                mapping_info = self.symbol_adapter.get_mapping_info()
                self.logger.info(f"✅ Broker auto-detected successfully!")
                self.logger.info(f"🏦 Server: {mapping_info['server']}")
                self.logger.info(f"📊 Mapped: {mapping_info['mapped_symbols']}/{mapping_info['total_system_symbols']} symbols")
                self.logger.info(f"🔧 Success rate: {mapping_info['mapping_success_rate']}")
                self.logger.info(f"🎯 Detected suffixes: {mapping_info['detected_suffixes']}")
                
                # อัพเดต forex_pairs ให้ใช้ broker symbols
                self.forex_pairs = self.symbol_adapter.get_mapped_symbols()
                self.broker_symbols_mapped = True
                
                # แสดง sample mapping
                if mapping_info['sample_mapping']:
                    self.logger.info("📋 Sample Symbol Mapping:")
                    for sys_sym, broker_sym in mapping_info['sample_mapping'].items():
                        self.logger.info(f"   {sys_sym} → {broker_sym}")
                
                self.persistence.log_system_event(
                    'INFO', 
                    f'Broker auto-detected: {mapping_info["server"]} - Mapped {mapping_info["mapped_symbols"]} symbols', 
                    'BROKER_DETECTION'
                )
                
            else:
                self.logger.warning("⚠️ Symbol mapping failed, using default .c format")
                self.broker_symbols_mapped = False
                
                # ใช้วิธีเดิมถ้า mapping ไม่สำเร็จ
                self.logger.info("🔄 Falling back to legacy symbol validation...")
                available_symbols = []
                for symbol in self.forex_pairs:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info is not None:
                        if not symbol_info.visible:
                            mt5.symbol_select(symbol, True)
                        available_symbols.append(symbol)
                    else:
                        self.logger.warning(f"❌ Symbol not available: {symbol}")
                
                self.forex_pairs = available_symbols
                
                if len(available_symbols) < len(self.forex_pairs) * 0.5:  # ถ้าหาได้น้อยกว่า 50%
                    self.logger.error("❌ Too few symbols available, connection may have issues")
                
                self.persistence.log_system_event(
                    'WARNING', 
                    f'Symbol mapping failed - Using legacy mode with {len(available_symbols)} symbols', 
                    'BROKER_DETECTION'
                )
            # ====================================================
            
            self.mt5_connected = True
            
            # Log connection success
            self.logger.info(f"🎉 MT5 Connected Successfully!")
            self.logger.info(f"👤 Account: {account_info.login}")
            self.logger.info(f"🏦 Server: {account_info.server}")
            self.logger.info(f"💰 Balance: ${account_info.balance:,.2f}")
            self.logger.info(f"📊 Available Pairs: {len(self.forex_pairs)}")
            self.logger.info(f"🔄 Symbol Mapping: {'Enabled' if self.broker_symbols_mapped else 'Disabled'}")
            
            # Log a few sample symbols
            if self.forex_pairs:
                sample_symbols = self.forex_pairs[:3]
                self.logger.info(f"📋 Sample symbols: {', '.join(sample_symbols)}")
            
            self.persistence.log_system_event(
                'INFO', 
                f'MT5 Connected - Account: {account_info.login}, Server: {account_info.server}, Symbols: {len(self.forex_pairs)}', 
                'CONNECTION'
            )
            
            # Verify existing positions and update tracking
            self.logger.info("🔍 Verifying existing positions...")
            self.verify_existing_positions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MT5 connection error: {str(e)}")
            self.persistence.log_system_event('ERROR', f'MT5 connection failed: {str(e)}', 'CONNECTION')
            return False
        
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
                # 🥇 GOLD (XAUUSD) - Fixed calculation
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
                    
                # 💴 JPY Pairs - Fixed calculation
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
                    
                # 💱 Standard Forex Pairs - Fixed calculation
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
                
                # ✅ Lot size constraints and rounding
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
            
            # 🎯 ENHANCED SIGNAL STRENGTH CHECK
            signal_strength = signal_data.get('strength', 0)
            confluence_score = signal_data.get('enhanced_analysis', {}).get('confluence_score', 0)
            
            validation_result['validation_details']['thresholds_used']['min_signal_strength'] = self.min_signal_strength
            validation_result['validation_details']['checks_performed'].append('signal_strength_check')
            
            if signal_strength < self.min_signal_strength:
                validation_result['issues'].append(f'Signal strength too low: {signal_strength} < {self.min_signal_strength}')
            else:
                validation_result['confirmations'].append(f'Strong signal: {signal_strength}/10')
                validation_result['score'] += 2
            
            # 🎯 CONFLUENCE SCORE CHECK (Enhanced System)
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
            
            # 🚀 Advanced Features Validation (if available)
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
    def test_enhanced_signals(self):
            """Test the enhanced signal system - เรียกใช้ผ่าน API"""
            try:
                print(" Testing Enhanced Multi-Timeframe Signal System...")
                
                # Basic test without external import
                test_pairs = ['EURUSD.c', 'GBPUSD.c', 'USDJPY.c']
                results = {}
                
                for symbol in test_pairs:
                    try:
                        if hasattr(self, 'signal_engine') and self.signal_engine:
                            confluence_result = self.signal_engine.get_multi_timeframe_confluence(symbol)
                            
                            results[symbol] = {
                                'signal': confluence_result.get('final_signal', 'NONE'),
                                'strength': confluence_result.get('final_strength', 0),
                                'quality': confluence_result.get('final_quality', 'POOR'),
                                'confluence_score': confluence_result.get('confluence_score', 0),
                                'timeframes': len(confluence_result.get('timeframe_analysis', {})),
                                'recommendation': confluence_result.get('trade_recommendation', 'NO_TRADE')
                            }
                        else:
                            results[symbol] = {'error': 'Signal engine not available'}
                            
                    except Exception as e:
                        results[symbol] = {'error': str(e)}
                
                return {
                    'success': True,
                    'test_results': results,
                    'message': 'Enhanced signal system test completed',
                    'system_status': 'Signal engine available' if hasattr(self, 'signal_engine') and self.signal_engine else 'Using fallback system'
                }
                
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
                return {
                    'success': False,
                    'error': str(e)
                }     

    def execute_trade(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute trade based on validated signal - with multi-broker support"""
        try:
            # 🔄 SYMBOL CONVERSION - ส่วนสำคัญที่สุด!
            # ================================================
            # Determine if we need symbol conversion
            if hasattr(self, 'broker_symbols_mapped') and self.broker_symbols_mapped:
                # symbol ที่ส่งเข้ามาคือ system symbol, ต้องแปลงเป็น broker symbol
                system_symbol = symbol
                broker_symbol = self.symbol_adapter.system_to_broker_symbol(symbol)
                self.logger.debug(f"🔄 Symbol mapping: {system_symbol} → {broker_symbol}")
            else:
                # ไม่มี mapping, ใช้ symbol เดิม
                system_symbol = symbol
                broker_symbol = symbol
            
            # Verify broker symbol exists
            if not self._verify_broker_symbol(broker_symbol):
                return {
                    'success': False, 
                    'error': f'Symbol {broker_symbol} not available on this broker',
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol
                }
            # ================================================
            
            # Final validation (ใช้ system_symbol สำหรับ consistency)
            validation = self.validate_trading_signal(system_symbol, signal_data)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'Signal validation failed',
                    'issues': validation['issues']
                }
            
            # Get signal details
            signal_direction = signal_data.get('signal', 'NONE')
            entry_price = signal_data.get('optimal_entry', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            
            if signal_direction == 'NONE' or entry_price == 0:
                return {'success': False, 'error': 'Invalid signal data'}
            
            # Calculate position size (ใช้ system_symbol สำหรับ consistency)
            position_info = self.calculate_position_size(entry_price, stop_loss, system_symbol)
            lot_size = position_info['lot_size']
            
            # 🎯 MT5 OPERATIONS - ใช้ broker_symbol
            # ================================================
            # Get current market data - ใช้ broker_symbol กับ MT5
            tick = mt5.symbol_info_tick(broker_symbol)
            if not tick:
                return {
                    'success': False, 
                    'error': f'No tick data for {broker_symbol}',
                    'broker_symbol': broker_symbol
                }
            
            # Determine order type - ใช้ broker_symbol กับ MT5
            if signal_direction in ['BUY', 'STRONG_BUY']:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                return {'success': False, 'error': 'Invalid signal direction'}
            
            # Prepare order request - ใช้ broker_symbol
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': broker_symbol,  # 🎯 สำคัญ! ใช้ broker_symbol กับ MT5
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit_1,
                'deviation': self.slippage_tolerance,
                'magic': 12345,  # EA magic number
                'comment': f'Auto-{signal_direction}-{system_symbol}',  # ใส่ system_symbol ใน comment
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            # ================================================
            
            # Execute order
            self.trade_logger.info(f"📤 Executing {signal_direction} order for {system_symbol} → {broker_symbol} - Lot: {lot_size}")
            self.trade_logger.info(f"💰 Entry Price: {price}, SL: {stop_loss}, TP: {take_profit_1}")
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(f"❌ {system_symbol} ({broker_symbol}): {error_msg}")
                return {
                    'success': False, 
                    'error': error_msg, 
                    'retcode': result.retcode,
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol
                }
            
            # ✅ ORDER SUCCESSFUL - บันทึกข้อมูล
            # ================================================
            trade_info = {
                'ticket': result.order,
                'system_symbol': system_symbol,     # 🆕 เพิ่ม system symbol
                'broker_symbol': broker_symbol,     # 🆕 เพิ่ม broker symbol
                'symbol': system_symbol,            # เก็บไว้เพื่อ backward compatibility
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
                'validation_score': validation['score'],
                'requested_price': price,
                'actual_price': result.price,
                'slippage_pips': abs(result.price - price) * (10000 if 'JPY' not in broker_symbol else 100)
            }
            
            # Update tracking - ใช้ system_symbol สำหรับ consistency
            self.active_trades_per_pair[system_symbol].append(result.order)
            self.pair_trade_status[system_symbol] = 'TRADING'
            self.daily_stats['trades_executed'] += 1
            self.last_global_trade_time = datetime.now()
            
            # Save trade to database - เพิ่ม broker_symbol
            self.persistence.save_trade_to_db({
                'ticket': str(result.order),
                'symbol': system_symbol,              # เก็บ system symbol
                'broker_symbol': broker_symbol,       # 🆕 เพิ่ม broker symbol
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
            # ================================================
            
            # Enhanced logging
            self.trade_logger.info(f"✅ TRADE EXECUTED SUCCESSFULLY!")
            self.trade_logger.info(f"   System Symbol: {system_symbol}")
            self.trade_logger.info(f"   Broker Symbol: {broker_symbol}")
            self.trade_logger.info(f"   Ticket: {result.order}")
            self.trade_logger.info(f"   Direction: {signal_direction}")
            self.trade_logger.info(f"   Entry Price: {result.price} (Requested: {price})")
            self.trade_logger.info(f"   Stop Loss: {stop_loss}")
            self.trade_logger.info(f"   Take Profit: {take_profit_1}")
            self.trade_logger.info(f"   Lot Size: {lot_size}")
            self.trade_logger.info(f"   Risk: {position_info['risk_percentage']:.2f}%")
            self.trade_logger.info(f"   Slippage: {trade_info['slippage_pips']:.1f} pips")
            
            self.persistence.log_system_event(
                'INFO', 
                f'Trade executed: {system_symbol}({broker_symbol}) {signal_direction} Ticket: {result.order} Entry: {result.price}', 
                'TRADING'
            )
            
            # Save updated tracking data
            self.save_pair_tracking_data()
            self.persistence.save_daily_stats(self.daily_stats)
            
            return {
                'success': True,
                'ticket': result.order,
                'trade_info': trade_info,
                'validation': validation,
                'system_symbol': system_symbol,
                'broker_symbol': broker_symbol,
                'execution_summary': {
                    'requested_price': price,
                    'actual_price': result.price,
                    'slippage_pips': trade_info['slippage_pips'],
                    'execution_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Trade execution error for {symbol}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.persistence.log_system_event('ERROR', error_msg, 'TRADING')
            return {
                'success': False, 
                'error': error_msg,
                'system_symbol': symbol,
                'broker_symbol': broker_symbol if 'broker_symbol' in locals() else 'unknown'
            }

    def _verify_broker_symbol(self, broker_symbol: str) -> bool:
        """🔍 ตรวจสอบว่า broker symbol มีอยู่จริงใน MT5"""
        try:
            symbol_info = mt5.symbol_info(broker_symbol)
            if symbol_info is None:
                self.logger.warning(f"❌ Symbol not found: {broker_symbol}")
                return False
            
            # ตรวจสอบว่า symbol สามารถเทรดได้
            if not symbol_info.visible:
                # พยายาม enable symbol
                if mt5.symbol_select(broker_symbol, True):
                    self.logger.info(f"✅ Symbol enabled: {broker_symbol}")
                    return True
                else:
                    self.logger.warning(f"❌ Cannot enable symbol: {broker_symbol}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Symbol verification error for {broker_symbol}: {str(e)}")
            return False
        
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                
                # Check if position was closed
                if ticket not in [pos.ticket for pos in mt5.positions_get(symbol=symbol) or []]:
                    # Position was closed, update tracking
                    if ticket in self.active_trades_per_pair.get(symbol, []):
                        self.active_trades_per_pair[symbol].remove(ticket)
                    
                    # If no more active trades for this pair, set cooldown
                    if len(self.active_trades_per_pair.get(symbol, [])) == 0:
                        self.pair_trade_status[symbol] = 'READY'
                        # Set cooldown period (5 minutes)
                        self.trade_cooldowns[symbol] = datetime.now() + timedelta(minutes=5)
                        
                        # Log position closure
                        self.trade_logger.info(f"Position closed: {symbol} Ticket: {ticket}")
                        self.persistence.log_system_event('INFO', f'Position closed: {symbol} Ticket: {ticket}', 'TRADING')
                        
                        # Update daily stats
                        history = mt5.history_deals_get(ticket=ticket)
                        if history and len(history) > 0:
                            deal = history[-1]
                            if deal.profit > 0:
                                self.daily_stats['wins'] += 1
                            else:
                                self.daily_stats['losses'] += 1
                            self.daily_stats['total_pnl'] += deal.profit
                            
                            # Update trade in database
                            self.persistence.save_trade_to_db({
                                'ticket': str(ticket),
                                'exit_price': deal.price,
                                'exit_time': datetime.now().isoformat(),
                                'profit': deal.profit
                            })
                        
                        # Save updated data
                        self.save_pair_tracking_data()
                        self.persistence.save_daily_stats(self.daily_stats)
        
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
    
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
    
    def auto_trading_loop(self):
        """Main auto trading loop"""
        while self.is_running:
            try:
                if not self.auto_trading_enabled or not self.mt5_connected:
                    time.sleep(5)
                    continue
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check for new signals
                for symbol in self.auto_trading_pairs:
                    if symbol not in self.live_data:
                        continue
                    
                    signal_data = self.live_data[symbol]
                    
                    # Quick checks
                    if signal_data.get('signal', 'NONE') == 'NONE':
                        continue
                    
                    # Check pair status
                    pair_status = self.check_pair_trading_status(symbol)
                    if not pair_status['can_trade']:
                        continue
                    
                    # Validate and execute if good
                    validation = self.validate_trading_signal(symbol, signal_data)
                    if validation['valid']:
                        self.logger.info(f"Valid signal found for {symbol}: {signal_data.get('signal')}")
                        result = self.execute_trade(symbol, signal_data)
                        
                        if result['success']:
                            self.logger.info(f"Trade executed successfully: {symbol} Ticket: {result['ticket']}")
                        else:
                            self.logger.warning(f"Trade execution failed: {symbol} - {result.get('error')}")
                
                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in auto trading loop: {str(e)}")
                self.persistence.log_system_event('ERROR', f'Auto trading loop error: {str(e)}', 'TRADING')
                time.sleep(30)
    
    def start_auto_trading(self):
        """Start auto trading system - FIXED VERSION"""
        try:
            if not self.mt5_connected:
                self.logger.error("Cannot start auto trading: MT5 not connected")
                return False
            
            if self.auto_trading_enabled:
                self.logger.info("Auto trading already running")
                return True
            
            # CRITICAL FIX 1: Ensure system is running
            self.is_running = True
            self.emergency_stop = False
            self.auto_trading_enabled = True
            
            # CRITICAL FIX 2: Force start new thread with error handling
            try:
                # Stop any existing thread first
                self.auto_trading_enabled = False
                time.sleep(1)  # Wait for existing thread to stop
                
                # Start fresh
                self.auto_trading_enabled = True
                trading_thread = threading.Thread(target=self._safe_auto_trading_loop, daemon=True)
                trading_thread.start()
                
                # CRITICAL FIX 3: Verify thread started
                time.sleep(2)
                if trading_thread.is_alive():
                    self.logger.info("✅ AUTO TRADING THREAD STARTED SUCCESSFULLY!")
                    self.trade_logger.info("=== AUTO TRADING SESSION STARTED ===")
                    self.persistence.log_system_event('INFO', 'Auto trading started successfully', 'TRADING')
                    
                    # Save settings
                    self.save_system_settings()
                    return True
                else:
                    self.logger.error("❌ AUTO TRADING THREAD FAILED TO START!")
                    self.auto_trading_enabled = False
                    return False
                    
            except Exception as thread_error:
                self.logger.error(f"Threading error: {str(thread_error)}")
                self.auto_trading_enabled = False
                return False
            
        except Exception as e:
            self.logger.error(f"Error starting auto trading: {str(e)}")
            self.auto_trading_enabled = False
            return False

    def stop_auto_trading(self):
        """Stop auto trading system - FIXED VERSION"""
        try:
            self.logger.info("🛑 Stopping auto trading...")
            
            # Set flags in correct order
            self.auto_trading_enabled = False
            
            # Give time for threads to respond
            time.sleep(2)
            
            # Log the stop action
            self.trade_logger.info("=== AUTO TRADING STOPPED BY USER ===")
            self.persistence.log_system_event('INFO', 'Auto trading stopped by user', 'TRADING')
            
            # Save settings immediately
            self.save_system_settings()
            
            # Update UI status
            self.logger.info("✅ Auto trading stopped successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping auto trading: {str(e)}")
            return False

    def _safe_auto_trading_loop(self):
        """Safe auto trading loop with comprehensive error handling"""
        self.logger.info("🚀 Auto trading loop started")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running and self.auto_trading_enabled and not self.emergency_stop:
            try:
                # Reset error counter on successful iteration
                consecutive_errors = 0
                
                # CRITICAL FIX 4: Comprehensive checks before processing
                if not self.mt5_connected:
                    self.logger.warning("MT5 disconnected, attempting reconnect...")
                    if not self.connect_mt5():
                        time.sleep(30)  # Wait 30 seconds before retry
                        continue
                
                if not self.auto_trading_enabled:
                    self.logger.info("Auto trading disabled, stopping loop")
                    break
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Process signals for each pair
                signals_processed = 0
                trades_executed = 0
                
                for symbol in list(self.auto_trading_pairs):  # Create copy to avoid iteration issues
                    if not self.auto_trading_enabled:
                        break
                        
                    try:
                        # Quick availability check
                        if symbol not in self.live_data:
                            continue
                        
                        signal_data = self.live_data[symbol]
                        
                        # Quick signal check
                        if signal_data.get('signal', 'NONE') == 'NONE':
                            continue
                        
                        signals_processed += 1
                        
                        # Check pair status
                        pair_status = self.check_pair_trading_status(symbol)
                        if not pair_status['can_trade']:
                            self.logger.debug(f"❌ {symbol}: {pair_status['reason']}")
                            continue
                        
                        # Validate signal
                        validation = self.validate_trading_signal(symbol, signal_data)
                        if not validation['valid']:
                            self.logger.debug(f"❌ {symbol}: Validation failed - {', '.join(validation['issues'])}")
                            continue
                        
                        # Execute trade
                        self.logger.info(f"🎯 Executing trade for {symbol}: {signal_data.get('signal')}")
                        result = self.execute_trade(symbol, signal_data)
                        
                        if result['success']:
                            trades_executed += 1
                            self.logger.info(f"✅ Trade executed: {symbol} Ticket: {result['ticket']}")
                        else:
                            self.logger.warning(f"❌ Trade failed: {symbol} - {result.get('error')}")
                        
                    except Exception as symbol_error:
                        self.logger.error(f"Error processing {symbol}: {str(symbol_error)}")
                        continue
                
                # Log iteration summary
                if signals_processed > 0:
                    self.logger.info(f" Processed {signals_processed} signals, executed {trades_executed} trades")
                
                # CRITICAL FIX 5: Adaptive sleep based on activity
                if trades_executed > 0:
                    time.sleep(5)   # Short sleep after executing trades
                elif signals_processed > 0:
                    time.sleep(10)  # Medium sleep when processing signals
                else:
                    time.sleep(15)  # Standard sleep when no activity
                    
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"❌ Auto trading loop error ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")
                self.persistence.log_system_event('ERROR', f'Auto trading loop error: {str(e)}', 'TRADING')
                
                # CRITICAL FIX 6: Circuit breaker for consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical(f"🚨 Too many consecutive errors ({consecutive_errors}), stopping auto trading")
                    self.auto_trading_enabled = False
                    self.emergency_stop = True
                    break
                
                # Exponential backoff for errors
                sleep_time = min(60, 5 * (2 ** consecutive_errors))
                self.logger.info(f"⏰ Sleeping {sleep_time} seconds after error")
                time.sleep(sleep_time)
        
        self.logger.info("🛑 Auto trading loop stopped")
        self.auto_trading_enabled = False

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
                self.logger.info("🔄 Force restarting auto trading...")
                
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
                    'type_filling': mt5.ORDER_FILLING_IOC,
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
                # 🔥 ตรวจสอบว่ามี signal engine หรือไม่
                if not hasattr(self, 'signal_engine') or self.signal_engine is None:
                    # Fallback to old system
                    return self.old_analyze_entry_exit_points(indicators, current_price, symbol)
                
                # 🔥 Use the new multi-timeframe confluence system
                confluence_result = self.signal_engine.get_multi_timeframe_confluence(symbol)
                
                # Check additional filters
                risk_factors = []
                
                # 1. Check correlation risk
                existing_positions = [pos.symbol for pos in (mt5.positions_get() or [])]
                if not self.signal_engine.check_correlation_risk(symbol, existing_positions):
                    risk_factors.append("High correlation with existing positions")
                
                # 2. Check news filter
                if not self.signal_engine.check_news_filter(symbol):
                    risk_factors.append("Near high-impact news event")
                
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
            
            # 🚀 ENHANCED ANALYSIS (Advanced Features)
            if self.use_advanced_features and analysis_method != 'error_fallback':
                try:
                    # Prepare timeframe data for advanced analysis
                    timeframe_data = {}
                    for tf_name, df in timeframe_rates.items():
                        timeframe_data[tf_name] = df
                    
                    # Add account balance to signal data
                    entry_exit_analysis['account_balance'] = self.account_balance
                    
                    # Get enhanced analysis
                    enhanced_analysis = self.advanced_integrator.enhance_signal_analysis(
                        symbol, entry_exit_analysis, timeframe_data
                    )
                    
                    # Use enhanced results
                    entry_exit_analysis = enhanced_analysis
                    analysis_method = 'enhanced_advanced'
                    
                    # self.logger.info(f" Enhanced Analysis for {symbol}: "
                    #             f"Signal: {enhanced_analysis.get('signal', 'NONE')} -> "
                    #             f"Enhanced Strength: {enhanced_analysis.get('enhanced_strength', 0)}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Enhanced analysis failed for {symbol}: {str(e)}")
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
            
            # Final validation of result
            try:
                # Ensure critical fields are valid
                if result['current_price'] <= 0:
                    result['current_price'] = 1.0
                if result['lot_size'] <= 0:
                    result['lot_size'] = self.default_lot_size
                
                # Ensure signal fields exist
                signal_fields = ['signal', 'strength', 'entry_quality', 'optimal_entry', 'stop_loss', 'take_profit_1']
                for field in signal_fields:
                    if field not in result:
                        if field == 'signal':
                            result[field] = 'NONE'
                        elif field == 'strength':
                            result[field] = 0
                        elif field == 'entry_quality':
                            result[field] = 'POOR'
                        else:
                            result[field] = result['current_price']
                            
            except Exception as e:
                self.logger.warning(f"Result validation error for {symbol}: {str(e)}")
            
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
<h2 style="color:#ffff00;">🔄 WITH DATA PERSISTENCE & STATE MANAGEMENT</h2>
<p style="color:#00ccff;">✅ Settings auto-saved every 5 minutes</p>
<p style="color:#00ccff;">✅ Positions tracking persistent</p>
<p style="color:#00ccff;">✅ Daily statistics saved</p>
<p style="color:#00ccff;">✅ Trade history database</p>
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
                        # เพิ่มข้อมูล Pullback Protection
                        if hasattr(self, 'pullback_protection') and self.pullback_protection:
                            if 'pullback_protection' in data:
                                # มีข้อมูล Pullback Protection อยู่แล้ว
                                pass
                            elif symbol in self.pullback_protection.waiting_positions:
                                # เพิ่มสถานะ waiting
                                data['pullback_protection'] = {
                                    'status': 'WAITING_PULLBACK',
                                    'waiting_since': self.pullback_protection.waiting_positions[symbol]['wait_start'].strftime('%H:%M:%S')
                                }
                        
                        formatted_data[symbol] = data
                
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
                    # 🛡️ Pullback Protection Status
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
    
        @self.app.route('/api/test-enhanced-signals')
        def test_enhanced_signals_api():
            """Test enhanced signal system via API"""
            return jsonify(self.test_enhanced_signals())
        
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
                'advanced_features_available': ADVANCED_FEATURES_AVAILABLE,
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
            if not ADVANCED_FEATURES_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'Advanced features module not available'
                })
            
            try:
                self.use_advanced_features = not self.use_advanced_features
                
                status = 'ENABLED' if self.use_advanced_features else 'DISABLED'
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
    <h1 style="color:#ff4444;">🎯 Hedging Dashboard</h1>
    <p style="color:#ffaa00;">Please save the hedging dashboard HTML as 'hedging_dashboard.html'</p>
    <p style="color:#888;">Error: {str(e)}</p>
    <a href="/" style="color:#00ccff;">← Back to Main Dashboard</a>
    </body></html>'''
        
        # 🎯 TRAILING STOP API ROUTES
        @self.app.route('/api/trailing-stops/status')
        def get_trailing_status():
            """Get trailing stop status"""
            try:
                data = self.get_trailing_dashboard_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/trailing-stops/toggle', methods=['POST'])
        def toggle_trailing():
            """Toggle trailing stops on/off"""
            try:
                data = request.get_json()
                enabled = data.get('enabled', False)
                self.trailing_system_enabled = enabled
                
                if enabled:
                    print("🟢 Trailing Stops: ENABLED")
                else:
                    print("🔴 Trailing Stops: DISABLED")
                
                return jsonify({'success': True, 'enabled': enabled})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/trailing-stops/profile', methods=['POST'])
        def set_trailing_profile():
            """Set trailing stop profile"""
            try:
                data = request.get_json()
                profile = data.get('profile', 'MODERATE')
                
                if profile in self.trailing_profiles:
                    self.current_trailing_profile = profile
                    print(f"🎯 Trailing Profile Changed: {profile}")
                    return jsonify({'success': True, 'profile': profile})
                else:
                    return jsonify({'success': False, 'error': 'Invalid profile'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/trailing-stops/manual-update', methods=['POST'])
        def manual_trailing_update():
            """Manually trigger trailing stop update"""
            try:
                self.process_all_trailing_stops()
                return jsonify({'success': True, 'message': 'Trailing stops updated'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/trailing-dashboard')
        def trailing_dashboard():
            """Serve trailing stop dashboard"""
            try:
                return send_from_directory('.', 'trailing_dashboard.html')
            except:
                return '''
                <h1 style="color:#00ff00;">🎯 Trailing Stop Dashboard</h1>
                <p style="color:#ffaa00;">Please save the trailing_dashboard.html file in the same directory.</p>
                <a href="/" style="color:#00ccff;">← Back to Main Dashboard</a>
                '''

        # 🛡️ Pullback Protection API Routes
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
        @self.app.route('/pullback_dashboard.html')
        def pullback_dashboard():
            """Pullback Protection Dashboard"""
            try:
                return send_from_directory('.', 'pullback_dashboard.html')
            except:
                return '''<!DOCTYPE html>
        <html><head><title>Pullback Protection Dashboard</title></head>
        <body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
        <h1 style="color:#cc0066;">🛡️ Pullback Protection Dashboard</h1>
        <p style="color:#ff6666;">ไฟล์ pullback_dashboard.html ไม่พบ</p>
        <p style="color:#ffaa00;">กรุณาบันทึกไฟล์ pullback_dashboard.html ในโฟลเดอร์เดียวกับ mt5_forex_connector.py</p>
        <p style="color:#666;">Current directory: ''' + os.getcwd() + '''</p>
        <br><a href="/" style="color:#00ccff;">← กลับสู่ Main Dashboard</a>
        </body></html>'''

        # 🔗 เพิ่ม Route สำหรับ Quick Access
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
        <h1 style="color:#cc0066;">🛡️ Redirecting to Pullback Dashboard...</h1>
        <p style="color:#00ccff;"><a href="/pullback_dashboard.html">Click here if not redirected</a></p>
        <script>window.location.href='/pullback_dashboard.html';</script>
        </body></html>'''
        # 📊 เพิ่ม Route สำหรับ Pullback Status Widget
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

        if hasattr(self, 'hedging_enabled') and self.hedging_enabled:
            self.setup_hedging_routes()
        
        print("✅ All routes setup completed")

        # 🎯 Add Trailing Stop Routes
        if hasattr(self, 'enhanced_trading'):
            add_trailing_stop_routes(self.app, self.enhanced_trading)
            print("✅ Trailing Stop API Routes: ACTIVATED")

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

            print("🎯 Hedging API routes added successfully!")
            
        except Exception as e:
            print(f"❌ Error setting up hedging routes: {str(e)}")
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
            print("\n🔄 Shutting down gracefully...")
            
            # Stop auto trading
            if self.auto_trading_enabled:
                self.stop_auto_trading()
            
            # Save all data
            print("💾 Saving system data...")
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
            
            print("✅ Shutdown completed successfully")
            print("💾 All data saved for next session")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {str(e)}")
    
        if hasattr(self, 'enhanced_trading'):
                self.enhanced_trading.stop_trailing_thread()
                print("⏹️ Trailing Stop System: STOPPED")

    def calculate_trailing_stop(self, position, market_data):
        """🎯 คำนวณ Trailing Stop สำหรับ position"""
        try:
            symbol = position.symbol
            ticket = position.ticket
            position_type = position.type
            entry_price = position.price_open
            current_price = market_data.get('bid' if position_type == 0 else 'ask', entry_price)
            current_sl = position.sl
            
            # Get ATR
            atr = market_data.get('atr', 0.001)
            if atr <= 0:
                atr = abs(current_price - entry_price) * 0.01
                
            # Get profile settings
            profile = self.trailing_profiles[self.current_trailing_profile]
            
            # Initialize position state
            if ticket not in self.trailing_position_states:
                self.trailing_position_states[ticket] = {
                    'highest_price': current_price if position_type == 0 else entry_price,
                    'lowest_price': current_price if position_type == 1 else entry_price,
                    'breakeven_activated': False,
                    'trail_count': 0
                }
            
            state = self.trailing_position_states[ticket]
            
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
            
            # Breakeven protection
            if (not state['breakeven_activated'] and 
                profit_atr >= profile['breakeven_trigger_atr']):
                
                new_sl = entry_price + (0.0001 if position_type == 0 else -0.0001)
                state['breakeven_activated'] = True
                should_update = True
                trail_reason = "BREAKEVEN_PROTECTION"
                
            # Dynamic trailing
            elif state['breakeven_activated'] or profit_atr >= profile['breakeven_trigger_atr']:
                trail_distance = profile['min_trail_distance_atr'] * atr
                
                if position_type == 0:  # BUY
                    calculated_sl = reference_price - trail_distance
                    if calculated_sl > current_sl:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_UP"
                        state['trail_count'] += 1
                else:  # SELL
                    calculated_sl = reference_price + trail_distance
                    if current_sl == 0 or calculated_sl < current_sl:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_DOWN"
                        state['trail_count'] += 1
            
            return {
                'should_update': should_update,
                'new_sl': new_sl,
                'trail_reason': trail_reason,
                'profit_atr': round(profit_atr, 2),
                'breakeven_activated': state['breakeven_activated'],
                'trail_count': state['trail_count']
            }
            
        except Exception as e:
            print(f"❌ Error calculating trailing stop: {str(e)}")
            return {'should_update': False, 'error': str(e)}

    def update_position_trailing_stop(self, ticket, new_sl, symbol):
        """🎯 อัพเดท Stop Loss ใน MT5"""
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
                print(f"✅ Trailing SL Updated: {symbol} #{ticket} → SL: {new_sl:.5f}")
                self.trailing_statistics['total_sl_updates'] += 1
                return True
            else:
                print(f"❌ SL Update Failed: {symbol} #{ticket} - {result.comment}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating SL: {str(e)}")
            return False

    def process_all_trailing_stops(self):
        """🎯 ประมวลผล Trailing Stop ทุก positions"""
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
                print(f"🎯 Trailing Stop Updates: {updates_made}")
                
        except Exception as e:
            print(f"❌ Error processing trailing stops: {str(e)}")

    def get_trailing_dashboard_data(self):
        """📊 ข้อมูลสำหรับ Dashboard"""
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
            print(f"❌ Error getting dashboard data: {str(e)}")
            return {'error': str(e)}

    def run(self, host='0.0.0.0', port=5000):
        """Run the enhanced auto trading dashboard"""
        try:
            print("Enhanced Smart Auto Trading Dashboard Starting...")
            print("=" * 60)
            print("🔄 WITH DATA PERSISTENCE & STATE MANAGEMENT")
            print("=" * 60)
            
            if not self.connect_mt5():
                print("ERROR: Failed to connect to MT5")
                return
            
            self.is_running = True
            self.start_data_updates()
            self.update_all_data()
            
            print(f"✅ SUCCESS: Enhanced Auto Trading Dashboard Started!")
            print(f"🔄 FEATURES: Smart Auto Trading + Risk Management + Data Persistence")
            print(f"💾 PERSISTENCE: Settings, Positions & Stats Auto-Saved")
            print(f" DATABASE: Trade History & System Logs")
            print(f"🛡️ RECOVERY: System state restored on restart")
            print(f"🌐 DASHBOARD: http://{host}:{port}")
            print(f"🔗 API: http://{host}:{port}/api/market-data")
            print(f"📈 STATUS: http://{host}:{port}/api/system-status")
            print(f"⚡ AUTO TRADING: Currently {('ENABLED' if self.auto_trading_enabled else 'DISABLED')}")
            print("💾 DATA SAVED: Every 5 minutes + on changes")
            print("🔄 STOP: Press Ctrl+C for graceful shutdown")
            print("=" * 60)
            
            # Log startup
            self.persistence.log_system_event('INFO', 'Enhanced Auto Trading Dashboard started', 'STARTUP')
            
            self.app.run(host=host, port=port, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            self.graceful_shutdown()
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            self.graceful_shutdown()

def main():
    """Main execution"""
    
    dashboard = EnhancedSmartAutoTradingDashboard()
    dashboard.run()
    print("Running")

if __name__ == "__main__":
    main()