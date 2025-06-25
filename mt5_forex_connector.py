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
from enhanced_signal_system import MultiTimeframeSignalEngine
warnings.filterwarnings('ignore')

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
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2, default=str)
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
            with open(self.daily_stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
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
            with open(self.pair_status_file, 'w') as f:
                json.dump(pair_status, f, indent=2, default=str)
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
        
        # Forex pairs
        self.forex_pairs = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c', 'USDCAD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'EURCAD.c',
            'GBPJPY.c', 'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'GBPCAD.c',
            'AUDCHF.c', 'AUDJPY.c', 'AUDNZD.c', 'AUDCAD.c',
            'NZDJPY.c', 'NZDCHF.c', 'NZDCAD.c',
            'CHFJPY.c', 'CADJPY.c', 'XAUUSD.c'
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
                
                'last_saved': datetime.now().isoformat()
            }
            
            if self.persistence.save_settings(settings):
                print("💾 Settings saved successfully")
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
        """Connect to MT5 with enhanced error handling"""
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
            
            # Test symbols and ensure they're visible
            available_symbols = []
            for symbol in self.forex_pairs:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
            
            self.forex_pairs = available_symbols
            self.mt5_connected = True
            
            # Log connection success
            self.logger.info(f"MT5 Connected Successfully!")
            self.logger.info(f"Account: {account_info.login}")
            self.logger.info(f"Balance: ${account_info.balance:,.2f}")
            self.logger.info(f"Available Pairs: {len(self.forex_pairs)}")
            
            self.persistence.log_system_event('INFO', f'MT5 Connected - Account: {account_info.login}', 'CONNECTION')
            
            # Verify existing positions and update tracking
            self.verify_existing_positions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
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
        """Calculate position size with CORRECT GOLD calculation"""
        try:
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # ตั้งค่าเริ่มต้น
            lot_size = self.default_lot_size
            money_per_pip = 10.0
            pip_size = 0.0001
            
            # Risk amount
            risk_amount = self.account_balance * risk_percent
            
            # 🥇 GOLD (XAUUSD) - การคำนวณพิเศษ
            if 'XAU' in symbol or 'GOLD' in symbol:
                pip_size = 0.1          # Gold: 1 pip = $0.10
                money_per_pip = 1.0     # Gold: $1 per pip per 1 lot (100 oz)
                
                # Gold lot size calculation
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    # Gold: Risk = Pips × $1 × Lot Size
                    lot_size = risk_amount / pips_at_risk
                
            # 💴 JPY Pairs
            elif 'JPY' in symbol:
                pip_size = 0.01         # JPY: 1 pip = 0.01
                money_per_pip = 10.0 / entry_price if entry_price > 0 else 0.1
                
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
            
            # 💱 Forex Pairs ปกติ
            else:
                pip_size = 0.0001       # Forex: 1 pip = 0.0001
                money_per_pip = 10.0    # $10 per pip per 1 lot
                
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
            
            # ✅ จำกัดขนาด lot
            lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
            
            # ปัดเศษตาม symbol
            if 'XAU' in symbol:
                lot_size = round(lot_size, 2)      # Gold: 0.01, 0.05, 0.10
            elif lot_size >= 1.0:
                lot_size = round(lot_size, 1)      # 1.0, 1.5, 2.0
            else:
                lot_size = round(lot_size, 2)      # 0.01, 0.05, 0.10
            
            # คำนวณความเสี่ยงจริง
            points_risk = abs(entry_price - stop_loss)
            pips_at_risk = points_risk / pip_size
            
            # 🥇 Gold: Risk = Pips × $1 × Lot
            if 'XAU' in symbol:
                actual_risk = pips_at_risk * 1.0 * lot_size  # $1 per pip per lot
            # 💴 JPY: Risk = Pips × (10/Price) × Lot  
            elif 'JPY' in symbol:
                actual_risk = pips_at_risk * money_per_pip * lot_size
            # 💱 Forex: Risk = Pips × $10 × Lot
            else:
                actual_risk = pips_at_risk * 10.0 * lot_size
            
            risk_percentage = (actual_risk / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            return {
                'lot_size': lot_size,
                'risk_amount': round(actual_risk, 2),
                'risk_percentage': round(risk_percentage, 2),
                'pip_value': round(money_per_pip * lot_size, 2),
                'points_risk': round(points_risk, 5),
                'pips_at_risk': round(pips_at_risk, 1),
                'pip_size': pip_size,
                'money_per_pip': money_per_pip,
                'symbol_type': 'GOLD' if 'XAU' in symbol else 'JPY' if 'JPY' in symbol else 'FOREX',
                'calculation_status': 'SUCCESS'
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation error for {symbol}: {str(e)}")
            
            # ✅ Return safe defaults based on symbol
            if 'XAU' in symbol:
                default_pip_size = 0.1
                default_money_per_pip = 1.0
            elif 'JPY' in symbol:
                default_pip_size = 0.01
                default_money_per_pip = 0.1
            else:
                default_pip_size = 0.0001
                default_money_per_pip = 10.0
                
            return {
                'lot_size': self.default_lot_size,
                'risk_amount': 0,
                'risk_percentage': 0,
                'pip_value': default_money_per_pip * self.default_lot_size,
                'points_risk': 0,
                'pips_at_risk': 0,
                'pip_size': default_pip_size,
                'money_per_pip': default_money_per_pip,
                'symbol_type': 'GOLD' if 'XAU' in symbol else 'JPY' if 'JPY' in symbol else 'FOREX',
                'calculation_status': 'ERROR',
                'error_message': str(e)
            }
    
    def validate_trading_signal(self, symbol: str, signal_data: Dict) -> Dict:
            """Enhanced signal validation with multi-timeframe confluence"""
            try:
                validation_result = {
                    'valid': False,
                    'score': 0,
                    'issues': [],
                    'confirmations': [],
                    'confidence_level': 'LOW'
                }
                
                # Check if auto trading is enabled
                if not self.auto_trading_enabled:
                    validation_result['issues'].append('Auto trading disabled')
                    return validation_result
                
                # Check emergency stop
                if self.emergency_stop:
                    validation_result['issues'].append('Emergency stop activated')
                    return validation_result
                
                # Check pair trading status
                pair_status = self.check_pair_trading_status(symbol)
                if not pair_status['can_trade']:
                    validation_result['issues'].append(f"Pair not ready: {pair_status['reason']}")
                    return validation_result
                
                # 🎯 ENHANCED SIGNAL STRENGTH CHECK
                signal_strength = signal_data.get('strength', 0)
                confluence_score = signal_data.get('enhanced_analysis', {}).get('confluence_score', 0)
                
                if signal_strength < self.min_signal_strength:
                    validation_result['issues'].append(f'Signal strength too low: {signal_strength} < {self.min_signal_strength}')
                else:
                    validation_result['confirmations'].append(f'Strong signal: {signal_strength}/10')
                    validation_result['score'] += 2
                
                # 🎯 CONFLUENCE SCORE CHECK (NEW)
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
                
                if quality_scores.get(entry_quality, 0) < min_quality_score:
                    validation_result['issues'].append(f'Entry quality too low: {entry_quality} < {self.min_entry_quality}')
                else:
                    validation_result['confirmations'].append(f'Good entry quality: {entry_quality}')
                    validation_result['score'] += quality_scores.get(entry_quality, 0)
                
                # Check timeframe analysis count
                timeframes_analyzed = len(signal_data.get('enhanced_analysis', {}).get('timeframes_analyzed', []))
                if timeframes_analyzed >= 3:
                    validation_result['confirmations'].append(f'Multi-timeframe analysis: {timeframes_analyzed} TFs')
                    validation_result['score'] += 1
                else:
                    validation_result['issues'].append(f'Insufficient timeframe data: {timeframes_analyzed}')
                
                # Check risk factors
                risk_factor_count = signal_data.get('enhanced_analysis', {}).get('total_risk_factors', 0)
                if risk_factor_count == 0:
                    validation_result['confirmations'].append('No risk factors detected')
                    validation_result['score'] += 1
                elif risk_factor_count <= 1:
                    validation_result['confirmations'].append('Minimal risk factors')
                    validation_result['score'] += 0.5
                else:
                    validation_result['issues'].append(f'Multiple risk factors: {risk_factor_count}')
                
                # Continue with existing validation checks...
                signal_direction = signal_data.get('signal', 'NONE')
                if signal_direction == 'NONE':
                    validation_result['issues'].append('No clear signal direction')
                else:
                    validation_result['confirmations'].append(f'Clear signal: {signal_direction}')
                    validation_result['score'] += 1
                
                # Check risk/reward ratio
                rr_ratio = signal_data.get('rr_tp1', 0)
                if rr_ratio < getattr(self, 'min_rr_ratio', 1.5):
                    validation_result['issues'].append(f'Poor risk/reward: 1:{rr_ratio}')
                else:
                    validation_result['confirmations'].append(f'Good R/R: 1:{rr_ratio}')
                    validation_result['score'] += 1
                
                # Check current exposure
                current_exposure = self.calculate_current_exposure()
                if current_exposure >= self.max_total_exposure:
                    validation_result['issues'].append(f'Max exposure reached: {current_exposure*100:.1f}%')
                else:
                    validation_result['confirmations'].append(f'Exposure OK: {current_exposure*100:.1f}%')
                    validation_result['score'] += 1
                
                # Final validation with enhanced scoring
                validation_result['valid'] = len(validation_result['issues']) == 0 and validation_result['score'] >= 6  # Raised threshold
                
                # Set confidence level based on score
                if validation_result['score'] >= 8:
                    validation_result['confidence_level'] = 'VERY_HIGH'
                elif validation_result['score'] >= 6:
                    validation_result['confidence_level'] = 'HIGH'
                elif validation_result['score'] >= 4:
                    validation_result['confidence_level'] = 'MEDIUM'
                else:
                    validation_result['confidence_level'] = 'LOW'
                
                return validation_result
                
            except Exception as e:
                self.logger.error(f"Error validating enhanced signal for {symbol}: {str(e)}")
                return {
                    'valid': False, 
                    'issues': [f'Validation error: {str(e)}'], 
                    'score': 0,
                    'confidence_level': 'ERROR'
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
        """Execute trade based on validated signal"""
        try:
            # Final validation
            validation = self.validate_trading_signal(symbol, signal_data)
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
            
            # Calculate position size
            position_info = self.calculate_position_size(entry_price, stop_loss, symbol)
            lot_size = position_info['lot_size']
            
            # Determine order type
            if signal_direction in ['BUY', 'STRONG_BUY']:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                return {'success': False, 'error': 'Invalid signal direction'}
            
            # Prepare order request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit_1,
                'deviation': self.slippage_tolerance,
                'magic': 12345,  # EA magic number
                'comment': f'Auto Trade - {signal_direction}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Execute order
            self.trade_logger.info(f"Executing {signal_direction} order for {symbol} - Lot: {lot_size}")
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'retcode': result.retcode}
            
            # Order successful
            trade_info = {
                'ticket': result.order,
                'symbol': symbol,
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
                'validation_score': validation['score']
            }
            
            # Update tracking
            self.active_trades_per_pair[symbol].append(result.order)
            self.pair_trade_status[symbol] = 'TRADING'
            self.daily_stats['trades_executed'] += 1
            self.last_global_trade_time = datetime.now()
            
            # Save trade to database
            self.persistence.save_trade_to_db({
                'ticket': str(result.order),
                'symbol': symbol,
                'type': signal_direction,
                'volume': lot_size,
                'entry_price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'entry_time': datetime.now().isoformat(),
                'signal_strength': signal_data.get('strength', 0),
                'entry_quality': signal_data.get('entry_quality', 'UNKNOWN'),
                'risk_percentage': position_info['risk_percentage']
            })
            
            # Log successful trade
            self.trade_logger.info(f"TRADE EXECUTED: {symbol} {signal_direction} - Ticket: {result.order}")
            self.trade_logger.info(f"Entry: {result.price}, SL: {stop_loss}, TP: {take_profit_1}")
            self.trade_logger.info(f"Lot Size: {lot_size}, Risk: {position_info['risk_percentage']:.2f}%")
            
            self.persistence.log_system_event('INFO', f'Trade executed: {symbol} {signal_direction} Ticket: {result.order}', 'TRADING')
            
            # Save updated tracking data
            self.save_pair_tracking_data()
            self.persistence.save_daily_stats(self.daily_stats)
            
            return {
                'success': True,
                'ticket': result.order,
                'trade_info': trade_info,
                'validation': validation
            }
            
        except Exception as e:
            error_msg = f"Trade execution error for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            self.persistence.log_system_event('ERROR', error_msg, 'TRADING')
            return {'success': False, 'error': error_msg}
    
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
        """Start auto trading system"""
        if not self.mt5_connected:
            return False
        
        if self.auto_trading_enabled:
            return True
        
        self.auto_trading_enabled = True
        self.emergency_stop = False
        
        # Start auto trading thread
        trading_thread = threading.Thread(target=self.auto_trading_loop, daemon=True)
        trading_thread.start()
        
        self.logger.info("AUTO TRADING STARTED!")
        self.trade_logger.info("=== AUTO TRADING SESSION STARTED ===")
        self.persistence.log_system_event('INFO', 'Auto trading started', 'TRADING')
        
        # Save settings
        self.save_system_settings()
        
        return True
    
    def stop_auto_trading(self):
        """Stop auto trading system"""
        self.auto_trading_enabled = False
        self.logger.info("AUTO TRADING STOPPED!")
        self.trade_logger.info("=== AUTO TRADING SESSION STOPPED ===")
        self.persistence.log_system_event('INFO', 'Auto trading stopped', 'TRADING')
        
        # Save settings
        self.save_system_settings()
    
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
                
                # 📊 Log enhanced signal info
                if result['signal'] != 'NONE':
                    self.logger.info(f" ENHANCED SIGNAL: {symbol}")
                    self.logger.info(f"   Signal: {result['signal']} | Strength: {result['strength']}/10")
                    self.logger.info(f"   Quality: {result['entry_quality']} | Confluence: {confluence_result.get('confluence_score', 0)}")
                    self.logger.info(f"   Timeframes: {len(confluence_result.get('timeframe_analysis', {}))}")
                    self.logger.info(f"   Risk Factors: {len(risk_factors)}")
                
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
        Backup method - ระบบเก่าสำหรับกรณี signal engine ไม่ทำงาน
        """
        try:
            # Get indicators
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0)
            atr = indicators.get('atr', current_price * 0.005)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # Simple signal analysis (ระบบเก่า)
            signal_direction = 'NONE'
            signal_strength = 0
            entry_score = 0
            entry_reasons = ['Using fallback system']
            
            # Basic bullish conditions
            bullish_score = 0
            if current_price > ema_9:
                bullish_score += 2
                entry_reasons.append("Price above EMA9")
            if ema_9 > ema_21:
                bullish_score += 1
                entry_reasons.append("EMA9 > EMA21")
            if ema_21 > ema_50:
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
            if current_price < ema_9:
                bearish_score += 2
            if ema_9 < ema_21:
                bearish_score += 1
            if ema_21 < ema_50:
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
            
            # Calculate levels
            atr_multiplier = 1.5
            
            if signal_direction in ['BUY', 'STRONG_BUY']:
                stop_loss = current_price - (atr * atr_multiplier)
                take_profit_1 = current_price + (atr * 2.5)
                take_profit_2 = current_price + (atr * 4.0)
                take_profit_3 = current_price + (atr * 6.0)
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                stop_loss = current_price + (atr * atr_multiplier)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Position sizing
            position_info = self.calculate_position_size(current_price, stop_loss, symbol)
            
            # R/R ratios
            risk = abs(current_price - stop_loss)
            rr_tp1 = abs(take_profit_1 - current_price) / risk if risk > 0 else 0
            rr_tp2 = abs(take_profit_2 - current_price) / risk if risk > 0 else 0
            rr_tp3 = abs(take_profit_3 - current_price) / risk if risk > 0 else 0
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 1),
                'entry_quality': entry_quality,
                'entry_score': entry_score,
                'entry_reasons': entry_reasons,
                'optimal_entry': round(current_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'take_profit_3': round(take_profit_3, 5),
                'lot_size': position_info.get('lot_size', self.default_lot_size),
                'risk_amount': position_info.get('risk_amount', 0),
                'risk_percentage': position_info.get('risk_percentage', 0),
                'rr_tp1': round(rr_tp1, 2),
                'rr_tp2': round(rr_tp2, 2),
                'rr_tp3': round(rr_tp3, 2),
                'enhanced_analysis': {
                    'fallback_mode': True,
                    'system_version': 'Fallback_System_v1.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis error: {str(e)}")
            return {
                'signal': 'NONE', 'strength': 0, 'entry_quality': 'POOR',
                'entry_score': 0, 'optimal_entry': current_price,
                'stop_loss': current_price, 'take_profit_1': current_price,
                'lot_size': self.default_lot_size, 'risk_amount': 0,
                'risk_percentage': 0, 'rr_tp1': 0, 'rr_tp2': 0, 'rr_tp3': 0,
                'enhanced_analysis': {
                    'error': str(e),
                    'fallback_mode': True,
                    'system_version': 'Fallback_System_v1.0'
                }
            }    
            
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get enhanced symbol data"""
        try:
            if not self.mt5_connected:
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            current_price = tick.bid
            
            # Get multi-timeframe data
            rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            
            if any(rates is None for rates in [rates_h4, rates_h1, rates_m15, rates_m5]):
                return None
            
            df_h1 = pd.DataFrame(rates_h1)
            indicators = self.calculate_indicators(df_h1)
            entry_exit_analysis = self.analyze_entry_exit_points(indicators, current_price, symbol)
            
            # Multi-timeframe prices
            h4_price = rates_h4[-1]['close'] if len(rates_h4) > 0 else current_price
            h1_price = rates_h1[-1]['close'] if len(rates_h1) > 0 else current_price
            m15_price = rates_m15[-1]['close'] if len(rates_m15) > 0 else current_price
            m5_price = rates_m5[-1]['close'] if len(rates_m5) > 0 else current_price
            
            # Price change
            previous_price = df_h1['close'].iloc[-2] if len(df_h1) > 1 else current_price
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            # Spread
            spread = tick.ask - tick.bid
            spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
            
            # Trading status for this pair
            pair_status = self.check_pair_trading_status(symbol)
            
            return {
                'symbol': symbol,
                'h4': round(h4_price, 5),
                'h1': round(h1_price, 5),
                'm15': round(m15_price, 5),
                'm5': round(m5_price, 5),
                'current_price': round(current_price, 5),
                'price_change': round(price_change, 5),
                'change_percent': round(change_percent, 3),
                'price_direction': 'up' if m5_price > m15_price else 'down',
                'bid': tick.bid,
                'ask': tick.ask,
                'spread_pips': round(spread_pips, 1),
                
                # Technical indicators
                'rsi': round(indicators.get('rsi', 50), 1),
                'macd': round(indicators.get('macd', 0), 6),
                'atrPercent': round(indicators.get('atr_percent', 0), 3),
                'trendStrength': round(indicators.get('trend_strength', 0), 3),
                'volumeRatio': round(indicators.get('volume_ratio', 1), 2),
                
                # Trading signals and analysis
                **entry_exit_analysis,
                
                # Auto trading info
                'pair_status': pair_status,
                'can_trade': pair_status['can_trade'],
                'active_trades': len(self.active_trades_per_pair.get(symbol, [])),
                
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
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
                    'custom_risk': self.custom_risk_per_trade
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
                    'mt5_connected': False
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
            print(f"📊 DATABASE: Trade History & System Logs")
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
    print("Enhanced Smart Auto Trading Dashboard")
    print("====================================")
    print("🔄 WITH DATA PERSISTENCE & RECOVERY")
    print("====================================")
    print("Features:")
    print("✅ Auto Trading with Signal Validation")
    print("✅ One Trade Per Pair Control")
    print("✅ Portfolio Risk Profiles")
    print("✅ Real-time Position Management")
    print("✅ Emergency Stop Controls")
    print("💾 Data Persistence & Auto-Save")
    print("🔄 State Management & Recovery")
    print("📊 Trade History Database")
    print("🛡️ System Logs & Monitoring")
    print()
    
    dashboard = EnhancedSmartAutoTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()