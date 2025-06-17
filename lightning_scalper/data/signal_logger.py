#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[ROCKET] Lightning Scalper - Signal Data Logger
Production-Grade Data Collection System for Active Learning

This module handles comprehensive data logging for all trading signals,
market conditions, execution results, and performance metrics.
Data is used for ML model training and system improvement.

Features:
- Real-time signal logging
- Market data capture
- Execution results tracking
- Performance analytics
- Data export for ML training
- Historical data management

Author: Phoenix Trading AI
Version: 1.0.0
License: Proprietary
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import pickle
import gzip
import shutil
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our core modules
try:
    from core.lightning_scalper_engine import FVGSignal, FVGType, CurrencyPair, MarketCondition, FVGStatus
    from execution.trade_executor import Order, Position, OrderStatus, TradeDirection
    from adapters.mt5_adapter import MT5Position, MT5AccountInfo
except ImportError as e:
    print(f"[X] Failed to import core modules: {e}")
    sys.exit(1)

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DataFormat(Enum):
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    PARQUET = "parquet"
    SQLITE = "sqlite"

@dataclass
class SignalLogEntry:
    """Comprehensive signal log entry for ML training"""
    # Signal identification
    signal_id: str
    timestamp: datetime
    log_type: str = "SIGNAL_GENERATED"
    
    # Signal data
    currency_pair: str = ""
    timeframe: str = ""
    signal_type: str = ""  # BULLISH/BEARISH
    confluence_score: float = 0.0
    
    # Price levels
    entry_price: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    stop_loss: float = 0.0
    gap_size: float = 0.0
    gap_percentage: float = 0.0
    
    # Market context
    market_condition: str = ""
    session: str = ""
    atr_ratio: float = 0.0
    volume_strength: float = 0.0
    momentum_score: float = 0.0
    structure_score: float = 0.0
    
    # Technical indicators (for ML features)
    rsi_value: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_position: Optional[float] = None
    support_resistance_distance: Optional[float] = None
    
    # Execution data
    execution_delay: Optional[float] = None
    slippage: Optional[float] = None
    actual_entry: Optional[float] = None
    
    # Results (for supervised learning)
    outcome: Optional[str] = None  # WIN/LOSS/PARTIAL
    pnl_pips: Optional[float] = None
    pnl_dollars: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    holding_time_minutes: Optional[int] = None
    
    # Client data
    client_id: Optional[str] = None
    lot_size: Optional[float] = None
    account_balance: Optional[float] = None
    
    # Additional metadata
    tags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ExecutionLogEntry:
    """Execution result log entry"""
    execution_id: str
    signal_id: str
    client_id: str
    timestamp: datetime
    
    # Order details
    order_type: str
    direction: str
    quantity: float
    requested_price: float
    
    # Execution results
    execution_status: str
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    execution_time_ms: Optional[float] = None
    slippage_pips: Optional[float] = None
    commission: Optional[float] = None
    
    # Market data at execution
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread: Optional[float] = None
    
    # Error info (if failed)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class PerformanceLogEntry:
    """Performance tracking log entry"""
    # Required fields (no default values)
    timestamp: datetime
    client_id: str
    period_type: str  # DAILY/WEEKLY/MONTHLY
    total_pnl: float
    winning_trades: int
    losing_trades: int
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    signals_generated: int
    signals_executed: int
    execution_rate: float
    avg_signal_quality: float
    
    # Optional fields (with default values)
    sharpe_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
class LightningScalperDataLogger:
    """
    [DATABASE] Lightning Scalper Data Logger
    Comprehensive data collection system for ML and performance analysis
    """
    
    def __init__(self, data_dir: str = "data", db_name: str = "lightning_scalper.db"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / db_name
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "signals").mkdir(exist_ok=True)
        (self.data_dir / "executions").mkdir(exist_ok=True)
        (self.data_dir / "performance").mkdir(exist_ok=True)
        (self.data_dir / "market_data").mkdir(exist_ok=True)
        (self.data_dir / "exports").mkdir(exist_ok=True)
        (self.data_dir / "backups").mkdir(exist_ok=True)
        
        # Database connection
        self.db_connection = None
        self.db_lock = threading.Lock()
        
        # In-memory buffers for performance
        self.signal_buffer: List[SignalLogEntry] = []
        self.execution_buffer: List[ExecutionLogEntry] = []
        self.performance_buffer: List[PerformanceLogEntry] = []
        
        # Buffer limits
        self.max_buffer_size = 1000
        self.flush_interval = 60  # seconds
        
        # Background tasks
        self.flush_thread = None
        self.is_running = False
        
        # Performance tracking
        self.stats = {
            'signals_logged': 0,
            'executions_logged': 0,
            'performance_records': 0,
            'last_flush': None,
            'buffer_flushes': 0,
            'db_writes': 0,
            'errors': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger('SignalDataLogger')
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info("[DATABASE] Lightning Scalper Data Logger initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        log_type TEXT NOT NULL,
                        currency_pair TEXT,
                        timeframe TEXT,
                        signal_type TEXT,
                        confluence_score REAL,
                        entry_price REAL,
                        target_1 REAL,
                        target_2 REAL,
                        target_3 REAL,
                        stop_loss REAL,
                        gap_size REAL,
                        gap_percentage REAL,
                        market_condition TEXT,
                        session TEXT,
                        atr_ratio REAL,
                        volume_strength REAL,
                        momentum_score REAL,
                        structure_score REAL,
                        rsi_value REAL,
                        macd_signal REAL,
                        bollinger_position REAL,
                        support_resistance_distance REAL,
                        execution_delay REAL,
                        slippage REAL,
                        actual_entry REAL,
                        outcome TEXT,
                        pnl_pips REAL,
                        pnl_dollars REAL,
                        max_favorable_excursion REAL,
                        max_adverse_excursion REAL,
                        holding_time_minutes INTEGER,
                        client_id TEXT,
                        lot_size REAL,
                        account_balance REAL,
                        tags TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Executions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT UNIQUE NOT NULL,
                        signal_id TEXT,
                        client_id TEXT,
                        timestamp TEXT NOT NULL,
                        order_type TEXT,
                        direction TEXT,
                        quantity REAL,
                        requested_price REAL,
                        execution_status TEXT,
                        fill_price REAL,
                        fill_quantity REAL,
                        execution_time_ms REAL,
                        slippage_pips REAL,
                        commission REAL,
                        bid_price REAL,
                        ask_price REAL,
                        spread REAL,
                        error_code TEXT,
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                    )
                ''')
                
                # Performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        client_id TEXT,
                        period_type TEXT,
                        total_pnl REAL,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        total_trades INTEGER,
                        win_rate REAL,
                        avg_win REAL,
                        avg_loss REAL,
                        profit_factor REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        calmar_ratio REAL,
                        signals_generated INTEGER,
                        signals_executed INTEGER,
                        execution_rate REAL,
                        avg_signal_quality REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Market data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        spread REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_currency_pair ON signals (currency_pair)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_client_id ON signals (client_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON executions (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_executions_client_id ON executions (client_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp)')
                
                conn.commit()
                
            self.logger.info("[CHECK] Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[X] Database initialization failed: {e}")
            raise
    
    def start_logging(self):
        """Start the data logging system"""
        if not self.is_running:
            self.is_running = True
            
            # Start background flush thread
            self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self.flush_thread.start()
            
            self.logger.info("[ROCKET] Data logging system started")
    
    def stop_logging(self):
        """Stop the data logging system and flush remaining data"""
        if self.is_running:
            self.is_running = False
            
            # Flush remaining data
            self.flush_buffers()
            
            # Wait for flush thread to complete
            if self.flush_thread:
                self.flush_thread.join(timeout=10)
            
            self.logger.info("? Data logging system stopped")
    
    def _flush_loop(self):
        """Background thread for periodic buffer flushing"""
        while self.is_running:
            try:
                time.sleep(self.flush_interval)
                self.flush_buffers()
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")
                time.sleep(5)
    
    def log_signal(self, signal: FVGSignal, **kwargs):
        """Log FVG signal data"""
        try:
            log_entry = SignalLogEntry(
                signal_id=signal.id,
                timestamp=signal.timestamp,
                log_type="SIGNAL_GENERATED",
                currency_pair=signal.currency_pair.value,
                timeframe=signal.timeframe,
                signal_type=signal.fvg_type.value,
                confluence_score=signal.confluence_score,
                entry_price=signal.entry_price,
                target_1=signal.target_1,
                target_2=signal.target_2,
                target_3=signal.target_3,
                stop_loss=signal.stop_loss,
                gap_size=signal.gap_size,
                gap_percentage=signal.gap_percentage,
                market_condition=signal.market_condition.value,
                session=signal.session,
                atr_ratio=signal.atr_ratio,
                volume_strength=signal.volume_strength,
                momentum_score=signal.momentum_score,
                structure_score=signal.structure_score,
                tags=signal.tags.copy(),
                notes=signal.notes,
                **kwargs
            )
            
            self.signal_buffer.append(log_entry)
            self.stats['signals_logged'] += 1
            
            # Auto-flush if buffer is full
            if len(self.signal_buffer) >= self.max_buffer_size:
                self.flush_signals_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging signal {signal.id}: {e}")
            self.stats['errors'] += 1
    
    def log_execution(self, execution_data: Dict[str, Any]):
        """Log order execution data"""
        try:
            log_entry = ExecutionLogEntry(**execution_data)
            
            self.execution_buffer.append(log_entry)
            self.stats['executions_logged'] += 1
            
            # Auto-flush if buffer is full
            if len(self.execution_buffer) >= self.max_buffer_size:
                self.flush_executions_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging execution: {e}")
            self.stats['errors'] += 1
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        try:
            log_entry = PerformanceLogEntry(**performance_data)
            
            self.performance_buffer.append(log_entry)
            self.stats['performance_records'] += 1
            
            # Auto-flush if buffer is full
            if len(self.performance_buffer) >= self.max_buffer_size:
                self.flush_performance_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging performance: {e}")
            self.stats['errors'] += 1
    
    def update_signal_outcome(self, signal_id: str, outcome_data: Dict[str, Any]):
        """Update signal with execution outcome for supervised learning"""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Update signal with outcome data
                    update_fields = []
                    values = []
                    
                    for field, value in outcome_data.items():
                        if field in ['outcome', 'pnl_pips', 'pnl_dollars', 'max_favorable_excursion',
                                   'max_adverse_excursion', 'holding_time_minutes', 'execution_delay',
                                   'slippage', 'actual_entry']:
                            update_fields.append(f"{field} = ?")
                            values.append(value)
                    
                    if update_fields:
                        values.append(signal_id)
                        query = f"UPDATE signals SET {', '.join(update_fields)} WHERE signal_id = ?"
                        cursor.execute(query, values)
                        conn.commit()
                        
                        self.logger.debug(f"Updated signal {signal_id} with outcome data")
                    
        except Exception as e:
            self.logger.error(f"Error updating signal outcome {signal_id}: {e}")
            self.stats['errors'] += 1
    
    def flush_buffers(self):
        """Flush all buffers to database"""
        try:
            self.flush_signals_buffer()
            self.flush_executions_buffer()
            self.flush_performance_buffer()
            
            self.stats['last_flush'] = datetime.now()
            self.stats['buffer_flushes'] += 1
            
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")
            self.stats['errors'] += 1
    
    def flush_signals_buffer(self):
        """Flush signals buffer to database"""
        if not self.signal_buffer:
            return
        
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for entry in self.signal_buffer:
                        data = asdict(entry)
                        # Convert lists to JSON strings
                        data['tags'] = json.dumps(data['tags'])
                        data['timestamp'] = data['timestamp'].isoformat()
                        
                        placeholders = ', '.join(['?' for _ in data.keys()])
                        columns = ', '.join(data.keys())
                        
                        cursor.execute(
                            f"INSERT OR REPLACE INTO signals ({columns}) VALUES ({placeholders})",
                            list(data.values())
                        )
                    
                    conn.commit()
                    
            # Clear buffer
            buffer_size = len(self.signal_buffer)
            self.signal_buffer.clear()
            self.stats['db_writes'] += buffer_size
            
            self.logger.debug(f"Flushed {buffer_size} signal records to database")
            
        except Exception as e:
            self.logger.error(f"Error flushing signals buffer: {e}")
            self.stats['errors'] += 1
    
    def flush_executions_buffer(self):
        """Flush executions buffer to database"""
        if not self.execution_buffer:
            return
        
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for entry in self.execution_buffer:
                        data = asdict(entry)
                        data['timestamp'] = data['timestamp'].isoformat()
                        
                        placeholders = ', '.join(['?' for _ in data.keys()])
                        columns = ', '.join(data.keys())
                        
                        cursor.execute(
                            f"INSERT OR REPLACE INTO executions ({columns}) VALUES ({placeholders})",
                            list(data.values())
                        )
                    
                    conn.commit()
                    
            # Clear buffer
            buffer_size = len(self.execution_buffer)
            self.execution_buffer.clear()
            self.stats['db_writes'] += buffer_size
            
            self.logger.debug(f"Flushed {buffer_size} execution records to database")
            
        except Exception as e:
            self.logger.error(f"Error flushing executions buffer: {e}")
            self.stats['errors'] += 1
    
    def flush_performance_buffer(self):
        """Flush performance buffer to database"""
        if not self.performance_buffer:
            return
        
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for entry in self.performance_buffer:
                        data = asdict(entry)
                        data['timestamp'] = data['timestamp'].isoformat()
                        
                        placeholders = ', '.join(['?' for _ in data.keys()])
                        columns = ', '.join(data.keys())
                        
                        cursor.execute(
                            f"INSERT OR REPLACE INTO performance ({columns}) VALUES ({placeholders})",
                            list(data.values())
                        )
                    
                    conn.commit()
                    
            # Clear buffer
            buffer_size = len(self.performance_buffer)
            self.performance_buffer.clear()
            self.stats['db_writes'] += buffer_size
            
            self.logger.debug(f"Flushed {buffer_size} performance records to database")
            
        except Exception as e:
            self.logger.error(f"Error flushing performance buffer: {e}")
            self.stats['errors'] += 1
    
    def log_market_data(self, symbol: str, timeframe: str, ohlcv_data: Dict[str, Any]):
        """Log market data for analysis"""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO market_data 
                        (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_data['timestamp'],
                        symbol,
                        timeframe,
                        ohlcv_data.get('open'),
                        ohlcv_data.get('high'),
                        ohlcv_data.get('low'),
                        ohlcv_data.get('close'),
                        ohlcv_data.get('volume'),
                        ohlcv_data.get('spread')
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error logging market data: {e}")
            self.stats['errors'] += 1
    
    def get_signals_for_ml_training(self, start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None,
                                  include_outcomes_only: bool = True) -> pd.DataFrame:
        """Get signals data formatted for ML training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM signals WHERE 1=1"
                params = []
                
                if include_outcomes_only:
                    query += " AND outcome IS NOT NULL"
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse JSON fields
                if 'tags' in df.columns:
                    df['tags'] = df['tags'].apply(lambda x: json.loads(x) if x else [])
                
                # Convert datetime columns
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                self.logger.info(f"Retrieved {len(df)} signals for ML training")
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting ML training data: {e}")
            return pd.DataFrame()
    
    def get_performance_analytics(self, client_id: Optional[str] = None,
                                period: str = "30d") -> Dict[str, Any]:
        """Get performance analytics from logged data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate period start
                if period.endswith('d'):
                    days = int(period[:-1])
                    start_date = datetime.now() - timedelta(days=days)
                elif period.endswith('w'):
                    weeks = int(period[:-1])
                    start_date = datetime.now() - timedelta(weeks=weeks)
                else:
                    start_date = datetime.now() - timedelta(days=30)
                
                # Base query
                query = """
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as winning_signals,
                        COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losing_signals,
                        AVG(confluence_score) as avg_confluence,
                        AVG(CASE WHEN outcome = 'WIN' THEN pnl_pips END) as avg_win_pips,
                        AVG(CASE WHEN outcome = 'LOSS' THEN pnl_pips END) as avg_loss_pips,
                        SUM(pnl_dollars) as total_pnl
                    FROM signals 
                    WHERE timestamp >= ? AND outcome IS NOT NULL
                """
                
                params = [start_date.isoformat()]
                
                if client_id:
                    query += " AND client_id = ?"
                    params.append(client_id)
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    analytics = dict(zip(columns, result))
                    
                    # Calculate additional metrics
                    if analytics['total_signals'] > 0:
                        analytics['win_rate'] = (analytics['winning_signals'] / analytics['total_signals']) * 100
                    else:
                        analytics['win_rate'] = 0
                    
                    if analytics['avg_loss_pips'] and analytics['avg_loss_pips'] != 0:
                        analytics['profit_factor'] = abs(analytics['avg_win_pips'] / analytics['avg_loss_pips'])
                    else:
                        analytics['profit_factor'] = 0
                    
                    return analytics
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error getting performance analytics: {e}")
            return {}
    
    def export_data(self, format_type: DataFormat = DataFormat.CSV, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   tables: List[str] = None) -> Dict[str, str]:
        """Export data in various formats for analysis"""
        try:
            if tables is None:
                tables = ['signals', 'executions', 'performance']
            
            export_files = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for table in tables:
                try:
                    # Query data
                    with sqlite3.connect(self.db_path) as conn:
                        query = f"SELECT * FROM {table}"
                        params = []
                        
                        if start_date:
                            query += " WHERE timestamp >= ?"
                            params.append(start_date.isoformat())
                            
                            if end_date:
                                query += " AND timestamp <= ?"
                                params.append(end_date.isoformat())
                        elif end_date:
                            query += " WHERE timestamp <= ?"
                            params.append(end_date.isoformat())
                        
                        df = pd.read_sql_query(query, conn, params=params)
                    
                    if df.empty:
                        continue
                    
                    # Export based on format
                    if format_type == DataFormat.CSV:
                        filename = f"{table}_{timestamp}.csv"
                        filepath = self.data_dir / "exports" / filename
                        df.to_csv(filepath, index=False)
                        
                    elif format_type == DataFormat.JSON:
                        filename = f"{table}_{timestamp}.json"
                        filepath = self.data_dir / "exports" / filename
                        df.to_json(filepath, orient='records', date_format='iso')
                        
                    elif format_type == DataFormat.PICKLE:
                        filename = f"{table}_{timestamp}.pkl"
                        filepath = self.data_dir / "exports" / filename
                        df.to_pickle(filepath)
                        
                    elif format_type == DataFormat.PARQUET:
                        filename = f"{table}_{timestamp}.parquet"
                        filepath = self.data_dir / "exports" / filename
                        df.to_parquet(filepath)
                    
                    export_files[table] = str(filepath)
                    self.logger.info(f"Exported {len(df)} {table} records to {filename}")
                    
                except Exception as e:
                    self.logger.error(f"Error exporting {table}: {e}")
            
            return export_files
            
        except Exception as e:
            self.logger.error(f"Error in data export: {e}")
            return {}
    
    def create_ml_dataset(self, target_column: str = 'outcome',
                         feature_columns: List[str] = None,
                         lookback_days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Create ML-ready dataset with features and target"""
        try:
            # Get training data
            start_date = datetime.now() - timedelta(days=lookback_days)
            df = self.get_signals_for_ml_training(start_date=start_date)
            
            if df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Default feature columns
            if feature_columns is None:
                feature_columns = [
                    'confluence_score', 'gap_percentage', 'atr_ratio',
                    'volume_strength', 'momentum_score', 'structure_score',
                    'rsi_value', 'macd_signal', 'bollinger_position'
                ]
            
            # Select available feature columns
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Create feature matrix
            X = df[available_features].copy()
            
            # Create target variable
            if target_column == 'outcome':
                # Convert outcome to binary (WIN=1, LOSS=0)
                y = (df['outcome'] == 'WIN').astype(int)
            elif target_column in df.columns:
                y = df[target_column]
            else:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Remove rows with missing values
            combined = pd.concat([X, y], axis=1)
            combined = combined.dropna()
            
            X = combined[available_features]
            y = combined.iloc[:, -1]
            
            self.logger.info(f"Created ML dataset: {len(X)} samples, {len(available_features)} features")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error creating ML dataset: {e}")
            return pd.DataFrame(), pd.Series()
    
    def backup_database(self) -> str:
        """Create database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"lightning_scalper_backup_{timestamp}.db"
            backup_path = self.data_dir / "backups" / backup_filename
            
            # Create backup directory
            backup_path.parent.mkdir(exist_ok=True)
            
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            
            # Compress backup
            compressed_backup = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_backup, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_path.unlink()
            
            self.logger.info(f"Database backup created: {compressed_backup}")
            return str(compressed_backup)
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics"""
        try:
            # Get database stats
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                db_stats = {}
                for table in ['signals', 'executions', 'performance', 'market_data']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    db_stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Combine with runtime stats
            stats = {
                **self.stats,
                **db_stats,
                'buffer_sizes': {
                    'signals': len(self.signal_buffer),
                    'executions': len(self.execution_buffer),
                    'performance': len(self.performance_buffer)
                },
                'is_running': self.is_running
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return self.stats

# Demo and testing
def run_demo():
    """Run demo of the data logger"""
    print("[DATABASE] Lightning Scalper Data Logger - Demo")
    print("=" * 50)
    
    # Initialize logger
    logger = LightningScalperDataLogger(data_dir="demo_data")
    logger.start_logging()
    
    try:
        # Create sample signal data
        from core.lightning_scalper_engine import FVGSignal, FVGType, CurrencyPair, MarketCondition, FVGStatus
        
        sample_signal = FVGSignal(
            id="DEMO_SIGNAL_001",
            timestamp=datetime.now(),
            timeframe="M5",
            currency_pair=CurrencyPair.EURUSD,
            fvg_type=FVGType.BULLISH,
            high=1.1050,
            low=1.1030,
            gap_size=0.0020,
            gap_percentage=0.18,
            confluence_score=85.0,
            market_condition=MarketCondition.TRENDING_UP,
            session="London",
            status=FVGStatus.ACTIVE,
            entry_price=1.1045,
            target_1=1.1065,
            target_2=1.1075,
            target_3=1.1085,
            stop_loss=1.1025,
            risk_reward_ratio=1.5,
            position_size_factor=1.2,
            urgency_level=4,
            atr_ratio=1.1,
            volume_strength=25.0,
            momentum_score=18.0,
            structure_score=42.0,
            tags=["session_London", "condition_TRENDING_UP", "high_volume"]
        )
        
        # Log signal
        logger.log_signal(sample_signal, client_id="DEMO_CLIENT", lot_size=0.1)
        print("[CHECK] Sample signal logged")
        
        # Log execution
        execution_data = {
            'execution_id': 'EXEC_DEMO_001',
            'signal_id': 'DEMO_SIGNAL_001',
            'client_id': 'DEMO_CLIENT',
            'timestamp': datetime.now(),
            'order_type': 'LIMIT',
            'direction': 'BUY',
            'quantity': 0.1,
            'requested_price': 1.1045,
            'execution_status': 'FILLED',
            'fill_price': 1.1046,
            'fill_quantity': 0.1,
            'execution_time_ms': 250.5,
            'slippage_pips': 0.1
        }
        
        logger.log_execution(execution_data)
        print("[CHECK] Sample execution logged")
        
        # Update signal outcome
        outcome_data = {
            'outcome': 'WIN',
            'pnl_pips': 15.0,
            'pnl_dollars': 150.0,
            'holding_time_minutes': 45
        }
        
        logger.update_signal_outcome('DEMO_SIGNAL_001', outcome_data)
        print("[CHECK] Signal outcome updated")
        
        # Flush data
        logger.flush_buffers()
        print("[CHECK] Data flushed to database")
        
        # Get statistics
        stats = logger.get_statistics()
        print(f"\n[CHART] Statistics:")
        print(f"   Signals logged: {stats['signals_logged']}")
        print(f"   Executions logged: {stats['executions_logged']}")
        print(f"   Database records: {stats['signals_count']}")
        
        # Export data
        export_files = logger.export_data(DataFormat.CSV)
        print(f"\n? Exported files:")
        for table, filepath in export_files.items():
            print(f"   {table}: {filepath}")
        
        # Create ML dataset
        X, y = logger.create_ml_dataset()
        print(f"\n? ML Dataset: {len(X)} samples, {len(X.columns) if not X.empty else 0} features")
        
    finally:
        logger.stop_logging()
        print("\n[CHECK] Demo completed!")

if __name__ == "__main__":
    run_demo()