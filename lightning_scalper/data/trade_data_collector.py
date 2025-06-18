#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[ROCKET] Lightning Scalper - Trade Data Collector
Active Learning Data Collection System

ระบบเก็บข้อมูลการเทรดจากลูกค้าเพื่อนำมาปรับปรุงโมเดล AI:
- ✅ Trade results collection
- ✅ Performance metrics tracking  
- ✅ Market condition analysis
- ✅ Client behavior patterns
- ✅ Data synchronization to server
- ✅ Model improvement feedback

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import hashlib

class TradeOutcome(Enum):
    """Trade outcome types"""
    WIN = "win"
    LOSS = "loss" 
    BREAKEVEN = "breakeven"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

class MarketSession(Enum):
    """Trading sessions"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"

@dataclass
class TradeRecord:
    """Complete trade record for analysis"""
    # Basic trade info
    trade_id: str
    client_id: str
    signal_id: str
    timestamp: datetime
    
    # Market data
    currency_pair: str
    timeframe: str
    market_session: MarketSession
    
    # Signal data
    signal_type: str  # BUY/SELL
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    
    # FVG specific data
    fvg_gap_size: float
    fvg_confluence_score: float
    fvg_timeframe_alignment: bool
    fvg_volume_confirmation: bool
    
    # Market conditions
    market_volatility: float
    spread: float
    news_events_nearby: bool
    session_overlap: bool
    
    # Execution data
    execution_delay: float  # seconds from signal to execution
    slippage: float
    
    # Results
    outcome: TradeOutcome
    profit_loss: float
    profit_loss_pips: float
    trade_duration: timedelta
    risk_reward_ratio: float
    
    # Performance metrics
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float    # MAE
    
    # Client behavior
    manual_override: bool
    risk_adjustment: float  # vs suggested lot size
    
    # Additional context
    account_balance_before: float
    account_balance_after: float
    drawdown_at_entry: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects
        data['timestamp'] = self.timestamp.isoformat()
        data['trade_duration'] = self.trade_duration.total_seconds()
        data['market_session'] = self.market_session.value
        data['outcome'] = self.outcome.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['trade_duration'] = timedelta(seconds=data['trade_duration'])
        data['market_session'] = MarketSession(data['market_session'])
        data['outcome'] = TradeOutcome(data['outcome'])
        return cls(**data)

@dataclass
class PerformanceMetrics:
    """Client performance metrics"""
    client_id: str
    period_start: datetime
    period_end: datetime
    
    # Basic stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_profit: float
    total_loss: float
    net_profit: float
    average_win: float
    average_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    max_consecutive_losses: int
    sharpe_ratio: float
    calmar_ratio: float
    
    # Execution metrics
    average_execution_delay: float
    average_slippage: float
    
    # Signal quality metrics
    high_confidence_win_rate: float  # Signals >80% confidence
    low_confidence_win_rate: float   # Signals <60% confidence
    
    # Market condition performance
    london_session_performance: float
    new_york_session_performance: float
    overlap_session_performance: float
    
    # FVG specific metrics
    avg_fvg_gap_size_wins: float
    avg_fvg_gap_size_losses: float
    confluence_score_threshold: float

class TradeDataCollector:
    """
    [📊] Trade Data Collector
    Collects and analyzes trading data for AI model improvement
    """
    
    def __init__(self, data_dir: str = "data", server_url: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.db_path = self.data_dir / "trade_data.db"
        self.server_url = server_url
        
        self.logger = logging.getLogger('TradeDataCollector')
        self.collection_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Background sync settings
        self.sync_thread = None
        self.is_syncing = False
        self.sync_interval = 300  # 5 minutes
        self.last_sync = datetime.now()
        
        # Performance cache
        self.performance_cache = {}
        self.cache_expiry = timedelta(minutes=30)
        
        self.logger.info("📊 Trade Data Collector initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trade records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_records (
                        trade_id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        signal_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        currency_pair TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        profit_loss REAL NOT NULL,
                        profit_loss_pips REAL NOT NULL,
                        outcome TEXT NOT NULL,
                        trade_duration REAL NOT NULL,
                        fvg_confluence_score REAL,
                        market_session TEXT,
                        execution_delay REAL,
                        slippage REAL,
                        data_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        synced_to_server BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id TEXT NOT NULL,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        total_trades INTEGER,
                        win_rate REAL,
                        net_profit REAL,
                        max_drawdown REAL,
                        metrics_json TEXT NOT NULL,
                        calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Market analysis table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        currency_pair TEXT NOT NULL,
                        session TEXT NOT NULL,
                        total_signals INTEGER,
                        successful_signals INTEGER,
                        avg_gap_size REAL,
                        avg_confluence_score REAL,
                        market_conditions TEXT,
                        analysis_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_client ON trade_records(client_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_records(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_pair ON trade_records(currency_pair)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_sync ON trade_records(synced_to_server)')
                
                conn.commit()
                
            self.logger.info("📊 Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
            raise
    
    def record_trade(self, trade_record: TradeRecord):
        """Record a completed trade"""
        with self.collection_lock:
            try:
                trade_data = trade_record.to_dict()
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO trade_records (
                            trade_id, client_id, signal_id, timestamp,
                            currency_pair, signal_type, entry_price, exit_price,
                            profit_loss, profit_loss_pips, outcome, trade_duration,
                            fvg_confluence_score, market_session, execution_delay,
                            slippage, data_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_record.trade_id,
                        trade_record.client_id,
                        trade_record.signal_id,
                        trade_record.timestamp.isoformat(),
                        trade_record.currency_pair,
                        trade_record.signal_type,
                        trade_record.entry_price,
                        trade_record.exit_price,
                        trade_record.profit_loss,
                        trade_record.profit_loss_pips,
                        trade_record.outcome.value,
                        trade_record.trade_duration.total_seconds(),
                        trade_record.fvg_confluence_score,
                        trade_record.market_session.value,
                        trade_record.execution_delay,
                        trade_record.slippage,
                        json.dumps(trade_data)
                    ))
                    
                    conn.commit()
                
                self.logger.info(f"📝 Trade recorded: {trade_record.trade_id} - {trade_record.outcome.value}")
                
                # Invalidate performance cache for this client
                if trade_record.client_id in self.performance_cache:
                    del self.performance_cache[trade_record.client_id]
                
                # Queue for server sync
                self._queue_for_sync(trade_record)
                
            except Exception as e:
                self.logger.error(f"❌ Error recording trade: {e}")
                raise
    
    def calculate_performance_metrics(self, client_id: str, 
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a client"""
        
        # Check cache first
        cache_key = f"{client_id}_{start_date}_{end_date}"
        if cache_key in self.performance_cache:
            cached_metrics, cache_time = self.performance_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                return cached_metrics
        
        try:
            # Default date range (last 30 days)
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get trade data
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM trade_records 
                    WHERE client_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=[client_id, start_date.isoformat(), end_date.isoformat()]
                )
            
            if df.empty:
                # Return empty metrics
                return PerformanceMetrics(
                    client_id=client_id,
                    period_start=start_date,
                    period_end=end_date,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_profit=0.0,
                    total_loss=0.0,
                    net_profit=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    profit_factor=0.0,
                    max_drawdown=0.0,
                    max_consecutive_losses=0,
                    sharpe_ratio=0.0,
                    calmar_ratio=0.0,
                    average_execution_delay=0.0,
                    average_slippage=0.0,
                    high_confidence_win_rate=0.0,
                    low_confidence_win_rate=0.0,
                    london_session_performance=0.0,
                    new_york_session_performance=0.0,
                    overlap_session_performance=0.0,
                    avg_fvg_gap_size_wins=0.0,
                    avg_fvg_gap_size_losses=0.0,
                    confluence_score_threshold=80.0
                )
            
            # Basic statistics
            total_trades = len(df)
            winning_trades = len(df[df['profit_loss'] > 0])
            losing_trades = len(df[df['profit_loss'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L calculations
            total_profit = df[df['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(df[df['profit_loss'] < 0]['profit_loss'].sum())
            net_profit = df['profit_loss'].sum()
            
            average_win = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Risk metrics
            cumulative_pnl = df['profit_loss'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max()
            
            # Consecutive losses
            loss_streak = 0
            max_consecutive_losses = 0
            for _, row in df.iterrows():
                if row['profit_loss'] < 0:
                    loss_streak += 1
                    max_consecutive_losses = max(max_consecutive_losses, loss_streak)
                else:
                    loss_streak = 0
            
            # Execution metrics
            average_execution_delay = df['execution_delay'].mean()
            average_slippage = df['slippage'].mean()
            
            # Signal quality analysis
            high_conf_trades = df[df['fvg_confluence_score'] >= 80]
            low_conf_trades = df[df['fvg_confluence_score'] < 60]
            
            high_confidence_win_rate = (
                len(high_conf_trades[high_conf_trades['profit_loss'] > 0]) / len(high_conf_trades)
                if len(high_conf_trades) > 0 else 0
            )
            
            low_confidence_win_rate = (
                len(low_conf_trades[low_conf_trades['profit_loss'] > 0]) / len(low_conf_trades)
                if len(low_conf_trades) > 0 else 0
            )
            
            # Session performance
            def session_performance(session_name: str) -> float:
                session_trades = df[df['market_session'] == session_name]
                return session_trades['profit_loss'].sum() if len(session_trades) > 0 else 0
            
            london_performance = session_performance('london')
            new_york_performance = session_performance('new_york')
            overlap_performance = session_performance('overlap')
            
            # FVG analysis
            winning_trades_df = df[df['profit_loss'] > 0]
            losing_trades_df = df[df['profit_loss'] < 0]
            
            # Parse JSON data to get FVG gap sizes
            def get_gap_size(data_json_str):
                try:
                    data = json.loads(data_json_str)
                    return data.get('fvg_gap_size', 0)
                except:
                    return 0
            
            df['gap_size'] = df['data_json'].apply(get_gap_size)
            
            avg_gap_size_wins = winning_trades_df['gap_size'].mean() if len(winning_trades_df) > 0 else 0
            avg_gap_size_losses = losing_trades_df['gap_size'].mean() if len(losing_trades_df) > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            returns = df['profit_loss']
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Calmar ratio
            calmar_ratio = abs(net_profit / max_drawdown) if max_drawdown > 0 else 0
            
            # Optimal confluence score threshold
            confluence_threshold = 80.0  # Default
            if len(df) >= 20:  # Enough data for analysis
                thresholds = [60, 70, 80, 85, 90]
                best_performance = -float('inf')
                for threshold in thresholds:
                    high_conf = df[df['fvg_confluence_score'] >= threshold]
                    if len(high_conf) >= 5:  # Minimum sample size
                        performance = high_conf['profit_loss'].sum()
                        if performance > best_performance:
                            best_performance = performance
                            confluence_threshold = threshold
            
            metrics = PerformanceMetrics(
                client_id=client_id,
                period_start=start_date,
                period_end=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_profit=total_profit,
                total_loss=total_loss,
                net_profit=net_profit,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                max_consecutive_losses=max_consecutive_losses,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                average_execution_delay=average_execution_delay,
                average_slippage=average_slippage,
                high_confidence_win_rate=high_confidence_win_rate,
                low_confidence_win_rate=low_confidence_win_rate,
                london_session_performance=london_performance,
                new_york_session_performance=new_york_performance,
                overlap_session_performance=overlap_performance,
                avg_fvg_gap_size_wins=avg_gap_size_wins,
                avg_fvg_gap_size_losses=avg_gap_size_losses,
                confluence_score_threshold=confluence_threshold
            )
            
            # Cache the result
            self.performance_cache[cache_key] = (metrics, datetime.now())
            
            # Store in database
            self._store_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            raise
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        client_id, period_start, period_end, total_trades,
                        win_rate, net_profit, max_drawdown, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.client_id,
                    metrics.period_start.isoformat(),
                    metrics.period_end.isoformat(),
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.net_profit,
                    metrics.max_drawdown,
                    json.dumps(asdict(metrics), default=str)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Error storing performance metrics: {e}")
    
    def analyze_market_conditions(self, date: datetime, 
                                currency_pair: str) -> Dict[str, Any]:
        """Analyze market conditions and signal performance"""
        try:
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM trade_records 
                    WHERE currency_pair = ? AND timestamp >= ? AND timestamp < ?
                '''
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[currency_pair, start_date.isoformat(), end_date.isoformat()]
                )
            
            if df.empty:
                return {
                    'date': date.isoformat(),
                    'currency_pair': currency_pair,
                    'total_signals': 0,
                    'analysis': 'No data available'
                }
            
            # Analyze by session
            session_analysis = {}
            for session in ['london', 'new_york', 'tokyo', 'overlap']:
                session_trades = df[df['market_session'] == session]
                if len(session_trades) > 0:
                    session_analysis[session] = {
                        'total_trades': len(session_trades),
                        'successful_trades': len(session_trades[session_trades['profit_loss'] > 0]),
                        'win_rate': len(session_trades[session_trades['profit_loss'] > 0]) / len(session_trades),
                        'avg_profit': session_trades['profit_loss'].mean(),
                        'avg_confluence_score': session_trades['fvg_confluence_score'].mean()
                    }
            
            # Overall analysis
            total_signals = len(df)
            successful_signals = len(df[df['profit_loss'] > 0])
            win_rate = successful_signals / total_signals if total_signals > 0 else 0
            
            # FVG analysis
            avg_gap_size = df['gap_size'].mean() if 'gap_size' in df.columns else 0
            avg_confluence_score = df['fvg_confluence_score'].mean()
            
            # Market volatility proxy
            price_ranges = []
            for _, trade in df.iterrows():
                try:
                    trade_data = json.loads(trade['data_json'])
                    max_fav = trade_data.get('max_favorable_excursion', 0)
                    max_adv = trade_data.get('max_adverse_excursion', 0)
                    price_ranges.append(max_fav + abs(max_adv))
                except:
                    pass
            
            avg_volatility = np.mean(price_ranges) if price_ranges else 0
            
            analysis = {
                'date': date.isoformat(),
                'currency_pair': currency_pair,
                'total_signals': total_signals,
                'successful_signals': successful_signals,
                'win_rate': win_rate,
                'avg_gap_size': avg_gap_size,
                'avg_confluence_score': avg_confluence_score,
                'avg_volatility': avg_volatility,
                'session_analysis': session_analysis,
                'recommendations': self._generate_recommendations(df)
            }
            
            # Store analysis
            self._store_market_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing market conditions: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on data analysis"""
        recommendations = []
        
        if len(df) < 10:
            recommendations.append("Insufficient data for meaningful recommendations")
            return recommendations
        
        # Win rate analysis
        win_rate = len(df[df['profit_loss'] > 0]) / len(df)
        if win_rate < 0.4:
            recommendations.append("Consider increasing confluence score threshold")
        elif win_rate > 0.7:
            recommendations.append("Excellent performance - consider increasing position size")
        
        # Confluence score analysis
        high_conf_trades = df[df['fvg_confluence_score'] >= 80]
        if len(high_conf_trades) > 0:
            high_conf_win_rate = len(high_conf_trades[high_conf_trades['profit_loss'] > 0]) / len(high_conf_trades)
            if high_conf_win_rate > win_rate + 0.1:
                recommendations.append("Focus on higher confluence score signals (>80%)")
        
        # Session analysis
        session_performance = {}
        for session in ['london', 'new_york', 'overlap']:
            session_trades = df[df['market_session'] == session]
            if len(session_trades) >= 3:
                session_performance[session] = session_trades['profit_loss'].sum()
        
        if session_performance:
            best_session = max(session_performance, key=session_performance.get)
            recommendations.append(f"Best performance during {best_session} session")
        
        # Execution analysis
        avg_delay = df['execution_delay'].mean()
        if avg_delay > 5.0:  # More than 5 seconds
            recommendations.append("Improve execution speed - high delays detected")
        
        avg_slippage = df['slippage'].mean()
        if avg_slippage > 2.0:  # More than 2 pips
            recommendations.append("Consider trading during higher liquidity periods")
        
        return recommendations
    
    def _store_market_analysis(self, analysis: Dict[str, Any]):
        """Store market analysis in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO market_analysis (
                        date, currency_pair, total_signals, successful_signals,
                        avg_gap_size, avg_confluence_score, analysis_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis['date'],
                    analysis['currency_pair'],
                    analysis['total_signals'],
                    analysis['successful_signals'],
                    analysis.get('avg_gap_size', 0),
                    analysis.get('avg_confluence_score', 0),
                    json.dumps(analysis)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Error storing market analysis: {e}")
    
    def _queue_for_sync(self, trade_record: TradeRecord):
        """Queue trade record for server synchronization"""
        if not self.server_url:
            return
        
        # Start sync thread if not running
        if not self.is_syncing:
            self.start_background_sync()
    
    def start_background_sync(self):
        """Start background synchronization with server"""
        if self.is_syncing or not self.server_url:
            return
        
        self.is_syncing = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        self.logger.info("🔄 Background sync started")
    
    def stop_background_sync(self):
        """Stop background synchronization"""
        self.is_syncing = False
        if self.sync_thread:
            self.sync_thread.join(timeout=10)
        self.logger.info("🛑 Background sync stopped")
    
    def _sync_loop(self):
        """Background synchronization loop"""
        while self.is_syncing:
            try:
                # Sync unsynced trade records
                self._sync_trade_records()
                
                # Sync performance metrics
                self._sync_performance_metrics()
                
                time.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Sync loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _sync_trade_records(self):
        """Sync unsynced trade records to server"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get unsynced records
                cursor.execute('''
                    SELECT trade_id, data_json FROM trade_records 
                    WHERE synced_to_server = FALSE
                    LIMIT 100
                ''')
                
                unsynced_records = cursor.fetchall()
                
                if not unsynced_records:
                    return
                
                # Prepare batch data
                batch_data = []
                trade_ids = []
                
                for trade_id, data_json in unsynced_records:
                    try:
                        trade_data = json.loads(data_json)
                        batch_data.append(trade_data)
                        trade_ids.append(trade_id)
                    except Exception as e:
                        self.logger.error(f"❌ Error parsing trade data {trade_id}: {e}")
                
                if not batch_data:
                    return
                
                # Send to server
                success = self._send_to_server('/api/trade_data', {
                    'trades': batch_data,
                    'timestamp': datetime.now().isoformat(),
                    'client_count': len(set(t['client_id'] for t in batch_data))
                })
                
                if success:
                    # Mark as synced
                    placeholders = ','.join(['?' for _ in trade_ids])
                    cursor.execute(f'''
                        UPDATE trade_records 
                        SET synced_to_server = TRUE 
                        WHERE trade_id IN ({placeholders})
                    ''', trade_ids)
                    
                    conn.commit()
                    self.last_sync = datetime.now()
                    
                    self.logger.info(f"📤 Synced {len(trade_ids)} trade records to server")
                
        except Exception as e:
            self.logger.error(f"❌ Error syncing trade records: {e}")
    
    def _sync_performance_metrics(self):
        """Sync performance metrics to server"""
        try:
            # Get recent performance metrics
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT metrics_json FROM performance_metrics 
                    WHERE calculated_at >= ?
                ''', (cutoff_time.isoformat(),))
                
                metrics_data = []
                for (metrics_json,) in cursor.fetchall():
                    try:
                        metrics_data.append(json.loads(metrics_json))
                    except Exception as e:
                        self.logger.error(f"❌ Error parsing metrics: {e}")
                
                if metrics_data:
                    success = self._send_to_server('/api/performance_data', {
                        'metrics': metrics_data,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if success:
                        self.logger.info(f"📤 Synced {len(metrics_data)} performance metrics")
                
        except Exception as e:
            self.logger.error(f"❌ Error syncing performance metrics: {e}")
    
    def _send_to_server(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send data to server"""
        try:
            url = f"{self.server_url.rstrip('/')}{endpoint}"
            
            # Compress data
            json_data = json.dumps(data).encode('utf-8')
            compressed_data = gzip.compress(json_data)
            
            # Add checksum
            checksum = hashlib.md5(json_data).hexdigest()
            
            headers = {
                'Content-Type': 'application/json',
                'Content-Encoding': 'gzip',
                'X-Data-Checksum': checksum,
                'X-Data-Source': 'lightning_scalper_client'
            }
            
            response = requests.post(
                url,
                data=compressed_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"❌ Server sync failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error sending to server: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trade records summary
                cursor.execute('SELECT COUNT(*) FROM trade_records')
                total_trades = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT client_id) FROM trade_records')
                unique_clients = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM trade_records WHERE synced_to_server = FALSE')
                unsynced_trades = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT 
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                        AVG(fvg_confluence_score) as avg_confluence
                    FROM trade_records
                ''')
                wins, losses, avg_confluence = cursor.fetchone()
                
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM trade_records')
                date_range = cursor.fetchone()
                
                return {
                    'total_trades': total_trades,
                    'unique_clients': unique_clients,
                    'unsynced_trades': unsynced_trades,
                    'winning_trades': wins or 0,
                    'losing_trades': losses or 0,
                    'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
                    'avg_confluence_score': avg_confluence or 0,
                    'date_range': date_range,
                    'last_sync': self.last_sync.isoformat(),
                    'sync_enabled': self.server_url is not None,
                    'database_size_mb': self.db_path.stat().st_size / 1024 / 1024
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting data summary: {e}")
            return {'error': str(e)}

# ================================
# TESTING AND DEMO
# ================================

def create_sample_trade_record() -> TradeRecord:
    """Create a sample trade record for testing"""
    return TradeRecord(
        trade_id=f"TRADE_{int(time.time())}",
        client_id="CLIENT_001",
        signal_id=f"SIGNAL_{int(time.time())}",
        timestamp=datetime.now(),
        currency_pair="EURUSD",
        timeframe="M5",
        market_session=MarketSession.LONDON,
        signal_type="BUY",
        entry_price=1.10850,
        exit_price=1.11150,
        stop_loss=1.10550,
        take_profit=1.11200,
        fvg_gap_size=15.0,
        fvg_confluence_score=85.5,
        fvg_timeframe_alignment=True,
        fvg_volume_confirmation=True,
        market_volatility=12.5,
        spread=1.2,
        news_events_nearby=False,
        session_overlap=False,
        execution_delay=2.3,
        slippage=0.5,
        outcome=TradeOutcome.WIN,
        profit_loss=300.0,
        profit_loss_pips=30.0,
        trade_duration=timedelta(minutes=45),
        risk_reward_ratio=2.0,
        max_favorable_excursion=35.0,
        max_adverse_excursion=-8.0,
        manual_override=False,
        risk_adjustment=1.0,
        account_balance_before=10000.0,
        account_balance_after=10300.0,
        drawdown_at_entry=0.0
    )

def test_trade_data_collector():
    """Test the trade data collector"""
    print("🧪 Testing Trade Data Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = TradeDataCollector(
        data_dir="test_data",
        server_url=None  # No server sync for testing
    )
    
    try:
        # Create sample trades
        print("📝 Creating sample trade records...")
        for i in range(5):
            trade = create_sample_trade_record()
            trade.client_id = f"CLIENT_{i % 2 + 1}"  # Two clients
            trade.profit_loss = (-1) ** i * abs(trade.profit_loss)  # Mix wins/losses
            trade.outcome = TradeOutcome.WIN if trade.profit_loss > 0 else TradeOutcome.LOSS
            
            collector.record_trade(trade)
            print(f"   Trade {i+1}: {trade.outcome.value} - ${trade.profit_loss}")
        
        # Calculate performance metrics
        print("\n📊 Calculating performance metrics...")
        for client_id in ["CLIENT_1", "CLIENT_2"]:
            metrics = collector.calculate_performance_metrics(client_id)
            print(f"\n{client_id} Performance:")
            print(f"   Total Trades: {metrics.total_trades}")
            print(f"   Win Rate: {metrics.win_rate:.1%}")
            print(f"   Net Profit: ${metrics.net_profit:.2f}")
            print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        
        # Market analysis
        print("\n📈 Market analysis...")
        analysis = collector.analyze_market_conditions(datetime.now(), "EURUSD")
        print(f"   Total Signals: {analysis['total_signals']}")
        print(f"   Win Rate: {analysis.get('win_rate', 0):.1%}")
        print(f"   Recommendations: {len(analysis.get('recommendations', []))}")
        
        # Data summary
        print("\n📋 Data summary...")
        summary = collector.get_data_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Trade Data Collector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trade_data_collector()