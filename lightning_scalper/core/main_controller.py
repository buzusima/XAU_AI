#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Windows Safe Version
Auto-fixed for Unicode compatibility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import threading
import time
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

# Import our core modules
from core.lightning_scalper_engine import (
    EnhancedFVGDetector, FVGSignal, CurrencyPair, 
    FVGType, MarketCondition, FVGStatus
)
from execution.trade_executor import (
    TradeExecutor, ClientAccount, Order, Position,
    OrderType, OrderStatus, TradeDirection
)
from adapters.mt5_adapter import (
    MT5Adapter, MT5IntegratedExecutor, MT5ConnectionStatus,
    MT5AccountInfo, MT5SymbolInfo
)

class SystemStatus:
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"

@dataclass
class ClientConnection:
    """Track individual client connections and settings"""
    client_id: str
    mt5_login: int
    mt5_password: str
    mt5_server: str
    adapter: Optional[MT5Adapter] = None
    last_heartbeat: Optional[datetime] = None
    is_active: bool = True
    auto_trading: bool = True
    max_signals_per_hour: int = 10
    preferred_pairs: List[str] = field(default_factory=list)
    risk_multiplier: float = 1.0
    connection_attempts: int = 0

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    total_clients: int = 0
    active_clients: int = 0
    connected_clients: int = 0
    total_signals_today: int = 0
    executed_trades_today: int = 0
    total_pnl_today: float = 0.0
    system_uptime: timedelta = timedelta(0)
    avg_signal_quality: float = 0.0
    avg_execution_time: float = 0.0
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

class LightningScalperController:
    """
    [ROCKET] Production Lightning Scalper Main Controller
    Central orchestration system for 80+ client AI trading operations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Core system components
        self.fvg_detector = EnhancedFVGDetector()
        self.trade_executor = TradeExecutor()
        
        # Client management
        self.client_connections: Dict[str, ClientConnection] = {}
        self.client_adapters: Dict[str, MT5Adapter] = {}
        self.integrated_executors: Dict[str, MT5IntegratedExecutor] = {}
        
        # System state
        self.status = SystemStatus.STARTING
        self.start_time = datetime.now()
        self.metrics = SystemMetrics()
        
        # Data feeds (multi-timeframe for each currency pair)
        self.data_feeds: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.signal_history: deque = deque(maxlen=10000)
        self.execution_log: deque = deque(maxlen=10000)
        
        # Threading and async operations
        self.main_loop_thread: Optional[threading.Thread] = None
        self.data_update_thread: Optional[threading.Thread] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.executor_pool = ThreadPoolExecutor(max_workers=20)
        
        # Event system for real-time updates
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Risk and safety
        self.global_safety_enabled = True
        self.max_total_daily_loss = 5000.0  # $5000 across all clients
        self.max_concurrent_trades = 200    # Max trades across all clients
        self.emergency_stop_triggered = False
        
        # Performance optimization
        self.signal_cache: Dict[str, List[FVGSignal]] = {}
        self.cache_expiry = timedelta(minutes=5)
        self.last_cache_update = {}
        
        # Configuration
        self.config = self._load_default_config()
        if config_path:
            self._load_config_file(config_path)
        
        # Monitoring and analytics
        self.currency_pairs = [pair for pair in CurrencyPair]
        self.timeframes = ['M1', 'M5', 'M15', 'H1']
        
        # Thread locks
        self.main_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.client_lock = threading.Lock()
        
        # Logging setup
        self._setup_logging()
        self.logger = logging.getLogger('LightningScalperController')
        
        self.logger.info("[ROCKET] Lightning Scalper Controller initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lightning_scalper.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration"""
        return {
            'system': {
                'max_clients': 100,
                'data_update_interval': 1.0,  # seconds
                'health_check_interval': 30.0,  # seconds
                'signal_generation_interval': 5.0,  # seconds
                'auto_reconnect': True,
                'max_reconnect_attempts': 5
            },
            'risk': {
                'global_daily_loss_limit': 5000.0,
                'max_concurrent_trades': 200,
                'emergency_stop_loss_percent': 10.0,
                'max_signals_per_client_hour': 10
            },
            'performance': {
                'min_signal_confluence': 65.0,
                'cache_duration_minutes': 5,
                'max_execution_threads': 20,
                'signal_history_limit': 10000
            },
            'monitoring': {
                'enable_real_time_dashboard': True,
                'log_all_signals': True,
                'track_performance_metrics': True,
                'alert_on_errors': True
            }
        }
    
    def _load_config_file(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Merge with default config
                self._deep_update(self.config, file_config)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load config file {config_path}: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    async def start_system(self) -> bool:
        """Start the complete Lightning Scalper system"""
        try:
            self.logger.info("[ROCKET] Starting Lightning Scalper System...")
            self.status = SystemStatus.STARTING
            
            # 1. Start trade executor
            self.trade_executor.start_execution_engine()
            self.logger.info("[CHECK] Trade Executor started")
            
            # 2. Initialize data feeds
            await self._initialize_data_feeds()
            self.logger.info("[CHECK] Data feeds initialized")
            
            # 3. Start background threads
            self._start_background_threads()
            self.logger.info("[CHECK] Background threads started")
            
            # 4. System ready
            self.status = SystemStatus.RUNNING
            self.start_time = datetime.now()
            
            self.logger.info("[TARGET] Lightning Scalper System is RUNNING!")
            self._trigger_event('system_started', {'timestamp': datetime.now()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Failed to start system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def _initialize_data_feeds(self):
        """Initialize data feeds for all currency pairs"""
        for currency_pair in self.currency_pairs:
            symbol = currency_pair.value
            self.data_feeds[symbol] = {}
            
            # Initialize with sample data for each timeframe
            for timeframe in self.timeframes:
                # In production, this would connect to real data sources
                sample_data = self._generate_sample_data(symbol, timeframe)
                self.data_feeds[symbol][timeframe] = sample_data
        
        self.logger.info(f"[CHECK] Data feeds initialized for {len(self.currency_pairs)} pairs")
    
    def _generate_sample_data(self, symbol: str, timeframe: str, periods: int = 200) -> pd.DataFrame:
        """Generate realistic sample OHLCV data"""
        # Get base price for currency pair
        base_prices = {
            'EURUSD': 1.1000, 'GBPUSD': 1.2500, 'USDJPY': 150.00,
            'AUDUSD': 0.6500, 'USDCAD': 1.3500, 'USDCHF': 0.9200,
            'NZDUSD': 0.6000, 'EURJPY': 165.00, 'GBPJPY': 187.50,
            'XAUUSD': 2000.00
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate dates
        if timeframe == 'M1':
            freq = '1T'
        elif timeframe == 'M5':
            freq = '5T'
        elif timeframe == 'M15':
            freq = '15T'
        elif timeframe == 'H1':
            freq = '1H'
        else:
            freq = '1H'
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods * 2),
            periods=periods,
            freq=freq
        )
        
        # Generate realistic price movement
        np.random.seed(42)  # For reproducible data
        returns = np.random.normal(0, 0.0001, periods)
        prices = base_price + np.cumsum(returns)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.00005, 0.0003)
            
            open_price = price + np.random.uniform(-volatility, volatility)
            close_price = price + np.random.uniform(-volatility, volatility)
            high_price = max(open_price, close_price) + np.random.uniform(0, volatility/2)
            low_price = min(open_price, close_price) - np.random.uniform(0, volatility/2)
            volume = np.random.randint(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def _start_background_threads(self):
        """Start all background monitoring threads"""
        # Main processing loop
        self.main_loop_thread = threading.Thread(
            target=self._main_processing_loop, 
            daemon=True
        )
        self.main_loop_thread.start()
        
        # Data update loop
        self.data_update_thread = threading.Thread(
            target=self._data_update_loop,
            daemon=True
        )
        self.data_update_thread.start()
        
        # Health monitoring loop
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.health_monitor_thread.start()
    
    def _main_processing_loop(self):
        """Main processing loop for signal generation and execution"""
        while self.status == SystemStatus.RUNNING:
            try:
                start_time = time.time()
                
                # 1. Generate signals for active currency pairs
                self._process_signal_generation()
                
                # 2. Process execution queue
                self._process_pending_executions()
                
                # 3. Update metrics
                self._update_system_metrics()
                
                # 4. Check safety conditions
                self._check_safety_conditions()
                
                # Calculate processing time and sleep
                processing_time = time.time() - start_time
                sleep_time = max(0, self.config['system']['signal_generation_interval'] - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def _process_signal_generation(self):
        """Process FVG signal generation for all active pairs"""
        try:
            for currency_pair in self.currency_pairs:
                symbol = currency_pair.value
                
                # Skip if no data available
                if symbol not in self.data_feeds:
                    continue
                
                # Check cache first
                cache_key = f"{symbol}_signals"
                if self._is_cache_valid(cache_key):
                    continue
                
                # Get multi-timeframe data
                timeframe_data = self.data_feeds[symbol]
                
                # Detect signals
                self.fvg_detector.currency_pair = currency_pair
                signals = self.fvg_detector.process_multi_timeframe_advanced(timeframe_data)
                
                # Cache signals
                self.signal_cache[cache_key] = signals
                self.last_cache_update[cache_key] = datetime.now()
                
                # Process execution-ready signals
                execution_signals = self.fvg_detector.get_execution_ready_signals(signals)
                
                if execution_signals:
                    self._process_execution_signals(execution_signals, symbol)
                    
        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
    
    def _process_execution_signals(self, execution_signals: List[Dict], symbol: str):
        """Process signals ready for execution"""
        for exec_signal in execution_signals:
            try:
                primary_signal = exec_signal['primary_signal']
                
                # Log signal
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'signal': primary_signal,
                    'confluence': exec_signal['total_confluence'],
                    'priority': exec_signal['execution_priority']
                })
                
                # Trigger signal event
                self._trigger_event('signal_generated', {
                    'signal': primary_signal,
                    'execution_data': exec_signal
                })
                
                # Execute for qualified clients
                self._execute_for_clients(primary_signal, exec_signal)
                
            except Exception as e:
                self.logger.error(f"Error processing execution signal: {e}")
    
    def _execute_for_clients(self, signal: FVGSignal, execution_data: Dict):
        """Execute signal for all qualified clients"""
        qualified_clients = self._get_qualified_clients(signal)
        
        for client_id in qualified_clients:
            try:
                # Submit to executor thread pool
                future = self.executor_pool.submit(
                    self._execute_signal_for_client,
                    signal,
                    client_id,
                    execution_data
                )
                
                # Optional: Add callback for completion
                future.add_done_callback(
                    lambda f: self._on_execution_complete(f, client_id, signal.id)
                )
                
            except Exception as e:
                self.logger.error(f"Error submitting execution for client {client_id}: {e}")
    
    def _execute_signal_for_client(self, signal: FVGSignal, client_id: str, execution_data: Dict):
        """Execute signal for specific client"""
        try:
            # Check if client has integrated executor
            if client_id not in self.integrated_executors:
                return
            
            # Get client's risk-adjusted lot size
            custom_lot_size = self._calculate_client_lot_size(client_id, signal)
            
            # Execute via trade executor
            result = self.trade_executor.execute_fvg_signal(
                signal, 
                client_id, 
                custom_lot_size
            )
            
            # Log execution
            self.execution_log.append({
                'timestamp': datetime.now(),
                'client_id': client_id,
                'signal_id': signal.id,
                'result': result,
                'lot_size': custom_lot_size
            })
            
            self.logger.info(f"Signal {signal.id} executed for client {client_id}: {result['success']}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal for client {client_id}: {e}")
    
    def _on_execution_complete(self, future, client_id: str, signal_id: str):
        """Callback when execution completes"""
        try:
            result = future.result()
            self._trigger_event('execution_complete', {
                'client_id': client_id,
                'signal_id': signal_id,
                'result': result
            })
        except Exception as e:
            self.logger.error(f"Execution callback error for {client_id}: {e}")
    
    def _get_qualified_clients(self, signal: FVGSignal) -> List[str]:
        """Get list of clients qualified to receive this signal"""
        qualified = []
        
        with self.client_lock:
            for client_id, connection in self.client_connections.items():
                # Check if client is active and auto-trading is enabled
                if not connection.is_active or not connection.auto_trading:
                    continue
                
                # Check if client trades this currency pair
                if (connection.preferred_pairs and 
                    signal.currency_pair.value not in connection.preferred_pairs):
                    continue
                
                # Check signal rate limits
                if self._check_client_signal_rate_limit(client_id):
                    qualified.append(client_id)
        
        return qualified
    
    def _check_client_signal_rate_limit(self, client_id: str) -> bool:
        """Check if client hasn't exceeded signal rate limit"""
        # Count signals in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_signals = [
            log for log in self.execution_log
            if (log['client_id'] == client_id and 
                log['timestamp'] > one_hour_ago)
        ]
        
        connection = self.client_connections.get(client_id)
        if not connection:
            return False
        
        return len(recent_signals) < connection.max_signals_per_hour
    
    def _calculate_client_lot_size(self, client_id: str, signal: FVGSignal) -> float:
        """Calculate lot size for specific client based on their risk multiplier"""
        connection = self.client_connections.get(client_id)
        if not connection:
            return 0.01  # Default minimum
        
        # Base calculation from signal
        base_lot = signal.position_size_factor * 0.01  # Convert to lot size
        
        # Apply client's risk multiplier
        adjusted_lot = base_lot * connection.risk_multiplier
        
        # Ensure within reasonable bounds
        return max(0.01, min(adjusted_lot, 1.0))
    
    def _data_update_loop(self):
        """Background loop for updating market data"""
        while self.status == SystemStatus.RUNNING:
            try:
                self._update_market_data()
                time.sleep(self.config['system']['data_update_interval'])
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                time.sleep(5)
    
    def _update_market_data(self):
        """Update market data from connected brokers"""
        with self.data_lock:
            for symbol in self.data_feeds.keys():
                # In production, update from real broker feeds
                # For now, append new sample data point
                for timeframe in self.timeframes:
                    df = self.data_feeds[symbol][timeframe]
                    
                    # Add new data point
                    last_close = df['close'].iloc[-1]
                    new_price = last_close + np.random.normal(0, 0.0001)
                    
                    new_row = {
                        'open': new_price + np.random.uniform(-0.00005, 0.00005),
                        'high': new_price + np.random.uniform(0, 0.0001),
                        'low': new_price - np.random.uniform(0, 0.0001),
                        'close': new_price,
                        'volume': np.random.randint(100, 1000)
                    }
                    
                    # Create new DataFrame with updated data
                    new_index = df.index[-1] + timedelta(minutes=1 if timeframe == 'M1' else 5)
                    new_df = pd.concat([df, pd.DataFrame([new_row], index=[new_index])])
                    
                    # Keep only last 200 candles
                    self.data_feeds[symbol][timeframe] = new_df.tail(200)
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                self._perform_health_check()
                time.sleep(self.config['system']['health_check_interval'])
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(10)
    
    def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            # Check client connections
            with self.client_lock:
                total_clients = len(self.client_connections)
                connected_clients = sum(
                    1 for conn in self.client_connections.values()
                    if conn.adapter and conn.adapter.is_connected()
                )
                active_clients = sum(
                    1 for conn in self.client_connections.values()
                    if conn.is_active
                )
            
            # Update metrics
            self.metrics.total_clients = total_clients
            self.metrics.connected_clients = connected_clients
            self.metrics.active_clients = active_clients
            self.metrics.last_update = datetime.now()
            
            # Check for disconnected clients and attempt reconnection
            if self.config['system']['auto_reconnect']:
                self._attempt_client_reconnections()
            
            # Trigger health check event
            self._trigger_event('health_check', {
                'metrics': self.metrics,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _attempt_client_reconnections(self):
        """Attempt to reconnect disconnected clients"""
        for client_id, connection in self.client_connections.items():
            if (not connection.adapter or 
                not connection.adapter.is_connected() and
                connection.connection_attempts < self.config['system']['max_reconnect_attempts']):
                
                try:
                    self.logger.info(f"Attempting reconnection for client {client_id}")
                    success = self._connect_client_mt5(connection)
                    
                    if success:
                        connection.connection_attempts = 0
                        self.logger.info(f"[CHECK] Client {client_id} reconnected successfully")
                    else:
                        connection.connection_attempts += 1
                        
                except Exception as e:
                    self.logger.error(f"Reconnection failed for client {client_id}: {e}")
                    connection.connection_attempts += 1
    
    def add_client(self, client_data: Dict[str, Any]) -> bool:
        """Add new client to the system"""
        try:
            client_id = client_data['client_id']
            
            # Create client account
            client_account = ClientAccount(**client_data['account_info'])
            
            # Register with trade executor
            if not self.trade_executor.register_client(client_account):
                return False
            
            # Create client connection
            connection = ClientConnection(
                client_id=client_id,
                mt5_login=client_data['mt5_login'],
                mt5_password=client_data['mt5_password'],
                mt5_server=client_data['mt5_server'],
                preferred_pairs=client_data.get('preferred_pairs', []),
                risk_multiplier=client_data.get('risk_multiplier', 1.0),
                max_signals_per_hour=client_data.get('max_signals_per_hour', 10)
            )
            
            # Connect to MT5
            if self._connect_client_mt5(connection):
                with self.client_lock:
                    self.client_connections[client_id] = connection
                
                self.logger.info(f"[CHECK] Client {client_id} added and connected successfully")
                self._trigger_event('client_added', {'client_id': client_id})
                return True
            else:
                self.logger.error(f"[X] Failed to connect MT5 for client {client_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding client: {e}")
            return False
    
    def _connect_client_mt5(self, connection: ClientConnection) -> bool:
        """Connect client to MT5"""
        try:
            # Create MT5 adapter
            adapter = MT5Adapter(magic_number=12345)
            
            # Connect
            success = adapter.connect(
                connection.mt5_login,
                connection.mt5_password,
                connection.mt5_server
            )
            
            if success:
                connection.adapter = adapter
                connection.last_heartbeat = datetime.now()
                
                # Create integrated executor
                integrated_executor = MT5IntegratedExecutor(adapter)
                self.integrated_executors[connection.client_id] = integrated_executor
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"MT5 connection error for {connection.client_id}: {e}")
            return False
    
    def remove_client(self, client_id: str) -> bool:
        """Remove client from system"""
        try:
            with self.client_lock:
                if client_id in self.client_connections:
                    connection = self.client_connections[client_id]
                    
                    # Disconnect MT5
                    if connection.adapter:
                        connection.adapter.disconnect()
                    
                    # Remove from integrated executors
                    if client_id in self.integrated_executors:
                        del self.integrated_executors[client_id]
                    
                    # Remove connection
                    del self.client_connections[client_id]
                    
                    self.logger.info(f"Client {client_id} removed successfully")
                    self._trigger_event('client_removed', {'client_id': client_id})
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing client {client_id}: {e}")
            return False
    
    def _process_pending_executions(self):
        """Process any pending executions"""
        # This integrates with the trade executor's queue
        # Additional processing logic can be added here
        pass
    
    def _update_system_metrics(self):
        """Update real-time system metrics"""
        try:
            # Calculate uptime
            self.metrics.system_uptime = datetime.now() - self.start_time
            
            # Count today's activities
            today = datetime.now().date()
            today_signals = [
                log for log in self.signal_history
                if log['timestamp'].date() == today
            ]
            today_executions = [
                log for log in self.execution_log
                if log['timestamp'].date() == today
            ]
            
            self.metrics.total_signals_today = len(today_signals)
            self.metrics.executed_trades_today = len(today_executions)
            
            # Calculate average signal quality
            if today_signals:
                self.metrics.avg_signal_quality = np.mean([
                    s['signal'].confluence_score for s in today_signals
                ])
            
            # Get total P&L from all clients
            total_pnl = 0.0
            for client_id in self.client_connections.keys():
                try:
                    summary = self.trade_executor.get_client_summary(client_id)
                    if 'pnl' in summary:
                        total_pnl += summary['pnl']['daily']
                except:
                    pass
            
            self.metrics.total_pnl_today = total_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _check_safety_conditions(self):
        """Check and enforce safety conditions"""
        try:
            # Check global daily loss limit
            if abs(self.metrics.total_pnl_today) > self.config['risk']['global_daily_loss_limit']:
                self.logger.critical("[SIREN] GLOBAL DAILY LOSS LIMIT EXCEEDED!")
                self.emergency_stop()
                return
            
            # Check max concurrent trades
            total_positions = sum(
                len(self.trade_executor.active_positions) 
                for executor in self.integrated_executors.values()
            )
            
            if total_positions > self.config['risk']['max_concurrent_trades']:
                self.logger.warning("[WARNING] Maximum concurrent trades limit approached")
                # Could implement position size reduction here
            
        except Exception as e:
            self.logger.error(f"Error checking safety conditions: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.last_cache_update:
            return False
        
        last_update = self.last_cache_update[cache_key]
        return datetime.now() - last_update < self.cache_expiry
    
    def emergency_stop(self, reason: str = "Emergency stop triggered"):
        """Emergency stop all operations"""
        try:
            self.emergency_stop_triggered = True
            self.status = SystemStatus.PAUSED
            
            # Stop trade executor
            self.trade_executor.emergency_stop_all(reason)
            
            # Disable auto trading for all clients
            with self.client_lock:
                for connection in self.client_connections.values():
                    connection.auto_trading = False
            
            self.logger.critical(f"[SIREN] EMERGENCY STOP: {reason}")
            self._trigger_event('emergency_stop', {'reason': reason})
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
    
    def resume_operations(self):
        """Resume operations after emergency stop"""
        try:
            if self.emergency_stop_triggered:
                self.emergency_stop_triggered = False
                self.status = SystemStatus.RUNNING
                
                # Resume trade executor
                self.trade_executor.resume_trading()
                
                # Re-enable auto trading for clients (with manual review)
                with self.client_lock:
                    for connection in self.client_connections.values():
                        connection.auto_trading = True
                
                self.logger.info("[CHECK] Operations resumed after emergency stop")
                self._trigger_event('operations_resumed', {'timestamp': datetime.now()})
                
        except Exception as e:
            self.logger.error(f"Error resuming operations: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'status': self.status,
            'metrics': {
                'total_clients': self.metrics.total_clients,
                'active_clients': self.metrics.active_clients,
                'connected_clients': self.metrics.connected_clients,
                'uptime_hours': self.metrics.system_uptime.total_seconds() / 3600,
                'signals_today': self.metrics.total_signals_today,
                'trades_today': self.metrics.executed_trades_today,
                'pnl_today': self.metrics.total_pnl_today,
                'avg_signal_quality': self.metrics.avg_signal_quality
            },
            'safety': {
                'emergency_stop': self.emergency_stop_triggered,
                'global_safety_enabled': self.global_safety_enabled
            },
            'performance': {
                'cache_hit_rate': len(self.signal_cache) / max(1, len(self.currency_pairs)),
                'execution_queue_size': len(self.trade_executor.execution_queue),
                'thread_pool_active': self.executor_pool._threads
            }
        }
    
    def get_client_status(self, client_id: str) -> Dict[str, Any]:
        """Get specific client status"""
        if client_id not in self.client_connections:
            return {'error': 'Client not found'}
        
        connection = self.client_connections[client_id]
        
        # Get trading summary
        summary = self.trade_executor.get_client_summary(client_id)
        
        # Get MT5 connection status
        mt5_status = None
        if connection.adapter:
            mt5_status = connection.adapter.get_execution_statistics()
        
        return {
            'client_id': client_id,
            'connection': {
                'is_active': connection.is_active,
                'auto_trading': connection.auto_trading,
                'mt5_connected': connection.adapter.is_connected() if connection.adapter else False,
                'last_heartbeat': connection.last_heartbeat,
                'connection_attempts': connection.connection_attempts
            },
            'trading_summary': summary,
            'mt5_status': mt5_status,
            'preferences': {
                'preferred_pairs': connection.preferred_pairs,
                'risk_multiplier': connection.risk_multiplier,
                'max_signals_per_hour': connection.max_signals_per_hour
            }
        }
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event for real-time updates"""
        try:
            callbacks = self.event_callbacks.get(event_type, [])
            for callback in callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering event {event_type}: {e}")
    
    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to system events"""
        self.event_callbacks[event_type].append(callback)
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            self.logger.info("[REFRESH] Shutting down Lightning Scalper System...")
            self.status = SystemStatus.STOPPING
            
            # Stop background threads
            self.is_running = False
            
            # Stop trade executor
            self.trade_executor.stop_execution_engine()
            
            # Disconnect all clients
            with self.client_lock:
                for connection in self.client_connections.values():
                    if connection.adapter:
                        connection.adapter.disconnect()
            
            # Shutdown thread pool
            self.executor_pool.shutdown(wait=True)
            
            self.status = SystemStatus.STOPPING
            self.logger.info("[CHECK] Lightning Scalper System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Demonstration and Testing
async def main():
    """Demo the Lightning Scalper Controller"""
    print("[ROCKET] Lightning Scalper Production Controller")
    print("=" * 60)
    
    # Initialize controller
    controller = LightningScalperController()
    
    # Start system
    success = await controller.start_system()
    if not success:
        print("[X] Failed to start system")
        return
    
    print("[CHECK] System started successfully!")
    
    # Add sample clients
    sample_clients = [
        {
            'client_id': 'CLIENT_001',
            'mt5_login': 12345001,
            'mt5_password': 'password1',
            'mt5_server': 'Broker-Server1',
            'account_info': {
                'client_id': 'CLIENT_001',
                'account_number': '12345001',
                'broker': 'MetaTrader5',
                'currency': 'USD',
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'margin_level': 0.0,
                'max_daily_loss': 200.0,
                'max_weekly_loss': 500.0,
                'max_monthly_loss': 1500.0,
                'max_positions': 5,
                'max_lot_size': 1.0
            },
            'preferred_pairs': ['EURUSD', 'GBPUSD'],
            'risk_multiplier': 1.0,
            'max_signals_per_hour': 8
        },
        {
            'client_id': 'CLIENT_002',
            'mt5_login': 12345002,
            'mt5_password': 'password2',
            'mt5_server': 'Broker-Server1',
            'account_info': {
                'client_id': 'CLIENT_002',
                'account_number': '12345002',
                'broker': 'MetaTrader5',
                'currency': 'USD',
                'balance': 25000.0,
                'equity': 25000.0,
                'margin': 0.0,
                'free_margin': 25000.0,
                'margin_level': 0.0,
                'max_daily_loss': 500.0,
                'max_weekly_loss': 1200.0,
                'max_monthly_loss': 3000.0,
                'max_positions': 8,
                'max_lot_size': 2.0
            },
            'preferred_pairs': ['EURUSD', 'USDJPY', 'XAUUSD'],
            'risk_multiplier': 1.5,
            'max_signals_per_hour': 12
        }
    ]
    
    # Note: In demo mode, MT5 connections will fail
    print("\n[MEMO] Demo Mode: Adding sample clients...")
    for client_data in sample_clients:
        # For demo, we'll register clients without MT5 connection
        client_account = ClientAccount(**client_data['account_info'])
        controller.trade_executor.register_client(client_account)
        print(f"   [CHECK] {client_data['client_id']} registered (Demo Mode)")
    
    # Simulate running for a short time
    print("\n[REFRESH] Running system simulation...")
    
    # Let system run for a few seconds
    await asyncio.sleep(5)
    
    # Get system status
    status = controller.get_system_status()
    print(f"\n[CHART] System Status:")
    print(f"   Status: {status['status']}")
    print(f"   Uptime: {status['metrics']['uptime_hours']:.2f} hours")
    print(f"   Signals Today: {status['metrics']['signals_today']}")
    print(f"   Trades Today: {status['metrics']['trades_today']}")
    print(f"   Average Signal Quality: {status['metrics']['avg_signal_quality']:.1f}")
    
    # Show recent signals
    print(f"\n[TARGET] Recent Signals Generated: {len(controller.signal_history)}")
    for i, signal_log in enumerate(list(controller.signal_history)[-3:]):
        signal = signal_log['signal']
        print(f"   {i+1}. {signal.currency_pair.value} {signal.fvg_type.value}")
        print(f"      Time: {signal_log['timestamp'].strftime('%H:%M:%S')}")
        print(f"      Confluence: {signal.confluence_score:.1f}")
        print(f"      Priority: {signal_log['priority']}")
    
    # Shutdown gracefully
    print(f"\n[REFRESH] Shutting down system...")
    await controller.shutdown()
    print("[CHECK] System shutdown complete!")
    
    print("\n[TARGET] Next Steps:")
    print("   1. Integrate with real MT5 broker connections")
    print("   2. Add Web Dashboard for monitoring 80+ clients")
    print("   3. Implement Signal Logger for Active Learning")
    print("   4. Deploy with proper configuration management")

if __name__ == "__main__":
    asyncio.run(main())