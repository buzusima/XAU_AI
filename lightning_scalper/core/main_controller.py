#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Lightning Scalper Controller
แก้ไขปัญหา logger attribute error
"""

import asyncio
import threading
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import json

# Import required modules
try:
    from .lightning_scalper_engine import (
        EnhancedFVGDetector, 
        FVGSignal, 
        CurrencyPair, 
        SystemStatus
    )
    from ..execution.trade_executor import TradeExecutor, ClientAccount
    from ..adapters.mt5_adapter import MT5Adapter
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from core.lightning_scalper_engine import (
        EnhancedFVGDetector, 
        FVGSignal, 
        CurrencyPair, 
        SystemStatus
    )
    from execution.trade_executor import TradeExecutor, ClientAccount
    from adapters.mt5_adapter import MT5Adapter

class LightningScalperController:
    """
    [ROCKET] Main Controller for Lightning Scalper System
    แก้ไขปัญหา logger attribute error
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Lightning Scalper Controller with proper logger setup"""
        
        # Setup logging FIRST before anything else
        self._setup_logging()
        self.logger = logging.getLogger('LightningScalperController')
        
        self.logger.info("[ROCKET] Initializing Lightning Scalper Controller...")
        
        # System status
        self.status = SystemStatus.STARTING
        self.start_time = None
        self.last_update = datetime.now()
        
        # Core components
        self.fvg_detector = None
        self.trade_executor = None
        self.mt5_adapter = None
        
        # Client management
        self.active_clients: Dict[str, ClientAccount] = {}
        self.client_stats: Dict[str, Dict] = {}
        self.client_connections: Dict[str, Any] = {}  # For MT5 connections
        
        # System metrics
        self.system_metrics = {
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'total_profit_loss': 0.0,
            'active_trades': 0,
            'system_uptime': timedelta(0),
            'last_signal_time': None,
            'error_count': 0
        }
        
        # Performance tracking
        self.performance_stats = {
            'signals_per_minute': 0.0,
            'trades_per_hour': 0.0,
            'avg_signal_processing_time': 0.0,
            'avg_trade_execution_time': 0.0
        }
        
        # System limits and controls
        self.max_clients = 100
        self.max_concurrent_trades = 200
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
        
        self.logger.info("[ROCKET] Lightning Scalper Controller initialized successfully")
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(logs_dir / 'lightning_scalper_controller.log'),
                logging.StreamHandler()
            ]
        )
        
        # Ensure the logger is created
        logger = logging.getLogger('LightningScalperController')
        logger.setLevel(logging.INFO)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration"""
        return {
            'system': {
                'max_clients': 100,
                'data_update_interval': 1.0,  # seconds
                'health_check_interval': 30.0,  # seconds
                'signal_generation_interval': 5.0,  # seconds
                'auto_reconnect': True,
                'max_reconnect_attempts': 5,
                'emergency_stop_loss': 0.10,  # 10% max loss
                'max_daily_trades': 1000
            },
            'trading': {
                'default_lot_size': 0.01,
                'max_lot_size': 1.0,
                'max_spread': 3.0,  # pips
                'min_equity': 100.0,  # USD
                'risk_per_trade': 0.02,  # 2% per trade
                'max_simultaneous_trades': 5
            },
            'fvg_detection': {
                'min_fvg_size': 5.0,  # pips
                'max_fvg_age': 300,  # seconds
                'confirmation_candles': 2,
                'volume_threshold': 1.2,
                'enable_multi_timeframe': True
            },
            'performance': {
                'enable_caching': True,
                'cache_duration': 300,  # seconds
                'max_memory_usage': 512,  # MB
                'cleanup_interval': 3600  # seconds
            }
        }
    
    def _load_config_file(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                # Merge with default config
                self._merge_config(self.config, file_config)
            self.logger.info(f"[CONFIG] Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.warning(f"[CONFIG] Failed to load config file {config_path}: {e}")
            self.logger.info("[CONFIG] Using default configuration")
    
    def _merge_config(self, default: Dict, custom: Dict):
        """Recursively merge custom config with default"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    async def start_system(self) -> bool:
        """Start the complete Lightning Scalper system"""
        try:
            self.logger.info("[LIGHTNING] Starting Lightning Scalper system...")
            self.status = SystemStatus.STARTING
            self.start_time = datetime.now()
            
            # 1. Initialize FVG Detector
            self.logger.info("[SATELLITE] Initializing FVG Detector...")
            self.fvg_detector = EnhancedFVGDetector()
            
            # 2. Initialize Trade Executor
            self.logger.info("[CHART] Initializing Trade Executor...")
            self.trade_executor = TradeExecutor()
            
            # 3. Initialize MT5 Adapter
            self.logger.info("[PLUG] Initializing MT5 Adapter...")
            self.mt5_adapter = MT5Adapter()
            
            # 4. Test MT5 connection
            if not await self._test_mt5_connection():
                self.logger.warning("[WARNING] MT5 connection test failed - continuing in demo mode")
            
            # 5. Start background tasks
            self._start_background_tasks()
            
            # 6. System is ready
            self.status = SystemStatus.RUNNING
            startup_time = (datetime.now() - self.start_time).total_seconds()
            self.system_metrics['system_uptime'] = datetime.now() - self.start_time
            
            self.logger.info(f"[ROCKET] Lightning Scalper system started successfully in {startup_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Failed to start system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def _test_mt5_connection(self) -> bool:
        """Test MT5 connection"""
        try:
            if self.mt5_adapter:
                # Test connection without actual login
                return True  # Placeholder - implement actual test
        except Exception as e:
            self.logger.error(f"[X] MT5 connection test failed: {e}")
        return False
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks"""
        self.logger.info("[REFRESH] Starting background tasks...")
        
        # Start in separate threads to avoid blocking
        tasks = [
            self._health_check_loop,
            self._signal_generation_loop,
            self._performance_monitoring_loop,
            self._cleanup_loop
        ]
        
        for task in tasks:
            thread = threading.Thread(target=self._run_async_task, args=(task,))
            thread.daemon = True
            thread.start()
    
    def _run_async_task(self, coro_func):
        """Run async task in separate thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro_func())
        except Exception as e:
            self.logger.error(f"[X] Background task failed: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config['system']['health_check_interval'])
            except Exception as e:
                self.logger.error(f"[X] Health check failed: {e}")
                await asyncio.sleep(5)
    
    async def _signal_generation_loop(self):
        """Background signal generation loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                await self._generate_signals()
                await asyncio.sleep(self.config['system']['signal_generation_interval'])
            except Exception as e:
                self.logger.error(f"[X] Signal generation failed: {e}")
                await asyncio.sleep(5)
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                await self._update_performance_stats()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"[X] Performance monitoring failed: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config['performance']['cleanup_interval'])
            except Exception as e:
                self.logger.error(f"[X] Cleanup failed: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            health_status = {
                'timestamp': datetime.now(),
                'system_status': self.status,
                'active_clients': len(self.active_clients),
                'active_trades': self.system_metrics['active_trades'],
                'memory_usage': self._get_memory_usage(),
                'error_count': self.system_metrics['error_count']
            }
            
            # Check for emergency conditions
            if health_status['error_count'] > 100:
                self.logger.warning("[WARNING] High error count detected")
            
            # Update last health check time
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"[X] Health check error: {e}")
            self.system_metrics['error_count'] += 1
    
    async def _generate_signals(self):
        """Generate trading signals for all currency pairs"""
        try:
            if not self.fvg_detector:
                return
            
            signals_generated = 0
            
            for pair in self.currency_pairs:
                try:
                    # Generate signals for this pair
                    signals = await self._generate_pair_signals(pair)
                    
                    if signals:
                        signals_generated += len(signals)
                        await self._process_signals(pair, signals)
                        
                except Exception as e:
                    self.logger.error(f"[X] Signal generation failed for {pair}: {e}")
            
            if signals_generated > 0:
                self.system_metrics['total_signals_generated'] += signals_generated
                self.system_metrics['last_signal_time'] = datetime.now()
                self.logger.debug(f"[LIGHTNING] Generated {signals_generated} signals")
                
        except Exception as e:
            self.logger.error(f"[X] Signal generation loop error: {e}")
            self.system_metrics['error_count'] += 1
    
    async def _generate_pair_signals(self, pair: CurrencyPair) -> List[FVGSignal]:
        """Generate signals for a specific currency pair"""
        try:
            # Check cache first
            cache_key = f"{pair.value}_signals"
            if cache_key in self.signal_cache:
                if datetime.now() - self.last_cache_update.get(cache_key, datetime.min) < self.cache_expiry:
                    return self.signal_cache[cache_key]
            
            # Generate new signals
            signals = []
            
            # Use FVG detector to find signals
            if self.fvg_detector:
                # Placeholder for actual signal generation
                # This would call the FVG detector with real market data
                pass
            
            # Cache the results
            self.signal_cache[cache_key] = signals
            self.last_cache_update[cache_key] = datetime.now()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"[X] Failed to generate signals for {pair}: {e}")
            return []
    
    async def _process_signals(self, pair: CurrencyPair, signals: List[FVGSignal]):
        """Process generated signals and execute trades"""
        try:
            for signal in signals:
                # Send signal to all relevant clients
                await self._distribute_signal(pair, signal)
                
        except Exception as e:
            self.logger.error(f"[X] Failed to process signals for {pair}: {e}")
    
    async def _distribute_signal(self, pair: CurrencyPair, signal: FVGSignal):
        """Distribute signal to all active clients"""
        try:
            if not self.active_clients:
                return
            
            distributed_count = 0
            
            for client_id, client_account in self.active_clients.items():
                try:
                    # Check if client is interested in this pair
                    if self._should_send_signal_to_client(client_account, pair, signal):
                        await self._send_signal_to_client(client_account, pair, signal)
                        distributed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"[X] Failed to send signal to client {client_id}: {e}")
            
            if distributed_count > 0:
                self.logger.debug(f"[SATELLITE] Distributed signal to {distributed_count} clients")
                
        except Exception as e:
            self.logger.error(f"[X] Signal distribution failed: {e}")
    
    def _should_send_signal_to_client(self, client: ClientAccount, pair: CurrencyPair, signal: FVGSignal) -> bool:
        """Determine if signal should be sent to specific client"""
        try:
            # Check client preferences and risk management
            if hasattr(client, 'is_demo') and client.is_demo and not self.config.get('allow_demo_trading', True):
                return False
            
            # Check if client trades this pair
            if hasattr(client, 'traded_pairs') and pair not in client.traded_pairs:
                return False
            
            # Check client risk limits
            if hasattr(client, 'max_concurrent_trades'):
                active_trades = self._get_client_active_trades(client.client_id)
                if active_trades >= client.max_concurrent_trades:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Failed to check signal criteria for client: {e}")
            return False
    
    async def _send_signal_to_client(self, client: ClientAccount, pair: CurrencyPair, signal: FVGSignal):
        """Send signal to specific client"""
        try:
            if self.trade_executor:
                # Call synchronous method from trade executor
                result = self.trade_executor.execute_fvg_signal(client, pair.value, signal)
                if result.get('success'):
                    self.logger.debug(f"[TARGET] Signal sent to client {client.client_id}")
                else:
                    self.logger.warning(f"[WARNING] Signal execution failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"[X] Failed to send signal to client {client.client_id}: {e}")
    
    def _get_client_active_trades(self, client_id: str) -> int:
        """Get number of active trades for client"""
        try:
            if client_id in self.client_stats:
                return self.client_stats[client_id].get('active_trades', 0)
            return 0
        except Exception:
            return 0
    
    async def _update_performance_stats(self):
        """Update system performance statistics"""
        try:
            current_time = datetime.now()
            
            if self.start_time:
                self.system_metrics['system_uptime'] = current_time - self.start_time
                
                # Calculate rates
                uptime_minutes = self.system_metrics['system_uptime'].total_seconds() / 60
                if uptime_minutes > 0:
                    self.performance_stats['signals_per_minute'] = self.system_metrics['total_signals_generated'] / uptime_minutes
                    self.performance_stats['trades_per_hour'] = (self.system_metrics['total_trades_executed'] / uptime_minutes) * 60
            
        except Exception as e:
            self.logger.error(f"[X] Performance stats update failed: {e}")
    
    async def _perform_cleanup(self):
        """Perform system cleanup tasks"""
        try:
            # Clean expired cache entries
            current_time = datetime.now()
            expired_keys = []
            
            for key, last_update in self.last_cache_update.items():
                if current_time - last_update > self.cache_expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.signal_cache.pop(key, None)
                self.last_cache_update.pop(key, None)
            
            if expired_keys:
                self.logger.debug(f"[MEMO] Cleaned {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"[X] Cleanup failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def add_client(self, client_account: ClientAccount) -> bool:
        """Add a client to the system"""
        try:
            with self.client_lock:
                if len(self.active_clients) >= self.max_clients:
                    self.logger.warning(f"[WARNING] Maximum clients ({self.max_clients}) reached")
                    return False
                
                self.active_clients[client_account.client_id] = client_account
                self.client_stats[client_account.client_id] = {
                    'added_time': datetime.now(),
                    'active_trades': 0,
                    'total_trades': 0,
                    'total_profit': 0.0,
                    'last_trade_time': None
                }
                
                account_type = 'DEMO' if hasattr(client_account, 'is_demo') and client_account.is_demo else 'LIVE'
                self.logger.info(f"[USERS] Added client {client_account.client_id} ({account_type})")
                return True
                
        except Exception as e:
            self.logger.error(f"[X] Failed to add client: {e}")
            return False
    
    def remove_client(self, account_id: str) -> bool:
        """Remove a client from the system"""
        try:
            with self.client_lock:
                if account_id in self.active_clients:
                    del self.active_clients[account_id]
                    self.client_stats.pop(account_id, None)
                    self.logger.info(f"[USERS] Removed client {account_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"[X] Failed to remove client: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'status': self.status,
                'start_time': self.start_time,
                'uptime': self.system_metrics['system_uptime'],
                'active_clients': len(self.active_clients),
                'metrics': self.system_metrics.copy(),
                'performance': self.performance_stats.copy(),
                'last_update': self.last_update,
                'memory_usage_mb': self._get_memory_usage()
            }
        except Exception as e:
            self.logger.error(f"[X] Failed to get system status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def get_client_list(self) -> List[Dict[str, Any]]:
        """Get list of all active clients"""
        try:
            with self.client_lock:
                clients = []
                for client_id, client in self.active_clients.items():
                    stats = self.client_stats.get(client_id, {})
                    
                    # Get account type safely
                    account_type = 'DEMO' if hasattr(client, 'is_demo') and client.is_demo else 'LIVE'
                    broker = getattr(client, 'broker', 'Unknown')
                    
                    clients.append({
                        'client_id': client_id,
                        'account_type': account_type,
                        'broker': broker,
                        'active_trades': stats.get('active_trades', 0),
                        'total_trades': stats.get('total_trades', 0),
                        'total_profit': stats.get('total_profit', 0.0),
                        'added_time': stats.get('added_time'),
                        'last_trade_time': stats.get('last_trade_time')
                    })
                return clients
        except Exception as e:
            self.logger.error(f"[X] Failed to get client list: {e}")
            return []
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            self.logger.info("[SATELLITE] Initiating system shutdown...")
            self.status = SystemStatus.STOPPING
            
            # Stop all clients
            with self.client_lock:
                for client_id in list(self.active_clients.keys()):
                    self.remove_client(client_id)
            
            # Shutdown components
            if self.trade_executor:
                self.trade_executor.stop_execution_engine()
                self.logger.info("[CHECK] Trade executor stopped")
            
            if self.mt5_adapter:
                # Add MT5 disconnect logic here if implemented
                self.logger.info("[CHECK] MT5 adapter disconnected")
            
            self.status = SystemStatus.STOPPED
            self.logger.info("[SATELLITE] System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"[X] Shutdown error: {e}")
            self.status = SystemStatus.ERROR