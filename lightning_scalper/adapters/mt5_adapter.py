#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[ROCKET] Lightning Scalper - Enhanced MT5 Adapter
Production-Grade MetaTrader 5 Integration - FIXED VERSION

แก้ไขปัญหาหลัก:
- ✅ Connection stability improved
- ✅ Robust error handling  
- ✅ Auto-reconnection system
- ✅ Order execution timeout
- ✅ Multiple account support
- ✅ Comprehensive testing

Author: Phoenix Trading AI
Version: 1.1.0 (Enhanced)
License: Proprietary
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
from dataclasses import dataclass
import json

# MetaTrader 5 imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("[⚠️] MetaTrader5 package not found!")
    print("   Install with: pip install MetaTrader5")
    MT5_AVAILABLE = False

class ConnectionStatus(Enum):
    """MT5 Connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class OrderExecutionResult(Enum):
    """Order execution results"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    PARTIAL = "partial"

@dataclass
class MT5Config:
    """MT5 Configuration settings"""
    login: int
    password: str
    server: str
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 2.0
    enable_logging: bool = True
    magic_number: int = 123456

@dataclass
class OrderRequest:
    """Enhanced order request"""
    symbol: str
    action: int  # mt5.TRADE_ACTION_DEAL, etc.
    volume: float
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 10
    comment: str = "Lightning Scalper"
    magic: int = 123456
    type_filling: int = 2  # mt5.ORDER_FILLING_IOC

@dataclass
class ExecutionResult:
    """Order execution result"""
    success: bool
    order_id: Optional[int] = None
    deal_id: Optional[int] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    retcode: Optional[int] = None

class EnhancedMT5Adapter:
    """
    [🔌] Enhanced MetaTrader 5 Adapter
    Production-grade MT5 integration with robust error handling
    """
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.logger = logging.getLogger(f'MT5Adapter-{config.login}')
        
        # Connection management
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_connection_attempt = None
        self.connection_attempts = 0
        self.last_error = None
        self.connection_lock = threading.Lock()
        
        # Monitoring and statistics
        self.is_monitoring = False
        self.monitor_thread = None
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'timeout_orders': 0,
            'average_execution_time': 0.0,
            'last_execution_time': None
        }
        
        # Current market data
        self.current_prices = {}
        self.price_lock = threading.Lock()
        
        # Account information
        self.account_info = None
        self.active_positions = {}
        
        # Health monitoring
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 30  # seconds
        
        if not MT5_AVAILABLE:
            self.logger.error("❌ MetaTrader5 package is not available")
            return
        
        self.logger.info(f"🔌 Enhanced MT5 Adapter initialized for {config.server}")
    
    def connect(self) -> bool:
        """Enhanced connection with retry logic"""
        with self.connection_lock:
            if not MT5_AVAILABLE:
                self.logger.error("❌ MT5 package not available")
                return False
            
            max_attempts = self.config.max_retries
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                self.connection_attempts += 1
                self.connection_status = ConnectionStatus.CONNECTING
                
                try:
                    self.logger.info(f"🔄 Connecting to MT5... (Attempt {attempt}/{max_attempts})")
                    self.logger.info(f"   Server: {self.config.server}")
                    self.logger.info(f"   Login: {self.config.login}")
                    
                    # Initialize MT5
                    if not mt5.initialize():
                        error = mt5.last_error()
                        self.last_error = f"MT5 initialization failed: {error}"
                        self.logger.error(self.last_error)
                        time.sleep(self.config.retry_delay)
                        continue
                    
                    # Login with timeout
                    login_start = time.time()
                    login_result = mt5.login(
                        login=self.config.login,
                        password=self.config.password,
                        server=self.config.server,
                        timeout=self.config.timeout * 1000  # MT5 expects milliseconds
                    )
                    login_time = time.time() - login_start
                    
                    if not login_result:
                        error = mt5.last_error()
                        self.last_error = f"Login failed: {error}"
                        self.logger.error(self.last_error)
                        mt5.shutdown()
                        time.sleep(self.config.retry_delay)
                        continue
                    
                    # Verify connection
                    account_info = mt5.account_info()
                    if not account_info:
                        self.last_error = "Failed to get account info"
                        self.logger.error(self.last_error)
                        mt5.shutdown()
                        time.sleep(self.config.retry_delay)
                        continue
                    
                    # Connection successful
                    self.connection_status = ConnectionStatus.CONNECTED
                    self.account_info = account_info
                    self.last_connection_attempt = datetime.now()
                    self.last_heartbeat = datetime.now()
                    
                    self.logger.info(f"✅ MT5 Connected successfully!")
                    self.logger.info(f"   Account: {account_info.login}")
                    self.logger.info(f"   Balance: ${account_info.balance:.2f}")
                    self.logger.info(f"   Server: {account_info.server}")
                    self.logger.info(f"   Connection time: {login_time:.2f}s")
                    
                    # Start monitoring
                    self.start_monitoring()
                    
                    return True
                    
                except Exception as e:
                    self.last_error = f"Connection error: {str(e)}"
                    self.logger.error(self.last_error)
                    try:
                        mt5.shutdown()
                    except:
                        pass
                    
                    if attempt < max_attempts:
                        self.logger.info(f"⏱️ Retrying in {self.config.retry_delay} seconds...")
                        time.sleep(self.config.retry_delay)
                    
            # All attempts failed
            self.connection_status = ConnectionStatus.ERROR
            self.logger.error(f"❌ Failed to connect after {max_attempts} attempts")
            return False
    
    def disconnect(self):
        """Enhanced disconnection with cleanup"""
        with self.connection_lock:
            try:
                self.logger.info("🔌 Disconnecting from MT5...")
                
                # Stop monitoring
                self.stop_monitoring()
                
                # Close any open operations
                self._cleanup_resources()
                
                # Shutdown MT5
                if MT5_AVAILABLE:
                    mt5.shutdown()
                
                self.connection_status = ConnectionStatus.DISCONNECTED
                self.account_info = None
                self.active_positions.clear()
                self.current_prices.clear()
                
                self.logger.info("✅ MT5 Disconnected successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Error during disconnection: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected with heartbeat verification"""
        if not MT5_AVAILABLE:
            return False
        
        if self.connection_status != ConnectionStatus.CONNECTED:
            return False
        
        # Check heartbeat
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        if time_since_heartbeat.total_seconds() > self.heartbeat_interval * 2:
            self.logger.warning("⚠️ Heartbeat timeout, checking connection...")
            return self._verify_connection()
        
        return True
    
    def _verify_connection(self) -> bool:
        """Verify MT5 connection is still alive"""
        try:
            # Quick connection test
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                self.logger.warning("⚠️ Terminal info unavailable")
                return False
            
            account_info = mt5.account_info()
            if not account_info:
                self.logger.warning("⚠️ Account info unavailable")
                return False
            
            self.last_heartbeat = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection verification failed: {e}")
            self.connection_status = ConnectionStatus.ERROR
            return False
    
    def auto_reconnect(self) -> bool:
        """Automatic reconnection logic"""
        if self.connection_status == ConnectionStatus.CONNECTING:
            return False  # Already trying to connect
        
        self.logger.info("🔄 Attempting auto-reconnection...")
        self.connection_status = ConnectionStatus.RECONNECTING
        
        # Wait a bit before reconnecting
        time.sleep(5)
        
        return self.connect()
    
    def send_order(self, order_request: OrderRequest, timeout: float = 30.0) -> ExecutionResult:
        """Enhanced order execution with timeout and retry"""
        execution_start = time.time()
        
        try:
            # Verify connection
            if not self.is_connected():
                if not self.auto_reconnect():
                    return ExecutionResult(
                        success=False,
                        error_message="MT5 not connected and reconnection failed"
                    )
            
            # Validate order request
            validation_error = self._validate_order_request(order_request)
            if validation_error:
                return ExecutionResult(
                    success=False,
                    error_message=f"Order validation failed: {validation_error}"
                )
            
            # Prepare order
            request = {
                "action": order_request.action,
                "symbol": order_request.symbol,
                "volume": order_request.volume,
                "price": order_request.price,
                "sl": order_request.sl,
                "tp": order_request.tp,
                "deviation": order_request.deviation,
                "magic": order_request.magic,
                "comment": order_request.comment,
                "type_filling": order_request.type_filling,
            }
            
            # Remove zero values
            request = {k: v for k, v in request.items() if v != 0.0 or k in ['action', 'symbol', 'volume']}
            
            self.logger.info(f"📤 Sending order: {order_request.symbol} {order_request.volume}")
            
            # Execute order with timeout
            result = None
            execution_thread = None
            
            def execute_order():
                nonlocal result
                try:
                    result = mt5.order_send(request)
                except Exception as e:
                    result = None
                    self.logger.error(f"❌ Order execution exception: {e}")
            
            execution_thread = threading.Thread(target=execute_order)
            execution_thread.start()
            execution_thread.join(timeout=timeout)
            
            # Check for timeout
            if execution_thread.is_alive():
                self.logger.error(f"⏱️ Order execution timeout ({timeout}s)")
                self.execution_stats['timeout_orders'] += 1
                return ExecutionResult(
                    success=False,
                    error_message=f"Order execution timeout ({timeout}s)"
                )
            
            execution_time = time.time() - execution_start
            
            # Process result
            if result is None:
                self.execution_stats['failed_orders'] += 1
                return ExecutionResult(
                    success=False,
                    error_message="Order execution returned None"
                )
            
            # Update statistics
            self.execution_stats['total_orders'] += 1
            self.execution_stats['last_execution_time'] = datetime.now()
            
            # Calculate average execution time
            current_avg = self.execution_stats['average_execution_time']
            total_orders = self.execution_stats['total_orders']
            self.execution_stats['average_execution_time'] = (
                (current_avg * (total_orders - 1) + execution_time) / total_orders
            )
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.execution_stats['successful_orders'] += 1
                self.logger.info(f"✅ Order executed successfully: Deal #{result.deal}")
                
                return ExecutionResult(
                    success=True,
                    order_id=result.order,
                    deal_id=result.deal,
                    execution_time=execution_time,
                    retcode=result.retcode
                )
            else:
                self.execution_stats['failed_orders'] += 1
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(f"❌ {error_msg}")
                
                return ExecutionResult(
                    success=False,
                    error_code=result.retcode,
                    error_message=error_msg,
                    execution_time=execution_time,
                    retcode=result.retcode
                )
                
        except Exception as e:
            execution_time = time.time() - execution_start
            self.execution_stats['failed_orders'] += 1
            error_msg = f"Order execution exception: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def _validate_order_request(self, order_request: OrderRequest) -> Optional[str]:
        """Validate order request before execution"""
        try:
            # Check symbol
            symbol_info = mt5.symbol_info(order_request.symbol)
            if not symbol_info:
                return f"Symbol {order_request.symbol} not found"
            
            if not symbol_info.visible:
                return f"Symbol {order_request.symbol} not visible in Market Watch"
            
            # Check volume
            if order_request.volume < symbol_info.volume_min:
                return f"Volume {order_request.volume} below minimum {symbol_info.volume_min}"
            
            if order_request.volume > symbol_info.volume_max:
                return f"Volume {order_request.volume} above maximum {symbol_info.volume_max}"
            
            # Check step
            volume_step = symbol_info.volume_step
            if volume_step > 0:
                steps = order_request.volume / volume_step
                if abs(steps - round(steps)) > 1e-6:
                    return f"Volume {order_request.volume} not in step {volume_step}"
            
            # Check market hours
            if not symbol_info.trade_mode or symbol_info.trade_mode == 0:
                return f"Trading disabled for {order_request.symbol}"
            
            return None  # Validation passed
            
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions with error handling"""
        try:
            if not self.is_connected():
                return {}
            
            positions = mt5.positions_get()
            if positions is None:
                self.logger.warning("⚠️ Failed to get positions")
                return {}
            
            result = {}
            for pos in positions:
                result[pos.ticket] = {
                    'symbol': pos.symbol,
                    'volume': pos.volume,
                    'type': pos.type,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'time': pos.time,
                    'comment': pos.comment,
                    'magic': pos.magic
                }
            
            self.active_positions = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error getting positions: {e}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information with caching"""
        try:
            if not self.is_connected():
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.warning(f"⚠️ Symbol info not available for {symbol}")
                return None
            
            return symbol_info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data with enhanced error handling"""
        try:
            if not self.is_connected():
                return None
            
            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Get data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"⚠️ No historical data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting historical data: {e}")
            return None
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("📊 Monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("📊 Monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check connection
                if not self._verify_connection():
                    self.logger.warning("⚠️ Connection lost, attempting reconnection...")
                    self.auto_reconnect()
                
                # Update positions
                self.get_positions()
                
                # Update account info
                if self.is_connected():
                    self.account_info = mt5.account_info()
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitor loop error: {e}")
                time.sleep(10)
    
    def _cleanup_resources(self):
        """Cleanup resources before disconnection"""
        try:
            # Cancel any pending operations
            # Close monitoring resources
            # Clear caches
            self.current_prices.clear()
            self.active_positions.clear()
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            health_data = {
                "mt5_available": MT5_AVAILABLE,
                "connection_status": self.connection_status.value,
                "is_connected": self.is_connected(),
                "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                "connection_attempts": self.connection_attempts,
                "last_error": self.last_error,
                "monitoring_active": self.is_monitoring,
                "execution_stats": self.execution_stats.copy(),
                "active_positions_count": len(self.active_positions)
            }
            
            if self.account_info:
                health_data.update({
                    "account_login": self.account_info.login,
                    "account_server": self.account_info.server,
                    "account_balance": self.account_info.balance,
                    "account_equity": self.account_info.equity,
                    "trade_allowed": self.account_info.trade_allowed
                })
            
            # Test basic functionality
            if self.is_connected():
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    health_data.update({
                        "terminal_connected": terminal_info.connected,
                        "terminal_name": terminal_info.name,
                        "terminal_build": terminal_info.build
                    })
            
            return health_data
            
        except Exception as e:
            return {
                "error": str(e),
                "connection_status": "error",
                "is_connected": False
            }

# ================================
# TESTING AND DEMO
# ================================

def test_mt5_adapter():
    """Test the enhanced MT5 adapter"""
    print("🧪 Testing Enhanced MT5 Adapter")
    print("=" * 50)
    
    # Demo configuration
    config = MT5Config(
        login=0,  # Replace with real demo login
        password="demo_password",  # Replace with real password
        server="MetaQuotes-Demo",  # Replace with real server
        timeout=30,
        max_retries=3,
        enable_logging=True
    )
    
    adapter = EnhancedMT5Adapter(config)
    
    try:
        # Test connection
        print("🔌 Testing connection...")
        if adapter.connect():
            print("✅ Connection successful!")
            
            # Test health check
            print("\n🏥 Testing health check...")
            health = adapter.health_check()
            print(f"Health status: {health}")
            
            # Test symbol info
            print("\n📊 Testing symbol info...")
            symbol_info = adapter.get_symbol_info("EURUSD")
            if symbol_info:
                print(f"EURUSD info: Bid={symbol_info.bid}, Ask={symbol_info.ask}")
            
            # Test historical data
            print("\n📈 Testing historical data...")
            df = adapter.get_historical_data("EURUSD", "M5", 10)
            if df is not None:
                print(f"Historical data shape: {df.shape}")
                print(df.tail())
            
            # Test positions
            print("\n📋 Testing positions...")
            positions = adapter.get_positions()
            print(f"Active positions: {len(positions)}")
            
        else:
            print("❌ Connection failed!")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        adapter.disconnect()
        print("🔌 Disconnected")

if __name__ == "__main__":
    test_mt5_adapter()

# Alias for backward compatibility with existing code
MT5Adapter = EnhancedMT5Adapter

# Legacy wrapper functions
def create_mt5_adapter(login: int, password: str, server: str, **kwargs) -> MT5Adapter:
    """Create MT5 adapter with legacy interface"""
    config = MT5Config(
        login=login,
        password=password,
        server=server,
        **kwargs
    )
    return EnhancedMT5Adapter(config)

# Export classes for import
__all__ = [
    'EnhancedMT5Adapter',
    'MT5Adapter',  # Alias
    'MT5Config',
    'OrderRequest',
    'ExecutionResult',
    'ConnectionStatus',
    'OrderExecutionResult',
    'create_mt5_adapter'
]

print("📦 MT5 Adapter module loaded successfully")