#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
import json
from collections import deque

# Import from our core modules
from core.lightning_scalper_engine import CurrencyPair, FVGSignal
from execution.trade_executor import Order, Position, TradeDirection, OrderType, OrderStatus

class MT5ConnectionStatus(Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    ERROR = "ERROR"

class MT5OrderType(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class MT5TimeFrame(Enum):
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1

@dataclass
class MT5AccountInfo:
    """MT5 Account Information"""
    login: int
    server: str
    name: str
    company: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    leverage: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int

@dataclass
class MT5SymbolInfo:
    """MT5 Symbol Information"""
    name: str
    bid: float
    ask: float
    spread: float
    digits: int
    trade_mode: int
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    margin_initial: float
    swap_long: float
    swap_short: float
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int

@dataclass
class MT5Position:
    """MT5 Position Information"""
    ticket: int
    time: datetime
    time_msc: int
    time_update: datetime
    time_update_msc: int
    type: int
    magic: int
    identifier: int
    reason: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    swap: float
    profit: float
    symbol: str
    comment: str
    external_id: str

class MT5Adapter:
    """
    Production-Grade MetaTrader 5 Adapter
    Real-time bridge between Lightning Scalper and MT5 platform
    """
    
    def __init__(self, magic_number: int = 12345, max_retries: int = 3):
        self.magic_number = magic_number  # Unique identifier for our trades
        self.max_retries = max_retries
        self.connection_status = MT5ConnectionStatus.DISCONNECTED
        
        # Connection parameters
        self.login = None
        self.password = None
        self.server = None
        
        # Real-time data
        self.current_prices: Dict[str, Dict] = {}
        self.account_info: Optional[MT5AccountInfo] = None
        self.symbol_info: Dict[str, MT5SymbolInfo] = {}
        
        # Position tracking
        self.active_positions: Dict[int, MT5Position] = {}  # ticket -> position
        self.position_mapping: Dict[str, int] = {}  # our_position_id -> mt5_ticket
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_execution_time': 0.0,
            'avg_slippage': 0.0,
            'last_error': None
        }
        
        # Real-time monitoring
        self.price_thread = None
        self.position_thread = None
        self.is_monitoring = False
        
        # Threading locks
        self.lock = threading.Lock()
        self.price_lock = threading.Lock()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MT5Adapter')
        
        # Error handling
        self.last_error = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
    
    def connect(self, login: int, password: str, server: str) -> bool:
        """
        Connect to MetaTrader 5
        """
        try:
            self.connection_status = MT5ConnectionStatus.CONNECTING
            self.login = login
            self.password = password
            self.server = server
            
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error}")
                self.connection_status = MT5ConnectionStatus.ERROR
                return False
            
            # Login to account
            if not mt5.login(login, password, server):
                error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error}")
                self.connection_status = MT5ConnectionStatus.ERROR
                mt5.shutdown()
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                self.connection_status = MT5ConnectionStatus.ERROR
                return False
            
            # Store account information
            self.account_info = MT5AccountInfo(
                login=account_info.login,
                server=account_info.server,
                name=account_info.name,
                company=account_info.company,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                profit=account_info.profit,
                leverage=account_info.leverage,
                margin_so_mode=account_info.margin_so_mode,
                trade_allowed=account_info.trade_allowed,
                trade_expert=account_info.trade_expert,
                margin_mode=account_info.margin_mode,
                currency_digits=account_info.currency_digits
            )
            
            self.connection_status = MT5ConnectionStatus.CONNECTED
            self.connection_attempts = 0
            
            # Start real-time monitoring
            self.start_monitoring()
            
            self.logger.info(f"Successfully connected to MT5 - Account: {login}, Server: {server}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.connection_status = MT5ConnectionStatus.ERROR
            self.last_error = str(e)
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        try:
            self.stop_monitoring()
            
            if mt5.initialize():
                mt5.shutdown()
            
            self.connection_status = MT5ConnectionStatus.DISCONNECTED
            self.logger.info("Disconnected from MT5")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to MT5"""
        return self.connection_status == MT5ConnectionStatus.CONNECTED
    
    def start_monitoring(self):
        """Start real-time price and position monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            
            # Start price monitoring thread
            self.price_thread = threading.Thread(target=self._price_monitoring_loop, daemon=True)
            self.price_thread.start()
            
            # Start position monitoring thread
            self.position_thread = threading.Thread(target=self._position_monitoring_loop, daemon=True)
            self.position_thread.start()
            
            self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        
        if self.price_thread:
            self.price_thread.join(timeout=2)
        
        if self.position_thread:
            self.position_thread.join(timeout=2)
        
        self.logger.info("Real-time monitoring stopped")
    
    def _price_monitoring_loop(self):
        """Real-time price monitoring loop"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD"]
        
        while self.is_monitoring:
            try:
                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        with self.price_lock:
                            self.current_prices[symbol] = {
                                'bid': tick.bid,
                                'ask': tick.ask,
                                'spread': tick.ask - tick.bid,
                                'time': datetime.fromtimestamp(tick.time),
                                'volume': tick.volume
                            }
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                self.logger.error(f"Price monitoring error: {e}")
                time.sleep(1)
    
    def _position_monitoring_loop(self):
        """Real-time position monitoring loop"""
        while self.is_monitoring:
            try:
                # Get all positions
                positions = mt5.positions_get()
                
                if positions is not None:
                    current_tickets = set()
                    
                    for pos in positions:
                        # Only track positions with our magic number
                        if pos.magic == self.magic_number:
                            ticket = pos.ticket
                            current_tickets.add(ticket)
                            
                            # Update position info
                            mt5_position = MT5Position(
                                ticket=pos.ticket,
                                time=datetime.fromtimestamp(pos.time),
                                time_msc=pos.time_msc,
                                time_update=datetime.fromtimestamp(pos.time_update),
                                time_update_msc=pos.time_update_msc,
                                type=pos.type,
                                magic=pos.magic,
                                identifier=pos.identifier,
                                reason=pos.reason,
                                volume=pos.volume,
                                price_open=pos.price_open,
                                sl=pos.sl,
                                tp=pos.tp,
                                price_current=pos.price_current,
                                swap=pos.swap,
                                profit=pos.profit,
                                symbol=pos.symbol,
                                comment=pos.comment,
                                external_id=pos.external_id
                            )
                            
                            with self.lock:
                                self.active_positions[ticket] = mt5_position
                    
                    # Remove closed positions
                    with self.lock:
                        closed_tickets = set(self.active_positions.keys()) - current_tickets
                        for ticket in closed_tickets:
                            del self.active_positions[ticket]
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                time.sleep(5)
    
    def get_symbol_info(self, symbol: str) -> Optional[MT5SymbolInfo]:
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            symbol_info = MT5SymbolInfo(
                name=info.name,
                bid=info.bid,
                ask=info.ask,
                spread=info.spread,
                digits=info.digits,
                trade_mode=info.trade_mode,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                contract_size=info.contract_size,
                margin_initial=info.margin_initial,
                swap_long=info.swap_long,
                swap_short=info.swap_short,
                session_deals=info.session_deals,
                session_buy_orders=info.session_buy_orders,
                session_sell_orders=info.session_sell_orders
            )
            
            # Cache symbol info
            self.symbol_info[symbol] = symbol_info
            return symbol_info
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for symbol"""
        with self.price_lock:
            return self.current_prices.get(symbol)
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        try:
            # Convert timeframe
            tf_map = {
                'M1': MT5TimeFrame.M1.value,
                'M5': MT5TimeFrame.M5.value,
                'M15': MT5TimeFrame.M15.value,
                'H1': MT5TimeFrame.H1.value,
                'H4': MT5TimeFrame.H4.value,
                'D1': MT5TimeFrame.D1.value
            }
            
            mt5_timeframe = tf_map.get(timeframe)
            if mt5_timeframe is None:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    def send_order(self, order: Order) -> Dict[str, Any]:
        """
        Send order to MT5
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to MT5"}
        
        try:
            execution_start = time.time()
            
            # Get symbol info
            symbol_info = self.get_symbol_info(order.symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Invalid symbol: {order.symbol}"}
            
            # Convert order type
            mt5_order_type = self._convert_order_type(order.order_type, order.direction)
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if order.order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": mt5_order_type,
                "magic": self.magic_number,
                "comment": f"Lightning_Scalper_{order.id[:8]}",
                "deviation": 20,  # Max slippage in points
            }
            
            # Add price for limit/stop orders
            if order.order_type != OrderType.MARKET:
                request["price"] = order.price
            
            # Add SL/TP if specified
            if order.stop_loss:
                request["sl"] = order.stop_loss
            if order.take_profit:
                request["tp"] = order.take_profit
            
            # Send order
            result = mt5.order_send(request)
            execution_time = time.time() - execution_start
            
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Order send failed: {error}")
                self._update_stats(False, execution_time, 0)
                return {"success": False, "error": f"MT5 error: {error}"}
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                self._update_stats(False, execution_time, 0)
                return {
                    "success": False, 
                    "error": f"Order failed: {result.retcode} - {result.comment}",
                    "retcode": result.retcode
                }
            
            # Calculate slippage
            requested_price = order.price if order.order_type != OrderType.MARKET else 0
            actual_price = result.price
            slippage = abs(actual_price - requested_price) if requested_price > 0 else 0
            
            # Update statistics
            self._update_stats(True, execution_time, slippage)
            
            # Store position mapping
            if result.order != 0:
                with self.lock:
                    self.position_mapping[order.id] = result.order
            
            self.logger.info(f"Order executed successfully: {result.order} at {result.price}")
            
            return {
                "success": True,
                "mt5_order": result.order,
                "mt5_deal": result.deal,
                "price": result.price,
                "volume": result.volume,
                "slippage": slippage,
                "execution_time": execution_time,
                "comment": result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def close_position(self, position_id: str, volume: Optional[float] = None) -> Dict[str, Any]:
        """Close position by position ID"""
        
        # Find MT5 ticket
        mt5_ticket = self.position_mapping.get(position_id)
        if mt5_ticket is None:
            return {"success": False, "error": "Position not found"}
        
        return self.close_position_by_ticket(mt5_ticket, volume)
    
    def close_position_by_ticket(self, ticket: int, volume: Optional[float] = None) -> Dict[str, Any]:
        """Close position by MT5 ticket"""
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": "Position not found"}
            
            pos = position[0]
            
            # Determine close volume
            close_volume = volume if volume else pos.volume
            
            # Determine order type for closing
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY
            
            # Get current price
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info is None:
                return {"success": False, "error": "Failed to get symbol info"}
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": order_type,
                "position": ticket,
                "magic": self.magic_number,
                "comment": f"Close_Lightning_Scalper_{ticket}",
                "deviation": 20
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return {"success": False, "error": f"Close failed: {error}"}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False, 
                    "error": f"Close failed: {result.retcode} - {result.comment}"
                }
            
            self.logger.info(f"Position {ticket} closed successfully")
            
            return {
                "success": True,
                "deal": result.deal,
                "volume": result.volume,
                "price": result.price
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict[str, Any]:
        """Modify position SL/TP"""
        
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": "Position not found"}
            
            pos = position[0]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp
            }
            
            # Send modification
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return {"success": False, "error": f"Modify failed: {error}"}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False, 
                    "error": f"Modify failed: {result.retcode} - {result.comment}"
                }
            
            return {"success": True, "message": "Position modified successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _convert_order_type(self, order_type: OrderType, direction: TradeDirection) -> int:
        """Convert our order types to MT5 order types"""
        
        if order_type == OrderType.MARKET:
            return MT5OrderType.BUY.value if direction == TradeDirection.BUY else MT5OrderType.SELL.value
        elif order_type == OrderType.LIMIT:
            return MT5OrderType.BUY_LIMIT.value if direction == TradeDirection.BUY else MT5OrderType.SELL_LIMIT.value
        elif order_type == OrderType.STOP:
            return MT5OrderType.BUY_STOP.value if direction == TradeDirection.BUY else MT5OrderType.SELL_STOP.value
        else:
            # Default to market order
            return MT5OrderType.BUY.value if direction == TradeDirection.BUY else MT5OrderType.SELL.value
    
    def _update_stats(self, success: bool, execution_time: float, slippage: float):
        """Update execution statistics"""
        with self.lock:
            self.execution_stats['total_orders'] += 1
            
            if success:
                self.execution_stats['successful_orders'] += 1
            else:
                self.execution_stats['failed_orders'] += 1
            
            # Update averages
            total = self.execution_stats['total_orders']
            self.execution_stats['avg_execution_time'] = (
                (self.execution_stats['avg_execution_time'] * (total - 1) + execution_time) / total
            )
            
            if success and slippage > 0:
                successful = self.execution_stats['successful_orders']
                self.execution_stats['avg_slippage'] = (
                    (self.execution_stats['avg_slippage'] * (successful - 1) + slippage) / successful
                )
    
    def get_account_info(self) -> Optional[MT5AccountInfo]:
        """Get current account information"""
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            self.account_info = MT5AccountInfo(
                login=account.login,
                server=account.server,
                name=account.name,
                company=account.company,
                currency=account.currency,
                balance=account.balance,
                equity=account.equity,
                margin=account.margin,
                free_margin=account.margin_free,
                margin_level=account.margin_level,
                profit=account.profit,
                leverage=account.leverage,
                margin_so_mode=account.margin_so_mode,
                trade_allowed=account.trade_allowed,
                trade_expert=account.trade_expert,
                margin_mode=account.margin_mode,
                currency_digits=account.currency_digits
            )
            
            return self.account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        with self.lock:
            positions = list(self.active_positions.values())
        
        if not positions:
            return {
                "total_positions": 0,
                "total_volume": 0.0,
                "total_profit": 0.0,
                "positions": []
            }
        
        total_volume = sum(pos.volume for pos in positions)
        total_profit = sum(pos.profit for pos in positions)
        
        position_summary = []
        for pos in positions:
            position_summary.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "profit": pos.profit,
                "swap": pos.swap,
                "open_time": pos.time
            })
        
        return {
            "total_positions": len(positions),
            "total_volume": total_volume,
            "total_profit": total_profit,
            "positions": position_summary
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get adapter execution statistics"""
        with self.lock:
            stats = self.execution_stats.copy()
        
        # Calculate success rate
        if stats['total_orders'] > 0:
            stats['success_rate'] = (stats['successful_orders'] / stats['total_orders']) * 100
        else:
            stats['success_rate'] = 0
        
        stats['connection_status'] = self.connection_status.value
        stats['active_positions'] = len(self.active_positions)
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            
            health_status = {
                "mt5_connected": self.is_connected(),
                "terminal_connected": terminal_info is not None,
                "account_connected": account_info is not None,
                "monitoring_active": self.is_monitoring,
                "total_symbols_tracked": len(self.current_prices),
                "active_positions": len(self.active_positions),
                "last_error": self.last_error,
                "connection_attempts": self.connection_attempts
            }
            
            if terminal_info:
                health_status.update({
                    "terminal_name": terminal_info.name,
                    "terminal_version": terminal_info.version,
                    "terminal_connected_to_server": terminal_info.connected
                })
            
            if account_info:
                health_status.update({
                    "account_login": account_info.login,
                    "account_server": account_info.server,
                    "trade_allowed": account_info.trade_allowed,
                    "expert_allowed": account_info.trade_expert
                })
            
            return health_status
            
        except Exception as e:
            return {
                "error": str(e),
                "mt5_connected": False,
                "status": "ERROR"
            }

# Enhanced MT5 Integration for Trade Executor
class MT5IntegratedExecutor:
    """
    Trade Executor with MT5 Integration
    Combines Trade Executor with MT5 Adapter for real trading
    """
    
    def __init__(self, mt5_adapter: MT5Adapter):
        self.mt5_adapter = mt5_adapter
        self.logger = logging.getLogger('MT5IntegratedExecutor')
    
    def execute_order_real(self, order: Order) -> Dict[str, Any]:
        """Execute order using MT5 adapter"""
        
        if not self.mt5_adapter.is_connected():
            return {"success": False, "error": "MT5 not connected"}
        
        # Send order to MT5
        result = self.mt5_adapter.send_order(order)
        
        if result['success']:
            # Update order with MT5 information
            order.status = OrderStatus.FILLED
            order.filled_price = result['price']
            order.filled_quantity = result['volume']
            order.fill_time = datetime.now()
            
            self.logger.info(f"Order executed via MT5: {order.id}")
        else:
            order.status = OrderStatus.REJECTED
            self.logger.error(f"Order execution failed: {result['error']}")
        
        return result
    
    def close_position_real(self, position_id: str, volume: Optional[float] = None) -> Dict[str, Any]:
        """Close position using MT5 adapter"""
        
        return self.mt5_adapter.close_position(position_id, volume)
    
    def get_real_time_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time prices from MT5"""
        
        prices = {}
        for symbol in symbols:
            price_data = self.mt5_adapter.get_current_price(symbol)
            if price_data:
                prices[symbol] = price_data
        
        return prices
    
    def get_account_status(self) -> Optional[MT5AccountInfo]:
        """Get current account status from MT5"""
        
        return self.mt5_adapter.get_account_info()

# Usage Example and Testing
if __name__ == "__main__":
    print("[ROCKET] Lightning Scalper MT5 Adapter")
    print("=" * 50)
    
    # Initialize MT5 Adapter
    adapter = MT5Adapter(magic_number=12345)
    
    # Connection parameters (replace with real credentials)
    login = 12345678  # Your MT5 login
    password = "your_password"  # Your MT5 password
    server = "YourBroker-Server"  # Your broker's server
    
    print(f"[SATELLITE] Attempting to connect to MT5...")
    print(f"   Login: {login}")
    print(f"   Server: {server}")
    
    # Note: This will fail without real MT5 credentials
    # For testing purposes, we'll simulate the connection
    try:
        # In real implementation, uncomment this:
        # success = adapter.connect(login, password, server)
        
        # For demo purposes:
        print("[WARNING]  Demo Mode: Replace with real MT5 credentials for live trading")
        success = False  # Set to True when you have real credentials
        
        if success:
            print("[CHECK] Connected to MT5 successfully!")
            
            # Get account info
            account_info = adapter.get_account_info()
            if account_info:
                print(f"\n[MONEY] Account Information:")
                print(f"   Login: {account_info.login}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Equity: ${account_info.equity:.2f}")
                print(f"   Free Margin: ${account_info.free_margin:.2f}")
                print(f"   Leverage: 1:{account_info.leverage}")
            
            # Get symbol info
            symbol_info = adapter.get_symbol_info("EURUSD")
            if symbol_info:
                print(f"\n[CHART] EURUSD Symbol Info:")
                print(f"   Bid: {symbol_info.bid:.5f}")
                print(f"   Ask: {symbol_info.ask:.5f}")
                print(f"   Spread: {symbol_info.spread:.1f} points")
                print(f"   Min Volume: {symbol_info.volume_min}")
                print(f"   Max Volume: {symbol_info.volume_max}")
            
            # Get historical data
            df = adapter.get_historical_data("EURUSD", "M5", 10)
            if df is not None:
                print(f"\n[TRENDING_UP] Historical Data (Last 10 M5 candles):")
                print(df.tail())
            
            # Test order (demo)
            from execution.trade_executor import Order, TradeDirection, OrderType
            
            test_order = Order(
                id="TEST_001",
                client_id="CLIENT_001", 
                symbol="EURUSD",
                direction=TradeDirection.BUY,
                order_type=OrderType.MARKET,
                quantity=0.01
            )
            
            print(f"\n[TARGET] Test Order Execution:")
            print(f"   Symbol: {test_order.symbol}")
            print(f"   Direction: {test_order.direction.value}")
            print(f"   Volume: {test_order.quantity}")
            
            # In real trading, this would execute:
            # result = adapter.send_order(test_order)
            # print(f"   Result: {result}")
            
            print("   [WARNING]  Demo Mode: Order not executed")
            
            # Health check
            health = adapter.health_check()
            print(f"\n? Health Check:")
            for key, value in health.items():
                print(f"   {key}: {value}")
            
        else:
            print("[X] Failed to connect to MT5")
            print("   Ensure MT5 is installed and credentials are correct")
            
    except Exception as e:
        print(f"[X] Connection error: {e}")
    
    finally:
        # Clean up
        adapter.disconnect()
    
    print("\n[CHECK] MT5 Adapter Ready for Production!")
    print("[TARGET] Next Step: Integrate with Trade Executor for Live Trading")
    
    # Integration example
    print("\n? Integration Example:")
    print("```python")
    print("# Initialize components")
    print("adapter = MT5Adapter()")
    print("adapter.connect(login, password, server)")
    print("")
    print("# Create integrated executor")
    print("executor = MT5IntegratedExecutor(adapter)")
    print("")
    print("# Execute real trades")
    print("result = executor.execute_order_real(order)")
    print("```")