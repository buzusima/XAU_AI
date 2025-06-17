#!/usr/bin/env python3
"""
🚀 Lightning Scalper - MetaTrader 5 Expert Advisor
Production-Grade MT5 Integration for Lightning Scalper Signals

📍 ไฟล์นี้: client_tools/lightning_scalper_mt5_ea.py

This EA connects to Lightning Scalper system to receive FVG signals
and automatically execute trades in MetaTrader 5 platform.

Features:
- Real-time signal reception
- Automatic trade execution  
- Advanced risk management
- Position monitoring
- Performance tracking
- Data synchronization back to server

Author: Phoenix Trading AI (อาจารย์ฟินิกซ์)
Version: 1.0.0
License: Proprietary

Installation:
1. Install MetaTrader5 package: pip install MetaTrader5
2. Configure your MT5 login credentials
3. Run this script while MT5 is open
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
from pathlib import Path

# MetaTrader 5 imports
try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 package not found!")
    print("   Install with: pip install MetaTrader5")
    sys.exit(1)

# Add client tools to path
CLIENT_TOOLS_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CLIENT_TOOLS_DIR))

# Import Lightning Scalper SDK
try:
    from lightning_scalper_sdk import (
        LightningScalperClient, TradingSignal, TradeResult, 
        SignalType, OrderStatus
    )
except ImportError:
    print("❌ Lightning Scalper SDK not found!")
    print("   Make sure lightning_scalper_sdk.py is in the same directory")
    sys.exit(1)

class MT5Position:
    """MT5 Position wrapper"""
    def __init__(self, position_info):
        self.ticket = position_info.ticket
        self.symbol = position_info.symbol
        self.type = position_info.type
        self.volume = position_info.volume
        self.price_open = position_info.price_open
        self.price_current = position_info.price_current
        self.profit = position_info.profit
        self.swap = position_info.swap
        self.commission = position_info.commission
        self.time = position_info.time
        self.comment = position_info.comment

class LightningScalperMT5EA:
    """
    🚀 Lightning Scalper MetaTrader 5 Expert Advisor
    Automated trading system connecting Lightning Scalper signals to MT5
    """
    
    def __init__(self, 
                 client_id: str,
                 api_key: str, 
                 api_secret: str,
                 mt5_login: int,
                 mt5_password: str,
                 mt5_server: str,
                 server_url: str = "ws://localhost:8080",
                 http_url: str = "http://localhost:5000"):
        
        # Lightning Scalper client
        self.client = LightningScalperClient(
            client_id=client_id,
            api_key=api_key,
            api_secret=api_secret,
            server_url=server_url,
            http_url=http_url
        )
        
        # MT5 connection details
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        self.mt5_connected = False
        
        # Trading settings
        self.auto_trading = True
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_positions = 5
        self.max_daily_trades = 20
        self.max_spread = 3.0  # Maximum spread in pips
        
        # Position tracking
        self.active_positions: Dict[str, MT5Position] = {}
        self.signal_to_position: Dict[str, int] = {}  # signal_id -> ticket
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
        
        # Risk management
        self.daily_profit = 0.0
        self.daily_loss = 0.0
        self.max_daily_loss = -500.0  # $500 daily loss limit
        self.emergency_stop = False
        
        # Performance tracking
        self.trade_results: List[TradeResult] = []
        self.stats = {
            'signals_received': 0,
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_pips': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Threading
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.position_monitor_thread: Optional[threading.Thread] = None
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'lightning_scalper_ea_{client_id}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'LightningEA-{client_id}')
        
        # Add signal callback
        self.client.add_signal_callback(self._handle_signal)
        
        self.logger.info(f"🚀 Lightning Scalper MT5 EA initialized for client {client_id}")
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("Failed to initialize MT5")
                return False
            
            # Login to account
            if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                self.logger.error(f"Failed to login to MT5: {mt5.last_error()}")
                return False
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            self.mt5_connected = True
            self.logger.info(f"✅ Connected to MT5 - Account: {account_info.login}")
            self.logger.info(f"   Balance: ${account_info.balance:.2f}")
            self.logger.info(f"   Equity: ${account_info.equity:.2f}")
            self.logger.info(f"   Server: {account_info.server}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MetaTrader 5"""
        try:
            mt5.shutdown()
            self.mt5_connected = False
            self.logger.info("Disconnected from MT5")
        except Exception as e:
            self.logger.error(f"MT5 disconnect error: {e}")
    
    def _handle_signal(self, signal: TradingSignal):
        """Handle incoming trading signal"""
        try:
            self.stats['signals_received'] += 1
            
            self.logger.info(f"📡 Received signal: {signal.currency_pair} {signal.signal_type.value}")
            self.logger.info(f"   Entry: {signal.entry_price}")
            self.logger.info(f"   SL: {signal.stop_loss}")
            self.logger.info(f"   TP1: {signal.take_profit_1}")
            self.logger.info(f"   Confidence: {signal.confidence:.2f}")
            
            # Check if we should execute this signal
            if self._should_execute_signal(signal):
                self._execute_signal(signal)
            else:
                self.logger.info(f"Signal {signal.signal_id} skipped due to filters")
                
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """Check if signal should be executed"""
        
        # Check if auto trading is enabled
        if not self.auto_trading:
            self.logger.info("Auto trading is disabled")
            return False
        
        # Check emergency stop
        if self.emergency_stop:
            self.logger.warning("Emergency stop is active")
            return False
        
        # Check MT5 connection
        if not self.mt5_connected:
            self.logger.error("MT5 not connected")
            return False
        
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.last_trade_date:
            self.daily_trades = 0
            self.daily_profit = 0.0
            self.daily_loss = 0.0
            self.last_trade_date = today
            self.logger.info(f"📅 New trading day: {today}")
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return False
        
        # Check maximum positions
        positions = mt5.positions_get()
        if len(positions) >= self.max_positions:
            self.logger.warning(f"Maximum positions reached: {len(positions)}")
            return False
        
        # Check daily loss limit
        if self.daily_loss <= self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: ${self.daily_loss:.2f}")
            return False
        
        # Check signal expiry
        if datetime.now() > signal.expires_at:
            self.logger.warning(f"Signal expired: {signal.signal_id}")
            return False
        
        # Check symbol availability and spread
        symbol_info = mt5.symbol_info(signal.currency_pair)
        if symbol_info is None:
            self.logger.error(f"Symbol not available: {signal.currency_pair}")
            return False
        
        if not symbol_info.visible:
            self.logger.warning(f"Symbol not visible: {signal.currency_pair}")
            return False
        
        # Check spread
        spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point
        if spread > self.max_spread:
            self.logger.warning(f"Spread too high: {spread:.1f} pips")
            return False
        
        # Check minimum confidence
        if signal.confidence < 0.7:  # 70% minimum confidence
            self.logger.info(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        return True
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute trading signal"""
        try:
            # Get account info for position sizing
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info for position sizing")
                return
            
            # Calculate position size
            lot_size = self._calculate_lot_size(signal, account_info.balance)
            
            if lot_size <= 0:
                self.logger.warning("Invalid lot size calculated")
                return
            
            # Prepare order
            symbol = signal.currency_pair
            order_type = mt5.ORDER_TYPE_BUY if signal.signal_type == SignalType.BUY else mt5.ORDER_TYPE_SELL
            
            # Get current price
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return
            
            price = symbol_info.ask if signal.signal_type == SignalType.BUY else symbol_info.bid
            
            # Calculate points for SL and TP
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Stop loss
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            sl_price = signal.stop_loss
            
            # Take profit (use TP1 as primary target)
            tp_distance = abs(signal.take_profit_1 - signal.entry_price)
            tp_price = signal.take_profit_1
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 123456,  # EA magic number
                "comment": f"Lightning-{signal.signal_id[:8]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error(f"Order send failed: {mt5.last_error()}")
                return
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return
            
            # Order successful
            self.logger.info(f"✅ Order executed successfully!")
            self.logger.info(f"   Ticket: {result.order}")
            self.logger.info(f"   Volume: {lot_size}")
            self.logger.info(f"   Price: {result.price}")
            self.logger.info(f"   SL: {sl_price}")
            self.logger.info(f"   TP: {tp_price}")
            
            # Track position
            self.signal_to_position[signal.signal_id] = result.order
            self.daily_trades += 1
            self.stats['trades_executed'] += 1
            
            # Create trade result for reporting
            trade_result = TradeResult(
                signal_id=signal.signal_id,
                client_id=self.client.client_id,
                timestamp=datetime.now(),
                executed=True,
                execution_price=result.price,
                lot_size=lot_size,
                slippage=abs(result.price - price),
                execution_delay=0.0  # Real-time execution
            )
            
            # Start monitoring this position
            self._start_position_monitoring(result.order, signal, trade_result)
            
        except Exception as e:
            self.logger.error(f"Failed to execute signal: {e}")
    
    def _calculate_lot_size(self, signal: TradingSignal, balance: float) -> float:
        """Calculate appropriate lot size based on risk management"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(signal.currency_pair)
            if symbol_info is None:
                return 0.0
            
            # Calculate risk amount
            risk_amount = balance * signal.max_risk_percent
            
            # Calculate stop loss distance in points
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            
            # Calculate lot size
            # Risk Amount = Lot Size * SL Distance * Point Value
            tick_value = symbol_info.trade_tick_value
            if tick_value == 0:
                tick_value = 1.0
            
            lot_size = risk_amount / (sl_distance * tick_value)
            
            # Round to valid lot size
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Check limits
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            
            lot_size = max(min_lot, min(lot_size, max_lot))
            
            # Apply suggested lot size if smaller
            if signal.suggested_lot_size > 0:
                lot_size = min(lot_size, signal.suggested_lot_size)
            
            self.logger.info(f"Calculated lot size: {lot_size} (Risk: ${risk_amount:.2f})")
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating lot size: {e}")
            return 0.0
    
    def _start_position_monitoring(self, ticket: int, signal: TradingSignal, trade_result: TradeResult):
        """Start monitoring a position"""
        def monitor_position():
            try:
                position_closed = False
                start_time = datetime.now()
                max_profit = 0.0
                max_drawdown = 0.0
                
                while not position_closed and self.is_running:
                    # Get position info
                    positions = mt5.positions_get(ticket=ticket)
                    
                    if not positions:
                        # Position closed
                        position_closed = True
                        
                        # Get deal history to find close details
                        deals = mt5.history_deals_get(ticket=ticket)
                        if deals:
                            close_deal = deals[-1]  # Last deal should be the close
                            
                            # Update trade result
                            trade_result.closed_at = datetime.fromtimestamp(close_deal.time)
                            trade_result.close_price = close_deal.price
                            trade_result.profit_loss = close_deal.profit
                            
                            # Calculate pips
                            symbol_info = mt5.symbol_info(signal.currency_pair)
                            if symbol_info:
                                pip_size = 0.0001 if symbol_info.digits == 5 else 0.01
                                price_diff = abs(close_deal.price - trade_result.execution_price)
                                trade_result.profit_loss_pips = price_diff / pip_size
                                
                                if signal.signal_type == SignalType.SELL:
                                    trade_result.profit_loss_pips *= -1 if close_deal.price > trade_result.execution_price else 1
                                else:
                                    trade_result.profit_loss_pips *= 1 if close_deal.price > trade_result.execution_price else -1
                            
                            # Check which targets were hit
                            if signal.signal_type == SignalType.BUY:
                                trade_result.tp1_hit = close_deal.price >= signal.take_profit_1
                                trade_result.tp2_hit = close_deal.price >= signal.take_profit_2
                                trade_result.tp3_hit = close_deal.price >= signal.take_profit_3
                                trade_result.sl_hit = close_deal.price <= signal.stop_loss
                            else:
                                trade_result.tp1_hit = close_deal.price <= signal.take_profit_1
                                trade_result.tp2_hit = close_deal.price <= signal.take_profit_2
                                trade_result.tp3_hit = close_deal.price <= signal.take_profit_3
                                trade_result.sl_hit = close_deal.price >= signal.stop_loss
                            
                            trade_result.max_profit = max_profit
                            trade_result.max_drawdown = max_drawdown
                            trade_result.duration_minutes = int((trade_result.closed_at - trade_result.timestamp).total_seconds() / 60)
                            
                            # Update statistics
                            self._update_stats(trade_result)
                            
                            # Submit result to server
                            self.client.submit_trade_result(trade_result)
                            
                            # Log result
                            result_type = "WIN" if trade_result.profit_loss > 0 else "LOSS"
                            self.logger.info(f"🏁 Position closed: {result_type}")
                            self.logger.info(f"   Ticket: {ticket}")
                            self.logger.info(f"   Profit: ${trade_result.profit_loss:.2f}")
                            self.logger.info(f"   Pips: {trade_result.profit_loss_pips:.1f}")
                            self.logger.info(f"   Duration: {trade_result.duration_minutes} minutes")
                        
                        break
                    
                    else:
                        # Position still open, track metrics
                        position = positions[0]
                        current_profit = position.profit + position.swap
                        
                        max_profit = max(max_profit, current_profit)
                        max_drawdown = min(max_drawdown, current_profit)
                    
                    time.sleep(5)  # Check every 5 seconds
                    
            except Exception as e:
                self.logger.error(f"Error monitoring position {ticket}: {e}")
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_position)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _update_stats(self, trade_result: TradeResult):
        """Update trading statistics"""
        try:
            if trade_result.profit_loss > 0:
                self.stats['winning_trades'] += 1
                self.daily_profit += trade_result.profit_loss
                self.stats['largest_win'] = max(self.stats['largest_win'], trade_result.profit_loss)
            else:
                self.stats['losing_trades'] += 1
                self.daily_loss += trade_result.profit_loss
                self.stats['largest_loss'] = min(self.stats['largest_loss'], trade_result.profit_loss)
            
            self.stats['total_profit'] += trade_result.profit_loss
            self.stats['total_pips'] += trade_result.profit_loss_pips or 0
            
            # Calculate win rate
            total_trades = self.stats['winning_trades'] + self.stats['losing_trades']
            if total_trades > 0:
                self.stats['win_rate'] = self.stats['winning_trades'] / total_trades
            
            # Calculate average profit/loss
            if self.stats['winning_trades'] > 0:
                self.stats['avg_profit'] = sum(t.profit_loss for t in self.trade_results if t.profit_loss > 0) / self.stats['winning_trades']
            
            if self.stats['losing_trades'] > 0:
                self.stats['avg_loss'] = sum(t.profit_loss for t in self.trade_results if t.profit_loss < 0) / self.stats['losing_trades']
            
            # Store trade result
            self.trade_results.append(trade_result)
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    def start(self) -> bool:
        """Start the EA"""
        try:
            self.logger.info("🚀 Starting Lightning Scalper MT5 EA...")
            
            # Connect to MT5
            if not self.connect_mt5():
                self.logger.error("Failed to connect to MT5")
                return False
            
            # Connect to Lightning Scalper server
            if not self.client.connect():
                self.logger.error("Failed to connect to Lightning Scalper server")
                return False
            
            self.is_running = True
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_account)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.logger.info("✅ Lightning Scalper MT5 EA started successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start EA: {e}")
            return False
    
    def stop(self):
        """Stop the EA"""
        try:
            self.logger.info("🛑 Stopping Lightning Scalper MT5 EA...")
            
            self.is_running = False
            
            # Disconnect from Lightning Scalper
            self.client.disconnect()
            
            # Disconnect from MT5
            self.disconnect_mt5()
            
            # Wait for threads to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.logger.info("✅ Lightning Scalper MT5 EA stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping EA: {e}")
    
    def _monitor_account(self):
        """Monitor account and send updates to server"""
        last_update = datetime.now()
        
        while self.is_running:
            try:
                # Update account balance every minute
                if datetime.now() - last_update >= timedelta(minutes=1):
                    account_info = mt5.account_info()
                    if account_info:
                        self.client.update_account_balance(
                            balance=account_info.balance,
                            equity=account_info.equity,
                            margin=account_info.margin
                        )
                        last_update = datetime.now()
                
                # Check emergency conditions
                account_info = mt5.account_info()
                if account_info:
                    # Check margin level
                    if account_info.margin_level < 200 and account_info.margin_level > 0:
                        self.logger.warning(f"⚠️ Low margin level: {account_info.margin_level:.1f}%")
                        if account_info.margin_level < 100:
                            self.emergency_stop = True
                            self.logger.error("🚨 EMERGENCY STOP: Margin call level reached!")
                    
                    # Check daily loss
                    if self.daily_loss <= self.max_daily_loss:
                        self.emergency_stop = True
                        self.logger.error(f"🚨 EMERGENCY STOP: Daily loss limit reached: ${self.daily_loss:.2f}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in account monitoring: {e}")
                time.sleep(30)
    
    def get_status(self) -> dict:
        """Get EA status"""
        account_info = mt5.account_info() if self.mt5_connected else None
        
        return {
            'ea_running': self.is_running,
            'mt5_connected': self.mt5_connected,
            'lightning_connected': self.client.is_connected,
            'auto_trading': self.auto_trading,
            'emergency_stop': self.emergency_stop,
            'daily_trades': self.daily_trades,
            'daily_profit': self.daily_profit,
            'active_positions': len(mt5.positions_get()) if self.mt5_connected else 0,
            'account_balance': account_info.balance if account_info else 0,
            'account_equity': account_info.equity if account_info else 0,
            'stats': self.stats
        }
    
    def set_auto_trading(self, enabled: bool):
        """Enable/disable auto trading"""
        self.auto_trading = enabled
        self.logger.info(f"Auto trading {'enabled' if enabled else 'disabled'}")
    
    def set_emergency_stop(self, stop: bool):
        """Set emergency stop"""
        self.emergency_stop = stop
        if stop:
            self.logger.warning("🚨 Emergency stop activated")
        else:
            self.logger.info("Emergency stop deactivated")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                self.logger.info("No positions to close")
                return
            
            for position in positions:
                self.logger.info(f"Closing position {position.ticket}")
                
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": "Lightning-CloseAll",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"✅ Position {position.ticket} closed")
                else:
                    self.logger.error(f"❌ Failed to close position {position.ticket}")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

def main():
    """Main function for standalone EA execution"""
    print("🚀 Lightning Scalper MetaTrader 5 Expert Advisor")
    print("=" * 60)
    
    # Configuration (in production, load from config file)
    config = {
        'client_id': 'your_client_id',
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret',
        'mt5_login': 12345678,
        'mt5_password': 'your_mt5_password',
        'mt5_server': 'YourBroker-Server',
        'server_url': 'ws://localhost:8080',
        'http_url': 'http://localhost:5000'
    }
    
    # Create EA instance
    ea = LightningScalperMT5EA(**config)
    
    try:
        # Start EA
        if ea.start():
            print("✅ EA started successfully!")
            print("Press Ctrl+C to stop...")
            
            # Run until interrupted
            while True:
                time.sleep(1)
                
                # Print status every 60 seconds
                if int(time.time()) % 60 == 0:
                    status = ea.get_status()
                    print(f"\n📊 Status Update:")
                    print(f"   Signals received: {status['stats']['signals_received']}")
                    print(f"   Trades executed: {status['stats']['trades_executed']}")
                    print(f"   Win rate: {status['stats']['win_rate']:.1%}")
                    print(f"   Daily P&L: ${status['daily_profit']:.2f}")
                    print(f"   Active positions: {status['active_positions']}")
        else:
            print("❌ Failed to start EA")
            
    except KeyboardInterrupt:
        print("\n⌨️ Stopping EA...")
        ea.stop()
        print("👋 Goodbye!")
    
    except Exception as e:
        print(f"\n💀 Critical error: {e}")
        ea.stop()

if __name__ == "__main__":
    main()