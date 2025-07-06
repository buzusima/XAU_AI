#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD All-in-One Trading System
ระบบเทรด XAUUSD ครบครันในไฟล์เดียว
- Trading Simulator
- Fractal + RSI Strategy  
- Recovery System
- Risk Management
- Real-time UI

รันด้วย: python all_in_one_trading.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

# ================================================================================
# 1. CONFIGURATION & DATA STRUCTURES
# ================================================================================

@dataclass
class TradingConfig:
    """Trading configuration"""
    # Entry Settings
    lot_size: float = 0.01
    rsi_up: int = 55
    rsi_down: int = 45
    rsi_period: int = 14
    fractal_period: int = 5
    trading_direction: int = 0  # 0=BOTH, 1=BUY_ONLY, 2=SELL_ONLY, 3=STOP
    
    # Exit Settings
    tp_first: int = 200  # Points
    exit_speed: int = 1  # 0=FAST, 1=MEDIUM, 2=SLOW
    dynamic_tp: bool = True
    
    # Recovery System
    recovery_price: int = 100  # Points loss to trigger recovery
    martingale: float = 2.0
    max_recovery: int = 3
    smart_recovery: bool = True
    
    # Risk Management
    daily_loss_limit: float = 100.0
    max_positions: int = 5
    max_drawdown: float = 10.0
    min_account_balance: float = 500.0
    max_spread: int = 100
    
    # System
    symbol: str = "XAUUSD"
    primary_tf: str = "M15"

@dataclass
class Position:
    """Trading position"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    open_price: float
    current_price: float
    open_time: datetime
    tp: float = 0.0
    sl: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    comment: str = ""
    recovery_level: int = 0
    is_recovery: bool = False

@dataclass 
class PriceTick:
    """Price tick data"""
    time: datetime
    bid: float
    ask: float
    last: float
    spread: float

class EngineState(Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"  
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

# ================================================================================
# 2. PRICE SIMULATION ENGINE
# ================================================================================

class XAUUSDPriceEngine:
    """XAUUSD price simulation engine"""
    
    def __init__(self, initial_price: float = 2650.0):
        self.current_price = initial_price
        self.trend_direction = 0  # -1=down, 0=sideways, 1=up
        self.trend_strength = 0.5
        self.volatility = 0.5
        self.last_update = datetime.now()
        
        # Historical data for indicators
        self.price_history = []
        self.ohlc_data = []
        
        # Initialize with some historical data
        self._generate_initial_history()
    
    def _generate_initial_history(self):
        """Generate initial price history for indicators"""
        base_price = self.current_price
        
        for i in range(100, 0, -1):
            timestamp = datetime.now() - timedelta(minutes=i * 15)  # M15 data
            
            # Random price movement
            change = random.gauss(0, 2.0)
            base_price += change
            base_price = max(2600, min(2700, base_price))
            
            # Create OHLC bar
            volatility = random.uniform(1.0, 3.0)
            open_price = base_price
            high_price = open_price + random.uniform(0, volatility)
            low_price = open_price - random.uniform(0, volatility)
            close_price = open_price + random.gauss(0, volatility/2)
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            ohlc = {
                'time': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(50, 200)
            }
            
            self.ohlc_data.append(ohlc)
            base_price = close_price
        
        # Set current price to last close
        if self.ohlc_data:
            self.current_price = self.ohlc_data[-1]['close']
    
    def generate_tick(self) -> PriceTick:
        """Generate next price tick"""
        now = datetime.now()
        
        # Price movement
        base_move = 0.0
        
        # Trend component
        if self.trend_direction != 0:
            trend_move = self.trend_direction * self.trend_strength * 0.1
            base_move += trend_move
        
        # Random component
        random_move = random.gauss(0, 0.5) * self.volatility
        base_move += random_move
        
        # Session volatility
        utc_hour = now.hour
        if 7 <= utc_hour < 16:  # European session
            base_move *= 1.2
        elif 13 <= utc_hour < 22:  # US session
            base_move *= 1.5
        else:  # Asian session
            base_move *= 0.8
        
        # Apply movement
        self.current_price += base_move
        self.current_price = max(2600, min(2700, self.current_price))
        
        # Calculate spread (20-60 points)
        base_spread = 0.30  # Base spread in USD
        spread_variance = random.uniform(0.8, 1.2)
        spread = base_spread * spread_variance
        
        spread_half = spread / 2
        bid = self.current_price - spread_half
        ask = self.current_price + spread_half
        
        # Update trend occasionally
        if random.random() < 0.01:
            self.trend_direction = random.choice([-1, 0, 1])
            self.trend_strength = random.uniform(0.3, 0.8)
            self.volatility = random.uniform(0.4, 0.8)
        
        tick = PriceTick(
            time=now,
            bid=round(bid, 2),
            ask=round(ask, 2),
            last=round(self.current_price, 2),
            spread=round(spread * 100, 1)  # Convert to points
        )
        
        self.price_history.append(tick)
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
        
        # Update OHLC data every 15 minutes
        if not self.ohlc_data or (now - self.ohlc_data[-1]['time']).total_seconds() >= 900:
            self._update_ohlc(tick)
        
        return tick
    
    def _update_ohlc(self, tick: PriceTick):
        """Update OHLC data with new tick"""
        if not self.ohlc_data:
            return
        
        # Create new M15 bar
        last_bar = self.ohlc_data[-1]
        
        new_bar = {
            'time': tick.time.replace(second=0, microsecond=0),
            'open': last_bar['close'],
            'high': tick.last,
            'low': tick.last,
            'close': tick.last,
            'volume': random.randint(50, 200)
        }
        
        self.ohlc_data.append(new_bar)
        if len(self.ohlc_data) > 500:
            self.ohlc_data = self.ohlc_data[-500:]
    
    def get_ohlc_data(self, bars: int = 100) -> List[Dict]:
        """Get OHLC data for analysis"""
        return self.ohlc_data[-bars:] if len(self.ohlc_data) >= bars else self.ohlc_data

# ================================================================================
# 3. TECHNICAL INDICATORS
# ================================================================================

class TechnicalIndicators:
    """Technical indicators for strategy"""
    
    @staticmethod
    def calculate_rsi(data: List[Dict], period: int = 14) -> float:
        """Calculate RSI"""
        if len(data) < period + 1:
            return 50.0  # Neutral RSI
        
        closes = [bar['close'] for bar in data]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    @staticmethod
    def find_fractals(data: List[Dict], period: int = 5) -> Tuple[bool, bool]:
        """Find fractal patterns"""
        if len(data) < period * 2 + 1:
            return False, False
        
        # Check last complete fractal (not current bar)
        check_index = -(period + 1)
        
        if abs(check_index) > len(data):
            return False, False
        
        center_bar = data[check_index]
        
        # Fractal Up (Resistance)
        fractal_up = True
        for i in range(1, period + 1):
            left_idx = check_index - i
            right_idx = check_index + i
            
            if (abs(left_idx) > len(data) or abs(right_idx) > len(data) or
                data[left_idx]['high'] >= center_bar['high'] or
                data[right_idx]['high'] >= center_bar['high']):
                fractal_up = False
                break
        
        # Fractal Down (Support)  
        fractal_down = True
        for i in range(1, period + 1):
            left_idx = check_index - i
            right_idx = check_index + i
            
            if (abs(left_idx) > len(data) or abs(right_idx) > len(data) or
                data[left_idx]['low'] <= center_bar['low'] or
                data[right_idx]['low'] <= center_bar['low']):
                fractal_down = False
                break
        
        return fractal_up, fractal_down

# ================================================================================
# 4. TRADING SIMULATOR
# ================================================================================

class TradingSimulator:
    """Complete trading simulator"""
    
    def __init__(self):
        self.price_engine = XAUUSDPriceEngine()
        self.positions: Dict[int, Position] = {}
        self.next_ticket = 1001
        
        # Account
        self.balance = 10000.0
        self.equity = 10000.0
        self.margin = 0.0
        self.free_margin = 10000.0
        
        # Settings
        self.point_value = 0.01  # $0.01 per point per 0.01 lot
        self.margin_rate = 100.0  # $100 margin per 0.01 lot
        
        # History
        self.trade_history = []
        
        # State
        self.running = False
        self.current_tick = None
        
        self.logger = logging.getLogger("Simulator")
    
    def start(self):
        """Start simulator"""
        if not self.running:
            self.running = True
            threading.Thread(target=self._price_loop, daemon=True).start()
            self.logger.info("Trading simulator started")
    
    def stop(self):
        """Stop simulator"""
        self.running = False
        self.logger.info("Trading simulator stopped")
    
    def _price_loop(self):
        """Price update loop"""
        while self.running:
            try:
                # Generate new tick
                self.current_tick = self.price_engine.generate_tick()
                
                # Update positions
                self._update_positions()
                
                # Update account
                self._update_account()
                
                # Sleep
                time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                self.logger.error(f"Price loop error: {e}")
    
    def _update_positions(self):
        """Update all positions"""
        if not self.current_tick:
            return
        
        for ticket, position in list(self.positions.items()):
            # Update current price
            if position.type == 0:  # BUY
                position.current_price = self.current_tick.bid
                price_diff = position.current_price - position.open_price
            else:  # SELL
                position.current_price = self.current_tick.ask
                price_diff = position.open_price - position.current_price
            
            # Calculate profit
            profit_points = price_diff * 100  # Convert to points
            position.profit = profit_points * position.volume * (self.point_value / 100)
            
            # Check TP/SL
            if position.tp > 0:
                if ((position.type == 0 and position.current_price >= position.tp) or
                    (position.type == 1 and position.current_price <= position.tp)):
                    self._close_position(ticket, "TP Hit")
                    continue
            
            if position.sl > 0:
                if ((position.type == 0 and position.current_price <= position.sl) or
                    (position.type == 1 and position.current_price >= position.sl)):
                    self._close_position(ticket, "SL Hit")
                    continue
    
    def _update_account(self):
        """Update account metrics"""
        self.equity = self.balance
        self.margin = 0.0
        
        for position in self.positions.values():
            self.equity += position.profit
            self.margin += position.volume * self.margin_rate
        
        self.free_margin = self.equity - self.margin
    
    def open_position(self, order_type: int, volume: float, tp: float = 0, sl: float = 0, comment: str = "") -> Dict:
        """Open new position"""
        if not self.current_tick:
            return {"success": False, "error": "No price data"}
        
        # Check margin
        required_margin = volume * self.margin_rate
        if required_margin > self.free_margin:
            return {"success": False, "error": "Not enough margin"}
        
        # Determine open price
        open_price = self.current_tick.ask if order_type == 0 else self.current_tick.bid
        
        # Create position
        ticket = self.next_ticket
        self.next_ticket += 1
        
        position = Position(
            ticket=ticket,
            symbol="XAUUSD",
            type=order_type,
            volume=volume,
            open_price=open_price,
            current_price=open_price,
            open_time=datetime.now(),
            tp=tp,
            sl=sl,
            comment=comment
        )
        
        self.positions[ticket] = position
        
        self.logger.info(f"Position opened: {ticket} {'BUY' if order_type == 0 else 'SELL'} {volume} at {open_price}")
        
        return {
            "success": True,
            "ticket": ticket,
            "price": open_price,
            "volume": volume
        }
    
    def _close_position(self, ticket: int, reason: str = "Manual"):
        """Close position"""
        if ticket not in self.positions:
            return False
        
        position = self.positions[ticket]
        
        # Update balance
        self.balance += position.profit
        
        # Add to history
        self.trade_history.append({
            "ticket": ticket,
            "type": "BUY" if position.type == 0 else "SELL",
            "volume": position.volume,
            "open_price": position.open_price,
            "close_price": position.current_price,
            "profit": position.profit,
            "open_time": position.open_time,
            "close_time": datetime.now(),
            "reason": reason,
            "comment": position.comment
        })
        
        # Remove position
        del self.positions[ticket]
        
        self.logger.info(f"Position closed: {ticket} - Profit: ${position.profit:.2f} ({reason})")
        return True
    
    def close_position(self, ticket: int) -> bool:
        """Close position manually"""
        return self._close_position(ticket, "Manual")
    
    def close_all_positions(self):
        """Close all positions"""
        tickets = list(self.positions.keys())
        for ticket in tickets:
            self.close_position(ticket)
    
    def get_current_tick(self) -> Optional[PriceTick]:
        """Get current price tick"""
        return self.current_tick
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return {
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "free_margin": self.free_margin,
            "margin_level": (self.equity / self.margin * 100) if self.margin > 0 else 0
        }
    
    def get_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    def get_ohlc_data(self, bars: int = 100) -> List[Dict]:
        """Get OHLC data for analysis"""
        return self.price_engine.get_ohlc_data(bars)

# ================================================================================
# 5. TRADING STRATEGY ENGINE
# ================================================================================

class StrategyEngine:
    """Fractal + RSI trading strategy"""
    
    def __init__(self, config: TradingConfig, simulator: TradingSimulator):
        self.config = config
        self.simulator = simulator
        self.indicators = TechnicalIndicators()
        
        # State
        self.state = EngineState.STOPPED
        self.running = False
        self.last_signal_check = None
        
        # Recovery tracking
        self.recovery_groups: Dict[str, List[Position]] = {}
        
        # Performance
        self.signals_generated = 0
        self.trades_executed = 0
        
        self.logger = logging.getLogger("Strategy")
    
    def start(self) -> bool:
        """Start strategy engine"""
        try:
            self.state = EngineState.STARTING
            self.running = True
            
            # Start strategy loop
            threading.Thread(target=self._strategy_loop, daemon=True).start()
            
            self.state = EngineState.RUNNING
            self.logger.info("Strategy engine started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start strategy: {e}")
            self.state = EngineState.ERROR
            return False
    
    def stop(self):
        """Stop strategy engine"""
        self.running = False
        self.state = EngineState.STOPPED
        self.logger.info("Strategy engine stopped")
    
    def pause(self):
        """Pause strategy"""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            self.logger.info("Strategy paused")
    
    def resume(self):
        """Resume strategy"""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            self.logger.info("Strategy resumed")
    
    def _strategy_loop(self):
        """Main strategy loop"""
        while self.running:
            try:
                if self.state == EngineState.RUNNING:
                    self._process_signals()
                    self._check_recovery()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}")
    
    def _process_signals(self):
        """Process trading signals"""
        # Check trading conditions
        if not self._check_trading_conditions():
            return
        
        # Get market data
        ohlc_data = self.simulator.get_ohlc_data(50)
        if len(ohlc_data) < 30:
            return
        
        # Calculate indicators
        rsi = self.indicators.calculate_rsi(ohlc_data, self.config.rsi_period)
        fractal_up, fractal_down = self.indicators.find_fractals(ohlc_data, self.config.fractal_period)
        
        current_tick = self.simulator.get_current_tick()
        if not current_tick:
            return
        
        # Generate signals
        signals = []
        
        # BUY Signal: Fractal Down + RSI > RSI_UP
        if (fractal_down and rsi > self.config.rsi_up and 
            self.config.trading_direction in [0, 1]):  # BOTH or BUY_ONLY
            signals.append({
                "type": "BUY",
                "rsi": rsi,
                "fractal": "DOWN",
                "price": current_tick.ask
            })
        
        # SELL Signal: Fractal Up + RSI < RSI_DOWN
        if (fractal_up and rsi < self.config.rsi_down and 
            self.config.trading_direction in [0, 2]):  # BOTH or SELL_ONLY
            signals.append({
                "type": "SELL", 
                "rsi": rsi,
                "fractal": "UP",
                "price": current_tick.bid
            })
        
        # Process signals
        for signal in signals:
            if self._check_anti_hedge(signal["type"]):
                self._execute_signal(signal)
    
    def _check_trading_conditions(self) -> bool:
        """Check if trading conditions are met"""
        # Check if trading is enabled
        if self.config.trading_direction == 3:  # STOP
            return False
        
        # Check account balance
        account = self.simulator.get_account_info()
        if account["balance"] < self.config.min_account_balance:
            return False
        
        # Check max positions
        if len(self.simulator.positions) >= self.config.max_positions:
            return False
        
        # Check spread
        current_tick = self.simulator.get_current_tick()
        if current_tick and current_tick.spread > self.config.max_spread:
            return False
        
        return True
    
    def _check_anti_hedge(self, signal_type: str) -> bool:
        """Check anti-hedge logic"""
        positions = self.simulator.get_positions()
        
        for position in positions:
            # If we have opposite position, block signal
            if ((position.type == 0 and signal_type == "SELL") or 
                (position.type == 1 and signal_type == "BUY")):
                return False
        
        return True
    
    def _execute_signal(self, signal: Dict):
        """Execute trading signal"""
        try:
            order_type = 0 if signal["type"] == "BUY" else 1
            volume = self.config.lot_size
            
            # Calculate TP/SL
            tp_points = self.config.tp_first
            tp_price = 0
            sl_price = 0
            
            if tp_points > 0:
                if order_type == 0:  # BUY
                    tp_price = signal["price"] + (tp_points * 0.01)
                else:  # SELL
                    tp_price = signal["price"] - (tp_points * 0.01)
            
            comment = f'{signal["type"]} Signal - RSI:{signal["rsi"]:.1f} Fractal:{signal["fractal"]}'
            
            # Execute order
            result = self.simulator.open_position(
                order_type=order_type,
                volume=volume,
                tp=tp_price,
                sl=sl_price,
                comment=comment
            )
            
            if result["success"]:
                self.trades_executed += 1
                self.signals_generated += 1
                self.logger.info(f'Signal executed: {signal["type"]} at {signal["price"]:.2f} - RSI:{signal["rsi"]:.1f}')
            else:
                self.logger.error(f'Signal execution failed: {result["error"]}')
                
        except Exception as e:
            self.logger.error(f"Signal execution error: {e}")
    
    def _check_recovery(self):
        """Check and execute recovery"""
        try:
            positions = self.simulator.get_positions()
            
            for position in positions:
                if self._needs_recovery(position):
                    self._execute_recovery(position)
                    
        except Exception as e:
            self.logger.error(f"Recovery check error: {e}")
    
    def _needs_recovery(self, position: Position) -> bool:
        """Check if position needs recovery"""
        if position.profit >= 0:
            return False
        
        # Calculate loss in points
        current_tick = self.simulator.get_current_tick()
        if not current_tick:
            return False
        
        if position.type == 0:  # BUY
            price_diff = position.open_price - current_tick.bid
        else:  # SELL
            price_diff = current_tick.ask - position.open_price
        
        loss_points = price_diff * 100
        
        return loss_points >= self.config.recovery_price
    
    def _execute_recovery(self, original_position: Position):
        """Execute recovery order"""
        try:
            # Check if already in recovery
            group_key = f"{original_position.ticket}_recovery"
            if group_key in self.recovery_groups:
                recovery_count = len(self.recovery_groups[group_key])
                if recovery_count >= self.config.max_recovery:
                    return
            else:
                self.recovery_groups[group_key] = [original_position]
                recovery_count = 0
            
            # Calculate recovery volume
            recovery_level = recovery_count + 1
            recovery_volume = original_position.volume * (self.config.martingale ** recovery_level)
            
            # Wait for same signal if smart recovery enabled
            if self.config.smart_recovery:
                if not self._wait_for_recovery_signal(original_position.type):
                    return
            
            comment = f"Recovery L{recovery_level} for {original_position.ticket}"
            
            # Execute recovery order
            result = self.simulator.open_position(
                order_type=original_position.type,
                volume=recovery_volume,
                comment=comment
            )
            
            if result["success"]:
                self.recovery_groups[group_key].append(self.simulator.positions[result["ticket"]])
                self.logger.info(f"Recovery executed: Level {recovery_level}, Volume {recovery_volume}")
            else:
                self.logger.error(f"Recovery failed: {result['error']}")
                
        except Exception as e:
            self.logger.error(f"Recovery execution error: {e}")
    
    def _wait_for_recovery_signal(self, original_type: int) -> bool:
        """Wait for same signal for smart recovery"""
        if not self.config.smart_recovery:
            return True
        
        # For now, return True (implement actual signal waiting logic)
        return True
    
    def get_status(self) -> Dict:
        """Get strategy status"""
        account = self.simulator.get_account_info()
        positions = self.simulator.get_positions()
        
        return {
            "state": self.state.value,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "current_positions": len(positions),
            "account_balance": account["balance"],
            "account_equity": account["equity"],
            "recovery_groups": len(self.recovery_groups)
        }

# ================================================================================
# 6. USER INTERFACE
# ================================================================================

class TradingUI:
    """Complete trading interface"""
    
    def __init__(self):
        # Initialize components
        self.config = TradingConfig()
        self.simulator = TradingSimulator()
        self.strategy = StrategyEngine(self.config, self.simulator)
        
        # UI state
        self.running = True
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("XAUUSD All-in-One Trading System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Setup logging
        self.setup_logging()
        
        # Create UI
        self.create_ui()
        
        # Start simulator
        self.simulator.start()
        
        # Start UI updates
        self.start_ui_updates()
        
        self.logger.info("XAUUSD Trading System initialized")
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("UI")
    
    def create_ui(self):
        """Create user interface"""
        # Main container with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Trading tab
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading")
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook) 
        notebook.add(analysis_frame, text="Analysis")
        
        # Create trading interface
        self.create_trading_interface(trading_frame)
        
        # Create analysis interface
        self.create_analysis_interface(analysis_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def create_trading_interface(self, parent):
        """Create main trading interface"""
        # Configure grid
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(2, weight=1)
        
        # Control panel
        self.create_control_panel(parent)
        
        # Market info panel
        self.create_market_panel(parent)
        
        # Strategy panel
        self.create_strategy_panel(parent)
        
        # Positions panel
        self.create_positions_panel(parent)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.LabelFrame(parent, text="Engine Control", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Engine state
        self.state_var = tk.StringVar(value="STOPPED")
        state_label = ttk.Label(control_frame, textvariable=self.state_var, 
                               font=("Arial", 12, "bold"))
        state_label.grid(row=0, column=0, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=1, padx=20)
        
        self.start_btn = ttk.Button(button_frame, text="START", command=self.start_strategy)
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="STOP", command=self.stop_strategy)
        self.stop_btn.grid(row=0, column=1, padx=2)
        
        self.pause_btn = ttk.Button(button_frame, text="PAUSE", command=self.pause_strategy)
        self.pause_btn.grid(row=0, column=2, padx=2)
        
        # Emergency stop
        emergency_btn = ttk.Button(button_frame, text="EMERGENCY STOP", 
                                 command=self.emergency_stop)
        emergency_btn.grid(row=0, column=3, padx=10)
        
        # Account info
        account_frame = ttk.Frame(control_frame)
        account_frame.grid(row=0, column=2, padx=20)
        
        self.balance_var = tk.StringVar(value="Balance: $0.00")
        balance_label = ttk.Label(account_frame, textvariable=self.balance_var)
        balance_label.grid(row=0, column=0)
        
        self.equity_var = tk.StringVar(value="Equity: $0.00") 
        equity_label = ttk.Label(account_frame, textvariable=self.equity_var)
        equity_label.grid(row=1, column=0)
    
    def create_market_panel(self, parent):
        """Create market information panel"""
        market_frame = ttk.LabelFrame(parent, text="Market Info", padding="10")
        market_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Price display
        self.price_var = tk.StringVar(value="XAUUSD: 0.00 / 0.00")
        price_label = ttk.Label(market_frame, textvariable=self.price_var, 
                               font=("Arial", 14, "bold"))
        price_label.pack()
        
        self.spread_var = tk.StringVar(value="Spread: 0.0 points")
        spread_label = ttk.Label(market_frame, textvariable=self.spread_var)
        spread_label.pack()
        
        # Quick trading
        trade_frame = ttk.Frame(market_frame)
        trade_frame.pack(pady=10)
        
        ttk.Button(trade_frame, text="BUY", command=self.manual_buy).pack(side=tk.LEFT, padx=5)
        ttk.Button(trade_frame, text="SELL", command=self.manual_sell).pack(side=tk.LEFT, padx=5)
        ttk.Button(trade_frame, text="Close All", command=self.close_all).pack(side=tk.LEFT, padx=10)
    
    def create_strategy_panel(self, parent):
        """Create strategy configuration panel"""
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Config", padding="10")
        strategy_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Parameters
        param_frame = ttk.Frame(strategy_frame)
        param_frame.pack(fill=tk.X)
        
        # Lot Size
        ttk.Label(param_frame, text="Lot Size:").grid(row=0, column=0, sticky="w", padx=5)
        self.lot_var = tk.DoubleVar(value=self.config.lot_size)
        ttk.Spinbox(param_frame, from_=0.01, to=1.0, increment=0.01, 
                   textvariable=self.lot_var, width=8).grid(row=0, column=1, padx=5)
        
        # RSI Upper
        ttk.Label(param_frame, text="RSI Upper:").grid(row=0, column=2, sticky="w", padx=5)
        self.rsi_up_var = tk.IntVar(value=self.config.rsi_up)
        ttk.Spinbox(param_frame, from_=50, to=80, increment=1, 
                   textvariable=self.rsi_up_var, width=8).grid(row=0, column=3, padx=5)
        
        # RSI Lower
        ttk.Label(param_frame, text="RSI Lower:").grid(row=1, column=0, sticky="w", padx=5)
        self.rsi_down_var = tk.IntVar(value=self.config.rsi_down)
        ttk.Spinbox(param_frame, from_=20, to=50, increment=1, 
                   textvariable=self.rsi_down_var, width=8).grid(row=1, column=1, padx=5)
        
        # TP Points
        ttk.Label(param_frame, text="TP Points:").grid(row=1, column=2, sticky="w", padx=5)
        self.tp_var = tk.IntVar(value=self.config.tp_first)
        ttk.Spinbox(param_frame, from_=50, to=500, increment=10, 
                   textvariable=self.tp_var, width=8).grid(row=1, column=3, padx=5)
        
        # Apply button
        apply_btn = ttk.Button(strategy_frame, text="Apply Config", command=self.apply_config)
        apply_btn.pack(pady=10)
        
        # Strategy status
        self.strategy_status_var = tk.StringVar(value="Strategy: Ready")
        status_label = ttk.Label(strategy_frame, textvariable=self.strategy_status_var)
        status_label.pack()
    
    def create_positions_panel(self, parent):
        """Create positions panel"""
        positions_frame = ttk.LabelFrame(parent, text="Open Positions", padding="5")
        positions_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Position tree
        columns = ("Ticket", "Type", "Volume", "Open Price", "Current Price", "Profit", "Comment")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        # Scrollbar
        pos_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", 
                                     command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pos_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_analysis_interface(self, parent):
        """Create analysis interface"""
        # Performance metrics
        perf_frame = ttk.LabelFrame(parent, text="Performance", padding="10")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.perf_text = tk.Text(perf_frame, height=8, bg="#1e1e1e", fg="#ffffff")
        self.perf_text.pack(fill=tk.X)
        
        # Trade history
        history_frame = ttk.LabelFrame(parent, text="Trade History", padding="5")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        hist_columns = ("Ticket", "Type", "Volume", "Open", "Close", "Profit", "Time", "Reason")
        self.history_tree = ttk.Treeview(history_frame, columns=hist_columns, show="headings")
        
        for col in hist_columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=80)
        
        hist_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", 
                                      command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=hist_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # System logs
        logs_frame = ttk.LabelFrame(parent, text="System Logs", padding="5")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(logs_frame, height=10, bg="#1e1e1e", fg="#ffffff", 
                               font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Setup log handler
        self.setup_log_handler()
    
    def setup_log_handler(self):
        """Setup log handler for UI"""
        class UILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.text_widget.after(0, self._add_log, msg)
                except:
                    pass
            
            def _add_log(self, msg):
                try:
                    self.text_widget.insert(tk.END, f"{msg}\n")
                    self.text_widget.see(tk.END)
                    
                    # Limit lines
                    lines = int(self.text_widget.index('end-1c').split('.')[0])
                    if lines > 500:
                        self.text_widget.delete('1.0', '100.0')
                except:
                    pass
        
        handler = UILogHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    
    def start_ui_updates(self):
        """Start UI update loop"""
        self.update_ui()
    
    def update_ui(self):
        """Update UI with current data"""
        if not self.running:
            return
        
        try:
            # Update price
            tick = self.simulator.get_current_tick()
            if tick:
                self.price_var.set(f"XAUUSD: {tick.bid:.2f} / {tick.ask:.2f}")
                self.spread_var.set(f"Spread: {tick.spread:.1f} points")
            
            # Update account
            account = self.simulator.get_account_info()
            self.balance_var.set(f"Balance: ${account['balance']:.2f}")
            self.equity_var.set(f"Equity: ${account['equity']:.2f}")
            
            # Update strategy state
            self.state_var.set(self.strategy.state.value)
            
            # Update strategy status
            status = self.strategy.get_status()
            self.strategy_status_var.set(
                f"Signals: {status['signals_generated']} | Trades: {status['trades_executed']} | Positions: {status['current_positions']}"
            )
            
            # Update positions
            self.update_positions_display()
            
            # Update performance
            self.update_performance_display()
            
            # Update trade history
            self.update_history_display()
            
        except Exception as e:
            self.logger.error(f"UI update error: {e}")
        
        # Schedule next update
        self.root.after(1000, self.update_ui)
    
    def update_positions_display(self):
        """Update positions display"""
        # Clear existing
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Add current positions
        for position in self.simulator.get_positions():
            profit_color = "green" if position.profit >= 0 else "red"
            
            self.positions_tree.insert('', 'end', values=(
                position.ticket,
                "BUY" if position.type == 0 else "SELL",
                f"{position.volume:.2f}",
                f"{position.open_price:.2f}",
                f"{position.current_price:.2f}",
                f"${position.profit:.2f}",
                position.comment
            ), tags=(profit_color,))
        
        # Configure colors
        self.positions_tree.tag_configure("green", foreground="green")
        self.positions_tree.tag_configure("red", foreground="red")
    
    def update_performance_display(self):
        """Update performance display"""
        try:
            account = self.simulator.get_account_info()
            status = self.strategy.get_status()
            history = self.simulator.trade_history
            
            # Calculate performance metrics
            total_trades = len(history)
            winning_trades = len([t for t in history if t["profit"] > 0])
            losing_trades = len([t for t in history if t["profit"] < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            gross_profit = sum([t["profit"] for t in history if t["profit"] > 0])
            gross_loss = sum([t["profit"] for t in history if t["profit"] < 0])
            net_profit = gross_profit + gross_loss
            
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0
            
            # Update display
            perf_text = f"""
ACCOUNT PERFORMANCE:
Balance: ${account['balance']:.2f}
Equity: ${account['equity']:.2f}
Free Margin: ${account['free_margin']:.2f}

TRADING STATISTICS:
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}
Win Rate: {win_rate:.1f}%

PROFIT & LOSS:
Gross Profit: ${gross_profit:.2f}
Gross Loss: ${gross_loss:.2f}
Net Profit: ${net_profit:.2f}
Profit Factor: {profit_factor:.2f}

STRATEGY STATUS:
Signals Generated: {status['signals_generated']}
Recovery Groups: {status['recovery_groups']}
Current Positions: {status['current_positions']}
"""
            
            self.perf_text.delete('1.0', tk.END)
            self.perf_text.insert('1.0', perf_text)
            
        except Exception as e:
            self.logger.error(f"Performance update error: {e}")
    
    def update_history_display(self):
        """Update trade history display"""
        # Clear existing
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Add recent trades (last 20)
        recent_trades = self.simulator.trade_history[-20:]
        
        for trade in recent_trades:
            profit_color = "green" if trade["profit"] >= 0 else "red"
            
            self.history_tree.insert('', 'end', values=(
                trade["ticket"],
                trade["type"],
                f"{trade['volume']:.2f}",
                f"{trade['open_price']:.2f}",
                f"{trade['close_price']:.2f}",
                f"${trade['profit']:.2f}",
                trade["close_time"].strftime("%H:%M:%S"),
                trade["reason"]
            ), tags=(profit_color,))
        
        self.history_tree.tag_configure("green", foreground="green")
        self.history_tree.tag_configure("red", foreground="red")
    
    # Event handlers
    def start_strategy(self):
        """Start strategy"""
        if self.strategy.start():
            self.logger.info("Strategy started")
            self.status_bar.config(text="Strategy Running")
        else:
            messagebox.showerror("Error", "Failed to start strategy")
    
    def stop_strategy(self):
        """Stop strategy"""
        self.strategy.stop()
        self.logger.info("Strategy stopped")
        self.status_bar.config(text="Strategy Stopped")
    
    def pause_strategy(self):
        """Pause strategy"""
        self.strategy.pause()
        self.logger.info("Strategy paused")
        self.status_bar.config(text="Strategy Paused")
    
    def emergency_stop(self):
        """Emergency stop"""
        if messagebox.askyesno("Emergency Stop", "Close all positions and stop strategy?"):
            self.simulator.close_all_positions()
            self.strategy.stop()
            self.logger.critical("EMERGENCY STOP EXECUTED")
            self.status_bar.config(text="EMERGENCY STOP")
    
    def apply_config(self):
        """Apply configuration changes"""
        try:
            # Update config
            self.config.lot_size = self.lot_var.get()
            self.config.rsi_up = self.rsi_up_var.get()
            self.config.rsi_down = self.rsi_down_var.get()
            self.config.tp_first = self.tp_var.get()
            
            # Validate
            if self.config.rsi_down >= self.config.rsi_up:
                messagebox.showerror("Config Error", "RSI Lower must be less than RSI Upper")
                return
            
            self.logger.info("Configuration updated")
            self.status_bar.config(text="Configuration Applied")
            
        except Exception as e:
            self.logger.error(f"Config update error: {e}")
            messagebox.showerror("Error", f"Failed to update config: {e}")
    
    def manual_buy(self):
        """Manual BUY order"""
        result = self.simulator.open_position(0, self.config.lot_size, comment="Manual BUY")
        if result["success"]:
            self.logger.info(f"Manual BUY executed at {result['price']}")
        else:
            messagebox.showerror("Error", result["error"])
    
    def manual_sell(self):
        """Manual SELL order"""
        result = self.simulator.open_position(1, self.config.lot_size, comment="Manual SELL")
        if result["success"]:
            self.logger.info(f"Manual SELL executed at {result['price']}")
        else:
            messagebox.showerror("Error", result["error"])
    
    def close_all(self):
        """Close all positions"""
        if messagebox.askyesno("Confirm", "Close all positions?"):
            self.simulator.close_all_positions()
            self.logger.info("All positions closed")
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.strategy.stop()
        self.simulator.stop()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.logger.info("🚀 XAUUSD All-in-One Trading System started")
        self.root.mainloop()

# ================================================================================
# 7. MAIN EXECUTION
# ================================================================================

def main():
    """Main function"""
    print("=" * 80)
    print("🏆 XAUUSD All-in-One Trading System")
    print("=" * 80)
    print("Features:")
    print("• Real-time XAUUSD price simulation")
    print("• Fractal + RSI trading strategy")
    print("• Smart recovery system with martingale")
    print("• Comprehensive risk management")
    print("• Live performance monitoring")
    print("• Manual trading capabilities")
    print("• Complete paper trading environment")
    print("=" * 80)
    print()
    
    try:
        app = TradingUI()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application terminated")

if __name__ == "__main__":
    main()