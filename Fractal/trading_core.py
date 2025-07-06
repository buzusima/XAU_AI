import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TradingConfig:
    """Configuration class for UI-adjustable parameters"""
    # Entry Settings
    lot_size: float = 0.01
    rsi_up: int = 55
    rsi_down: int = 45
    rsi_period: int = 14
    fractal_period: int = 5
    trading_direction: int = 0  # 0=BOTH, 1=BUY_ONLY, 2=SELL_ONLY, 3=STOP
    
    # Take Profit Settings
    tp_first: int = 200  # Points for first position
    exit_speed: int = 1  # 0=FAST, 1=MEDIUM, 2=SLOW
    dynamic_tp: bool = True
    
    # Recovery System
    recovery_price: int = 100  # Points loss to trigger recovery
    martingale: float = 2.0
    max_recovery: int = 3
    smart_recovery: bool = True
    
    # Spread Management
    spread_mode: int = 0  # 0=AUTO, 1=FIXED, 2=SMART, 3=NONE
    spread_buffer: int = 5
    max_spread_alert: int = 30
    
    # Timeframe Settings
    primary_tf: str = "M15"
    tf_mode: int = 0  # 0=SINGLE, 1=MULTI, 2=CASCADE, 3=ADAPTIVE
    
    # Risk Management
    daily_loss_limit: float = 100.0
    max_positions: int = 5
    max_drawdown: float = 10.0
    
    # System Settings
    symbol: str = "XAUUSD.v"
    auto_symbol_detect: bool = True
    
    def update_from_dict(self, params: Dict):
        """Update parameters from dictionary (for UI updates)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for UI display"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def get_timeframe_enum(self) -> int:
        """Convert timeframe string to MT5 enum"""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        return tf_map.get(self.primary_tf, mt5.TIMEFRAME_M15)

class XAUUSDTradingCore:
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        
        # Internal tracking
        self.positions = {}
        self.recovery_levels = {}
        self.last_signals = {}
        self.is_trading = False
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update_config(self, new_params: Dict):
        """Update trading configuration from UI"""
        old_config = self.config.to_dict()
        self.config.update_from_dict(new_params)
        
        # Log changes
        changes = {k: v for k, v in new_params.items() if old_config.get(k) != v}
        if changes:
            self.logger.info(f"Config updated: {changes}")
    
    def get_config(self) -> Dict:
        """Get current configuration for UI"""
        return self.config.to_dict()
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        
        # Auto-detect symbol if enabled
        if self.config.auto_symbol_detect:
            detected_symbol = self._detect_gold_symbol()
            if detected_symbol:
                self.config.symbol = detected_symbol
        
        # Check symbol availability
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {self.config.symbol} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.config.symbol, True):
                self.logger.error(f"Failed to select symbol {self.config.symbol}")
                return False
        
        self.logger.info(f"MT5 initialized successfully for {self.config.symbol}")
        return True
    
    def _detect_gold_symbol(self) -> Optional[str]:
        """Auto-detect gold symbol variations"""
        gold_symbols = ["XAUUSD", "XAUUSD.m", "XAUUSD.raw", "#XAUUSD", "GOLD"]
        
        for symbol in gold_symbols:
            if mt5.symbol_info(symbol) is not None:
                self.logger.info(f"Detected gold symbol: {symbol}")
                return symbol
        
        return None
    
    def get_market_data(self, bars: int = 100) -> pd.DataFrame:
        """Get market data for analysis"""
        timeframe = self.config.get_timeframe_enum()
        rates = mt5.copy_rates_from_pos(self.config.symbol, timeframe, 0, bars)
        if rates is None:
            self.logger.error("Failed to get market data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        period = period or self.config.rsi_period
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_fractals(self, data: pd.DataFrame, period: int = None) -> Tuple[pd.Series, pd.Series]:
        """Find fractal highs and lows"""
        period = period or self.config.fractal_period
        high = data['high']
        low = data['low']
        
        fractal_up = pd.Series(False, index=data.index)
        fractal_down = pd.Series(False, index=data.index)
        
        for i in range(period, len(data) - period):
            # Fractal Up (Resistance)
            if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, period+1)) and \
               all(high.iloc[i] >= high.iloc[i+j] for j in range(1, period+1)):
                fractal_up.iloc[i] = True
            
            # Fractal Down (Support)
            if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, period+1)) and \
               all(low.iloc[i] <= low.iloc[i+j] for j in range(1, period+1)):
                fractal_down.iloc[i] = True
        
        return fractal_up, fractal_down
    
    def get_current_spread(self) -> float:
        """Get current spread in points"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 0
        
        spread = symbol_info.spread
        return spread  # Already in points for XAUUSD
    
    def calculate_spread_buffer(self) -> int:
        """Calculate spread buffer based on mode"""
        current_spread = self.get_current_spread()
        
        if self.config.spread_mode == 0:  # AUTO
            return int(current_spread * 1.5) + 2
        elif self.config.spread_mode == 1:  # FIXED
            return self.config.spread_buffer
        elif self.config.spread_mode == 2:  # SMART
            # TODO: Implement smart spread calculation based on history
            return int(current_spread * 1.2) + 1
        else:  # NONE
            return 0
    
    def check_trading_conditions(self) -> Dict:
        """Check if trading conditions are met"""
        # Check if trading is enabled
        if self.config.trading_direction == 3:  # STOP
            return {"can_trade": False, "reason": "Trading stopped"}
        
        # Check spread
        current_spread = self.get_current_spread()
        if current_spread > self.config.max_spread_alert:
            return {"can_trade": False, "reason": f"Spread too high: {current_spread}"}
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.daily_loss_limit:
            return {"can_trade": False, "reason": "Daily loss limit reached"}
        
        # Check max positions
        active_positions = len(self.positions)
        if active_positions >= self.config.max_positions:
            return {"can_trade": False, "reason": "Max positions reached"}
        
        return {"can_trade": True, "spread": current_spread}
    
    def analyze_entry_signals(self) -> Dict:
        """Analyze entry signals based on Fractal + RSI"""
        data = self.get_market_data()
        if data is None or len(data) < 50:
            return {"signal": None, "reason": "Insufficient data"}
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        fractal_up, fractal_down = self.find_fractals(data)
        
        current_rsi = rsi.iloc[-1]
        latest_fractal_up = fractal_up.iloc[-self.config.fractal_period:].any()
        latest_fractal_down = fractal_down.iloc[-self.config.fractal_period:].any()
        
        signals = {}
        
        # BUY Signal: Fractal Down + RSI > RSI_UP
        if latest_fractal_down and current_rsi > self.config.rsi_up:
            if self.config.trading_direction in [0, 1]:  # BOTH or BUY_ONLY
                signals["BUY"] = {
                    "rsi": current_rsi,
                    "fractal_down": True,
                    "strength": min(100, (current_rsi - self.config.rsi_up) * 2)
                }
        
        # SELL Signal: Fractal Up + RSI < RSI_DOWN
        if latest_fractal_up and current_rsi < self.config.rsi_down:
            if self.config.trading_direction in [0, 2]:  # BOTH or SELL_ONLY
                signals["SELL"] = {
                    "rsi": current_rsi,
                    "fractal_up": True,
                    "strength": min(100, (self.config.rsi_down - current_rsi) * 2)
                }
        
        return signals