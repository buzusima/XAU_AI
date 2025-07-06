import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import json
from enum import Enum
from trading_core import TradingConfig
from position_manager import PositionManager
from order_executor import OrderExecutor

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradingSession(Enum):
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"
    OVERLAP = "overlap"

@dataclass
class RiskMetrics:
    """Risk metrics tracking"""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    exposure: float = 0.0  # % of balance in open positions
    correlation_risk: float = 0.0

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    daily_loss_limit: float = 100.0
    weekly_loss_limit: float = 500.0
    monthly_loss_limit: float = 2000.0
    max_drawdown_percent: float = 10.0
    max_positions: int = 5
    max_lot_size: float = 1.0
    max_exposure_percent: float = 50.0
    max_correlation: float = 0.8
    min_account_balance: float = 1000.0
    max_spread_trading: int = 30
    max_slippage_points: int = 50

@dataclass
class MarketCondition:
    """Market condition assessment"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    session: TradingSession = TradingSession.ASIAN
    news_risk: bool = False
    weekend_gap: bool = False
    low_liquidity: bool = False
    high_spread: bool = False

class RiskManager:
    def __init__(self, config: TradingConfig, position_manager: PositionManager):
        self.config = config
        self.position_manager = position_manager
        self.risk_limits = RiskLimits()
        self.risk_metrics = RiskMetrics()
        
        # Trading restrictions
        self.trading_allowed = True
        self.risk_level = RiskLevel.LOW
        self.current_restrictions = []
        
        # Market condition tracking
        self.market_condition = MarketCondition()
        
        # Performance tracking
        self.balance_history = []
        self.equity_history = []
        self.trade_history = []
        
        # News and event tracking
        self.high_impact_news_times = []
        self.trading_sessions = self._init_trading_sessions()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with current account info
        self._update_account_info()
    
    def _init_trading_sessions(self) -> Dict[TradingSession, Dict]:
        """Initialize trading session times (UTC)"""
        return {
            TradingSession.ASIAN: {"start": time(0, 0), "end": time(9, 0)},
            TradingSession.EUROPEAN: {"start": time(7, 0), "end": time(16, 0)},
            TradingSession.AMERICAN: {"start": time(13, 0), "end": time(22, 0)}
        }
    
    def update_risk_limits(self, new_limits: Dict):
        """Update risk limits from UI"""
        for key, value in new_limits.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                self.logger.info(f"Risk limit updated: {key} = {value}")
    
    def check_trading_allowed(self) -> Tuple[bool, List[str]]:
        """Check if trading is allowed based on risk conditions"""
        restrictions = []
        
        # Check daily loss limit
        if self.risk_metrics.daily_pnl <= -self.risk_limits.daily_loss_limit:
            restrictions.append(f"Daily loss limit reached: ${abs(self.risk_metrics.daily_pnl):.2f}")
        
        # Check weekly loss limit
        if self.risk_metrics.weekly_pnl <= -self.risk_limits.weekly_loss_limit:
            restrictions.append(f"Weekly loss limit reached: ${abs(self.risk_metrics.weekly_pnl):.2f}")
        
        # Check monthly loss limit
        if self.risk_metrics.monthly_pnl <= -self.risk_limits.monthly_loss_limit:
            restrictions.append(f"Monthly loss limit reached: ${abs(self.risk_metrics.monthly_pnl):.2f}")
        
        # Check max drawdown
        if self.risk_metrics.current_drawdown >= self.risk_limits.max_drawdown_percent:
            restrictions.append(f"Max drawdown exceeded: {self.risk_metrics.current_drawdown:.2f}%")
        
        # Check max positions
        active_positions = len(self.position_manager.positions)
        if active_positions >= self.risk_limits.max_positions:
            restrictions.append(f"Max positions reached: {active_positions}")
        
        # Check account balance
        if self.risk_metrics.current_balance < self.risk_limits.min_account_balance:
            restrictions.append(f"Account balance too low: ${self.risk_metrics.current_balance:.2f}")
        
        # Check market conditions
        market_restrictions = self._check_market_conditions()
        restrictions.extend(market_restrictions)
        
        # Check news events
        if self._is_high_impact_news_time():
            restrictions.append("High impact news event")
        
        self.current_restrictions = restrictions
        self.trading_allowed = len(restrictions) == 0
        
        return self.trading_allowed, restrictions
    
    def _check_market_conditions(self) -> List[str]:
        """Check market conditions for trading restrictions"""
        restrictions = []
        
        # Check spread
        current_spread = self._get_current_spread()
        if current_spread > self.risk_limits.max_spread_trading:
            restrictions.append(f"Spread too high: {current_spread} points")
            self.market_condition.high_spread = True
        
        # Check volatility
        volatility = self._calculate_volatility()
        if volatility > 3.0:  # High volatility threshold
            restrictions.append(f"High volatility detected: {volatility:.2f}")
        
        # Check liquidity (weekend gaps)
        if self._is_weekend_gap():
            restrictions.append("Weekend gap detected")
            self.market_condition.weekend_gap = True
        
        # Check low liquidity periods
        if self._is_low_liquidity_period():
            restrictions.append("Low liquidity period")
            self.market_condition.low_liquidity = True
        
        return restrictions
    
    def validate_order_size(self, volume: float, order_type: str) -> Tuple[bool, float, str]:
        """Validate and adjust order size based on risk limits"""
        # Check maximum lot size
        if volume > self.risk_limits.max_lot_size:
            volume = self.risk_limits.max_lot_size
            self.logger.warning(f"Order size reduced to max limit: {volume}")
        
        # Check exposure limit
        current_exposure = self._calculate_current_exposure()
        additional_exposure = self._calculate_order_exposure(volume)
        
        if (current_exposure + additional_exposure) > self.risk_limits.max_exposure_percent:
            # Reduce volume to stay within exposure limit
            max_additional_exposure = self.risk_limits.max_exposure_percent - current_exposure
            if max_additional_exposure <= 0:
                return False, 0, "Maximum exposure reached"
            
            # Calculate maximum allowed volume
            max_volume = self._calculate_max_volume_for_exposure(max_additional_exposure)
            volume = min(volume, max_volume)
        
        # Validate minimum volume
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info and volume < symbol_info.volume_min:
            return False, 0, f"Volume below minimum: {symbol_info.volume_min}"
        
        return True, volume, "Order size validated"
    
    def assess_risk_level(self) -> RiskLevel:
        """Assess current risk level"""
        risk_score = 0
        
        # Drawdown risk
        if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_percent * 0.8:
            risk_score += 3
        elif self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_percent * 0.5:
            risk_score += 2
        elif self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_percent * 0.3:
            risk_score += 1
        
        # Daily loss risk
        daily_loss_ratio = abs(self.risk_metrics.daily_pnl) / self.risk_limits.daily_loss_limit
        if daily_loss_ratio > 0.8:
            risk_score += 3
        elif daily_loss_ratio > 0.5:
            risk_score += 2
        elif daily_loss_ratio > 0.3:
            risk_score += 1
        
        # Exposure risk
        if self.risk_metrics.exposure > self.risk_limits.max_exposure_percent * 0.8:
            risk_score += 2
        elif self.risk_metrics.exposure > self.risk_limits.max_exposure_percent * 0.5:
            risk_score += 1
        
        # Market condition risk
        if self.market_condition.high_spread or self.market_condition.news_risk:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            self.risk_level = RiskLevel.CRITICAL
        elif risk_score >= 5:
            self.risk_level = RiskLevel.HIGH
        elif risk_score >= 3:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
        
        return self.risk_level
    
    def update_metrics(self):
        """Update risk metrics and performance tracking"""
        self._update_account_info()
        self._update_pnl_metrics()
        self._update_drawdown_metrics()
        self._update_trade_metrics()
        self._update_exposure_metrics()
        self._update_market_condition()
        
        # Assess current risk level
        self.assess_risk_level()
    
    def _update_account_info(self):
        """Update account information"""
        account_info = mt5.account_info()
        if account_info:
            self.risk_metrics.current_balance = account_info.balance
            
            # Update peak balance
            if account_info.balance > self.risk_metrics.peak_balance:
                self.risk_metrics.peak_balance = account_info.balance
            
            # Add to history
            self.balance_history.append({
                "timestamp": datetime.now(),
                "balance": account_info.balance,
                "equity": account_info.equity
            })
            
            # Keep only last 1000 records
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-1000:]
    
    def _update_pnl_metrics(self):
        """Update P&L metrics for different periods"""
        if not self.balance_history:
            return
        
        now = datetime.now()
        
        # Daily P&L
        daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_records = [r for r in self.balance_history if r["timestamp"] >= daily_start]
        if daily_records:
            self.risk_metrics.daily_pnl = daily_records[-1]["balance"] - daily_records[0]["balance"]
        
        # Weekly P&L
        weekly_start = now - timedelta(days=7)
        weekly_records = [r for r in self.balance_history if r["timestamp"] >= weekly_start]
        if weekly_records:
            self.risk_metrics.weekly_pnl = weekly_records[-1]["balance"] - weekly_records[0]["balance"]
        
        # Monthly P&L
        monthly_start = now - timedelta(days=30)
        monthly_records = [r for r in self.balance_history if r["timestamp"] >= monthly_start]
        if monthly_records:
            self.risk_metrics.monthly_pnl = monthly_records[-1]["balance"] - monthly_records[0]["balance"]
    
    def _update_drawdown_metrics(self):
        """Update drawdown metrics"""
        if self.risk_metrics.peak_balance > 0:
            current_drawdown_amount = self.risk_metrics.peak_balance - self.risk_metrics.current_balance
            self.risk_metrics.current_drawdown = (current_drawdown_amount / self.risk_metrics.peak_balance) * 100
            
            # Update max drawdown
            if self.risk_metrics.current_drawdown > self.risk_metrics.max_drawdown:
                self.risk_metrics.max_drawdown = self.risk_metrics.current_drawdown
    
    def _update_trade_metrics(self):
        """Update trading performance metrics"""
        # Get today's deals
        deals = mt5.history_deals_get(
            datetime.now().replace(hour=0, minute=0, second=0),
            datetime.now()
        )
        
        if deals:
            profits = [deal.profit for deal in deals if deal.symbol == self.config.symbol and deal.profit != 0]
            
            if profits:
                self.risk_metrics.total_trades = len(profits)
                self.risk_metrics.winning_trades = len([p for p in profits if p > 0])
                self.risk_metrics.losing_trades = len([p for p in profits if p < 0])
                
                if self.risk_metrics.total_trades > 0:
                    self.risk_metrics.win_rate = (self.risk_metrics.winning_trades / self.risk_metrics.total_trades) * 100
                
                # Calculate profit factor
                gross_profit = sum([p for p in profits if p > 0])
                gross_loss = abs(sum([p for p in profits if p < 0]))
                
                if gross_loss > 0:
                    self.risk_metrics.profit_factor = gross_profit / gross_loss
    
    def _update_exposure_metrics(self):
        """Update exposure metrics"""
        if self.risk_metrics.current_balance > 0:
            total_margin = 0
            for position in self.position_manager.positions.values():
                symbol_info = mt5.symbol_info(self.config.symbol)
                if symbol_info:
                    margin_required = position.volume * symbol_info.margin_initial
                    total_margin += margin_required
            
            self.risk_metrics.exposure = (total_margin / self.risk_metrics.current_balance) * 100
    
    def _update_market_condition(self):
        """Update market condition assessment"""
        # Update trading session
        current_time = datetime.utcnow().time()
        for session, times in self.trading_sessions.items():
            if times["start"] <= current_time <= times["end"]:
                self.market_condition.session = session
                break
        
        # Update volatility
        self.market_condition.volatility = self._calculate_volatility()
        
        # Update trend strength
        self.market_condition.trend_strength = self._calculate_trend_strength()
    
    def _calculate_volatility(self) -> float:
        """Calculate market volatility (ATR-based)"""
        try:
            # Get recent data
            rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H1, 0, 24)
            if rates is None or len(rates) < 14:
                return 0.0
            
            df = pd.DataFrame(rates)
            
            # Calculate True Range
            df['hl'] = df['high'] - df['low']
            df['hc'] = abs(df['high'] - df['close'].shift(1))
            df['lc'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
            
            # Calculate ATR
            atr = df['tr'].rolling(window=14).mean().iloc[-1]
            
            # Normalize ATR as percentage of price
            current_price = df['close'].iloc[-1]
            volatility = (atr / current_price) * 100
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength using ADX-like indicator"""
        try:
            rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H4, 0, 50)
            if rates is None or len(rates) < 20:
                return 0.0
            
            df = pd.DataFrame(rates)
            
            # Simple trend strength calculation
            close_prices = df['close']
            sma_short = close_prices.rolling(window=10).mean()
            sma_long = close_prices.rolling(window=20).mean()
            
            # Calculate trend strength as difference between SMAs
            trend_diff = abs(sma_short.iloc[-1] - sma_long.iloc[-1])
            trend_strength = (trend_diff / close_prices.iloc[-1]) * 100
            
            return min(trend_strength, 100)  # Cap at 100%
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _get_current_spread(self) -> float:
        """Get current spread in points"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        return symbol_info.spread if symbol_info else 0
    
    def _is_high_impact_news_time(self) -> bool:
        """Check if current time is during high impact news"""
        current_time = datetime.now()
        
        for news_time in self.high_impact_news_times:
            if abs((current_time - news_time).total_seconds()) < 1800:  # 30 minutes window
                return True
        
        return False
    
    def _is_weekend_gap(self) -> bool:
        """Detect weekend gap"""
        rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H1, 0, 5)
        if rates is None or len(rates) < 2:
            return False
        
        # Check for significant gap between last two candles
        last_close = rates[-2]['close']
        current_open = rates[-1]['open']
        gap_size = abs(current_open - last_close)
        
        # Consider gap significant if > 0.5% of price
        gap_threshold = last_close * 0.005
        
        return gap_size > gap_threshold
    
    def _is_low_liquidity_period(self) -> bool:
        """Check if current time is low liquidity period"""
        current_time = datetime.utcnow().time()
        
        # Low liquidity periods (UTC)
        low_liquidity_periods = [
            (time(22, 0), time(23, 59)),  # After NY close
            (time(0, 0), time(1, 0)),     # Weekend transition
        ]
        
        for start, end in low_liquidity_periods:
            if start <= current_time <= end:
                return True
        
        return False
    
    def _calculate_current_exposure(self) -> float:
        """Calculate current exposure percentage"""
        return self.risk_metrics.exposure
    
    def _calculate_order_exposure(self, volume: float) -> float:
        """Calculate exposure for new order"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if not symbol_info or self.risk_metrics.current_balance <= 0:
            return 0
        
        margin_required = volume * symbol_info.margin_initial
        exposure = (margin_required / self.risk_metrics.current_balance) * 100
        
        return exposure
    
    def _calculate_max_volume_for_exposure(self, max_exposure_percent: float) -> float:
        """Calculate maximum volume for given exposure limit"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if not symbol_info or self.risk_metrics.current_balance <= 0:
            return 0
        
        max_margin = (max_exposure_percent / 100) * self.risk_metrics.current_balance
        max_volume = max_margin / symbol_info.margin_initial
        
        return max_volume
    
    def add_high_impact_news(self, news_time: datetime):
        """Add high impact news time"""
        self.high_impact_news_times.append(news_time)
        
        # Keep only future news events
        current_time = datetime.now()
        self.high_impact_news_times = [
            t for t in self.high_impact_news_times 
            if t > current_time - timedelta(hours=1)
        ]
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            "risk_level": self.risk_level.value,
            "trading_allowed": self.trading_allowed,
            "restrictions": self.current_restrictions,
            "metrics": {
                "daily_pnl": self.risk_metrics.daily_pnl,
                "weekly_pnl": self.risk_metrics.weekly_pnl,
                "monthly_pnl": self.risk_metrics.monthly_pnl,
                "current_drawdown": self.risk_metrics.current_drawdown,
                "max_drawdown": self.risk_metrics.max_drawdown,
                "exposure": self.risk_metrics.exposure,
                "win_rate": self.risk_metrics.win_rate,
                "profit_factor": self.risk_metrics.profit_factor,
                "total_trades": self.risk_metrics.total_trades
            },
            "market_condition": {
                "volatility": self.market_condition.volatility,
                "trend_strength": self.market_condition.trend_strength,
                "session": self.market_condition.session.value,
                "high_spread": self.market_condition.high_spread,
                "low_liquidity": self.market_condition.low_liquidity,
                "news_risk": self.market_condition.news_risk
            },
            "limits": {
                "daily_loss_limit": self.risk_limits.daily_loss_limit,
                "max_drawdown_percent": self.risk_limits.max_drawdown_percent,
                "max_positions": self.risk_limits.max_positions,
                "max_exposure_percent": self.risk_limits.max_exposure_percent
            }
        }
    
    def emergency_risk_shutdown(self) -> Dict:
        """Emergency shutdown due to risk conditions"""
        self.trading_allowed = False
        self.risk_level = RiskLevel.CRITICAL
        
        shutdown_reason = []
        
        if self.risk_metrics.current_drawdown >= self.risk_limits.max_drawdown_percent:
            shutdown_reason.append(f"Max drawdown exceeded: {self.risk_metrics.current_drawdown:.2f}%")
        
        if abs(self.risk_metrics.daily_pnl) >= self.risk_limits.daily_loss_limit:
            shutdown_reason.append(f"Daily loss limit exceeded: ${abs(self.risk_metrics.daily_pnl):.2f}")
        
        self.logger.critical(f"EMERGENCY RISK SHUTDOWN: {', '.join(shutdown_reason)}")
        
        return {
            "shutdown": True,
            "reason": shutdown_reason,
            "timestamp": datetime.now(),
            "risk_level": self.risk_level.value
        }