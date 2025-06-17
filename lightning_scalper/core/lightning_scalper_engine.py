#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper Engine - Fixed Version
เพิ่ม SystemStatus enum ที่หายไป
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import threading
import time

class SystemStatus(Enum):
    """System status enumeration - เพิ่มที่หายไป"""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"

class FVGType(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

class FVGStatus(Enum):
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"

class MarketCondition(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class CurrencyPair(Enum):
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"
    AUDUSD = "AUDUSD"
    USDCAD = "USDCAD"
    USDCHF = "USDCHF"
    NZDUSD = "NZDUSD"
    EURJPY = "EURJPY"
    GBPJPY = "GBPJPY"
    GOLD = "XAUUSD"

@dataclass
class MarketSession:
    """Track market sessions for optimal FVG hunting"""
    name: str
    start_hour: int
    end_hour: int
    volatility_multiplier: float
    is_major: bool

@dataclass
class FVGSignal:
    """Enhanced FVG Signal with comprehensive metadata"""
    id: str
    timestamp: datetime
    timeframe: str
    currency_pair: CurrencyPair
    fvg_type: FVGType
    high: float
    low: float
    gap_size: float
    gap_percentage: float
    confluence_score: float
    market_condition: MarketCondition
    session: str
    status: FVGStatus
    
    # Price levels
    entry_price: float
    target_1: float
    target_2: float
    target_3: float
    stop_loss: float
    
    # Risk metrics
    risk_reward_ratio: float
    position_size_factor: float
    urgency_level: int  # 1-5, 5 = highest
    
    # Market context
    atr_ratio: float
    volume_strength: float
    momentum_score: float
    structure_score: float
    
    # Tracking
    fill_percentage: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl_pips: Optional[float] = None
    
    # Metadata for learning
    tags: List[str] = field(default_factory=list)
    notes: str = ""

class EnhancedFVGDetector:
    """
    Production-Grade Lightning Scalper FVG Detection Engine
    Advanced SMC-based Fair Value Gap detection with ML enhancement
    """
    
    def __init__(self, 
                 min_gap_size: float = 5.0,
                 min_confluence: float = 60.0,
                 max_fvg_age: int = 300):
        """
        Initialize Enhanced FVG Detector
        
        Args:
            min_gap_size: Minimum gap size in pips
            min_confluence: Minimum confluence score (0-100)
            max_fvg_age: Maximum age of FVG in seconds
        """
        self.min_gap_size = min_gap_size
        self.min_confluence = min_confluence
        self.max_fvg_age = max_fvg_age
        
        # Detection parameters
        self.volume_threshold = 1.2  # Volume multiplier
        self.atr_multiplier = 0.8    # ATR-based validation
        self.confluence_weights = {
            'structure': 0.25,
            'volume': 0.20,
            'momentum': 0.20,
            'session': 0.15,
            'fibonacci': 0.10,
            'support_resistance': 0.10
        }
        
        # Market sessions for timing
        self.sessions = {
            'london': MarketSession('London', 8, 17, 1.3, True),
            'new_york': MarketSession('New York', 13, 22, 1.4, True),
            'tokyo': MarketSession('Tokyo', 23, 8, 1.1, True),
            'sydney': MarketSession('Sydney', 21, 6, 0.9, False)
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_detected': 0,
            'high_quality_signals': 0,
            'successful_signals': 0,
            'avg_processing_time': 0.0
        }
        
        # Cache for processed data
        self.price_cache = {}
        self.fvg_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        
        # Thread safety
        self.detection_lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('EnhancedFVGDetector')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("[SATELLITE] Enhanced FVG Detector initialized")
    
    def detect_fvgs(self, 
                   data: pd.DataFrame, 
                   currency_pair: CurrencyPair,
                   timeframe: str = 'M5') -> List[FVGSignal]:
        """
        Detect Fair Value Gaps in price data
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            currency_pair: Currency pair being analyzed
            timeframe: Timeframe of the data
            
        Returns:
            List of detected FVG signals
        """
        try:
            start_time = time.time()
            
            with self.detection_lock:
                signals = []
                
                if len(data) < 10:  # Need minimum data
                    return signals
                
                # Prepare data
                df = data.copy()
                df = self._prepare_data(df)
                
                # Detect bullish FVGs
                bullish_fvgs = self._detect_bullish_fvgs(df, currency_pair, timeframe)
                signals.extend(bullish_fvgs)
                
                # Detect bearish FVGs
                bearish_fvgs = self._detect_bearish_fvgs(df, currency_pair, timeframe)
                signals.extend(bearish_fvgs)
                
                # Filter by confluence
                high_quality_signals = [s for s in signals if s.confluence_score >= self.min_confluence]
                
                # Update stats
                processing_time = time.time() - start_time
                self._update_detection_stats(len(signals), len(high_quality_signals), processing_time)
                
                self.logger.debug(f"[LIGHTNING] Detected {len(high_quality_signals)} high-quality FVGs for {currency_pair.value}")
                
                return high_quality_signals
                
        except Exception as e:
            self.logger.error(f"[X] FVG detection failed: {e}")
            return []
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and enrich price data for analysis"""
        try:
            # Calculate technical indicators
            df['atr'] = self._calculate_atr(df, period=14)
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price momentum
            df['price_change'] = df['close'].pct_change()
            df['momentum'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=20).mean()
            
            # Support and Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            return df
            
        except Exception as e:
            self.logger.error(f"[X] Data preparation failed: {e}")
            return df
    
    def _detect_bullish_fvgs(self, df: pd.DataFrame, currency_pair: CurrencyPair, timeframe: str) -> List[FVGSignal]:
        """Detect bullish Fair Value Gaps"""
        signals = []
        
        try:
            for i in range(2, len(df) - 1):
                # Check for bullish FVG pattern
                # Pattern: candle[i-1].low > candle[i+1].high (gap between)
                
                prev_candle = df.iloc[i-1]
                current_candle = df.iloc[i]
                next_candle = df.iloc[i+1]
                
                if prev_candle['low'] > next_candle['high']:
                    # We have a bullish FVG
                    gap_size = (prev_candle['low'] - next_candle['high']) * 10000  # Convert to pips
                    
                    if gap_size >= self.min_gap_size:
                        signal = self._create_bullish_fvg_signal(
                            df, i, currency_pair, timeframe, gap_size
                        )
                        
                        if signal and signal.confluence_score >= self.min_confluence:
                            signals.append(signal)
                            
        except Exception as e:
            self.logger.error(f"[X] Bullish FVG detection failed: {e}")
        
        return signals
    
    def _detect_bearish_fvgs(self, df: pd.DataFrame, currency_pair: CurrencyPair, timeframe: str) -> List[FVGSignal]:
        """Detect bearish Fair Value Gaps"""
        signals = []
        
        try:
            for i in range(2, len(df) - 1):
                # Check for bearish FVG pattern
                # Pattern: candle[i-1].high < candle[i+1].low (gap between)
                
                prev_candle = df.iloc[i-1]
                current_candle = df.iloc[i]
                next_candle = df.iloc[i+1]
                
                if prev_candle['high'] < next_candle['low']:
                    # We have a bearish FVG
                    gap_size = (next_candle['low'] - prev_candle['high']) * 10000  # Convert to pips
                    
                    if gap_size >= self.min_gap_size:
                        signal = self._create_bearish_fvg_signal(
                            df, i, currency_pair, timeframe, gap_size
                        )
                        
                        if signal and signal.confluence_score >= self.min_confluence:
                            signals.append(signal)
                            
        except Exception as e:
            self.logger.error(f"[X] Bearish FVG detection failed: {e}")
        
        return signals
    
    def _create_bullish_fvg_signal(self, df: pd.DataFrame, index: int, 
                                  currency_pair: CurrencyPair, timeframe: str, 
                                  gap_size: float) -> Optional[FVGSignal]:
        """Create a bullish FVG signal"""
        try:
            candle = df.iloc[index]
            prev_candle = df.iloc[index-1]
            next_candle = df.iloc[index+1]
            
            # FVG boundaries
            fvg_high = prev_candle['low']
            fvg_low = next_candle['high']
            
            # Entry and targets
            entry_price = fvg_low + (gap_size * 0.0001 * 0.2)  # Enter at 20% into the gap
            
            # Calculate targets and stop loss
            atr = candle['atr']
            target_1 = entry_price + (atr * 1.0)
            target_2 = entry_price + (atr * 2.0)
            target_3 = entry_price + (atr * 3.0)
            stop_loss = fvg_low - (atr * 0.5)
            
            # Risk-reward ratio
            risk = entry_price - stop_loss
            reward = target_1 - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(df, index, 'BULLISH')
            
            # Create signal
            signal = FVGSignal(
                id=f"BULL_{currency_pair.value}_{timeframe}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                timeframe=timeframe,
                currency_pair=currency_pair,
                fvg_type=FVGType.BULLISH,
                high=fvg_high,
                low=fvg_low,
                gap_size=gap_size,
                gap_percentage=(gap_size / candle['close']) * 100,
                confluence_score=confluence_score,
                market_condition=self._assess_market_condition(df, index),
                session=self._get_current_session(),
                status=FVGStatus.ACTIVE,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                stop_loss=stop_loss,
                risk_reward_ratio=rr_ratio,
                position_size_factor=self._calculate_position_size_factor(confluence_score, rr_ratio),
                urgency_level=self._calculate_urgency_level(confluence_score, gap_size),
                atr_ratio=gap_size / (atr * 10000),
                volume_strength=candle.get('volume_ratio', 1.0),
                momentum_score=candle.get('momentum', 1.0),
                structure_score=self._calculate_structure_score(df, index),
                tags=['bullish_fvg', timeframe, currency_pair.value.lower()]
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create bullish FVG signal: {e}")
            return None
    
    def _create_bearish_fvg_signal(self, df: pd.DataFrame, index: int, 
                                  currency_pair: CurrencyPair, timeframe: str, 
                                  gap_size: float) -> Optional[FVGSignal]:
        """Create a bearish FVG signal"""
        try:
            candle = df.iloc[index]
            prev_candle = df.iloc[index-1]
            next_candle = df.iloc[index+1]
            
            # FVG boundaries
            fvg_high = next_candle['low']
            fvg_low = prev_candle['high']
            
            # Entry and targets
            entry_price = fvg_high - (gap_size * 0.0001 * 0.2)  # Enter at 20% into the gap
            
            # Calculate targets and stop loss
            atr = candle['atr']
            target_1 = entry_price - (atr * 1.0)
            target_2 = entry_price - (atr * 2.0)
            target_3 = entry_price - (atr * 3.0)
            stop_loss = fvg_high + (atr * 0.5)
            
            # Risk-reward ratio
            risk = stop_loss - entry_price
            reward = entry_price - target_1
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(df, index, 'BEARISH')
            
            # Create signal
            signal = FVGSignal(
                id=f"BEAR_{currency_pair.value}_{timeframe}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                timeframe=timeframe,
                currency_pair=currency_pair,
                fvg_type=FVGType.BEARISH,
                high=fvg_high,
                low=fvg_low,
                gap_size=gap_size,
                gap_percentage=(gap_size / candle['close']) * 100,
                confluence_score=confluence_score,
                market_condition=self._assess_market_condition(df, index),
                session=self._get_current_session(),
                status=FVGStatus.ACTIVE,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                stop_loss=stop_loss,
                risk_reward_ratio=rr_ratio,
                position_size_factor=self._calculate_position_size_factor(confluence_score, rr_ratio),
                urgency_level=self._calculate_urgency_level(confluence_score, gap_size),
                atr_ratio=gap_size / (atr * 10000),
                volume_strength=candle.get('volume_ratio', 1.0),
                momentum_score=candle.get('momentum', 1.0),
                structure_score=self._calculate_structure_score(df, index),
                tags=['bearish_fvg', timeframe, currency_pair.value.lower()]
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create bearish FVG signal: {e}")
            return None
    
    def _calculate_confluence_score(self, df: pd.DataFrame, index: int, fvg_type: str) -> float:
        """Calculate confluence score for FVG signal"""
        try:
            scores = {}
            
            # Structure score (20%)
            scores['structure'] = self._calculate_structure_score(df, index)
            
            # Volume score (20%)
            candle = df.iloc[index]
            volume_ratio = candle.get('volume_ratio', 1.0)
            scores['volume'] = min(100, volume_ratio * 50)
            
            # Momentum score (20%)
            momentum = candle.get('momentum', 1.0)
            if fvg_type == 'BULLISH':
                scores['momentum'] = max(0, min(100, (momentum - 1) * 200))
            else:
                scores['momentum'] = max(0, min(100, (1 - momentum) * 200))
            
            # Session score (15%)
            scores['session'] = self._calculate_session_score()
            
            # Fibonacci score (10%)
            scores['fibonacci'] = self._calculate_fibonacci_score(df, index)
            
            # Support/Resistance score (10%)
            scores['support_resistance'] = self._calculate_sr_score(df, index)
            
            # Weighted average
            total_score = sum(scores[key] * self.confluence_weights[key] for key in scores)
            
            return min(100, max(0, total_score))
            
        except Exception as e:
            self.logger.error(f"[X] Confluence calculation failed: {e}")
            return 50.0  # Default score
    
    def _calculate_structure_score(self, df: pd.DataFrame, index: int) -> float:
        """Calculate market structure score"""
        try:
            # Look at higher highs and higher lows for trend structure
            window = min(20, len(df) - index - 1)
            if window < 5:
                return 50.0
            
            recent_data = df.iloc[max(0, index-window):index+1]
            
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Check for trending structure
            hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
            
            # Score based on trend consistency
            trend_strength = (hh_count + hl_count) / (len(highs) - 1) * 100
            
            return min(100, trend_strength)
            
        except Exception:
            return 50.0
    
    def _calculate_session_score(self) -> float:
        """Calculate score based on current market session"""
        try:
            current_hour = datetime.now().hour
            
            for session in self.sessions.values():
                if session.start_hour <= current_hour <= session.end_hour:
                    return session.volatility_multiplier * 50
            
            return 30.0  # Off-session hours
            
        except Exception:
            return 50.0
    
    def _calculate_fibonacci_score(self, df: pd.DataFrame, index: int) -> float:
        """Calculate Fibonacci retracement score"""
        try:
            # Simplified fibonacci score based on price position
            window = min(50, len(df))
            recent_data = df.iloc[max(0, index-window):index+1]
            
            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            current_price = df.iloc[index]['close']
            
            # Calculate retracement level
            if high_price != low_price:
                retracement = (current_price - low_price) / (high_price - low_price)
                
                # Score higher if near key fibonacci levels
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                min_distance = min(abs(retracement - level) for level in fib_levels)
                
                return max(0, 100 - (min_distance * 300))
            
            return 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_sr_score(self, df: pd.DataFrame, index: int) -> float:
        """Calculate support/resistance score"""
        try:
            candle = df.iloc[index]
            current_price = candle['close']
            
            # Check proximity to support/resistance levels
            support = candle.get('support', current_price)
            resistance = candle.get('resistance', current_price)
            
            # Score based on distance from S/R levels
            price_range = resistance - support
            if price_range > 0:
                support_distance = abs(current_price - support) / price_range
                resistance_distance = abs(current_price - resistance) / price_range
                
                min_distance = min(support_distance, resistance_distance)
                return max(0, 100 - (min_distance * 200))
            
            return 50.0
            
        except Exception:
            return 50.0
    
    def _assess_market_condition(self, df: pd.DataFrame, index: int) -> MarketCondition:
        """Assess current market condition"""
        try:
            window = min(20, len(df))
            recent_data = df.iloc[max(0, index-window):index+1]
            
            # Calculate volatility
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
            
            # Calculate trend
            first_price = recent_data['close'].iloc[0]
            last_price = recent_data['close'].iloc[-1]
            trend_change = (last_price - first_price) / first_price
            
            # Classify condition
            if volatility > 0.02:  # High volatility threshold
                return MarketCondition.HIGH_VOLATILITY
            elif volatility < 0.005:  # Low volatility threshold
                return MarketCondition.LOW_VOLATILITY
            elif trend_change > 0.01:  # Uptrend threshold
                return MarketCondition.TRENDING_UP
            elif trend_change < -0.01:  # Downtrend threshold
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.RANGING
                
        except Exception:
            return MarketCondition.RANGING
    
    def _get_current_session(self) -> str:
        """Get current market session"""
        try:
            current_hour = datetime.now().hour
            
            for name, session in self.sessions.items():
                if session.start_hour <= current_hour <= session.end_hour:
                    return name.title()
            
            return "Off-Market"
            
        except Exception:
            return "Unknown"
    
    def _calculate_position_size_factor(self, confluence_score: float, rr_ratio: float) -> float:
        """Calculate position size factor based on signal quality"""
        try:
            # Base factor on confluence score
            base_factor = confluence_score / 100
            
            # Adjust based on risk-reward ratio
            rr_adjustment = min(1.5, max(0.5, rr_ratio / 2))
            
            return min(1.0, base_factor * rr_adjustment)
            
        except Exception:
            return 0.5
    
    def _calculate_urgency_level(self, confluence_score: float, gap_size: float) -> int:
        """Calculate urgency level (1-5)"""
        try:
            # Base on confluence score
            confluence_urgency = int(confluence_score / 20)
            
            # Adjust based on gap size
            if gap_size > 20:
                gap_urgency = 5
            elif gap_size > 15:
                gap_urgency = 4
            elif gap_size > 10:
                gap_urgency = 3
            elif gap_size > 7:
                gap_urgency = 2
            else:
                gap_urgency = 1
            
            return min(5, max(1, max(confluence_urgency, gap_urgency)))
            
        except Exception:
            return 3
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"[X] ATR calculation failed: {e}")
            return pd.Series([0.0001] * len(df), index=df.index)
    
    def _update_detection_stats(self, total_signals: int, high_quality_signals: int, processing_time: float):
        """Update detection statistics"""
        try:
            self.detection_stats['total_detected'] += total_signals
            self.detection_stats['high_quality_signals'] += high_quality_signals
            
            # Update average processing time
            current_avg = self.detection_stats['avg_processing_time']
            self.detection_stats['avg_processing_time'] = (current_avg + processing_time) / 2
            
        except Exception as e:
            self.logger.error(f"[X] Stats update failed: {e}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        try:
            stats = self.detection_stats.copy()
            
            # Calculate success rate
            if stats['high_quality_signals'] > 0:
                stats['success_rate'] = (stats['successful_signals'] / stats['high_quality_signals']) * 100
            else:
                stats['success_rate'] = 0.0
            
            # Calculate quality rate
            if stats['total_detected'] > 0:
                stats['quality_rate'] = (stats['high_quality_signals'] / stats['total_detected']) * 100
            else:
                stats['quality_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"[X] Failed to get stats: {e}")
            return {}
    
    def update_signal_outcome(self, signal_id: str, success: bool, pnl_pips: float):
        """Update signal outcome for learning"""
        try:
            if success:
                self.detection_stats['successful_signals'] += 1
            
            # Here we could implement ML feedback learning
            # For now, just log the outcome
            self.logger.info(f"[CHART] Signal {signal_id} outcome: {'SUCCESS' if success else 'FAILURE'}, PnL: {pnl_pips} pips")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to update signal outcome: {e}")
    
    def cleanup_expired_signals(self, active_signals: List[FVGSignal]) -> List[FVGSignal]:
        """Clean up expired signals"""
        try:
            current_time = datetime.now()
            valid_signals = []
            
            for signal in active_signals:
                age = (current_time - signal.timestamp).total_seconds()
                
                if age < self.max_fvg_age:
                    valid_signals.append(signal)
                else:
                    # Mark as expired
                    signal.status = FVGStatus.EXPIRED
                    self.logger.debug(f"[CLOCK] Signal {signal.id} expired after {age:.0f}s")
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"[X] Signal cleanup failed: {e}")
            return active_signals