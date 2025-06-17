#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import threading
import time

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
    Advanced SMC-based Fair Value Gap hunting with ML-ready features
    """
    
    def __init__(self, currency_pair: CurrencyPair = CurrencyPair.EURUSD):
        self.currency_pair = currency_pair
        self.timeframes = ['M1', 'M5', 'M15', 'H1']
        
        # Active FVG tracking
        self.active_fvgs: Dict[str, List[FVGSignal]] = {tf: [] for tf in self.timeframes}
        self.signal_history: deque = deque(maxlen=1000)  # Keep last 1000 signals
        
        # Currency-specific parameters
        self.pair_configs = self._init_pair_configs()
        self.current_config = self.pair_configs[currency_pair]
        
        # Market sessions
        self.sessions = {
            'Sydney': MarketSession('Sydney', 22, 6, 0.8, False),
            'Tokyo': MarketSession('Tokyo', 0, 8, 1.1, True),
            'London': MarketSession('London', 8, 16, 1.3, True),
            'NewYork': MarketSession('NewYork', 13, 21, 1.2, True),
            'Overlap_Asian': MarketSession('Asian_Overlap', 0, 8, 1.4, True),
            'Overlap_London_NY': MarketSession('London_NY_Overlap', 13, 16, 1.5, True)
        }
        
        # Performance tracking
        self.stats = {
            'total_signals': 0,
            'high_quality_signals': 0,
            'win_rate': 0.0,
            'avg_rr': 0.0,
            'best_session': '',
            'best_timeframe': ''
        }
        
        # Threading for real-time processing
        self.lock = threading.Lock()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f'FVGDetector_{currency_pair.value}')
    
    def _init_pair_configs(self) -> Dict[CurrencyPair, Dict]:
        """Initialize currency-specific configurations"""
        configs = {}
        
        # Major pairs
        major_config = {
            'min_gap_percentage': 0.008,
            'max_gap_percentage': 0.35,
            'confluence_threshold': 65,
            'atr_multiplier': 1.2,
            'volume_threshold': 1.3,
            'pip_value': 0.0001,
            'spread_factor': 1.5
        }
        
        # JPY pairs
        jpy_config = {
            'min_gap_percentage': 0.01,
            'max_gap_percentage': 0.4,
            'confluence_threshold': 70,
            'atr_multiplier': 1.1,
            'volume_threshold': 1.4,
            'pip_value': 0.01,
            'spread_factor': 1.8
        }
        
        # Gold
        gold_config = {
            'min_gap_percentage': 0.015,
            'max_gap_percentage': 0.8,
            'confluence_threshold': 75,
            'atr_multiplier': 1.5,
            'volume_threshold': 1.2,
            'pip_value': 0.1,
            'spread_factor': 2.0
        }
        
        # Assign configurations
        for pair in CurrencyPair:
            if 'JPY' in pair.value:
                configs[pair] = jpy_config.copy()
            elif pair == CurrencyPair.GOLD:
                configs[pair] = gold_config.copy()
            else:
                configs[pair] = major_config.copy()
        
        return configs
    
    def detect_advanced_fvg(self, df: pd.DataFrame, timeframe: str) -> List[FVGSignal]:
        """
        Advanced FVG detection with market structure analysis
        """
        if len(df) < 20:  # Need sufficient history
            return []
        
        signals = []
        current_session = self._get_current_session()
        market_condition = self._analyze_market_condition(df)
        atr = self._calculate_atr(df, period=14)
        
        # Enhanced pattern scanning
        for i in range(10, len(df) - 3):  # Start from index 10 for structure analysis
            
            # Get extended candle context
            context_start = max(0, i - 10)
            context_candles = df.iloc[context_start:i+4]
            
            # Core 3-candle pattern
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Check bullish FVG
            bullish_signal = self._detect_bullish_fvg_advanced(
                prev_candle, curr_candle, next_candle, context_candles,
                timeframe, current_session, market_condition, atr
            )
            if bullish_signal:
                signals.append(bullish_signal)
            
            # Check bearish FVG
            bearish_signal = self._detect_bearish_fvg_advanced(
                prev_candle, curr_candle, next_candle, context_candles,
                timeframe, current_session, market_condition, atr
            )
            if bearish_signal:
                signals.append(bearish_signal)
        
        # Filter and rank signals
        quality_signals = self._filter_quality_signals(signals, timeframe)
        return quality_signals
    
    def _detect_bullish_fvg_advanced(self, prev: pd.Series, curr: pd.Series, next: pd.Series,
                                   context: pd.DataFrame, timeframe: str, session: str,
                                   market_condition: MarketCondition, atr: float) -> Optional[FVGSignal]:
        """Advanced bullish FVG detection with structure analysis"""
        
        # Basic gap check
        if prev['high'] >= next['low']:
            return None
        
        gap_high = next['low']
        gap_low = prev['high']
        gap_size = gap_high - gap_low
        
        # Enhanced gap validation
        mid_price = (prev['high'] + prev['low']) / 2
        gap_percentage = (gap_size / mid_price) * 100
        
        config = self.current_config
        if (gap_percentage < config['min_gap_percentage'] or 
            gap_percentage > config['max_gap_percentage']):
            return None
        
        # Market structure validation
        structure_score = self._analyze_bullish_structure(context, prev, curr, next)
        if structure_score < 30:  # Minimum structure requirement
            return None
        
        # Enhanced confluence calculation
        confluence_data = self._calculate_advanced_confluence(
            prev, curr, next, context, gap_size, atr, FVGType.BULLISH
        )
        
        total_confluence = confluence_data['total_score']
        if total_confluence < config['confluence_threshold']:
            return None
        
        # Smart entry and target calculation
        entry_data = self._calculate_smart_levels_bullish(
            gap_high, gap_low, gap_size, atr, market_condition
        )
        
        # Create signal
        signal_id = f"FVG_{self.currency_pair.value}_{timeframe}_{curr.name.strftime('%Y%m%d_%H%M%S')}"
        
        signal = FVGSignal(
            id=signal_id,
            timestamp=curr.name,
            timeframe=timeframe,
            currency_pair=self.currency_pair,
            fvg_type=FVGType.BULLISH,
            high=gap_high,
            low=gap_low,
            gap_size=gap_size,
            gap_percentage=gap_percentage,
            confluence_score=total_confluence,
            market_condition=market_condition,
            session=session,
            status=FVGStatus.ACTIVE,
            
            entry_price=entry_data['entry'],
            target_1=entry_data['target_1'],
            target_2=entry_data['target_2'],
            target_3=entry_data['target_3'],
            stop_loss=entry_data['stop_loss'],
            
            risk_reward_ratio=entry_data['rr_ratio'],
            position_size_factor=self._calculate_position_size_factor(total_confluence, market_condition),
            urgency_level=self._calculate_urgency(total_confluence, gap_percentage, session),
            
            atr_ratio=gap_size / atr if atr > 0 else 0,
            volume_strength=confluence_data['volume_score'],
            momentum_score=confluence_data['momentum_score'],
            structure_score=structure_score,
            
            tags=self._generate_signal_tags(confluence_data, market_condition, session)
        )
        
        return signal
    
    def _detect_bearish_fvg_advanced(self, prev: pd.Series, curr: pd.Series, next: pd.Series,
                                   context: pd.DataFrame, timeframe: str, session: str,
                                   market_condition: MarketCondition, atr: float) -> Optional[FVGSignal]:
        """Advanced bearish FVG detection with structure analysis"""
        
        # Basic gap check
        if prev['low'] <= next['high']:
            return None
        
        gap_high = prev['low']
        gap_low = next['high']
        gap_size = gap_high - gap_low
        
        # Enhanced gap validation
        mid_price = (prev['high'] + prev['low']) / 2
        gap_percentage = (gap_size / mid_price) * 100
        
        config = self.current_config
        if (gap_percentage < config['min_gap_percentage'] or 
            gap_percentage > config['max_gap_percentage']):
            return None
        
        # Market structure validation
        structure_score = self._analyze_bearish_structure(context, prev, curr, next)
        if structure_score < 30:  # Minimum structure requirement
            return None
        
        # Enhanced confluence calculation
        confluence_data = self._calculate_advanced_confluence(
            prev, curr, next, context, gap_size, atr, FVGType.BEARISH
        )
        
        total_confluence = confluence_data['total_score']
        if total_confluence < config['confluence_threshold']:
            return None
        
        # Smart entry and target calculation
        entry_data = self._calculate_smart_levels_bearish(
            gap_high, gap_low, gap_size, atr, market_condition
        )
        
        # Create signal
        signal_id = f"FVG_{self.currency_pair.value}_{timeframe}_{curr.name.strftime('%Y%m%d_%H%M%S')}"
        
        signal = FVGSignal(
            id=signal_id,
            timestamp=curr.name,
            timeframe=timeframe,
            currency_pair=self.currency_pair,
            fvg_type=FVGType.BEARISH,
            high=gap_high,
            low=gap_low,
            gap_size=gap_size,
            gap_percentage=gap_percentage,
            confluence_score=total_confluence,
            market_condition=market_condition,
            session=session,
            status=FVGStatus.ACTIVE,
            
            entry_price=entry_data['entry'],
            target_1=entry_data['target_1'],
            target_2=entry_data['target_2'],
            target_3=entry_data['target_3'],
            stop_loss=entry_data['stop_loss'],
            
            risk_reward_ratio=entry_data['rr_ratio'],
            position_size_factor=self._calculate_position_size_factor(total_confluence, market_condition),
            urgency_level=self._calculate_urgency(total_confluence, gap_percentage, session),
            
            atr_ratio=gap_size / atr if atr > 0 else 0,
            volume_strength=confluence_data['volume_score'],
            momentum_score=confluence_data['momentum_score'],
            structure_score=structure_score,
            
            tags=self._generate_signal_tags(confluence_data, market_condition, session)
        )
        
        return signal
    
    def _calculate_advanced_confluence(self, prev: pd.Series, curr: pd.Series, next: pd.Series,
                                     context: pd.DataFrame, gap_size: float, atr: float,
                                     fvg_type: FVGType) -> Dict:
        """Calculate comprehensive confluence score"""
        
        confluence_data = {
            'volume_score': 0,
            'momentum_score': 0,
            'structure_score': 0,
            'session_score': 0,
            'volatility_score': 0,
            'pattern_score': 0,
            'total_score': 0
        }
        
        # 1. Volume Analysis (25 points)
        if 'volume' in curr.index and curr['volume'] > 0:
            recent_volumes = context['volume'].tail(5).mean() if 'volume' in context.columns else 0
            if recent_volumes > 0:
                volume_ratio = curr['volume'] / recent_volumes
                if volume_ratio > 2.0:
                    confluence_data['volume_score'] = 25
                elif volume_ratio > 1.5:
                    confluence_data['volume_score'] = 18
                elif volume_ratio > 1.2:
                    confluence_data['volume_score'] = 12
        
        # 2. Momentum Analysis (20 points)
        momentum_score = self._analyze_momentum(context, fvg_type)
        confluence_data['momentum_score'] = momentum_score
        
        # 3. Market Structure (20 points)
        if fvg_type == FVGType.BULLISH:
            structure_score = self._analyze_bullish_structure(context, prev, curr, next)
        else:
            structure_score = self._analyze_bearish_structure(context, prev, curr, next)
        confluence_data['structure_score'] = min(structure_score, 20)
        
        # 4. Session Strength (15 points)
        current_session = self._get_current_session()
        session_multiplier = self.sessions.get(current_session, self.sessions['Sydney']).volatility_multiplier
        confluence_data['session_score'] = min(int(session_multiplier * 10), 15)
        
        # 5. Volatility Context (10 points)
        if atr > 0:
            gap_atr_ratio = gap_size / atr
            if 0.5 <= gap_atr_ratio <= 2.0:  # Optimal volatility range
                confluence_data['volatility_score'] = 10
            elif 0.3 <= gap_atr_ratio <= 3.0:
                confluence_data['volatility_score'] = 6
        
        # 6. Pattern Quality (10 points)
        pattern_score = self._analyze_pattern_quality(prev, curr, next, fvg_type)
        confluence_data['pattern_score'] = pattern_score
        
        # Calculate total
        confluence_data['total_score'] = sum([
            confluence_data['volume_score'],
            confluence_data['momentum_score'],
            confluence_data['structure_score'],
            confluence_data['session_score'],
            confluence_data['volatility_score'],
            confluence_data['pattern_score']
        ])
        
        return confluence_data
    
    def _analyze_bullish_structure(self, context: pd.DataFrame, prev: pd.Series, 
                                 curr: pd.Series, next: pd.Series) -> float:
        """Analyze bullish market structure"""
        score = 0
        
        # Higher highs and higher lows
        recent_highs = context['high'].tail(10)
        recent_lows = context['low'].tail(10)
        
        if len(recent_highs) >= 3:
            if recent_highs.iloc[-1] > recent_highs.iloc[-3]:  # Higher high
                score += 15
            if recent_lows.iloc[-1] > recent_lows.iloc[-3]:   # Higher low
                score += 15
        
        # Support/Resistance context
        current_price = curr['close']
        resistance_broken = any(current_price > h for h in recent_highs.iloc[:-1])
        if resistance_broken:
            score += 20
        
        # Bullish engulfing or strong bullish candle
        if curr['close'] > curr['open'] and (curr['close'] - curr['open']) > (curr['high'] - curr['low']) * 0.6:
            score += 10
        
        return min(score, 60)
    
    def _analyze_bearish_structure(self, context: pd.DataFrame, prev: pd.Series, 
                                 curr: pd.Series, next: pd.Series) -> float:
        """Analyze bearish market structure"""
        score = 0
        
        # Lower highs and lower lows
        recent_highs = context['high'].tail(10)
        recent_lows = context['low'].tail(10)
        
        if len(recent_highs) >= 3:
            if recent_highs.iloc[-1] < recent_highs.iloc[-3]:  # Lower high
                score += 15
            if recent_lows.iloc[-1] < recent_lows.iloc[-3]:   # Lower low
                score += 15
        
        # Support/Resistance context
        current_price = curr['close']
        support_broken = any(current_price < l for l in recent_lows.iloc[:-1])
        if support_broken:
            score += 20
        
        # Bearish engulfing or strong bearish candle
        if curr['close'] < curr['open'] and (curr['open'] - curr['close']) > (curr['high'] - curr['low']) * 0.6:
            score += 10
        
        return min(score, 60)
    
    def _analyze_momentum(self, context: pd.DataFrame, fvg_type: FVGType) -> float:
        """Analyze momentum strength"""
        if len(context) < 5:
            return 0
        
        score = 0
        closes = context['close'].tail(5)
        
        if fvg_type == FVGType.BULLISH:
            # Check for upward momentum
            upward_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] > closes.iloc[i-1])
            score = (upward_moves / (len(closes) - 1)) * 20
        else:
            # Check for downward momentum
            downward_moves = sum(1 for i in range(1, len(closes)) if closes.iloc[i] < closes.iloc[i-1])
            score = (downward_moves / (len(closes) - 1)) * 20
        
        return score
    
    def _analyze_pattern_quality(self, prev: pd.Series, curr: pd.Series, next: pd.Series, fvg_type: FVGType) -> float:
        """Analyze the quality of the FVG pattern formation"""
        score = 0
        
        # Check candle body sizes
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        next_body = abs(next['close'] - next['open'])
        
        # Prefer strong momentum candles
        prev_range = prev['high'] - prev['low']
        curr_range = curr['high'] - curr['low']
        
        if prev_range > 0 and prev_body / prev_range > 0.6:  # Strong previous candle
            score += 3
        
        if curr_range > 0 and curr_body / curr_range > 0.6:  # Strong current candle
            score += 4
        
        # Check for clean gaps (no overlapping wicks)
        if fvg_type == FVGType.BULLISH:
            if prev['high'] < next['low']:  # Clean gap
                score += 3
        else:
            if prev['low'] > next['high']:  # Clean gap
                score += 3
        
        return score
    
    def _calculate_smart_levels_bullish(self, gap_high: float, gap_low: float, gap_size: float, 
                                      atr: float, market_condition: MarketCondition) -> Dict:
        """Calculate smart entry and target levels for bullish FVG"""
        
        # Dynamic risk-reward based on market condition
        rr_base = 1.2 if market_condition == MarketCondition.TRENDING_UP else 1.0
        
        entry = gap_high - (gap_size * 0.1)  # Enter slightly inside the gap
        
        target_1 = entry + (gap_size * rr_base)
        target_2 = entry + (gap_size * rr_base * 1.5)
        target_3 = entry + (gap_size * rr_base * 2.0)
        
        # Adaptive stop loss
        stop_buffer = max(gap_size * 0.3, atr * 0.5) if atr > 0 else gap_size * 0.3
        stop_loss = gap_low - stop_buffer
        
        rr_ratio = (target_1 - entry) / (entry - stop_loss) if entry > stop_loss else 0
        
        return {
            'entry': entry,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'stop_loss': stop_loss,
            'rr_ratio': rr_ratio
        }
    
    def _calculate_smart_levels_bearish(self, gap_high: float, gap_low: float, gap_size: float, 
                                      atr: float, market_condition: MarketCondition) -> Dict:
        """Calculate smart entry and target levels for bearish FVG"""
        
        # Dynamic risk-reward based on market condition
        rr_base = 1.2 if market_condition == MarketCondition.TRENDING_DOWN else 1.0
        
        entry = gap_low + (gap_size * 0.1)  # Enter slightly inside the gap
        
        target_1 = entry - (gap_size * rr_base)
        target_2 = entry - (gap_size * rr_base * 1.5)
        target_3 = entry - (gap_size * rr_base * 2.0)
        
        # Adaptive stop loss
        stop_buffer = max(gap_size * 0.3, atr * 0.5) if atr > 0 else gap_size * 0.3
        stop_loss = gap_high + stop_buffer
        
        rr_ratio = (entry - target_1) / (stop_loss - entry) if stop_loss > entry else 0
        
        return {
            'entry': entry,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'stop_loss': stop_loss,
            'rr_ratio': rr_ratio
        }
    
    def _get_current_session(self) -> str:
        """Determine current trading session"""
        current_hour = datetime.now().hour
        
        # Check for overlaps first (higher priority)
        if 13 <= current_hour <= 16:
            return 'Overlap_London_NY'
        elif 0 <= current_hour <= 8:
            return 'Overlap_Asian'
        elif 8 <= current_hour <= 12:
            return 'London'
        elif 13 <= current_hour <= 21:
            return 'NewYork'
        elif 22 <= current_hour <= 23:
            return 'Sydney'
        else:
            return 'Tokyo'
    
    def _analyze_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """Analyze current market condition"""
        if len(df) < 20:
            return MarketCondition.RANGING
        
        recent_data = df.tail(20)
        closes = recent_data['close']
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Trend analysis
        sma_short = closes.tail(5).mean()
        sma_long = closes.tail(15).mean()
        
        trend_strength = abs(sma_short - sma_long) / sma_long * 100
        
        # Volatility analysis
        atr = self._calculate_atr(recent_data, 14)
        avg_range = (highs - lows).mean()
        volatility_ratio = atr / avg_range if avg_range > 0 else 0
        
        # Determine condition
        if trend_strength > 0.15:
            if sma_short > sma_long:
                return MarketCondition.TRENDING_UP
            else:
                return MarketCondition.TRENDING_DOWN
        elif volatility_ratio > 1.5:
            return MarketCondition.HIGH_VOLATILITY
        elif volatility_ratio < 0.5:
            return MarketCondition.LOW_VOLATILITY
        else:
            return MarketCondition.RANGING
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0
    
    def _calculate_position_size_factor(self, confluence: float, market_condition: MarketCondition) -> float:
        """Calculate position size factor based on signal quality"""
        base_factor = 1.0
        
        # Confluence adjustment
        if confluence >= 90:
            base_factor = 1.5
        elif confluence >= 80:
            base_factor = 1.3
        elif confluence >= 70:
            base_factor = 1.1
        elif confluence < 60:
            base_factor = 0.7
        
        # Market condition adjustment
        if market_condition in [MarketCondition.HIGH_VOLATILITY, MarketCondition.LOW_VOLATILITY]:
            base_factor *= 0.8
        elif market_condition in [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN]:
            base_factor *= 1.1
        
        return min(base_factor, 2.0)  # Cap at 2x
    
    def _calculate_urgency(self, confluence: float, gap_percentage: float, session: str) -> int:
        """Calculate urgency level (1-5)"""
        urgency = 1
        
        # Confluence factor
        if confluence >= 90:
            urgency += 2
        elif confluence >= 80:
            urgency += 1
        
        # Gap size factor
        if gap_percentage >= 0.05:
            urgency += 1
        
        # Session factor
        if session in ['Overlap_London_NY', 'London', 'NewYork']:
            urgency += 1
        
        return min(urgency, 5)
    
    def _generate_signal_tags(self, confluence_data: Dict, market_condition: MarketCondition, session: str) -> List[str]:
        """Generate tags for ML feature engineering"""
        tags = []
        
        tags.append(f"session_{session}")
        tags.append(f"condition_{market_condition.value}")
        
        if confluence_data['volume_score'] > 15:
            tags.append("high_volume")
        if confluence_data['momentum_score'] > 15:
            tags.append("strong_momentum")
        if confluence_data['structure_score'] > 15:
            tags.append("good_structure")
        
        return tags
    
    def _filter_quality_signals(self, signals: List[FVGSignal], timeframe: str) -> List[FVGSignal]:
        """Filter and rank signals by quality"""
        if not signals:
            return []
        
        # Sort by confluence score
        signals.sort(key=lambda x: x.confluence_score, reverse=True)
        
        # Take top signals based on timeframe
        max_signals = {'M1': 3, 'M5': 2, 'M15': 1, 'H1': 1}.get(timeframe, 1)
        
        return signals[:max_signals]
    
    def process_multi_timeframe_advanced(self, data_feeds: Dict[str, pd.DataFrame]) -> Dict[str, List[FVGSignal]]:
        """Process multiple timeframes with advanced analysis"""
        results = {}
        
        with self.lock:
            for timeframe, df in data_feeds.items():
                if df is not None and len(df) > 20:
                    signals = self.detect_advanced_fvg(df, timeframe)
                    results[timeframe] = signals
                    
                    # Update active FVGs
                    self.active_fvgs[timeframe] = signals
                    
                    # Update statistics
                    self.stats['total_signals'] += len(signals)
                    self.stats['high_quality_signals'] += len([s for s in signals if s.confluence_score >= 80])
                else:
                    results[timeframe] = []
        
        return results
    
    def get_execution_ready_signals(self, multi_tf_results: Dict[str, List[FVGSignal]]) -> List[Dict]:
        """Get signals ready for immediate execution"""
        execution_signals = []
        
        # Priority order: Higher timeframe confluence
        for m15_signal in multi_tf_results.get('M15', []):
            for m5_signal in multi_tf_results.get('M5', []):
                if (m15_signal.fvg_type == m5_signal.fvg_type and
                    abs((m15_signal.timestamp - m5_signal.timestamp).total_seconds()) < 900):
                    
                    # Find M1 confirmation
                    for m1_signal in multi_tf_results.get('M1', []):
                        if (m1_signal.fvg_type == m15_signal.fvg_type and
                            abs((m5_signal.timestamp - m1_signal.timestamp).total_seconds()) < 300):
                            
                            # Create execution package
                            execution_signal = {
                                'primary_signal': m1_signal,
                                'confluence_signals': [m15_signal, m5_signal],
                                'total_confluence': m15_signal.confluence_score + m5_signal.confluence_score + m1_signal.confluence_score,
                                'execution_priority': min(m1_signal.urgency_level + 1, 5),
                                'position_size_multiplier': m1_signal.position_size_factor,
                                'risk_level': 'LOW' if m1_signal.risk_reward_ratio > 1.5 else 'MEDIUM',
                                'recommended_action': 'EXECUTE_IMMEDIATELY' if m1_signal.urgency_level >= 4 else 'PREPARE_ENTRY'
                            }
                            execution_signals.append(execution_signal)
        
        # Sort by execution priority
        execution_signals.sort(key=lambda x: x['execution_priority'], reverse=True)
        
        return execution_signals[:5]  # Limit to top 5 for execution focus

# Usage Example
if __name__ == "__main__":
    # Initialize for EURUSD
    detector = EnhancedFVGDetector(CurrencyPair.EURUSD)
    
    print("[ROCKET] Enhanced Lightning Scalper FVG Detection Engine")
    print("=" * 60)
    print(f"Currency Pair: {detector.currency_pair.value}")
    print(f"Configuration: {detector.current_config}")
    print()
    
    # Test with sample data
    dates = pd.date_range(start='2024-01-01 08:00:00', periods=200, freq='1T')
    np.random.seed(42)
    
    # Create more realistic OHLCV data
    base_price = 1.1000
    price_walk = np.random.randn(200).cumsum() * 0.0001
    
    sample_data = pd.DataFrame({
        'open': base_price + price_walk,
        'high': base_price + price_walk + np.random.uniform(0, 0.0005, 200),
        'low': base_price + price_walk - np.random.uniform(0, 0.0005, 200),
        'close': base_price + price_walk + np.random.uniform(-0.0003, 0.0003, 200),
        'volume': np.random.randint(500, 2000, 200)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        row = sample_data.iloc[i]
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = max(row['open'], row['close']) + np.random.uniform(0, 0.0002)
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = min(row['open'], row['close']) - np.random.uniform(0, 0.0002)
    
    # Test advanced detection
    signals = detector.detect_advanced_fvg(sample_data, 'M1')
    
    print(f"[CHART] Advanced FVG Signals Detected: {len(signals)}")
    
    if signals:
        print("\n[TARGET] Top Quality Signals:")
        for i, signal in enumerate(signals[:3]):
            print(f"\n{i+1}. {signal.fvg_type.value} FVG")
            print(f"   Time: {signal.timestamp}")
            print(f"   Confluence: {signal.confluence_score:.1f}")
            print(f"   Gap Size: {signal.gap_size:.5f} ({signal.gap_percentage:.3f}%)")
            print(f"   Entry: {signal.entry_price:.5f}")
            print(f"   Targets: {signal.target_1:.5f} | {signal.target_2:.5f} | {signal.target_3:.5f}")
            print(f"   Stop Loss: {signal.stop_loss:.5f}")
            print(f"   R:R Ratio: {signal.risk_reward_ratio:.2f}")
            print(f"   Urgency: {signal.urgency_level}/5")
            print(f"   Position Factor: {signal.position_size_factor:.2f}x")
            print(f"   Market: {signal.market_condition.value}")
            print(f"   Session: {signal.session}")
            print(f"   Tags: {', '.join(signal.tags)}")
    
    print(f"\n[TRENDING_UP] Detection Statistics:")
    print(f"   Total Signals: {detector.stats['total_signals']}")
    print(f"   High Quality: {detector.stats['high_quality_signals']}")
    
    print("\n[CHECK] Enhanced FVG Detection Engine Ready for Production!")
    print("[TARGET] Next: Real-time Data Integration & Trade Execution Module")