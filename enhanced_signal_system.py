"""
Enhanced Multi-Timeframe Signal Generation System - UNICODE FIXED
===============================================
Professional-grade signal analysis with timeframe confluence
Win Rate Target: 65-75% (up from 55%)
NO EMOJI VERSION - WINDOWS COMPATIBLE
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

class MultiTimeframeSignalEngine:
    """
    Professional Multi-Timeframe Signal Generation Engine
    H4 = Main Trend Direction (Macro View)
    H1 = Setup Confirmation (Entry Setup)  
    M15 = Precise Entry Timing (Execution)
    M5 = Risk Management & Monitoring
    """
    
    def __init__(self):
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1, 
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5
        }
        
        # Signal Cache for performance
        self.signal_cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # Market Regime Detection
        self.market_regimes = {}
        
        # Correlation Matrix
        self.correlation_pairs = {
            'EURUSD.c': ['GBPUSD.c', 'AUDUSD.c', 'NZDUSD.c'],
            'EURJPY.c': ['GBPJPY.c', 'AUDJPY.c', 'NZDJPY.c'],
            'XAUUSD.c': ['XAGUSD.c'],  # Gold-Silver correlation
            'USDJPY.c': ['USDCHF.c'],  # Safe haven correlation
        }
        
        # News Event Times (UTC)
        self.high_impact_times = [
            '07:00', '08:30',  # London Session
            '12:30', '13:30', '14:30',  # NY Session  
            '15:00', '19:00', '21:00'   # Major News
        ]
        
        print("Enhanced Multi-Timeframe Signal Engine Initialized")
        print("H4 Trend + H1 Setup + M15 Entry + M5 Management")
        print("Signal Confluence System Active")
    
    def get_cached_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached signal to reduce computation"""
        try:
            if symbol in self.signal_cache:
                cache_time, signal_data = self.signal_cache[symbol]
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                    return signal_data
            return None
        except Exception as e:
            print(f"Cache error: {str(e)}")
            return None
    
    def cache_signal(self, symbol: str, signal_data: Dict):
        """Cache signal data"""
        try:
            self.signal_cache[symbol] = (datetime.now(), signal_data)
        except Exception as e:
            print(f"Cache save error: {str(e)}")
    
    def get_timeframe_data(self, symbol: str, timeframe: int, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLC data for specific timeframe"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
            if rates is None or len(rates) < 50:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            print(f"Error getting {symbol} data: {str(e)}")
            return None
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"RSI calculation error: {str(e)}")
            return pd.Series([50.0] * len(close), index=close.index)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            return atr
        except Exception as e:
            print(f"ATR calculation error: {str(e)}")
            return pd.Series([0.001] * len(close), index=close.index)
    
    def calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
        except Exception as e:
            print(f"MACD calculation error: {str(e)}")
            default_series = pd.Series([0.0] * len(close), index=close.index)
            return default_series, default_series, default_series
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            
            # EMA SYSTEM
            ema_9 = close.ewm(span=9, adjust=False).mean()
            ema_21 = close.ewm(span=21, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else ema_50
            
            # RSI MULTI-PERIOD
            rsi_14 = self.calculate_rsi(close, 14)
            rsi_21 = self.calculate_rsi(close, 21)
            
            # ATR & VOLATILITY
            atr_14 = self.calculate_atr(high, low, close, 14)
            atr_21 = self.calculate_atr(high, low, close, 21)
            
            # MACD SYSTEM
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # VOLUME ANALYSIS
            volume_sma = volume.rolling(window=20, min_periods=1).mean()
            volume_ratio = volume / volume_sma
            
            # TREND ANALYSIS
            trend_strength = self.calculate_trend_strength(close, ema_9, ema_21, ema_50)
            trend_direction = self.get_trend_direction(ema_9, ema_21, ema_50, ema_200)
            
            # MOMENTUM
            momentum_10 = close / close.shift(10) - 1
            momentum_20 = close / close.shift(20) - 1
            
            return {
                # EMAs
                'ema_9': float(ema_9.iloc[-1]) if not pd.isna(ema_9.iloc[-1]) else float(close.iloc[-1]),
                'ema_21': float(ema_21.iloc[-1]) if not pd.isna(ema_21.iloc[-1]) else float(close.iloc[-1]),
                'ema_50': float(ema_50.iloc[-1]) if not pd.isna(ema_50.iloc[-1]) else float(close.iloc[-1]),
                'ema_200': float(ema_200.iloc[-1]) if not pd.isna(ema_200.iloc[-1]) else float(close.iloc[-1]),
                
                # RSI
                'rsi_14': float(rsi_14.iloc[-1]) if not pd.isna(rsi_14.iloc[-1]) else 50.0,
                'rsi_21': float(rsi_21.iloc[-1]) if not pd.isna(rsi_21.iloc[-1]) else 50.0,
                
                # ATR & Volatility
                'atr_14': float(atr_14.iloc[-1]) if not pd.isna(atr_14.iloc[-1]) else 0.001,
                'atr_21': float(atr_21.iloc[-1]) if not pd.isna(atr_21.iloc[-1]) else 0.001,
                'atr_percent': float((atr_14.iloc[-1] / close.iloc[-1]) * 100) if not pd.isna(atr_14.iloc[-1]) else 0.1,
                
                # MACD
                'macd_line': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
                'macd_histogram': float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0,
                
                # Volume
                'volume_ratio': float(volume_ratio.iloc[-1]) if not pd.isna(volume_ratio.iloc[-1]) else 1.0,
                
                # Trend
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                
                # Momentum
                'momentum_10': float(momentum_10.iloc[-1]) if not pd.isna(momentum_10.iloc[-1]) else 0,
                'momentum_20': float(momentum_20.iloc[-1]) if not pd.isna(momentum_20.iloc[-1]) else 0,
                
                # Current Price
                'current_price': float(close.iloc[-1])
            }
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return self.get_default_indicators()
    
    def calculate_trend_strength(self, close: pd.Series, ema_9: pd.Series, ema_21: pd.Series, ema_50: pd.Series) -> float:
        """Calculate trend strength (0-1)"""
        try:
            current_price = close.iloc[-1]
            
            # EMA Alignment Score
            ema_conditions = [
                current_price > ema_9.iloc[-1],
                ema_9.iloc[-1] > ema_21.iloc[-1],
                ema_21.iloc[-1] > ema_50.iloc[-1]
            ]
            uptrend_score = sum(ema_conditions) / len(ema_conditions)
            
            ema_conditions_down = [
                current_price < ema_9.iloc[-1],
                ema_9.iloc[-1] < ema_21.iloc[-1],
                ema_21.iloc[-1] < ema_50.iloc[-1]
            ]
            downtrend_score = sum(ema_conditions_down) / len(ema_conditions_down)
            
            return max(uptrend_score, downtrend_score)
            
        except Exception:
            return 0.0
    
    def get_trend_direction(self, ema_9: pd.Series, ema_21: pd.Series, ema_50: pd.Series, ema_200: pd.Series) -> str:
        """Get overall trend direction"""
        try:
            # Current values
            e9 = ema_9.iloc[-1]
            e21 = ema_21.iloc[-1]
            e50 = ema_50.iloc[-1]
            e200 = ema_200.iloc[-1]
            
            # Strong uptrend
            if e9 > e21 > e50 > e200:
                return 'STRONG_UPTREND'
            # Uptrend
            elif e9 > e21 > e50:
                return 'UPTREND'
            # Strong downtrend
            elif e9 < e21 < e50 < e200:
                return 'STRONG_DOWNTREND'
            # Downtrend
            elif e9 < e21 < e50:
                return 'DOWNTREND'
            # Sideways
            else:
                return 'SIDEWAYS'
                
        except Exception:
            return 'UNKNOWN'
    
    def analyze_timeframe_signal(self, symbol: str, timeframe: str, indicators: Dict) -> Dict:
        """Analyze signal for specific timeframe"""
        tf_analysis = {
            'timeframe': timeframe,
            'signal': 'NONE',
            'strength': 0,
            'score': 0,
            'factors': [],
            'trend_bias': 'NEUTRAL'
        }
        
        try:
            # Get indicator values
            current_price = indicators['current_price']
            ema_9 = indicators['ema_9']
            ema_21 = indicators['ema_21']
            ema_50 = indicators['ema_50']
            ema_200 = indicators['ema_200']
            rsi_14 = indicators['rsi_14']
            macd_line = indicators['macd_line']
            macd_signal = indicators['macd_signal']
            macd_histogram = indicators['macd_histogram']
            trend_strength = indicators['trend_strength']
            volume_ratio = indicators['volume_ratio']
            
            # TIMEFRAME-SPECIFIC ANALYSIS
            
            if timeframe == 'H4':
                # H4 = MAIN TREND DIRECTION
                tf_analysis = self.analyze_h4_trend(current_price, ema_9, ema_21, ema_50, ema_200, trend_strength)
                
            elif timeframe == 'H1':
                # H1 = SETUP CONFIRMATION
                tf_analysis = self.analyze_h1_setup(current_price, ema_9, ema_21, rsi_14, macd_line, macd_signal, volume_ratio)
                
            elif timeframe == 'M15':
                # M15 = ENTRY TIMING
                tf_analysis = self.analyze_m15_entry(current_price, ema_9, ema_21, rsi_14, macd_histogram, volume_ratio)
                
            elif timeframe == 'M5':
                # M5 = CONFIRMATION & RISK MANAGEMENT
                tf_analysis = self.analyze_m5_confirmation(current_price, ema_9, rsi_14, macd_histogram)
            
            return tf_analysis
            
        except Exception as e:
            print(f"Error analyzing {timeframe} signal: {str(e)}")
            return tf_analysis
    
    def analyze_h4_trend(self, price: float, ema_9: float, ema_21: float, ema_50: float, ema_200: float, trend_strength: float) -> Dict:
        """H4 = Main Trend Direction Analysis"""
        score = 0
        factors = []
        signal = 'NONE'
        trend_bias = 'NEUTRAL'
        
        # MAJOR TREND CONDITIONS
        
        # EMA Alignment (important for H4)
        if price > ema_9 > ema_21 > ema_50:
            score += 4
            factors.append("Strong Bullish EMA Alignment")
            trend_bias = 'BULLISH'
        elif price > ema_9 > ema_21:
            score += 2
            factors.append("Bullish EMA Short-term")
            trend_bias = 'BULLISH'
        elif price < ema_9 < ema_21 < ema_50:
            score -= 4
            factors.append("Strong Bearish EMA Alignment")
            trend_bias = 'BEARISH'
        elif price < ema_9 < ema_21:
            score -= 2
            factors.append("Bearish EMA Short-term")
            trend_bias = 'BEARISH'
        
        # Long-term trend (EMA 200)
        if price > ema_200:
            score += 1
            factors.append("Above EMA 200")
        elif price < ema_200:
            score -= 1
            factors.append("Below EMA 200")
        
        # Trend Strength
        if trend_strength >= 0.8:
            score += 2
            factors.append("Very Strong Trend")
        elif trend_strength >= 0.6:
            score += 1
            factors.append("Strong Trend")
        
        # SIGNAL DETERMINATION
        if score >= 4:
            signal = 'STRONG_BUY' if trend_bias == 'BULLISH' else 'STRONG_SELL'
        elif score >= 2:
            signal = 'BUY' if trend_bias == 'BULLISH' else 'SELL'
        elif score <= -4:
            signal = 'STRONG_SELL'
        elif score <= -2:
            signal = 'SELL'
        
        strength = min(10, abs(score) * 1.5)
        
        return {
            'timeframe': 'H4',
            'signal': signal,
            'strength': strength,
            'score': score,
            'factors': factors,
            'trend_bias': trend_bias
        }
    
    def analyze_h1_setup(self, price: float, ema_9: float, ema_21: float, rsi_14: float, 
                        macd_line: float, macd_signal: float, volume_ratio: float) -> Dict:
        """H1 = Setup Confirmation Analysis"""
        score = 0
        factors = []
        signal = 'NONE'
        
        # SETUP CONDITIONS
        
        # EMA Setup
        if price > ema_9 > ema_21:
            score += 2
            factors.append("Bullish EMA Setup")
        elif price < ema_9 < ema_21:
            score -= 2
            factors.append("Bearish EMA Setup")
        
        # RSI Conditions
        if 30 <= rsi_14 <= 70:
            score += 1
            factors.append("RSI in safe zone")
        if 40 <= rsi_14 <= 60:
            score += 1
            factors.append("RSI optimal zone")
        
        # MACD Confirmation
        if macd_line > macd_signal and macd_line > 0:
            score += 2
            factors.append("MACD Bullish")
        elif macd_line < macd_signal and macd_line < 0:
            score -= 2
            factors.append("MACD Bearish")
        elif macd_line > macd_signal:
            score += 1
            factors.append("MACD Momentum Up")
        elif macd_line < macd_signal:
            score -= 1
            factors.append("MACD Momentum Down")
        
        # Volume Confirmation
        if volume_ratio >= 1.2:
            score += 1
            factors.append("High Volume")
        
        # SIGNAL DETERMINATION
        if score >= 5:
            signal = 'STRONG_BUY'
        elif score >= 3:
            signal = 'BUY'
        elif score <= -5:
            signal = 'STRONG_SELL'
        elif score <= -3:
            signal = 'SELL'
        
        strength = min(10, abs(score) * 1.2)
        
        return {
            'timeframe': 'H1',
            'signal': signal,
            'strength': strength,
            'score': score,
            'factors': factors,
            'trend_bias': 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'
        }
    
    def analyze_m15_entry(self, price: float, ema_9: float, ema_21: float, rsi_14: float, 
                         macd_histogram: float, volume_ratio: float) -> Dict:
        """M15 = Entry Timing Analysis"""
        score = 0
        factors = []
        signal = 'NONE'
        
        # ENTRY TIMING CONDITIONS
        
        # Price vs EMA
        if price > ema_9:
            score += 1
            factors.append("Price above EMA9")
        elif price < ema_9:
            score -= 1
            factors.append("Price below EMA9")
        
        # EMA Direction
        if ema_9 > ema_21:
            score += 1
            factors.append("EMA9 > EMA21")
        elif ema_9 < ema_21:
            score -= 1
            factors.append("EMA9 < EMA21")
        
        # RSI Entry Conditions
        if 25 <= rsi_14 <= 35:  # Oversold but not extreme
            score += 2
            factors.append("RSI Oversold Entry")
        elif 65 <= rsi_14 <= 75:  # Overbought but not extreme
            score -= 2
            factors.append("RSI Overbought Entry")
        elif 45 <= rsi_14 <= 55:  # Neutral zone
            score += 1
            factors.append("RSI Neutral")
        
        # MACD Histogram (Momentum)
        if macd_histogram > 0:
            score += 1
            factors.append("Positive Momentum")
        elif macd_histogram < 0:
            score -= 1
            factors.append("Negative Momentum")
        
        # Volume
        if volume_ratio >= 1.1:
            score += 1
            factors.append("Volume Support")
        
        # SIGNAL DETERMINATION
        if score >= 4:
            signal = 'BUY'
        elif score >= 2:
            signal = 'WEAK_BUY'
        elif score <= -4:
            signal = 'SELL'
        elif score <= -2:
            signal = 'WEAK_SELL'
        
        strength = min(10, abs(score) * 2)
        
        return {
            'timeframe': 'M15',
            'signal': signal,
            'strength': strength,
            'score': score,
            'factors': factors,
            'trend_bias': 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'
        }
    
    def analyze_m5_confirmation(self, price: float, ema_9: float, rsi_14: float, macd_histogram: float) -> Dict:
        """M5 = Final Confirmation & Risk Management"""
        score = 0
        factors = []
        signal = 'NONE'
        
        # CONFIRMATION CONDITIONS
        
        # Immediate price action
        if price > ema_9:
            score += 1
            factors.append("Immediate bullish bias")
        elif price < ema_9:
            score -= 1
            factors.append("Immediate bearish bias")
        
        # RSI momentum
        if 30 <= rsi_14 <= 70:
            score += 1
            factors.append("RSI safe for entry")
        
        # MACD momentum confirmation
        if macd_histogram > 0:
            score += 1
            factors.append("Momentum confirmation")
        elif macd_histogram < 0:
            score -= 1
            factors.append("Momentum divergence")
        
        # SIGNAL DETERMINATION
        if score >= 2:
            signal = 'CONFIRM_BUY'
        elif score <= -2:
            signal = 'CONFIRM_SELL'
        else:
            signal = 'WAIT'
        
        strength = min(10, abs(score) * 3)
        
        return {
            'timeframe': 'M5',
            'signal': signal,
            'strength': strength,
            'score': score,
            'factors': factors,
            'trend_bias': 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'
        }
    
    def get_multi_timeframe_confluence(self, symbol: str) -> Dict:
        """
        CORE FUNCTION: Multi-Timeframe Signal Confluence
        This is where the magic happens!
        """
        
        # Check cache first
        cached_signal = self.get_cached_signal(symbol)
        if cached_signal:
            return cached_signal
        
        # Initialize result
        confluence_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'final_signal': 'NONE',
            'final_strength': 0,
            'final_quality': 'POOR',
            'confluence_score': 0,
            'timeframe_analysis': {},
            'risk_factors': [],
            'entry_conditions': {},
            'trade_recommendation': 'NO_TRADE'
        }
        
        try:
            # GET DATA FOR ALL TIMEFRAMES
            timeframe_data = {}
            timeframe_indicators = {}
            
            for tf_name, tf_value in self.timeframes.items():
                df = self.get_timeframe_data(symbol, tf_value, 100)
                if df is not None:
                    timeframe_data[tf_name] = df
                    timeframe_indicators[tf_name] = self.calculate_advanced_indicators(df)
            
            if len(timeframe_data) < 3:  # Need at least 3 timeframes
                confluence_result['risk_factors'].append('Insufficient timeframe data')
                return confluence_result
            
            # ANALYZE EACH TIMEFRAME
            for tf_name in ['H4', 'H1', 'M15', 'M5']:
                if tf_name in timeframe_indicators:
                    tf_analysis = self.analyze_timeframe_signal(symbol, tf_name, timeframe_indicators[tf_name])
                    confluence_result['timeframe_analysis'][tf_name] = tf_analysis
            
            # CONFLUENCE CALCULATION
            confluence_score = 0
            bullish_votes = 0
            bearish_votes = 0
            
            # H4 = 40% weight (Most Important)
            if 'H4' in confluence_result['timeframe_analysis']:
                h4_analysis = confluence_result['timeframe_analysis']['H4']
                if h4_analysis['signal'] in ['STRONG_BUY', 'BUY']:
                    confluence_score += 4
                    bullish_votes += 4
                elif h4_analysis['signal'] in ['STRONG_SELL', 'SELL']:
                    confluence_score -= 4
                    bearish_votes += 4
            
            # H1 = 30% weight
            if 'H1' in confluence_result['timeframe_analysis']:
                h1_analysis = confluence_result['timeframe_analysis']['H1']
                if h1_analysis['signal'] in ['STRONG_BUY', 'BUY']:
                    confluence_score += 3
                    bullish_votes += 3
                elif h1_analysis['signal'] in ['STRONG_SELL', 'SELL']:
                    confluence_score -= 3
                    bearish_votes += 3
            
            # M15 = 20% weight
            if 'M15' in confluence_result['timeframe_analysis']:
                m15_analysis = confluence_result['timeframe_analysis']['M15']
                if m15_analysis['signal'] in ['BUY', 'WEAK_BUY']:
                    confluence_score += 2
                    bullish_votes += 2
                elif m15_analysis['signal'] in ['SELL', 'WEAK_SELL']:
                    confluence_score -= 2
                    bearish_votes += 2
            
            # M5 = 10% weight (Confirmation only)
            if 'M5' in confluence_result['timeframe_analysis']:
                m5_analysis = confluence_result['timeframe_analysis']['M5']
                if m5_analysis['signal'] == 'CONFIRM_BUY':
                    confluence_score += 1
                    bullish_votes += 1
                elif m5_analysis['signal'] == 'CONFIRM_SELL':
                    confluence_score -= 1
                    bearish_votes += 1
            
            # FINAL SIGNAL DETERMINATION
            confluence_result['confluence_score'] = confluence_score
            
            if confluence_score >= 6:  # Strong confluence
                confluence_result['final_signal'] = 'STRONG_BUY'
                confluence_result['final_strength'] = min(10, 6 + (confluence_score - 6) * 0.5)
                confluence_result['final_quality'] = 'EXCELLENT'
                confluence_result['trade_recommendation'] = 'STRONG_BUY'
                
            elif confluence_score >= 4:  # Good confluence
                confluence_result['final_signal'] = 'BUY'
                confluence_result['final_strength'] = min(10, 4 + (confluence_score - 4) * 0.8)
                confluence_result['final_quality'] = 'GOOD'
                confluence_result['trade_recommendation'] = 'BUY'
                
            elif confluence_score >= 2:  # Weak confluence
                confluence_result['final_signal'] = 'WEAK_BUY'
                confluence_result['final_strength'] = min(10, 2 + (confluence_score - 2) * 1.0)
                confluence_result['final_quality'] = 'FAIR'
                confluence_result['trade_recommendation'] = 'CONSIDER_BUY'
                
            elif confluence_score <= -6:  # Strong bearish confluence
                confluence_result['final_signal'] = 'STRONG_SELL'
                confluence_result['final_strength'] = min(10, 6 + abs(confluence_score + 6) * 0.5)
                confluence_result['final_quality'] = 'EXCELLENT'
                confluence_result['trade_recommendation'] = 'STRONG_SELL'
                
            elif confluence_score <= -4:  # Good bearish confluence
                confluence_result['final_signal'] = 'SELL'
                confluence_result['final_strength'] = min(10, 4 + abs(confluence_score + 4) * 0.8)
                confluence_result['final_quality'] = 'GOOD'
                confluence_result['trade_recommendation'] = 'SELL'
                
            elif confluence_score <= -2:  # Weak bearish confluence
                confluence_result['final_signal'] = 'WEAK_SELL'
                confluence_result['final_strength'] = min(10, 2 + abs(confluence_score + 2) * 1.0)
                confluence_result['final_quality'] = 'FAIR'
                confluence_result['trade_recommendation'] = 'CONSIDER_SELL'
            
            # ENTRY CONDITIONS (use H1 and M15)
            if 'H1' in timeframe_indicators:
                h1_data = timeframe_indicators['H1']
                current_price = h1_data['current_price']
                atr = h1_data['atr_14']
                
                if confluence_result['final_signal'] in ['STRONG_BUY', 'BUY', 'WEAK_BUY']:
                    confluence_result['entry_conditions'] = {
                        'optimal_entry': round(current_price, 5),
                        'stop_loss': round(current_price - (atr * 1.5), 5),
                        'take_profit_1': round(current_price + (atr * 2.0), 5),
                        'take_profit_2': round(current_price + (atr * 3.5), 5),
                        'take_profit_3': round(current_price + (atr * 5.0), 5),
                        'risk_reward_tp1': round((atr * 2.0) / (atr * 1.5), 2),
                        'risk_reward_tp2': round((atr * 3.5) / (atr * 1.5), 2),
                        'risk_reward_tp3': round((atr * 5.0) / (atr * 1.5), 2)
                    }
                elif confluence_result['final_signal'] in ['STRONG_SELL', 'SELL', 'WEAK_SELL']:
                    confluence_result['entry_conditions'] = {
                        'optimal_entry': round(current_price, 5),
                        'stop_loss': round(current_price + (atr * 1.5), 5),
                        'take_profit_1': round(current_price - (atr * 2.0), 5),
                        'take_profit_2': round(current_price - (atr * 3.5), 5),
                        'take_profit_3': round(current_price - (atr * 5.0), 5),
                        'risk_reward_tp1': round((atr * 2.0) / (atr * 1.5), 2),
                        'risk_reward_tp2': round((atr * 3.5) / (atr * 1.5), 2),
                        'risk_reward_tp3': round((atr * 5.0) / (atr * 1.5), 2)
                    }
            
            # Cache the result
            self.cache_signal(symbol, confluence_result)
            
            return confluence_result
            
        except Exception as e:
            print(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            confluence_result['risk_factors'].append(f'Analysis error: {str(e)}')
            return confluence_result
    
    def check_correlation_risk(self, symbol: str, existing_positions: List[str]) -> bool:
        """Check if new symbol conflicts with existing positions"""
        try:
            if symbol not in self.correlation_pairs:
                return True  # No known correlations
            
            correlated_symbols = self.correlation_pairs[symbol]
            
            for existing_symbol in existing_positions:
                if existing_symbol in correlated_symbols:
                    return False  # High correlation risk
            
            return True
        except Exception as e:
            print(f"Correlation check error: {str(e)}")
            return True  # Default to allow trading
    
    def check_news_filter(self, symbol: str) -> bool:
        """Check if current time is safe for trading (avoid news events)"""
        try:
            current_time = datetime.now().strftime('%H:%M')
            
            for news_time in self.high_impact_times:
                news_dt = datetime.strptime(news_time, '%H:%M')
                current_dt = datetime.strptime(current_time, '%H:%M')
                time_diff = abs((current_dt - news_dt).total_seconds() / 60)
                
                if time_diff <= 15:  # Within 15 minutes of news
                    return False
            
            return True
        except Exception as e:
            print(f"News filter error: {str(e)}")
            return True  # Default to allow trading
    
    def get_default_indicators(self) -> Dict:
        """Return default indicators when calculation fails"""
        return {
            'ema_9': 1.0, 'ema_21': 1.0, 'ema_50': 1.0, 'ema_200': 1.0,
            'rsi_14': 50.0, 'rsi_21': 50.0,
            'atr_14': 0.001, 'atr_21': 0.001, 'atr_percent': 0.1,
            'macd_line': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'volume_ratio': 1.0, 'trend_strength': 0.0, 'trend_direction': 'UNKNOWN',
            'momentum_10': 0.0, 'momentum_20': 0.0,
            'current_price': 1.0
        }

print("Enhanced Signal System - Windows Compatible Version")
print("All emoji characters removed for Windows compatibility")
print("Ready for Multi-Timeframe Analysis")