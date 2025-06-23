"""
🎯 PROFESSIONAL INDICATOR-BASED TRADING AI
==========================================
Advanced Trading System with MT5 Integration
Multi-Timeframe Analysis & Professional Risk Management
"""

from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ProfessionalTradingAI:
    """Professional Indicator-Based Trading AI System"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize Professional Trading AI"""
        self.symbol = symbol
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Trading Configuration
        self.max_risk_per_trade = 0.02  # 2%
        self.max_total_exposure = 0.06  # 6%
        self.min_reward_ratio = 2.0     # 1:2 minimum R/R
        
        # Indicator Parameters (Professional Grade)
        self.ema_periods = [9, 21, 50, 200]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
        self.stoch_k = 14
        self.stoch_d = 3
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Signal Scoring Weights
        self.weights = {
            'trend': 0.30,      # 30% - Most important
            'momentum': 0.25,   # 25%
            'volume': 0.20,     # 20%
            'confluence': 0.15, # 15%
            'risk_reward': 0.10 # 10%
        }
        
        # Timeframes for Multi-TF Analysis
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        
        # Data Storage
        self.market_data = {}
        self.signals = {}
        self.performance_stats = {
            'signals_generated': 0,
            'high_quality_signals': 0,
            'filtered_signals': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        self.is_running = False
        self.last_update = datetime.now()
        
        self.setup_logging()
        self.setup_routes()
    
    def setup_logging(self):
        """Setup professional logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('professional_trading_ai.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 with professional validation"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Validate account connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            # Validate symbol
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to select symbol {self.symbol}")
                    return False
            
            self.logger.info(f"✅ MT5 Connected - Account: {account_info.login}")
            self.logger.info(f"✅ Symbol: {self.symbol} - Spread: {symbol_info.spread}")
            self.logger.info(f"✅ Balance: ${account_info.balance:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def get_market_data(self, timeframe: int, bars: int = 200) -> Optional[pd.DataFrame]:
        """Get market data for specific timeframe"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # Price data
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            
            # EMAs
            for period in self.ema_periods:
                indicators[f'ema_{period}'] = self.calculate_ema(close, period).iloc[-1]
            
            # RSI
            rsi = self.calculate_rsi(close, self.rsi_period)
            indicators['rsi'] = rsi.iloc[-1]
            
            # MACD
            macd_data = self.calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
            indicators['macd'] = macd_data['macd'].iloc[-1]
            indicators['macd_signal'] = macd_data['signal'].iloc[-1]
            indicators['macd_histogram'] = macd_data['histogram'].iloc[-1]
            
            # ATR
            atr = self.calculate_atr(high, low, close, self.atr_period)
            indicators['atr'] = atr.iloc[-1]
            indicators['atr_percent'] = (atr.iloc[-1] / close.iloc[-1]) * 100
            
            # Stochastic
            stoch_data = self.calculate_stochastic(high, low, close, self.stoch_k, self.stoch_d)
            indicators['stoch_k'] = stoch_data['k_percent'].iloc[-1]
            indicators['stoch_d'] = stoch_data['d_percent'].iloc[-1]
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(close, self.bb_period, self.bb_std)
            indicators['bb_upper'] = bb_data['upper'].iloc[-1]
            indicators['bb_middle'] = bb_data['middle'].iloc[-1]
            indicators['bb_lower'] = bb_data['lower'].iloc[-1]
            
            # Volume Analysis
            volume_sma = volume.rolling(window=20).mean()
            indicators['volume_ratio'] = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
            
            # Current Price
            indicators['current_price'] = close.iloc[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def analyze_trend_strength(self, indicators: Dict) -> float:
        """Analyze trend strength (0-1 scale)"""
        try:
            current_price = indicators.get('current_price', 0)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            ema_200 = indicators.get('ema_200', current_price)
            
            # Bullish conditions
            bullish_conditions = [
                current_price > ema_9,
                ema_9 > ema_21,
                ema_21 > ema_50,
                ema_50 > ema_200,
                indicators.get('macd', 0) > indicators.get('macd_signal', 0)
            ]
            
            # Bearish conditions
            bearish_conditions = [
                current_price < ema_9,
                ema_9 < ema_21,
                ema_21 < ema_50,
                ema_50 < ema_200,
                indicators.get('macd', 0) < indicators.get('macd_signal', 0)
            ]
            
            bullish_strength = sum(bullish_conditions) / len(bullish_conditions)
            bearish_strength = sum(bearish_conditions) / len(bearish_conditions)
            
            # Return the stronger trend (0.5 = neutral, 0 = strong bearish, 1 = strong bullish)
            if bullish_strength > bearish_strength:
                return 0.5 + (bullish_strength * 0.5)
            else:
                return 0.5 - (bearish_strength * 0.5)
                
        except Exception as e:
            self.logger.error(f"Error analyzing trend strength: {str(e)}")
            return 0.5
    
    def analyze_momentum(self, indicators: Dict) -> float:
        """Analyze momentum strength (0-1 scale)"""
        try:
            rsi = indicators.get('rsi', 50)
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            # RSI momentum (optimal range 40-60 for trending)
            rsi_momentum = 1.0 - abs(rsi - 50) / 50
            
            # Stochastic momentum
            stoch_momentum = (stoch_k + stoch_d) / 200  # 0-1 scale
            
            # MACD momentum
            macd_momentum = 0.5 + (macd_histogram * 1000)  # Normalized
            macd_momentum = max(0, min(1, macd_momentum))
            
            return (rsi_momentum + stoch_momentum + macd_momentum) / 3
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {str(e)}")
            return 0.5
    
    def analyze_volume_confirmation(self, indicators: Dict) -> float:
        """Analyze volume confirmation (0-1 scale)"""
        try:
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # Volume confirmation score
            if volume_ratio >= 1.5:
                return 1.0  # Strong volume
            elif volume_ratio >= 1.2:
                return 0.8  # Good volume
            elif volume_ratio >= 1.0:
                return 0.6  # Normal volume
            else:
                return 0.3  # Low volume
                
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            return 0.5
    
    def check_confluence(self, indicators: Dict) -> float:
        """Check indicator confluence (0-1 scale)"""
        try:
            confluence_score = 0.0
            max_score = 0.0
            
            # RSI confluence
            rsi = indicators.get('rsi', 50)
            if 35 <= rsi <= 65:  # Good trading range
                confluence_score += 0.2
            max_score += 0.2
            
            # MACD confluence
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if abs(macd - macd_signal) > 0.0001:  # Clear signal
                confluence_score += 0.2
            max_score += 0.2
            
            # Bollinger Bands confluence
            current_price = indicators.get('current_price', 0)
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            bb_middle = indicators.get('bb_middle', current_price)
            
            # Price position relative to BB
            if bb_lower < current_price < bb_upper:
                confluence_score += 0.2
            max_score += 0.2
            
            # ATR confluence (volatility check)
            atr_percent = indicators.get('atr_percent', 1.0)
            if 0.5 <= atr_percent <= 3.0:  # Normal volatility for Gold
                confluence_score += 0.2
            max_score += 0.2
            
            # Stochastic confluence
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            if 20 <= stoch_k <= 80 and 20 <= stoch_d <= 80:
                confluence_score += 0.2
            max_score += 0.2
            
            return confluence_score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error checking confluence: {str(e)}")
            return 0.0
    
    def calculate_risk_reward(self, entry_price: float, signal_direction: str, indicators: Dict) -> Tuple[float, float, List[float]]:
        """Calculate risk/reward levels"""
        try:
            atr = indicators.get('atr', entry_price * 0.01)
            
            if signal_direction.upper() == 'BUY':
                stop_loss = entry_price - (atr * 1.5)
                take_profit_1 = entry_price + (atr * 2.5)  # 1:1.67 R/R
                take_profit_2 = entry_price + (atr * 3.75) # 1:2.5 R/R
                take_profit_3 = entry_price + (atr * 6.0)  # 1:4.0 R/R
            else:  # SELL
                stop_loss = entry_price + (atr * 1.5)
                take_profit_1 = entry_price - (atr * 2.5)
                take_profit_2 = entry_price - (atr * 3.75)
                take_profit_3 = entry_price - (atr * 6.0)
            
            # Calculate R/R ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit_1 - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            return rr_ratio, stop_loss, [take_profit_1, take_profit_2, take_profit_3]
            
        except Exception as e:
            self.logger.error(f"Error calculating risk/reward: {str(e)}")
            return 0.0, entry_price, [entry_price, entry_price, entry_price]
    
    def generate_signal(self, all_timeframes_data: Dict) -> Dict:
        """Generate professional trading signal"""
        try:
            # Use H1 as primary timeframe for signal generation
            h1_indicators = all_timeframes_data.get('H1', {})
            if not h1_indicators:
                return self.create_no_signal()
            
            # Calculate component scores
            trend_score = self.analyze_trend_strength(h1_indicators)
            momentum_score = self.analyze_momentum(h1_indicators)
            volume_score = self.analyze_volume_confirmation(h1_indicators)
            confluence_score = self.check_confluence(h1_indicators)
            
            # Determine signal direction
            signal_direction = 'NONE'
            current_price = h1_indicators.get('current_price', 0)
            
            # Signal logic based on trend and confluence
            if trend_score > 0.7 and confluence_score > 0.6:
                if (h1_indicators.get('macd', 0) > h1_indicators.get('macd_signal', 0) and
                    h1_indicators.get('rsi', 50) > 45):
                    signal_direction = 'BUY'
                    
            elif trend_score < 0.3 and confluence_score > 0.6:
                if (h1_indicators.get('macd', 0) < h1_indicators.get('macd_signal', 0) and
                    h1_indicators.get('rsi', 50) < 55):
                    signal_direction = 'SELL'
            
            # Calculate risk/reward if signal exists
            rr_ratio = 0.0
            stop_loss = current_price
            take_profits = [current_price, current_price, current_price]
            
            if signal_direction != 'NONE':
                rr_ratio, stop_loss, take_profits = self.calculate_risk_reward(
                    current_price, signal_direction, h1_indicators
                )
                
                # Filter by minimum R/R ratio
                if rr_ratio < self.min_reward_ratio:
                    signal_direction = 'NONE'
                    rr_ratio = 0.0
            
            # Calculate RR score
            rr_score = min(1.0, rr_ratio / 3.0) if rr_ratio > 0 else 0.0
            
            # Calculate final signal strength
            signal_strength = (
                trend_score * self.weights['trend'] +
                momentum_score * self.weights['momentum'] +
                volume_score * self.weights['volume'] +
                confluence_score * self.weights['confluence'] +
                rr_score * self.weights['risk_reward']
            ) * 10  # Scale to 0-10
            
            # Determine confidence level
            if signal_strength >= 8.5 and signal_direction != 'NONE':
                confidence = 'VERY_HIGH'
            elif signal_strength >= 7.5 and signal_direction != 'NONE':
                confidence = 'HIGH'
            elif signal_strength >= 6.5 and signal_direction != 'NONE':
                confidence = 'MEDIUM'
            elif signal_strength >= 5.0:
                confidence = 'LOW'
            else:
                confidence = 'FILTERED'
                signal_direction = 'NONE'
            
            return {
                'direction': signal_direction,
                'strength': round(signal_strength, 2),
                'confidence': confidence,
                'trend_score': round(trend_score, 3),
                'momentum_score': round(momentum_score, 3),
                'volume_score': round(volume_score, 3),
                'confluence_score': round(confluence_score, 3),
                'rr_ratio': round(rr_ratio, 2),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profits[0], 5),
                'take_profit_2': round(take_profits[1], 5),
                'take_profit_3': round(take_profits[2], 5),
                'entry_price': round(current_price, 5),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return self.create_no_signal()
    
    def create_no_signal(self) -> Dict:
        """Create empty signal structure"""
        return {
            'direction': 'NONE',
            'strength': 0.0,
            'confidence': 'FILTERED',
            'trend_score': 0.0,
            'momentum_score': 0.0,
            'volume_score': 0.0,
            'confluence_score': 0.0,
            'rr_ratio': 0.0,
            'stop_loss': 0.0,
            'take_profit_1': 0.0,
            'take_profit_2': 0.0,
            'take_profit_3': 0.0,
            'entry_price': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_all_timeframes(self) -> Dict:
        """Analyze all timeframes and generate comprehensive signal"""
        try:
            all_data = {}
            
            # Get data for all timeframes
            for tf_name, tf_value in self.timeframes.items():
                df = self.get_market_data(tf_value, 200)
                if df is not None and len(df) >= 50:
                    indicators = self.calculate_all_indicators(df)
                    all_data[tf_name] = indicators
                    
                    self.logger.info(f"✅ {tf_name}: Price={indicators.get('current_price', 0):.5f}, RSI={indicators.get('rsi', 0):.1f}")
            
            # Generate signal from multi-timeframe analysis
            signal = self.generate_signal(all_data)
            
            # Update performance stats
            self.performance_stats['signals_generated'] += 1
            if signal['confidence'] in ['HIGH', 'VERY_HIGH']:
                self.performance_stats['high_quality_signals'] += 1
            elif signal['confidence'] == 'FILTERED':
                self.performance_stats['filtered_signals'] += 1
            
            # Store results
            self.market_data = all_data
            self.signals = signal
            self.last_update = datetime.now()
            
            return {
                'timeframes': all_data,
                'signal': signal,
                'performance': self.performance_stats,
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframes: {str(e)}")
            return {}
    
    def setup_routes(self):
        """Setup Flask routes for web interface"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/analysis')
        def get_analysis():
            """API endpoint for complete analysis"""
            return jsonify(self.analyze_all_timeframes())
        
        @self.app.route('/api/signal')
        def get_signal():
            """API endpoint for current signal"""
            return jsonify(self.signals)
        
        @self.app.route('/api/performance')
        def get_performance():
            """API endpoint for performance stats"""
            return jsonify(self.performance_stats)
    
    def get_dashboard_html(self) -> str:
        """Professional Trading Dashboard HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Professional Trading AI - XAUUSD</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', 'Consolas', monospace;
            background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #2d1b4e 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
            border-bottom: 3px solid #00ff88;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.3);
        }
        
        .header h1 {
            font-size: 2.5rem;
            color: #00ff88;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .header p {
            font-size: 1.2rem;
            color: #ffffff;
            opacity: 0.9;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            padding: 2rem;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .signal-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            transition: transform 0.3s ease;
        }
        
        .signal-panel:hover {
            transform: translateY(-5px);
        }
        
        .signal-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .signal-direction {
            font-size: 3rem;
            font-weight: bold;
            margin: 1rem 0;
            padding: 1rem 2rem;
            border-radius: 15px;
            text-transform: uppercase;
        }
        
        .signal-buy {
            background: linear-gradient(135deg, #00ff88, #00d46a);
            color: #000;
            animation: pulse-green 2s infinite;
        }
        
        .signal-sell {
            background: linear-gradient(135deg, #ff4757, #ff3838);
            color: #fff;
            animation: pulse-red 2s infinite;
        }
        
        .signal-none {
            background: rgba(255, 255, 255, 0.1);
            color: #ccc;
        }
        
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.5); }
            50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.8); }
        }
        
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 71, 87, 0.5); }
            50% { box-shadow: 0 0 40px rgba(255, 71, 87, 0.8); }
        }
        
        .strength-meter {
            margin: 2rem 0;
        }
        
        .strength-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .strength-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff4757, #ffff00, #00ff88);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .timeframes-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .timeframe-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        
        .timeframe-name {
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 0.5rem;
        }
        
        .timeframe-price {
            font-size: 1.1rem;
            color: #fff;
        }
        
        .indicators-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .indicator-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .indicator-name {
            font-size: 0.9rem;
            color: #aaa;
            margin-bottom: 0.5rem;
        }
        
        .indicator-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #00ff88;
        }
        
        .trading-levels {
            background: rgba(0, 123, 255, 0.1);
            border: 2px solid rgba(0, 123, 255, 0.3);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
        }
        
        .level-row {
            display: flex;
            justify-content: space-between;
            padding: 0.8rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .level-label {
            color: #aaa;
        }
        
        .level-value {
            color: #00ff88;
            font-weight: bold;
            font-family: 'Consolas', monospace;
        }
        
        .performance-stats {
            grid-column: 1 / -1;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 2rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
        }
        
        .stat-card {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #00ff88;
        }
        
        .stat-label {
            color: #aaa;
            margin-top: 0.5rem;
        }
        
        .loading {
            text-align: center;
            padding: 3rem;
            font-size: 1.2rem;
            color: #00ff88;
        }
        
        .loading::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            animation: blink 1s infinite;
            margin-left: 10px;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Professional Trading AI</h1>
        <p>XAUUSD Multi-Timeframe Analysis • Indicator-Based Signals <span class="status-indicator"></span></p>
    </div>
    
    <div class="main-container">
        <!-- Current Signal Panel -->
        <div class="signal-panel">
            <div class="signal-header">
                <h2>📊 Current Signal</h2>
            </div>
            
            <div id="signalDirection" class="signal-direction signal-none">
                Loading...
            </div>
            
            <div class="strength-meter">
                <h3>Signal Strength</h3>
                <div class="strength-bar">
                    <div id="strengthFill" class="strength-fill" style="width: 0%"></div>
                </div>
                <div style="text-align: center; margin-top: 0.5rem;">
                    <span id="strengthValue">0.0</span>/10 • <span id="confidenceLevel">-</span>
                </div>
            </div>
            
            <div class="indicators-grid">
                <div class="indicator-card">
                    <div class="indicator-name">Trend Score</div>
                    <div id="trendScore" class="indicator-value">0.0</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">Momentum</div>
                    <div id="momentumScore" class="indicator-value">0.0</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">Volume</div>
                    <div id="volumeScore" class="indicator-value">0.0</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">Confluence</div>
                    <div id="confluenceScore" class="indicator-value">0.0</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">R/R Ratio</div>
                    <div id="rrRatio" class="indicator-value">0.0</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">Entry Price</div>
                    <div id="entryPrice" class="indicator-value">0.0</div>
                </div>
            </div>
        </div>
        
        <!-- Trading Levels Panel -->
        <div class="signal-panel">
            <div class="signal-header">
                <h2>🎯 Trading Levels</h2>
            </div>
            
            <div class="trading-levels">
                <div class="level-row">
                    <span class="level-label">Entry Price:</span>
                    <span id="levelEntry" class="level-value">-</span>
                </div>
                <div class="level-row">
                    <span class="level-label">Stop Loss:</span>
                    <span id="levelStopLoss" class="level-value">-</span>
                </div>
                <div class="level-row">
                    <span class="level-label">Take Profit 1:</span>
                    <span id="levelTP1" class="level-value">-</span>
                </div>
                <div class="level-row">
                    <span class="level-label">Take Profit 2:</span>
                    <span id="levelTP2" class="level-value">-</span>
                </div>
                <div class="level-row">
                    <span class="level-label">Take Profit 3:</span>
                    <span id="levelTP3" class="level-value">-</span>
                </div>
            </div>
            
            <div class="timeframes-grid">
                <div class="timeframe-card">
                    <div class="timeframe-name">H4</div>
                    <div id="priceH4" class="timeframe-price">-</div>
                </div>
                <div class="timeframe-card">
                    <div class="timeframe-name">H1</div>
                    <div id="priceH1" class="timeframe-price">-</div>
                </div>
                <div class="timeframe-card">
                    <div class="timeframe-name">M15</div>
                    <div id="priceM15" class="timeframe-price">-</div>
                </div>
                <div class="timeframe-card">
                    <div class="timeframe-name">M5</div>
                    <div id="priceM5" class="timeframe-price">-</div>
                </div>
                <div class="timeframe-card">
                    <div class="timeframe-name">M1</div>
                    <div id="priceM1" class="timeframe-price">-</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Performance Statistics -->
    <div class="performance-stats">
        <h2 style="text-align: center; margin-bottom: 2rem;">📈 Performance Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div id="totalSignals" class="stat-value">0</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-card">
                <div id="highQualitySignals" class="stat-value">0</div>
                <div class="stat-label">High Quality</div>
            </div>
            <div class="stat-card">
                <div id="filteredSignals" class="stat-value">0</div>
                <div class="stat-label">Filtered Signals</div>
            </div>
            <div class="stat-card">
                <div id="lastUpdate" class="stat-value">-</div>
                <div class="stat-label">Last Update</div>
            </div>
        </div>
    </div>

    <script>
        async function loadAnalysis() {
            try {
                const response = await fetch('/api/analysis');
                const data = await response.json();
                
                updateSignalDisplay(data.signal || {});
                updateTimeframesPrices(data.timeframes || {});
                updatePerformanceStats(data.performance || {});
                
            } catch (error) {
                console.error('Error loading analysis:', error);
            }
        }
        
        function updateSignalDisplay(signal) {
            // Signal Direction
            const directionEl = document.getElementById('signalDirection');
            directionEl.textContent = signal.direction || 'NONE';
            directionEl.className = `signal-direction signal-${(signal.direction || 'none').toLowerCase()}`;
            
            // Signal Strength
            const strength = signal.strength || 0;
            document.getElementById('strengthFill').style.width = `${Math.min(100, strength * 10)}%`;
            document.getElementById('strengthValue').textContent = strength.toFixed(1);
            document.getElementById('confidenceLevel').textContent = signal.confidence || '-';
            
            // Component Scores
            document.getElementById('trendScore').textContent = (signal.trend_score || 0).toFixed(2);
            document.getElementById('momentumScore').textContent = (signal.momentum_score || 0).toFixed(2);
            document.getElementById('volumeScore').textContent = (signal.volume_score || 0).toFixed(2);
            document.getElementById('confluenceScore').textContent = (signal.confluence_score || 0).toFixed(2);
            document.getElementById('rrRatio').textContent = `1:${(signal.rr_ratio || 0).toFixed(1)}`;
            document.getElementById('entryPrice').textContent = (signal.entry_price || 0).toFixed(2);
            
            // Trading Levels
            document.getElementById('levelEntry').textContent = (signal.entry_price || 0).toFixed(2);
            document.getElementById('levelStopLoss').textContent = (signal.stop_loss || 0).toFixed(2);
            document.getElementById('levelTP1').textContent = (signal.take_profit_1 || 0).toFixed(2);
            document.getElementById('levelTP2').textContent = (signal.take_profit_2 || 0).toFixed(2);
            document.getElementById('levelTP3').textContent = (signal.take_profit_3 || 0).toFixed(2);
        }
        
        function updateTimeframesPrices(timeframes) {
            Object.entries(timeframes).forEach(([tf, data]) => {
                const priceEl = document.getElementById(`price${tf}`);
                if (priceEl) {
                    priceEl.textContent = (data.current_price || 0).toFixed(2);
                }
            });
        }
        
        function updatePerformanceStats(performance) {
            document.getElementById('totalSignals').textContent = performance.signals_generated || 0;
            document.getElementById('highQualitySignals').textContent = performance.high_quality_signals || 0;
            document.getElementById('filteredSignals').textContent = performance.filtered_signals || 0;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadAnalysis, 30000);
        
        // Initial load
        loadAnalysis();
    </script>
</body>
</html>
        '''
    
    def start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    self.analyze_all_timeframes()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    self.logger.error(f"Monitoring error: {str(e)}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("✅ Background monitoring started")
    
    def run(self, host='127.0.0.1', port=5000):
        """Run the Professional Trading AI"""
        try:
            # Connect to MT5
            if not self.connect_mt5():
                print("❌ Failed to connect to MT5")
                return
            
            self.is_running = True
            
            # Start background monitoring
            self.start_monitoring()
            
            # Initial analysis
            self.analyze_all_timeframes()
            
            print("🎯 Professional Trading AI Starting...")
            print(f"📊 Symbol: {self.symbol}")
            print(f"🌐 Dashboard: http://{host}:{port}")
            print("⏹️ Press Ctrl+C to stop")
            
            # Run Flask app
            self.app.run(host=host, port=port, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping Professional Trading AI...")
            self.is_running = False
            mt5.shutdown()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.is_running = False
            mt5.shutdown()

def main():
    """Main execution function"""
    print("🎯 PROFESSIONAL INDICATOR-BASED TRADING AI")
    print("=" * 50)
    print("📊 Multi-Timeframe Analysis")
    print("🎯 Professional Risk Management")
    print("📈 Real-time Signal Generation")
    print("=" * 50)
    
    # Initialize and run the AI
    trading_ai = ProfessionalTradingAI("XAUUSD.c")
    trading_ai.run()

if __name__ == "__main__":
    main()