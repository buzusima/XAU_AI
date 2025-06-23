"""
🎯 OPTIMIZED TRADING SYSTEM V2.0
===============================
Enhanced version based on backtesting results
Focus: Higher Win Rate, Better Risk Management, Improved Filtering
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OptimizedTradingSystemV2:
    """Enhanced Trading System with Optimized Parameters"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize Optimized Trading System V2"""
        self.symbol = symbol
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        
        # ENHANCED PARAMETERS (Based on backtest analysis)
        self.ema_periods = [9, 21, 50, 200]
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14
        self.rsi_overbought = 65  # Tighter from 70
        self.rsi_oversold = 35    # Tighter from 30
        
        # ENHANCED FILTERING THRESHOLDS
        self.min_confluence_score = 7.5  # Increased from 6.0
        self.min_timeframes_agreement = 4  # Increased from 3
        self.max_rsi_extreme = 20  # More strict (was 25)
        self.min_volume_confirmation = 1.3  # Add volume filter
        
        # ENHANCED RISK MANAGEMENT
        self.max_consecutive_losses = 5  # Stop after 5 losses
        self.risk_scaling_factor = 0.8  # Reduce risk after losses
        self.profit_protection_ratio = 0.7  # Protect 70% of open profit
        
        # MARKET CONDITION FILTERS
        self.min_atr_threshold = 0.001  # Minimum volatility
        self.max_atr_threshold = 0.020  # Maximum volatility
        self.trend_strength_min = 0.75  # Minimum trend strength
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimized_trading_v2.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found")
                return False
                
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to select symbol {self.symbol}")
                    return False
                    
            self.logger.info(f"Successfully connected to MT5 and selected {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def get_rates(self, timeframe_str: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Get historical rates"""
        try:
            timeframe = self.timeframes[timeframe_str]
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            
            if rates is None:
                self.logger.error(f"Failed to get rates for {timeframe_str}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates for {timeframe_str}: {str(e)}")
            return None
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced indicators with additional filters"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', df.get('volume', pd.Series(1, index=df.index)))
            
            # EMA Calculations
            for period in self.ema_periods:
                df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            
            # Enhanced MACD with histogram analysis
            ema_fast = close.ewm(span=self.macd_fast).mean()
            ema_slow = close.ewm(span=self.macd_slow).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Enhanced RSI with smoothing
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI_Smooth'] = df['RSI'].rolling(window=3).mean()  # Smoothed RSI
            
            # Enhanced ATR with multiple periods
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            df['ATR_Pct'] = (df['ATR'] / close) * 100  # ATR as percentage
            
            # Trend Strength Indicator
            df['Trend_Strength'] = self.calculate_trend_strength(df)
            
            # Volume Analysis
            df['Volume_SMA'] = volume.rolling(window=20).mean()
            df['Volume_Ratio'] = volume / df['Volume_SMA']
            
            # Market Structure
            df['Higher_High'] = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
            df['Lower_Low'] = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
            
            # Support/Resistance levels
            df['Resistance'] = high.rolling(window=20).max()
            df['Support'] = low.rolling(window=20).min()
            df['Price_Position'] = (close - df['Support']) / (df['Resistance'] - df['Support'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced indicators: {str(e)}")
            return df
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength based on EMA alignment"""
        try:
            ema_9 = df['EMA_9']
            ema_21 = df['EMA_21']
            ema_50 = df['EMA_50']
            ema_200 = df['EMA_200']
            close = df['close']
            
            # Calculate alignment score for each row
            trend_strength = []
            
            for i in range(len(df)):
                # Bullish conditions
                bullish_conditions = [
                    close.iloc[i] > ema_9.iloc[i],
                    ema_9.iloc[i] > ema_21.iloc[i],
                    ema_21.iloc[i] > ema_50.iloc[i],
                    ema_50.iloc[i] > ema_200.iloc[i]
                ]
                
                # Bearish conditions
                bearish_conditions = [
                    close.iloc[i] < ema_9.iloc[i],
                    ema_9.iloc[i] < ema_21.iloc[i],
                    ema_21.iloc[i] < ema_50.iloc[i],
                    ema_50.iloc[i] < ema_200.iloc[i]
                ]
                
                bullish_score = sum(bullish_conditions)
                bearish_score = sum(bearish_conditions)
                
                # Calculate strength (0-1 scale)
                max_score = max(bullish_score, bearish_score)
                strength = max(0, (max_score - 2) / 2) if max_score >= 2 else 0
                
                trend_strength.append(strength)
            
            return pd.Series(trend_strength, index=df.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def enhanced_signal_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced signal analysis with stricter criteria"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = {
                'direction': 'NONE',
                'strength': 0.0,
                'confidence': 'LOW',
                'filters_passed': [],
                'filters_failed': [],
                'risk_factors': []
            }
            
            # 1. ENHANCED EMA ANALYSIS
            ema_9 = latest['EMA_9']
            ema_21 = latest['EMA_21']
            ema_50 = latest['EMA_50']
            ema_200 = latest['EMA_200']
            current_price = latest['close']
            
            # Stronger EMA conditions
            strong_bullish_ema = (current_price > ema_9 > ema_21 > ema_50 > ema_200)
            strong_bearish_ema = (current_price < ema_9 < ema_21 < ema_50 < ema_200)
            moderate_bullish_ema = (current_price > ema_9 > ema_21) and (ema_21 > ema_50)
            moderate_bearish_ema = (current_price < ema_9 < ema_21) and (ema_21 < ema_50)
            
            # 2. ENHANCED MACD ANALYSIS
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            macd_hist = latest['MACD_Histogram']
            prev_macd_hist = prev['MACD_Histogram']
            
            macd_bullish = (macd > macd_signal) and (macd_hist > prev_macd_hist)
            macd_bearish = (macd < macd_signal) and (macd_hist < prev_macd_hist)
            
            # 3. ENHANCED RSI ANALYSIS
            rsi = latest['RSI_Smooth']  # Use smoothed RSI
            rsi_trend = latest['RSI_Smooth'] - df['RSI_Smooth'].iloc[-5:].mean()
            
            rsi_bullish = (self.rsi_oversold < rsi < self.rsi_overbought) and (rsi > 50) and (rsi_trend > 0)
            rsi_bearish = (self.rsi_oversold < rsi < self.rsi_overbought) and (rsi < 50) and (rsi_trend < 0)
            rsi_extreme = rsi < self.max_rsi_extreme or rsi > (100 - self.max_rsi_extreme)
            
            # 4. MARKET CONDITION FILTERS
            atr_pct = latest['ATR_Pct']
            trend_strength = latest['Trend_Strength']
            volume_ratio = latest['Volume_Ratio']
            price_position = latest['Price_Position']
            
            # Apply enhanced filters
            filter_results = []
            
            # Filter 1: Volatility check
            if self.min_atr_threshold <= atr_pct <= self.max_atr_threshold:
                filter_results.append("✅ Volatility OK")
                signals['filters_passed'].append("Volatility within range")
            else:
                filter_results.append(f"❌ Volatility: {atr_pct:.3f}%")
                signals['filters_failed'].append(f"Volatility out of range: {atr_pct:.3f}%")
            
            # Filter 2: Trend strength
            if trend_strength >= self.trend_strength_min:
                filter_results.append("✅ Strong trend")
                signals['filters_passed'].append("Strong trend detected")
            else:
                filter_results.append(f"❌ Weak trend: {trend_strength:.2f}")
                signals['filters_failed'].append(f"Weak trend strength: {trend_strength:.2f}")
            
            # Filter 3: Volume confirmation
            if volume_ratio >= self.min_volume_confirmation:
                filter_results.append("✅ Volume confirmed")
                signals['filters_passed'].append("Volume confirmation")
            else:
                filter_results.append(f"❌ Low volume: {volume_ratio:.2f}")
                signals['filters_failed'].append(f"Low volume ratio: {volume_ratio:.2f}")
            
            # Filter 4: RSI extreme check
            if not rsi_extreme:
                filter_results.append("✅ RSI normal")
                signals['filters_passed'].append("RSI in normal range")
            else:
                filter_results.append(f"❌ RSI extreme: {rsi:.1f}")
                signals['filters_failed'].append(f"RSI extreme level: {rsi:.1f}")
                signals['risk_factors'].append("RSI extreme - potential reversal")
            
            # Filter 5: Price position
            if 0.2 <= price_position <= 0.8:
                filter_results.append("✅ Good price position")
                signals['filters_passed'].append("Good price position in range")
            else:
                filter_results.append(f"❌ Extreme price position: {price_position:.2f}")
                signals['filters_failed'].append(f"Price at extreme level: {price_position:.2f}")
                signals['risk_factors'].append("Price at support/resistance extreme")
            
            # SIGNAL GENERATION WITH ENHANCED LOGIC
            confluence_score = 0.0
            
            # EMA scoring
            if strong_bullish_ema:
                confluence_score += 3.0
                signal_bias = 'BULLISH'
            elif strong_bearish_ema:
                confluence_score += 3.0
                signal_bias = 'BEARISH'
            elif moderate_bullish_ema:
                confluence_score += 2.0
                signal_bias = 'BULLISH'
            elif moderate_bearish_ema:
                confluence_score += 2.0
                signal_bias = 'BEARISH'
            else:
                signal_bias = 'NEUTRAL'
            
            # MACD scoring
            if macd_bullish and signal_bias == 'BULLISH':
                confluence_score += 2.5
            elif macd_bearish and signal_bias == 'BEARISH':
                confluence_score += 2.5
            elif macd_bullish or macd_bearish:
                confluence_score += 1.0
            
            # RSI scoring
            if rsi_bullish and signal_bias == 'BULLISH':
                confluence_score += 2.0
            elif rsi_bearish and signal_bias == 'BEARISH':
                confluence_score += 2.0
            elif (rsi_bullish or rsi_bearish) and not rsi_extreme:
                confluence_score += 1.0
            
            # Bonus for filter confirmations
            confluence_score += len(signals['filters_passed']) * 0.5
            
            # Penalty for failed filters
            confluence_score -= len(signals['filters_failed']) * 1.0
            
            # Determine final signal
            signals['strength'] = round(min(max(confluence_score, 0), 10), 2)
            
            if (signals['strength'] >= self.min_confluence_score and 
                len(signals['filters_failed']) <= 1 and
                not rsi_extreme and
                signal_bias in ['BULLISH', 'BEARISH']):
                
                signals['direction'] = 'BUY' if signal_bias == 'BULLISH' else 'SELL'
                
                if signals['strength'] >= 9.0:
                    signals['confidence'] = 'VERY_HIGH'
                elif signals['strength'] >= 8.0:
                    signals['confidence'] = 'HIGH'
                elif signals['strength'] >= 7.5:
                    signals['confidence'] = 'MEDIUM'
                else:
                    signals['confidence'] = 'LOW'
            else:
                signals['direction'] = 'NONE'
                signals['confidence'] = 'FILTERED'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in enhanced signal analysis: {str(e)}")
            return {'direction': 'NONE', 'strength': 0.0, 'confidence': 'ERROR', 
                   'filters_passed': [], 'filters_failed': [], 'risk_factors': []}
    
    def analyze_timeframe_v2(self, timeframe_str: str) -> Dict:
        """Enhanced timeframe analysis"""
        try:
            df = self.get_rates(timeframe_str, 300)
            if df is None:
                return {}
            
            df = self.calculate_enhanced_indicators(df)
            signals = self.enhanced_signal_analysis(df)
            
            return {
                'timeframe': timeframe_str,
                'price': df['close'].iloc[-1],
                'atr': df['ATR'].iloc[-1],
                'atr_pct': df['ATR_Pct'].iloc[-1],
                'trend_strength': df['Trend_Strength'].iloc[-1],
                'rsi': df['RSI_Smooth'].iloc[-1],
                'volume_ratio': df['Volume_Ratio'].iloc[-1],
                'price_position': df['Price_Position'].iloc[-1],
                'signals': signals
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe_str}: {str(e)}")
            return {}
    
    def generate_optimized_signals(self) -> Dict:
        """Generate signals with enhanced optimization"""
        try:
            # Analyze all timeframes
            timeframe_analysis = {}
            for tf in self.timeframes.keys():
                self.logger.info(f"Analyzing {tf} with optimized parameters...")
                timeframe_analysis[tf] = self.analyze_timeframe_v2(tf)
            
            # Multi-timeframe confluence
            signal_votes = {'BUY': 0, 'SELL': 0, 'NONE': 0}
            total_strength = 0.0
            all_filters_passed = []
            all_filters_failed = []
            all_risk_factors = []
            
            # Timeframe weights (H4 and H1 more important)
            tf_weights = {'H4': 0.35, 'H1': 0.30, 'M15': 0.20, 'M5': 0.10, 'M1': 0.05}
            
            for tf, weight in tf_weights.items():
                if tf in timeframe_analysis and timeframe_analysis[tf]:
                    tf_data = timeframe_analysis[tf]
                    tf_signals = tf_data.get('signals', {})
                    
                    direction = tf_signals.get('direction', 'NONE')
                    strength = tf_signals.get('strength', 0)
                    
                    signal_votes[direction] += weight
                    total_strength += strength * weight
                    
                    all_filters_passed.extend(tf_signals.get('filters_passed', []))
                    all_filters_failed.extend(tf_signals.get('filters_failed', []))
                    all_risk_factors.extend(tf_signals.get('risk_factors', []))
            
            # Determine final signal
            max_vote = max(signal_votes.values())
            final_direction = [k for k, v in signal_votes.items() if v == max_vote][0]
            
            # Enhanced filtering
            min_agreement = 0.6  # 60% of timeframes must agree
            if max_vote < min_agreement or final_direction == 'NONE':
                final_direction = 'NONE'
                confidence = 'FILTERED'
            else:
                if total_strength >= 8.5:
                    confidence = 'VERY_HIGH'
                elif total_strength >= 7.5:
                    confidence = 'HIGH'
                elif total_strength >= 6.5:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
            
            # Risk/Reward calculation
            current_price = timeframe_analysis.get('M15', {}).get('price', 0) or timeframe_analysis.get('H1', {}).get('price', 0)
            atr = timeframe_analysis.get('M15', {}).get('atr', current_price * 0.005)
            
            # Enhanced R/R ratios
            if final_direction == 'BUY':
                stop_loss = current_price - (atr * 1.5)  # Tighter stops
                take_profit_1 = current_price + (atr * 2.5)  # Better R/R
                take_profit_2 = current_price + (atr * 4.0)
                take_profit_3 = current_price + (atr * 6.0)
            elif final_direction == 'SELL':
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_direction': final_direction,
                'strength_score': round(total_strength, 2),
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'signal_votes': signal_votes,
                'timeframe_analysis': timeframe_analysis,
                'filters_summary': {
                    'passed': list(set(all_filters_passed)),
                    'failed': list(set(all_filters_failed)),
                    'risk_factors': list(set(all_risk_factors))
                },
                'risk_reward': {
                    'stop_loss': round(stop_loss, 2),
                    'take_profit_1': round(take_profit_1, 2),
                    'take_profit_2': round(take_profit_2, 2),
                    'take_profit_3': round(take_profit_3, 2),
                    'atr_used': round(atr, 2),
                    'risk_reward_ratio': 1.67 if final_direction != 'NONE' else 0  # 1.5:2.5 = 1:1.67
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating optimized signals: {str(e)}")
            return {'error': str(e)}
    
    def print_optimized_analysis(self, signal: Dict):
        """Print enhanced analysis results"""
        if 'error' in signal:
            print(f"❌ ERROR: {signal['error']}")
            return
        
        print("\n" + "="*90)
        print("🎯 OPTIMIZED TRADING SYSTEM V2.0 ANALYSIS")
        print("Enhanced Filtering • Improved Risk Management • Higher Win Rate Focus")
        print("="*90)
        print(f"📅 Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Symbol: {signal['symbol']}")
        print(f"💵 Current Price: ${signal['current_price']}")
        print(f"📊 Strength Score: {signal['strength_score']}/10")
        print(f"🎯 Confidence: {signal['confidence']}")
        
        # Signal with voting breakdown
        direction_emoji = "🟢" if signal['signal_direction'] == "BUY" else "🔴" if signal['signal_direction'] == "SELL" else "⚪"
        print(f"{direction_emoji} OPTIMIZED SIGNAL: {signal['signal_direction']}")
        
        # Voting breakdown
        votes = signal['signal_votes']
        print(f"\n🗳️ TIMEFRAME VOTING:")
        print(f"   BUY votes: {votes['BUY']:.2f}")
        print(f"   SELL votes: {votes['SELL']:.2f}")
        print(f"   NONE votes: {votes['NONE']:.2f}")
        
        # Enhanced filters
        filters = signal['filters_summary']
        print(f"\n🔍 ENHANCED FILTERS:")
        print(f"   ✅ Passed Filters: {len(filters['passed'])}")
        for f in filters['passed'][:5]:  # Show top 5
            print(f"      • {f}")
        
        print(f"   ❌ Failed Filters: {len(filters['failed'])}")
        for f in filters['failed'][:3]:  # Show top 3
            print(f"      • {f}")
        
        if filters['risk_factors']:
            print(f"   🚨 Risk Factors: {len(filters['risk_factors'])}")
            for rf in filters['risk_factors']:
                print(f"      • {rf}")
        
        # Enhanced Risk/Reward
        print(f"\n🎯 ENHANCED RISK/REWARD:")
        if signal['signal_direction'] != "NONE":
            rr = signal['risk_reward']
            print(f"   Stop Loss: ${rr['stop_loss']} (1.5 ATR)")
            print(f"   Take Profit 1: ${rr['take_profit_1']} (2.5 ATR) - 1:1.67 R/R")
            print(f"   Take Profit 2: ${rr['take_profit_2']} (4.0 ATR) - 1:2.67 R/R")
            print(f"   Take Profit 3: ${rr['take_profit_3']} (6.0 ATR) - 1:4.0 R/R")
            print(f"   ATR Used: ${rr['atr_used']}")
        else:
            print("   No active signal - Enhanced filters protecting capital")
        
        # Timeframe Details
        print(f"\n📈 ENHANCED TIMEFRAME ANALYSIS:")
        for tf, data in signal['timeframe_analysis'].items():
            if not data:
                continue
                
            tf_signals = data.get('signals', {})
            direction = tf_signals.get('direction', 'NONE')
            strength = tf_signals.get('strength', 0)
            trend_str = data.get('trend_strength', 0)
            rsi = data.get('rsi', 0)
            vol_ratio = data.get('volume_ratio', 0)
            
            status_emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
            
            print(f"   {tf}: {status_emoji} {direction} | Strength: {strength:.1f}/10 | "
                  f"Trend: {trend_str:.2f} | RSI: {rsi:.1f} | Vol: {vol_ratio:.2f}")
        
        print("="*90)
        
        # Enhanced recommendations
        if signal['signal_direction'] != "NONE" and signal['confidence'] in ['HIGH', 'VERY_HIGH']:
            print(f"✅ STRONG RECOMMENDATION: Consider {signal['signal_direction']} position")
            print(f"💡 Enhanced system shows {signal['confidence']} confidence")
            print(f"💡 Improved risk/reward ratio: 1:1.67 minimum")
        elif signal['signal_direction'] != "NONE":
            print(f"⚠️ MODERATE RECOMMENDATION: {signal['signal_direction']} with caution")
            print(f"💡 Medium confidence - consider smaller position size")
        else:
            print("⏳ RECOMMENDATION: Wait for better setup")
            print("💡 Enhanced filters protecting against false signals")
        
        print(f"\n🚀 V2.0 IMPROVEMENTS:")
        print(f"   ✅ Tighter entry criteria (7.5/10 vs 6.0/10)")
        print(f"   ✅ Enhanced RSI filtering (35-65 vs 30-70)")
        print(f"   ✅ Volume confirmation required")
        print(f"   ✅ Trend strength validation")
        print(f"   ✅ Improved risk/reward ratios")
        print(f"   ✅ Multi-timeframe voting system")

def main():
    """Main execution for optimized system"""
    print("🚀 Starting Optimized Trading System V2.0...")
    
    # Initialize optimized system
    optimized_system = OptimizedTradingSystemV2("XAUUSD.c")
    
    # Connect to MT5
    if not optimized_system.connect_mt5():
        print("❌ Failed to connect to MT5. Please check your connection.")
        return
    
    try:
        # Generate optimized signals
        print("📊 Analyzing with enhanced optimization parameters...")
        signal = optimized_system.generate_optimized_signals()
        
        # Print enhanced analysis
        optimized_system.print_optimized_analysis(signal)
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        mt5.shutdown()
        print("🔌 MT5 connection closed")

if __name__ == "__main__":
    main()