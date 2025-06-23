"""
🎯 PROFESSIONAL MULTI-TIMEFRAME EMA TRADING SYSTEM
=================================================
Created for: XAUUSD.c Trading
Timeframes: H4, H1, M15, M5, M1
EMA Periods: 9, 21, 50, 200
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EMAMultiTimeframeSystem:
    """Professional EMA Trading System with Multi-Timeframe Analysis"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize EMA Trading System"""
        self.symbol = symbol
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        self.ema_periods = [9, 21, 50, 200]
        self.setup_logging()
        
        # Signal scoring weights
        self.scoring_weights = {
            'trend_alignment': 0.30,
            'crossover_strength': 0.25,
            'timeframe_confluence': 0.20,
            'momentum': 0.15,
            'distance_factor': 0.10
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ema_trading.log'),
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
                
            # Check symbol availability
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
        """Get historical rates for specified timeframe"""
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
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_all_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all EMA periods for given dataframe"""
        try:
            close = df['close']
            
            for period in self.ema_periods:
                df[f'EMA_{period}'] = self.calculate_ema(close, period)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating EMAs: {str(e)}")
            return df
    
    def get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction based on EMA alignment"""
        try:
            latest = df.iloc[-1]
            
            # Check EMA alignment for trend
            ema_9 = latest['EMA_9']
            ema_21 = latest['EMA_21']
            ema_50 = latest['EMA_50']
            ema_200 = latest['EMA_200']
            current_price = latest['close']
            
            # Strong bullish: Price > EMA9 > EMA21 > EMA50 > EMA200
            if (current_price > ema_9 > ema_21 > ema_50 > ema_200):
                return "STRONG_BULLISH"
            
            # Bullish: Price > EMA9 > EMA21 and trending up
            elif (current_price > ema_9 > ema_21) and (ema_21 > ema_50):
                return "BULLISH"
                
            # Strong bearish: Price < EMA9 < EMA21 < EMA50 < EMA200
            elif (current_price < ema_9 < ema_21 < ema_50 < ema_200):
                return "STRONG_BEARISH"
                
            # Bearish: Price < EMA9 < EMA21 and trending down
            elif (current_price < ema_9 < ema_21) and (ema_21 < ema_50):
                return "BEARISH"
                
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            self.logger.error(f"Error determining trend: {str(e)}")
            return "UNKNOWN"
    
    def detect_ema_crossovers(self, df: pd.DataFrame) -> Dict:
        """Detect EMA crossovers and their strength"""
        try:
            crossovers = {}
            
            # Check main crossovers
            crossover_pairs = [
                ('EMA_9', 'EMA_21'),
                ('EMA_21', 'EMA_50'),
                ('EMA_50', 'EMA_200')
            ]
            
            for fast, slow in crossover_pairs:
                # Current and previous values
                curr_fast = df[fast].iloc[-1]
                curr_slow = df[slow].iloc[-1]
                prev_fast = df[fast].iloc[-2]
                prev_slow = df[slow].iloc[-2]
                
                # Detect crossover
                if prev_fast <= prev_slow and curr_fast > curr_slow:
                    crossovers[f'{fast}_{slow}'] = {
                        'type': 'GOLDEN_CROSS',
                        'strength': abs(curr_fast - curr_slow) / curr_slow * 100,
                        'bars_ago': 0
                    }
                elif prev_fast >= prev_slow and curr_fast < curr_slow:
                    crossovers[f'{fast}_{slow}'] = {
                        'type': 'DEATH_CROSS',
                        'strength': abs(curr_fast - curr_slow) / curr_slow * 100,
                        'bars_ago': 0
                    }
            
            return crossovers
            
        except Exception as e:
            self.logger.error(f"Error detecting crossovers: {str(e)}")
            return {}
    
    def calculate_signal_strength(self, timeframe_data: Dict) -> float:
        """Calculate signal strength score (0-10)"""
        try:
            score = 0.0
            max_score = 10.0
            
            # 1. Trend Alignment Score (30%)
            trend_scores = {
                'STRONG_BULLISH': 10,
                'BULLISH': 7,
                'SIDEWAYS': 3,
                'BEARISH': 7,
                'STRONG_BEARISH': 10,
                'UNKNOWN': 0
            }
            
            h4_trend = trend_scores.get(timeframe_data['H4']['trend'], 0)
            h1_trend = trend_scores.get(timeframe_data['H1']['trend'], 0)
            
            # Bonus for trend alignment across timeframes
            if timeframe_data['H4']['trend'] == timeframe_data['H1']['trend']:
                trend_alignment = (h4_trend + h1_trend) / 2
            else:
                trend_alignment = min(h4_trend, h1_trend) * 0.7
                
            score += (trend_alignment / 10) * self.scoring_weights['trend_alignment'] * max_score
            
            # 2. Crossover Strength (25%)
            crossover_score = 0
            for tf in ['H1', 'M15']:
                if timeframe_data[tf]['crossovers']:
                    for cross_data in timeframe_data[tf]['crossovers'].values():
                        if cross_data['type'] in ['GOLDEN_CROSS', 'DEATH_CROSS']:
                            crossover_score += min(cross_data['strength'], 5)
            
            crossover_score = min(crossover_score, 10)
            score += (crossover_score / 10) * self.scoring_weights['crossover_strength'] * max_score
            
            # 3. Multi-Timeframe Confluence (20%)
            confluence_count = 0
            target_trend = timeframe_data['H4']['trend']
            
            for tf in ['H1', 'M15', 'M5']:
                if timeframe_data[tf]['trend'] == target_trend:
                    confluence_count += 1
                    
            confluence_score = (confluence_count / 3) * 10
            score += (confluence_score / 10) * self.scoring_weights['timeframe_confluence'] * max_score
            
            # 4. Momentum Score (15%)
            # Based on EMA spacing and direction
            momentum_score = 5  # Base momentum
            try:
                m15_data = timeframe_data['M15']['data']
                latest = m15_data.iloc[-1]
                
                # Check EMA spacing (wider = stronger momentum)
                ema_diff = abs(latest['EMA_9'] - latest['EMA_21']) / latest['close'] * 100
                momentum_score += min(ema_diff * 2, 5)
                
            except:
                pass
                
            momentum_score = min(momentum_score, 10)
            score += (momentum_score / 10) * self.scoring_weights['momentum'] * max_score
            
            # 5. Distance Factor (10%)
            # Penalize signals too far from EMAs
            distance_score = 10
            try:
                m15_data = timeframe_data['M15']['data']
                latest = m15_data.iloc[-1]
                current_price = latest['close']
                ema_9 = latest['EMA_9']
                
                distance_pct = abs(current_price - ema_9) / current_price * 100
                if distance_pct > 2:  # If price is >2% away from EMA9
                    distance_score = max(10 - (distance_pct - 2) * 2, 0)
                    
            except:
                pass
                
            score += (distance_score / 10) * self.scoring_weights['distance_factor'] * max_score
            
            return round(min(max(score, 0), 10), 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0
    
    def analyze_all_timeframes(self) -> Dict:
        """Analyze all timeframes and return comprehensive data"""
        try:
            analysis = {}
            
            for tf_name in self.timeframes.keys():
                self.logger.info(f"Analyzing {tf_name}...")
                
                # Get data
                df = self.get_rates(tf_name, 300)
                if df is None:
                    continue
                    
                # Calculate EMAs
                df = self.calculate_all_emas(df)
                
                # Get trend direction
                trend = self.get_trend_direction(df)
                
                # Detect crossovers
                crossovers = self.detect_ema_crossovers(df)
                
                analysis[tf_name] = {
                    'data': df,
                    'trend': trend,
                    'crossovers': crossovers,
                    'current_price': df['close'].iloc[-1],
                    'emas': {
                        'EMA_9': df['EMA_9'].iloc[-1],
                        'EMA_21': df['EMA_21'].iloc[-1],
                        'EMA_50': df['EMA_50'].iloc[-1],
                        'EMA_200': df['EMA_200'].iloc[-1]
                    }
                }
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            return {}
    
    def generate_trading_signals(self) -> Dict:
        """Generate comprehensive trading signals"""
        try:
            # Get multi-timeframe analysis
            timeframe_data = self.analyze_all_timeframes()
            
            if not timeframe_data:
                return {'error': 'Failed to get timeframe data'}
            
            # Calculate signal strength
            signal_strength = self.calculate_signal_strength(timeframe_data)
            
            # Determine signal direction based on H4 bias and H1 setup
            h4_trend = timeframe_data.get('H4', {}).get('trend', 'UNKNOWN')
            h1_trend = timeframe_data.get('H1', {}).get('trend', 'UNKNOWN')
            m15_crossovers = timeframe_data.get('M15', {}).get('crossovers', {})
            
            signal_direction = "NONE"
            confidence = "LOW"
            
            # Signal Logic: H4 bias + H1 setup + M15 entry
            if h4_trend in ['STRONG_BULLISH', 'BULLISH'] and h1_trend in ['STRONG_BULLISH', 'BULLISH']:
                if any(cross['type'] == 'GOLDEN_CROSS' for cross in m15_crossovers.values()):
                    signal_direction = "BUY"
                elif h1_trend == 'STRONG_BULLISH' and signal_strength >= 6.0:
                    signal_direction = "BUY"
                    
            elif h4_trend in ['STRONG_BEARISH', 'BEARISH'] and h1_trend in ['STRONG_BEARISH', 'BEARISH']:
                if any(cross['type'] == 'DEATH_CROSS' for cross in m15_crossovers.values()):
                    signal_direction = "SELL"
                elif h1_trend == 'STRONG_BEARISH' and signal_strength >= 6.0:
                    signal_direction = "SELL"
            
            # Determine confidence level
            if signal_strength >= 8.0:
                confidence = "VERY_HIGH"
            elif signal_strength >= 6.5:
                confidence = "HIGH"
            elif signal_strength >= 5.0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Calculate risk/reward levels
            current_price = timeframe_data['M15']['current_price']
            atr_approx = current_price * 0.005  # Approximate ATR for XAUUSD
            
            if signal_direction == "BUY":
                stop_loss = current_price - (atr_approx * 2)
                take_profit_1 = current_price + (atr_approx * 2)  # 1:1
                take_profit_2 = current_price + (atr_approx * 4)  # 1:2
                take_profit_3 = current_price + (atr_approx * 6)  # 1:3
                
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr_approx * 2)
                take_profit_1 = current_price - (atr_approx * 2)  # 1:1
                take_profit_2 = current_price - (atr_approx * 4)  # 1:2
                take_profit_3 = current_price - (atr_approx * 6)  # 1:3
                
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Compile final signal
            signal = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'h4_trend': h4_trend,
                'h1_trend': h1_trend,
                'timeframe_analysis': {
                    tf: {
                        'trend': data['trend'],
                        'crossovers': len(data['crossovers']),
                        'emas': data['emas']
                    } for tf, data in timeframe_data.items()
                },
                'risk_reward': {
                    'stop_loss': round(stop_loss, 2),
                    'take_profit_1': round(take_profit_1, 2),
                    'take_profit_2': round(take_profit_2, 2),
                    'take_profit_3': round(take_profit_3, 2),
                    'risk_reward_ratio': 2.0 if signal_direction != "NONE" else 0
                }
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {'error': str(e)}
    
    def print_signal_summary(self, signal: Dict):
        """Print formatted signal summary"""
        if 'error' in signal:
            print(f"❌ ERROR: {signal['error']}")
            return
            
        print("\n" + "="*60)
        print("🎯 XAUUSD EMA TRADING SIGNAL ANALYSIS")
        print("="*60)
        print(f"📅 Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Symbol: {signal['symbol']}")
        print(f"💵 Current Price: ${signal['current_price']}")
        print(f"📊 Signal Strength: {signal['signal_strength']}/10")
        print(f"🎯 Confidence: {signal['confidence']}")
        
        # Signal Direction
        direction_emoji = "🟢" if signal['signal_direction'] == "BUY" else "🔴" if signal['signal_direction'] == "SELL" else "⚪"
        print(f"{direction_emoji} Signal: {signal['signal_direction']}")
        
        print(f"\n📈 TREND ANALYSIS:")
        print(f"   H4 Trend: {signal['h4_trend']}")
        print(f"   H1 Trend: {signal['h1_trend']}")
        
        print(f"\n🎯 RISK/REWARD SETUP:")
        if signal['signal_direction'] != "NONE":
            print(f"   Stop Loss: ${signal['risk_reward']['stop_loss']}")
            print(f"   Take Profit 1: ${signal['risk_reward']['take_profit_1']} (1:1)")
            print(f"   Take Profit 2: ${signal['risk_reward']['take_profit_2']} (1:2)")
            print(f"   Take Profit 3: ${signal['risk_reward']['take_profit_3']} (1:3)")
            print(f"   Risk/Reward: 1:{signal['risk_reward']['risk_reward_ratio']}")
        else:
            print("   No active signal - Wait for better setup")
        
        print("\n📊 TIMEFRAME CONFLUENCE:")
        for tf, data in signal['timeframe_analysis'].items():
            crossover_text = f"({data['crossovers']} crossovers)" if data['crossovers'] > 0 else ""
            print(f"   {tf}: {data['trend']} {crossover_text}")
        
        print("="*60)

def main():
    """Main execution function"""
    print("🚀 Starting Professional EMA Trading System...")
    
    # Initialize system
    ema_system = EMAMultiTimeframeSystem("XAUUSD.c")
    
    # Connect to MT5
    if not ema_system.connect_mt5():
        print("❌ Failed to connect to MT5. Please check your connection.")
        return
    
    try:
        # Generate signals
        print("📊 Analyzing multi-timeframe data...")
        signal = ema_system.generate_trading_signals()
        
        # Print results
        ema_system.print_signal_summary(signal)
        
        # Optional: Run continuous monitoring
        print(f"\n💡 TIP: Run this script every 15 minutes for fresh signals!")
        print(f"💡 TIP: Combine with other indicators for better confluence!")
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        mt5.shutdown()
        print("🔌 MT5 connection closed")

if __name__ == "__main__":
    main()

# Example Usage:
"""
# Basic usage
system = EMAMultiTimeframeSystem("XAUUSD.c")
if system.connect_mt5():
    signal = system.generate_trading_signals()
    system.print_signal_summary(signal)

# Advanced usage with custom parameters
system.ema_periods = [8, 21, 50, 100]  # Custom EMA periods
system.scoring_weights['trend_alignment'] = 0.40  # Adjust weights
signal = system.generate_trading_signals()
"""