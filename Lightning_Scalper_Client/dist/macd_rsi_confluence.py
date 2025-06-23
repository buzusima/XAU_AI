"""
📊 PROFESSIONAL MACD + RSI CONFLUENCE SYSTEM
===========================================
Enhanced Technical Analysis for XAUUSD.c
MACD: (12,26,9) + RSI: (14) + Divergence Detection
Multi-Timeframe Confluence with Volume Confirmation
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MACDRSIConfluenceSystem:
    """Professional MACD + RSI Confluence Trading System"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize MACD + RSI Confluence System"""
        self.symbol = symbol
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        
        # MACD Parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # RSI Parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Confluence requirements
        self.min_timeframes_agreement = 3
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('macd_rsi_trading.log'),
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
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD with histogram and signal line"""
        try:
            close = df['close']
            
            # Calculate EMAs for MACD
            ema_fast = close.ewm(span=self.macd_fast).mean()
            ema_slow = close.ewm(span=self.macd_slow).mean()
            
            # MACD Line
            df['MACD'] = ema_fast - ema_slow
            
            # Signal Line
            df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal).mean()
            
            # Histogram
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # MACD trend analysis
            df['MACD_Trend'] = 'NEUTRAL'
            df.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Trend'] = 'BULLISH'
            df.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Trend'] = 'BEARISH'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return df
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI with overbought/oversold levels"""
        try:
            close = df['close']
            delta = close.diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # RSI conditions
            df['RSI_Condition'] = 'NEUTRAL'
            df.loc[df['RSI'] > self.rsi_overbought, 'RSI_Condition'] = 'OVERBOUGHT'
            df.loc[df['RSI'] < self.rsi_oversold, 'RSI_Condition'] = 'OVERSOLD'
            
            # RSI trend
            df['RSI_Trend'] = 'NEUTRAL'
            df.loc[df['RSI'] > 50, 'RSI_Trend'] = 'BULLISH'
            df.loc[df['RSI'] < 50, 'RSI_Trend'] = 'BEARISH'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return df
    
    def detect_macd_signals(self, df: pd.DataFrame) -> Dict:
        """Detect MACD crossovers and divergences"""
        try:
            signals = {
                'crossover': None,
                'zero_line': None,
                'histogram_trend': None,
                'divergence': None
            }
            
            if len(df) < 5:
                return signals
            
            # Current and previous values
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_Signal'].iloc[-1]
            current_hist = df['MACD_Histogram'].iloc[-1]
            
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_Signal'].iloc[-2]
            prev_hist = df['MACD_Histogram'].iloc[-2]
            
            # 1. MACD Line Crossovers
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals['crossover'] = {
                    'type': 'BULLISH_CROSSOVER',
                    'strength': abs(current_macd - current_signal),
                    'location': 'ABOVE_ZERO' if current_macd > 0 else 'BELOW_ZERO'
                }
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals['crossover'] = {
                    'type': 'BEARISH_CROSSOVER',
                    'strength': abs(current_macd - current_signal),
                    'location': 'ABOVE_ZERO' if current_macd > 0 else 'BELOW_ZERO'
                }
            
            # 2. Zero Line Crossovers
            if df['MACD'].iloc[-3] <= 0 and current_macd > 0:
                signals['zero_line'] = 'BULLISH_ZERO_CROSS'
            elif df['MACD'].iloc[-3] >= 0 and current_macd < 0:
                signals['zero_line'] = 'BEARISH_ZERO_CROSS'
            
            # 3. Histogram Trend
            hist_last_3 = df['MACD_Histogram'].iloc[-3:].tolist()
            if all(hist_last_3[i] < hist_last_3[i+1] for i in range(len(hist_last_3)-1)):
                signals['histogram_trend'] = 'INCREASING'
            elif all(hist_last_3[i] > hist_last_3[i+1] for i in range(len(hist_last_3)-1)):
                signals['histogram_trend'] = 'DECREASING'
            
            # 4. Simple Divergence Detection
            try:
                price_highs = df['high'].iloc[-20:].rolling(3).max()
                price_lows = df['low'].iloc[-20:].rolling(3).min()
                macd_values = df['MACD'].iloc[-20:]
                
                # Bullish divergence: Price making lower lows, MACD making higher lows
                recent_price_low = price_lows.iloc[-1]
                prev_price_low = price_lows.iloc[-10]
                recent_macd = macd_values.iloc[-1]
                prev_macd = macd_values.iloc[-10]
                
                if (recent_price_low < prev_price_low and 
                    recent_macd > prev_macd and 
                    recent_macd < 0):
                    signals['divergence'] = 'BULLISH_DIVERGENCE'
                elif (recent_price_low > prev_price_low and 
                      recent_macd < prev_macd and 
                      recent_macd > 0):
                    signals['divergence'] = 'BEARISH_DIVERGENCE'
                    
            except:
                pass
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD signals: {str(e)}")
            return {'crossover': None, 'zero_line': None, 'histogram_trend': None, 'divergence': None}
    
    def detect_rsi_signals(self, df: pd.DataFrame) -> Dict:
        """Detect RSI signals and divergences"""
        try:
            signals = {
                'condition': df['RSI_Condition'].iloc[-1],
                'trend': df['RSI_Trend'].iloc[-1],
                'divergence': None,
                'momentum': None
            }
            
            if len(df) < 10:
                return signals
            
            current_rsi = df['RSI'].iloc[-1]
            
            # RSI Momentum
            rsi_change = df['RSI'].iloc[-1] - df['RSI'].iloc[-5]
            if abs(rsi_change) > 5:
                signals['momentum'] = 'STRONG_BULLISH' if rsi_change > 0 else 'STRONG_BEARISH'
            elif abs(rsi_change) > 2:
                signals['momentum'] = 'MODERATE_BULLISH' if rsi_change > 0 else 'MODERATE_BEARISH'
            else:
                signals['momentum'] = 'WEAK'
            
            # RSI Divergence Detection
            try:
                price_data = df['close'].iloc[-20:]
                rsi_data = df['RSI'].iloc[-20:]
                
                # Find recent highs and lows
                price_recent_high = price_data.iloc[-5:].max()
                price_prev_high = price_data.iloc[-15:-10].max()
                rsi_recent_high = rsi_data.iloc[-5:].max()
                rsi_prev_high = rsi_data.iloc[-15:-10].max()
                
                price_recent_low = price_data.iloc[-5:].min()
                price_prev_low = price_data.iloc[-15:-10].min()
                rsi_recent_low = rsi_data.iloc[-5:].min()
                rsi_prev_low = rsi_data.iloc[-15:-10].min()
                
                # Bullish divergence
                if (price_recent_low < price_prev_low and rsi_recent_low > rsi_prev_low):
                    signals['divergence'] = 'BULLISH_DIVERGENCE'
                # Bearish divergence
                elif (price_recent_high > price_prev_high and rsi_recent_high < rsi_prev_high):
                    signals['divergence'] = 'BEARISH_DIVERGENCE'
                    
            except:
                pass
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting RSI signals: {str(e)}")
            return {'condition': 'NEUTRAL', 'trend': 'NEUTRAL', 'divergence': None, 'momentum': None}
    
    def calculate_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based confirmation"""
        try:
            if 'tick_volume' not in df.columns:
                return {'volume_trend': 'NO_DATA', 'volume_strength': 0}
            
            # Volume moving averages
            vol_sma_short = df['tick_volume'].rolling(10).mean()
            vol_sma_long = df['tick_volume'].rolling(30).mean()
            
            current_volume = df['tick_volume'].iloc[-1]
            avg_volume_short = vol_sma_short.iloc[-1]
            avg_volume_long = vol_sma_long.iloc[-1]
            
            # Volume trend
            volume_trend = 'INCREASING' if avg_volume_short > avg_volume_long else 'DECREASING'
            
            # Volume strength (compared to average)
            volume_ratio = current_volume / avg_volume_long if avg_volume_long > 0 else 1
            
            if volume_ratio > 1.5:
                volume_strength = 'HIGH'
            elif volume_ratio > 1.2:
                volume_strength = 'MEDIUM'
            else:
                volume_strength = 'LOW'
            
            return {
                'volume_trend': volume_trend,
                'volume_strength': volume_strength,
                'volume_ratio': round(volume_ratio, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume: {str(e)}")
            return {'volume_trend': 'NO_DATA', 'volume_strength': 'LOW', 'volume_ratio': 1.0}
    
    def analyze_timeframe(self, timeframe_str: str) -> Dict:
        """Analyze single timeframe for MACD + RSI signals"""
        try:
            # Get data
            df = self.get_rates(timeframe_str, 200)
            if df is None:
                return {}
            
            # Calculate indicators
            df = self.calculate_macd(df)
            df = self.calculate_rsi(df)
            
            # Detect signals
            macd_signals = self.detect_macd_signals(df)
            rsi_signals = self.detect_rsi_signals(df)
            volume_info = self.calculate_volume_confirmation(df)
            
            # Current values
            current_data = {
                'price': df['close'].iloc[-1],
                'macd': {
                    'macd_line': round(df['MACD'].iloc[-1], 4),
                    'signal_line': round(df['MACD_Signal'].iloc[-1], 4),
                    'histogram': round(df['MACD_Histogram'].iloc[-1], 4),
                    'trend': df['MACD_Trend'].iloc[-1]
                },
                'rsi': {
                    'value': round(df['RSI'].iloc[-1], 2),
                    'condition': rsi_signals['condition'],
                    'trend': rsi_signals['trend']
                },
                'signals': {
                    'macd': macd_signals,
                    'rsi': rsi_signals
                },
                'volume': volume_info
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe_str}: {str(e)}")
            return {}
    
    def calculate_confluence_score(self, timeframe_analysis: Dict) -> float:
        """Calculate confluence score based on all timeframes"""
        try:
            total_score = 0.0
            max_score = 10.0
            
            # Timeframe weights
            tf_weights = {
                'H4': 0.30,
                'H1': 0.25,
                'M15': 0.20,
                'M5': 0.15,
                'M1': 0.10
            }
            
            bullish_signals = 0
            bearish_signals = 0
            total_weight = 0
            
            for tf, weight in tf_weights.items():
                if tf not in timeframe_analysis:
                    continue
                    
                tf_data = timeframe_analysis[tf]
                tf_score = 0
                
                # MACD Analysis
                macd_signals = tf_data.get('signals', {}).get('macd', {})
                
                # MACD Crossover signals
                if macd_signals.get('crossover'):
                    crossover = macd_signals['crossover']
                    if crossover['type'] == 'BULLISH_CROSSOVER':
                        tf_score += 3
                        bullish_signals += 1
                    elif crossover['type'] == 'BEARISH_CROSSOVER':
                        tf_score += 3
                        bearish_signals += 1
                
                # MACD Zero line
                if macd_signals.get('zero_line') == 'BULLISH_ZERO_CROSS':
                    tf_score += 2
                    bullish_signals += 1
                elif macd_signals.get('zero_line') == 'BEARISH_ZERO_CROSS':
                    tf_score += 2
                    bearish_signals += 1
                
                # MACD Histogram trend
                if macd_signals.get('histogram_trend') == 'INCREASING':
                    tf_score += 1
                    bullish_signals += 0.5
                elif macd_signals.get('histogram_trend') == 'DECREASING':
                    tf_score += 1
                    bearish_signals += 0.5
                
                # RSI Analysis
                rsi_signals = tf_data.get('signals', {}).get('rsi', {})
                
                # RSI Divergence
                if rsi_signals.get('divergence') == 'BULLISH_DIVERGENCE':
                    tf_score += 3
                    bullish_signals += 1
                elif rsi_signals.get('divergence') == 'BEARISH_DIVERGENCE':
                    tf_score += 3
                    bearish_signals += 1
                
                # RSI Momentum
                momentum = rsi_signals.get('momentum', '')
                if 'BULLISH' in momentum:
                    tf_score += 1 if 'STRONG' in momentum else 0.5
                    bullish_signals += 0.5
                elif 'BEARISH' in momentum:
                    tf_score += 1 if 'STRONG' in momentum else 0.5
                    bearish_signals += 0.5
                
                # Volume confirmation
                volume_info = tf_data.get('volume', {})
                if volume_info.get('volume_strength') in ['HIGH', 'MEDIUM']:
                    tf_score += 1
                
                # Add weighted score
                total_score += (tf_score / 10) * weight * max_score
                total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                normalized_score = total_score / total_weight
            else:
                normalized_score = 0
            
            # Confluence bonus
            signal_ratio = abs(bullish_signals - bearish_signals) / max(bullish_signals + bearish_signals, 1)
            confluence_bonus = signal_ratio * 2  # Up to 2 points bonus for strong confluence
            
            final_score = min(normalized_score + confluence_bonus, 10.0)
            
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating confluence score: {str(e)}")
            return 0.0
    
    def determine_signal_direction(self, timeframe_analysis: Dict) -> Tuple[str, str]:
        """Determine overall signal direction and confidence"""
        try:
            bullish_votes = 0
            bearish_votes = 0
            
            # Weight votes by timeframe importance
            tf_importance = {
                'H4': 3,
                'H1': 2.5,
                'M15': 2,
                'M5': 1.5,
                'M1': 1
            }
            
            for tf, importance in tf_importance.items():
                if tf not in timeframe_analysis:
                    continue
                    
                tf_data = timeframe_analysis[tf]
                
                # MACD signals
                macd_signals = tf_data.get('signals', {}).get('macd', {})
                macd_trend = tf_data.get('macd', {}).get('trend', 'NEUTRAL')
                
                # RSI signals
                rsi_signals = tf_data.get('signals', {}).get('rsi', {})
                rsi_trend = rsi_signals.get('trend', 'NEUTRAL')
                
                # Vote based on crossovers
                crossover = macd_signals.get('crossover')
                if crossover:
                    if crossover['type'] == 'BULLISH_CROSSOVER':
                        bullish_votes += importance * 2
                    elif crossover['type'] == 'BEARISH_CROSSOVER':
                        bearish_votes += importance * 2
                
                # Vote based on trends
                if macd_trend == 'BULLISH' and rsi_trend == 'BULLISH':
                    bullish_votes += importance
                elif macd_trend == 'BEARISH' and rsi_trend == 'BEARISH':
                    bearish_votes += importance
                
                # Vote based on divergences
                if (macd_signals.get('divergence') == 'BULLISH_DIVERGENCE' or 
                    rsi_signals.get('divergence') == 'BULLISH_DIVERGENCE'):
                    bullish_votes += importance * 1.5
                elif (macd_signals.get('divergence') == 'BEARISH_DIVERGENCE' or 
                      rsi_signals.get('divergence') == 'BEARISH_DIVERGENCE'):
                    bearish_votes += importance * 1.5
            
            # Determine direction
            total_votes = bullish_votes + bearish_votes
            if total_votes == 0:
                return "NONE", "LOW"
            
            vote_ratio = max(bullish_votes, bearish_votes) / total_votes
            
            # Signal direction
            if bullish_votes > bearish_votes:
                direction = "BUY"
            elif bearish_votes > bullish_votes:
                direction = "SELL"
            else:
                direction = "NONE"
            
            # Confidence level
            if vote_ratio >= 0.8:
                confidence = "VERY_HIGH"
            elif vote_ratio >= 0.7:
                confidence = "HIGH"
            elif vote_ratio >= 0.6:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            return direction, confidence
            
        except Exception as e:
            self.logger.error(f"Error determining signal direction: {str(e)}")
            return "NONE", "LOW"
    
    def generate_confluence_signals(self) -> Dict:
        """Generate comprehensive MACD + RSI confluence signals"""
        try:
            # Analyze all timeframes
            timeframe_analysis = {}
            for tf in self.timeframes.keys():
                self.logger.info(f"Analyzing {tf} for MACD + RSI...")
                timeframe_analysis[tf] = self.analyze_timeframe(tf)
            
            # Calculate confluence score
            confluence_score = self.calculate_confluence_score(timeframe_analysis)
            
            # Determine signal direction
            signal_direction, confidence = self.determine_signal_direction(timeframe_analysis)
            
            # Calculate risk/reward
            current_price = timeframe_analysis.get('M15', {}).get('price', 0)
            if current_price == 0:
                current_price = timeframe_analysis.get('H1', {}).get('price', 0)
            
            atr_approx = current_price * 0.005  # Approximate ATR for XAUUSD
            
            if signal_direction == "BUY":
                stop_loss = current_price - (atr_approx * 2)
                take_profit_1 = current_price + (atr_approx * 2)
                take_profit_2 = current_price + (atr_approx * 4)
                take_profit_3 = current_price + (atr_approx * 6)
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr_approx * 2)
                take_profit_1 = current_price - (atr_approx * 2)
                take_profit_2 = current_price - (atr_approx * 4)
                take_profit_3 = current_price - (atr_approx * 6)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Compile signal
            signal = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_direction': signal_direction,
                'confluence_score': confluence_score,
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'timeframe_analysis': timeframe_analysis,
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
            self.logger.error(f"Error generating confluence signals: {str(e)}")
            return {'error': str(e)}
    
    def print_confluence_summary(self, signal: Dict):
        """Print detailed confluence analysis"""
        if 'error' in signal:
            print(f"❌ ERROR: {signal['error']}")
            return
        
        print("\n" + "="*70)
        print("📊 MACD + RSI CONFLUENCE ANALYSIS")
        print("="*70)
        print(f"📅 Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Symbol: {signal['symbol']}")
        print(f"💵 Current Price: ${signal['current_price']}")
        print(f"📊 Confluence Score: {signal['confluence_score']}/10")
        print(f"🎯 Confidence: {signal['confidence']}")
        
        # Signal Direction
        direction_emoji = "🟢" if signal['signal_direction'] == "BUY" else "🔴" if signal['signal_direction'] == "SELL" else "⚪"
        print(f"{direction_emoji} Signal: {signal['signal_direction']}")
        
        print(f"\n🎯 RISK/REWARD SETUP:")
        if signal['signal_direction'] != "NONE":
            print(f"   Stop Loss: ${signal['risk_reward']['stop_loss']}")
            print(f"   Take Profit 1: ${signal['risk_reward']['take_profit_1']} (1:1)")
            print(f"   Take Profit 2: ${signal['risk_reward']['take_profit_2']} (1:2)")
            print(f"   Take Profit 3: ${signal['risk_reward']['take_profit_3']} (1:3)")
        else:
            print("   No active signal - Wait for confluence")
        
        print(f"\n📈 TIMEFRAME CONFLUENCE:")
        for tf, data in signal['timeframe_analysis'].items():
            if not data:
                continue
                
            macd_trend = data.get('macd', {}).get('trend', 'N/A')
            rsi_value = data.get('rsi', {}).get('value', 0)
            rsi_condition = data.get('rsi', {}).get('condition', 'N/A')
            
            # Signal indicators
            signals_text = ""
            macd_signals = data.get('signals', {}).get('macd', {})
            rsi_signals = data.get('signals', {}).get('rsi', {})
            
            if macd_signals.get('crossover'):
                signals_text += f"📈{macd_signals['crossover']['type']} "
            if macd_signals.get('divergence'):
                signals_text += f"🔄{macd_signals['divergence']} "
            if rsi_signals.get('divergence'):
                signals_text += f"📊{rsi_signals['divergence']} "
            
            print(f"   {tf}: MACD={macd_trend} | RSI={rsi_value:.1f}({rsi_condition}) {signals_text}")
        
        print("="*70)

def main():
    """Main execution function"""
    print("🚀 Starting MACD + RSI Confluence System...")
    
    # Initialize system
    confluence_system = MACDRSIConfluenceSystem("XAUUSD.c")
    
    # Connect to MT5
    if not confluence_system.connect_mt5():
        print("❌ Failed to connect to MT5. Please check your connection.")
        return
    
    try:
        # Generate confluence signals
        print("📊 Analyzing MACD + RSI confluence across timeframes...")
        signal = confluence_system.generate_confluence_signals()
        
        # Print results
        confluence_system.print_confluence_summary(signal)
        
        print(f"\n💡 TIP: Look for confluence score >6.0 for high-probability trades!")
        print(f"💡 TIP: Combine with volume confirmation for better results!")
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        mt5.shutdown()
        print("🔌 MT5 connection closed")

if __name__ == "__main__":
    main()