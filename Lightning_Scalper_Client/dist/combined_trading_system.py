"""
🎯 PROFESSIONAL COMBINED TRADING SYSTEM
======================================
EMA + MACD + RSI Confluence with Advanced Filtering
Reduces False Signals & Improves Signal Quality
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CombinedTradingSystem:
    """Professional Combined EMA + MACD + RSI Trading System"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize Combined Trading System"""
        self.symbol = symbol
        self.timeframes = {
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5,
            'M1': mt5.TIMEFRAME_M1
        }
        
        # EMA Parameters
        self.ema_periods = [9, 21, 50, 200]
        
        # MACD Parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # RSI Parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Signal filtering thresholds
        self.min_confluence_score = 6.0
        self.min_timeframes_agreement = 3
        self.max_rsi_extreme = 25  # Don't trade if RSI < 25 or > 75
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('combined_trading.log'),
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
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators: EMA, MACD, RSI"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # EMA Calculations
            for period in self.ema_periods:
                df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            
            # MACD Calculations
            ema_fast = close.ewm(span=self.macd_fast).mean()
            ema_slow = close.ewm(span=self.macd_slow).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI Calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # ATR for volatility measurement
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def analyze_ema_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze EMA-based signals"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = {
                'trend': 'NEUTRAL',
                'strength': 0,
                'crossovers': [],
                'price_position': 'NEUTRAL'
            }
            
            # Get current EMAs
            ema_9 = latest['EMA_9']
            ema_21 = latest['EMA_21']
            ema_50 = latest['EMA_50']
            ema_200 = latest['EMA_200']
            current_price = latest['close']
            
            # Trend Analysis
            if current_price > ema_9 > ema_21 > ema_50 > ema_200:
                signals['trend'] = 'STRONG_BULLISH'
                signals['strength'] = 10
            elif current_price > ema_9 > ema_21 and ema_21 > ema_50:
                signals['trend'] = 'BULLISH'
                signals['strength'] = 7
            elif current_price < ema_9 < ema_21 < ema_50 < ema_200:
                signals['trend'] = 'STRONG_BEARISH'
                signals['strength'] = 10
            elif current_price < ema_9 < ema_21 and ema_21 < ema_50:
                signals['trend'] = 'BEARISH'
                signals['strength'] = 7
            else:
                signals['trend'] = 'SIDEWAYS'
                signals['strength'] = 3
            
            # Crossover Detection
            crossover_pairs = [('EMA_9', 'EMA_21'), ('EMA_21', 'EMA_50')]
            
            for fast, slow in crossover_pairs:
                curr_fast = latest[fast]
                curr_slow = latest[slow]
                prev_fast = prev[fast]
                prev_slow = prev[slow]
                
                if prev_fast <= prev_slow and curr_fast > curr_slow:
                    signals['crossovers'].append(f'{fast}_ABOVE_{slow}')
                elif prev_fast >= prev_slow and curr_fast < curr_slow:
                    signals['crossovers'].append(f'{fast}_BELOW_{slow}')
            
            # Price Position relative to EMAs
            if current_price > ema_9:
                signals['price_position'] = 'ABOVE_FAST_EMA'
            elif current_price < ema_9:
                signals['price_position'] = 'BELOW_FAST_EMA'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing EMA signals: {str(e)}")
            return {'trend': 'NEUTRAL', 'strength': 0, 'crossovers': [], 'price_position': 'NEUTRAL'}
    
    def analyze_macd_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze MACD-based signals"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = {
                'trend': 'NEUTRAL',
                'crossover': None,
                'histogram_trend': 'NEUTRAL',
                'zero_line_position': 'NEUTRAL',
                'strength': 0
            }
            
            current_macd = latest['MACD']
            current_signal = latest['MACD_Signal']
            current_hist = latest['MACD_Histogram']
            
            prev_macd = prev['MACD']
            prev_signal = prev['MACD_Signal']
            
            # MACD Trend
            if current_macd > current_signal:
                signals['trend'] = 'BULLISH'
                signals['strength'] += 5
            else:
                signals['trend'] = 'BEARISH'
                signals['strength'] += 5
            
            # Crossover Detection
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals['crossover'] = 'BULLISH_CROSSOVER'
                signals['strength'] += 3
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals['crossover'] = 'BEARISH_CROSSOVER'
                signals['strength'] += 3
            
            # Histogram Trend (last 3 bars)
            hist_trend = df['MACD_Histogram'].iloc[-3:].diff().mean()
            if hist_trend > 0:
                signals['histogram_trend'] = 'INCREASING'
                signals['strength'] += 2
            elif hist_trend < 0:
                signals['histogram_trend'] = 'DECREASING'
                signals['strength'] += 2
            
            # Zero line position
            if current_macd > 0:
                signals['zero_line_position'] = 'ABOVE_ZERO'
                signals['strength'] += 1
            else:
                signals['zero_line_position'] = 'BELOW_ZERO'
                signals['strength'] += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing MACD signals: {str(e)}")
            return {'trend': 'NEUTRAL', 'crossover': None, 'histogram_trend': 'NEUTRAL', 
                   'zero_line_position': 'NEUTRAL', 'strength': 0}
    
    def analyze_rsi_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI-based signals with extreme filtering"""
        try:
            current_rsi = df['RSI'].iloc[-1]
            
            signals = {
                'value': round(current_rsi, 2),
                'condition': 'NEUTRAL',
                'trend': 'NEUTRAL',
                'extreme_warning': False,
                'strength': 0
            }
            
            # RSI Conditions
            if current_rsi > self.rsi_overbought:
                signals['condition'] = 'OVERBOUGHT'
                signals['extreme_warning'] = current_rsi > 75
            elif current_rsi < self.rsi_oversold:
                signals['condition'] = 'OVERSOLD'
                signals['extreme_warning'] = current_rsi < 25
            else:
                signals['condition'] = 'NEUTRAL'
            
            # RSI Trend
            if current_rsi > 50:
                signals['trend'] = 'BULLISH'
                signals['strength'] = min((current_rsi - 50) / 10, 5)
            else:
                signals['trend'] = 'BEARISH'
                signals['strength'] = min((50 - current_rsi) / 10, 5)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing RSI signals: {str(e)}")
            return {'value': 50, 'condition': 'NEUTRAL', 'trend': 'NEUTRAL', 
                   'extreme_warning': False, 'strength': 0}
    
    def calculate_timeframe_confluence(self, timeframe_analysis: Dict) -> Dict:
        """Calculate confluence across timeframes with advanced filtering"""
        try:
            confluence = {
                'score': 0.0,
                'agreement_count': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'conflicts': [],
                'warnings': []
            }
            
            # Timeframe weights
            tf_weights = {
                'H4': 0.30,
                'H1': 0.25,
                'M15': 0.20,
                'M5': 0.15,
                'M1': 0.10
            }
            
            total_score = 0.0
            bullish_weight = 0.0
            bearish_weight = 0.0
            
            for tf, weight in tf_weights.items():
                if tf not in timeframe_analysis:
                    continue
                
                tf_data = timeframe_analysis[tf]
                tf_score = 0.0
                
                # EMA Analysis
                ema_signals = tf_data.get('ema', {})
                ema_trend = ema_signals.get('trend', 'NEUTRAL')
                ema_strength = ema_signals.get('strength', 0)
                
                if ema_trend in ['STRONG_BULLISH', 'BULLISH']:
                    tf_score += ema_strength * 0.4
                    bullish_weight += weight
                    confluence['bullish_signals'] += 1
                elif ema_trend in ['STRONG_BEARISH', 'BEARISH']:
                    tf_score += ema_strength * 0.4
                    bearish_weight += weight
                    confluence['bearish_signals'] += 1
                
                # MACD Analysis
                macd_signals = tf_data.get('macd', {})
                macd_trend = macd_signals.get('trend', 'NEUTRAL')
                macd_strength = macd_signals.get('strength', 0)
                
                if macd_trend == 'BULLISH':
                    tf_score += macd_strength * 0.3
                    if ema_trend not in ['BULLISH', 'STRONG_BULLISH']:
                        confluence['conflicts'].append(f'{tf}: EMA-MACD conflict')
                elif macd_trend == 'BEARISH':
                    tf_score += macd_strength * 0.3
                    if ema_trend not in ['BEARISH', 'STRONG_BEARISH']:
                        confluence['conflicts'].append(f'{tf}: EMA-MACD conflict')
                
                # RSI Analysis with extreme filtering
                rsi_signals = tf_data.get('rsi', {})
                rsi_trend = rsi_signals.get('trend', 'NEUTRAL')
                rsi_extreme = rsi_signals.get('extreme_warning', False)
                
                if rsi_extreme:
                    confluence['warnings'].append(f'{tf}: RSI extreme level {rsi_signals.get("value", 0)}')
                    tf_score *= 0.5  # Reduce score for extreme RSI
                
                if rsi_trend == 'BULLISH' and not rsi_extreme:
                    tf_score += 3.0
                elif rsi_trend == 'BEARISH' and not rsi_extreme:
                    tf_score += 3.0
                
                # Add to total score
                total_score += (tf_score / 15) * weight * 10  # Normalize to 0-10
            
            # Calculate final confluence score
            confluence['score'] = min(total_score, 10.0)
            
            # Agreement analysis
            total_signals = confluence['bullish_signals'] + confluence['bearish_signals']
            if total_signals > 0:
                agreement_ratio = max(confluence['bullish_signals'], confluence['bearish_signals']) / total_signals
                confluence['agreement_count'] = int(agreement_ratio * len(tf_weights))
            
            return confluence
            
        except Exception as e:
            self.logger.error(f"Error calculating confluence: {str(e)}")
            return {'score': 0.0, 'agreement_count': 0, 'bullish_signals': 0, 
                   'bearish_signals': 0, 'conflicts': [], 'warnings': []}
    
    def apply_signal_filters(self, signal_data: Dict) -> Tuple[bool, List[str]]:
        """Apply professional signal filters"""
        try:
            filters_passed = True
            filter_reasons = []
            
            confluence_score = signal_data.get('confluence_score', 0)
            confluence_analysis = signal_data.get('confluence_analysis', {})
            
            # Filter 1: Minimum confluence score
            if confluence_score < self.min_confluence_score:
                filters_passed = False
                filter_reasons.append(f"Low confluence score: {confluence_score} < {self.min_confluence_score}")
            
            # Filter 2: Minimum timeframe agreement
            agreement_count = confluence_analysis.get('agreement_count', 0)
            if agreement_count < self.min_timeframes_agreement:
                filters_passed = False
                filter_reasons.append(f"Insufficient timeframe agreement: {agreement_count} < {self.min_timeframes_agreement}")
            
            # Filter 3: RSI extreme levels
            warnings = confluence_analysis.get('warnings', [])
            rsi_extreme_warnings = [w for w in warnings if 'extreme level' in w]
            if rsi_extreme_warnings:
                filters_passed = False
                filter_reasons.append(f"RSI extreme levels detected: {rsi_extreme_warnings}")
            
            # Filter 4: Major conflicts between indicators
            conflicts = confluence_analysis.get('conflicts', [])
            if len(conflicts) >= 3:  # Too many conflicts
                filters_passed = False
                filter_reasons.append(f"Too many indicator conflicts: {conflicts}")
            
            # Filter 5: Signal strength vs confluence mismatch
            if confluence_score > 7.0 and agreement_count < 2:
                filters_passed = False
                filter_reasons.append("High score but low agreement - potential false signal")
            
            return filters_passed, filter_reasons
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
            return False, [f"Filter error: {str(e)}"]
    
    def analyze_all_timeframes(self) -> Dict:
        """Analyze all timeframes with combined indicators"""
        try:
            timeframe_analysis = {}
            
            for tf_name in self.timeframes.keys():
                self.logger.info(f"Analyzing {tf_name} with combined indicators...")
                
                # Get data
                df = self.get_rates(tf_name, 300)
                if df is None:
                    continue
                
                # Calculate all indicators
                df = self.calculate_all_indicators(df)
                
                # Analyze each indicator type
                ema_analysis = self.analyze_ema_signals(df)
                macd_analysis = self.analyze_macd_signals(df)
                rsi_analysis = self.analyze_rsi_signals(df)
                
                timeframe_analysis[tf_name] = {
                    'price': df['close'].iloc[-1],
                    'atr': df['ATR'].iloc[-1],
                    'ema': ema_analysis,
                    'macd': macd_analysis,
                    'rsi': rsi_analysis
                }
            
            return timeframe_analysis
            
        except Exception as e:
            self.logger.error(f"Error in timeframe analysis: {str(e)}")
            return {}
    
    def determine_signal_direction(self, confluence_analysis: Dict) -> Tuple[str, str, List[str]]:
        """Determine signal direction with reasoning"""
        try:
            bullish_signals = confluence_analysis.get('bullish_signals', 0)
            bearish_signals = confluence_analysis.get('bearish_signals', 0)
            score = confluence_analysis.get('score', 0)
            conflicts = confluence_analysis.get('conflicts', [])
            warnings = confluence_analysis.get('warnings', [])
            
            reasoning = []
            
            # Determine direction
            if bullish_signals > bearish_signals and len(conflicts) <= 1:
                direction = "BUY"
                reasoning.append(f"Bullish signals: {bullish_signals} vs Bearish: {bearish_signals}")
            elif bearish_signals > bullish_signals and len(conflicts) <= 1:
                direction = "SELL" 
                reasoning.append(f"Bearish signals: {bearish_signals} vs Bullish: {bullish_signals}")
            else:
                direction = "NONE"
                reasoning.append(f"Mixed signals or too many conflicts: {len(conflicts)}")
            
            # Determine confidence
            if score >= 8.0 and len(conflicts) == 0 and len(warnings) == 0:
                confidence = "VERY_HIGH"
            elif score >= 7.0 and len(conflicts) <= 1:
                confidence = "HIGH"
            elif score >= 6.0 and len(conflicts) <= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
                
            if conflicts:
                reasoning.append(f"Conflicts detected: {conflicts}")
            if warnings:
                reasoning.append(f"Warnings: {warnings}")
            
            return direction, confidence, reasoning
            
        except Exception as e:
            self.logger.error(f"Error determining signal direction: {str(e)}")
            return "NONE", "LOW", [f"Error: {str(e)}"]
    
    def generate_combined_signals(self) -> Dict:
        """Generate combined trading signals with advanced filtering"""
        try:
            # Analyze all timeframes
            timeframe_analysis = self.analyze_all_timeframes()
            
            if not timeframe_analysis:
                return {'error': 'Failed to get timeframe analysis'}
            
            # Calculate confluence
            confluence_analysis = self.calculate_timeframe_confluence(timeframe_analysis)
            
            # Determine signal direction
            signal_direction, confidence, reasoning = self.determine_signal_direction(confluence_analysis)
            
            # Prepare signal data for filtering
            signal_data = {
                'confluence_score': confluence_analysis['score'],
                'confluence_analysis': confluence_analysis,
                'direction': signal_direction
            }
            
            # Apply professional filters
            filters_passed, filter_reasons = self.apply_signal_filters(signal_data)
            
            # Override signal if filters failed
            if not filters_passed:
                signal_direction = "NONE"
                confidence = "FILTERED"
                reasoning.extend(filter_reasons)
            
            # Calculate risk/reward
            current_price = timeframe_analysis.get('M15', {}).get('price', 0) or timeframe_analysis.get('H1', {}).get('price', 0)
            atr = timeframe_analysis.get('M15', {}).get('atr', current_price * 0.005)
            
            if signal_direction == "BUY":
                stop_loss = current_price - (atr * 2)
                take_profit_1 = current_price + (atr * 2)
                take_profit_2 = current_price + (atr * 4) 
                take_profit_3 = current_price + (atr * 6)
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr * 2)
                take_profit_1 = current_price - (atr * 2)
                take_profit_2 = current_price - (atr * 4)
                take_profit_3 = current_price - (atr * 6)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Compile final signal
            combined_signal = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_direction': signal_direction,
                'confluence_score': round(confluence_analysis['score'], 2),
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'filters_passed': filters_passed,
                'reasoning': reasoning,
                'confluence_analysis': confluence_analysis,
                'timeframe_analysis': timeframe_analysis,
                'risk_reward': {
                    'stop_loss': round(stop_loss, 2),
                    'take_profit_1': round(take_profit_1, 2),
                    'take_profit_2': round(take_profit_2, 2),
                    'take_profit_3': round(take_profit_3, 2),
                    'atr_used': round(atr, 2),
                    'risk_reward_ratio': 2.0 if signal_direction != "NONE" else 0
                }
            }
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error generating combined signals: {str(e)}")
            return {'error': str(e)}
    
    def print_combined_analysis(self, signal: Dict):
        """Print comprehensive combined analysis"""
        if 'error' in signal:
            print(f"❌ ERROR: {signal['error']}")
            return
        
        print("\n" + "="*80)
        print("🎯 PROFESSIONAL COMBINED TRADING ANALYSIS")
        print("EMA + MACD + RSI with Advanced Filtering")
        print("="*80)
        print(f"📅 Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Symbol: {signal['symbol']}")
        print(f"💵 Current Price: ${signal['current_price']}")
        print(f"📊 Confluence Score: {signal['confluence_score']}/10")
        print(f"🎯 Confidence: {signal['confidence']}")
        print(f"🔍 Filters Passed: {'✅ YES' if signal['filters_passed'] else '❌ NO'}")
        
        # Signal Direction
        direction_emoji = "🟢" if signal['signal_direction'] == "BUY" else "🔴" if signal['signal_direction'] == "SELL" else "⚪"
        print(f"{direction_emoji} FINAL SIGNAL: {signal['signal_direction']}")
        
        # Reasoning
        print(f"\n🧠 SIGNAL REASONING:")
        for reason in signal['reasoning']:
            print(f"   • {reason}")
        
        # Risk/Reward
        print(f"\n🎯 RISK/REWARD SETUP:")
        if signal['signal_direction'] != "NONE":
            print(f"   Stop Loss: ${signal['risk_reward']['stop_loss']}")
            print(f"   Take Profit 1: ${signal['risk_reward']['take_profit_1']} (1:1)")
            print(f"   Take Profit 2: ${signal['risk_reward']['take_profit_2']} (1:2)")
            print(f"   Take Profit 3: ${signal['risk_reward']['take_profit_3']} (1:3)")
            print(f"   ATR Used: ${signal['risk_reward']['atr_used']}")
        else:
            print("   No active signal - Waiting for better setup")
        
        # Confluence Analysis
        confluence = signal['confluence_analysis']
        print(f"\n📊 CONFLUENCE ANALYSIS:")
        print(f"   Bullish Signals: {confluence['bullish_signals']}")
        print(f"   Bearish Signals: {confluence['bearish_signals']}")
        print(f"   Agreement Count: {confluence['agreement_count']}/5 timeframes")
        
        if confluence['conflicts']:
            print(f"   ⚠️ Conflicts: {len(confluence['conflicts'])}")
            for conflict in confluence['conflicts']:
                print(f"      • {conflict}")
        
        if confluence['warnings']:
            print(f"   🚨 Warnings: {len(confluence['warnings'])}")
            for warning in confluence['warnings']:
                print(f"      • {warning}")
        
        # Timeframe Details
        print(f"\n📈 TIMEFRAME BREAKDOWN:")
        for tf, data in signal['timeframe_analysis'].items():
            ema_trend = data.get('ema', {}).get('trend', 'N/A')
            macd_trend = data.get('macd', {}).get('trend', 'N/A')
            rsi_value = data.get('rsi', {}).get('value', 0)
            rsi_condition = data.get('rsi', {}).get('condition', 'N/A')
            
            warning_emoji = "🚨" if data.get('rsi', {}).get('extreme_warning', False) else ""
            
            print(f"   {tf}: EMA={ema_trend} | MACD={macd_trend} | RSI={rsi_value:.1f}({rsi_condition}) {warning_emoji}")
        
        print("="*80)
        
        # Trading recommendation
        if signal['signal_direction'] != "NONE" and signal['filters_passed']:
            print(f"✅ RECOMMENDATION: Consider {signal['signal_direction']} position")
            print(f"💡 Confluence score {signal['confluence_score']}/10 meets minimum threshold")
        else:
            print("⏳ RECOMMENDATION: Wait for better setup")
            print("💡 Current market conditions do not meet trading criteria")

def main():
    """Main execution function"""
    print("🚀 Starting Professional Combined Trading System...")
    
    # Initialize system
    combined_system = CombinedTradingSystem("XAUUSD.c")
    
    # Connect to MT5
    if not combined_system.connect_mt5():
        print("❌ Failed to connect to MT5. Please check your connection.")
        return
    
    try:
        # Generate combined signals
        print("📊 Analyzing combined EMA + MACD + RSI signals...")
        signal = combined_system.generate_combined_signals()
        
        # Print comprehensive analysis
        combined_system.print_combined_analysis(signal)
        
        print(f"\n💡 SYSTEM FEATURES:")
        print(f"   ✅ Minimum confluence score: {combined_system.min_confluence_score}/10")
        print(f"   ✅ Minimum timeframe agreement: {combined_system.min_timeframes_agreement}/5")
        print(f"   ✅ RSI extreme filtering: Active")
        print(f"   ✅ Conflict detection: Active")
        print(f"   ✅ Professional risk management: Active")
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        mt5.shutdown()
        print("🔌 MT5 connection closed")

if __name__ == "__main__":
    main()