"""
📟 CONSOLE-BASED TRADING DASHBOARD
==================================
Lightweight Dashboard for VPS/Server Environment
Terminal-Based UI with Real-time Updates
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import threading
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ConsoleDashboard:
    """Lightweight console-based trading dashboard"""
    
    def __init__(self, symbol: str = "XAUUSD.c"):
        """Initialize console dashboard"""
        self.symbol = symbol
        self.is_running = False
        self.update_interval = 30  # seconds
        
        # Dashboard data
        self.data = {
            'current_price': 0.0,
            'previous_price': 0.0,
            'signal_direction': 'NONE',
            'strength_score': 0.0,
            'confidence': 'LOW',
            'rsi': 0.0,
            'atr': 0.0,
            'trend_strength': 0.0,
            'volume_ratio': 0.0,
            'stop_loss': 0.0,
            'take_profit_1': 0.0,
            'take_profit_2': 0.0,
            'take_profit_3': 0.0,
            'timeframes': {},
            'filters_passed': [],
            'filters_failed': [],
            'session_stats': {
                'total_signals': 0,
                'high_quality_signals': 0,
                'filtered_signals': 0,
                'start_time': datetime.now(),
                'last_signal_time': None
            },
            'alerts': []
        }
        
        # Optimized system parameters (simplified)
        self.min_confluence_score = 7.5
        self.min_timeframes_agreement = 4
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for console dashboard"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('console_dashboard.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
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
                    
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def get_current_price(self) -> Optional[float]:
        """Get current price from MT5"""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            return tick.bid
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate basic indicators for console display"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # EMA calculations
            ema_9 = close.ewm(span=9).mean().iloc[-1]
            ema_21 = close.ewm(span=21).mean().iloc[-1]
            ema_50 = close.ewm(span=50).mean().iloc[-1]
            
            # RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Simple trend strength
            current_price = close.iloc[-1]
            trend_conditions = [
                current_price > ema_9,
                ema_9 > ema_21,
                ema_21 > ema_50
            ]
            trend_strength = sum(trend_conditions) / len(trend_conditions)
            
            # Volume ratio (simplified)
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            volume_avg = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
            
            return {
                'rsi': rsi,
                'atr': atr,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_50': ema_50
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def analyze_signal(self, indicators: Dict, current_price: float) -> Dict:
        """Analyze current trading signal"""
        try:
            signal_direction = 'NONE'
            strength_score = 0.0
            confidence = 'LOW'
            filters_passed = []
            filters_failed = []
            
            if not indicators:
                return {
                    'direction': signal_direction,
                    'strength': strength_score,
                    'confidence': confidence,
                    'filters_passed': filters_passed,
                    'filters_failed': filters_failed
                }
            
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            
            # Filter checks
            if 35 <= rsi <= 65:
                filters_passed.append("RSI in normal range")
                strength_score += 2.0
            else:
                filters_failed.append(f"RSI extreme: {rsi:.1f}")
            
            if trend_strength >= 0.7:
                filters_passed.append("Strong trend detected")
                strength_score += 3.0
            else:
                filters_failed.append(f"Weak trend: {trend_strength:.2f}")
            
            if volume_ratio >= 1.2:
                filters_passed.append("Volume confirmation")
                strength_score += 1.5
            else:
                filters_failed.append(f"Low volume: {volume_ratio:.2f}")
            
            # Signal logic
            if current_price > ema_9 > ema_21 and trend_strength >= 0.7:
                signal_direction = 'BUY'
                strength_score += 2.0
            elif current_price < ema_9 < ema_21 and trend_strength >= 0.7:
                signal_direction = 'SELL'
                strength_score += 2.0
            
            # Confidence levels
            if strength_score >= 8.0 and len(filters_failed) == 0:
                confidence = 'VERY_HIGH'
            elif strength_score >= 7.0 and len(filters_failed) <= 1:
                confidence = 'HIGH'
            elif strength_score >= 5.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            # Apply minimum threshold
            if strength_score < self.min_confluence_score:
                signal_direction = 'NONE'
                confidence = 'FILTERED'
            
            return {
                'direction': signal_direction,
                'strength': round(strength_score, 2),
                'confidence': confidence,
                'filters_passed': filters_passed,
                'filters_failed': filters_failed
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal: {str(e)}")
            return {
                'direction': 'NONE',
                'strength': 0.0,
                'confidence': 'ERROR',
                'filters_passed': [],
                'filters_failed': ['Analysis error']
            }
    
    def calculate_trading_levels(self, current_price: float, signal_direction: str, atr: float) -> Dict:
        """Calculate trading levels"""
        try:
            if signal_direction == 'NONE':
                return {
                    'stop_loss': 0,
                    'take_profit_1': 0,
                    'take_profit_2': 0,
                    'take_profit_3': 0
                }
            
            if signal_direction == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                take_profit_1 = current_price + (atr * 2.5)
                take_profit_2 = current_price + (atr * 4.0)
                take_profit_3 = current_price + (atr * 6.0)
            else:  # SELL
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            
            return {
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'take_profit_3': round(take_profit_3, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trading levels: {str(e)}")
            return {'stop_loss': 0, 'take_profit_1': 0, 'take_profit_2': 0, 'take_profit_3': 0}
    
    def update_data(self):
        """Update dashboard data"""
        try:
            # Get current price
            current_price = self.get_current_price()
            if current_price is None:
                return
            
            # Store previous price for change calculation
            if self.data['current_price'] > 0:
                self.data['previous_price'] = self.data['current_price']
            self.data['current_price'] = current_price
            
            # Get market data for analysis
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return
            
            df = pd.DataFrame(rates)
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            
            # Update indicator values
            self.data.update({
                'rsi': round(indicators.get('rsi', 0), 1),
                'atr': round(indicators.get('atr', 0), 4),
                'trend_strength': round(indicators.get('trend_strength', 0), 2),
                'volume_ratio': round(indicators.get('volume_ratio', 1), 2)
            })
            
            # Analyze signal
            signal_analysis = self.analyze_signal(indicators, current_price)
            
            # Update signal data
            previous_signal = self.data['signal_direction']
            self.data.update({
                'signal_direction': signal_analysis['direction'],
                'strength_score': signal_analysis['strength'],
                'confidence': signal_analysis['confidence'],
                'filters_passed': signal_analysis['filters_passed'],
                'filters_failed': signal_analysis['filters_failed']
            })
            
            # Calculate trading levels
            atr = indicators.get('atr', current_price * 0.005)
            levels = self.calculate_trading_levels(current_price, signal_analysis['direction'], atr)
            self.data.update(levels)
            
            # Update session stats
            if previous_signal != signal_analysis['direction'] and signal_analysis['direction'] != 'NONE':
                self.data['session_stats']['total_signals'] += 1
                self.data['session_stats']['last_signal_time'] = datetime.now()
                
                if signal_analysis['confidence'] in ['HIGH', 'VERY_HIGH']:
                    self.data['session_stats']['high_quality_signals'] += 1
                    
                # Add alert
                alert_msg = f"{signal_analysis['direction']} signal - {signal_analysis['confidence']} confidence"
                self.add_alert(alert_msg)
            
            if signal_analysis['confidence'] == 'FILTERED':
                self.data['session_stats']['filtered_signals'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
    
    def add_alert(self, message: str):
        """Add alert to log"""
        alert = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'message': message
        }
        self.data['alerts'].insert(0, alert)
        
        # Keep only last 5 alerts
        if len(self.data['alerts']) > 5:
            self.data['alerts'] = self.data['alerts'][:5]
    
    def format_price_change(self) -> str:
        """Format price change display"""
        if self.data['previous_price'] == 0:
            return "±0.00 (±0.00%)"
        
        change = self.data['current_price'] - self.data['previous_price']
        change_percent = (change / self.data['previous_price']) * 100
        
        if change >= 0:
            return f"+{change:.2f} (+{change_percent:.2f}%)"
        else:
            return f"{change:.2f} ({change_percent:.2f}%)"
    
    def get_signal_emoji(self) -> str:
        """Get emoji for signal direction"""
        if self.data['signal_direction'] == 'BUY':
            return "🟢"
        elif self.data['signal_direction'] == 'SELL':
            return "🔴"
        else:
            return "⚪"
    
    def get_confidence_bar(self, length: int = 20) -> str:
        """Create ASCII confidence bar"""
        confidence_levels = {'VERY_HIGH': 100, 'HIGH': 80, 'MEDIUM': 60, 'LOW': 40, 'FILTERED': 0}
        percent = confidence_levels.get(self.data['confidence'], 0)
        
        filled = int((percent / 100) * length)
        empty = length - filled
        
        return "█" * filled + "░" * empty
    
    def display_dashboard(self):
        """Display console dashboard"""
        self.clear_screen()
        
        # Header
        print("=" * 80)
        print("🎯 PROFESSIONAL TRADING DASHBOARD - VPS OPTIMIZED")
        print("=" * 80)
        print(f"📊 Symbol: {self.symbol} | 🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 📡 LIVE")
        print("=" * 80)
        
        # Current Price Section
        price_change = self.format_price_change()
        print(f"\n💰 CURRENT PRICE")
        print(f"   Price: ${self.data['current_price']:.2f}")
        print(f"   Change: {price_change}")
        
        # Current Signal Section
        signal_emoji = self.get_signal_emoji()
        confidence_bar = self.get_confidence_bar()
        print(f"\n📊 CURRENT SIGNAL")
        print(f"   Direction: {signal_emoji} {self.data['signal_direction']}")
        print(f"   Strength: {self.data['strength_score']}/10")
        print(f"   Confidence: {self.data['confidence']}")
        print(f"   Meter: [{confidence_bar}] {self.data['confidence']}")
        
        # Trading Levels
        if self.data['signal_direction'] != 'NONE':
            print(f"\n🎯 TRADING LEVELS")
            print(f"   Stop Loss: ${self.data['stop_loss']:.2f}")
            print(f"   Take Profit 1: ${self.data['take_profit_1']:.2f}")
            print(f"   Take Profit 2: ${self.data['take_profit_2']:.2f}")
            print(f"   Take Profit 3: ${self.data['take_profit_3']:.2f}")
        else:
            print(f"\n🎯 TRADING LEVELS")
            print(f"   No active signal - Waiting for setup")
        
        # Market Analysis
        print(f"\n📈 MARKET ANALYSIS")
        print(f"   RSI: {self.data['rsi']:.1f}")
        print(f"   ATR: {self.data['atr']:.4f}")
        print(f"   Trend Strength: {self.data['trend_strength']:.2f}")
        print(f"   Volume Ratio: {self.data['volume_ratio']:.2f}")
        
        # Filters Status
        print(f"\n🔍 FILTERS STATUS")
        print(f"   ✅ Passed: {len(self.data['filters_passed'])}")
        for filter_pass in self.data['filters_passed'][:3]:  # Show top 3
            print(f"      • {filter_pass}")
        
        print(f"   ❌ Failed: {len(self.data['filters_failed'])}")
        for filter_fail in self.data['filters_failed'][:3]:  # Show top 3
            print(f"      • {filter_fail}")
        
        # Session Statistics
        uptime = datetime.now() - self.data['session_stats']['start_time']
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        print(f"\n📊 SESSION STATISTICS")
        print(f"   Total Signals: {self.data['session_stats']['total_signals']}")
        print(f"   High Quality: {self.data['session_stats']['high_quality_signals']}")
        print(f"   Filtered: {self.data['session_stats']['filtered_signals']}")
        print(f"   Uptime: {uptime_str}")
        
        last_signal = self.data['session_stats']['last_signal_time']
        if last_signal:
            print(f"   Last Signal: {last_signal.strftime('%H:%M:%S')}")
        else:
            print(f"   Last Signal: None")
        
        # Recent Alerts
        print(f"\n🔔 RECENT ALERTS")
        if self.data['alerts']:
            for alert in self.data['alerts']:
                print(f"   {alert['time']} - {alert['message']}")
        else:
            print(f"   No alerts yet...")
        
        # Footer
        print("\n" + "=" * 80)
        print("⏹️ Press Ctrl+C to stop monitoring | 🔄 Auto-refresh every 30 seconds")
        print("=" * 80)
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                self.update_data()
                self.display_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def start_monitoring(self):
        """Start console monitoring"""
        try:
            # Connect to MT5
            if not self.connect_mt5():
                print("❌ Failed to connect to MT5")
                return
            
            print("🚀 Connecting to MT5...")
            time.sleep(2)
            
            self.is_running = True
            self.add_alert("Console dashboard started")
            
            # Start monitoring loop
            self.monitoring_loop()
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping console dashboard...")
            self.stop_monitoring()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        
        # Display final stats
        uptime = datetime.now() - self.data['session_stats']['start_time']
        print(f"\n📊 FINAL SESSION SUMMARY")
        print(f"   Session Duration: {str(uptime).split('.')[0]}")
        print(f"   Total Signals: {self.data['session_stats']['total_signals']}")
        print(f"   High Quality Signals: {self.data['session_stats']['high_quality_signals']}")
        print(f"   Filtered Signals: {self.data['session_stats']['filtered_signals']}")
        
        mt5.shutdown()
        print("🔌 MT5 connection closed")
        print("👋 Console dashboard stopped")

def main():
    """Main execution"""
    print("📟 Starting Console Trading Dashboard...")
    print("🖥️ VPS Optimized - Lightweight Interface")
    
    # Initialize console dashboard
    dashboard = ConsoleDashboard("XAUUSD.c")
    
    # Start monitoring
    dashboard.start_monitoring()

if __name__ == "__main__":
    main()