"""
Enhanced MT5 Dashboard with Risk Management & Entry/Exit Points
==============================================================
Advanced Trading Dashboard with Position Sizing and Risk Controls
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedRiskDashboard:
    """Enhanced Dashboard with Risk Management Features"""
    
    def __init__(self):
        """Initialize Enhanced Risk Dashboard"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Forex pairs
        self.forex_pairs = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c', 'USDCAD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'EURCAD.c',
            'GBPJPY.c', 'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'GBPCAD.c',
            'AUDCHF.c', 'AUDJPY.c', 'AUDNZD.c', 'AUDCAD.c',
            'NZDJPY.c', 'NZDCHF.c', 'NZDCAD.c',
            'CHFJPY.c', 'CADJPY.c', 'XAUUSD.c'
        ]
        
        # Risk Management Settings
        self.account_balance = 10000.0  # Default account balance
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_total_exposure = 0.06  # 6% total portfolio
        self.max_daily_loss = 0.05      # 5% daily loss limit
        
        # Position Sizing Settings
        self.default_lot_size = 0.01
        self.max_lot_size = 1.0
        self.min_lot_size = 0.01
        
        # Entry/Exit Settings
        self.entry_confirmation_required = True
        self.partial_exit_enabled = True
        self.trailing_stop_enabled = True
        
        # Data storage
        self.live_data = {}
        self.account_info = {}
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.is_running = False
        self.mt5_connected = False
        self.last_update = datetime.now()
        
        self.setup_logging()
        self.setup_routes()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_dashboard.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 and get account info"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get real account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            # Update account settings with real data
            self.account_balance = account_info.balance
            self.account_info = {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'leverage': account_info.leverage
            }
            
            # Test symbols
            available_symbols = []
            for symbol in self.forex_pairs:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
            
            self.forex_pairs = available_symbols
            self.mt5_connected = True
            
            self.logger.info(f"MT5 Connected - Account: {account_info.login}")
            self.logger.info(f"Balance: ${account_info.balance:,.2f}")
            self.logger.info(f"Available Pairs: {len(self.forex_pairs)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str, risk_percent: float = None) -> Dict:
        """Calculate optimal position size based on risk management"""
        try:
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # Risk amount in account currency
            risk_amount = self.account_balance * risk_percent
            
            # Pip value calculation
            pip_size = 0.0001 if 'JPY' not in symbol else 0.01
            if 'XAU' in symbol:
                pip_size = 0.01
            
            # Points difference between entry and stop loss
            points_risk = abs(entry_price - stop_loss)
            
            # Position size calculation
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'lot_size': self.default_lot_size, 'risk_amount': risk_amount, 'error': 'Symbol info not available'}
            
            # Calculate lot size
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            if tick_size > 0 and points_risk > 0:
                # Money per pip
                money_per_pip = (tick_value / tick_size) * pip_size
                
                # Lot size based on risk
                lot_size = risk_amount / (points_risk / pip_size * money_per_pip)
                
                # Apply limits
                lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
                lot_size = round(lot_size, 2)
            else:
                lot_size = self.default_lot_size
            
            # Calculate actual risk with calculated lot size
            actual_risk = (points_risk / pip_size * (tick_value / tick_size) * pip_size * lot_size)
            risk_percentage = (actual_risk / self.account_balance) * 100
            
            return {
                'lot_size': lot_size,
                'risk_amount': round(actual_risk, 2),
                'risk_percentage': round(risk_percentage, 2),
                'pip_value': round(money_per_pip * lot_size, 2),
                'points_risk': round(points_risk, 5)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {'lot_size': self.default_lot_size, 'risk_amount': 0, 'error': str(e)}
    
    def calculate_reward_risk_ratio(self, entry_price: float, stop_loss: float, take_profits: List[float]) -> Dict:
        """Calculate reward to risk ratios"""
        try:
            risk = abs(entry_price - stop_loss)
            
            ratios = {}
            for i, tp in enumerate(take_profits, 1):
                reward = abs(tp - entry_price)
                ratio = reward / risk if risk > 0 else 0
                ratios[f'tp_{i}_ratio'] = round(ratio, 2)
                ratios[f'tp_{i}_reward'] = round(reward, 5)
            
            ratios['risk'] = round(risk, 5)
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating R/R ratio: {str(e)}")
            return {}
    
    def analyze_entry_exit_points(self, indicators: Dict, current_price: float, symbol: str) -> Dict:
        """Analyze optimal entry and exit points"""
        try:
            # Your existing indicator calculation logic here
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0)
            atr = indicators.get('atr', current_price * 0.005)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            
            # Entry point analysis
            entry_quality = 'POOR'
            entry_score = 0
            entry_reasons = []
            
            # Signal direction from your existing logic
            signal_direction = 'NONE'
            if current_price > ema_9 > ema_21 > ema_50 and 35 <= rsi <= 65 and trend_strength >= 0.67:
                signal_direction = 'BUY'
                entry_reasons.append("Strong uptrend confirmed")
                entry_score += 3
            elif current_price < ema_9 < ema_21 < ema_50 and 35 <= rsi <= 65 and trend_strength >= 0.67:
                signal_direction = 'SELL'
                entry_reasons.append("Strong downtrend confirmed")
                entry_score += 3
            
            # Entry quality assessment
            if 45 <= rsi <= 55:
                entry_reasons.append("RSI in optimal range")
                entry_score += 2
            
            if trend_strength >= 0.8:
                entry_reasons.append("Very strong trend")
                entry_score += 2
            elif trend_strength >= 0.7:
                entry_reasons.append("Strong trend")
                entry_score += 1
            
            # Determine entry quality
            if entry_score >= 6:
                entry_quality = 'EXCELLENT'
            elif entry_score >= 4:
                entry_quality = 'GOOD'
            elif entry_score >= 2:
                entry_quality = 'FAIR'
            
            # Calculate stop loss and take profits using your existing logic
            if signal_direction == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                take_profit_1 = current_price + (atr * 2.5)
                take_profit_2 = current_price + (atr * 4.0)
                take_profit_3 = current_price + (atr * 6.0)
            elif signal_direction == 'SELL':
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Exit strategy
            exit_strategy = {
                'tp1_percentage': 50,  # Close 50% at TP1
                'tp2_percentage': 30,  # Close 30% at TP2
                'tp3_percentage': 20,  # Close 20% at TP3
                'trailing_stop': True,
                'break_even_level': take_profit_1 if signal_direction != 'NONE' else current_price
            }
            
            # Position sizing
            position_info = self.calculate_position_size(current_price, stop_loss, symbol)
            
            # R/R analysis
            rr_analysis = self.calculate_reward_risk_ratio(
                current_price, stop_loss, [take_profit_1, take_profit_2, take_profit_3]
            )
            
            return {
                'signal_direction': signal_direction,
                'entry_quality': entry_quality,
                'entry_score': entry_score,
                'entry_reasons': entry_reasons,
                'optimal_entry': round(current_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'take_profit_3': round(take_profit_3, 5),
                'position_sizing': position_info,
                'risk_reward': rr_analysis,
                'exit_strategy': exit_strategy,
                'max_risk_amount': round(self.account_balance * self.max_risk_per_trade, 2),
                'recommended_lot_size': position_info.get('lot_size', self.default_lot_size)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry/exit points: {str(e)}")
            return {}
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate indicators (using your existing logic)"""
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
            
            # Trend strength
            current_price = close.iloc[-1]
            trend_conditions = [
                current_price > ema_9,
                ema_9 > ema_21,
                ema_21 > ema_50
            ]
            trend_strength = sum(trend_conditions) / len(trend_conditions)
            
            # Volume analysis
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            volume_avg = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
            
            return {
                'rsi': rsi,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_50': ema_50
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get enhanced symbol data with risk management"""
        try:
            if not self.mt5_connected:
                return None
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            current_price = tick.bid
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            
            # Analyze entry/exit points
            entry_exit_analysis = self.analyze_entry_exit_points(indicators, current_price, symbol)
            
            # Price change calculation
            previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100
            
            # Spread calculation
            spread = tick.ask - tick.bid
            spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 5),
                'price_change': round(price_change, 5),
                'change_percent': round(change_percent, 3),
                'bid': tick.bid,
                'ask': tick.ask,
                'spread_pips': round(spread_pips, 1),
                
                # Technical Analysis
                'rsi': round(indicators.get('rsi', 50), 1),
                'atr_percent': round(indicators.get('atr_percent', 0), 3),
                'trend_strength': round(indicators.get('trend_strength', 0), 3),
                'volume_ratio': round(indicators.get('volume_ratio', 1), 2),
                
                # Enhanced Entry/Exit Analysis
                **entry_exit_analysis,
                
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def update_all_data(self):
        """Update all symbols data"""
        try:
            if not self.mt5_connected:
                self.logger.warning("MT5 not connected")
                return
            
            updated_count = 0
            for symbol in self.forex_pairs:
                data = self.get_symbol_data(symbol)
                if data:
                    self.live_data[symbol] = data
                    updated_count += 1
                time.sleep(0.1)
            
            self.last_update = datetime.now()
            self.logger.info(f"Updated {updated_count}/{len(self.forex_pairs)} pairs")
            
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return send_from_directory('.', 'enhanced_risk_dashboard.html')
        
        @self.app.route('/api/market-data')
        def get_market_data():
            """Enhanced API with risk management data"""
            try:
                formatted_data = {}
                
                for symbol, data in self.live_data.items():
                    formatted_data[symbol] = data
                
                return jsonify({
                    'success': True,
                    'data': formatted_data,
                    'account': self.account_info,
                    'risk_settings': {
                        'max_risk_per_trade': self.max_risk_per_trade * 100,
                        'max_total_exposure': self.max_total_exposure * 100,
                        'max_daily_loss': self.max_daily_loss * 100,
                        'account_balance': self.account_balance
                    },
                    'timestamp': datetime.now().isoformat(),
                    'mt5_connected': self.mt5_connected
                })
                
            except Exception as e:
                self.logger.error(f"Error in market-data API: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {},
                    'mt5_connected': False
                })
        
        @self.app.route('/api/risk-analysis/<symbol>')
        def get_risk_analysis(symbol):
            """Get detailed risk analysis for specific symbol"""
            if symbol in self.live_data:
                return jsonify(self.live_data[symbol])
            return jsonify({'error': 'Symbol not found'}), 404
    
    def start_data_updates(self):
        """Start automatic data updates"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(15)
                except Exception as e:
                    self.logger.error(f"Update error: {str(e)}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.logger.info("Auto-update system started")
    
    def run(self, host='127.0.0.1', port=5000):
        """Run the enhanced dashboard"""
        try:
            print("Enhanced Risk Management Dashboard Starting...")
            print("=" * 50)
            
            if not self.connect_mt5():
                print("ERROR Failed to connect to MT5")
                return
            
            self.is_running = True
            self.start_data_updates()
            self.update_all_data()
            
            print(f"SUCCESS Dashboard Started!")
            print(f"FEATURES Risk Management + Entry/Exit Analysis")
            print(f"DASHBOARD http://{host}:{port}")
            print(f"API http://{host}:{port}/api/market-data")
            print("STOP Press Ctrl+C to stop")
            print("=" * 50)
            
            self.app.run(host=host, port=port, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            print("\nSTOPPING Enhanced Dashboard...")
            self.is_running = False
            mt5.shutdown()
            print("STOPPED successfully")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.is_running = False
            mt5.shutdown()

def main():
    """Main execution"""
    print("Enhanced Risk Management Dashboard")
    print("==================================")
    print("Features: Position Sizing, Entry/Exit Analysis, Risk Controls")
    print()
    
    dashboard = EnhancedRiskDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()