"""
🔗 FIXED DASHBOARD SYSTEM
========================
Dashboard ที่เชื่อมต่อ MT5 ได้แน่นอน
ใช้ HTTP polling แทน WebSocket
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FixedDashboardSystem:
    """Fixed dashboard system with guaranteed MT5 connection"""
    
    def __init__(self):
        """Initialize fixed dashboard system"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Forex pairs (from your diagnostic results)
        self.forex_pairs = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'GBPJPY.c',
            'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'AUDCHF.c', 'AUDJPY.c', 'NZDJPY.c',
            'XAUUSD.c'  # Gold
        ]
        
        # Data storage
        self.market_data = {}
        self.last_update = datetime.now()
        self.is_running = False
        self.mt5_connected = False
        
        # Trading system parameters
        self.min_confluence_score = 7.5
        self.setup_logging()
        self.setup_routes()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 (same as working diagnostic)"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info to verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            self.logger.info(f"✅ Connected to MT5 - Account: {account_info.login}")
            self.logger.info(f"✅ Server: {account_info.server}")
            self.logger.info(f"✅ Balance: ${account_info.balance}")
            
            # Test and prepare symbols
            available_symbols = []
            for symbol in self.forex_pairs:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
                    self.logger.info(f"✅ Symbol ready: {symbol}")
                else:
                    self.logger.warning(f"⚠️ Symbol not found: {symbol}")
            
            self.forex_pairs = available_symbols
            self.mt5_connected = True
            self.logger.info(f"✅ Ready with {len(self.forex_pairs)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate indicators (simplified but working)"""
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
            
            return {
                'rsi': rsi,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'trend_strength': trend_strength,
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_50': ema_50
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def analyze_signal(self, indicators: Dict, current_price: float) -> Dict:
        """Analyze trading signal (simplified V2.0)"""
        try:
            signal_direction = 'NONE'
            strength_score = 0.0
            confidence = 'LOW'
            
            if not indicators:
                return {'direction': signal_direction, 'strength': strength_score, 'confidence': confidence}
            
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            
            # Signal logic
            if (current_price > ema_9 > ema_21 > ema_50 and 
                trend_strength >= 0.67 and 
                35 <= rsi <= 65):
                signal_direction = 'BUY'
                strength_score = 6.0 + (trend_strength * 2) + ((60 - abs(rsi - 50)) / 10)
                
            elif (current_price < ema_9 < ema_21 < ema_50 and 
                  trend_strength >= 0.67 and 
                  35 <= rsi <= 65):
                signal_direction = 'SELL'
                strength_score = 6.0 + (trend_strength * 2) + ((60 - abs(rsi - 50)) / 10)
            
            # Confidence levels
            if strength_score >= 8.5:
                confidence = 'VERY_HIGH'
            elif strength_score >= 7.5:
                confidence = 'HIGH'
            elif strength_score >= 6.5:
                confidence = 'MEDIUM'
            elif strength_score >= 5.0:
                confidence = 'LOW'
            else:
                confidence = 'FILTERED'
            
            # Apply minimum threshold
            if strength_score < self.min_confluence_score:
                signal_direction = 'NONE'
                confidence = 'FILTERED'
            
            return {
                'direction': signal_direction,
                'strength': round(strength_score, 2),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal: {str(e)}")
            return {'direction': 'NONE', 'strength': 0.0, 'confidence': 'ERROR'}
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time data from MT5"""
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
            
            # Analyze signal
            signal_analysis = self.analyze_signal(indicators, current_price)
            
            # Price change
            previous_price = df['close'].iloc[-2]
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100
            
            # Trading levels
            atr = indicators.get('atr', current_price * 0.005)
            
            if signal_analysis['direction'] == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                take_profit_1 = current_price + (atr * 2.5)
                take_profit_2 = current_price + (atr * 4.0)
            elif signal_analysis['direction'] == 'SELL':
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = current_price
            
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
                'signal_direction': signal_analysis['direction'],
                'confidence': signal_analysis['confidence'],
                'strength_score': signal_analysis['strength'],
                'rsi': round(indicators.get('rsi', 50), 1),
                'atr_percent': round(indicators.get('atr_percent', 0), 3),
                'trend_strength': round(indicators.get('trend_strength', 0), 3),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
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
                    self.market_data[symbol] = data
                    updated_count += 1
            
            self.last_update = datetime.now()
            self.logger.info(f"✅ Updated {updated_count}/{len(self.forex_pairs)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
    
    def calculate_market_stats(self) -> Dict:
        """Calculate market statistics"""
        try:
            active_signals = 0
            high_confidence = 0
            
            for data in self.market_data.values():
                if data['signal_direction'] != 'NONE':
                    active_signals += 1
                if data['confidence'] in ['HIGH', 'VERY_HIGH']:
                    high_confidence += 1
            
            return {
                'active_signals': active_signals,
                'high_confidence': high_confidence,
                'total_pairs': len(self.market_data),
                'mt5_connected': self.mt5_connected,
                'last_update': self.last_update.strftime('%H:%M:%S'),
                'sentiment': 'BULLISH' if active_signals > 5 else 'BEARISH' if active_signals < 2 else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating stats: {str(e)}")
            return {}
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint for all market data"""
            return jsonify({
                'pairs': self.market_data,
                'stats': self.calculate_market_stats(),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/status')
        def get_status():
            """API endpoint for connection status"""
            return jsonify({
                'mt5_connected': self.mt5_connected,
                'symbols_count': len(self.forex_pairs),
                'last_update': self.last_update.isoformat()
            })
    
    def get_dashboard_html(self) -> str:
        """Dashboard HTML with auto-refresh"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>🔗 Fixed Forex Dashboard</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="15">
    <style>
        body { 
            font-family: 'Segoe UI', Arial; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; margin: 0; padding: 1rem;
        }
        .header { 
            text-align: center; padding: 2rem; 
            background: rgba(255,255,255,0.1); 
            border-radius: 15px; margin-bottom: 2rem;
            backdrop-filter: blur(10px);
        }
        .status { 
            display: inline-block; padding: 0.5rem 1rem; 
            border-radius: 20px; font-weight: bold; margin: 0.5rem;
        }
        .connected { background: #00ff88; color: #000; }
        .disconnected { background: #ff4757; color: #fff; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 1.5rem; 
        }
        .card { 
            background: rgba(255,255,255,0.1); 
            border-radius: 15px; padding: 1.5rem; 
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .pair-name { 
            font-size: 1.3rem; font-weight: bold; 
            color: #00ff88; margin-bottom: 0.5rem;
        }
        .price { 
            font-size: 1.8rem; font-weight: bold; 
            margin: 1rem 0; text-align: center;
        }
        .change { font-size: 1rem; text-align: center; margin-bottom: 1rem; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .signal { 
            text-align: center; padding: 0.8rem; 
            border-radius: 10px; margin: 1rem 0; 
            font-weight: bold; text-transform: uppercase;
        }
        .signal-buy { background: #00ff88; color: #000; }
        .signal-sell { background: #ff4757; color: #fff; }
        .signal-none { background: rgba(255,255,255,0.2); color: #ccc; }
        .metrics { 
            display: grid; grid-template-columns: 1fr 1fr; 
            gap: 0.5rem; margin-top: 1rem;
        }
        .metric { 
            text-align: center; padding: 0.5rem; 
            background: rgba(255,255,255,0.1); border-radius: 8px;
        }
        .metric-value { color: #00ff88; font-weight: bold; }
        .levels { 
            margin-top: 1rem; font-size: 0.9rem;
            background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;
        }
        .level-row { 
            display: flex; justify-content: space-between; 
            padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .loading { text-align: center; padding: 2rem; font-size: 1.2rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔗 Fixed Forex Dashboard</h1>
        <p>Real-time MT5 Connection • Auto-refresh every 15 seconds</p>
        <div id="statusArea">
            <div class="status disconnected">🔴 Loading...</div>
        </div>
    </div>
    
    <div id="dataArea" class="loading">
        📡 Loading market data from MT5...
    </div>

    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                updateStatus(data.stats);
                updatePairs(data.pairs);
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('statusArea').innerHTML = 
                    '<div class="status disconnected">🔴 Connection Error</div>';
            }
        }
        
        function updateStatus(stats) {
            const statusHtml = stats.mt5_connected ? 
                `<div class="status connected">🟢 MT5 Connected</div>
                 <div class="status connected">📊 ${stats.total_pairs} Pairs</div>
                 <div class="status connected">⚡ ${stats.active_signals} Signals</div>
                 <div class="status connected">🕒 ${stats.last_update}</div>` :
                '<div class="status disconnected">🔴 MT5 Disconnected</div>';
            
            document.getElementById('statusArea').innerHTML = statusHtml;
        }
        
        function updatePairs(pairs) {
            const dataArea = document.getElementById('dataArea');
            
            if (Object.keys(pairs).length === 0) {
                dataArea.innerHTML = '<div class="loading">📡 No data available</div>';
                return;
            }
            
            let html = '<div class="grid">';
            
            Object.entries(pairs).forEach(([symbol, data]) => {
                const displaySymbol = symbol.replace('.c', '');
                const changeClass = data.price_change >= 0 ? 'positive' : 'negative';
                const changeSymbol = data.price_change >= 0 ? '+' : '';
                const precision = symbol.includes('JPY') ? 3 : 5;
                
                html += `
                    <div class="card">
                        <div class="pair-name">${displaySymbol}</div>
                        <div class="price">${data.current_price.toFixed(precision)}</div>
                        <div class="change ${changeClass}">
                            ${changeSymbol}${Math.abs(data.price_change).toFixed(precision)} 
                            (${changeSymbol}${data.change_percent.toFixed(2)}%)
                        </div>
                        
                        <div class="signal signal-${data.signal_direction.toLowerCase()}">
                            ${data.signal_direction}
                        </div>
                        
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-value">${data.rsi.toFixed(1)}</div>
                                <div>RSI</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.spread_pips.toFixed(1)}</div>
                                <div>Spread</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.strength_score.toFixed(1)}</div>
                                <div>Strength</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.confidence}</div>
                                <div>Confidence</div>
                            </div>
                        </div>
                        
                        ${data.signal_direction !== 'NONE' ? `
                        <div class="levels">
                            <div class="level-row">
                                <span>Stop Loss:</span>
                                <span>${data.stop_loss.toFixed(precision)}</span>
                            </div>
                            <div class="level-row">
                                <span>Take Profit 1:</span>
                                <span>${data.take_profit_1.toFixed(precision)}</span>
                            </div>
                            <div class="level-row">
                                <span>Take Profit 2:</span>
                                <span>${data.take_profit_2.toFixed(precision)}</span>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            
            html += '</div>';
            dataArea.innerHTML = html;
        }
        
        // Load data immediately and then every 15 seconds
        loadData();
        setInterval(loadData, 15000);
    </script>
</body>
</html>
        '''
    
    def start_data_updates(self):
        """Start background data updates"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    self.logger.error(f"Update error: {str(e)}")
                    time.sleep(5)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self.logger.info("✅ Data update thread started")
    
    def run(self):
        """Run the fixed dashboard"""
        try:
            # Connect to MT5
            if not self.connect_mt5():
                print("❌ Failed to connect to MT5")
                return
            
            self.is_running = True
            
            # Start data updates
            self.start_data_updates()
            
            # Initial data load
            self.update_all_data()
            
            print("🚀 Fixed Dashboard Starting...")
            print("📊 MT5 Connected Successfully")
            print("🌐 Open: http://127.0.0.1:5000")
            print("⏹️ Press Ctrl+C to stop")
            
            # Run Flask app
            self.app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            print("\n⏹️ Shutting down...")
            self.is_running = False
            mt5.shutdown()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.is_running = False
            mt5.shutdown()

def main():
    """Main execution"""
    print("🔗 Starting Fixed Dashboard System...")
    
    # Initialize and run
    dashboard = FixedDashboardSystem()
    dashboard.run()

if __name__ == "__main__":
    main()