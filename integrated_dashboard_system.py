"""
🔗 INTEGRATED DASHBOARD SYSTEM
==============================
เชื่อม MT5 กับ HTML Dashboard แบบ Real-time
ใช้ Flask WebSocket สำหรับ Live Data Streaming
"""

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
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

class IntegratedDashboardSystem:
    """Integrated system connecting MT5 to HTML Dashboard"""
    
    def __init__(self):
        """Initialize integrated dashboard system"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'forex_dashboard_secret'
        
        # Configure SocketIO with specific settings to avoid network issues
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        # Forex pairs configuration (with .c suffix)
        self.forex_pairs = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'GBPJPY.c',
            'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'AUDCHF.c', 'AUDJPY.c', 'NZDJPY.c'
        ]
        
        # Data storage
        self.market_data = {}
        self.last_update = datetime.now()
        self.is_running = False
        self.connected_clients = 0
        
        # Trading system parameters (from Optimized V2.0)
        self.min_confluence_score = 7.5
        self.rsi_period = 14
        self.atr_period = 14
        self.ema_periods = [9, 21, 50, 200]
        
        self.setup_logging()
        self.setup_routes()
        self.setup_socketio_events()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('integrated_dashboard.log'),
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
            
            # Check if symbols are available
            available_symbols = []
            for symbol in self.forex_pairs:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
                else:
                    self.logger.warning(f"Symbol {symbol} not available")
            
            self.forex_pairs = available_symbols
            self.logger.info(f"Connected to MT5 with {len(self.forex_pairs)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators (from Optimized System V2.0)"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            
            # EMA calculations
            emas = {}
            for period in self.ema_periods:
                emas[f'EMA_{period}'] = close.ewm(span=period).mean().iloc[-1]
            
            # RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
            
            # MACD calculation
            ema_fast = close.ewm(span=12).mean()
            ema_slow = close.ewm(span=26).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Trend strength calculation
            current_price = close.iloc[-1]
            trend_conditions = [
                current_price > emas['EMA_9'],
                emas['EMA_9'] > emas['EMA_21'],
                emas['EMA_21'] > emas['EMA_50'],
                emas['EMA_50'] > emas['EMA_200']
            ]
            trend_strength = sum(trend_conditions) / len(trend_conditions)
            
            # Volume analysis
            volume_avg = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
            
            return {
                'rsi': rsi,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_histogram': macd_histogram.iloc[-1],
                **emas
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def analyze_trading_signal(self, indicators: Dict, current_price: float, symbol: str) -> Dict:
        """Analyze trading signal (Enhanced V2.0 Logic)"""
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
            atr_percent = indicators.get('atr_percent', 0)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # Enhanced Filter System (V2.0)
            
            # Filter 1: RSI Range (Tighter)
            if 35 <= rsi <= 65:
                filters_passed.append("RSI in normal range")
                strength_score += 2.0
            else:
                filters_failed.append(f"RSI extreme: {rsi:.1f}")
            
            # Filter 2: Trend Strength
            if trend_strength >= 0.75:
                filters_passed.append("Strong trend detected")
                strength_score += 3.0
            else:
                filters_failed.append(f"Weak trend: {trend_strength:.2f}")
            
            # Filter 3: Volume Confirmation
            if volume_ratio >= 1.3:
                filters_passed.append("Volume confirmation")
                strength_score += 1.5
            else:
                filters_failed.append(f"Low volume: {volume_ratio:.2f}")
            
            # Filter 4: Volatility Check
            if 0.05 <= atr_percent <= 2.0:  # Normal volatility for forex
                filters_passed.append("Normal volatility")
                strength_score += 1.0
            else:
                filters_failed.append(f"Unusual volatility: {atr_percent:.2f}%")
            
            # Filter 5: MACD Confirmation
            if macd > macd_signal:
                filters_passed.append("MACD bullish")
                strength_score += 1.5
            elif macd < macd_signal:
                filters_passed.append("MACD bearish")
                strength_score += 1.5
            
            # Signal Generation Logic
            ema_9 = indicators.get('EMA_9', current_price)
            ema_21 = indicators.get('EMA_21', current_price)
            ema_50 = indicators.get('EMA_50', current_price)
            
            # Enhanced signal logic
            if (current_price > ema_9 > ema_21 > ema_50 and 
                trend_strength >= 0.75 and 
                macd > macd_signal and
                35 <= rsi <= 65):
                signal_direction = 'BUY'
                strength_score += 2.0
                
            elif (current_price < ema_9 < ema_21 < ema_50 and 
                  trend_strength >= 0.75 and 
                  macd < macd_signal and
                  35 <= rsi <= 65):
                signal_direction = 'SELL'
                strength_score += 2.0
            
            # Confidence Levels (V2.0 Enhanced)
            if strength_score >= 9.0 and len(filters_failed) == 0:
                confidence = 'VERY_HIGH'
            elif strength_score >= 8.0 and len(filters_failed) <= 1:
                confidence = 'HIGH'
            elif strength_score >= 6.5 and len(filters_failed) <= 2:
                confidence = 'MEDIUM'
            elif strength_score >= 5.0:
                confidence = 'LOW'
            else:
                confidence = 'FILTERED'
            
            # Apply minimum threshold (V2.0)
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
        """Calculate trading levels (V2.0 Enhanced R/R)"""
        try:
            if signal_direction == 'NONE':
                return {
                    'stop_loss': 0,
                    'take_profit_1': 0,
                    'take_profit_2': 0,
                    'take_profit_3': 0
                }
            
            # V2.0 Enhanced ratios
            if signal_direction == 'BUY':
                stop_loss = current_price - (atr * 1.5)  # Tighter stops
                take_profit_1 = current_price + (atr * 2.5)  # 1:1.67 R/R
                take_profit_2 = current_price + (atr * 4.0)  # 1:2.67 R/R
                take_profit_3 = current_price + (atr * 6.0)  # 1:4.0 R/R
            else:  # SELL
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            
            return {
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'take_profit_3': round(take_profit_3, 5)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trading levels: {str(e)}")
            return {'stop_loss': 0, 'take_profit_1': 0, 'take_profit_2': 0, 'take_profit_3': 0}
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time data for a symbol from MT5"""
        try:
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            current_price = tick.bid
            
            # Get historical data for indicators
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 300)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            
            # Analyze signal
            signal_analysis = self.analyze_trading_signal(indicators, current_price, symbol)
            
            # Calculate trading levels
            atr = indicators.get('atr', current_price * 0.005)
            levels = self.calculate_trading_levels(current_price, signal_analysis['direction'], atr)
            
            # Calculate price change
            if len(df) >= 2:
                previous_price = df['close'].iloc[-2]
                price_change = current_price - previous_price
                change_percent = (price_change / previous_price) * 100
            else:
                price_change = 0
                change_percent = 0
            
            # Calculate spread
            spread = tick.ask - tick.bid
            spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 5),
                'price_change': round(price_change, 5),
                'change_percent': round(change_percent, 3),
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': round(spread, 5),
                'spread_pips': round(spread_pips, 1),
                'signal_direction': signal_analysis['direction'],
                'confidence': signal_analysis['confidence'],
                'strength_score': signal_analysis['strength'],
                'rsi': round(indicators.get('rsi', 50), 1),
                'atr': round(indicators.get('atr', 0), 6),
                'atr_percent': round(indicators.get('atr_percent', 0), 3),
                'trend_strength': round(indicators.get('trend_strength', 0), 3),
                'volume_ratio': round(indicators.get('volume_ratio', 1), 2),
                'macd': round(indicators.get('macd', 0), 5),
                'macd_signal': round(indicators.get('macd_signal', 0), 5),
                'stop_loss': levels['stop_loss'],
                'take_profit_1': levels['take_profit_1'],
                'take_profit_2': levels['take_profit_2'],
                'take_profit_3': levels['take_profit_3'],
                'filters_passed': signal_analysis.get('filters_passed', []),
                'filters_failed': signal_analysis.get('filters_failed', []),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def update_all_symbols(self):
        """Update data for all symbols"""
        try:
            updated_data = {}
            
            for symbol in self.forex_pairs:
                data = self.get_symbol_data(symbol)
                if data:
                    updated_data[symbol] = data
                    self.market_data[symbol] = data
            
            # Calculate market statistics
            market_stats = self.calculate_market_stats()
            
            # Emit to all connected clients
            if self.connected_clients > 0:
                self.socketio.emit('market_update', {
                    'pairs': updated_data,
                    'market_stats': market_stats,
                    'timestamp': datetime.now().isoformat()
                })
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating symbols: {str(e)}")
    
    def calculate_market_stats(self) -> Dict:
        """Calculate overall market statistics"""
        try:
            active_signals = 0
            high_confidence = 0
            trending_pairs = 0
            
            # Currency strength calculation
            currency_strength = {'USD': 0, 'EUR': 0, 'GBP': 0, 'JPY': 0, 'AUD': 0, 'NZD': 0, 'CHF': 0}
            
            for symbol, data in self.market_data.items():
                if data['signal_direction'] != 'NONE':
                    active_signals += 1
                    
                if data['confidence'] in ['HIGH', 'VERY_HIGH']:
                    high_confidence += 1
                    
                if abs(data['change_percent']) > 0.3:
                    trending_pairs += 1
                
                # Calculate currency strength
                base = symbol[:3]  # First 3 characters (EURUSD.c -> EUR)
                quote = symbol[3:6]  # Characters 3-6 (EURUSD.c -> USD)
                
                if base in currency_strength and quote in currency_strength:
                    if data['change_percent'] > 0:
                        currency_strength[base] += abs(data['change_percent'])
                        currency_strength[quote] -= abs(data['change_percent'])
                    else:
                        currency_strength[base] -= abs(data['change_percent'])
                        currency_strength[quote] += abs(data['change_percent'])
            
            # Find strongest and weakest currencies
            sorted_currencies = sorted(currency_strength.items(), key=lambda x: x[1], reverse=True)
            strongest = sorted_currencies[0][0] if sorted_currencies else 'USD'
            weakest = sorted_currencies[-1][0] if sorted_currencies else 'JPY'
            
            return {
                'active_signals': active_signals,
                'high_confidence': high_confidence,
                'trending_pairs': trending_pairs,
                'total_pairs': len(self.market_data),
                'strongest_currency': strongest,
                'weakest_currency': weakest,
                'sentiment': 'BULLISH' if active_signals > 6 else 'BEARISH' if active_signals < 3 else 'NEUTRAL',
                'volatility': 'HIGH' if trending_pairs > 5 else 'LOW' if trending_pairs < 2 else 'MEDIUM',
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market stats: {str(e)}")
            return {}
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/market_data')
        def get_market_data():
            """API endpoint for market data"""
            return jsonify({
                'pairs': self.market_data,
                'market_stats': self.calculate_market_stats(),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/symbol/<symbol>')
        def get_symbol_data_api(symbol):
            """API endpoint for specific symbol"""
            if symbol in self.market_data:
                return jsonify(self.market_data[symbol])
            else:
                return jsonify({'error': 'Symbol not found'}), 404
    
    def setup_socketio_events(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.connected_clients += 1
            self.logger.info(f"Client connected. Total clients: {self.connected_clients}")
            
            # Send initial data
            emit('market_update', {
                'pairs': self.market_data,
                'market_stats': self.calculate_market_stats(),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients -= 1
            self.logger.info(f"Client disconnected. Total clients: {self.connected_clients}")
    
    def get_dashboard_html(self) -> str:
        """Get the dashboard HTML template with real-time updates"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔗 Live Forex Dashboard - MT5 Connected</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        /* Include all the CSS from the previous dashboard here */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #2d1b4e 100%);
            color: #ffffff; min-height: 100vh;
        }
        .dashboard-header {
            background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
            border-bottom: 3px solid #00ff88; padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.3);
        }
        .live-status { 
            background: #00ff88; color: #000; padding: 0.5rem 1rem; 
            border-radius: 20px; font-weight: bold; animation: pulse 2s infinite;
        }
        .currency-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem; padding: 2rem; max-width: 1600px; margin: 0 auto;
        }
        .currency-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.08) 100%);
            backdrop-filter: blur(15px); border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px; padding: 1.5rem; transition: all 0.4s ease;
        }
        .signal-buy { background: linear-gradient(135deg, #00ff88, #00d46a); color: #000; }
        .signal-sell { background: linear-gradient(135deg, #ff4757, #ff3838); color: #fff; }
        .signal-none { background: rgba(255, 255, 255, 0.1); color: #ccc; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>🔗 Live Forex Dashboard</h1>
            <div class="live-status" id="connectionStatus">🔴 CONNECTING...</div>
        </div>
    </div>
    
    <div class="currency-grid" id="currencyGrid">
        <div style="grid-column: 1 / -1; text-align: center; padding: 2rem;">
            <h2>📡 Connecting to MT5...</h2>
            <p>Please wait while we establish connection with MetaTrader 5</p>
        </div>
    </div>

    <script>
        const socket = io();
        let isConnected = false;
        
        socket.on('connect', function() {
            isConnected = true;
            document.getElementById('connectionStatus').innerHTML = '🟢 LIVE - MT5 CONNECTED';
            document.getElementById('connectionStatus').style.background = '#00ff88';
            console.log('Connected to server');
        });
        
        socket.on('disconnect', function() {
            isConnected = false;
            document.getElementById('connectionStatus').innerHTML = '🔴 DISCONNECTED';
            document.getElementById('connectionStatus').style.background = '#ff4757';
            console.log('Disconnected from server');
        });
        
        socket.on('market_update', function(data) {
            updateDashboard(data);
        });
        
        function updateDashboard(data) {
            const grid = document.getElementById('currencyGrid');
            grid.innerHTML = '';
            
            // Add market overview
            const overview = createMarketOverview(data.market_stats);
            grid.appendChild(overview);
            
            // Add currency cards
            Object.entries(data.pairs).forEach(([symbol, pairData]) => {
                const card = createCurrencyCard(symbol, pairData);
                grid.appendChild(card);
            });
        }
        
        function createMarketOverview(stats) {
            const div = document.createElement('div');
            div.style.cssText = `
                grid-column: 1 / -1; background: rgba(0, 123, 255, 0.15);
                border: 2px solid rgba(255, 255, 255, 0.2); border-radius: 15px;
                padding: 2rem; margin-bottom: 1rem; text-align: center;
            `;
            div.innerHTML = `
                <h2>📊 Live Market Overview</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #00ff88;">${stats.active_signals || 0}</div>
                        <div>Active Signals</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #00ff88;">${stats.high_confidence || 0}</div>
                        <div>High Confidence</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #00ff88;">${stats.strongest_currency || 'USD'}</div>
                        <div>Strongest Currency</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #00ff88;">${stats.sentiment || 'NEUTRAL'}</div>
                        <div>Market Sentiment</div>
                    </div>
                </div>
            `;
            return div;
        }
        
        function createCurrencyCard(symbol, data) {
            const div = document.createElement('div');
            div.className = 'currency-card';
            
            // Remove .c suffix for display
            const displaySymbol = symbol.replace('.c', '');
            
            const changeClass = data.price_change >= 0 ? 'positive' : 'negative';
            const changeSymbol = data.price_change >= 0 ? '+' : '';
            const precision = symbol.includes('JPY') ? 3 : 5;
            
            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <div>
                        <h3 style="color: #00ff88; font-size: 1.4rem;">${displaySymbol}</h3>
                        <div style="color: #aaa; font-size: 0.9rem;">Live from MT5</div>
                    </div>
                    <div style="font-size: 1.5rem;">${getCountryFlags(displaySymbol)}</div>
                </div>
                
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 1rem 0;">
                    <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">${data.current_price.toFixed(precision)}</div>
                    <div class="${changeClass}" style="font-size: 1.1rem;">
                        ${changeSymbol}${Math.abs(data.price_change).toFixed(precision)} (${changeSymbol}${data.change_percent.toFixed(2)}%)
                    </div>
                </div>
                
                <div style="text-align: center; margin: 1rem 0;">
                    <div class="signal-${data.signal_direction.toLowerCase()}" style="padding: 0.8rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: bold; margin-bottom: 1rem;">
                        ${data.signal_direction}
                    </div>
                    <div style="color: #aaa;">${data.confidence} (${data.strength_score}/10)</div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.08); border-radius: 8px;">
                        <div style="color: #00ff88; font-size: 1.2rem;">${data.rsi.toFixed(1)}</div>
                        <div style="color: #aaa; font-size: 0.8rem;">RSI</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.08); border-radius: 8px;">
                        <div style="color: #00ff88; font-size: 1.2rem;">${data.spread_pips.toFixed(1)}</div>
                        <div style="color: #aaa; font-size: 0.8rem;">Spread (pips)</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.08); border-radius: 8px;">
                        <div style="color: #00ff88; font-size: 1.2rem;">${data.atr_percent.toFixed(2)}%</div>
                        <div style="color: #aaa; font-size: 0.8rem;">ATR</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.08); border-radius: 8px;">
                        <div style="color: #00ff88; font-size: 1.2rem;">${data.trend_strength.toFixed(2)}</div>
                        <div style="color: #aaa; font-size: 0.8rem;">Trend</div>
                    </div>
                </div>
                
                ${data.signal_direction !== 'NONE' ? `
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <h4 style="color: #00ff88; margin-bottom: 0.8rem; text-align: center;">🎯 Trading Levels</h4>
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <span style="color: #aaa;">Stop Loss:</span>
                        <span style="color: #00ff88; font-weight: bold;">${data.stop_loss.toFixed(precision)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <span style="color: #aaa;">TP1:</span>
                        <span style="color: #00ff88; font-weight: bold;">${data.take_profit_1.toFixed(precision)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
                        <span style="color: #aaa;">TP2:</span>
                        <span style="color: #00ff88; font-weight: bold;">${data.take_profit_2.toFixed(precision)}</span>
                    </div>
                </div>
                ` : `
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem; text-align: center; color: #aaa;">
                    ⏳ No Active Signal<br>
                    <small>Waiting for better market conditions</small>
                </div>
                `}
            `;
            
            return div;
        }
        
        function getCountryFlags(symbol) {
            const flags = {
                'EUR': '🇪🇺', 'USD': '🇺🇸', 'GBP': '🇬🇧', 'JPY': '🇯🇵',
                'AUD': '🇦🇺', 'NZD': '🇳🇿', 'CHF': '🇨🇭'
            };
            const base = symbol.substring(0, 3);
            const quote = symbol.substring(3, 6);
            return (flags[base] || '💱') + (flags[quote] || '💱');
        }
    </script>
</body>
</html>
        '''
    
    def start_data_updates(self):
        """Start the data update thread"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_symbols()
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    self.logger.error(f"Error in update loop: {str(e)}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.logger.info("Data update thread started")
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the integrated dashboard system"""
        try:
            # Connect to MT5
            if not self.connect_mt5():
                self.logger.error("Failed to connect to MT5")
                return
            
            self.is_running = True
            
            # Start data updates
            self.start_data_updates()
            
            self.logger.info(f"Starting integrated dashboard on http://{host}:{port}")
            self.logger.info(f"Monitoring {len(self.forex_pairs)} forex pairs")
            
            # Run Flask-SocketIO app with specific configurations
            self.socketio.run(
                self.app, 
                host=host, 
                port=port, 
                debug=debug,
                allow_unsafe_werkzeug=True,
                use_reloader=False,
                log_output=True
            )
            
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.is_running = False
            mt5.shutdown()
        except Exception as e:
            self.logger.error(f"Error running dashboard: {str(e)}")
            self.is_running = False
            mt5.shutdown()

def main():
    """Main execution function"""
    print("🔗 Starting Integrated Dashboard System...")
    print("📊 Connecting Trading System to HTML Dashboard")
    
    try:
        # Initialize system
        dashboard_system = IntegratedDashboardSystem()
        
        # Run the system with localhost IP
        dashboard_system.run(host='127.0.0.1', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Failed to start dashboard: {str(e)}")
        print("💡 Trying alternative simple server...")
        
        # Alternative: Run simple Flask without SocketIO
        run_simple_dashboard()

def run_simple_dashboard():
    """Fallback: Simple dashboard without WebSocket"""
    from flask import Flask, jsonify, render_template_string
    import webbrowser
    import threading
    
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>🔗 Simple Forex Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body { font-family: Arial; background: #1a1a2e; color: white; padding: 2rem; }
                .card { background: #16213e; padding: 1rem; margin: 1rem; border-radius: 10px; border: 1px solid #0f3460; }
                .price { font-size: 2rem; color: #00ff88; font-weight: bold; }
                .positive { color: #00ff88; }
                .negative { color: #ff4757; }
            </style>
        </head>
        <body>
            <h1>🔗 Simple Forex Dashboard</h1>
            <p>📡 Auto-refresh every 10 seconds</p>
            <div id="status">🟢 Running - Connect to MT5 for live data</div>
            
            <div class="card">
                <h3>📊 System Status</h3>
                <p>✅ Flask Server: Running</p>
                <p>⚠️ SocketIO: Disabled (compatibility mode)</p>
                <p>🔗 MT5 Connection: Check console</p>
                <p>💡 Open <strong>http://127.0.0.1:5000</strong> in browser</p>
            </div>
            
            <div class="card">
                <h3>🎯 Next Steps</h3>
                <p>1. ✅ Server is running successfully</p>
                <p>2. 🔗 Connect MT5 manually for live data</p>
                <p>3. 📊 Use console dashboard for real-time monitoring</p>
                <p>4. 🌐 Access via: <strong>http://127.0.0.1:5000</strong></p>
            </div>
            
            <div class="card">
                <h3>🔧 Alternative Solutions</h3>
                <p>• Use Console Dashboard (console_dashboard.py)</p>
                <p>• Run Optimized V2.0 System (optimized_trading_system_v2.py)</p>
                <p>• Check network/firewall settings</p>
            </div>
        </body>
        </html>
        '''
    
    try:
        print("🌐 Starting simple Flask server...")
        print("📡 Open: http://127.0.0.1:5000")
        
        # Open browser automatically
        threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
        
        app.run(host='127.0.0.1', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Simple server also failed: {str(e)}")
        print("💡 Use console dashboard instead:")
        print("   python console_dashboard.py")

if __name__ == "__main__":
    main()