# ================================
# COMPLETE PROFESSIONAL AI TRADING SYSTEM
# ระบบ AI เทรดครบครัน ระดับมืออาชีพ
# ================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
import asyncio
from datetime import datetime, timedelta
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib

# News and Market Data
import requests
import feedparser
from textblob import TextBlob

# Technical Analysis
import numba
from numba import jit

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalDataManager:
    """
    🗄️ Professional Data Management System
    จัดการข้อมูลแบบมืออาชีพ Multi-timeframe + News
    """
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.db_path = "professional_trading.db"
        self.is_streaming = False
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        self.data_cache = {}
        self.news_cache = []
        
        self._setup_database()
        self._connect_mt5()
    
    def _setup_database(self):
        """สร้าง Professional Database Schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Multi-timeframe OHLC data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    spread REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Technical indicators cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    indicator_name TEXT,
                    value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # News and events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time DATETIME,
                    currency TEXT,
                    impact TEXT,
                    event_name TEXT,
                    forecast TEXT,
                    previous TEXT,
                    actual TEXT,
                    sentiment_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # AI predictions and results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_time DATETIME,
                    model_name TEXT,
                    prediction_type TEXT,
                    confidence REAL,
                    predicted_direction INTEGER,
                    predicted_price REAL,
                    actual_price REAL,
                    accuracy INTEGER,
                    features TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading positions and performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE,
                    symbol TEXT,
                    type TEXT,
                    entry_time DATETIME,
                    entry_price REAL,
                    exit_time DATETIME,
                    exit_price REAL,
                    volume REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    profit REAL,
                    pips REAL,
                    ai_confidence REAL,
                    ai_reasoning TEXT,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Professional database initialized")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def _connect_mt5(self):
        """เชื่อมต่อ MT5 Professional"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Auto-detect symbol
            symbols = mt5.symbols_get()
            available_symbols = [s.name for s in symbols] if symbols else []
            
            # Find XAUUSD variant
            possible_names = ["XAUUSD", "XAUUSDm", "XAUUSD.c", "GOLD", "GOLDm"]
            for name in possible_names:
                if name in available_symbols:
                    self.symbol = name
                    break
            
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                mt5.symbol_select(self.symbol, True)
            
            logger.info(f"MT5 connected - Symbol: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_multi_timeframe_data(self, periods=1000):
        """ดึงข้อมูล Multi-timeframe แบบมืออาชีพ"""
        try:
            multi_data = {}
            
            for tf_name, tf_const in self.timeframes.items():
                try:
                    rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, periods)
                    if rates is not None:
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        
                        # Add technical indicators
                        df = self._add_technical_indicators(df, tf_name)
                        
                        multi_data[tf_name] = df
                        logger.info(f"Loaded {len(df)} {tf_name} bars")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {tf_name}: {e}")
                    continue
            
            return multi_data
            
        except Exception as e:
            logger.error(f"Multi-timeframe data error: {e}")
            return {}
    
    @jit(nopython=True)
    def _fast_sma(self, data, period):
        """Numba-optimized SMA"""
        n = len(data)
        sma = np.empty(n)
        sma[:period-1] = np.nan
        for i in range(period-1, n):
            sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def _add_technical_indicators(self, df, timeframe):
        """เพิ่ม Technical Indicators แบบครบครัน"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['tick_volume'].values
            
            # Moving Averages
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_200'] = talib.SMA(close, timeperiod=200)
            
            df['ema_10'] = talib.EMA(close, timeperiod=10)
            df['ema_20'] = talib.EMA(close, timeperiod=20)
            df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Momentum Indicators
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_7'] = talib.RSI(close, timeperiod=7)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
            
            # ADX
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # ATR and Volatility
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['true_range'] = talib.TRANGE(high, low, close)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # CCI
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)
            
            # Price Action
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_bullish'] = (df['close'] > df['open']).astype(int)
            
            # Volume indicators (if available)
            if len(volume) > 0:
                df['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=20)
                df['volume_ratio'] = volume / df['volume_sma']
                
            # Support/Resistance levels
            df['pivot_high'] = (df['high'] == df['high'].rolling(5, center=True).max()).astype(int)
            df['pivot_low'] = (df['low'] == df['low'].rolling(5, center=True).min()).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicators error: {e}")
            return df

class NewsAndSentimentAnalyzer:
    """
    📰 News and Market Sentiment Analysis
    วิเคราะห์ข่าวและความรู้สึกตลาด
    """
    
    def __init__(self):
        self.economic_calendar = []
        self.news_feeds = [
            'https://feeds.feedburner.com/forexlive/feed',
            'https://www.fxstreet.com/feeds/all/rss',
        ]
        self.sentiment_scores = {}
        
    def get_economic_calendar(self):
        """ดึง Economic Calendar (ต้องใช้ API จริง)"""
        try:
            # ตัวอย่าง mock data - ใช้งานจริงต้องเชื่อม API
            events = [
                {
                    'time': datetime.now() + timedelta(hours=2),
                    'currency': 'USD',
                    'impact': 'HIGH',
                    'event': 'Non-Farm Payrolls',
                    'forecast': '180K',
                    'previous': '150K'
                }
            ]
            return events
        except Exception as e:
            logger.error(f"Economic calendar error: {e}")
            return []
    
    def analyze_news_sentiment(self, hours_back=24):
        """วิเคราะห์ sentiment จากข่าว"""
        try:
            sentiment_score = 0.0
            news_count = 0
            
            for feed_url in self.news_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:  # ล่าสุด 10 ข่าว
                        # Analyze sentiment
                        text = entry.title + " " + entry.get('summary', '')
                        blob = TextBlob(text)
                        sentiment_score += blob.sentiment.polarity
                        news_count += 1
                except:
                    continue
            
            avg_sentiment = sentiment_score / news_count if news_count > 0 else 0.0
            
            return {
                'sentiment_score': avg_sentiment,
                'news_count': news_count,
                'interpretation': self._interpret_sentiment(avg_sentiment)
            }
            
        except Exception as e:
            logger.error(f"News sentiment error: {e}")
            return {'sentiment_score': 0.0, 'news_count': 0, 'interpretation': 'NEUTRAL'}
    
    def _interpret_sentiment(self, score):
        """แปลค่า sentiment"""
        if score > 0.1:
            return 'BULLISH'
        elif score < -0.1:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

class ProfessionalMLEngine:
    """
    🤖 Professional Machine Learning Engine
    ระบบ ML แบบมืออาชีพ Multiple Models
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Model configurations
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    def prepare_ml_features(self, multi_tf_data):
        """เตรียม ML Features จาก Multi-timeframe"""
        try:
            features = []
            
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    continue
                
                # Get latest values
                latest = df.iloc[-1]
                
                # Price features
                features.extend([
                    latest.get('close', 0),
                    latest.get('open', 0),
                    latest.get('high', 0),
                    latest.get('low', 0),
                ])
                
                # Technical indicators
                features.extend([
                    latest.get('sma_10', 0),
                    latest.get('sma_20', 0),
                    latest.get('sma_50', 0),
                    latest.get('ema_10', 0),
                    latest.get('ema_20', 0),
                    latest.get('rsi_14', 50),
                    latest.get('rsi_7', 50),
                    latest.get('macd', 0),
                    latest.get('macd_signal', 0),
                    latest.get('macd_hist', 0),
                    latest.get('bb_position', 0.5),
                    latest.get('bb_width', 0),
                    latest.get('atr', 0),
                    latest.get('adx', 0),
                    latest.get('stoch_k', 50),
                    latest.get('stoch_d', 50),
                    latest.get('williams_r', -50),
                    latest.get('cci', 0),
                    latest.get('volume_ratio', 1),
                ])
                
                # Price action
                features.extend([
                    latest.get('body_size', 0),
                    latest.get('upper_shadow', 0),
                    latest.get('lower_shadow', 0),
                    latest.get('is_bullish', 0),
                ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return np.array([]).reshape(1, -1)
    
    def create_training_data(self, multi_tf_data, lookback=1000):
        """สร้างข้อมูลเทรน ML"""
        try:
            X, y = [], []
            
            # ใช้ M1 เป็นหลักในการสร้าง labels
            m1_data = multi_tf_data.get('M1')
            if m1_data is None or len(m1_data) < lookback:
                return None, None
            
            for i in range(100, len(m1_data) - 10):
                # Features จาก multi-timeframe
                features = []
                
                for tf, df in multi_tf_data.items():
                    if df is None:
                        continue
                    
                    # Map index ให้เหมาะสม
                    if tf == 'M1':
                        idx = i
                    elif tf == 'M5':
                        idx = i // 5
                    elif tf == 'M15':
                        idx = i // 15
                    elif tf == 'H1':
                        idx = i // 60
                    elif tf == 'H4':
                        idx = i // 240
                    elif tf == 'D1':
                        idx = i // 1440
                    else:
                        continue
                    
                    if idx >= len(df):
                        continue
                    
                    row = df.iloc[idx]
                    features.extend([
                        row.get('close', 0),
                        row.get('rsi_14', 50),
                        row.get('macd', 0),
                        row.get('bb_position', 0.5),
                        row.get('atr', 0),
                        row.get('adx', 0),
                        row.get('volume_ratio', 1),
                    ])
                
                if len(features) < 10:
                    continue
                
                # Label จาก M1
                current_price = m1_data.iloc[i]['close']
                future_price = m1_data.iloc[i + 10]['close']  # 10 นาทีข้างหน้า
                
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.002:  # +0.2%
                    label = 2  # BUY
                elif price_change < -0.002:  # -0.2%
                    label = 0  # SELL
                else:
                    label = 1  # HOLD
                
                X.append(features)
                y.append(label)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Training data creation error: {e}")
            return None, None
    
    def train_ensemble_models(self, X, y):
        """เทรน Ensemble Models"""
        try:
            logger.info("Training professional ML models...")
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            self.scalers['standard'] = StandardScaler()
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            self.models['lightgbm'] = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            # Train XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(**self.xgb_params)
            self.models['xgboost'].fit(X_train_scaled, y_train)
            
            # Train Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['random_forest'].fit(X_train_scaled, y_train)
            
            # Create ensemble
            self.models['ensemble'] = VotingClassifier([
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ], voting='soft')
            self.models['ensemble'].fit(X_train_scaled, y_train)
            
            # Evaluate
            self._evaluate_models(X_test_scaled, y_test)
            
            self.is_trained = True
            logger.info("ML models trained successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def _evaluate_models(self, X_test, y_test):
        """ประเมินประสิทธิภาพ models"""
        try:
            for name, model in self.models.items():
                if name == 'lightgbm':
                    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = model.predict(X_test)
                
                accuracy = np.mean(y_pred == y_test)
                logger.info(f"{name} accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
    
    def predict_market_direction(self, features):
        """พยากรณ์ทิศทางตลาด"""
        try:
            if not self.is_trained:
                return None
            
            # Scale features
            features_scaled = self.scalers['standard'].transform(features)
            
            predictions = {}
            
            # LightGBM prediction
            if 'lightgbm' in self.models:
                lgb_pred = self.models['lightgbm'].predict(features_scaled)
                predictions['lightgbm'] = {
                    'probs': lgb_pred[0],
                    'direction': np.argmax(lgb_pred[0]),
                    'confidence': np.max(lgb_pred[0])
                }
            
            # XGBoost prediction
            if 'xgboost' in self.models:
                xgb_pred = self.models['xgboost'].predict_proba(features_scaled)
                predictions['xgboost'] = {
                    'probs': xgb_pred[0],
                    'direction': np.argmax(xgb_pred[0]),
                    'confidence': np.max(xgb_pred[0])
                }
            
            # Random Forest prediction
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict_proba(features_scaled)
                predictions['random_forest'] = {
                    'probs': rf_pred[0],
                    'direction': np.argmax(rf_pred[0]),
                    'confidence': np.max(rf_pred[0])
                }
            
            # Ensemble prediction
            if 'ensemble' in self.models:
                ens_pred = self.models['ensemble'].predict_proba(features_scaled)
                predictions['ensemble'] = {
                    'probs': ens_pred[0],
                    'direction': np.argmax(ens_pred[0]),
                    'confidence': np.max(ens_pred[0])
                }
            
            # Final consensus
            directions = [p['direction'] for p in predictions.values()]
            confidences = [p['confidence'] for p in predictions.values()]
            
            final_direction = max(set(directions), key=directions.count)
            final_confidence = np.mean(confidences)
            consensus_ratio = directions.count(final_direction) / len(directions)
            
            return {
                'direction': final_direction,
                'confidence': final_confidence,
                'consensus_ratio': consensus_ratio,
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

class AdvancedRiskManager:
    """
    🛡️ Advanced Risk Management System
    ระบบบริหารความเสี่ยงแบบมืออาชีพ
    """
    
    def __init__(self, max_risk_per_trade=0.02, max_total_risk=0.10):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_total_risk = max_total_risk  # 10% total exposure
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.05  # 5% daily stop
        
    def calculate_position_size(self, balance, entry_price, stop_loss, confidence, atr):
        """คำนวณขนาดไม้แบบ dynamic"""
        try:
            # Basic risk per trade
            risk_amount = balance * self.max_risk_per_trade
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 1.5, 1.0)
            risk_amount *= confidence_multiplier
            
            # Adjust based on volatility
            volatility_ratio = atr / entry_price
            if volatility_ratio > 0.003:  # High volatility
                risk_amount *= 0.7
            elif volatility_ratio < 0.001:  # Low volatility
                risk_amount *= 1.2
            
            # Calculate position size
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
            else:
                position_size = 0
            
            # Apply limits
            max_size = balance * 0.1  # Max 10% of balance
            position_size = min(position_size, max_size)
            
            return {
                'size': position_size,
                'risk_amount': risk_amount,
                'risk_percentage': (position_size * stop_distance) / balance * 100
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return {'size': 0, 'risk_amount': 0, 'risk_percentage': 0}
    
    def calculate_stop_loss_take_profit(self, entry_price, direction, atr, confidence):
        """คำนวณ SL/TP แบบ dynamic"""
        try:
            # Base multipliers
            sl_multiplier = 2.0
            tp_multiplier = 3.0
            
            # Adjust based on confidence
            if confidence > 0.8:
                tp_multiplier = 4.0  # Higher target for high confidence
            elif confidence < 0.6:
                sl_multiplier = 1.5  # Tighter stop for low confidence
            
            # Calculate levels
            if direction == 'BUY':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_distance': abs(entry_price - stop_loss),
                'tp_distance': abs(take_profit - entry_price),
                'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            }
            
        except Exception as e:
            logger.error(f"SL/TP calculation error: {e}")
            return None
    
    def check_risk_limits(self, new_position_risk):
        """ตรวจสอบขีดจำกัดความเสี่ยง"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return False, "Daily loss limit exceeded"
            
            # Check total exposure
            total_risk = sum([pos.get('risk', 0) for pos in self.current_positions.values()])
            if total_risk + new_position_risk > self.max_total_risk:
                return False, "Total risk limit exceeded"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, "Risk check error"

class ProfessionalTradingEngine:
    """
    🚀 Professional Trading Engine
    ระบบเทรดแบบมืออาชีพครบครัน
    """
    
    def __init__(self):
        self.data_manager = ProfessionalDataManager()
        self.news_analyzer = NewsAndSentimentAnalyzer()
        self.ml_engine = ProfessionalMLEngine()
        self.risk_manager = AdvancedRiskManager()
        
        self.is_trading = False
        self.current_positions = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
    async def initialize_system(self):
        """เริ่มต้นระบบแบบครบครัน"""
        try:
            logger.info("🚀 Initializing Professional AI Trading System...")
            
            # Load multi-timeframe data
            logger.info("📊 Loading multi-timeframe data...")
            multi_data = self.data_manager.get_multi_timeframe_data(periods=2000)
            
            if len(multi_data) < 3:
                logger.error("❌ Insufficient market data")
                return False
            
            # Train ML models
            logger.info("🤖 Training ML models...")
            X, y = self.ml_engine.create_training_data(multi_data)
            
            if X is not None and len(X) > 100:
                success = self.ml_engine.train_ensemble_models(X, y)
                if not success:
                    logger.error("❌ ML model training failed")
                    return False
            else:
                logger.error("❌ Insufficient training data")
                return False
            
            # Initialize news analysis
            logger.info("📰 Initializing news analysis...")
            news_sentiment = self.news_analyzer.analyze_news_sentiment()
            logger.info(f"📊 Market sentiment: {news_sentiment['interpretation']}")
            
            logger.info("✅ Professional AI Trading System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization error: {e}")
            return False
    
    async def analyze_market(self):
        """วิเคราะห์ตลาดแบบครบครัน"""
        try:
            # Get fresh multi-timeframe data
            multi_data = self.data_manager.get_multi_timeframe_data(periods=200)
            
            if not multi_data:
                return None
            
            # Prepare ML features
            features = self.ml_engine.prepare_ml_features(multi_data)
            
            if features.size == 0:
                return None
            
            # Get ML predictions
            ml_prediction = self.ml_engine.predict_market_direction(features)
            
            if not ml_prediction:
                return None
            
            # Get news sentiment
            news_sentiment = self.news_analyzer.analyze_news_sentiment()
            
            # Get current market conditions
            m1_data = multi_data.get('M1')
            if m1_data is None:
                return None
            
            current_price = m1_data['close'].iloc[-1]
            atr = m1_data['atr'].iloc[-1]
            
            # Combine all analysis
            analysis = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'atr': atr,
                'ml_prediction': ml_prediction,
                'news_sentiment': news_sentiment,
                'market_data': {tf: data.tail(1).to_dict('records')[0] for tf, data in multi_data.items()},
                'recommendation': self._generate_recommendation(ml_prediction, news_sentiment, multi_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return None
    
    def _generate_recommendation(self, ml_prediction, news_sentiment, multi_data):
        """สร้างคำแนะนำการเทรด"""
        try:
            recommendation = {
                'action': 'WAIT',
                'confidence': 0.0,
                'reasoning': [],
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'position_size': 0.0
            }
            
            # ML signals
            ml_direction = ml_prediction['direction']
            ml_confidence = ml_prediction['confidence']
            consensus = ml_prediction['consensus_ratio']
            
            # News sentiment adjustment
            sentiment_score = news_sentiment['sentiment_score']
            
            # Market conditions
            m1_data = multi_data.get('M1')
            if m1_data is None:
                return recommendation
            
            current_price = m1_data['close'].iloc[-1]
            atr = m1_data['atr'].iloc[-1]
            rsi = m1_data['rsi_14'].iloc[-1]
            
            # Decision logic
            base_confidence = ml_confidence * consensus
            
            # Adjust for news sentiment
            if ml_direction == 2 and sentiment_score > 0:  # BUY + Bullish news
                base_confidence *= 1.15
            elif ml_direction == 0 and sentiment_score < 0:  # SELL + Bearish news
                base_confidence *= 1.15
            elif (ml_direction == 2 and sentiment_score < -0.1) or (ml_direction == 0 and sentiment_score > 0.1):
                base_confidence *= 0.8  # Conflicting signals
            
            # RSI filter
            if ml_direction == 2 and rsi > 80:  # Overbought
                base_confidence *= 0.7
            elif ml_direction == 0 and rsi < 20:  # Oversold
                base_confidence *= 0.7
            
            # Final decision
            if base_confidence > 0.75:
                if ml_direction == 2:
                    recommendation['action'] = 'BUY'
                elif ml_direction == 0:
                    recommendation['action'] = 'SELL'
                
                if recommendation['action'] != 'WAIT':
                    recommendation['confidence'] = base_confidence
                    
                    # Calculate risk parameters
                    direction = recommendation['action']
                    risk_params = self.risk_manager.calculate_stop_loss_take_profit(
                        current_price, direction, atr, base_confidence
                    )
                    
                    if risk_params:
                        recommendation.update({
                            'entry_price': current_price,
                            'stop_loss': risk_params['stop_loss'],
                            'take_profit': risk_params['take_profit'],
                        })
                        
                        # Calculate position size
                        size_info = self.risk_manager.calculate_position_size(
                            10000,  # Assuming $10k balance
                            current_price,
                            risk_params['stop_loss'],
                            base_confidence,
                            atr
                        )
                        recommendation['position_size'] = size_info['size']
                    
                    # Add reasoning
                    recommendation['reasoning'] = [
                        f"ML Models consensus: {consensus:.2f}",
                        f"ML Confidence: {ml_confidence:.3f}",
                        f"News sentiment: {news_sentiment['interpretation']}",
                        f"Final confidence: {base_confidence:.3f}",
                        f"Risk/Reward: {risk_params.get('risk_reward_ratio', 0):.2f}" if risk_params else ""
                    ]
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return recommendation
    
    async def start_real_time_trading(self):
        """เริ่มการเทรดแบบ real-time"""
        try:
            logger.info("🔥 Starting real-time professional trading...")
            self.is_trading = True
            
            while self.is_trading:
                try:
                    # Analyze market
                    analysis = await self.analyze_market()
                    
                    if analysis:
                        recommendation = analysis['recommendation']
                        
                        # Log analysis
                        logger.info(f"🎯 Market Analysis at {analysis['timestamp'].strftime('%H:%M:%S')}")
                        logger.info(f"💰 Price: {analysis['current_price']:.2f}")
                        logger.info(f"🤖 ML Prediction: {['SELL', 'HOLD', 'BUY'][analysis['ml_prediction']['direction']]}")
                        logger.info(f"🎯 Confidence: {analysis['ml_prediction']['confidence']:.3f}")
                        logger.info(f"📰 News Sentiment: {analysis['news_sentiment']['interpretation']}")
                        
                        if recommendation['action'] != 'WAIT':
                            logger.info(f"🚨 SIGNAL: {recommendation['action']}")
                            logger.info(f"💪 Confidence: {recommendation['confidence']:.3f}")
                            logger.info(f"💰 Entry: {recommendation['entry_price']:.2f}")
                            logger.info(f"🛑 SL: {recommendation['stop_loss']:.2f}")
                            logger.info(f"🎯 TP: {recommendation['take_profit']:.2f}")
                            logger.info(f"📊 Size: {recommendation['position_size']:.4f}")
                            logger.info(f"💭 Reasoning: {', '.join(recommendation['reasoning'])}")
                            
                            # Here you would execute the trade in real system
                            # await self.execute_trade(recommendation)
                        
                        else:
                            logger.info("⏳ WAITING - No strong signal")
                    
                    # Wait before next analysis
                    await asyncio.sleep(30)  # Analyze every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Real-time trading error: {e}")
    
    def stop_trading(self):
        """หยุดการเทรด"""
        self.is_trading = False
        logger.info("🛑 Trading stopped")
    
    def get_performance_report(self):
        """รายงานประสิทธิภาพ"""
        return {
            'total_trades': self.performance_stats['total_trades'],
            'winning_trades': self.performance_stats['winning_trades'],
            'win_rate': self.performance_stats['winning_trades'] / max(self.performance_stats['total_trades'], 1),
            'total_pnl': self.performance_stats['total_pnl'],
            'max_drawdown': self.performance_stats['max_drawdown']
        }

# ================================
# MAIN EXECUTION
# ================================

async def main():
    """เริ่มระบบ Professional AI Trading"""
    try:
        print("🚀 PROFESSIONAL AI TRADING SYSTEM")
        print("=" * 50)
        
        # Create trading engine
        engine = ProfessionalTradingEngine()
        
        # Initialize system
        print("🔄 Initializing system...")
        success = await engine.initialize_system()
        
        if not success:
            print("❌ System initialization failed")
            return
        
        print("✅ System ready!")
        print("\n🔥 Starting real-time analysis...")
        print("Press Ctrl+C to stop\n")
        
        # Start real-time trading
        try:
            await engine.start_real_time_trading()
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
            engine.stop_trading()
            
            # Show performance report
            report = engine.get_performance_report()
            print(f"\n📊 Performance Report:")
            print(f"Total Trades: {report['total_trades']}")
            print(f"Win Rate: {report['win_rate']:.2%}")
            print(f"Total PnL: ${report['total_pnl']:.2f}")
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    # Run the professional system
    asyncio.run(main())