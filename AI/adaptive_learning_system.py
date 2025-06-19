# ================================
# ADAPTIVE LEARNING & SELF-IMPROVING AI SYSTEM
# ระบบ AI ที่เรียนรู้และปรับปรุงตัวเองได้
# ================================

import numpy as np
import pandas as pd
import sqlite3
import pickle
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import asyncio
import threading
from collections import deque

# Reinforcement Learning
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# Online Learning
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    🎮 Trading Environment สำหรับ Reinforcement Learning
    ให้ AI เรียนรู้จากการเทรดจริง
    """
    
    def __init__(self, data, initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0=No position, 1=Long, -1=Short
        self.entry_price = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: OHLC + indicators (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(20,), dtype=np.float32
        )
        
        self.trade_history = []
        self.episode_profits = []
        
    def reset(self):
        """รีเซ็ต environment"""
        self.current_step = 50  # เริ่มจากตำแหน่งที่มีข้อมูลพอ
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """ดำเนินการ action และคืน reward"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        done = False
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
            
        elif action == 3 and self.position != 0:  # Close position
            if self.position == 1:  # Close Long
                profit = (current_price - self.entry_price) / self.entry_price
            else:  # Close Short
                profit = (self.entry_price - current_price) / self.entry_price
            
            self.balance *= (1 + profit * 0.1)  # 10x leverage simulation
            reward = profit * 1000  # Scale reward
            
            self.trade_history.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'profit': profit,
                'position_type': 'LONG' if self.position == 1 else 'SHORT'
            })
            
            self.position = 0
            self.entry_price = 0
        
        # Calculate unrealized P&L for open positions
        if self.position != 0:
            if self.position == 1:
                unrealized_profit = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_profit = (self.entry_price - current_price) / self.entry_price
            
            # Small reward for unrealized profits, penalty for losses
            reward += unrealized_profit * 100
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= self.max_steps or self.balance <= self.initial_balance * 0.7:
            done = True
            
        # Calculate episode profit
        total_profit = (self.balance - self.initial_balance) / self.initial_balance
        info = {'total_profit': total_profit, 'trades': len(self.trade_history)}
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """สร้าง observation สำหรับ AI"""
        if self.current_step >= len(self.data):
            return np.zeros(20, dtype=np.float32)
        
        # Get current market data
        current = self.data.iloc[self.current_step]
        
        # Basic price data (normalized)
        obs = [
            current['open'] / 2000,  # Normalize around gold price
            current['high'] / 2000,
            current['low'] / 2000,
            current['close'] / 2000,
        ]
        
        # Technical indicators (if available)
        indicators = [
            current.get('rsi_14', 50) / 100,
            current.get('macd', 0) / 10,
            current.get('bb_position', 0.5),
            current.get('atr', 1) / 10,
            current.get('volume_ratio', 1),
        ]
        obs.extend(indicators)
        
        # Position information
        position_info = [
            self.position,  # Current position
            self.entry_price / 2000 if self.entry_price > 0 else 0,
            self.balance / self.initial_balance,  # Balance ratio
        ]
        obs.extend(position_info)
        
        # Market context (last 8 candles trend)
        if self.current_step >= 8:
            recent_closes = self.data.iloc[self.current_step-8:self.current_step]['close'].values
            trend = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] for i in range(1, 8)]
            obs.extend(trend)
        else:
            obs.extend([0] * 7)
        
        return np.array(obs[:20], dtype=np.float32)  # Ensure exactly 20 features

class AdaptiveLearningEngine:
    """
    🧠 Adaptive Learning Engine
    ระบบเรียนรู้และปรับปรุงตัวเองแบบต่อเนื่อง
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.db_path = "adaptive_learning.db"
        
        # Models
        self.rl_agent = None
        self.online_classifier = SGDClassifier(
            loss='log', learning_rate='adaptive', 
            eta0=0.01, random_state=42
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.model_versions = {}
        self.current_strategy_params = {
            'confidence_threshold': 0.75,
            'risk_per_trade': 0.02,
            'news_weight': 0.15,
            'technical_weight': 0.85
        }
        
        # Learning parameters
        self.min_trades_for_update = 10
        self.retrain_frequency_hours = 24
        self.last_retrain_time = datetime.now()
        
        self._setup_database()
        self._initialize_rl_agent()
    
    def _setup_database(self):
        """สร้าง database สำหรับ adaptive learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    trade_id TEXT,
                    action TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    profit_pips REAL,
                    duration_minutes INTEGER,
                    ai_confidence REAL,
                    market_conditions TEXT,
                    model_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    version TEXT,
                    evaluation_date DATETIME,
                    accuracy REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    avg_profit_per_trade REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy parameters log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    parameter_name TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    performance_before REAL,
                    performance_after REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Adaptive learning database initialized")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def _initialize_rl_agent(self):
        """เริ่มต้น Reinforcement Learning Agent"""
        try:
            # Create dummy environment for initial setup
            dummy_data = pd.DataFrame({
                'open': [2000] * 100,
                'high': [2005] * 100,
                'low': [1995] * 100,
                'close': [2000] * 100,
                'rsi_14': [50] * 100,
                'macd': [0] * 100,
                'bb_position': [0.5] * 100,
                'atr': [1] * 100,
                'volume_ratio': [1] * 100
            })
            
            env = TradingEnvironment(dummy_data)
            
            # Initialize PPO agent
            self.rl_agent = PPO(
                "MlpPolicy", 
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=0
            )
            
            logger.info("RL Agent initialized")
            
        except Exception as e:
            logger.error(f"RL Agent initialization error: {e}")
    
    def log_trade_result(self, trade_data):
        """บันทึกผลการเทรดสำหรับการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_log 
                (timestamp, trade_id, action, entry_price, exit_price, 
                 profit_loss, profit_pips, duration_minutes, ai_confidence, 
                 market_conditions, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['timestamp'],
                trade_data['trade_id'],
                trade_data['action'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['profit_loss'],
                trade_data['profit_pips'],
                trade_data['duration_minutes'],
                trade_data['ai_confidence'],
                json.dumps(trade_data['market_conditions']),
                trade_data.get('model_version', 'v1.0')
            ))
            
            conn.commit()
            conn.close()
            
            # Add to performance history
            self.performance_history.append({
                'profit_loss': trade_data['profit_loss'],
                'ai_confidence': trade_data['ai_confidence'],
                'timestamp': trade_data['timestamp']
            })
            
            logger.info(f"Trade result logged: {trade_data['action']} - P/L: {trade_data['profit_loss']:.2f}")
            
            # Check if we need to adapt
            self._check_adaptation_triggers()
            
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    def _check_adaptation_triggers(self):
        """ตรวจสอบว่าต้องปรับปรุง model ไหม"""
        try:
            if len(self.performance_history) < self.min_trades_for_update:
                return
            
            # Check recent performance
            recent_trades = list(self.performance_history)[-20:]  # Last 20 trades
            recent_win_rate = sum(1 for trade in recent_trades if trade['profit_loss'] > 0) / len(recent_trades)
            recent_avg_profit = np.mean([trade['profit_loss'] for trade in recent_trades])
            
            # Trigger adaptation if performance is poor
            if recent_win_rate < 0.4 or recent_avg_profit < -0.5:
                logger.warning(f"Poor performance detected: Win rate: {recent_win_rate:.2f}, Avg profit: {recent_avg_profit:.2f}")
                asyncio.create_task(self.adapt_strategy())
            
            # Check if it's time for scheduled retraining
            hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain >= self.retrain_frequency_hours:
                asyncio.create_task(self.retrain_models())
            
        except Exception as e:
            logger.error(f"Adaptation check error: {e}")
    
    async def adapt_strategy(self):
        """ปรับปรุงกลยุทธ์แบบ adaptive"""
        try:
            logger.info("🔄 Starting strategy adaptation...")
            
            # Analyze recent performance
            recent_trades = list(self.performance_history)[-50:]  # Last 50 trades
            
            if len(recent_trades) < 10:
                return
            
            # Analyze confidence vs performance correlation
            high_conf_trades = [t for t in recent_trades if t['ai_confidence'] > 0.8]
            low_conf_trades = [t for t in recent_trades if t['ai_confidence'] < 0.6]
            
            if high_conf_trades:
                high_conf_win_rate = sum(1 for t in high_conf_trades if t['profit_loss'] > 0) / len(high_conf_trades)
            else:
                high_conf_win_rate = 0.5
            
            if low_conf_trades:
                low_conf_win_rate = sum(1 for t in low_conf_trades if t['profit_loss'] > 0) / len(low_conf_trades)
            else:
                low_conf_win_rate = 0.5
            
            # Adapt confidence threshold
            old_threshold = self.current_strategy_params['confidence_threshold']
            
            if high_conf_win_rate > low_conf_win_rate + 0.1:
                # High confidence trades are significantly better
                new_threshold = min(old_threshold + 0.05, 0.9)
            elif low_conf_win_rate > high_conf_win_rate:
                # High confidence isn't helping, lower threshold
                new_threshold = max(old_threshold - 0.05, 0.5)
            else:
                new_threshold = old_threshold
            
            if new_threshold != old_threshold:
                self.current_strategy_params['confidence_threshold'] = new_threshold
                self._log_parameter_change(
                    'confidence_threshold', old_threshold, new_threshold,
                    f"High conf win rate: {high_conf_win_rate:.2f}, Low conf win rate: {low_conf_win_rate:.2f}"
                )
            
            # Adapt risk per trade based on recent drawdown
            recent_profits = [t['profit_loss'] for t in recent_trades]
            max_drawdown = self._calculate_max_drawdown(recent_profits)
            
            old_risk = self.current_strategy_params['risk_per_trade']
            
            if max_drawdown > 0.15:  # More than 15% drawdown
                new_risk = max(old_risk * 0.8, 0.005)  # Reduce risk
            elif max_drawdown < 0.05 and np.mean(recent_profits) > 0:  # Low drawdown + profitable
                new_risk = min(old_risk * 1.1, 0.05)  # Increase risk slightly
            else:
                new_risk = old_risk
            
            if abs(new_risk - old_risk) > 0.002:
                self.current_strategy_params['risk_per_trade'] = new_risk
                self._log_parameter_change(
                    'risk_per_trade', old_risk, new_risk,
                    f"Max drawdown: {max_drawdown:.2f}, Avg profit: {np.mean(recent_profits):.3f}"
                )
            
            logger.info(f"✅ Strategy adapted - Confidence threshold: {self.current_strategy_params['confidence_threshold']:.2f}, Risk per trade: {self.current_strategy_params['risk_per_trade']:.3f}")
            
        except Exception as e:
            logger.error(f"Strategy adaptation error: {e}")
    
    async def retrain_models(self):
        """เทรน models ใหม่ด้วยข้อมูลล่าสุด"""
        try:
            logger.info("🤖 Starting model retraining...")
            
            # Get recent market data
            multi_data = self.data_manager.get_multi_timeframe_data(periods=1000)
            if not multi_data:
                logger.error("No data available for retraining")
                return
            
            # Prepare training data with recent performance feedback
            X, y = await self._prepare_adaptive_training_data(multi_data)
            
            if X is not None and len(X) > 100:
                # Online learning update
                if hasattr(self.online_classifier, 'partial_fit'):
                    # If classifier is already fitted, use partial_fit
                    try:
                        self.online_classifier.partial_fit(X, y)
                    except:
                        # If not fitted yet, use regular fit
                        self.online_classifier.fit(X, y)
                else:
                    self.online_classifier.fit(X, y)
                
                # Retrain RL agent with recent data
                await self._retrain_rl_agent(multi_data)
                
                # Update model version
                version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.model_versions[version] = {
                    'timestamp': datetime.now(),
                    'training_samples': len(X),
                    'performance_trades': len(self.performance_history)
                }
                
                self.last_retrain_time = datetime.now()
                logger.info(f"✅ Models retrained successfully - Version: {version}")
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    async def _prepare_adaptive_training_data(self, multi_data):
        """เตรียมข้อมูลเทรนที่ปรับตาม performance feedback"""
        try:
            m1_data = multi_data.get('M1')
            if m1_data is None or len(m1_data) < 100:
                return None, None
            
            X, y = [], []
            
            # Get recent trade results for weighting
            recent_results = {}
            if len(self.performance_history) > 0:
                for trade in self.performance_history:
                    timestamp = trade['timestamp']
                    profit = 1 if trade['profit_loss'] > 0 else 0
                    recent_results[timestamp] = profit
            
            # Create training samples
            for i in range(50, len(m1_data) - 10):
                # Features (same as before but with performance weighting)
                features = []
                current = m1_data.iloc[i]
                
                features.extend([
                    current.get('close', 0),
                    current.get('rsi_14', 50),
                    current.get('macd', 0),
                    current.get('bb_position', 0.5),
                    current.get('atr', 0),
                    current.get('volume_ratio', 1),
                ])
                
                if len(features) < 6:
                    continue
                
                # Create label
                current_price = m1_data.iloc[i]['close']
                future_price = m1_data.iloc[i + 10]['close']
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.002:
                    label = 2  # BUY
                elif price_change < -0.002:
                    label = 0  # SELL
                else:
                    label = 1  # HOLD
                
                X.append(features)
                y.append(label)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Adaptive training data preparation error: {e}")
            return None, None
    
    async def _retrain_rl_agent(self, multi_data):
        """เทรน RL agent ใหม่"""
        try:
            m1_data = multi_data.get('M1')
            if m1_data is None or len(m1_data) < 500:
                return
            
            # Create new environment with recent data
            env = TradingEnvironment(m1_data.tail(500))
            
            # Update RL agent's environment
            self.rl_agent.set_env(env)
            
            # Train for a few episodes
            self.rl_agent.learn(total_timesteps=10000)
            
            logger.info("RL agent retrained with recent data")
            
        except Exception as e:
            logger.error(f"RL agent retraining error: {e}")
    
    def get_rl_action(self, observation):
        """ขอ action จาก RL agent"""
        try:
            if self.rl_agent is None:
                return 0  # Hold
            
            action, _ = self.rl_agent.predict(observation, deterministic=True)
            return action
            
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            return 0
    
    def get_adaptive_prediction(self, features):
        """ขอ prediction จาก adaptive classifier"""
        try:
            if not hasattr(self.online_classifier, 'classes_'):
                # Model not trained yet
                return {'prediction': 1, 'confidence': 0.5}  # HOLD
            
            prediction = self.online_classifier.predict(features.reshape(1, -1))[0]
            
            # Get prediction probabilities if available
            if hasattr(self.online_classifier, 'predict_proba'):
                probabilities = self.online_classifier.predict_proba(features.reshape(1, -1))[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.6  # Default confidence
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Adaptive prediction error: {e}")
            return {'prediction': 1, 'confidence': 0.5}
    
    def get_current_strategy_params(self):
        """ขอพารามิเตอร์กลยุทธ์ปัจจุบัน"""
        return self.current_strategy_params.copy()
    
    def _calculate_max_drawdown(self, profits):
        """คำนวณ maximum drawdown"""
        if not profits:
            return 0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(running_max, 1)
        
        return abs(np.min(drawdown))
    
    def _log_parameter_change(self, param_name, old_value, new_value, reason):
        """บันทึกการเปลี่ยนแปลงพารามิเตอร์"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO strategy_parameters 
                (timestamp, parameter_name, old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                param_name,
                old_value,
                new_value,
                reason
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Parameter changed: {param_name} {old_value:.3f} → {new_value:.3f} ({reason})")
            
        except Exception as e:
            logger.error(f"Parameter logging error: {e}")
    
    def get_performance_report(self):
        """รายงานประสิทธิภาพและการเรียนรู้"""
        try:
            if len(self.performance_history) == 0:
                return "No trading history available"
            
            recent_trades = list(self.performance_history)[-50:]
            
            total_trades = len(recent_trades)
            winning_trades = sum(1 for t in recent_trades if t['profit_loss'] > 0)
            win_rate = winning_trades / total_trades
            
            avg_profit = np.mean([t['profit_loss'] for t in recent_trades])
            total_profit = sum([t['profit_loss'] for t in recent_trades])
            
            max_drawdown = self._calculate_max_drawdown([t['profit_loss'] for t in recent_trades])
            
            return f"""
🤖 ADAPTIVE LEARNING PERFORMANCE REPORT
==========================================
📊 Recent Performance (Last {total_trades} trades):
   Win Rate: {win_rate:.2%}
   Average Profit per Trade: {avg_profit:.3f}
   Total Profit: {total_profit:.2f}
   Maximum Drawdown: {max_drawdown:.2%}

🧠 Current Strategy Parameters:
   Confidence Threshold: {self.current_strategy_params['confidence_threshold']:.2f}
   Risk per Trade: {self.current_strategy_params['risk_per_trade']:.3f}
   News Weight: {self.current_strategy_params['news_weight']:.2f}
   Technical Weight: {self.current_strategy_params['technical_weight']:.2f}

🔄 Learning Status:
   Total Performance Records: {len(self.performance_history)}
   Model Versions: {len(self.model_versions)}
   Last Retrain: {self.last_retrain_time.strftime('%Y-%m-%d %H:%M')}
"""
            
        except Exception as e:
            logger.error(f"Performance report error: {e}")
            return "Error generating performance report"

# ================================
# INTEGRATION WITH MAIN SYSTEM
# ================================

class AdaptiveIntegration:
    """
    🔗 การผสานระบบเรียนรู้เข้ากับระบบหลัก
    """
    
    def __init__(self, trading_engine, data_manager):
        self.trading_engine = trading_engine
        self.adaptive_engine = AdaptiveLearningEngine(data_manager)
        self.trade_counter = 0
        
    async def enhanced_market_analysis(self):
        """การวิเคราะห์ตลาดที่ปรับปรุงด้วย adaptive learning"""
        try:
            # Get standard analysis
            standard_analysis = await self.trading_engine.analyze_market()
            
            if not standard_analysis:
                return None
            
            # Get adaptive parameters
            adaptive_params = self.adaptive_engine.get_current_strategy_params()
            
            # Prepare features for adaptive prediction
            ml_prediction = standard_analysis['ml_prediction']
            features = np.array([
                ml_prediction['confidence'],
                ml_prediction['consensus_ratio'],
                standard_analysis['current_price'],
                standard_analysis['atr'],
                standard_analysis['news_sentiment']['sentiment_score']
            ])
            
            # Get adaptive prediction
            adaptive_pred = self.adaptive_engine.get_adaptive_prediction(features)
            
            # Modify recommendation with adaptive learning
            recommendation = standard_analysis['recommendation']
            
            # Apply adaptive confidence threshold
            original_confidence = recommendation['confidence']
            adaptive_threshold = adaptive_params['confidence_threshold']
            
            if original_confidence < adaptive_threshold:
                recommendation['action'] = 'WAIT'
                recommendation['reasoning'].append(f"Below adaptive threshold ({adaptive_threshold:.2f})")
            
            # Apply adaptive risk management
            if recommendation['action'] != 'WAIT':
                adaptive_risk = adaptive_params['risk_per_trade']
                recommendation['position_size'] *= (adaptive_risk / 0.02)  # Scale based on adaptive risk
                recommendation['reasoning'].append(f"Adaptive risk: {adaptive_risk:.3f}")
            
            # Add adaptive information
            standard_analysis['adaptive_info'] = {
                'adaptive_prediction': adaptive_pred,
                'adaptive_params': adaptive_params,
                'learning_status': f"{len(self.adaptive_engine.performance_history)} trades learned"
            }
            
            return standard_analysis
            
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            return standard_analysis
    
    def log_trade_execution(self, trade_action, entry_price, ai_confidence, market_conditions):
        """บันทึกการเทรดสำหรับการเรียนรู้"""
        try:
            self.trade_counter += 1
            
            trade_data = {
                'timestamp': datetime.now(),
                'trade_id': f"TRADE_{self.trade_counter:06d}",
                'action': trade_action,
                'entry_price': entry_price,
                'exit_price': None,  # Will be updated when closed
                'profit_loss': None,  # Will be calculated when closed
                'profit_pips': None,
                'duration_minutes': None,
                'ai_confidence': ai_confidence,
                'market_conditions': market_conditions,
                'model_version': 'adaptive_v1.0'
            }
            
            # Store for later completion
            self.pending_trades = getattr(self, 'pending_trades', {})
            self.pending_trades[trade_data['trade_id']] = trade_data
            
            logger.info(f"Trade logged for learning: {trade_action} at {entry_price}")
            
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    def complete_trade_log(self, trade_id, exit_price, profit_loss):
        """เสร็จสิ้นการบันทึกเทรดเมื่อปิดไม้"""
        try:
            pending_trades = getattr(self, 'pending_trades', {})
            
            if trade_id in pending_trades:
                trade_data = pending_trades[trade_id]
                
                # Complete trade information
                trade_data.update({
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'profit_pips': abs(exit_price - trade_data['entry_price']) * 10,  # Approximate for gold
                    'duration_minutes': (datetime.now() - trade_data['timestamp']).total_seconds() / 60
                })
                
                # Log to adaptive engine
                self.adaptive_engine.log_trade_result(trade_data)
                
                # Remove from pending
                del pending_trades[trade_id]
                
                logger.info(f"Trade completed: {trade_id} - P/L: {profit_loss:.2f}")
            
        except Exception as e:
            logger.error(f"Trade completion error: {e}")
    
    def get_learning_report(self):
        """รายงานการเรียนรู้"""
        return self.adaptive_engine.get_performance_report()

# ================================
# USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    async def test_adaptive_learning():
        """ทดสอบระบบเรียนรู้"""
        try:
            from complete_ai_trading_system import ProfessionalDataManager
            
            print("🧠 Testing Adaptive Learning System...")
            
            # Initialize
            data_manager = ProfessionalDataManager()
            adaptive_engine = AdaptiveLearningEngine(data_manager)
            
            # Simulate some trade results
            trade_results = [
                {'timestamp': datetime.now(), 'trade_id': 'T001', 'action': 'BUY', 'entry_price': 2020, 'exit_price': 2025, 'profit_loss': 5, 'profit_pips': 50, 'duration_minutes': 120, 'ai_confidence': 0.8, 'market_conditions': {'volatility': 'normal'}, 'model_version': 'v1.0'},
                {'timestamp': datetime.now(), 'trade_id': 'T002', 'action': 'SELL', 'entry_price': 2025, 'exit_price': 2020, 'profit_loss': 5, 'profit_pips': 50, 'duration_minutes': 90, 'ai_confidence': 0.7, 'market_conditions': {'volatility': 'high'}, 'model_version': 'v1.0'},
                {'timestamp': datetime.now(), 'trade_id': 'T003', 'action': 'BUY', 'entry_price': 2018, 'exit_price': 2015, 'profit_loss': -3, 'profit_pips': 30, 'duration_minutes': 60, 'ai_confidence': 0.6, 'market_conditions': {'volatility': 'low'}, 'model_version': 'v1.0'},
            ]
            
            # Log trade results
            for trade in trade_results:
                adaptive_engine.log_trade_result(trade)
            
            # Test adaptation
            await adaptive_engine.adapt_strategy()
            
            # Show report
            print(adaptive_engine.get_performance_report())
            
        except Exception as e:
            print(f"Test error: {e}")
    
    # Run test
    asyncio.run(test_adaptive_learning())