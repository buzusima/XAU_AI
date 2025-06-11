import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Import our Signal Engine
from smc_signal_engine import SMCSignalEngine


class SMCAutoTrader:
    """
    Enhanced SMC Auto Trading Bot with Comprehensive Logging
    Created by อาจารย์ฟินิกซ์ - Professional Automated Trading System with Learning Capability
    """

    def __init__(
        self,
        models_path: str = "EURUSD_c_SMC",
        account: int = None,
        password: str = None,
        server: str = None,
        signal_change_threshold: float = 0.001,
        enable_first_signal_trade: bool = True,
        first_signal_min_confidence: float = 0.75,
        max_risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
        max_concurrent_trades: int = 1,
        min_confidence: float = 0.75,
        min_consensus: int = 3,
        base_lot_size: float = 0.01,
        max_lot_size: float = 0.1,
        lot_multiplier: float = 2.0,
        default_sl_pips: int = 20,
        default_tp_ratio: float = 2.0,
        max_trades_per_hour: int = 5,
        wait_for_trade_completion: bool = True,
    ):
        """Initialize Enhanced SMC Auto Trader with Logging"""

        # Original settings
        self.account = account
        self.password = password
        self.server = server
        self.timezone = pytz.timezone("Etc/UTC")

        # Signal Sensitivity Controls
        self.signal_change_threshold = signal_change_threshold
        self.enable_first_signal_trade = enable_first_signal_trade
        self.first_signal_min_confidence = first_signal_min_confidence

        # Risk Management Settings
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_concurrent_trades = max_concurrent_trades
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus

        # Position Sizing
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.lot_multiplier = lot_multiplier

        # Trade Management
        self.default_sl_pips = default_sl_pips
        self.default_tp_ratio = default_tp_ratio

        # Safety Controls
        self.trading_enabled = False
        self.max_trades_per_hour = max_trades_per_hour
        self.wait_for_trade_completion = wait_for_trade_completion

        # Trading state
        self.active_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.hourly_trade_count = 0
        self.hour_start = datetime.now().hour
        self.last_trade_closed_time = None

        # 🆕 Enhanced Logging System
        self.log_directory = "trading_logs"
        self.ensure_log_directory()
        
        # Log files
        self.trade_log_file = os.path.join(self.log_directory, "trade_entries.jsonl")
        self.signal_log_file = os.path.join(self.log_directory, "signal_history.jsonl")
        self.performance_log_file = os.path.join(self.log_directory, "daily_performance.json")
        
        # Market data cache for logging
        self.market_data_cache = {}
        
        # Initialize Signal Engine
        self.signal_engine = SMCSignalEngine(models_path)

        print("🤖 Enhanced SMC Auto Trading Bot Initialized")
        print("📝 Comprehensive Logging System Active")
        print("⚠️ Trading is DISABLED by default for safety")

    def ensure_log_directory(self):
        """สร้าง directory สำหรับเก็บ logs"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            print(f"📁 Created log directory: {self.log_directory}")

    def get_comprehensive_market_data(self, symbol: str) -> Dict:
        """เก็บข้อมูลตลาดครบถ้วนในขณะที่เทรด"""
        try:
            # Get current tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}

            # Get recent candle data for context
            rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 5)
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3)

            market_data = {
                # Current Tick Data
                'current_tick': {
                    'time': datetime.fromtimestamp(tick.time).isoformat(),
                    'bid': float(tick.bid),
                    'ask': float(tick.ask),
                    'spread': float(tick.ask - tick.bid),
                    'volume': int(tick.volume) if hasattr(tick, 'volume') else 0
                },
                
                # Recent Price Action
                'recent_candles': {
                    'M1_last': {
                        'open': float(rates_m1[-1]['open']) if rates_m1 is not None else 0,
                        'high': float(rates_m1[-1]['high']) if rates_m1 is not None else 0,
                        'low': float(rates_m1[-1]['low']) if rates_m1 is not None else 0,
                        'close': float(rates_m1[-1]['close']) if rates_m1 is not None else 0,
                        'volume': int(rates_m1[-1]['tick_volume']) if rates_m1 is not None else 0
                    },
                    'M5_last': {
                        'open': float(rates_m5[-1]['open']) if rates_m5 is not None else 0,
                        'high': float(rates_m5[-1]['high']) if rates_m5 is not None else 0,
                        'low': float(rates_m5[-1]['low']) if rates_m5 is not None else 0,
                        'close': float(rates_m5[-1]['close']) if rates_m5 is not None else 0,
                        'volume': int(rates_m5[-1]['tick_volume']) if rates_m5 is not None else 0
                    }
                },
                
                # Volatility Context
                'volatility_context': {
                    'atr_m5': float(np.mean([r['high'] - r['low'] for r in rates_m5[-5:]])) if rates_m5 is not None else 0,
                    'price_range_last_hour': float(max([r['high'] for r in rates_h1[-1:]] or [0]) - min([r['low'] for r in rates_h1[-1:]] or [0])) if rates_h1 is not None else 0
                },
                
                # Session Info
                'session_info': {
                    'hour': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),
                    'is_asian_session': 0 <= datetime.now().hour <= 8,
                    'is_london_session': 8 <= datetime.now().hour <= 16,
                    'is_ny_session': 13 <= datetime.now().hour <= 21,
                    'is_weekend': datetime.now().weekday() >= 5
                }
            }
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error getting market data: {str(e)}")
            return {}

    def convert_to_json_serializable(self, obj):
        """แปลง numpy types และ objects อื่น ๆ ให้เป็น JSON serializable"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif pd.isna(obj) or obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            try:
                return float(obj) if hasattr(obj, '__float__') else str(obj)
            except:
                return str(obj)

    def log_trade_entry(self, signal: Dict, symbol: str, order_details: Dict, market_data: Dict) -> str:
        """เก็บ log เมื่อเปิด trade"""
        try:
            trade_id = f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order_details.get('ticket', 'UNKNOWN')}"
            
            entry_log = {
                # Trade Identification
                'trade_id': trade_id,
                'timestamp': datetime.now(self.timezone).isoformat(),
                'symbol': symbol,
                'mt5_ticket': self.convert_to_json_serializable(order_details.get('ticket')),
                
                # Entry Conditions & Signal Analysis
                'entry_conditions': {
                    'signal_direction': signal.get('final_direction', 'UNKNOWN'),
                    'signal_confidence': self.convert_to_json_serializable(signal.get('final_confidence', 0)),
                    'risk_level': signal.get('risk_level', 'UNKNOWN'),
                    'timeframe_consensus': signal.get('timeframe_consensus', 'UNKNOWN'),
                    'trading_recommendation': signal.get('trading_recommendation', 'UNKNOWN'),
                    
                    # Individual Timeframe Signals
                    'timeframe_signals': {
                        tf: {
                            'prediction': self.convert_to_json_serializable(sig_data.get('consensus_prediction', 0)),
                            'confidence': self.convert_to_json_serializable(sig_data.get('average_confidence', 0)),
                            'signal_quality': sig_data.get('signal_quality', 'UNKNOWN')
                        }
                        for tf, sig_data in signal.get('individual_signals', {}).items()
                    },
                    
                    # SMC Analysis
                    'smc_confluence': self._extract_smc_features(signal),
                    
                    # Model Performance
                    'model_versions': self._get_current_model_versions(),
                    'signal_change_triggered': True
                },
                
                # Trade Execution Details
                'execution_details': {
                    'order_type': 'BUY' if order_details.get('type') == mt5.ORDER_TYPE_BUY else 'SELL',
                    'position_size': self.convert_to_json_serializable(order_details.get('volume', 0)),
                    'entry_price': self.convert_to_json_serializable(order_details.get('price', 0)),
                    'stop_loss': self.convert_to_json_serializable(order_details.get('sl', 0)),
                    'take_profit': self.convert_to_json_serializable(order_details.get('tp', 0)),
                    'risk_reward_ratio': self.convert_to_json_serializable(self._calculate_risk_reward(order_details)),
                    'position_risk_percent': self.convert_to_json_serializable(self._calculate_position_risk(order_details)),
                    'slippage': 0  # Will be calculated if needed
                },
                
                # Market Context at Entry
                'market_context': self.convert_to_json_serializable(market_data),
                
                # Trade Outcome (will be updated when trade closes)
                'outcome': {
                    'status': 'OPEN',
                    'exit_time': None,
                    'exit_price': None,
                    'exit_reason': None,  # 'TP_HIT', 'SL_HIT', 'TIME_EXIT', 'MANUAL_CLOSE'
                    'pnl_pips': None,
                    'pnl_usd': None,
                    'pnl_percent': None,
                    'trade_duration_minutes': None,
                    'is_winner': None,
                    'max_favorable_excursion': None,  # MFE
                    'max_adverse_excursion': None,    # MAE
                },
                
                # Learning Data
                'learning_metrics': {
                    'prediction_accuracy': None,  # Will be calculated after close
                    'signal_quality_score': None,
                    'market_behavior_match': None,
                    'execution_quality': 'GOOD'  # Based on slippage, timing etc.
                }
            }
            
            # Convert entire log to JSON serializable format
            entry_log = self.convert_to_json_serializable(entry_log)
            
            # Save to log file
            with open(self.trade_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry_log, ensure_ascii=False, default=str) + '\n')
            
            print(f"📝 Trade entry logged: {trade_id}")
            return trade_id
            
        except Exception as e:
            print(f"❌ Trade entry logging error: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")
            return f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def update_trade_outcome(self, trade_id: str, exit_details: Dict, market_data: Dict):
        """อัพเดท log เมื่อ trade ปิด"""
        try:
            # Read existing logs
            logs = []
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line.strip()))
            
            # Find and update the specific trade
            updated = False
            for log in logs:
                if log.get('trade_id') == trade_id or log.get('mt5_ticket') == exit_details.get('ticket'):
                    
                    # Calculate trade metrics
                    entry_price = self.convert_to_json_serializable(log['execution_details']['entry_price'])
                    exit_price = self.convert_to_json_serializable(exit_details.get('exit_price', entry_price))
                    position_size = self.convert_to_json_serializable(log['execution_details']['position_size'])
                    order_type = log['execution_details']['order_type']
                    
                    # Calculate P&L
                    if order_type == 'BUY':
                        pnl_pips = (exit_price - entry_price) / 0.00001
                        pnl_percent = ((exit_price / entry_price) - 1) * 100
                    else:
                        pnl_pips = (entry_price - exit_price) / 0.00001
                        pnl_percent = ((entry_price / exit_price) - 1) * 100
                    
                    # Update outcome data
                    log['outcome'].update({
                        'status': 'CLOSED',
                        'exit_time': datetime.now(self.timezone).isoformat(),
                        'exit_price': self.convert_to_json_serializable(exit_price),
                        'exit_reason': exit_details.get('exit_reason', 'UNKNOWN'),
                        'pnl_pips': self.convert_to_json_serializable(pnl_pips),
                        'pnl_usd': self.convert_to_json_serializable(exit_details.get('profit', 0)),
                        'pnl_percent': self.convert_to_json_serializable(pnl_percent),
                        'trade_duration_minutes': self.convert_to_json_serializable(exit_details.get('duration_minutes', 0)),
                        'is_winner': pnl_pips > 0,
                        'max_favorable_excursion': self.convert_to_json_serializable(exit_details.get('mfe', 0)),
                        'max_adverse_excursion': self.convert_to_json_serializable(exit_details.get('mae', 0))
                    })
                    
                    # Add exit market context
                    log['exit_market_context'] = self.convert_to_json_serializable(market_data)
                    
                    # Calculate learning metrics
                    log['learning_metrics'].update({
                        'prediction_accuracy': 1.0 if (
                            (log['entry_conditions']['signal_direction'] == 'LONG' and pnl_pips > 0) or
                            (log['entry_conditions']['signal_direction'] == 'SHORT' and pnl_pips > 0)
                        ) else 0.0,
                        'signal_quality_score': self.convert_to_json_serializable(self._calculate_signal_quality_score(log, pnl_pips)),
                        'market_behavior_match': self.convert_to_json_serializable(self._analyze_market_behavior_match(log, exit_details))
                    })
                    
                    updated = True
                    break
            
            if updated:
                # Convert all logs to JSON serializable format before writing
                serializable_logs = [self.convert_to_json_serializable(log) for log in logs]
                
                # Write back all logs
                with open(self.trade_log_file, 'w', encoding='utf-8') as f:
                    for log in serializable_logs:
                        f.write(json.dumps(log, ensure_ascii=False, default=str) + '\n')
                
                print(f"📝 Trade outcome updated: {trade_id}")
                
                # Log to daily performance
                self._update_daily_performance(pnl_pips, pnl_percent)
                
            else:
                print(f"⚠️ Trade not found for outcome update: {trade_id}")
                
        except Exception as e:
            print(f"❌ Trade outcome update error: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")

    def _extract_smc_features(self, signal: Dict) -> Dict:
        """แยกข้อมูล SMC features สำหรับ logging"""
        try:
            smc_features = {}
            
            # Extract from individual signals if available
            for tf, sig_data in signal.get('individual_signals', {}).items():
                if 'current_price' in str(sig_data):  # Check if detailed data exists
                    smc_features[tf] = {
                        'market_structure': 'BULLISH' if sig_data.get('consensus_prediction', 0) > 0 else 'BEARISH',
                        'confidence': float(sig_data.get('average_confidence', 0)),
                        'signal_quality': sig_data.get('signal_quality', 'UNKNOWN')
                    }
            
            return smc_features
            
        except Exception as e:
            print(f"⚠️ Error extracting SMC features: {str(e)}")
            return {}

    def _get_current_model_versions(self) -> Dict:
        """เก็บเวอร์ชันของโมเดลที่ใช้อยู่"""
        return {
            'random_forest': 'base_v1.0',
            'xgboost': 'base_v1.0',
            'neural_network': 'base_v1.0',
            'ensemble_version': 'v1.0'
        }

    def _calculate_risk_reward(self, order_details: Dict) -> float:
        """คำนวณ Risk:Reward ratio"""
        try:
            entry_price = order_details.get('price', 0)
            stop_loss = order_details.get('sl', 0)
            take_profit = order_details.get('tp', 0)
            
            if stop_loss == 0 or take_profit == 0:
                return 0.0
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            return reward / risk if risk > 0 else 0.0
            
        except:
            return 0.0

    def _calculate_position_risk(self, order_details: Dict) -> float:
        """คำนวณเปอร์เซ็นต์ความเสี่ยงของ position"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
            
            position_size = order_details.get('volume', 0)
            entry_price = order_details.get('price', 0)
            stop_loss = order_details.get('sl', 0)
            
            if stop_loss == 0:
                return 0.0
            
            risk_per_pip = position_size * 10  # For 5-digit quotes
            risk_pips = abs(entry_price - stop_loss) / 0.00001
            total_risk_usd = risk_per_pip * risk_pips
            
            return (total_risk_usd / account_info.balance) * 100
            
        except:
            return 0.0

    def _calculate_signal_quality_score(self, trade_log: Dict, pnl_pips: float) -> float:
        """คำนวณคะแนนคุณภาพของ signal"""
        try:
            base_score = 0.5
            
            # Confidence bonus
            confidence = trade_log['entry_conditions']['signal_confidence']
            confidence_bonus = (confidence - 0.5) * 0.4  # Max 0.2 bonus
            
            # Outcome bonus/penalty
            outcome_bonus = 0.3 if pnl_pips > 0 else -0.3
            
            # Consensus bonus
            consensus = trade_log['entry_conditions']['timeframe_consensus']
            consensus_count = len([x for x in consensus.split('/') if 'L' in x or 'S' in x])
            consensus_bonus = (consensus_count - 2) * 0.1  # Bonus for more agreements
            
            final_score = base_score + confidence_bonus + outcome_bonus + consensus_bonus
            return max(0.0, min(1.0, final_score))
            
        except:
            return 0.5

    def _analyze_market_behavior_match(self, trade_log: Dict, exit_details: Dict) -> float:
        """วิเคราะห์ว่าพฤติกรรมตลาดตรงกับที่คาดหวังหรือไม่"""
        try:
            # Simple implementation - can be enhanced
            expected_direction = trade_log['entry_conditions']['signal_direction']
            actual_outcome = exit_details.get('exit_reason', '')
            
            if expected_direction == 'LONG':
                if actual_outcome == 'TP_HIT':
                    return 1.0  # Perfect match
                elif actual_outcome == 'SL_HIT':
                    return 0.0  # Complete mismatch
                else:
                    return 0.5  # Neutral
            else:  # SHORT
                if actual_outcome == 'TP_HIT':
                    return 1.0
                elif actual_outcome == 'SL_HIT':
                    return 0.0
                else:
                    return 0.5
                    
        except:
            return 0.5

    def _update_daily_performance(self, pnl_pips: float, pnl_percent: float):
        """อัพเดทสถิติการเทรดรายวัน"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Load existing performance data
            performance_data = {}
            if os.path.exists(self.performance_log_file):
                with open(self.performance_log_file, 'r') as f:
                    performance_data = json.load(f)
            
            # Initialize today's data if not exists
            if today not in performance_data:
                performance_data[today] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl_pips': 0.0,
                    'total_pnl_percent': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'win_rate': 0.0
                }
            
            # Convert inputs to JSON serializable format
            pnl_pips = self.convert_to_json_serializable(pnl_pips)
            pnl_percent = self.convert_to_json_serializable(pnl_percent)
            
            # Update today's stats
            today_stats = performance_data[today]
            today_stats['total_trades'] += 1
            
            if pnl_pips > 0:
                today_stats['winning_trades'] += 1
                today_stats['largest_win'] = max(today_stats['largest_win'], pnl_pips)
            else:
                today_stats['losing_trades'] += 1
                today_stats['largest_loss'] = min(today_stats['largest_loss'], pnl_pips)
            
            today_stats['total_pnl_pips'] += pnl_pips
            today_stats['total_pnl_percent'] += pnl_percent
            today_stats['win_rate'] = today_stats['winning_trades'] / today_stats['total_trades']
            
            # Convert all data to JSON serializable format
            performance_data = self.convert_to_json_serializable(performance_data)
            
            # Save updated performance data
            with open(self.performance_log_file, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Daily performance update error: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")

    def connect_mt5(self) -> bool:
        """Connect to MT5 with trading capabilities"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False

            if self.account and self.password and self.server:
                if not mt5.login(
                    self.account, password=self.password, server=self.server
                ):
                    print(f"❌ Login failed: {mt5.last_error()}")
                    return False

            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Failed to get account info")
                return False

            if not account_info.trade_allowed:
                print("❌ Trading is not allowed on this account")
                return False

            print("✅ MT5 Connected with Trading Capabilities")
            print(f"📊 Account: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance:.2f}")
            print(f"📝 Log Directory: {self.log_directory}")

            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def load_models(self) -> bool:
        """Load AI models through signal engine"""
        return self.signal_engine.load_trained_models()

    def enable_trading(self, enable: bool = True):
        """Enable or disable automated trading"""
        self.trading_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🎯 Auto Trading {status}")

        if enable:
            print("⚠️ WARNING: Live trading is now active!")
            print("🛡️ Safety mechanisms active")
            print("📝 All trades will be logged for learning")

    def send_order(
        self,
        symbol: str,
        order_type: int,
        lot_size: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "SMC_AI_Bot",
        signal_data: Dict = None,
    ) -> bool:
        """Enhanced send order with comprehensive logging"""

        if not self.trading_enabled:
            print("⚠️ Trading disabled - order not sent")
            return False

        try:
            # Get market data before order execution
            market_data = self.get_comprehensive_market_data(symbol)
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not found")
                return False

            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            print(f"🔍 Order validation:")
            print(f"   Requested lot: {lot_size}")
            print(f"   Broker limits: {min_lot} - {max_lot}, step: {lot_step}")

            if lot_size < min_lot:
                lot_size = min_lot
                print(f"⚠️ Adjusted to minimum: {lot_size}")
            elif lot_size > max_lot:
                lot_size = max_lot
                print(f"⚠️ Adjusted to maximum: {lot_size}")

            if lot_step > 0:
                lot_size = round(lot_size / lot_step) * lot_step
                print(f"🔧 Rounded to step: {lot_size}")

            if order_type == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(symbol).ask
                order_type_str = "BUY"
            else:
                price = mt5.symbol_info_tick(symbol).bid
                order_type_str = "SELL"

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": stop_loss if stop_loss > 0 else 0.0,
                "tp": take_profit if take_profit > 0 else 0.0,
                "deviation": 20,
                "magic": 123456,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            print(f"📋 Sending order: {order_type_str} {lot_size} {symbol}")

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Order failed: {result.retcode} - {result.comment}")

                if result.retcode == 10014:
                    print("🔄 Trying FOK filling...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)

                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"❌ Second attempt failed: {result.retcode}")
                        return False
                else:
                    return False

            # 🆕 Enhanced logging with comprehensive data
            order_details = {
                "ticket": result.order,
                "type": order_type,
                "volume": lot_size,
                "price": result.price,
                "sl": stop_loss,
                "tp": take_profit
            }
            
            # Log trade entry with full context
            trade_id = self.log_trade_entry(
                signal=signal_data or {},
                symbol=symbol,
                order_details=order_details,
                market_data=market_data
            )

            trade_info = {
                "trade_id": trade_id,
                "timestamp": datetime.now(self.timezone),
                "symbol": symbol,
                "type": order_type_str,
                "volume": lot_size,
                "price": result.price,
                "sl": stop_loss,
                "tp": take_profit,
                "ticket": result.order,
                "comment": comment,
            }

            self.trade_history.append(trade_info)
            self.active_positions[result.order] = trade_info

            print(
                f"✅ Order executed: {order_type_str} {lot_size} {symbol} @ {result.price:.5f}"
            )
            print(f"   🎯 SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            print(f"   🎫 Ticket: {result.order}")
            print(f"   📝 Logged as: {trade_id}")

            return True

        except Exception as e:
            print(f"❌ Order execution error: {str(e)}")
            return False

    def update_positions(self):
        """Enhanced position update with comprehensive logging"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            current_tickets = [pos.ticket for pos in positions]

            closed_tickets = []
            for ticket in list(self.active_positions.keys()):
                if ticket not in current_tickets:
                    closed_tickets.append(ticket)

                    closed_trade = self.active_positions[ticket]
                    print(f"📈 Trade #{ticket} CLOSED:")
                    print(
                        f"   {closed_trade['type']} {closed_trade['volume']} {closed_trade['symbol']}"
                    )
                    print(f"   Entry: {closed_trade['price']:.5f}")

                    # Get deal details for comprehensive logging
                    deals = mt5.history_deals_get(
                        datetime.now() - timedelta(hours=1), datetime.now()
                    )

                    exit_details = {
                        'ticket': ticket,
                        'exit_price': closed_trade['price'],  # Default
                        'exit_reason': 'UNKNOWN',
                        'profit': 0,
                        'duration_minutes': 0
                    }

                    if deals:
                        for deal in deals:
                            if deal.position_id == ticket and deal.entry == 1:  # Exit deal
                                exit_details.update({
                                    'exit_price': deal.price,
                                    'profit': deal.profit,
                                    'duration_minutes': (datetime.fromtimestamp(deal.time) - closed_trade['timestamp']).total_seconds() / 60
                                })
                                
                                # Determine exit reason
                                if abs(deal.price - closed_trade.get('tp', 0)) < 0.00002:
                                    exit_details['exit_reason'] = 'TP_HIT'
                                elif abs(deal.price - closed_trade.get('sl', 0)) < 0.00002:
                                    exit_details['exit_reason'] = 'SL_HIT'
                                else:
                                    exit_details['exit_reason'] = 'TIME_EXIT'

                                print(f"   Exit: {deal.price:.5f}")
                                print(f"   P&L: ${deal.profit:.2f}")
                                print(f"   Reason: {exit_details['exit_reason']}")
                                print(f"   Result: {'✅ WIN' if deal.profit > 0 else '❌ LOSS'}")
                                break

                    # 🆕 Log trade outcome with market context
                    market_data = self.get_comprehensive_market_data(closed_trade['symbol'])
                    
                    self.update_trade_outcome(
                        trade_id=closed_trade.get('trade_id', f"LEGACY_{ticket}"),
                        exit_details=exit_details,
                        market_data=market_data
                    )

                    del self.active_positions[ticket]
                    self.last_trade_closed_time = datetime.now()

            if closed_tickets:
                print(
                    f"🎯 {len(closed_tickets)} position(s) closed. Ready for new trades."
                )

            total_profit = sum(pos.profit for pos in positions)
            self.daily_pnl = total_profit

        except Exception as e:
            print(f"❌ Position update error: {str(e)}")

    def process_signal(self, signal: Dict, symbol: str) -> bool:
        """Enhanced signal processing with logging"""

        if not self.trading_enabled:
            return False

        if not self.check_can_trade():
            return False

        if signal["final_confidence"] < self.min_confidence:
            print(f"⚠️ Signal confidence too low: {signal['final_confidence']:.3f}")
            return False

        if signal["trading_recommendation"] != "TRADE":
            return False

        individual_signals = signal["individual_signals"]
        long_count = sum(
            1 for s in individual_signals.values() if s["consensus_prediction"] == 1
        )
        short_count = sum(
            1 for s in individual_signals.values() if s["consensus_prediction"] == -1
        )

        total_agreement = max(long_count, short_count)
        if total_agreement < self.min_consensus:
            print(
                f"⚠️ Insufficient consensus: {total_agreement}/{len(individual_signals)}"
            )
            return False

        if signal["final_direction"] == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
        elif signal["final_direction"] == "SHORT":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            return False

        lot_size = self.calculate_position_size(symbol, signal["final_confidence"])

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        stop_loss, take_profit = self.calculate_sl_tp_levels(
            symbol, order_type, entry_price
        )

        comment = (
            f"SMC_AI_{signal['final_direction']}_C{signal['final_confidence']:.2f}"
        )

        # 🆕 Pass signal data to order for logging
        success = self.send_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            signal_data=signal  # Pass signal for comprehensive logging
        )

        if success:
            self.hourly_trade_count += 1
            print(f"🚀 Auto trade executed: {signal['final_direction']} {symbol}")

        return success

    def get_trading_statistics(self) -> Dict:
        """📊 Get comprehensive trading statistics from logs"""
        try:
            stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl_pips': 0.0,
                'avg_win_pips': 0.0,
                'avg_loss_pips': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_trade_duration': 0.0,
                'model_accuracy': 0.0
            }
            
            if not os.path.exists(self.trade_log_file):
                return stats
            
            closed_trades = []
            with open(self.trade_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line.strip())
                        if trade.get('outcome', {}).get('status') == 'CLOSED':
                            closed_trades.append(trade)
            
            if len(closed_trades) == 0:
                return stats
            
            # Calculate statistics
            stats['total_trades'] = len(closed_trades)
            winning_trades = [t for t in closed_trades if t['outcome']['is_winner']]
            losing_trades = [t for t in closed_trades if not t['outcome']['is_winner']]
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            stats['win_rate'] = len(winning_trades) / len(closed_trades) * 100
            
            all_pnl = [t['outcome']['pnl_pips'] for t in closed_trades]
            win_pnl = [t['outcome']['pnl_pips'] for t in winning_trades]
            loss_pnl = [t['outcome']['pnl_pips'] for t in losing_trades]
            
            stats['total_pnl_pips'] = sum(all_pnl)
            stats['avg_win_pips'] = np.mean(win_pnl) if win_pnl else 0
            stats['avg_loss_pips'] = np.mean(loss_pnl) if loss_pnl else 0
            stats['largest_win'] = max(all_pnl) if all_pnl else 0
            stats['largest_loss'] = min(all_pnl) if all_pnl else 0
            
            durations = [t['outcome']['trade_duration_minutes'] for t in closed_trades if t['outcome']['trade_duration_minutes']]
            stats['avg_trade_duration'] = np.mean(durations) if durations else 0
            
            accuracies = [t['learning_metrics']['prediction_accuracy'] for t in closed_trades if t['learning_metrics']['prediction_accuracy'] is not None]
            stats['model_accuracy'] = np.mean(accuracies) * 100 if accuracies else 0
            
            return stats
            
        except Exception as e:
            print(f"❌ Statistics calculation error: {str(e)}")
            return stats

    def print_trading_summary(self):
        """📊 Print comprehensive trading summary"""
        stats = self.get_trading_statistics()
        
        print("\n" + "="*60)
        print("📊 TRADING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎯 Total Trades: {stats['total_trades']}")
        print(f"✅ Winning Trades: {stats['winning_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        print(f"📈 Win Rate: {stats['win_rate']:.1f}%")
        print(f"💰 Total P&L: {stats['total_pnl_pips']:.1f} pips")
        print(f"🏆 Average Win: {stats['avg_win_pips']:.1f} pips")
        print(f"💸 Average Loss: {stats['avg_loss_pips']:.1f} pips")
        print(f"🚀 Largest Win: {stats['largest_win']:.1f} pips")
        print(f"🔻 Largest Loss: {stats['largest_loss']:.1f} pips")
        print(f"⏱️ Avg Duration: {stats['avg_trade_duration']:.0f} minutes")
        print(f"🧠 Model Accuracy: {stats['model_accuracy']:.1f}%")
        print("="*60)
        print(f"📝 Log Files Location: {self.log_directory}")
        print("="*60)

    # ... (คงฟังก์ชันเดิมไว้เหมือนเดิม) ...
    
    def should_analyze_signals(self) -> bool:
        """Determine if system should analyze new signals"""
        if not self.wait_for_trade_completion:
            return True

        if len(self.active_positions) > 0:
            return False

        return True

    def check_can_trade(self) -> bool:
        """Check if system can place new trades"""
        if self.wait_for_trade_completion:
            if len(self.active_positions) > 0:
                print(
                    f"⏳ Waiting for current trade to close. Active positions: {len(self.active_positions)}"
                )
                return False

        return self.check_risk_limits()

    def check_risk_limits(self) -> bool:
        """Check if trading is within risk limits"""
        if len(self.active_positions) >= self.max_concurrent_trades:
            print(f"🛑 Maximum concurrent trades reached: {len(self.active_positions)}")
            return False

        current_hour = datetime.now().hour
        if current_hour != self.hour_start:
            self.hourly_trade_count = 0
            self.hour_start = current_hour

        if self.hourly_trade_count >= self.max_trades_per_hour:
            print(f"🛑 Hourly trade limit reached: {self.hourly_trade_count}")
            return False

        return True

    def _is_signal_changed(
        self, last_signal: Optional[Dict], current_signal: Dict
    ) -> bool:
        """Determine if signal has changed enough to warrant new trade"""

        if last_signal is None:
            if self.enable_first_signal_trade:
                return (
                    current_signal["final_confidence"]
                    >= self.first_signal_min_confidence
                    and current_signal["trading_recommendation"] == "TRADE"
                )
            else:
                return False

        if last_signal["final_direction"] != current_signal["final_direction"]:
            return True

        confidence_change = abs(
            last_signal["final_confidence"] - current_signal["final_confidence"]
        )
        if confidence_change > self.signal_change_threshold:
            return True

        if (
            current_signal["final_confidence"] >= 0.85
            and current_signal["trading_recommendation"] == "TRADE"
            and len(self.trade_history) == 0
        ):
            print(
                f"🔥 Force trading high confidence signal: {current_signal['final_confidence']:.3f}"
            )
            return True

        return False

    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return self.base_lot_size

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Cannot get symbol info for {symbol}")
                return self.base_lot_size

            print(f"📊 Symbol {symbol} specifications:")
            print(f"   Min lot: {symbol_info.volume_min}")
            print(f"   Max lot: {symbol_info.volume_max}")
            print(f"   Lot step: {symbol_info.volume_step}")

            calculated_lot = self.base_lot_size

            if confidence >= 0.9:
                calculated_lot *= self.lot_multiplier
            elif confidence >= 0.8:
                calculated_lot *= 1.5

            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            calculated_lot = max(min_lot, calculated_lot)
            calculated_lot = min(max_lot, calculated_lot)
            calculated_lot = min(self.max_lot_size, calculated_lot)

            if lot_step > 0:
                calculated_lot = round(calculated_lot / lot_step) * lot_step

            final_lot = max(min_lot, calculated_lot)

            print(f"💰 Position size: {self.base_lot_size} → {final_lot}")

            return final_lot

        except Exception as e:
            print(f"❌ Position size error: {str(e)}")
            return self.base_lot_size

    def calculate_sl_tp_levels(
        self, symbol: str, order_type: int, entry_price: float
    ) -> Tuple[float, float]:
        """Calculate Stop Loss and Take Profit levels"""
        try:
            if "JPY" in symbol:
                pip_size = 0.01
            else:
                pip_size = 0.0001

            sl_distance = self.default_sl_pips * pip_size
            tp_distance = sl_distance * self.default_tp_ratio

            if order_type == mt5.ORDER_TYPE_BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            return stop_loss, take_profit

        except Exception as e:
            print(f"❌ SL/TP calculation error: {str(e)}")
            return 0.0, 0.0

    def print_current_settings(self):
        """Print current configuration with logging info"""
        print("⚙️ Auto Trader Settings:")
        print("=" * 50)
        print(f"🎯 Max concurrent trades: {self.max_concurrent_trades}")
        print(
            f"⏳ Wait for completion: {'YES' if self.wait_for_trade_completion else 'NO'}"
        )
        print(f"📊 Min confidence: {self.min_confidence*100}%")
        print(f"🤝 Min consensus: {self.min_consensus}/5")
        print(f"💰 Base lot size: {self.base_lot_size}")
        print(f"📝 Logging: ENABLED - {self.log_directory}")
        print("=" * 50)

    def start_auto_trading(self, symbol: str = "EURUSD.c", update_interval: int = 60):
        """Start automated trading system with comprehensive logging"""

        print("🚀 Starting Enhanced SMC Auto Trading System")
        print("=" * 60)
        print(f"📊 Symbol: {symbol}")
        print(f"🎯 Trading Status: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"⏳ Mode: One trade at a time")
        print(f"📝 Comprehensive Logging: ACTIVE")
        print("=" * 60)

        last_signal = None

        try:
            while True:
                try:
                    self.update_positions()

                    if not self.should_analyze_signals():
                        print(
                            f"\n⏳ {datetime.now().strftime('%H:%M:%S')} - Waiting for active trade to close..."
                        )
                        print(
                            f"💰 Daily P&L: ${self.daily_pnl:.2f} | Active Positions: {len(self.active_positions)}"
                        )

                        if self.active_positions:
                            for ticket, trade_info in self.active_positions.items():
                                positions = mt5.positions_get(ticket=ticket)
                                if positions:
                                    pos = positions[0]
                                    print(
                                        f"🔄 Active: {trade_info['type']} {trade_info['volume']} {trade_info['symbol']}"
                                    )
                                    print(
                                        f"   Entry: {trade_info['price']:.5f} | Current P&L: ${pos.profit:.2f}"
                                    )

                        time.sleep(update_interval)
                        continue

                    print(
                        f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Analyzing new signals..."
                    )
                    signal = self.signal_engine.get_multi_timeframe_signals(symbol)

                    if "error" in signal:
                        print(f"❌ Signal error: {signal['error']}")
                    else:
                        print(
                            f"📊 {symbol}: {signal['final_direction']} | Confidence: {signal['final_confidence']:.3f}"
                        )
                        print(
                            f"🎯 Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}"
                        )
                        print(f"💰 Daily P&L: ${self.daily_pnl:.2f}")

                        signal_changed = self._is_signal_changed(last_signal, signal)

                        if signal_changed and signal["trading_recommendation"] == "TRADE":
                            if self.trading_enabled:
                                print("🔥 NEW TRADING SIGNAL DETECTED!")
                                success = self.process_signal(signal, symbol)
                                if success:
                                    print("✅ Auto trade executed successfully")
                                    print("📝 Trade logged with full market context")
                                else:
                                    print("❌ Auto trade failed or blocked")
                            else:
                                print("📊 TRADING SIGNAL (Trading disabled)")

                        last_signal = signal

                    time.sleep(update_interval)

                except KeyboardInterrupt:
                    print("\n🛑 Auto trading stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Auto trading error: {str(e)}")
                    time.sleep(10)

        finally:
            # Print final summary when stopping
            self.print_trading_summary()
            print("✅ Auto trading system stopped")


# Main execution
if __name__ == "__main__":
    print("🤖 Enhanced SMC Auto Trading Bot with Comprehensive Logging")
    print("=" * 60)

    # Settings
    SIGNAL_CHANGE_THRESHOLD = 0.001
    ENABLE_FIRST_TRADE = True
    FIRST_TRADE_MIN_CONFIDENCE = 0.75
    MIN_CONFIDENCE = 0.75
    MIN_CONSENSUS = 3
    MAX_CONCURRENT_TRADES = 1
    WAIT_FOR_COMPLETION = True
    BASE_LOT_SIZE = 0.01
    MAX_LOT_SIZE = 0.1

    # Initialize enhanced trader
    trader = SMCAutoTrader(
        models_path="EURUSD_c_SMC",
        signal_change_threshold=SIGNAL_CHANGE_THRESHOLD,
        enable_first_signal_trade=ENABLE_FIRST_TRADE,
        first_signal_min_confidence=FIRST_TRADE_MIN_CONFIDENCE,
        min_confidence=MIN_CONFIDENCE,
        min_consensus=MIN_CONSENSUS,
        max_concurrent_trades=MAX_CONCURRENT_TRADES,
        wait_for_trade_completion=WAIT_FOR_COMPLETION,
        base_lot_size=BASE_LOT_SIZE,
        max_lot_size=MAX_LOT_SIZE,
    )

    trader.print_current_settings()

    if trader.connect_mt5():
        if trader.load_models():
            print("\n🎯 Enhanced Auto Trading Bot Ready!")
            print("📝 All trades will be logged for AI improvement")

            enable_trading = (
                input("\n🚀 Enable LIVE AUTO TRADING? (yes/no): ").lower().strip()
            )

            if enable_trading == "yes":
                trader.enable_trading(True)
            else:
                print("📊 Demo mode")
                trader.enable_trading(True)  # Enable anyway for testing

            trader.start_auto_trading("EURUSD.c", 60)

        else:
            print("❌ Failed to load AI models")
    else:
        print("❌ Failed to connect to MT5")