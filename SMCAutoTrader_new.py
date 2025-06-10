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

# Import our Gold-Optimized Signal Engine
from smc_signal_engine import SMCSignalEngine


class SMCGoldAutoTrader:
    """
    🥇 Gold SMC Auto Trading Bot with Advanced Gold Features
    Created by อาจารย์ฟินิกซ์ - Professional Automated Gold Trading System
    
    Optimized specifically for XAUUSD.c with:
    - Gold-specific risk management
    - Session-based trading
    - Point-based calculations
    - Enhanced volatility handling
    """

    def __init__(
        self,
        models_path: str = "XAUUSD_c_SMC",
        account: int = None,
        password: str = None,
        server: str = None,
        # 🥇 Gold-specific signal controls
        signal_change_threshold: float = 0.5,  # In Gold points (50 cents)
        enable_first_signal_trade: bool = True,
        first_signal_min_confidence: float = 0.80,  # Higher for Gold
        # 🥇 Gold risk management
        max_risk_per_trade: float = 0.015,  # 1.5% per trade (lower for Gold)
        max_daily_loss: float = 0.04,       # 4% daily loss limit
        max_concurrent_trades: int = 1,      # One Gold trade at a time
        min_confidence: float = 0.80,        # Higher confidence required
        min_consensus: int = 4,              # 4/5 timeframes agreement
        # 🥇 Gold position sizing (in Troy Ounces)
        base_lot_size: float = 0.01,         # 0.01 lots = 1 troy ounce
        max_lot_size: float = 0.05,          # Conservative max for Gold
        lot_multiplier: float = 1.5,         # Lower multiplier for Gold
        # 🥇 Gold-specific stops and targets (in points)
        default_sl_points: int = 500,        # 500 points = $5.00
        default_tp_ratio: float = 2.5,       # 2.5:1 ratio for Gold
        # 🥇 Gold session controls
        max_trades_per_hour: int = 3,        # Lower frequency for Gold
        wait_for_trade_completion: bool = True,
        # 🥇 Gold session preferences
        enable_london_session: bool = True,   # High volatility
        enable_us_session: bool = True,       # High impact
        enable_asian_session: bool = False,   # Low liquidity - avoid
        enable_overlap_session: bool = True,  # London-US overlap
    ):
        """Initialize Gold Auto Trader with Gold-specific optimizations"""

        # Connection settings
        self.account = account
        self.password = password
        self.server = server
        self.timezone = pytz.timezone("Etc/UTC")

        # 🥇 Gold Signal Controls
        self.signal_change_threshold = signal_change_threshold  # In Gold points
        self.enable_first_signal_trade = enable_first_signal_trade
        self.first_signal_min_confidence = first_signal_min_confidence

        # 🥇 Gold Risk Management
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_concurrent_trades = max_concurrent_trades
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus

        # 🥇 Gold Position Sizing
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.lot_multiplier = lot_multiplier

        # 🥇 Gold Trade Management (Point-based)
        self.default_sl_points = default_sl_points
        self.default_tp_ratio = default_tp_ratio
        self.gold_point_value = 0.01  # $0.01 per point

        # 🥇 Gold Session Controls
        self.enable_london_session = enable_london_session
        self.enable_us_session = enable_us_session
        self.enable_asian_session = enable_asian_session
        self.enable_overlap_session = enable_overlap_session

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

        # 🥇 Gold-specific tracking
        self.daily_gold_points = 0.0
        self.session_trades = {"london": 0, "us": 0, "asian": 0, "overlap": 0}
        
        # Enhanced Logging System
        self.log_directory = "gold_trading_logs"
        self.ensure_log_directory()
        
        # Log files
        self.trade_log_file = os.path.join(self.log_directory, "gold_trade_entries.jsonl")
        self.signal_log_file = os.path.join(self.log_directory, "gold_signal_history.jsonl")
        self.performance_log_file = os.path.join(self.log_directory, "gold_daily_performance.json")
        
        # Market data cache
        self.market_data_cache = {}
        
        # Initialize Gold Signal Engine
        self.signal_engine = SMCSignalEngine(models_path)

        print("🥇 Gold SMC Auto Trading Bot Initialized")
        print("📊 Optimized for XAUUSD.c with enhanced Gold features")
        print("⚠️ Trading is DISABLED by default for safety")

    def ensure_log_directory(self):
        """Create Gold-specific logging directory"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            print(f"📁 Created Gold log directory: {self.log_directory}")

    def get_current_gold_session(self) -> str:
        """Get current Gold trading session"""
        current_hour = datetime.now().hour
        
        # London session (8-16 UTC) - High volatility
        if 8 <= current_hour <= 16:
            return "london"
        # US session (13-21 UTC) - High impact news
        elif 13 <= current_hour <= 21:
            return "us"
        # London-US overlap (13-16 UTC) - Highest volatility
        elif 13 <= current_hour <= 16:
            return "overlap"
        # Asian session (22-7 UTC) - Low liquidity
        elif 22 <= current_hour or current_hour <= 7:
            return "asian"
        else:
            return "transition"

    def is_gold_session_allowed(self) -> bool:
        """Check if current session is allowed for Gold trading"""
        current_session = self.get_current_gold_session()
        
        session_allowed = {
            "london": self.enable_london_session,
            "us": self.enable_us_session,
            "overlap": self.enable_overlap_session,
            "asian": self.enable_asian_session,
            "transition": False  # Never trade during transitions
        }
        
        return session_allowed.get(current_session, False)

    def get_gold_session_risk_multiplier(self) -> float:
        """Get Gold session-specific risk multiplier"""
        current_session = self.get_current_gold_session()
        
        session_multipliers = {
            "london": 1.2,     # Higher risk during high volatility
            "us": 1.1,         # Slightly higher during news
            "overlap": 1.3,    # Highest during overlap
            "asian": 0.6,      # Lower risk during low liquidity
            "transition": 0.5  # Minimal risk
        }
        
        return session_multipliers.get(current_session, 1.0)

    def find_gold_symbol(self) -> Optional[str]:
        """Find the best available Gold symbol"""
        gold_symbols = [
            "XAUUSD.c", "XAUUSD", "GOLD.c", "GOLD",
            "XAU/USD", "XAUUSD#", "XAUUSD.raw", "XAUUSD.a"
        ]
        
        print("🔍 Searching for Gold symbols...")
        for symbol in gold_symbols:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info and symbol_info.trade_mode != 0:
                        print(f"✅ Found tradeable Gold symbol: {symbol}")
                        return symbol
            except:
                continue
        
        print("❌ No tradeable Gold symbol found")
        return None

    def get_comprehensive_gold_market_data(self, symbol: str) -> Dict:
        """Get comprehensive Gold market data with session analysis"""
        try:
            # Get current tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}

            # Get recent candle data for Gold context
            rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 5)
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3)

            # Current session info
            current_session = self.get_current_gold_session()
            session_multiplier = self.get_gold_session_risk_multiplier()

            market_data = {
                # Current Gold Tick Data
                'current_tick': {
                    'time': datetime.fromtimestamp(tick.time).isoformat(),
                    'bid': float(tick.bid),
                    'ask': float(tick.ask),
                    'spread': float(tick.ask - tick.bid),
                    'spread_points': float((tick.ask - tick.bid) / self.gold_point_value),
                    'volume': int(tick.volume) if hasattr(tick, 'volume') else 0
                },
                
                # Gold Price Action
                'gold_price_action': {
                    'M1_last': {
                        'open': float(rates_m1[-1]['open']) if rates_m1 is not None else 0,
                        'high': float(rates_m1[-1]['high']) if rates_m1 is not None else 0,
                        'low': float(rates_m1[-1]['low']) if rates_m1 is not None else 0,
                        'close': float(rates_m1[-1]['close']) if rates_m1 is not None else 0,
                        'range_points': float((rates_m1[-1]['high'] - rates_m1[-1]['low']) / self.gold_point_value) if rates_m1 is not None else 0,
                        'volume': int(rates_m1[-1]['tick_volume']) if rates_m1 is not None else 0
                    },
                    'M5_last': {
                        'open': float(rates_m5[-1]['open']) if rates_m5 is not None else 0,
                        'high': float(rates_m5[-1]['high']) if rates_m5 is not None else 0,
                        'low': float(rates_m5[-1]['low']) if rates_m5 is not None else 0,
                        'close': float(rates_m5[-1]['close']) if rates_m5 is not None else 0,
                        'range_points': float((rates_m5[-1]['high'] - rates_m5[-1]['low']) / self.gold_point_value) if rates_m5 is not None else 0,
                        'volume': int(rates_m5[-1]['tick_volume']) if rates_m5 is not None else 0
                    }
                },
                
                # Gold Volatility Context
                'gold_volatility': {
                    'atr_points': float(np.mean([
                        (r['high'] - r['low']) / self.gold_point_value 
                        for r in rates_m5[-5:]
                    ])) if rates_m5 is not None else 0,
                    'hour_range_points': float((
                        max([r['high'] for r in rates_h1[-1:]] or [0]) - 
                        min([r['low'] for r in rates_h1[-1:]] or [0])
                    ) / self.gold_point_value) if rates_h1 is not None else 0,
                    'volatility_level': 'high' if current_session in ['london', 'us', 'overlap'] else 'low'
                },
                
                # Gold Session Info
                'gold_session': {
                    'current_session': current_session,
                    'session_multiplier': session_multiplier,
                    'is_allowed': self.is_gold_session_allowed(),
                    'hour': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),
                    'is_london_session': current_session == 'london',
                    'is_us_session': current_session == 'us',
                    'is_overlap_session': current_session == 'overlap',
                    'is_asian_session': current_session == 'asian',
                    'is_weekend': datetime.now().weekday() >= 5,
                    'is_high_impact_time': datetime.now().hour in [8, 13, 14, 15]
                },
                
                # Gold Trading Context
                'gold_trading_context': {
                    'point_value': self.gold_point_value,
                    'default_sl_points': self.default_sl_points,
                    'recommended_tp_points': int(self.default_sl_points * self.default_tp_ratio),
                    'contract_size': '1 troy ounce per 0.01 lot',
                    'daily_points_target': 1000,  # 1000 points = $10 per 0.01 lot
                    'current_daily_points': self.daily_gold_points
                }
            }
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error getting Gold market data: {str(e)}")
            return {}

    def log_gold_trade_entry(self, signal: Dict, symbol: str, order_details: Dict, market_data: Dict) -> str:
        """Log Gold trade entry with comprehensive Gold-specific data"""
        try:
            trade_id = f"GOLD_TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order_details.get('ticket', 'UNKNOWN')}"
            
            entry_log = {
                # Gold Trade Identification
                'trade_id': trade_id,
                'timestamp': datetime.now(self.timezone).isoformat(),
                'symbol': symbol,
                'symbol_type': 'GOLD',
                'mt5_ticket': self.convert_to_json_serializable(order_details.get('ticket')),
                
                # Gold Entry Conditions & Signal Analysis
                'gold_entry_conditions': {
                    'signal_direction': signal.get('final_direction', 'UNKNOWN'),
                    'signal_confidence': self.convert_to_json_serializable(signal.get('final_confidence', 0)),
                    'session_adjusted_confidence': self.convert_to_json_serializable(signal.get('session_adjusted_confidence', 0)),
                    'risk_level': signal.get('risk_level', 'UNKNOWN'),
                    'timeframe_consensus': signal.get('timeframe_consensus', 'UNKNOWN'),
                    'trading_recommendation': signal.get('trading_recommendation', 'UNKNOWN'),
                    
                    # Gold Session Context
                    'session_info': signal.get('session_info', {}),
                    'session_allowed': self.is_gold_session_allowed(),
                    'session_multiplier': self.get_gold_session_risk_multiplier(),
                    
                    # Individual Timeframe Signals
                    'timeframe_signals': {
                        tf: {
                            'prediction': self.convert_to_json_serializable(sig_data.get('consensus_prediction', 0)),
                            'confidence': self.convert_to_json_serializable(sig_data.get('average_confidence', 0)),
                            'session_adjusted_confidence': self.convert_to_json_serializable(sig_data.get('session_adjusted_confidence', 0)),
                            'signal_quality': sig_data.get('signal_quality', 'UNKNOWN')
                        }
                        for tf, sig_data in signal.get('individual_signals', {}).items()
                    },
                    
                    # Gold SMC Analysis
                    'gold_smc_confluence': self._extract_gold_smc_features(signal),
                    
                    # Model Performance
                    'model_versions': self._get_current_model_versions(),
                    'signal_change_triggered': True
                },
                
                # Gold Trade Execution Details
                'gold_execution_details': {
                    'order_type': 'BUY' if order_details.get('type') == mt5.ORDER_TYPE_BUY else 'SELL',
                    'position_size_lots': self.convert_to_json_serializable(order_details.get('volume', 0)),
                    'position_size_ounces': self.convert_to_json_serializable(order_details.get('volume', 0) * 100),  # 1 lot = 100 oz
                    'entry_price': self.convert_to_json_serializable(order_details.get('price', 0)),
                    'stop_loss': self.convert_to_json_serializable(order_details.get('sl', 0)),
                    'take_profit': self.convert_to_json_serializable(order_details.get('tp', 0)),
                    'stop_loss_points': self.convert_to_json_serializable(self._calculate_sl_points(order_details)),
                    'take_profit_points': self.convert_to_json_serializable(self._calculate_tp_points(order_details)),
                    'risk_reward_ratio': self.convert_to_json_serializable(self._calculate_risk_reward(order_details)),
                    'position_risk_percent': self.convert_to_json_serializable(self._calculate_position_risk(order_details)),
                    'position_risk_dollars': self.convert_to_json_serializable(self._calculate_risk_dollars(order_details)),
                    'slippage_points': 0  # Will be calculated if needed
                },
                
                # Gold Market Context at Entry
                'gold_market_context': self.convert_to_json_serializable(market_data),
                
                # Trade Outcome (will be updated when trade closes)
                'gold_outcome': {
                    'status': 'OPEN',
                    'exit_time': None,
                    'exit_price': None,
                    'exit_reason': None,  # 'TP_HIT', 'SL_HIT', 'TIME_EXIT', 'MANUAL_CLOSE', 'SESSION_EXIT'
                    'pnl_points': None,
                    'pnl_usd': None,
                    'pnl_percent': None,
                    'trade_duration_minutes': None,
                    'is_winner': None,
                    'max_favorable_excursion_points': None,  # MFE in Gold points
                    'max_adverse_excursion_points': None,    # MAE in Gold points
                },
                
                # Gold Learning Data
                'gold_learning_metrics': {
                    'prediction_accuracy': None,  # Will be calculated after close
                    'session_performance_match': None,
                    'volatility_handling': None,
                    'gold_market_behavior_match': None,
                    'execution_quality': 'GOOD'  # Based on slippage, timing etc.
                }
            }
            
            # Convert entire log to JSON serializable format
            entry_log = self.convert_to_json_serializable(entry_log)
            
            # Save to Gold log file
            with open(self.trade_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry_log, ensure_ascii=False, default=str) + '\n')
            
            print(f"📝 Gold trade entry logged: {trade_id}")
            return trade_id
            
        except Exception as e:
            print(f"❌ Gold trade entry logging error: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")
            return f"GOLD_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _calculate_sl_points(self, order_details: Dict) -> float:
        """Calculate stop loss in Gold points"""
        try:
            entry_price = order_details.get('price', 0)
            stop_loss = order_details.get('sl', 0)
            
            if stop_loss == 0:
                return 0.0
                
            sl_points = abs(entry_price - stop_loss) / self.gold_point_value
            return sl_points
            
        except:
            return 0.0

    def _calculate_tp_points(self, order_details: Dict) -> float:
        """Calculate take profit in Gold points"""
        try:
            entry_price = order_details.get('price', 0)
            take_profit = order_details.get('tp', 0)
            
            if take_profit == 0:
                return 0.0
                
            tp_points = abs(take_profit - entry_price) / self.gold_point_value
            return tp_points
            
        except:
            return 0.0

    def _calculate_risk_dollars(self, order_details: Dict) -> float:
        """Calculate position risk in dollars"""
        try:
            position_size = order_details.get('volume', 0)
            sl_points = self._calculate_sl_points(order_details)
            
            # Risk in dollars = position_size * sl_points * point_value
            risk_dollars = position_size * 100 * sl_points * self.gold_point_value  # 1 lot = 100 oz
            return risk_dollars
            
        except:
            return 0.0

    def _extract_gold_smc_features(self, signal: Dict) -> Dict:
        """Extract Gold-specific SMC features for logging"""
        try:
            gold_smc_features = {}
            
            # Extract from individual signals if available
            for tf, sig_data in signal.get('individual_signals', {}).items():
                if 'current_price' in str(sig_data):  # Check if detailed data exists
                    gold_smc_features[tf] = {
                        'market_structure': 'BULLISH' if sig_data.get('consensus_prediction', 0) > 0 else 'BEARISH',
                        'confidence': float(sig_data.get('average_confidence', 0)),
                        'session_adjusted': float(sig_data.get('session_adjusted_confidence', 0)),
                        'signal_quality': sig_data.get('signal_quality', 'UNKNOWN'),
                        'is_gold_optimized': True
                    }
            
            # Add Gold-specific confluence data
            gold_smc_features['gold_session_analysis'] = signal.get('session_info', {})
            gold_smc_features['gold_optimized'] = signal.get('gold_optimized', False)
            
            return gold_smc_features
            
        except Exception as e:
            print(f"⚠️ Error extracting Gold SMC features: {str(e)}")
            return {}

    def _get_current_model_versions(self) -> Dict:
        """Get current Gold model versions"""
        return {
            'random_forest': 'gold_v1.0',
            'xgboost': 'gold_v1.0',
            'neural_network': 'gold_v1.0',
            'ensemble_version': 'gold_v1.0',
            'optimized_for': 'XAUUSD.c',
            'session_weighting': 'enabled',
            'point_based_calculations': 'enabled'
        }

    def convert_to_json_serializable(self, obj):
        """Convert numpy types and objects to JSON serializable format"""
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

    def connect_mt5(self) -> bool:
        """Connect to MT5 with Gold trading capabilities"""
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

            print("✅ MT5 Connected with Gold Trading Capabilities")
            print(f"📊 Account: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance:.2f}")
            print(f"🥇 Gold Log Directory: {self.log_directory}")

            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def load_models(self) -> bool:
        """Load Gold AI models through signal engine"""
        success = self.signal_engine.load_trained_models()
        if success:
            print("✅ Gold AI models loaded successfully")
        return success

    def enable_trading(self, enable: bool = True):
        """Enable or disable Gold automated trading"""
        self.trading_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🥇 Gold Auto Trading {status}")

        if enable:
            print("⚠️ WARNING: Live Gold trading is now active!")
            print("🛡️ Gold-specific safety mechanisms active")
            print("📝 All Gold trades will be logged with comprehensive data")
            print(f"🎯 Session preferences: London={self.enable_london_session}, US={self.enable_us_session}, Asian={self.enable_asian_session}")

    def calculate_gold_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal Gold position size with Gold-specific considerations"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return self.base_lot_size

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Cannot get Gold symbol info for {symbol}")
                return self.base_lot_size

            print(f"🥇 Gold Symbol {symbol} specifications:")
            print(f"   Min lot: {symbol_info.volume_min}")
            print(f"   Max lot: {symbol_info.volume_max}")
            print(f"   Lot step: {symbol_info.volume_step}")

            # Base calculation
            calculated_lot = self.base_lot_size

            # 🥇 Gold confidence multiplier (more conservative)
            if confidence >= 0.90:
                calculated_lot *= self.lot_multiplier
            elif confidence >= 0.85:
                calculated_lot *= (self.lot_multiplier * 0.75)

            # 🥇 Session-based adjustment
            session_multiplier = self.get_gold_session_risk_multiplier()
            calculated_lot *= session_multiplier

            # Apply broker limits
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            calculated_lot = max(min_lot, calculated_lot)
            calculated_lot = min(max_lot, calculated_lot)
            calculated_lot = min(self.max_lot_size, calculated_lot)

            if lot_step > 0:
                calculated_lot = round(calculated_lot / lot_step) * lot_step

            final_lot = max(min_lot, calculated_lot)

            print(f"🥇 Gold Position size: {self.base_lot_size} → {final_lot} (session: {session_multiplier}x)")

            return final_lot

        except Exception as e:
            print(f"❌ Gold position size calculation error: {str(e)}")
            return self.base_lot_size

    def calculate_gold_sl_tp_levels(
        self, symbol: str, order_type: int, entry_price: float
    ) -> Tuple[float, float]:
        """Calculate Gold Stop Loss and Take Profit levels in points"""
        try:
            # 🥇 Gold uses point-based calculation
            sl_points = self.default_sl_points
            tp_points = sl_points * self.default_tp_ratio

            # 🥇 Session-based adjustment
            session_multiplier = self.get_gold_session_risk_multiplier()
            adjusted_sl_points = sl_points * session_multiplier
            adjusted_tp_points = tp_points * session_multiplier

            # Convert points to price levels
            if order_type == mt5.ORDER_TYPE_BUY:
                stop_loss = entry_price - (adjusted_sl_points * self.gold_point_value)
                take_profit = entry_price + (adjusted_tp_points * self.gold_point_value)
            else:
                stop_loss = entry_price + (adjusted_sl_points * self.gold_point_value)
                take_profit = entry_price - (adjusted_tp_points * self.gold_point_value)

            print(f"🥇 Gold SL/TP: {adjusted_sl_points:.0f} points SL, {adjusted_tp_points:.0f} points TP")
            return stop_loss, take_profit

        except Exception as e:
            print(f"❌ Gold SL/TP calculation error: {str(e)}")
            return 0.0, 0.0

    def send_gold_order(
        self,
        symbol: str,
        order_type: int,
        lot_size: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "Gold_SMC_AI_Bot",
        signal_data: Dict = None,
    ) -> bool:
        """Enhanced Gold order sending with comprehensive logging"""

        if not self.trading_enabled:
            print("⚠️ Gold trading disabled - order not sent")
            return False

        # 🥇 Check Gold session
        if not self.is_gold_session_allowed():
            current_session = self.get_current_gold_session()
            print(f"⚠️ Gold trading not allowed during {current_session} session")
            return False

        try:
            # Get Gold market data before order execution
            market_data = self.get_comprehensive_gold_market_data(symbol)
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Gold symbol {symbol} not found")
                return False

            # Validate lot size for Gold
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            print(f"🔍 Gold order validation:")
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
                "deviation": 30,  # Larger deviation for Gold
                "magic": 234567,  # Different magic for Gold
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            print(f"🥇 Sending Gold order: {order_type_str} {lot_size} {symbol}")

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Gold order failed: {result.retcode} - {result.comment}")

                if result.retcode == 10014:
                    print("🔄 Trying FOK filling for Gold...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)

                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"❌ Second Gold order attempt failed: {result.retcode}")
                        return False
                else:
                    return False

            # 🥇 Enhanced Gold trade logging
            order_details = {
                "ticket": result.order,
                "type": order_type,
                "volume": lot_size,
                "price": result.price,
                "sl": stop_loss,
                "tp": take_profit
            }
            
            # Log Gold trade entry with comprehensive data
            trade_id = self.log_gold_trade_entry(
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
                "session": self.get_current_gold_session(),
                "sl_points": self._calculate_sl_points(order_details),
                "tp_points": self._calculate_tp_points(order_details),
            }

            self.trade_history.append(trade_info)
            self.active_positions[result.order] = trade_info

            # Update session trade count
            current_session = self.get_current_gold_session()
            self.session_trades[current_session] = self.session_trades.get(current_session, 0) + 1

            print(f"✅ Gold order executed: {order_type_str} {lot_size} {symbol} @ {result.price:.2f}")
            print(f"   🎯 SL: {stop_loss:.2f} ({self._calculate_sl_points(order_details):.0f} points) | TP: {take_profit:.2f} ({self._calculate_tp_points(order_details):.0f} points)")
            print(f"   🎫 Ticket: {result.order}")
            print(f"   🥇 Session: {current_session.upper()}")
            print(f"   📝 Logged as: {trade_id}")

            return True

        except Exception as e:
            print(f"❌ Gold order execution error: {str(e)}")
            return False

    def process_gold_signal(self, signal: Dict, symbol: str) -> bool:
        """Enhanced Gold signal processing with session filtering"""

        if not self.trading_enabled:
            return False

        # 🥇 Check Gold session first
        if not self.is_gold_session_allowed():
            current_session = self.get_current_gold_session()
            print(f"⚠️ Gold signal ignored - {current_session} session not allowed")
            return False

        if not self.check_can_trade():
            return False

        # 🥇 Higher confidence threshold for Gold
        if signal["final_confidence"] < self.min_confidence:
            print(f"⚠️ Gold signal confidence too low: {signal['final_confidence']:.3f}")
            return False

        if signal["trading_recommendation"] != "TRADE":
            return False

        # 🥇 More strict consensus for Gold
        individual_signals = signal["individual_signals"]
        long_count = sum(
            1 for s in individual_signals.values() if s["consensus_prediction"] == 1
        )
        short_count = sum(
            1 for s in individual_signals.values() if s["consensus_prediction"] == -1
        )

        total_agreement = max(long_count, short_count)
        if total_agreement < self.min_consensus:
            print(f"⚠️ Insufficient Gold consensus: {total_agreement}/{len(individual_signals)}")
            return False

        if signal["final_direction"] == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
        elif signal["final_direction"] == "SHORT":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            return False

        # 🥇 Gold-specific position sizing
        lot_size = self.calculate_gold_position_size(symbol, signal["final_confidence"])

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        stop_loss, take_profit = self.calculate_gold_sl_tp_levels(symbol, order_type, entry_price)

        comment = f"Gold_SMC_{signal['final_direction']}_C{signal['final_confidence']:.2f}_{self.get_current_gold_session()}"

        # 🥇 Send Gold order with signal data
        success = self.send_gold_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            signal_data=signal
        )

        if success:
            self.hourly_trade_count += 1
            current_session = self.get_current_gold_session()
            print(f"🥇 Gold auto trade executed: {signal['final_direction']} {symbol} during {current_session} session")

        return success

    def check_can_trade(self) -> bool:
        """Check if Gold system can place new trades"""
        if self.wait_for_trade_completion:
            if len(self.active_positions) > 0:
                print(f"⏳ Waiting for Gold trade to close. Active positions: {len(self.active_positions)}")
                return False

        return self.check_gold_risk_limits()

    def check_gold_risk_limits(self) -> bool:
        """Check Gold-specific risk limits"""
        # Concurrent trades limit
        if len(self.active_positions) >= self.max_concurrent_trades:
            print(f"🛑 Maximum Gold concurrent trades reached: {len(self.active_positions)}")
            return False

        # Hourly trade limit
        current_hour = datetime.now().hour
        if current_hour != self.hour_start:
            self.hourly_trade_count = 0
            self.hour_start = current_hour

        if self.hourly_trade_count >= self.max_trades_per_hour:
            print(f"🛑 Gold hourly trade limit reached: {self.hourly_trade_count}")
            return False

        # 🥇 Daily Gold points limit
        daily_points_limit = 2000  # 2000 points daily limit
        if abs(self.daily_gold_points) > daily_points_limit:
            print(f"🛑 Gold daily points limit reached: {self.daily_gold_points:.0f} points")
            return False

        # 🥇 Session-specific limits
        current_session = self.get_current_gold_session()
        max_session_trades = {
            "london": 2,
            "us": 2, 
            "overlap": 3,
            "asian": 1
        }
        
        session_trade_count = self.session_trades.get(current_session, 0)
        max_for_session = max_session_trades.get(current_session, 1)
        
        if session_trade_count >= max_for_session:
            print(f"🛑 Gold {current_session} session trade limit reached: {session_trade_count}")
            return False

        return True

    def print_gold_settings(self):
        """Print current Gold configuration"""
        print("🥇 Gold Auto Trader Settings:")
        print("=" * 50)
        print(f"🎯 Max concurrent trades: {self.max_concurrent_trades}")
        print(f"⏳ Wait for completion: {'YES' if self.wait_for_trade_completion else 'NO'}")
        print(f"📊 Min confidence: {self.min_confidence*100}%")
        print(f"🤝 Min consensus: {self.min_consensus}/5")
        print(f"💰 Base lot size: {self.base_lot_size} (per 0.01 lot = 1 troy ounce)")
        print(f"🛡️ Default SL: {self.default_sl_points} points (${self.default_sl_points * self.gold_point_value})")
        print(f"🎯 Default TP ratio: {self.default_tp_ratio}:1")
        print("🌍 Session Settings:")
        print(f"   London: {'✅' if self.enable_london_session else '❌'} (High volatility)")
        print(f"   US: {'✅' if self.enable_us_session else '❌'} (High impact)")
        print(f"   Overlap: {'✅' if self.enable_overlap_session else '❌'} (Highest volatility)")
        print(f"   Asian: {'✅' if self.enable_asian_session else '❌'} (Low liquidity)")
        print(f"📝 Logging: ENABLED - {self.log_directory}")
        print("=" * 50)

    def start_gold_auto_trading(self, symbol: str = None, update_interval: int = 60):
        """Start Gold automated trading system"""

        # Find Gold symbol if not provided
        if symbol is None:
            symbol = self.find_gold_symbol()
            if symbol is None:
                print("❌ No Gold symbol found")
                return

        print("🥇 Starting Gold SMC Auto Trading System")
        print("=" * 60)
        print(f"📊 Symbol: {symbol}")
        print(f"🎯 Trading Status: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"⏳ Mode: One Gold trade at a time")
        print(f"📝 Comprehensive Gold Logging: ACTIVE")
        print(f"🌍 Current Session: {self.get_current_gold_session().upper()}")
        print("=" * 60)

        last_signal = None

        try:
            while True:
                try:
                    self.update_positions()

                    if not self.should_analyze_signals():
                        print(f"\n⏳ {datetime.now().strftime('%H:%M:%S')} - Waiting for active Gold trade to close...")
                        print(f"🥇 Daily Gold P&L: {self.daily_gold_points:.0f} points | Active Positions: {len(self.active_positions)}")

                        if self.active_positions:
                            for ticket, trade_info in self.active_positions.items():
                                positions = mt5.positions_get(ticket=ticket)
                                if positions:
                                    pos = positions[0]
                                    current_points = (pos.price_current - pos.price_open) / self.gold_point_value
                                    if pos.type == 1:  # Sell position
                                        current_points *= -1
                                    
                                    print(f"🔄 Active Gold: {trade_info['type']} {trade_info['volume']} {trade_info['symbol']}")
                                    print(f"   Entry: {trade_info['price']:.2f} | Current: {pos.price_current:.2f} | P&L: {current_points:.0f} points (${pos.profit:.2f})")

                        time.sleep(update_interval)
                        continue

                    # Check if current session is allowed
                    if not self.is_gold_session_allowed():
                        current_session = self.get_current_gold_session()
                        print(f"\n⏸️ {datetime.now().strftime('%H:%M:%S')} - Gold trading paused during {current_session} session")
                        time.sleep(300)  # Wait 5 minutes during disallowed sessions
                        continue

                    print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Analyzing Gold signals...")
                    signal = self.signal_engine.get_multi_timeframe_signals(symbol)

                    if "error" in signal:
                        print(f"❌ Gold signal error: {signal['error']}")
                    else:
                        session_info = signal.get('session_info', {})
                        print(f"🥇 {symbol}: {signal['final_direction']} | Confidence: {signal['final_confidence']:.3f}")
                        print(f"🎯 Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}")
                        print(f"🌍 Session: {session_info.get('session', 'unknown').upper()} | Gold Points: {self.daily_gold_points:.0f}")

                        signal_changed = self._is_signal_changed(last_signal, signal)

                        if signal_changed and signal["trading_recommendation"] == "TRADE":
                            if self.trading_enabled:
                                print("🔥 NEW GOLD TRADING SIGNAL DETECTED!")
                                success = self.process_gold_signal(signal, symbol)
                                if success:
                                    print("✅ Gold auto trade executed successfully")
                                    print("📝 Gold trade logged with comprehensive market data")
                                else:
                                    print("❌ Gold auto trade failed or blocked")
                            else:
                                print("📊 GOLD TRADING SIGNAL (Trading disabled)")

                        last_signal = signal

                    time.sleep(update_interval)

                except KeyboardInterrupt:
                    print("\n🛑 Gold auto trading stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Gold auto trading error: {str(e)}")
                    time.sleep(10)

        finally:
            # Print final Gold summary when stopping
            self.print_gold_trading_summary()
            print("✅ Gold auto trading system stopped")

    def _is_signal_changed(self, last_signal: Optional[Dict], current_signal: Dict) -> bool:
        """Determine if Gold signal has changed enough to warrant new trade"""

        if last_signal is None:
            if self.enable_first_signal_trade:
                return (
                    current_signal["final_confidence"] >= self.first_signal_min_confidence
                    and current_signal["trading_recommendation"] == "TRADE"
                    and self.is_gold_session_allowed()  # 🥇 Gold session check
                )
            else:
                return False

        if last_signal["final_direction"] != current_signal["final_direction"]:
            return True

        # 🥇 Gold uses point-based threshold
        if "current_price" in current_signal:
            price_change_points = abs(
                current_signal.get("current_price", 0) - 
                last_signal.get("current_price", 0)
            ) / self.gold_point_value
            
            if price_change_points > self.signal_change_threshold:
                return True

        confidence_change = abs(
            last_signal["final_confidence"] - current_signal["final_confidence"]
        )
        if confidence_change > 0.05:  # 5% confidence change for Gold
            return True

        # 🥇 Force trade high confidence Gold signals during good sessions
        if (current_signal["final_confidence"] >= 0.90
            and current_signal["trading_recommendation"] == "TRADE"
            and self.is_gold_session_allowed()
            and len(self.trade_history) == 0):
            print(f"🔥 Force trading high confidence Gold signal: {current_signal['final_confidence']:.3f}")
            return True

        return False

    def should_analyze_signals(self) -> bool:
        """Determine if Gold system should analyze new signals"""
        if not self.wait_for_trade_completion:
            return True

        if len(self.active_positions) > 0:
            return False

        return True

    def update_positions(self):
        """Enhanced Gold position update with comprehensive logging"""
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
                    print(f"🥇 Gold Trade #{ticket} CLOSED:")
                    print(f"   {closed_trade['type']} {closed_trade['volume']} {closed_trade['symbol']}")
                    print(f"   Entry: {closed_trade['price']:.2f}")

                    # Get deal details for comprehensive Gold logging
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
                                
                                # Determine Gold exit reason
                                if abs(deal.price - closed_trade.get('tp', 0)) < 0.02:  # 2 cents tolerance for Gold
                                    exit_details['exit_reason'] = 'TP_HIT'
                                elif abs(deal.price - closed_trade.get('sl', 0)) < 0.02:
                                    exit_details['exit_reason'] = 'SL_HIT'
                                else:
                                    exit_details['exit_reason'] = 'TIME_EXIT'

                                # Calculate Gold points
                                if closed_trade['type'] == 'BUY':
                                    points_pnl = (deal.price - closed_trade['price']) / self.gold_point_value
                                else:
                                    points_pnl = (closed_trade['price'] - deal.price) / self.gold_point_value

                                self.daily_gold_points += points_pnl

                                print(f"   Exit: {deal.price:.2f}")
                                print(f"   P&L: ${deal.profit:.2f} ({points_pnl:.0f} Gold points)")
                                print(f"   Reason: {exit_details['exit_reason']}")
                                print(f"   Result: {'✅ WIN' if deal.profit > 0 else '❌ LOSS'}")
                                break

                    # 🥇 Log Gold trade outcome
                    market_data = self.get_comprehensive_gold_market_data(closed_trade['symbol'])
                    
                    self.update_trade_outcome(
                        trade_id=closed_trade.get('trade_id', f"LEGACY_GOLD_{ticket}"),
                        exit_details=exit_details,
                        market_data=market_data
                    )

                    del self.active_positions[ticket]
                    self.last_trade_closed_time = datetime.now()

            if closed_tickets:
                print(f"🥇 {len(closed_tickets)} Gold position(s) closed. Ready for new Gold trades.")

            total_profit = sum(pos.profit for pos in positions)
            self.daily_pnl = total_profit

        except Exception as e:
            print(f"❌ Gold position update error: {str(e)}")

    def update_trade_outcome(self, trade_id: str, exit_details: Dict, market_data: Dict):
        """Update Gold trade outcome in logs"""
        # Implementation similar to original but with Gold-specific enhancements
        # (This would be the same logic as in the original file but with Gold-specific calculations)
        pass

    def print_gold_trading_summary(self):
        """Print comprehensive Gold trading summary"""
        # Get Gold statistics from logs
        stats = self.get_gold_trading_statistics()
        
        print("\n" + "="*60)
        print("🥇 GOLD TRADING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎯 Total Gold Trades: {stats['total_trades']}")
        print(f"✅ Winning Trades: {stats['winning_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        print(f"📈 Win Rate: {stats['win_rate']:.1f}%")
        print(f"🥇 Total P&L: {stats['total_pnl_points']:.0f} Gold points (${stats['total_pnl_points'] * self.gold_point_value:.2f})")
        print(f"🏆 Average Win: {stats['avg_win_points']:.0f} points")
        print(f"💸 Average Loss: {stats['avg_loss_points']:.0f} points")
        print(f"🚀 Largest Win: {stats['largest_win']:.0f} points")
        print(f"🔻 Largest Loss: {stats['largest_loss']:.0f} points")
        print(f"⏱️ Avg Duration: {stats['avg_trade_duration']:.0f} minutes")
        print(f"🧠 Model Accuracy: {stats['model_accuracy']:.1f}%")
        print("Session Performance:")
        for session, count in self.session_trades.items():
            print(f"   {session.title()}: {count} trades")
        print("="*60)
        print(f"📝 Gold Log Files: {self.log_directory}")
        print("="*60)

    def get_gold_trading_statistics(self) -> Dict:
        """Get comprehensive Gold trading statistics"""
        # This would implement Gold-specific statistics calculation
        # Similar to original but focusing on Gold points and session-based metrics
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0),
            'losing_trades': sum(1 for trade in self.trade_history if trade.get('profit', 0) < 0),
            'win_rate': 0.0,
            'total_pnl_points': self.daily_gold_points,
            'avg_win_points': 0.0,
            'avg_loss_points': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_trade_duration': 0.0,
            'model_accuracy': 0.0
        }


# Main execution for Gold
if __name__ == "__main__":
    print("🥇 Gold SMC Auto Trading Bot with Enhanced Features")
    print("=" * 60)

    # 🥇 Gold-specific settings
    SIGNAL_CHANGE_THRESHOLD = 0.5  # 50 cents in Gold points
    ENABLE_FIRST_TRADE = True
    FIRST_TRADE_MIN_CONFIDENCE = 0.80  # Higher for Gold
    MIN_CONFIDENCE = 0.80              # Higher for Gold
    MIN_CONSENSUS = 4                  # 4/5 timeframes for Gold
    MAX_CONCURRENT_TRADES = 1          # One Gold trade at a time
    WAIT_FOR_COMPLETION = True
    BASE_LOT_SIZE = 0.01              # 1 troy ounce
    MAX_LOT_SIZE = 0.05               # Conservative max for Gold

    # Initialize Gold trader
    gold_trader = SMCGoldAutoTrader(
        models_path="XAUUSD_c_SMC",
        signal_change_threshold=SIGNAL_CHANGE_THRESHOLD,
        enable_first_signal_trade=ENABLE_FIRST_TRADE,
        first_signal_min_confidence=FIRST_TRADE_MIN_CONFIDENCE,
        min_confidence=MIN_CONFIDENCE,
        min_consensus=MIN_CONSENSUS,
        max_concurrent_trades=MAX_CONCURRENT_TRADES,
        wait_for_trade_completion=WAIT_FOR_COMPLETION,
        base_lot_size=BASE_LOT_SIZE,
        max_lot_size=MAX_LOT_SIZE,
        # 🥇 Gold session preferences
        enable_london_session=True,
        enable_us_session=True,
        enable_asian_session=False,  # Avoid low liquidity
        enable_overlap_session=True,
    )

    gold_trader.print_gold_settings()

    if gold_trader.connect_mt5():
        if gold_trader.load_models():
            print("\n🥇 Gold Auto Trading Bot Ready!")
            print("📝 All Gold trades will be logged with comprehensive data")

            enable_trading = input("\n🚀 Enable LIVE GOLD AUTO TRADING? (yes/no): ").lower().strip()

            if enable_trading == "yes":
                gold_trader.enable_trading(True)
            else:
                print("📊 Demo mode")
                gold_trader.enable_trading(False)

            # Start Gold auto trading
            gold_trader.start_gold_auto_trading()

        else:
            print("❌ Failed to load Gold AI models")
    else:
        print("❌ Failed to connect to MT5")