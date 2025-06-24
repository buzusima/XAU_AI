"""
Smart Auto Trading System with Portfolio Risk Management
========================================================
Professional Auto Trading with ONE TRADE PER PAIR Control
"""

from flask import Flask, jsonify, send_from_directory, request
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

class SmartAutoTradingDashboard:
    """Smart Auto Trading Dashboard with Portfolio Risk Control"""
    
    def __init__(self):
        """Initialize Smart Auto Trading Dashboard"""
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
        
        # Portfolio Risk Management Settings
        self.account_balance = 10000.0
        self.portfolio_risk_profiles = {
            'CONSERVATIVE': {'risk_per_trade': 0.5, 'max_total_exposure': 2.0, 'max_daily_loss': 2.0},
            'MODERATE': {'risk_per_trade': 1.0, 'max_total_exposure': 4.0, 'max_daily_loss': 3.0},
            'BALANCED': {'risk_per_trade': 1.5, 'max_total_exposure': 6.0, 'max_daily_loss': 4.0},
            'AGGRESSIVE': {'risk_per_trade': 2.0, 'max_total_exposure': 8.0, 'max_daily_loss': 5.0},
            'HIGH_RISK': {'risk_per_trade': 3.0, 'max_total_exposure': 12.0, 'max_daily_loss': 8.0}
        }
        
        # Current Portfolio Settings
        self.current_risk_profile = 'BALANCED'
        self.custom_risk_per_trade = 1.5  # Custom risk percentage
        self.max_risk_per_trade = 0.015   # 1.5% per trade
        self.max_total_exposure = 0.06    # 6% total portfolio
        self.max_daily_loss = 0.04        # 4% daily loss limit
        
        # ONE TRADE PER PAIR CONTROL
        self.one_trade_per_pair = True    # Enforce one trade per pair
        self.active_trades_per_pair = {}  # Track active trades per pair
        self.pair_trade_status = {}       # Track if pair can trade
        
        # AUTO TRADING CORE SETTINGS
        self.auto_trading_enabled = False
        self.auto_trading_pairs = set(self.forex_pairs)  # All pairs enabled by default
        self.min_signal_strength = 6.0   # Minimum signal strength (1-10)
        self.min_entry_quality = 'GOOD'  # Minimum entry quality (POOR/FAIR/GOOD/EXCELLENT)
        self.max_simultaneous_trades = 8  # Maximum total open trades
        
        # Signal Filtering
        self.required_confirmations = {
            'trend_alignment': True,      # Require trend alignment
            'volume_confirmation': True,  # Require volume confirmation
            'rsi_filter': True,          # RSI not in extreme zones
            'multiple_timeframe': True    # Multiple timeframe confirmation
        }
        
        # Trading Time Controls
        self.trading_sessions = {
            'ASIAN': {'start': '00:00', 'end': '09:00', 'enabled': False},
            'LONDON': {'start': '08:00', 'end': '17:00', 'enabled': True},
            'NEWYORK': {'start': '13:00', 'end': '22:00', 'enabled': True},
            'OVERLAP': {'start': '13:00', 'end': '17:00', 'enabled': True}  # London-NY overlap
        }
        
        # Position Management
        self.default_lot_size = 0.01
        self.max_lot_size = 2.0
        self.min_lot_size = 0.01
        self.partial_close_enabled = True
        self.trailing_stop_enabled = True
        
        # Trade Execution Settings
        self.slippage_tolerance = 3       # Max slippage in pips
        self.max_spread_threshold = 2.0   # Max spread in pips
        self.trade_timeout = 30           # Order timeout in seconds
        
        # Data Storage
        self.live_data = {}
        self.account_info = {}
        self.open_positions = {}
        self.trade_history = []
        self.pending_signals = {}
        self.signal_log = []
        self.daily_stats = {
            'trades_executed': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # System State
        self.is_running = False
        self.mt5_connected = False
        self.last_update = datetime.now()
        self.emergency_stop = False
        
        # Cooldown Management
        self.trade_cooldowns = {}         # Per-pair cooldowns
        self.global_cooldown = 60         # Global cooldown between any trades
        self.last_global_trade_time = None
        
        self.setup_logging()
        self.setup_routes()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('auto_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create separate logger for trades
        trade_logger = logging.getLogger('trades')
        trade_handler = logging.FileHandler('trade_execution.log')
        trade_handler.setFormatter(logging.Formatter('%(asctime)s - TRADE - %(message)s'))
        trade_logger.addHandler(trade_handler)
        trade_logger.setLevel(logging.INFO)
        self.trade_logger = trade_logger
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 with enhanced error handling"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            # Update account settings
            self.account_balance = account_info.balance
            self.account_info = {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'leverage': account_info.leverage,
                'trade_allowed': account_info.trade_allowed
            }
            
            # Check if trading is allowed
            if not account_info.trade_allowed:
                self.logger.warning("Trading is not allowed on this account!")
                self.auto_trading_enabled = False
            
            # Initialize pair tracking
            for pair in self.forex_pairs:
                self.active_trades_per_pair[pair] = []
                self.pair_trade_status[pair] = 'READY'  # READY, TRADING, COOLDOWN
                self.trade_cooldowns[pair] = None
            
            # Test symbols and ensure they're visible
            available_symbols = []
            for symbol in self.forex_pairs:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
            
            self.forex_pairs = available_symbols
            self.mt5_connected = True
            
            self.logger.info(f"MT5 Connected Successfully!")
            self.logger.info(f"Account: {account_info.login}")
            self.logger.info(f"Balance: ${account_info.balance:,.2f}")
            self.logger.info(f"Available Pairs: {len(self.forex_pairs)}")
            self.logger.info(f"Trading Allowed: {account_info.trade_allowed}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def update_risk_profile(self, profile_name: str = None, custom_risk: float = None):
        """Update portfolio risk profile"""
        try:
            if custom_risk:
                self.custom_risk_per_trade = custom_risk
                self.max_risk_per_trade = custom_risk / 100
                self.logger.info(f"Custom risk set to {custom_risk}% per trade")
            elif profile_name and profile_name in self.portfolio_risk_profiles:
                profile = self.portfolio_risk_profiles[profile_name]
                self.current_risk_profile = profile_name
                self.max_risk_per_trade = profile['risk_per_trade'] / 100
                self.max_total_exposure = profile['max_total_exposure'] / 100
                self.max_daily_loss = profile['max_daily_loss'] / 100
                
                self.logger.info(f"Risk profile updated to {profile_name}")
                self.logger.info(f"Risk per trade: {profile['risk_per_trade']}%")
                self.logger.info(f"Max total exposure: {profile['max_total_exposure']}%")
                
        except Exception as e:
            self.logger.error(f"Error updating risk profile: {str(e)}")
    
    def check_pair_trading_status(self, symbol: str) -> Dict:
        """Check if pair can trade based on one-trade-per-pair rule"""
        try:
            if not self.one_trade_per_pair:
                return {'can_trade': True, 'reason': 'Multiple trades allowed'}
            
            # Get current positions for this pair
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is None:
                positions = []
            
            # Update active trades tracking
            self.active_trades_per_pair[symbol] = list(positions)
            
            # Check if pair has active trades
            if len(positions) > 0:
                self.pair_trade_status[symbol] = 'TRADING'
                return {
                    'can_trade': False, 
                    'reason': f'Active trade exists (Ticket: {positions[0].ticket})',
                    'active_trades': len(positions)
                }
            
            # Check cooldown
            if self.trade_cooldowns.get(symbol):
                cooldown_remaining = (self.trade_cooldowns[symbol] - datetime.now()).total_seconds()
                if cooldown_remaining > 0:
                    self.pair_trade_status[symbol] = 'COOLDOWN'
                    return {
                        'can_trade': False,
                        'reason': f'Cooldown active ({int(cooldown_remaining)}s remaining)',
                        'cooldown_remaining': int(cooldown_remaining)
                    }
            
            # Pair is ready to trade
            self.pair_trade_status[symbol] = 'READY'
            return {'can_trade': True, 'reason': 'Ready to trade'}
            
        except Exception as e:
            self.logger.error(f"Error checking pair status {symbol}: {str(e)}")
            return {'can_trade': False, 'reason': f'Error: {str(e)}'}
    
    # แทนที่ฟังก์ชัน calculate_position_size ด้วยการคำนวณทองคำที่ถูกต้อง

    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str, risk_percent: float = None) -> Dict:
        """Calculate position size with CORRECT GOLD calculation"""
        try:
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # ตั้งค่าเริ่มต้น
            lot_size = self.default_lot_size
            money_per_pip = 10.0
            pip_size = 0.0001
            
            # Risk amount
            risk_amount = self.account_balance * risk_percent
            
            # 🥇 GOLD (XAUUSD) - การคำนวณพิเศษ
            if 'XAU' in symbol or 'GOLD' in symbol:
                pip_size = 0.1          # Gold: 1 pip = $0.10
                money_per_pip = 1.0     # Gold: $1 per pip per 1 lot (100 oz)
                
                # Gold lot size calculation
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    # Gold: Risk = Pips × $1 × Lot Size
                    lot_size = risk_amount / pips_at_risk
                
            # 💴 JPY Pairs
            elif 'JPY' in symbol:
                pip_size = 0.01         # JPY: 1 pip = 0.01
                money_per_pip = 10.0 / entry_price if entry_price > 0 else 0.1
                
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
            
            # 💱 Forex Pairs ปกติ
            else:
                pip_size = 0.0001       # Forex: 1 pip = 0.0001
                money_per_pip = 10.0    # $10 per pip per 1 lot
                
                points_risk = abs(entry_price - stop_loss)
                pips_at_risk = points_risk / pip_size
                
                if pips_at_risk > 0:
                    lot_size = risk_amount / (pips_at_risk * money_per_pip)
            
            # ✅ จำกัดขนาด lot
            lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
            
            # ปัดเศษตาม symbol
            if 'XAU' in symbol:
                lot_size = round(lot_size, 2)      # Gold: 0.01, 0.05, 0.10
            elif lot_size >= 1.0:
                lot_size = round(lot_size, 1)      # 1.0, 1.5, 2.0
            else:
                lot_size = round(lot_size, 2)      # 0.01, 0.05, 0.10
            
            # คำนวณความเสี่ยงจริง
            points_risk = abs(entry_price - stop_loss)
            pips_at_risk = points_risk / pip_size
            
            # 🥇 Gold: Risk = Pips × $1 × Lot
            if 'XAU' in symbol:
                actual_risk = pips_at_risk * 1.0 * lot_size  # $1 per pip per lot
            # 💴 JPY: Risk = Pips × (10/Price) × Lot  
            elif 'JPY' in symbol:
                actual_risk = pips_at_risk * money_per_pip * lot_size
            # 💱 Forex: Risk = Pips × $10 × Lot
            else:
                actual_risk = pips_at_risk * 10.0 * lot_size
            
            risk_percentage = (actual_risk / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            return {
                'lot_size': lot_size,
                'risk_amount': round(actual_risk, 2),
                'risk_percentage': round(risk_percentage, 2),
                'pip_value': round(money_per_pip * lot_size, 2),
                'points_risk': round(points_risk, 5),
                'pips_at_risk': round(pips_at_risk, 1),
                'pip_size': pip_size,
                'money_per_pip': money_per_pip,
                'symbol_type': 'GOLD' if 'XAU' in symbol else 'JPY' if 'JPY' in symbol else 'FOREX',
                'calculation_status': 'SUCCESS'
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation error for {symbol}: {str(e)}")
            
            # ✅ Return safe defaults based on symbol
            if 'XAU' in symbol:
                default_pip_size = 0.1
                default_money_per_pip = 1.0
            elif 'JPY' in symbol:
                default_pip_size = 0.01
                default_money_per_pip = 0.1
            else:
                default_pip_size = 0.0001
                default_money_per_pip = 10.0
                
            return {
                'lot_size': self.default_lot_size,
                'risk_amount': 0,
                'risk_percentage': 0,
                'pip_value': default_money_per_pip * self.default_lot_size,
                'points_risk': 0,
                'pips_at_risk': 0,
                'pip_size': default_pip_size,
                'money_per_pip': default_money_per_pip,
                'symbol_type': 'GOLD' if 'XAU' in symbol else 'JPY' if 'JPY' in symbol else 'FOREX',
                'calculation_status': 'ERROR',
                'error_message': str(e)
            }    
    def validate_trading_signal(self, symbol: str, signal_data: Dict) -> Dict:
        """Validate trading signal with comprehensive checks"""
        try:
            validation_result = {
                'valid': False,
                'score': 0,
                'issues': [],
                'confirmations': []
            }
            
            # Check if auto trading is enabled
            if not self.auto_trading_enabled:
                validation_result['issues'].append('Auto trading disabled')
                return validation_result
            
            # Check emergency stop
            if self.emergency_stop:
                validation_result['issues'].append('Emergency stop activated')
                return validation_result
            
            # Check pair trading status
            pair_status = self.check_pair_trading_status(symbol)
            if not pair_status['can_trade']:
                validation_result['issues'].append(f"Pair not ready: {pair_status['reason']}")
                return validation_result
            
            # Check signal strength
            signal_strength = signal_data.get('strength', 0)
            if signal_strength < self.min_signal_strength:
                validation_result['issues'].append(f'Signal strength too low: {signal_strength} < {self.min_signal_strength}')
            else:
                validation_result['confirmations'].append(f'Strong signal: {signal_strength}/10')
                validation_result['score'] += 2
            
            # Check entry quality
            entry_quality = signal_data.get('entry_quality', 'POOR')
            quality_scores = {'POOR': 0, 'FAIR': 1, 'GOOD': 2, 'EXCELLENT': 3}
            min_quality_score = quality_scores.get(self.min_entry_quality, 2)
            
            if quality_scores.get(entry_quality, 0) < min_quality_score:
                validation_result['issues'].append(f'Entry quality too low: {entry_quality} < {self.min_entry_quality}')
            else:
                validation_result['confirmations'].append(f'Good entry quality: {entry_quality}')
                validation_result['score'] += quality_scores.get(entry_quality, 0)
            
            # Check signal direction
            signal_direction = signal_data.get('signal', 'NONE')
            if signal_direction == 'NONE':
                validation_result['issues'].append('No clear signal direction')
            else:
                validation_result['confirmations'].append(f'Clear signal: {signal_direction}')
                validation_result['score'] += 1
            
            # Check risk/reward ratio
            rr_ratio = signal_data.get('rr_tp1', 0)
            if rr_ratio < getattr(self, 'min_rr_ratio', 1.5):
                validation_result['issues'].append(f'Poor risk/reward: 1:{rr_ratio}')
            else:
                validation_result['confirmations'].append(f'Good R/R: 1:{rr_ratio}')
                validation_result['score'] += 1
            
            # Check current exposure
            current_exposure = self.calculate_current_exposure()
            if current_exposure >= self.max_total_exposure:
                validation_result['issues'].append(f'Max exposure reached: {current_exposure*100:.1f}%')
            else:
                validation_result['confirmations'].append(f'Exposure OK: {current_exposure*100:.1f}%')
                validation_result['score'] += 1
            
            # Check daily loss limit
            if abs(self.daily_stats['total_pnl']) >= (self.account_balance * self.max_daily_loss):
                validation_result['issues'].append('Daily loss limit reached')
            else:
                validation_result['confirmations'].append('Daily loss limit OK')
                validation_result['score'] += 1
            
            # Check trading session
            if not self.is_trading_session_active():
                validation_result['issues'].append('Outside trading hours')
            else:
                validation_result['confirmations'].append('Trading session active')
                validation_result['score'] += 1
            
            # Check maximum trades
            open_positions = mt5.positions_total()
            if open_positions >= self.max_simultaneous_trades:
                validation_result['issues'].append(f'Max trades reached: {open_positions}/{self.max_simultaneous_trades}')
            else:
                validation_result['confirmations'].append(f'Trade slots available: {open_positions}/{self.max_simultaneous_trades}')
                validation_result['score'] += 1
            
            # Final validation
            validation_result['valid'] = len(validation_result['issues']) == 0 and validation_result['score'] >= 5
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating signal for {symbol}: {str(e)}")
            return {'valid': False, 'issues': [f'Validation error: {str(e)}'], 'score': 0}
    
    def execute_trade(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute trade based on validated signal"""
        try:
            # Final validation
            validation = self.validate_trading_signal(symbol, signal_data)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'Signal validation failed',
                    'issues': validation['issues']
                }
            
            # Get signal details
            signal_direction = signal_data.get('signal', 'NONE')
            entry_price = signal_data.get('optimal_entry', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            
            if signal_direction == 'NONE' or entry_price == 0:
                return {'success': False, 'error': 'Invalid signal data'}
            
            # Calculate position size
            position_info = self.calculate_position_size(entry_price, stop_loss, symbol)
            lot_size = position_info['lot_size']
            
            # Determine order type
            if signal_direction in ['BUY', 'STRONG_BUY']:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                return {'success': False, 'error': 'Invalid signal direction'}
            
            # Prepare order request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit_1,
                'deviation': self.slippage_tolerance,
                'magic': 12345,  # EA magic number
                'comment': f'Auto Trade - {signal_direction}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Execute order
            self.trade_logger.info(f"Executing {signal_direction} order for {symbol} - Lot: {lot_size}")
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'retcode': result.retcode}
            
            # Order successful
            trade_info = {
                'ticket': result.order,
                'symbol': symbol,
                'lot_size': lot_size,
                'entry_price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'signal_direction': signal_direction,
                'signal_strength': signal_data.get('strength', 0),
                'entry_quality': signal_data.get('entry_quality', 'UNKNOWN'),
                'risk_amount': position_info['risk_amount'],
                'risk_percentage': position_info['risk_percentage'],
                'timestamp': datetime.now(),
                'validation_score': validation['score']
            }
            
            # Update tracking
            self.active_trades_per_pair[symbol].append(result.order)
            self.pair_trade_status[symbol] = 'TRADING'
            self.daily_stats['trades_executed'] += 1
            self.last_global_trade_time = datetime.now()
            
            # Log successful trade
            self.trade_logger.info(f"TRADE EXECUTED: {symbol} {signal_direction} - Ticket: {result.order}")
            self.trade_logger.info(f"Entry: {result.price}, SL: {stop_loss}, TP: {take_profit_1}")
            self.trade_logger.info(f"Lot Size: {lot_size}, Risk: {position_info['risk_percentage']:.2f}%")
            
            return {
                'success': True,
                'ticket': result.order,
                'trade_info': trade_info,
                'validation': validation
            }
            
        except Exception as e:
            error_msg = f"Trade execution error for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                
                # Check if position was closed
                if ticket not in [pos.ticket for pos in mt5.positions_get(symbol=symbol) or []]:
                    # Position was closed, update tracking
                    if ticket in self.active_trades_per_pair.get(symbol, []):
                        self.active_trades_per_pair[symbol].remove(ticket)
                    
                    # If no more active trades for this pair, set cooldown
                    if len(self.active_trades_per_pair.get(symbol, [])) == 0:
                        self.pair_trade_status[symbol] = 'READY'
                        # Set cooldown period (5 minutes)
                        self.trade_cooldowns[symbol] = datetime.now() + timedelta(minutes=5)
                        
                        # Log position closure
                        self.trade_logger.info(f"Position closed: {symbol} Ticket: {ticket}")
                        
                        # Update daily stats
                        history = mt5.history_deals_get(ticket=ticket)
                        if history and len(history) > 0:
                            deal = history[-1]
                            if deal.profit > 0:
                                self.daily_stats['wins'] += 1
                            else:
                                self.daily_stats['losses'] += 1
                            self.daily_stats['total_pnl'] += deal.profit
        
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
    
    def calculate_current_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0.0
            
            total_risk = 0.0
            for position in positions:
                # Calculate risk for each position
                entry_price = position.price_open
                if position.sl > 0:
                    risk_points = abs(entry_price - position.sl)
                    risk_amount = risk_points * position.volume * 10  # Simplified calculation
                    total_risk += risk_amount
            
            return (total_risk / self.account_balance) if self.account_balance > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure: {str(e)}")
            return 0.0
    
    def is_trading_session_active(self) -> bool:
        """Check if current time is within active trading sessions"""
        try:
            current_time = datetime.now().strftime('%H:%M')
            current_hour = int(current_time.split(':')[0])
            
            for session_name, session_info in self.trading_sessions.items():
                if not session_info['enabled']:
                    continue
                
                start_hour = int(session_info['start'].split(':')[0])
                end_hour = int(session_info['end'].split(':')[0])
                
                if start_hour <= current_hour <= end_hour:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading session: {str(e)}")
            return True  # Default to allow trading if error
    
    def auto_trading_loop(self):
        """Main auto trading loop"""
        while self.is_running:
            try:
                if not self.auto_trading_enabled or not self.mt5_connected:
                    time.sleep(5)
                    continue
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check for new signals
                for symbol in self.auto_trading_pairs:
                    if symbol not in self.live_data:
                        continue
                    
                    signal_data = self.live_data[symbol]
                    
                    # Quick checks
                    if signal_data.get('signal', 'NONE') == 'NONE':
                        continue
                    
                    # Check pair status
                    pair_status = self.check_pair_trading_status(symbol)
                    if not pair_status['can_trade']:
                        continue
                    
                    # Validate and execute if good
                    validation = self.validate_trading_signal(symbol, signal_data)
                    if validation['valid']:
                        self.logger.info(f"Valid signal found for {symbol}: {signal_data.get('signal')}")
                        result = self.execute_trade(symbol, signal_data)
                        
                        if result['success']:
                            self.logger.info(f"Trade executed successfully: {symbol} Ticket: {result['ticket']}")
                        else:
                            self.logger.warning(f"Trade execution failed: {symbol} - {result.get('error')}")
                
                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in auto trading loop: {str(e)}")
                time.sleep(30)
    
    def start_auto_trading(self):
        """Start auto trading system"""
        if not self.mt5_connected:
            return False
        
        if self.auto_trading_enabled:
            return True
        
        self.auto_trading_enabled = True
        self.emergency_stop = False
        
        # Start auto trading thread
        trading_thread = threading.Thread(target=self.auto_trading_loop, daemon=True)
        trading_thread.start()
        
        self.logger.info("AUTO TRADING STARTED!")
        self.trade_logger.info("=== AUTO TRADING SESSION STARTED ===")
        
        return True
    
    def stop_auto_trading(self):
        """Stop auto trading system"""
        self.auto_trading_enabled = False
        self.logger.info("AUTO TRADING STOPPED!")
        self.trade_logger.info("=== AUTO TRADING SESSION STOPPED ===")
    
    def emergency_stop_all(self):
        """Emergency stop with position closure"""
        self.emergency_stop = True
        self.auto_trading_enabled = False
        
        # Close all positions
        positions = mt5.positions_get()
        if positions:
            for position in positions:
                close_request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': position.symbol,
                    'volume': position.volume,
                    'type': mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    'position': position.ticket,
                    'magic': 12345,
                    'comment': 'Emergency Stop',
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
        
        self.logger.critical("EMERGENCY STOP ACTIVATED - ALL POSITIONS CLOSED")
        self.trade_logger.critical("=== EMERGENCY STOP ACTIVATED ===")
    
    # Include all other methods from the previous version
    # (calculate_indicators, get_symbol_data, update_all_data, etc.)
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            if len(df) < 50:
                return self.get_default_indicators()
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('tick_volume', pd.Series(1, index=df.index))
            
            # EMA calculations
            ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
            ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            
            # RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
            
            # MACD calculation
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = (ema_12 - ema_26).iloc[-1]
            
            # Trend strength
            current_price = close.iloc[-1]
            trend_conditions = [
                current_price > ema_9,
                ema_9 > ema_21,
                ema_21 > ema_50
            ]
            uptrend_strength = sum(trend_conditions) / len(trend_conditions)
            
            downtrend_conditions = [
                current_price < ema_9,
                ema_9 < ema_21,
                ema_21 < ema_50
            ]
            downtrend_strength = sum(downtrend_conditions) / len(downtrend_conditions)
            trend_strength = max(uptrend_strength, downtrend_strength)
            
            # Volume analysis
            volume_avg = volume.rolling(window=20, min_periods=1).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
            
            return {
                'rsi': float(rsi) if not pd.isna(rsi) else 50.0,
                'atr': float(atr) if not pd.isna(atr) else current_price * 0.001,
                'atr_percent': float((atr / current_price) * 100) if not pd.isna(atr) else 0.1,
                'macd': float(macd) if not pd.isna(macd) else 0.0,
                'trend_strength': float(trend_strength),
                'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
                'ema_9': float(ema_9) if not pd.isna(ema_9) else current_price,
                'ema_21': float(ema_21) if not pd.isna(ema_21) else current_price,
                'ema_50': float(ema_50) if not pd.isna(ema_50) else current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return self.get_default_indicators()
    
    def get_default_indicators(self) -> Dict:
        """Return default indicators"""
        return {
            'rsi': 50.0, 'atr': 0.001, 'atr_percent': 0.1, 'macd': 0.0,
            'trend_strength': 0.0, 'volume_ratio': 1.0,
            'ema_9': 1.0, 'ema_21': 1.0, 'ema_50': 1.0
        }
    
    def analyze_entry_exit_points(self, indicators: Dict, current_price: float, symbol: str) -> Dict:
        """Analyze entry/exit points with enhanced signal generation"""
        try:
            # Get indicators
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0)
            atr = indicators.get('atr', current_price * 0.005)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # Signal analysis
            signal_direction = 'NONE'
            signal_strength = 0
            entry_score = 0
            entry_reasons = []
            
            # Bullish conditions
            if (current_price > ema_9 > ema_21 > ema_50 and 
                35 <= rsi <= 65 and 
                trend_strength >= 0.67 and 
                volume_ratio >= 1.2):
                
                signal_direction = 'BUY'
                entry_reasons.append("Strong uptrend confirmed")
                entry_score += 3
                signal_strength = 7 + (trend_strength - 0.67) * 10
                
                if trend_strength >= 0.8:
                    signal_direction = 'STRONG_BUY'
                    signal_strength = 8 + (trend_strength - 0.8) * 10
            
            # Bearish conditions
            elif (current_price < ema_9 < ema_21 < ema_50 and 
                  35 <= rsi <= 65 and 
                  trend_strength >= 0.67 and 
                  volume_ratio >= 1.2):
                
                signal_direction = 'SELL'
                entry_reasons.append("Strong downtrend confirmed")
                entry_score += 3
                signal_strength = 7 + (trend_strength - 0.67) * 10
                
                if trend_strength >= 0.8:
                    signal_direction = 'STRONG_SELL'
                    signal_strength = 8 + (trend_strength - 0.8) * 10
            
            # Additional scoring
            if 45 <= rsi <= 55:
                entry_reasons.append("RSI optimal")
                entry_score += 2
            if trend_strength >= 0.8:
                entry_reasons.append("Very strong trend")
                entry_score += 2
            elif trend_strength >= 0.7:
                entry_reasons.append("Strong trend")
                entry_score += 1
            if volume_ratio >= 1.5:
                entry_reasons.append("High volume")
                entry_score += 1
            
            # Entry quality
            if entry_score >= 6:
                entry_quality = 'EXCELLENT'
            elif entry_score >= 4:
                entry_quality = 'GOOD'
            elif entry_score >= 2:
                entry_quality = 'FAIR'
            else:
                entry_quality = 'POOR'
            
            # Calculate levels
            atr_multiplier = 1.5
            
            if signal_direction in ['BUY', 'STRONG_BUY']:
                stop_loss = current_price - (atr * atr_multiplier)
                take_profit_1 = current_price + (atr * 2.5)
                take_profit_2 = current_price + (atr * 4.0)
                take_profit_3 = current_price + (atr * 6.0)
            elif signal_direction in ['SELL', 'STRONG_SELL']:
                stop_loss = current_price + (atr * atr_multiplier)
                take_profit_1 = current_price - (atr * 2.5)
                take_profit_2 = current_price - (atr * 4.0)
                take_profit_3 = current_price - (atr * 6.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            # Position sizing
            position_info = self.calculate_position_size(current_price, stop_loss, symbol)
            
            # R/R ratios
            risk = abs(current_price - stop_loss)
            rr_tp1 = abs(take_profit_1 - current_price) / risk if risk > 0 else 0
            rr_tp2 = abs(take_profit_2 - current_price) / risk if risk > 0 else 0
            rr_tp3 = abs(take_profit_3 - current_price) / risk if risk > 0 else 0
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 1),
                'entry_quality': entry_quality,
                'entry_score': entry_score,
                'entry_reasons': entry_reasons,
                'optimal_entry': round(current_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'take_profit_3': round(take_profit_3, 5),
                'lot_size': position_info.get('lot_size', self.default_lot_size),
                'risk_amount': position_info.get('risk_amount', 0),
                'risk_percentage': position_info.get('risk_percentage', 0),
                'rr_tp1': round(rr_tp1, 2),
                'rr_tp2': round(rr_tp2, 2),
                'rr_tp3': round(rr_tp3, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry/exit: {str(e)}")
            return {
                'signal': 'NONE', 'strength': 0, 'entry_quality': 'POOR',
                'entry_score': 0, 'optimal_entry': current_price,
                'stop_loss': current_price, 'take_profit_1': current_price,
                'lot_size': self.default_lot_size, 'risk_amount': 0,
                'risk_percentage': 0, 'rr_tp1': 0, 'rr_tp2': 0, 'rr_tp3': 0
            }
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get enhanced symbol data"""
        try:
            if not self.mt5_connected:
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            current_price = tick.bid
            
            # Get multi-timeframe data
            rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            
            if any(rates is None for rates in [rates_h4, rates_h1, rates_m15, rates_m5]):
                return None
            
            df_h1 = pd.DataFrame(rates_h1)
            indicators = self.calculate_indicators(df_h1)
            entry_exit_analysis = self.analyze_entry_exit_points(indicators, current_price, symbol)
            
            # Multi-timeframe prices
            h4_price = rates_h4[-1]['close'] if len(rates_h4) > 0 else current_price
            h1_price = rates_h1[-1]['close'] if len(rates_h1) > 0 else current_price
            m15_price = rates_m15[-1]['close'] if len(rates_m15) > 0 else current_price
            m5_price = rates_m5[-1]['close'] if len(rates_m5) > 0 else current_price
            
            # Price change
            previous_price = df_h1['close'].iloc[-2] if len(df_h1) > 1 else current_price
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            # Spread
            spread = tick.ask - tick.bid
            spread_pips = spread * (10000 if 'JPY' not in symbol else 100)
            
            # Trading status for this pair
            pair_status = self.check_pair_trading_status(symbol)
            
            return {
                'symbol': symbol,
                'h4': round(h4_price, 5),
                'h1': round(h1_price, 5),
                'm15': round(m15_price, 5),
                'm5': round(m5_price, 5),
                'current_price': round(current_price, 5),
                'price_change': round(price_change, 5),
                'change_percent': round(change_percent, 3),
                'price_direction': 'up' if m5_price > m15_price else 'down',
                'bid': tick.bid,
                'ask': tick.ask,
                'spread_pips': round(spread_pips, 1),
                
                # Technical indicators
                'rsi': round(indicators.get('rsi', 50), 1),
                'macd': round(indicators.get('macd', 0), 6),
                'atrPercent': round(indicators.get('atr_percent', 0), 3),
                'trendStrength': round(indicators.get('trend_strength', 0), 3),
                'volumeRatio': round(indicators.get('volume_ratio', 1), 2),
                
                # Trading signals and analysis
                **entry_exit_analysis,
                
                # Auto trading info
                'pair_status': pair_status,
                'can_trade': pair_status['can_trade'],
                'active_trades': len(self.active_trades_per_pair.get(symbol, [])),
                
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def update_all_data(self):
        """Update all symbols data"""
        try:
            if not self.mt5_connected:
                return
            
            updated_count = 0
            for symbol in self.forex_pairs:
                try:
                    data = self.get_symbol_data(symbol)
                    if data:
                        self.live_data[symbol] = data
                        updated_count += 1
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error updating {symbol}: {str(e)}")
                    continue
            
            self.last_update = datetime.now()
            self.logger.info(f"Updated {updated_count}/{len(self.forex_pairs)} pairs")
            
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
    
    def setup_routes(self):
        """Setup Flask routes for auto trading control"""
        
        @self.app.route('/')
        def dashboard():
            # ใช้ชื่อไฟล์เดิมที่คุณมี
            try:
                return send_from_directory('.', 'forex_dashboard.html')
            except:
                # ถ้าไม่เจอ ให้ลองหาไฟล์อื่น
                try:
                    return send_from_directory('.', 'auto_trading_dashboard.html')
                except:
                    return '''<!DOCTYPE html>
<html><head><title>Auto Trading Dashboard</title></head>
<body style="background:#000;color:#fff;font-family:monospace;padding:2rem;">
<h1 style="color:#00ff00;">Smart Auto Trading Dashboard</h1>
<p style="color:#ffff00;">Please save the HTML code as 'forex_dashboard.html' in the same directory as the Python script.</p>
<p style="color:#ff6666;">Current directory: ''' + os.getcwd() + '''</p>
<br><a href="/api/market-data" style="color:#00ccff;">API Test - Market Data</a>
</body></html>'''
        
        @self.app.route('/api/market-data')
        def get_market_data():
            """Get market data with auto trading status"""
            try:
                formatted_data = {}
                for symbol, data in self.live_data.items():
                    if data:
                        formatted_data[symbol] = data
                
                # Auto trading status
                auto_trading_status = {
                    'enabled': self.auto_trading_enabled,
                    'emergency_stop': self.emergency_stop,
                    'active_pairs': list(self.auto_trading_pairs),
                    'total_trades': len([pos for positions in self.active_trades_per_pair.values() for pos in positions]),
                    'daily_stats': self.daily_stats,
                    'current_exposure': self.calculate_current_exposure() * 100,
                    'max_exposure': self.max_total_exposure * 100,
                    'risk_profile': self.current_risk_profile,
                    'custom_risk': self.custom_risk_per_trade
                }
                
                return jsonify({
                    'success': True,
                    'data': formatted_data,
                    'account': self.account_info,
                    'auto_trading': auto_trading_status,
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
        
        @self.app.route('/api/auto-trading/start', methods=['POST'])
        def start_auto_trading_api():
            """Start auto trading"""
            if self.start_auto_trading():
                return jsonify({'success': True, 'message': 'Auto trading started'})
            else:
                return jsonify({'success': False, 'error': 'Failed to start auto trading'})
        
        @self.app.route('/api/auto-trading/stop', methods=['POST'])
        def stop_auto_trading_api():
            """Stop auto trading"""
            self.stop_auto_trading()
            return jsonify({'success': True, 'message': 'Auto trading stopped'})
        
        @self.app.route('/api/auto-trading/emergency-stop', methods=['POST'])
        def emergency_stop_api():
            """Emergency stop all trading"""
            self.emergency_stop_all()
            return jsonify({'success': True, 'message': 'Emergency stop activated'})
        
        @self.app.route('/api/auto-trading/settings', methods=['POST'])
        def update_auto_trading_settings():
            """Update auto trading settings"""
            try:
                data = request.get_json()
                
                if 'risk_profile' in data:
                    self.update_risk_profile(data['risk_profile'])
                
                if 'custom_risk' in data:
                    self.update_risk_profile(custom_risk=data['custom_risk'])
                
                if 'min_signal_strength' in data:
                    self.min_signal_strength = float(data['min_signal_strength'])
                    self.logger.info(f"Min signal strength updated to: {self.min_signal_strength}")
                
                if 'min_entry_quality' in data:
                    self.min_entry_quality = data['min_entry_quality']
                    self.logger.info(f"Min entry quality updated to: {self.min_entry_quality}")
                
                if 'enabled_pairs' in data:
                    self.auto_trading_pairs = set(data['enabled_pairs'])
                
                if 'max_simultaneous_trades' in data:
                    self.max_simultaneous_trades = int(data['max_simultaneous_trades'])
                
                # New signal settings
                if 'require_trend_alignment' in data:
                    self.required_confirmations['trend_alignment'] = data['require_trend_alignment']
                
                if 'require_volume_confirmation' in data:
                    self.required_confirmations['volume_confirmation'] = data['require_volume_confirmation']
                
                if 'require_rsi_filter' in data:
                    self.required_confirmations['rsi_filter'] = data['require_rsi_filter']
                
                if 'require_multiple_timeframe' in data:
                    self.required_confirmations['multiple_timeframe'] = data['require_multiple_timeframe']
                
                if 'min_rr_ratio' in data:
                    self.min_rr_ratio = float(data['min_rr_ratio'])
                
                return jsonify({
                    'success': True, 
                    'message': 'Settings updated',
                    'current_settings': {
                        'min_signal_strength': self.min_signal_strength,
                        'min_entry_quality': self.min_entry_quality,
                        'confirmations': self.required_confirmations,
                        'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5)
                    }
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/auto-trading/current-settings')
        def get_current_settings():
            """Get current auto trading settings"""
            try:
                return jsonify({
                    'success': True,
                    'settings': {
                        'min_signal_strength': self.min_signal_strength,
                        'min_entry_quality': self.min_entry_quality,
                        'max_simultaneous_trades': self.max_simultaneous_trades,
                        'confirmations': self.required_confirmations,
                        'min_rr_ratio': getattr(self, 'min_rr_ratio', 1.5),
                        'auto_trading_enabled': self.auto_trading_enabled,
                        'risk_profile': self.current_risk_profile,
                        'custom_risk': self.custom_risk_per_trade
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            try:
                positions = mt5.positions_get()
                if positions is None:
                    positions = []
                
                formatted_positions = []
                for pos in positions:
                    formatted_positions.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit,
                        'time': pos.time
                    })
                
                return jsonify({
                    'success': True,
                    'positions': formatted_positions,
                    'total_positions': len(formatted_positions)
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/refresh')
        def refresh_data():
            """Manual refresh"""
            try:
                self.update_all_data()
                return jsonify({
                    'success': True,
                    'message': 'Data refreshed',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def start_data_updates(self):
        """Start automatic data updates"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(15)
                except Exception as e:
                    self.logger.error(f"Update loop error: {str(e)}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.logger.info("Auto-update system started")
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the auto trading dashboard"""
        try:
            print("Smart Auto Trading Dashboard Starting...")
            print("=" * 60)
            
            if not self.connect_mt5():
                print("ERROR: Failed to connect to MT5")
                return
            
            self.is_running = True
            self.start_data_updates()
            self.update_all_data()
            
            print(f"SUCCESS: Auto Trading Dashboard Started!")
            print(f"FEATURES: Smart Auto Trading + Risk Management")
            print(f"PORTFOLIO: One Trade Per Pair + Risk Profiles")
            print(f"DASHBOARD: http://{host}:{port}")
            print(f"API: http://{host}:{port}/api/market-data")
            print(f"AUTO TRADING: Currently {('ENABLED' if self.auto_trading_enabled else 'DISABLED')}")
            print("CONTROLS: Use dashboard to start/stop auto trading")
            print("STOP: Press Ctrl+C to stop")
            print("=" * 60)
            
            self.app.run(host=host, port=port, debug=False, threaded=True)
            
        except KeyboardInterrupt:
            print("\nSTOPPING Auto Trading Dashboard...")
            self.is_running = False
            self.auto_trading_enabled = False
            if self.mt5_connected:
                mt5.shutdown()
            print("STOPPED successfully")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.is_running = False
            if self.mt5_connected:
                mt5.shutdown()

def main():
    """Main execution"""
    print("Smart Auto Trading Dashboard")
    print("============================")
    print("Features:")
    print("- Auto Trading with Signal Validation")
    print("- One Trade Per Pair Control")
    print("- Portfolio Risk Profiles")
    print("- Real-time Position Management")
    print("- Emergency Stop Controls")
    print()
    
    dashboard = SmartAutoTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()