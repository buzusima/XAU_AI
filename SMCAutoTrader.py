# Professional SMC Auto Trader with Advanced Logging
# Created by อาจารย์ฟินิกซ์ - Clean, Professional Code

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import json
import time
import os
import csv
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

warnings.filterwarnings("ignore")

# Import our Signal Engine
from smc_signal_engine import SMCSignalEngine


class ProfessionalSMCAutoTrader:
    """
    Professional SMC Auto Trading Bot with Advanced Logging
    
    Features:
    - Clean, professional code structure
    - Comprehensive error handling
    - Advanced logging system
    - Risk management
    - MT5 integration with proper order handling
    """

    def __init__(
        self,
        models_path: str = "XAUUSD_v_SMC",
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        # Signal settings
        signal_change_threshold: float = 0.001,
        enable_first_signal_trade: bool = True,
        first_signal_min_confidence: float = 0.75,
        # Risk management
        max_risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
        max_concurrent_trades: int = 1,
        min_confidence: float = 0.75,
        min_consensus: int = 3,
        # Position sizing
        base_lot_size: float = 0.01,
        max_lot_size: float = 0.1,
        lot_multiplier: float = 2.0,
        # Trade management
        default_sl_pips: int = 20,
        default_tp_ratio: float = 2.0,
        max_trades_per_hour: int = 5,
        wait_for_trade_completion: bool = True,
        # Logging
        enable_logging: bool = True,
        log_directory: str = "trading_logs"
    ):
        """Initialize Professional SMC Auto Trader"""
        
        # Connection settings
        self.account = account
        self.password = password
        self.server = server
        self.timezone = pytz.timezone("Etc/UTC")
        
        # Signal settings
        self.signal_change_threshold = signal_change_threshold
        self.enable_first_signal_trade = enable_first_signal_trade
        self.first_signal_min_confidence = first_signal_min_confidence
        
        # Risk management
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_concurrent_trades = max_concurrent_trades
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        
        # Position sizing
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.lot_multiplier = lot_multiplier
        
        # Trade management
        self.default_sl_pips = default_sl_pips
        self.default_tp_ratio = default_tp_ratio
        self.max_trades_per_hour = max_trades_per_hour
        self.wait_for_trade_completion = wait_for_trade_completion
        
        # Trading state
        self.trading_enabled = False
        self.active_positions: Dict[int, Dict] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.hourly_trade_count = 0
        self.hour_start = datetime.now().hour
        self.last_trade_closed_time: Optional[datetime] = None
        
        # Logging system
        self.enable_logging = enable_logging
        self.log_directory = log_directory
        self.signal_history: List[Dict] = []
        self.trade_performance: List[Dict] = []
        self.model_training_data: List[Dict] = []
        
        # Initialize systems
        self._setup_logging()
        self.signal_engine = SMCSignalEngine(models_path)
        
        self._print_banner()

    def _print_banner(self):
        """Print professional banner"""
        print("=" * 80)
        print("🤖 PROFESSIONAL SMC AUTO TRADING BOT")
        print("Created by อาจารย์ฟินิกซ์ - Advanced AI Trading System")
        print("=" * 80)
        print(f"📊 Logging: {'ENABLED' if self.enable_logging else 'DISABLED'}")
        print(f"🎯 Trading: DISABLED (Enable manually for safety)")
        print(f"⚡ Status: Ready for Professional Trading")
        print("=" * 80)

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        if not self.enable_logging:
            return
            
        try:
            # Create directories
            os.makedirs(self.log_directory, exist_ok=True)
            os.makedirs(f"{self.log_directory}/signals", exist_ok=True)
            os.makedirs(f"{self.log_directory}/trades", exist_ok=True)
            os.makedirs(f"{self.log_directory}/performance", exist_ok=True)
            os.makedirs(f"{self.log_directory}/training_data", exist_ok=True)
            
            # Setup file paths
            today = datetime.now().strftime("%Y%m%d")
            self.signal_log_file = f"{self.log_directory}/signals/signals_{today}.csv"
            self.trade_log_file = f"{self.log_directory}/trades/trades_{today}.csv"
            self.performance_log_file = f"{self.log_directory}/performance/performance_{today}.json"
            
            # Initialize signal log
            if not os.path.exists(self.signal_log_file):
                with open(self.signal_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'symbol', 'direction', 'confidence', 'risk_level',
                        'recommendation', 'price', 'consensus', 'trade_executed', 'reason'
                    ])
            
            # Initialize trade log
            if not os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'entry_time', 'exit_time', 'symbol', 'direction', 'volume',
                        'entry_price', 'exit_price', 'sl', 'tp', 'pnl', 'pips',
                        'duration_min', 'ticket', 'result'
                    ])
            
            print(f"📁 Logging system initialized: {self.log_directory}")
            
        except Exception as e:
            print(f"❌ Logging setup error: {e}")
            self.enable_logging = False

    def connect_mt5(self) -> bool:
        """Connect to MT5 with proper error handling"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if all([self.account, self.password, self.server]):
                if not mt5.login(self.account, password=self.password, server=self.server):
                    print(f"❌ Login failed: {mt5.last_error()}")
                    return False
            
            # Verify account
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Failed to get account info")
                return False
            
            if not account_info.trade_allowed:
                print("❌ Trading not allowed on this account")
                return False
            
            # Success
            print("✅ MT5 Connected Successfully")
            print(f"📊 Account: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance:,.2f}")
            print(f"🏦 Server: {account_info.server}")
            print(f"🏢 Company: {account_info.company}")
            
            return True
            
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False

    def load_models(self) -> bool:
        """Load AI models with proper validation"""
        try:
            print("🔄 Loading AI Models...")
            success = self.signal_engine.load_trained_models()
            
            if success:
                print("✅ AI Models loaded successfully")
                return True
            else:
                print("❌ Failed to load AI models")
                return False
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False

    def debug_symbol_issues(self, symbol: str):
        """Debug symbol-related issues comprehensively"""
        print(f"\n🔍 COMPREHENSIVE SYMBOL DEBUG: {symbol}")
        print("=" * 60)
        
        try:
            # 1. Check if symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol '{symbol}' not found")
                
                # Search for similar symbols
                print("\n🔍 Searching for similar symbols...")
                symbols = mt5.symbols_get()
                if symbols:
                    gold_symbols = []
                    for s in symbols:
                        if any(keyword in s.name.upper() for keyword in ['XAU', 'GOLD', 'GOL']):
                            gold_symbols.append(s.name)
                    
                    if gold_symbols:
                        print(f"📊 Found Gold-related symbols:")
                        for gs in gold_symbols:
                            print(f"   - {gs}")
                        
                        print(f"\n💡 Try using: {gold_symbols[0]} instead of {symbol}")
                return False
            
            # 2. Symbol exists - check details
            print(f"✅ Symbol found: {symbol_info.description}")
            print(f"   Visible: {symbol_info.visible}")
            print(f"   Selected: {symbol_info.select}")
            
            # 3. Enable symbol if not selected
            if not symbol_info.select:
                print("🔧 Symbol not selected, attempting to select...")
                if mt5.symbol_select(symbol, True):
                    print("✅ Symbol selected successfully")
                else:
                    print("❌ Failed to select symbol")
                    return False
            
            # 4. Check trading permissions
            trade_modes = {
                0: "Disabled",
                1: "Long only",
                2: "Short only", 
                4: "Full trading"
            }
            
            print(f"\n📊 TRADING PERMISSIONS:")
            print(f"   Trade Mode: {trade_modes.get(symbol_info.trade_mode, 'Unknown')} ({symbol_info.trade_mode})")
            
            if symbol_info.trade_mode == 0:
                print("❌ Trading is completely disabled for this symbol")
                return False
            
            # 5. Check market hours
            print(f"\n⏰ MARKET STATUS:")
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print("❌ No tick data available - market might be closed")
                return False
            else:
                print(f"✅ Tick data available")
                print(f"   Bid: {tick.bid}")
                print(f"   Ask: {tick.ask}")
                print(f"   Time: {datetime.fromtimestamp(tick.time)}")
            
            # 6. Check account permissions
            account_info = mt5.account_info()
            if account_info:
                print(f"\n👤 ACCOUNT STATUS:")
                print(f"   Trade Allowed: {account_info.trade_allowed}")
                print(f"   Trade Expert: {account_info.trade_expert}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Equity: ${account_info.equity:.2f}")
                print(f"   Margin Free: ${account_info.margin_free:.2f}")
                print(f"   Margin Level: {account_info.margin_level:.2f}%")
                
                if not account_info.trade_allowed:
                    print("❌ Trading not allowed on account")
                    return False
                
                if not account_info.trade_expert:
                    print("❌ Expert Advisor trading disabled")
                    return False
            
            # 7. Volume requirements
            print(f"\n📊 VOLUME REQUIREMENTS:")
            print(f"   Min Volume: {symbol_info.volume_min}")
            print(f"   Max Volume: {symbol_info.volume_max}")
            print(f"   Volume Step: {symbol_info.volume_step}")
            
            # 8. Margin requirements
            print(f"\n💰 MARGIN REQUIREMENTS:")
            margin_initial = getattr(symbol_info, 'margin_initial', 'Not available')
            margin_maintenance = getattr(symbol_info, 'margin_maintenance', 'Not available')
            print(f"   Initial Margin: {margin_initial}")
            print(f"   Maintenance Margin: {margin_maintenance}")
            
            # 9. Price restrictions
            print(f"\n💱 PRICE RESTRICTIONS:")
            print(f"   Point: {symbol_info.point}")
            print(f"   Digits: {symbol_info.digits}")
            
            # Check if stops_level exists (some MT5 versions don't have it)
            stops_level = getattr(symbol_info, 'stops_level', 'Not available')
            freeze_level = getattr(symbol_info, 'freeze_level', 'Not available')
            print(f"   Stops Level: {stops_level}")
            print(f"   Freeze Level: {freeze_level}")
            
            # 10. Execution modes
            execution_modes = {
                0: "Request",
                1: "Instant", 
                2: "Market",
                3: "Exchange"
            }
            execution_mode = getattr(symbol_info, 'execution_mode', None)
            if execution_mode is not None:
                print(f"   Execution Mode: {execution_modes.get(execution_mode, 'Unknown')} ({execution_mode})")
            else:
                print(f"   Execution Mode: Not available")
            
            # Check filling modes
            filling_mode = getattr(symbol_info, 'filling_mode', 0)
            print(f"   Filling Mode Value: {filling_mode}")
            
            filling_modes = []
            if filling_mode & 1:
                filling_modes.append("IOC")
            if filling_mode & 2:
                filling_modes.append("FOK") 
            if filling_mode & 4:
                filling_modes.append("RETURN")
            
            if filling_modes:
                print(f"   Available Filling Modes: {', '.join(filling_modes)}")
            else:
                print(f"   Available Filling Modes: None detected")
            
            # 11. Test a small order (dry run)
            print(f"\n🧪 ORDER VALIDATION TEST:")
            test_volume = symbol_info.volume_min
            
            # Determine available filling mode for testing
            filling_mode = getattr(symbol_info, 'filling_mode', 0)
            if filling_mode & 2:  # FOK available
                test_filling = mt5.ORDER_FILLING_FOK
                filling_name = "FOK"
            elif filling_mode & 1:  # IOC available
                test_filling = mt5.ORDER_FILLING_IOC  
                filling_name = "IOC"
            else:  # Default to RETURN
                test_filling = mt5.ORDER_FILLING_RETURN
                filling_name = "RETURN"
            
            print(f"   Using filling mode: {filling_name}")
            
            # Test request structure
            test_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": test_volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "TEST_ORDER",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": test_filling,
            }
            
            print(f"   Test Volume: {test_volume}")
            print(f"   Test Price: {tick.ask}")
            
            # Check what would happen (don't actually send)
            try:
                result = mt5.order_check(test_request)
                if result is None:
                    print("❌ Order check failed - result is None")
                else:
                    print(f"   Check Result: {result.retcode}")
                    if hasattr(result, 'comment'):
                        print(f"   Comment: {result.comment}")
                    if result.retcode == 0:
                        print("✅ Order structure is valid")
                    else:
                        print(f"❌ Order validation failed: {result.retcode}")
                        
                        # Show what went wrong
                        error_meanings = {
                            10004: "Requote",
                            10014: "Invalid volume",
                            10015: "Invalid price", 
                            10016: "Invalid stops",
                            10017: "Trade disabled",
                            10019: "Not enough money",
                            10030: "Invalid filling type"
                        }
                        
                        if result.retcode in error_meanings:
                            print(f"   Meaning: {error_meanings[result.retcode]}")
                            
            except Exception as e:
                print(f"❌ Order check exception: {e}")
            
            # 12. Show all available attributes
            print(f"\n📋 ALL SYMBOL ATTRIBUTES:")
            symbol_attrs = dir(symbol_info)
            relevant_attrs = []
            for attr in symbol_attrs:
                if not attr.startswith('_') and not callable(getattr(symbol_info, attr, None)):
                    try:
                        value = getattr(symbol_info, attr)
                        relevant_attrs.append(f"   {attr}: {value}")
                    except:
                        relevant_attrs.append(f"   {attr}: Unable to read")
            
            # Show first 20 attributes to avoid flooding
            for attr_info in relevant_attrs[:20]:
                print(attr_info)
            
            if len(relevant_attrs) > 20:
                print(f"   ... and {len(relevant_attrs) - 20} more attributes")
            
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_symbol_info(self, symbol: str) -> bool:
        """Check symbol information and trading permissions"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not found")
                return False
            
            print(f"\n🔍 Symbol Info: {symbol}")
            print("-" * 40)
            print(f"Description: {symbol_info.description}")
            print(f"Volume: {symbol_info.volume_min} - {symbol_info.volume_max}")
            print(f"Volume Step: {symbol_info.volume_step}")
            print(f"Contract Size: {symbol_info.trade_contract_size}")
            print(f"Point: {symbol_info.point}")
            print(f"Digits: {symbol_info.digits}")
            print(f"Spread: {symbol_info.spread}")
            
            # Check trading permissions
            if symbol_info.trade_mode == 0:
                print("❌ Trading disabled for this symbol")
                return False
            
            # Check filling modes
            filling_modes = []
            if symbol_info.filling_mode & 1:
                filling_modes.append("IOC")
            if symbol_info.filling_mode & 2:
                filling_modes.append("FOK")
            if symbol_info.filling_mode & 4:
                filling_modes.append("RETURN")
            
            print(f"Filling Modes: {', '.join(filling_modes)}")
            
            # Check current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print("❌ Cannot get current price")
                return False
            
            print(f"Current Price: {tick.bid} / {tick.ask}")
            print("-" * 40)
            
            return True
            
        except Exception as e:
            print(f"❌ Symbol info error: {e}")
            return False

    def enable_trading(self, enable: bool = True):
        """Enable or disable trading with safety checks"""
        if enable:
            print("⚠️  ENABLING LIVE TRADING")
            print("⚠️  WARNING: This will execute real trades with real money!")
            print("⚠️  Make sure you understand the risks involved.")
            
            confirm = input("Type 'CONFIRM' to enable live trading: ")
            if confirm != 'CONFIRM':
                print("❌ Trading NOT enabled")
                return
        
        self.trading_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🎯 Trading {status}")

    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal position size based on confidence and risk"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return self.base_lot_size
            
            # Base calculation
            calculated_lot = self.base_lot_size
            
            # Adjust for confidence
            if confidence >= 0.9:
                calculated_lot *= self.lot_multiplier
            elif confidence >= 0.8:
                calculated_lot *= 1.5
            
            # Adjust for symbol type (Gold has higher volatility)
            if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
                calculated_lot *= 0.5  # Reduce for Gold
            
            # Apply constraints
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, self.max_lot_size)
            lot_step = symbol_info.volume_step
            
            # Round to valid step
            if lot_step > 0:
                calculated_lot = round(calculated_lot / lot_step) * lot_step
            
            # Final validation
            final_lot = max(min_lot, min(max_lot, calculated_lot))
            
            return final_lot
            
        except Exception as e:
            print(f"❌ Position size calculation error: {e}")
            return self.base_lot_size

    def calculate_sl_tp_levels(self, symbol: str, order_type: int, entry_price: float) -> Tuple[float, float]:
        """Calculate Stop Loss and Take Profit levels"""
        try:
            # Determine pip size based on symbol
            if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
                pip_size = 0.01
                sl_pips = self.default_sl_pips * 2  # Wider SL for Gold
            elif "JPY" in symbol.upper():
                pip_size = 0.01
                sl_pips = self.default_sl_pips
            else:
                pip_size = 0.0001
                sl_pips = self.default_sl_pips
            
            sl_distance = sl_pips * pip_size
            tp_distance = sl_distance * self.default_tp_ratio
            
            if order_type == mt5.ORDER_TYPE_BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            print(f"❌ SL/TP calculation error: {e}")
            return 0.0, 0.0

    def send_order(self, symbol: str, order_type: int, lot_size: float, 
                   stop_loss: float = 0.0, take_profit: float = 0.0, 
                   comment: str = "SMC_AI") -> bool:
        """Send order with comprehensive error handling and debugging"""
        
        if not self.trading_enabled:
            print("⚠️ Trading disabled - order not sent")
            return False
        
        try:
            print(f"\n🔄 ORDER PREPARATION:")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
            print(f"   Volume: {lot_size}")
            print(f"   SL: {stop_loss}")
            print(f"   TP: {take_profit}")
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not found")
                # Try to find similar symbols
                symbols = mt5.symbols_get()
                if symbols:
                    gold_symbols = [s.name for s in symbols if 'XAU' in s.name.upper() or 'GOLD' in s.name.upper()]
                    if gold_symbols:
                        print(f"💡 Available Gold symbols: {gold_symbols}")
                return False
            
            print(f"✅ Symbol info retrieved:")
            print(f"   Description: {symbol_info.description}")
            print(f"   Trade Mode: {symbol_info.trade_mode}")
            print(f"   Volume Min/Max: {symbol_info.volume_min}/{symbol_info.volume_max}")
            print(f"   Volume Step: {symbol_info.volume_step}")
            print(f"   Filling Mode: {symbol_info.filling_mode}")
            print(f"   Execution Mode: {symbol_info.execution_mode}")
            
            # Check if trading is allowed
            if symbol_info.trade_mode == 0:
                print("❌ Trading is disabled for this symbol")
                return False
            
            # Validate and adjust lot size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            print(f"\n🔧 LOT SIZE VALIDATION:")
            print(f"   Original: {lot_size}")
            
            if lot_size < min_lot:
                lot_size = min_lot
                print(f"   Adjusted to minimum: {lot_size}")
            elif lot_size > max_lot:
                lot_size = max_lot
                print(f"   Adjusted to maximum: {lot_size}")
            
            if lot_step > 0:
                lot_size = round(lot_size / lot_step) * lot_step
                print(f"   Rounded to step: {lot_size}")
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print("❌ Cannot get current price")
                return False
            
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            order_type_str = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            
            print(f"\n💰 CURRENT PRICES:")
            print(f"   Bid: {tick.bid}")
            print(f"   Ask: {tick.ask}")
            print(f"   Spread: {tick.ask - tick.bid}")
            print(f"   Using Price: {price}")
            
            # Determine best filling mode
            filling_mode = symbol_info.filling_mode
            filling_modes_available = []
            
            if filling_mode & 1:
                filling_modes_available.append(("IOC", mt5.ORDER_FILLING_IOC))
            if filling_mode & 2:
                filling_modes_available.append(("FOK", mt5.ORDER_FILLING_FOK))
            if filling_mode & 4:
                filling_modes_available.append(("RETURN", mt5.ORDER_FILLING_RETURN))
            
            print(f"\n🎯 FILLING MODES AVAILABLE:")
            for mode_name, mode_value in filling_modes_available:
                print(f"   {mode_name}: {mode_value}")
            
            if not filling_modes_available:
                print("❌ No filling modes available")
                return False
            
            # Try each filling mode
            for mode_name, filling_type in filling_modes_available:
                print(f"\n🔄 TRYING {mode_name} FILLING MODE:")
                
                # Check account info
                account_info = mt5.account_info()
                if account_info is None:
                    print("❌ Cannot get account info")
                    continue
                
                print(f"   Account Balance: ${account_info.balance:.2f}")
                print(f"   Account Equity: ${account_info.equity:.2f}")
                print(f"   Account Margin Free: ${account_info.margin_free:.2f}")
                
                # Calculate required margin
                margin_initial = getattr(symbol_info, 'margin_initial', None)
                if margin_initial and margin_initial > 0:
                    required_margin = lot_size * margin_initial
                    print(f"   Required Margin: ${required_margin:.2f}")
                    
                    if required_margin > account_info.margin_free:
                        print("❌ Insufficient margin")
                        continue
                else:
                    print("   Margin calculation: Not available")
                
                # Create order request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_type,
                }
                
                # Add SL/TP only if they are valid
                if stop_loss > 0:
                    # Validate SL distance (use default if stops_level not available)
                    min_distance = getattr(symbol_info, 'stops_level', 10) * symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY:
                        sl_distance = price - stop_loss
                    else:
                        sl_distance = stop_loss - price
                    
                    if sl_distance >= min_distance:
                        request["sl"] = stop_loss
                        print(f"   SL added: {stop_loss}")
                    else:
                        print(f"   SL too close, skipping (min: {min_distance})")
                
                if take_profit > 0:
                    # Validate TP distance (use default if stops_level not available)
                    min_distance = getattr(symbol_info, 'stops_level', 10) * symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY:
                        tp_distance = take_profit - price
                    else:
                        tp_distance = price - take_profit
                    
                    if tp_distance >= min_distance:
                        request["tp"] = take_profit
                        print(f"   TP added: {take_profit}")
                    else:
                        print(f"   TP too close, skipping (min: {min_distance})")
                
                print(f"   📋 Final Request: {request}")
                
                # Send order
                result = mt5.order_send(request)
                
                print(f"   📤 Order Result:")
                print(f"      Retcode: {result.retcode}")
                print(f"      Comment: {result.comment}")
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success!
                    trade_info = {
                        "timestamp": datetime.now(self.timezone),
                        "symbol": symbol,
                        "type": order_type_str,
                        "volume": lot_size,
                        "price": result.price,
                        "sl": request.get("sl", 0),
                        "tp": request.get("tp", 0),
                        "ticket": result.order,
                        "comment": comment,
                    }
                    
                    self.trade_history.append(trade_info)
                    self.active_positions[result.order] = trade_info
                    
                    # Format price display
                    if "XAU" in symbol.upper():
                        price_str = f"{result.price:.2f}"
                        sl_str = f"{request.get('sl', 0):.2f}" if request.get('sl', 0) > 0 else "None"
                        tp_str = f"{request.get('tp', 0):.2f}" if request.get('tp', 0) > 0 else "None"
                    else:
                        price_str = f"{result.price:.5f}"
                        sl_str = f"{request.get('sl', 0):.5f}" if request.get('sl', 0) > 0 else "None"
                        tp_str = f"{request.get('tp', 0):.5f}" if request.get('tp', 0) > 0 else "None"
                    
                    print(f"\n✅ ORDER SUCCESSFUL!")
                    print(f"   {order_type_str} {lot_size} {symbol} @ {price_str}")
                    print(f"   🎯 SL: {sl_str} | TP: {tp_str}")
                    print(f"   🎫 Ticket: {result.order}")
                    print(f"   💰 Order Value: ${result.volume * result.price:.2f}")
                    
                    return True
                    
                else:
                    # Failed
                    print(f"   ❌ Order failed with {mode_name}: {result.retcode} - {result.comment}")
                    
                    # Common error codes
                    error_meanings = {
                        10004: "Requote",
                        10006: "Request rejected",
                        10007: "Request canceled by trader",
                        10008: "Order placed",
                        10009: "Request completed",
                        10010: "Only part of the request was completed",
                        10011: "Request processing error",
                        10012: "Request canceled by timeout",
                        10013: "Invalid request",
                        10014: "Invalid volume in the request",
                        10015: "Invalid price in the request",
                        10016: "Invalid stops in the request",
                        10017: "Trade is disabled",
                        10018: "Market is closed",
                        10019: "There is not enough money to complete the request",
                        10020: "Prices changed",
                        10021: "There are no quotes to process the request",
                        10022: "Invalid order expiration date in the request",
                        10023: "Order state changed",
                        10024: "Too frequent requests",
                        10025: "No changes in request",
                        10026: "Autotrading disabled by server",
                        10027: "Autotrading disabled by client terminal",
                        10028: "Request locked for processing",
                        10029: "Order or position frozen",
                        10030: "Invalid order filling type",
                        10031: "No connection with the trade server"
                    }
                    
                    if result.retcode in error_meanings:
                        print(f"   📖 Meaning: {error_meanings[result.retcode]}")
                    
                    # Specific handling for common errors
                    if result.retcode == 10030:  # Invalid filling type
                        print(f"   🔄 Filling type {mode_name} not supported, trying next...")
                        continue
                    elif result.retcode == 10017:  # Trade disabled
                        print(f"   ❌ Trading is disabled")
                        return False
                    elif result.retcode == 10019:  # Not enough money
                        print(f"   ❌ Insufficient funds")
                        return False
                    elif result.retcode == 10031:  # No connection
                        print(f"   ❌ No connection to trade server")
                        return False
            
            print(f"\n❌ ALL FILLING MODES FAILED")
            return False
            
        except Exception as e:
            print(f"❌ Order execution exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def log_signal(self, signal: Dict, symbol: str, trade_executed: bool = False, reason: str = ""):
        """Log signal data for analysis"""
        if not self.enable_logging:
            return
        
        try:
            log_entry = [
                datetime.now().isoformat(),
                symbol,
                signal.get('final_direction', 'HOLD'),
                signal.get('final_confidence', 0),
                signal.get('risk_level', 'HIGH'),
                signal.get('trading_recommendation', 'WAIT'),
                signal.get('current_price', 0),
                signal.get('timeframe_consensus', '0L/0S/5H'),
                trade_executed,
                reason
            ]
            
            with open(self.signal_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(log_entry)
            
            print(f"📊 Signal logged: {signal['final_direction']} ({signal['final_confidence']:.3f})")
            
        except Exception as e:
            print(f"❌ Signal logging error: {e}")

    def log_trade_exit(self, ticket: int, exit_price: float, pnl: float):
        """Log trade exit with performance metrics"""
        if not self.enable_logging or ticket not in self.active_positions:
            return
        
        try:
            trade_info = self.active_positions[ticket]
            
            # Calculate metrics
            entry_time = trade_info['timestamp']
            exit_time = datetime.now(self.timezone)
            duration = (exit_time - entry_time).total_seconds() / 60  # minutes
            
            # Calculate pips
            symbol = trade_info['symbol']
            entry_price = trade_info['price']
            
            if "XAU" in symbol.upper():
                pip_value = 0.01
            elif "JPY" in symbol.upper():
                pip_value = 0.01
            else:
                pip_value = 0.0001
            
            if trade_info['type'] == 'BUY':
                pips = (exit_price - entry_price) / pip_value
            else:
                pips = (entry_price - exit_price) / pip_value
            
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
            
            # Log trade
            trade_log_entry = [
                entry_time.isoformat(),
                exit_time.isoformat(),
                symbol,
                trade_info['type'],
                trade_info['volume'],
                entry_price,
                exit_price,
                trade_info['sl'],
                trade_info['tp'],
                pnl,
                pips,
                duration,
                ticket,
                result
            ]
            
            with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(trade_log_entry)
            
            print(f"📉 Trade logged: {result} ${pnl:.2f} ({pips:.1f} pips)")
            
        except Exception as e:
            print(f"❌ Trade logging error: {e}")

    def update_positions(self):
        """Update active positions and handle closed trades"""
        try:
            # Get current positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            current_tickets = [pos.ticket for pos in positions]
            
            # Check for closed positions
            for ticket in list(self.active_positions.keys()):
                if ticket not in current_tickets:
                    # Position closed
                    closed_trade = self.active_positions[ticket]
                    
                    # Get closing details from history
                    deals = mt5.history_deals_get(
                        datetime.now() - timedelta(hours=1), 
                        datetime.now()
                    )
                    
                    if deals:
                        for deal in deals:
                            if deal.position_id == ticket and deal.entry == 1:  # Exit deal
                                # Log the closed trade
                                self.log_trade_exit(ticket, deal.price, deal.profit)
                                
                                # Display result
                                symbol = closed_trade['symbol']
                                if "XAU" in symbol.upper():
                                    entry_str = f"{closed_trade['price']:.2f}"
                                    exit_str = f"{deal.price:.2f}"
                                else:
                                    entry_str = f"{closed_trade['price']:.5f}"
                                    exit_str = f"{deal.price:.5f}"
                                
                                result_icon = "✅" if deal.profit > 0 else "❌"
                                print(f"📈 Trade #{ticket} CLOSED:")
                                print(f"   {closed_trade['type']} {closed_trade['volume']} {symbol}")
                                print(f"   Entry: {entry_str} → Exit: {exit_str}")
                                print(f"   {result_icon} P&L: ${deal.profit:.2f}")
                                break
                    
                    # Remove from active positions
                    del self.active_positions[ticket]
                    self.last_trade_closed_time = datetime.now()
            
            # Update daily P&L
            self.daily_pnl = sum(pos.profit for pos in positions)
            
        except Exception as e:
            print(f"❌ Position update error: {e}")

    def should_analyze_signals(self) -> bool:
        """Check if system should analyze new signals"""
        if not self.wait_for_trade_completion:
            return True
        
        return len(self.active_positions) == 0

    def check_risk_limits(self) -> bool:
        """Check if trading is within risk limits"""
        # Check concurrent trades
        if len(self.active_positions) >= self.max_concurrent_trades:
            return False
        
        # Check hourly limit
        current_hour = datetime.now().hour
        if current_hour != self.hour_start:
            self.hourly_trade_count = 0
            self.hour_start = current_hour
        
        if self.hourly_trade_count >= self.max_trades_per_hour:
            return False
        
        return True

    def is_signal_changed(self, last_signal: Optional[Dict], current_signal: Dict) -> bool:
        """Check if signal has changed significantly"""
        if last_signal is None:
            if self.enable_first_signal_trade:
                return (current_signal["final_confidence"] >= self.first_signal_min_confidence and
                        current_signal["trading_recommendation"] == "TRADE")
            return False
        
        # Direction change
        if last_signal["final_direction"] != current_signal["final_direction"]:
            return True
        
        # Confidence change
        confidence_change = abs(last_signal["final_confidence"] - current_signal["final_confidence"])
        if confidence_change > self.signal_change_threshold:
            return True
        
        # High confidence signal
        if (current_signal["final_confidence"] >= 0.85 and
            current_signal["trading_recommendation"] == "TRADE" and
            len(self.trade_history) == 0):
            return True
        
        return False

    def process_signal(self, signal: Dict, symbol: str) -> bool:
        """Process AI signal and execute trade if conditions are met"""
        
        # Check if trading is enabled
        if not self.trading_enabled:
            self.log_signal(signal, symbol, False, "TRADING_DISABLED")
            return False
        
        # Check risk limits
        if not self.check_risk_limits():
            self.log_signal(signal, symbol, False, "RISK_LIMITS")
            return False
        
        # Check confidence
        if signal["final_confidence"] < self.min_confidence:
            self.log_signal(signal, symbol, False, "LOW_CONFIDENCE")
            return False
        
        # Check recommendation
        if signal["trading_recommendation"] != "TRADE":
            self.log_signal(signal, symbol, False, "NO_TRADE_RECOMMENDATION")
            return False
        
        # Check consensus
        individual_signals = signal["individual_signals"]
        long_count = sum(1 for s in individual_signals.values() if s["consensus_prediction"] == 1)
        short_count = sum(1 for s in individual_signals.values() if s["consensus_prediction"] == -1)
        total_agreement = max(long_count, short_count)
        
        if total_agreement < self.min_consensus:
            self.log_signal(signal, symbol, False, "INSUFFICIENT_CONSENSUS")
            return False
        
        # Determine order type
        if signal["final_direction"] == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
        elif signal["final_direction"] == "SHORT":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            self.log_signal(signal, symbol, False, "INVALID_DIRECTION")
            return False
        
        # Calculate position size
        lot_size = self.calculate_position_size(symbol, signal["final_confidence"])
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.log_signal(signal, symbol, False, "NO_TICK_DATA")
            return False
        
        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        stop_loss, take_profit = self.calculate_sl_tp_levels(symbol, order_type, entry_price)
        
        # Create comment
        comment = f"SMC_AI_{signal['final_direction']}_C{signal['final_confidence']:.2f}"
        
        # Execute order
        success = self.send_order(symbol, order_type, lot_size, stop_loss, take_profit, comment)
        
        if success:
            self.hourly_trade_count += 1
            self.log_signal(signal, symbol, True, "TRADE_EXECUTED")
            print(f"🚀 Trade executed: {signal['final_direction']} {symbol}")
        else:
            self.log_signal(signal, symbol, False, "ORDER_FAILED")
        
        return success

    def print_status(self):
        """Print current system status"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM STATUS")
        print("=" * 60)
        print(f"🎯 Trading: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"📈 Active Positions: {len(self.active_positions)}")
        print(f"💰 Daily P&L: ${self.daily_pnl:.2f}")
        print(f"📊 Trades Today: {self.hourly_trade_count}")
        print(f"⏱️ Wait for Completion: {'YES' if self.wait_for_trade_completion else 'NO'}")
        print(f"📁 Logging: {'ENABLED' if self.enable_logging else 'DISABLED'}")
        print("=" * 60)

    def start_auto_trading(self, symbol: str = "XAUUSD.v", update_interval: int = 60):
        """Start the automated trading system"""
        
        print(f"\n🚀 Starting Auto Trading System")
        print(f"📊 Symbol: {symbol}")
        print(f"⏱️ Update Interval: {update_interval} seconds")
        print(f"🎯 Mode: {'Wait for completion' if self.wait_for_trade_completion else 'Multiple trades'}")
        
        # Verify symbol with debugging
        print(f"\n🔍 Verifying symbol: {symbol}")
        if not self.debug_symbol_issues(symbol):
            print("❌ Symbol verification failed")
            return
        
        self.print_status()
        
        last_signal = None
        
        try:
            while True:
                try:
                    # Update positions
                    self.update_positions()
                    
                    # Check if should analyze
                    if not self.should_analyze_signals():
                        print(f"\n⏳ {datetime.now().strftime('%H:%M:%S')} - Waiting for trade completion...")
                        print(f"💰 Daily P&L: ${self.daily_pnl:.2f} | Active: {len(self.active_positions)}")
                        
                        # Show active positions
                        for ticket, trade_info in self.active_positions.items():
                            positions = mt5.positions_get(ticket=ticket)
                            if positions:
                                pos = positions[0]
                                print(f"🔄 {trade_info['type']} {trade_info['volume']} {trade_info['symbol']} | P&L: ${pos.profit:.2f}")
                        
                        time.sleep(update_interval)
                        continue
                    
                    # Analyze signals
                    print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Analyzing signals...")
                    signal = self.signal_engine.get_multi_timeframe_signals(symbol)
                    
                    if "error" in signal:
                        print(f"❌ Signal error: {signal['error']}")
                        time.sleep(update_interval)
                        continue
                    
                    # Display signal
                    print(f"📊 {symbol}: {signal['final_direction']} | Confidence: {signal['final_confidence']:.3f}")
                    print(f"🎯 Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}")
                    print(f"💰 Daily P&L: ${self.daily_pnl:.2f}")
                    
                    # Check if signal changed
                    if self.is_signal_changed(last_signal, signal) and signal["trading_recommendation"] == "TRADE":
                        print("🔥 NEW TRADING SIGNAL DETECTED!")
                        success = self.process_signal(signal, symbol)
                        if success:
                            print("✅ Trade executed successfully")
                        else:
                            print("❌ Trade execution failed")
                    else:
                        # Log signal even if not trading
                        reason = "NO_CHANGE" if not self.is_signal_changed(last_signal, signal) else "NO_TRADE_RECOMMENDATION"
                        self.log_signal(signal, symbol, False, reason)
                    
                    last_signal = signal
                    time.sleep(update_interval)
                    
                except KeyboardInterrupt:
                    print("\n🛑 Auto trading stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Trading loop error: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            print(f"❌ Critical error: {e}")
        finally:
            print("✅ Auto trading system stopped")
            self.print_status()


def main():
    """Main execution function"""
    print("🎯 Professional SMC Auto Trading Bot")
    print("Created by อาจารย์ฟินิกซ์")
    
    # Initialize trader
    trader = ProfessionalSMCAutoTrader(
        models_path="XAUUSD_v_SMC",
        min_confidence=0.75,
        min_consensus=3,
        max_concurrent_trades=1,
        wait_for_trade_completion=True,
        base_lot_size=0.01,
        max_lot_size=0.05,
        enable_logging=True
    )
    
    # Connect to MT5
    if not trader.connect_mt5():
        print("❌ Failed to connect to MT5")
        return
    
    # Load models
    if not trader.load_models():
        print("❌ Failed to load models")
        return
    
    print("\n🎯 System Ready!")
    
    # Ask user about trading
    while True:
        choice = input("\nSelect option:\n1. Enable Live Trading\n2. Demo Mode (Signals Only)\n3. Exit\nChoice: ").strip()
        
        if choice == "1":
            trader.enable_trading(True)
            break
        elif choice == "2":
            trader.enable_trading(False)
            print("📊 Demo mode - signals will be logged but no trades executed")
            break
        elif choice == "3":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")
    
    # Start trading
    try:
        trader.start_auto_trading("XAUUSD.v", 60)
    except KeyboardInterrupt:
        print("\n👋 Trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()