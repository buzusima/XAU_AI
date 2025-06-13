import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Import our enhanced components
from smc_signal_engine import SMCSignalEngine
from gold_trading_config import GoldTradingConfig


class GoldSMCAutoTrader:
    """
    Gold SMC Auto Trading Bot - Universal Broker Support
    Created by อาจารย์ฟินิกซ์ - Professional Gold Trading System
    
    ✨ Features:
    - Auto-detect Gold symbol across brokers
    - Dynamic risk calculation with position sizing
    - Gold-specific volatility handling
    - News time filtering
    - Comprehensive risk metrics
    """

    def __init__(
        self,
        models_path: str = "XAUUSD_SMC",
        account: int = None,
        password: str = None,
        server: str = None,
        # Risk Management
        target_risk_per_trade: float = 2.0,  # 2% risk per trade
        max_daily_risk: float = 6.0,         # 6% max daily risk
        max_drawdown_limit: float = 15.0,    # 15% max drawdown
        # Trading Controls
        max_concurrent_trades: int = 1,
        wait_for_trade_completion: bool = True,
        # Signal Sensitivity
        min_confidence: float = 0.75,
        min_consensus: int = 3,
        signal_change_threshold: float = 0.001,
        enable_first_signal_trade: bool = True,
        first_signal_min_confidence: float = 0.80,
        # Gold-Specific Settings
        auto_detect_symbol: bool = True,
        gold_symbol: str = None,
        dynamic_lot_sizing: bool = True,
        news_filter_enabled: bool = True,
        spread_filter_enabled: bool = True,
        max_trades_per_hour: int = 3,
    ):
        """Initialize Gold SMC Auto Trader"""

        # Connection settings
        self.account = account
        self.password = password
        self.server = server
        self.timezone = pytz.timezone("Etc/UTC")

        # Gold Configuration
        self.gold_config = GoldTradingConfig()
        self.auto_detect_symbol = auto_detect_symbol
        self.gold_symbol = gold_symbol
        self.detected_symbol = None

        # Risk Management
        self.target_risk_per_trade = target_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.dynamic_lot_sizing = dynamic_lot_sizing

        # Trading Controls
        self.max_concurrent_trades = max_concurrent_trades
        self.wait_for_trade_completion = wait_for_trade_completion
        self.trading_enabled = False

        # Signal Settings
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.signal_change_threshold = signal_change_threshold
        self.enable_first_signal_trade = enable_first_signal_trade
        self.first_signal_min_confidence = first_signal_min_confidence

        # Gold-Specific Filters
        self.news_filter_enabled = news_filter_enabled
        self.spread_filter_enabled = spread_filter_enabled
        self.max_trades_per_hour = max_trades_per_hour

        # Trading State
        self.active_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.hourly_trade_count = 0
        self.hour_start = datetime.now().hour
        self.total_risk_exposure = 0.0
        self.last_trade_closed_time = None

        # Performance Tracking
        self.daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "risk_taken": 0.0
        }

        # Initialize Signal Engine
        self.signal_engine = SMCSignalEngine(models_path)

        print("🏆 Gold SMC Auto Trading Bot Initialized")
        print("⚠️ Trading is DISABLED by default for safety")

    def connect_mt5(self) -> bool:
        """Connect to MT5 with Gold symbol detection"""
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
            print(f"🎯 Leverage: 1:{account_info.leverage}")

    def connect_mt5(self) -> bool:
        """Connect to MT5 with enhanced Gold symbol detection"""
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
            print(f"🎯 Leverage: 1:{account_info.leverage}")

            # Enhanced Gold symbol detection
            if self.auto_detect_symbol:
                potential_symbols = self.gold_config.detect_gold_symbol()
                
                if potential_symbols:
                    if isinstance(potential_symbols, list):
                        # Multiple symbols found - let user choose
                        self.detected_symbol = self._select_gold_symbol(potential_symbols)
                    else:
                        # Single symbol found
                        self.detected_symbol = potential_symbols
                        
                    if self.detected_symbol:
                        print(f"🥇 Selected Gold Symbol: {self.detected_symbol}")
                    else:
                        print("❌ No Gold symbol selected")
                        return self._manual_symbol_setup()
                else:
                    print("❌ No Gold symbols found automatically")
                    return self._manual_symbol_setup()
            else:
                self.detected_symbol = self.gold_symbol
                if not self._verify_symbol_exists(self.detected_symbol):
                    print(f"❌ Manual symbol '{self.gold_symbol}' not found")
                    return self._manual_symbol_setup()

            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def _select_gold_symbol(self, potential_symbols: List[Dict]) -> Optional[str]:
        """Let user select from detected Gold symbols"""
        print("\n🎯 Multiple Gold symbols detected. Please select:")
        print("=" * 50)
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(potential_symbols)}) or 'm' for manual: ").strip().lower()
                
                if choice == 'm':
                    return self._get_manual_symbol()
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(potential_symbols):
                    selected = potential_symbols[choice_num - 1]
                    symbol_name = selected['name']
                    
                    # Confirm selection
                    print(f"\n✅ Selected: {symbol_name}")
                    print(f"📊 Current Price: ${selected['current_price']:.2f}")
                    print(f"📈 Spread: {selected['spread']:.4f}")
                    
                    confirm = input("\nConfirm this selection? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return symbol_name
                    else:
                        continue
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'm'")
            except KeyboardInterrupt:
                print("\n🛑 Selection cancelled")
                return None

    def _manual_symbol_setup(self) -> bool:
        """Manual symbol setup when auto-detection fails"""
        print("\n🔧 Manual Symbol Setup Required")
        print("=" * 40)
        
        # Show available symbols that might be Gold
        self._show_available_symbols()
        
        symbol = self._get_manual_symbol()
        if symbol:
            self.detected_symbol = symbol
            print(f"✅ Manual symbol set: {symbol}")
            return True
        else:
            print("❌ No symbol provided")
            return False

    def _show_available_symbols(self):
        """Show available symbols for manual selection"""
        print("🔍 Searching all available symbols...")
        try:
            symbols = mt5.symbols_get()
            if symbols:
                # Filter symbols that might be metals/commodities
                metal_symbols = []
                for symbol in symbols:
                    name = symbol.name.upper()
                    desc = getattr(symbol, 'description', '').upper()
                    
                    # Look for metal-related keywords
                    metal_keywords = ['METAL', 'GOLD', 'SILVER', 'XAU', 'XAG', 'GC', 'SI', 'AU', 'AG']
                    if any(keyword in name or keyword in desc for keyword in metal_keywords):
                        try:
                            tick = mt5.symbol_info_tick(symbol.name)
                            if tick:
                                metal_symbols.append({
                                    'name': symbol.name,
                                    'desc': desc,
                                    'price': tick.bid
                                })
                        except:
                            continue
                
                if metal_symbols:
                    print("\n💎 Available Metal/Commodity symbols:")
                    print("-" * 60)
                    for symbol in metal_symbols[:10]:  # Show first 10
                        print(f"   {symbol['name']} - {symbol['desc']} (${symbol['price']:.2f})")
                    
                    if len(metal_symbols) > 10:
                        print(f"   ... and {len(metal_symbols) - 10} more symbols")
                else:
                    print("❌ No metal symbols found")
                    
        except Exception as e:
            print(f"❌ Error searching symbols: {str(e)}")

    def _get_manual_symbol(self) -> Optional[str]:
        """Get symbol manually from user"""
        print("\n📝 Enter Gold symbol manually:")
        print("Examples: XAUUSD, GOLD, GOLDUSD, XAU/USD, etc.")
        
        while True:
            symbol = input("\nGold Symbol: ").strip().upper()
            
            if not symbol:
                return None
                
            if self._verify_symbol_exists(symbol):
                return symbol
            else:
                print(f"❌ Symbol '{symbol}' not found or not tradeable")
                retry = input("Try another symbol? (y/n): ").strip().lower()
                if retry != 'y':
                    return None

    def _verify_symbol_exists(self, symbol: str) -> bool:
        """Verify if symbol exists and is tradeable"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
                
            # Check if symbol is visible and tradeable
            if not symbol_info.visible:
                # Try to make it visible
                if not mt5.symbol_select(symbol, True):
                    return False
                    
            # Get current price to verify it's working
            tick = mt5.symbol_info_tick(symbol)
            if not tick or tick.bid <= 0:
                return False
                
            print(f"✅ Symbol verified: {symbol}")
            print(f"   Current Price: ${tick.bid:.2f}")
            print(f"   Spread: {tick.ask - tick.bid:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Symbol verification error: {str(e)}")
            return False

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def get_trading_symbol(self) -> str:
        """Get the trading symbol (auto-detected or manual)"""
        return self.detected_symbol or self.gold_symbol or "XAUUSD"

    def check_trading_conditions(self) -> Dict:
        """Check if conditions are suitable for Gold trading"""
        symbol = self.get_trading_symbol()
        conditions = self.gold_config.check_gold_trading_conditions(symbol)
        
        # Additional risk checks
        account_info = mt5.account_info()
        if account_info:
            # Check daily risk limit
            daily_risk_pct = (abs(self.daily_pnl) / account_info.balance) * 100
            conditions["daily_risk_exceeded"] = daily_risk_pct > self.max_daily_risk
            
            # Check total exposure
            conditions["high_exposure"] = self.total_risk_exposure > (account_info.balance * 0.10)
            
            # Update trading recommendation
            if conditions["daily_risk_exceeded"]:
                conditions["recommendation"] = "STOP - Daily risk limit exceeded"
            elif conditions["high_exposure"]:
                conditions["recommendation"] = "REDUCE - High exposure"
                
        return conditions

    def calculate_optimal_position_size(self, confidence: float, sl_pips: int = None) -> Dict:
        """Calculate optimal position size with comprehensive risk analysis"""
        symbol = self.get_trading_symbol()
        
        if sl_pips is None:
            sl_pips = self.gold_config.gold_sl_pips
            
        try:
            account_info = mt5.account_info()
            if not account_info:
                return {"error": "Cannot get account info"}
            
            # Adjust risk based on confidence
            base_risk = self.target_risk_per_trade
            confidence_multiplier = min(1.5, max(0.5, confidence / 0.75))
            adjusted_risk = base_risk * confidence_multiplier
            
            # Get optimal position size
            optimization = self.gold_config.optimize_position_size(
                symbol, account_info.balance, adjusted_risk, sl_pips
            )
            
            if "error" in optimization:
                return optimization
                
            optimal_lot = optimization["optimal_lot_size"]
            
            # Additional safety checks
            if self.daily_trades >= 5:  # Limit daily trades
                optimal_lot *= 0.7  # Reduce size for excessive trading
                
            if len(self.active_positions) > 0:  # Already have positions
                optimal_lot *= 0.5  # Reduce size significantly
                
            # Final risk calculation
            risk_metrics = self.gold_config.calculate_risk_metrics(
                symbol, optimal_lot, sl_pips
            )
            
            return {
                "optimal_lot_size": optimal_lot,
                "adjusted_risk_pct": adjusted_risk,
                "confidence_multiplier": confidence_multiplier,
                "risk_metrics": risk_metrics,
                "optimization_details": optimization
            }
            
        except Exception as e:
            print(f"❌ Position sizing error: {str(e)}")
            return {"error": str(e)}

    def send_gold_order(
        self,
        order_type: int,
        lot_size: float,
        confidence: float,
        signal_info: Dict,
        comment: str = "Gold_SMC_AI",
    ) -> bool:
        """Send optimized Gold trading order"""

        if not self.trading_enabled:
            print("⚠️ Trading disabled - order not sent")
            return False

        symbol = self.get_trading_symbol()

        try:
            # Pre-trade checks
            conditions = self.check_trading_conditions()
            if not conditions.get("tradeable", False):
                print(f"🛑 Trading conditions not met: {conditions['recommendation']}")
                return False

            # Get current price and calculate SL/TP
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"❌ Cannot get {symbol} price")
                return False

            entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Calculate SL/TP using Gold-optimized method
            stop_loss, take_profit = self.gold_config.get_gold_sl_tp_levels(
                symbol, order_type, entry_price
            )

            # Final risk validation
            sl_pips = abs(entry_price - stop_loss) / self.gold_config.gold_pip_size
            risk_metrics = self.gold_config.calculate_risk_metrics(
                symbol, lot_size, int(sl_pips), entry_price
            )

            if "error" in risk_metrics:
                print(f"❌ Risk calculation failed: {risk_metrics['error']}")
                return False

            # Check risk limits
            if risk_metrics["risk_percentage"] > self.target_risk_per_trade * 1.5:
                print(f"🛑 Risk too high: {risk_metrics['risk_percentage']:.2f}%")
                return False

            # Prepare order request
            order_type_str = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 30,  # Higher deviation for Gold
                "magic": 234567,
                "comment": f"{comment}_C{confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            print(f"📋 Sending Gold order:")
            print(f"   {order_type_str} {lot_size} {symbol} @ {entry_price:.2f}")
            print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
            print(f"   Risk: ${risk_metrics['max_loss_usd']:.2f} ({risk_metrics['risk_percentage']:.2f}%)")

            # Send order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Order failed: {result.retcode} - {result.comment}")

                # Try alternative filling mode
                if result.retcode == 10014:  # Invalid volume
                    print("🔄 Trying FOK filling...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)

                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"❌ Second attempt failed: {result.retcode}")
                        return False
                else:
                    return False

            # Store trade information
            trade_info = {
                "timestamp": datetime.now(self.timezone),
                "symbol": symbol,
                "type": order_type_str,
                "volume": lot_size,
                "price": result.price,
                "sl": stop_loss,
                "tp": take_profit,
                "ticket": result.order,
                "comment": request["comment"],
                "confidence": confidence,
                "signal_info": signal_info,
                "risk_metrics": risk_metrics,
                "entry_conditions": conditions
            }

            self.trade_history.append(trade_info)
            self.active_positions[result.order] = trade_info

            # Update counters
            self.hourly_trade_count += 1
            self.daily_trades += 1
            self.total_risk_exposure += risk_metrics['max_loss_usd']

            print(f"✅ Gold trade executed:")
            print(f"   🎫 Ticket: {result.order}")
            print(f"   💰 Position Value: ${risk_metrics['position_value']:,.2f}")
            print(f"   📊 R:R Ratio: 1:{risk_metrics['risk_reward_ratio']:.1f}")

            return True

        except Exception as e:
            print(f"❌ Gold order execution error: {str(e)}")
            return False

    def process_gold_signal(self, signal: Dict) -> bool:
        """Process AI signal for Gold trading with enhanced validation"""

        if not self.trading_enabled:
            return False

        # Check if we can trade
        if not self.check_can_trade():
            return False

        # Signal validation
        if signal["final_confidence"] < self.min_confidence:
            print(f"⚠️ Signal confidence too low: {signal['final_confidence']:.3f}")
            return False

        if signal["trading_recommendation"] != "TRADE":
            print(f"📊 Signal recommendation: {signal['trading_recommendation']}")
            return False

        # Consensus check
        individual_signals = signal["individual_signals"]
        long_count = sum(
            1 for s in individual_signals.values() 
            if s.get("consensus_prediction") == 1
        )
        short_count = sum(
            1 for s in individual_signals.values() 
            if s.get("consensus_prediction") == -1
        )

        total_agreement = max(long_count, short_count)
        if total_agreement < self.min_consensus:
            print(f"⚠️ Insufficient consensus: {total_agreement}/{len(individual_signals)}")
            return False

        # Determine order type
        if signal["final_direction"] == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
        elif signal["final_direction"] == "SHORT":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            print(f"⚠️ Invalid signal direction: {signal['final_direction']}")
            return False

        # Calculate optimal position size
        position_calc = self.calculate_optimal_position_size(signal["final_confidence"])
        
        if "error" in position_calc:
            print(f"❌ Position calculation failed: {position_calc['error']}")
            return False

        lot_size = position_calc["optimal_lot_size"]

        # Execute trade
        success = self.send_gold_order(
            order_type=order_type,
            lot_size=lot_size,
            confidence=signal["final_confidence"],
            signal_info=signal,
            comment=f"Gold_SMC_{signal['final_direction']}"
        )

        if success:
            print(f"🚀 Gold auto trade executed: {signal['final_direction']}")
            print(f"   📊 Confidence: {signal['final_confidence']:.3f}")
            print(f"   💰 Lot Size: {lot_size}")
            print(f"   🎯 Risk: {position_calc['adjusted_risk_pct']:.2f}%")

        return success

    def check_can_trade(self) -> bool:
        """Enhanced trading permission check for Gold"""
        
        # Basic checks
        if self.wait_for_trade_completion and len(self.active_positions) > 0:
            print(f"⏳ Waiting for trade completion. Active: {len(self.active_positions)}")
            return False

        if len(self.active_positions) >= self.max_concurrent_trades:
            print(f"🛑 Max concurrent trades reached: {len(self.active_positions)}")
            return False

        # Hourly limit check
        current_hour = datetime.now().hour
        if current_hour != self.hour_start:
            self.hourly_trade_count = 0
            self.hour_start = current_hour

        if self.hourly_trade_count >= self.max_trades_per_hour:
            print(f"🛑 Hourly trade limit reached: {self.hourly_trade_count}")
            return False

        # Gold-specific conditions
        conditions = self.check_trading_conditions()
        if not conditions.get("tradeable", False):
            print(f"🛑 Gold conditions: {conditions.get('recommendation', 'Not suitable')}")
            return False

        return True

    def update_positions_and_stats(self):
        """Update positions and calculate performance statistics"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            current_tickets = [pos.ticket for pos in positions]
            closed_tickets = []

            # Check for closed positions
            for ticket in list(self.active_positions.keys()):
                if ticket not in current_tickets:
                    closed_tickets.append(ticket)
                    self._process_closed_trade(ticket)

            # Update daily P&L
            total_profit = sum(pos.profit for pos in positions)
            self.daily_pnl = total_profit

            # Update risk exposure
            self.total_risk_exposure = sum(
                trade_info.get("risk_metrics", {}).get("max_loss_usd", 0)
                for trade_info in self.active_positions.values()
            )

            if closed_tickets:
                print(f"🎯 {len(closed_tickets)} position(s) closed")
                self._update_daily_stats()

        except Exception as e:
            print(f"❌ Position update error: {str(e)}")

    def _process_closed_trade(self, ticket: int):
        """Process a closed trade and update statistics"""
        try:
            trade_info = self.active_positions[ticket]
            
            print(f"📈 Trade #{ticket} CLOSED:")
            print(f"   {trade_info['type']} {trade_info['volume']} {trade_info['symbol']}")
            print(f"   Entry: {trade_info['price']:.2f}")

            # Get trade result from history
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(hours=24), datetime.now()
            )

            if deals:
                for deal in deals:
                    if deal.position_id == ticket and deal.entry == 1:  # Closing deal
                        profit = deal.profit
                        close_price = deal.price
                        close_time = datetime.fromtimestamp(deal.time)

                        print(f"   Exit: {close_price:.2f}")
                        print(f"   P&L: ${profit:.2f}")
                        
                        # Determine win/loss
                        is_win = profit > 0
                        result_emoji = "✅ WIN" if is_win else "❌ LOSS"
                        print(f"   Result: {result_emoji}")

                        # Update statistics
                        if is_win:
                            self.daily_stats["wins"] += 1
                        else:
                            self.daily_stats["losses"] += 1
                            
                        self.daily_stats["pnl"] += profit
                        break

            # Remove from active positions
            del self.active_positions[ticket]
            self.last_trade_closed_time = datetime.now()

            # Reduce risk exposure
            risk_amount = trade_info.get("risk_metrics", {}).get("max_loss_usd", 0)
            self.total_risk_exposure = max(0, self.total_risk_exposure - risk_amount)

        except Exception as e:
            print(f"❌ Error processing closed trade: {str(e)}")

    def _update_daily_stats(self):
        """Update and display daily statistics"""
        total_trades = self.daily_stats["wins"] + self.daily_stats["losses"]
        if total_trades > 0:
            win_rate = (self.daily_stats["wins"] / total_trades) * 100
            print(f"📊 Daily Stats: {self.daily_stats['wins']}W/{self.daily_stats['losses']}L ({win_rate:.1f}%)")
            print(f"💰 Daily P&L: ${self.daily_stats['pnl']:.2f}")

    def print_current_settings(self):
        """Print current Gold trading configuration"""
        print("⚙️ Gold SMC Auto Trader Settings:")
        print("=" * 60)
        print(f"🥇 Symbol: {self.get_trading_symbol()}")
        print(f"🎯 Target Risk/Trade: {self.target_risk_per_trade}%")
        print(f"📊 Max Daily Risk: {self.max_daily_risk}%")
        print(f"🛡️ Max Drawdown: {self.max_drawdown_limit}%")
        print(f"⏳ Wait for completion: {'YES' if self.wait_for_trade_completion else 'NO'}")
        print(f"📈 Min confidence: {self.min_confidence*100}%")
        print(f"🤝 Min consensus: {self.min_consensus}/5")
        print(f"📰 News filter: {'ON' if self.news_filter_enabled else 'OFF'}")
        print(f"📊 Spread filter: {'ON' if self.spread_filter_enabled else 'OFF'}")
        print("=" * 60)

    def start_gold_auto_trading(self, update_interval: int = 60):
        """Start automated Gold trading system"""
        
        symbol = self.get_trading_symbol()
        
        print("🏆 Starting Gold SMC Auto Trading System")
        print("=" * 60)
        print(f"🥇 Symbol: {symbol}")
        print(f"🎯 Status: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"⏳ Mode: One trade at a time")
        print(f"🔄 Update: {update_interval}s")
        print("=" * 60)

        last_signal = None

        while True:
            try:
                # Update positions and stats
                self.update_positions_and_stats()

                # Check if we should analyze signals
                if not self.should_analyze_signals():
                    self._display_waiting_status()
                    time.sleep(update_interval)
                    continue

                # Generate new signals
                print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Analyzing Gold signals...")
                signal = self.signal_engine.get_multi_timeframe_signals(symbol)

                if "error" in signal:
                    print(f"❌ Signal error: {signal['error']}")
                else:
                    self._display_signal_status(signal)

                    # Check for trading signal
                    signal_changed = self._is_signal_changed(last_signal, signal)

                    if signal_changed and signal["trading_recommendation"] == "TRADE":
                        if self.trading_enabled:
                            print("🔥 NEW GOLD TRADING SIGNAL DETECTED!")
                            
                            # Show market conditions
                            conditions = self.check_trading_conditions()
                            print(f"📊 Market: {conditions.get('recommendation', 'Unknown')}")
                            print(f"📈 Spread: {conditions.get('spread', 0):.1f} pips")
                            
                            success = self.process_gold_signal(signal)
                            if success:
                                print("✅ Gold auto trade executed successfully")
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
                print(f"❌ Auto trading error: {str(e)}")
                time.sleep(10)

        print("✅ Gold auto trading system stopped")

    def should_analyze_signals(self) -> bool:
        """Check if we should analyze new signals"""
        if not self.wait_for_trade_completion:
            return True
        return len(self.active_positions) == 0

    def _display_waiting_status(self):
        """Display waiting status with active positions"""
        print(f"\n⏳ {datetime.now().strftime('%H:%M:%S')} - Waiting for Gold trade completion...")
        print(f"💰 Daily P&L: ${self.daily_pnl:.2f} | Risk Exposure: ${self.total_risk_exposure:.2f}")

        if self.active_positions:
            for ticket, trade_info in self.active_positions.items():
                positions = mt5.positions_get(ticket=ticket)
                if positions:
                    pos = positions[0]
                    pnl_pips = (pos.profit / trade_info.get('risk_metrics', {}).get('pip_value', 1))
                    print(f"🔄 Active: {trade_info['type']} {trade_info['volume']} {trade_info['symbol']}")
                    print(f"   Entry: {trade_info['price']:.2f} | P&L: ${pos.profit:.2f} ({pnl_pips:.1f} pips)")

    def _display_signal_status(self, signal: Dict):
        """Display current signal status"""
        symbol = self.get_trading_symbol()
        print(f"🥇 {symbol}: {signal['final_direction']} | Confidence: {signal['final_confidence']:.3f}")
        print(f"🎯 Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}")
        print(f"💰 Daily P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades}")

    def _is_signal_changed(self, last_signal: Optional[Dict], current_signal: Dict) -> bool:
        """Check if signal has changed significantly"""
        if last_signal is None:
            if self.enable_first_signal_trade:
                return (
                    current_signal["final_confidence"] >= self.first_signal_min_confidence
                    and current_signal["trading_recommendation"] == "TRADE"
                )
            return False

        # Direction change
        if last_signal["final_direction"] != current_signal["final_direction"]:
            return True

        # Confidence change
        confidence_change = abs(
            last_signal["final_confidence"] - current_signal["final_confidence"]
        )
        if confidence_change > self.signal_change_threshold:
            return True

        # High confidence override
        if (
            current_signal["final_confidence"] >= 0.85
            and current_signal["trading_recommendation"] == "TRADE"
            and len(self.trade_history) == 0
        ):
            print(f"🔥 High confidence Gold signal: {current_signal['final_confidence']:.3f}")
            return True

        return False

    def enable_trading(self, enable: bool = True):
        """Enable or disable automated trading"""
        self.trading_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🎯 Gold Auto Trading {status}")

        if enable:
            print("⚠️ WARNING: Live Gold trading is now active!")
            print("🛡️ Enhanced safety mechanisms active")
            print(f"🥇 Trading symbol: {self.get_trading_symbol()}")


# Main execution for Gold trading
if __name__ == "__main__":
    print("🏆 Gold SMC Auto Trading Bot - Universal Broker Support")
    print("=" * 60)

    # Gold-optimized settings
    SYMBOL_AUTO_DETECT = True
    TARGET_RISK_PER_TRADE = 2.0      # 2% risk per trade
    MAX_DAILY_RISK = 6.0             # 6% max daily risk
    MIN_CONFIDENCE = 0.75            # 75% minimum confidence
    MIN_CONSENSUS = 3                # 3/5 timeframes agreement
    ENABLE_FIRST_TRADE = True
    FIRST_TRADE_MIN_CONF = 0.80      # 80% confidence for first trade
    NEWS_FILTER = True
    SPREAD_FILTER = True

    # Initialize Gold trader
    trader = GoldSMCAutoTrader(
        models_path="XAUUSD_SMC",
        auto_detect_symbol=SYMBOL_AUTO_DETECT,
        target_risk_per_trade=TARGET_RISK_PER_TRADE,
        max_daily_risk=MAX_DAILY_RISK,
        min_confidence=MIN_CONFIDENCE,
        min_consensus=MIN_CONSENSUS,
        enable_first_signal_trade=ENABLE_FIRST_TRADE,
        first_signal_min_confidence=FIRST_TRADE_MIN_CONF,
        news_filter_enabled=NEWS_FILTER,
        spread_filter_enabled=SPREAD_FILTER,
        dynamic_lot_sizing=True,
        wait_for_trade_completion=True,
    )

    trader.print_current_settings()

    if trader.connect_mt5():
        if trader.signal_engine.load_trained_models():
            print("\n🎯 Gold Auto Trading Bot Ready!")

            enable_trading = (
                input("\n🚀 Enable LIVE GOLD AUTO TRADING? (yes/no): ").lower().strip()
            )

            if enable_trading == "yes":
                trader.enable_trading(True)
                
                # Show final risk warning
                print("\n" + "="*60)
                print("⚠️  GOLD TRADING RISK WARNING")
                print("="*60)
                print(f"💰 Max risk per trade: {TARGET_RISK_PER_TRADE}%")
                print(f"📊 Max daily risk: {MAX_DAILY_RISK}%")
                print("🥇 Gold is highly volatile - monitor closely!")
                print("="*60)
                
                confirm = input("\n✅ I understand the risks. Start trading? (YES/no): ")
                if confirm.upper() == "YES":
                    trader.start_gold_auto_trading(60)
                else:
                    print("📊 Demo mode - trading disabled")
            else:
                print("📊 Demo mode - signals only")
                trader.enable_trading(False)
                trader.start_gold_auto_trading(60)

        else:
            print("❌ Failed to load AI models")
    else:
        print("❌ Failed to connect to MT5")