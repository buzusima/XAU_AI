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

# Import our Signal Engine
from smc_signal_engine import SMCSignalEngine


class SMCAutoTrader:
    """
    SMC Auto Trading Bot - One Trade at a Time
    Created by อาจารย์ฟินิกซ์ - Professional Automated Trading System
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
        """Initialize SMC Auto Trader with configurable parameters"""

        # Connection settings
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

        # Initialize Signal Engine
        self.signal_engine = SMCSignalEngine(models_path)

        print("🤖 SMC Auto Trading Bot Initialized")
        print("⚠️ Trading is DISABLED by default for safety")

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

    def send_order(
        self,
        symbol: str,
        order_type: int,
        lot_size: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "SMC_AI_Bot",
    ) -> bool:
        """Send trading order to MT5"""

        if not self.trading_enabled:
            print("⚠️ Trading disabled - order not sent")
            return False

        try:
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

            trade_info = {
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

            return True

        except Exception as e:
            print(f"❌ Order execution error: {str(e)}")
            return False

    def process_signal(self, signal: Dict, symbol: str) -> bool:
        """Process AI signal and execute trade if conditions are met"""

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

        success = self.send_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

        if success:
            self.hourly_trade_count += 1
            print(f"🚀 Auto trade executed: {signal['final_direction']} {symbol}")

        return success

    def update_positions(self):
        """Update active positions and calculate P&L"""
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

                    deals = mt5.history_deals_get(
                        datetime.now() - timedelta(hours=1), datetime.now()
                    )

                    if deals:
                        for deal in deals:
                            if deal.position_id == ticket and deal.entry == 1:
                                profit = deal.profit
                                close_price = deal.price
                                close_time = datetime.fromtimestamp(deal.time)

                                print(f"   Exit: {close_price:.5f}")
                                print(f"   P&L: ${profit:.2f}")
                                print(
                                    f"   Result: {'✅ WIN' if profit > 0 else '❌ LOSS'}"
                                )
                                break

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

    def print_current_settings(self):
        """Print current configuration"""
        print("⚙️ Auto Trader Settings:")
        print("=" * 50)
        print(f"🎯 Max concurrent trades: {self.max_concurrent_trades}")
        print(
            f"⏳ Wait for completion: {'YES' if self.wait_for_trade_completion else 'NO'}"
        )
        print(f"📊 Min confidence: {self.min_confidence*100}%")
        print(f"🤝 Min consensus: {self.min_consensus}/5")
        print(f"💰 Base lot size: {self.base_lot_size}")
        print("=" * 50)

    def start_auto_trading(self, symbol: str = "EURUSD.c", update_interval: int = 60):
        """Start automated trading system"""

        print("🚀 Starting SMC Auto Trading System - One Trade at a Time")
        print("=" * 60)
        print(f"📊 Symbol: {symbol}")
        print(f"🎯 Trading Status: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"⏳ Mode: One trade at a time")
        print("=" * 60)

        last_signal = None

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

        print("✅ Auto trading system stopped")


# Main execution
if __name__ == "__main__":
    print("🤖 SMC Auto Trading Bot - One Trade at a Time")
    print("=" * 50)

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

    # Initialize trader
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
            print("\n🎯 Auto Trading Bot Ready!")

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
