# XAUUSD SMC Auto Trading Bot - Gold Trading System
# Modified for XAUUSD from the original EURUSD system

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


class XAUUSDAutoTrader:
    """
    XAUUSD SMC Auto Trading Bot - Specialized for Gold Trading
    Created by อาจารย์ฟินิกซ์ - Professional Gold Trading System
    """

    def __init__(
        self,
        models_path: str = "XAUUSD_v_SMC",  # แก้ไขเป็น XAUUSD
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
        default_sl_pips: int = 20,  # สำหรับทองคำใช้ pips เป็น points
        default_tp_ratio: float = 2.0,
        max_trades_per_hour: int = 5,
        wait_for_trade_completion: bool = True,
    ):
        """Initialize XAUUSD Auto Trader with Gold-specific parameters"""

        # Connection settings
        self.account = account
        self.password = password
        self.server = server
        self.timezone = pytz.timezone("Etc/UTC")

        # Signal Sensitivity Controls
        self.signal_change_threshold = signal_change_threshold
        self.enable_first_signal_trade = enable_first_signal_trade
        self.first_signal_min_confidence = first_signal_min_confidence

        # Risk Management Settings (ปรับสำหรับทองคำ)
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_concurrent_trades = max_concurrent_trades
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus

        # Position Sizing (ปรับสำหรับทองคำ)
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.lot_multiplier = lot_multiplier

        # Trade Management (ปรับสำหรับทองคำ)
        self.default_sl_pips = default_sl_pips  # ทองคำ 1 pip = $0.01
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

        # Initialize Signal Engine สำหรับ XAUUSD
        self.signal_engine = SMCSignalEngine(models_path)

        print("🥇 XAUUSD SMC Auto Trading Bot Initialized")
        print("💰 Specialized for Gold Trading")
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

            print("✅ MT5 Connected with Gold Trading Capabilities")
            print(f"📊 Account: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance:.2f}")
            print(f"🥇 Currency: {account_info.currency}")

            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def load_models(self) -> bool:
        """Load AI models through signal engine"""
        print("🔄 Loading XAUUSD AI Models...")
        success = self.signal_engine.load_trained_models()
        if success:
            print("✅ XAUUSD Models loaded successfully")
        else:
            print("❌ Failed to load XAUUSD models")
        return success

    def enable_trading(self, enable: bool = True):
        """Enable or disable automated trading"""
        self.trading_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🎯 XAUUSD Auto Trading {status}")

        if enable:
            print("⚠️ WARNING: Live Gold trading is now active!")
            print("🛡️ Safety mechanisms active")

    def calculate_gold_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal position size for Gold trading"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return self.base_lot_size

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Cannot get symbol info for {symbol}")
                return self.base_lot_size

            print(f"🥇 Gold {symbol} specifications:")
            print(f"   Min lot: {symbol_info.volume_min}")
            print(f"   Max lot: {symbol_info.volume_max}")
            print(f"   Lot step: {symbol_info.volume_step}")
            print(f"   Contract size: {symbol_info.trade_contract_size}")

            # ปรับขนาด lot สำหรับทองคำ
            calculated_lot = self.base_lot_size

            # เพิ่มขนาดตาม confidence
            if confidence >= 0.9:
                calculated_lot *= self.lot_multiplier * 1.5  # เพิ่มสำหรับทองคำ
            elif confidence >= 0.8:
                calculated_lot *= self.lot_multiplier
            elif confidence >= 0.75:
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

            print(f"💰 Gold position size: {self.base_lot_size} → {final_lot}")

            return final_lot

        except Exception as e:
            print(f"❌ Gold position size error: {str(e)}")
            return self.base_lot_size

    def calculate_gold_sl_tp_levels(
        self, symbol: str, order_type: int, entry_price: float
    ) -> Tuple[float, float]:
        """Calculate Stop Loss and Take Profit levels for Gold"""
        try:
            # ทองคำใช้ point size แทน pip
            # XAUUSD: 1 point = $0.01
            point_size = 0.01
            
            sl_distance = self.default_sl_pips * point_size
            tp_distance = sl_distance * self.default_tp_ratio

            if order_type == mt5.ORDER_TYPE_BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            print(f"🎯 Gold SL/TP calculation:")
            print(f"   Entry: {entry_price:.2f}")
            print(f"   SL: {stop_loss:.2f} (Risk: {sl_distance:.2f})")
            print(f"   TP: {take_profit:.2f} (Reward: {tp_distance:.2f})")

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
        comment: str = "XAUUSD_SMC_AI_Bot",
    ) -> bool:
        """Send Gold trading order to MT5"""

        if not self.trading_enabled:
            print("⚠️ Trading disabled - Gold order not sent")
            return False

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Gold symbol {symbol} not found")
                return False

            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            print(f"🔍 Gold order validation:")
            print(f"   Requested lot: {lot_size}")
            print(f"   Broker limits: {min_lot} - {max_lot}, step: {lot_step}")

            # ปรับขนาด lot ให้เหมาะสม
            if lot_size < min_lot:
                lot_size = min_lot
                print(f"⚠️ Adjusted to minimum: {lot_size}")
            elif lot_size > max_lot:
                lot_size = max_lot
                print(f"⚠️ Adjusted to maximum: {lot_size}")

            if lot_step > 0:
                lot_size = round(lot_size / lot_step) * lot_step
                print(f"🔧 Rounded to step: {lot_size}")

            # ดึงราคาปัจจุบัน
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
                "deviation": 50,  # เพิ่ม deviation สำหรับทองคำ
                "magic": 999999,  # Magic number สำหรับทองคำ
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            print(f"📋 Sending Gold order: {order_type_str} {lot_size} {symbol}")

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Gold order failed: {result.retcode} - {result.comment}")

                # ลองเปลี่ยน filling mode
                if result.retcode == 10014:
                    print("🔄 Trying FOK filling for Gold...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)

                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"❌ Second attempt failed: {result.retcode}")
                        return False
                else:
                    return False

            # บันทึกข้อมูลการเทรด
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

            print(f"✅ Gold order executed: {order_type_str} {lot_size} {symbol} @ ${result.price:.2f}")
            print(f"   🎯 SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            print(f"   🎫 Ticket: {result.order}")

            return True

        except Exception as e:
            print(f"❌ Gold order execution error: {str(e)}")
            return False

    def process_gold_signal(self, signal: Dict, symbol: str) -> bool:
        """Process AI signal for Gold trading"""

        if not self.trading_enabled:
            return False

        # เช็คสถานะการเทรด
        if self.wait_for_trade_completion and len(self.active_positions) > 0:
            print(f"⏳ Waiting for Gold trade to close. Active: {len(self.active_positions)}")
            return False

        if signal["final_confidence"] < self.min_confidence:
            print(f"⚠️ Gold signal confidence too low: {signal['final_confidence']:.3f}")
            return False

        if signal["trading_recommendation"] != "TRADE":
            return False

        # ตรวจสอบ consensus
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

        # กำหนดทิศทางการเทรด
        if signal["final_direction"] == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
        elif signal["final_direction"] == "SHORT":
            order_type = mt5.ORDER_TYPE_SELL
        else:
            return False

        # คำนวณขนาด position สำหรับทองคำ
        lot_size = self.calculate_gold_position_size(symbol, signal["final_confidence"])

        # ดึงราคาปัจจุบัน
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        stop_loss, take_profit = self.calculate_gold_sl_tp_levels(
            symbol, order_type, entry_price
        )

        comment = f"XAUUSD_SMC_{signal['final_direction']}_C{signal['final_confidence']:.2f}"

        # ส่งคำสั่งซื้อขาย
        success = self.send_gold_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

        if success:
            self.hourly_trade_count += 1
            print(f"🚀 Gold auto trade executed: {signal['final_direction']} {symbol}")

        return success

    def update_gold_positions(self):
        """Update Gold positions and calculate P&L"""
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
                    print(f"📈 Gold Trade #{ticket} CLOSED:")
                    print(f"   {closed_trade['type']} {closed_trade['volume']} {closed_trade['symbol']}")
                    print(f"   Entry: ${closed_trade['price']:.2f}")

                    # ดึงข้อมูลการปิดออเดอร์
                    deals = mt5.history_deals_get(
                        datetime.now() - timedelta(hours=1), datetime.now()
                    )

                    if deals:
                        for deal in deals:
                            if deal.position_id == ticket and deal.entry == 1:
                                profit = deal.profit
                                close_price = deal.price
                                close_time = datetime.fromtimestamp(deal.time)

                                print(f"   Exit: ${close_price:.2f}")
                                print(f"   P&L: ${profit:.2f}")
                                print(f"   Result: {'✅ WIN' if profit > 0 else '❌ LOSS'}")
                                break

                    del self.active_positions[ticket]
                    self.last_trade_closed_time = datetime.now()

            if closed_tickets:
                print(f"🎯 {len(closed_tickets)} Gold position(s) closed. Ready for new trades.")

            # คำนวณ P&L รวม
            total_profit = sum(pos.profit for pos in positions)
            self.daily_pnl = total_profit

        except Exception as e:
            print(f"❌ Gold position update error: {str(e)}")

    def start_gold_auto_trading(self, symbol: str = "XAUUSD", update_interval: int = 60):
        """Start XAUUSD automated trading system"""

        print("🚀 Starting XAUUSD SMC Auto Trading System")
        print("=" * 60)
        print(f"🥇 Symbol: {symbol}")
        print(f"🎯 Trading Status: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print(f"⏳ Mode: One Gold trade at a time")
        print("=" * 60)

        last_signal = None

        while True:
            try:
                self.update_gold_positions()

                print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Analyzing Gold signals...")
                signal = self.signal_engine.get_multi_timeframe_signals(symbol)

                if "error" in signal:
                    print(f"❌ Gold signal error: {signal['error']}")
                else:
                    print(f"🥇 {symbol}: {signal['final_direction']} | Confidence: {signal['final_confidence']:.3f}")
                    print(f"🎯 Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}")
                    print(f"💰 Daily P&L: ${self.daily_pnl:.2f}")

                    # ตรวจสอบการเปลี่ยนแปลงสัญญาณ
                    signal_changed = self._is_signal_changed(last_signal, signal)

                    if signal_changed and signal["trading_recommendation"] == "TRADE":
                        if self.trading_enabled:
                            print("🔥 NEW GOLD TRADING SIGNAL DETECTED!")
                            success = self.process_gold_signal(signal, symbol)
                            if success:
                                print("✅ Gold auto trade executed successfully")
                            else:
                                print("❌ Gold auto trade failed or blocked")
                        else:
                            print("📊 GOLD TRADING SIGNAL (Trading disabled)")

                    last_signal = signal

                # แสดงสถานะ position ที่เปิดอยู่
                if self.active_positions:
                    for ticket, trade_info in self.active_positions.items():
                        positions = mt5.positions_get(ticket=ticket)
                        if positions:
                            pos = positions[0]
                            print(f"🔄 Active Gold: {trade_info['type']} {trade_info['volume']} {trade_info['symbol']}")
                            print(f"   Entry: ${trade_info['price']:.2f} | Current P&L: ${pos.profit:.2f}")

                time.sleep(update_interval)

            except KeyboardInterrupt:
                print("\n🛑 Gold auto trading stopped by user")
                break
            except Exception as e:
                print(f"❌ Gold auto trading error: {str(e)}")
                time.sleep(10)

        print("✅ Gold auto trading system stopped")

    def _is_signal_changed(self, last_signal: Optional[Dict], current_signal: Dict) -> bool:
        """Determine if Gold signal has changed enough to warrant new trade"""

        if last_signal is None:
            if self.enable_first_signal_trade:
                return (
                    current_signal["final_confidence"] >= self.first_signal_min_confidence
                    and current_signal["trading_recommendation"] == "TRADE"
                )
            else:
                return False

        # ตรวจสอบการเปลี่ยนทิศทาง
        if last_signal["final_direction"] != current_signal["final_direction"]:
            return True

        # ตรวจสอบการเปลี่ยน confidence
        confidence_change = abs(
            last_signal["final_confidence"] - current_signal["final_confidence"]
        )
        if confidence_change > self.signal_change_threshold:
            return True

        # สัญญาณแรงสำหรับทองคำ
        if (
            current_signal["final_confidence"] >= 0.85
            and current_signal["trading_recommendation"] == "TRADE"
            and len(self.trade_history) == 0
        ):
            print(f"🔥 Force trading high confidence Gold signal: {current_signal['final_confidence']:.3f}")
            return True

        return False

    def print_gold_settings(self):
        """Print current Gold trading configuration"""
        print("⚙️ XAUUSD Auto Trader Settings:")
        print("=" * 50)
        print(f"🥇 Trading Gold (XAUUSD)")
        print(f"🎯 Max concurrent trades: {self.max_concurrent_trades}")
        print(f"⏳ Wait for completion: {'YES' if self.wait_for_trade_completion else 'NO'}")
        print(f"📊 Min confidence: {self.min_confidence*100}%")
        print(f"🤝 Min consensus: {self.min_consensus}/5")
        print(f"💰 Base lot size: {self.base_lot_size}")
        print(f"🎯 Default SL: {self.default_sl_pips} points")
        print(f"📈 TP Ratio: {self.default_tp_ratio}:1")
        print("=" * 50)


# Main execution สำหรับ Gold Trading
if __name__ == "__main__":
    print("🥇 XAUUSD SMC Auto Trading Bot")
    print("=" * 50)

    # Settings สำหรับทองคำ
    MODELS_PATH = "XAUUSD_v_SMC"  # เปลี่ยนเป็น path ที่ถูกต้อง
    SYMBOL = "XAUUSD"  # หรือ "XAUUSD.c" ขึ้นอยู่กับโบรกเกอร์
    
    SIGNAL_CHANGE_THRESHOLD = 0.001
    ENABLE_FIRST_TRADE = True
    FIRST_TRADE_MIN_CONFIDENCE = 0.75
    MIN_CONFIDENCE = 0.75
    MIN_CONSENSUS = 3
    MAX_CONCURRENT_TRADES = 1
    WAIT_FOR_COMPLETION = True
    BASE_LOT_SIZE = 0.01
    MAX_LOT_SIZE = 0.1

    # Initialize Gold trader
    gold_trader = XAUUSDAutoTrader(
        models_path=MODELS_PATH,
        signal_change_threshold=SIGNAL_CHANGE_THRESHOLD,
        enable_first_signal_trade=ENABLE_FIRST_TRADE,
        first_signal_min_confidence=FIRST_TRADE_MIN_CONFIDENCE,
        min_confidence=MIN_CONFIDENCE,
        min_consensus=MIN_CONSENSUS,
        max_concurrent_trades=MAX_CONCURRENT_TRADES,
        wait_for_trade_completion=WAIT_FOR_COMPLETION,
        base_lot_size=BASE_LOT_SIZE,
        max_lot_size=MAX_LOT_SIZE,
        default_sl_pips=200,  # ทองคำใช้ SL ใหญ่กว่า (200 points = $2.00)
        default_tp_ratio=2.0,
    )

    gold_trader.print_gold_settings()

    if gold_trader.connect_mt5():
        if gold_trader.load_models():
            print("\n🎯 Gold Auto Trading Bot Ready!")

            enable_trading = input("\n🚀 Enable LIVE GOLD TRADING? (yes/no): ").lower().strip()

            if enable_trading == "yes":
                gold_trader.enable_trading(True)
            else:
                print("📊 Demo mode - signals only")

            # เริ่มเทรดทองคำ
            gold_trader.start_gold_auto_trading(SYMBOL, 60)

        else:
            print("❌ Failed to load XAUUSD AI models")
    else:
        print("❌ Failed to connect to MT5")