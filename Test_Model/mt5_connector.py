import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import asyncio
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    """การตั้งค่าเชื่อมต่อ MT5"""
    login: int
    password: str
    server: str
    timeout: int = 60000
    path: str = ""

@dataclass
class MT5Position:
    """ข้อมูลโพซิชั่นจาก MT5"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    comment: str
    time: datetime
    magic: int = 0

class MT5Connector:
    """คลาสหลักสำหรับเชื่อมต่อและควบคุม MT5"""
    
    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config
        self.is_connected = False
        self.account_info = None
        self.symbols_info = {}
        self.last_prices = {}
        
    async def auto_connect(self) -> bool:
        """เชื่อมต่อ MT5 อัตโนมัติ (ใช้บัญชีที่ login ไว้แล้ว)"""
        try:
            logger.info("🔄 Attempting MT5 Auto-Connection...")
            
            # เริ่มต้น MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialize failed: {error_code}")
                return False
            
            # ตรวจสอบว่ามีการ login อยู่แล้วหรือไม่
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ No active MT5 account found. Please login to MT5 first.")
                mt5.shutdown()
                return False
            
            # ตรวจสอบการเชื่อมต่อ terminal
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("❌ Cannot get MT5 terminal info")
                mt5.shutdown()
                return False
            
            # เก็บข้อมูลบัญชี
            self.account_info = account_info
            self.is_connected = True
            
            logger.info("🎉 MT5 Auto-Connected Successfully!")
            logger.info(f"📊 Account: {account_info.login}")
            logger.info(f"💰 Balance: ${account_info.balance:,.2f}")
            logger.info(f"📈 Equity: ${account_info.equity:,.2f}")
            logger.info(f"🏢 Broker: {account_info.company}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error auto-connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """ตัดการเชื่อมต่อ MT5"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                logger.info("🔌 Disconnected from MT5")
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    def get_account_info(self) -> Optional[Dict]:
        """ได้ข้อมูลบัญชีแบบ Real-time"""
        try:
            if not self.is_connected:
                return None
            
            # ดึงข้อมูลล่าสุดจาก MT5
            account = mt5.account_info()
            if account is None:
                logger.warning("Cannot retrieve account info - connection lost?")
                return None
            
            # อัพเดทข้อมูลใน cache
            self.account_info = account
            
            # คำนวณข้อมูลเพิ่มเติม
            margin_level = 0
            if account.margin > 0:
                margin_level = (account.equity / account.margin) * 100
            
            profit_percentage = 0
            if account.balance > 0:
                profit_percentage = (account.profit / account.balance) * 100
            
            return {
                "login": account.login,
                "balance": round(account.balance, 2),
                "equity": round(account.equity, 2),
                "margin": round(account.margin, 2),
                "free_margin": round(account.margin_free, 2),
                "margin_level": round(margin_level, 2),
                "profit": round(account.profit, 2),
                "profit_percentage": round(profit_percentage, 2),
                "currency": account.currency,
                "leverage": account.leverage,
                "server": account.server,
                "company": account.company,
                "name": account.name,
                "trade_allowed": account.trade_allowed,
                "trade_expert": account.trade_expert,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """ได้ข้อมูลคู่สกุลเงิน"""
        try:
            if not self.is_connected:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol {symbol} not found")
                return None
            
            return {
                "name": symbol_info.name,
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "spread": symbol_info.spread,
                "digits": symbol_info.digits,
                "point": symbol_info.point,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
                "contract_size": symbol_info.trade_contract_size,
                "margin_initial": symbol_info.margin_initial,
                "swap_long": symbol_info.swap_long,
                "swap_short": symbol_info.swap_short
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """ได้ราคาปัจจุบันของคู่สกุลเงิน"""
        try:
            if not self.is_connected:
                return {}
            
            prices = {}
            for symbol in symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    prices[symbol] = {
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": tick.ask - tick.bid,
                        "time": datetime.fromtimestamp(tick.time),
                        "volume": tick.volume if hasattr(tick, 'volume') else 0
                    }
                    
                    # เก็บราคาล่าสุด
                    self.last_prices[symbol] = prices[symbol]
                else:
                    logger.warning(f"Cannot get tick data for {symbol}")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def get_positions(self) -> List[MT5Position]:
        """ได้รายการโพซิชั่นปัจจุบัน"""
        try:
            if not self.is_connected:
                return []
            
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            mt5_positions = []
            for pos in positions:
                mt5_pos = MT5Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission,
                    comment=pos.comment,
                    time=datetime.fromtimestamp(pos.time),
                    magic=pos.magic
                )
                mt5_positions.append(mt5_pos)
            
            return mt5_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def place_market_order(self, symbol: str, order_type: str, volume: float, 
                          comment: str = "AI_Recovery_Bot", magic: int = 234000) -> Optional[Dict]:
        """วางออเดอร์ Market Order"""
        try:
            if not self.is_connected:
                logger.error("Not connected to MT5")
                return None
            
            # ตรวจสอบว่า symbol มีอยู่
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # เปิดใช้งาน symbol
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to enable symbol {symbol}")
                    return None
            
            # กำหนดประเภทออเดอร์
            if order_type.upper() == "BUY":
                trade_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            elif order_type.upper() == "SELL":
                trade_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None
            
            # ปรับปรุงขนาด volume ให้ตรงกับข้อกำหนด
            volume = self._normalize_volume(symbol, volume)
            
            # สร้างคำขอ
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "price": price,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # ส่งออเดอร์
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"✅ Order placed successfully: {order_type} {volume} {symbol} at {result.price}")
            
            return {
                "ticket": result.order,
                "symbol": symbol,
                "type": order_type,
                "volume": volume,
                "price": result.price,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, ticket: int) -> bool:
        """ปิดโพซิชั่น"""
        try:
            if not self.is_connected:
                return False
            
            # หาข้อมูลโพซิชั่น
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            
            # กำหนดประเภทออเดอร์ปิด
            if pos.type == mt5.POSITION_TYPE_BUY:
                trade_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(pos.symbol).bid
            else:
                trade_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).ask
            
            # สร้างคำขอปิดโพซิชั่น
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": trade_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "AI_Recovery_Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # ส่งคำขอปิด
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Close position failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"✅ Position {ticket} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """ปิดโพซิชั่นทั้งหมด"""
        try:
            positions = self.get_positions()
            closed_count = 0
            
            for pos in positions:
                # ถ้าระบุ symbol แล้วปิดเฉพาะ symbol นั้น
                if symbol and pos.symbol != symbol:
                    continue
                
                if self.close_position(pos.ticket):
                    closed_count += 1
                    # รอสักพักก่อนปิดตัวต่อไป
                    time.sleep(0.1)
            
            logger.info(f"✅ Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """ปรับปรุงขนาด volume ให้ถูกต้อง"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return volume
            
            # ปรับให้อยู่ในช่วงที่อนุญาต
            min_vol = symbol_info.volume_min
            max_vol = symbol_info.volume_max
            step = symbol_info.volume_step
            
            # ปรับให้ไม่น้อยกว่า minimum
            if volume < min_vol:
                volume = min_vol
            
            # ปรับให้ไม่เกิน maximum
            if volume > max_vol:
                volume = max_vol
            
            # ปรับให้ตรงกับ step
            volume = round(volume / step) * step
            
            return volume
            
        except Exception as e:
            logger.error(f"Error normalizing volume: {e}")
            return volume
    
    def check_market_hours(self) -> bool:
        """ตรวจสอบว่าตลาดเปิดอยู่หรือไม่"""
        try:
            # ตรวจสอบเวลาปัจจุบัน
            now = datetime.now()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # ตลาด Forex ปิดตั้งแต่วันเสาร์ถึงอาทิตย์
            if weekday == 5:  # Saturday
                return False
            elif weekday == 6:  # Sunday
                # เปิดหลัง 17:00 UTC วันอาทิตย์
                return now.hour >= 17
            else:
                # วันธรรมดาเปิดตลอด ยกเว้นวันศุกร์หลัง 17:00 UTC
                if weekday == 4 and now.hour >= 17:  # Friday after 17:00
                    return False
                return True
                
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return True  # Default เปิด

# ฟังก์ชันสำหรับ API Server ใช้
async def initialize_mt5_auto():
    """เริ่มต้น MT5 แบบอัตโนมัติสำหรับ API Server"""
    connector = MT5Connector()
    
    if await connector.auto_connect():
        logger.info("✅ MT5 Auto-Connected for API Server")
        return connector
    else:
        logger.error("❌ Failed to auto-connect MT5")
        return None

# ตัวอย่างการใช้งาน AUTO CONNECT
async def test_auto_connection():
    """ทดสอบการเชื่อมต่อแบบอัตโนมัติ"""
    connector = MT5Connector()
    
    print("🔄 Attempting MT5 Auto-Connection...")
    print("📝 Make sure MT5 is running and logged in first!")
    print("-" * 50)
    
    try:
        # เชื่อมต่อแบบอัตโนมัติ
        if await connector.auto_connect():
            print("✅ Auto-Connected Successfully!")
            print()
            
            # แสดงข้อมูลบัญชีแบบละเอียด
            account = connector.get_account_info()
            if account:
                print("📊 Account Information:")
                print(f"   Account Number: {account['login']}")
                print(f"   Account Name: {account.get('name', 'N/A')}")
                print(f"   Broker: {account['company']}")
                print(f"   Server: {account['server']}")
                print(f"   Currency: {account['currency']}")
                print(f"   Leverage: 1:{account['leverage']}")
                print()
                print("💰 Financial Information:")
                print(f"   Balance: ${account['balance']:,.2f}")
                print(f"   Equity: ${account['equity']:,.2f}")
                print(f"   Margin Used: ${account['margin']:,.2f}")
                print(f"   Free Margin: ${account['free_margin']:,.2f}")
                print(f"   Margin Level: {account['margin_level']:.2f}%")
                print(f"   Current Profit: ${account['profit']:,.2f} ({account['profit_percentage']:+.2f}%)")
                print()
                print("⚙️ Trading Settings:")
                print(f"   Trading Allowed: {'✅ Yes' if account['trade_allowed'] else '❌ No'}")
                print(f"   Expert Advisors: {'✅ Enabled' if account['trade_expert'] else '❌ Disabled'}")
            
            # ทดสอบดึงราคา
            print("\n📈 Testing Price Data:")
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            prices = connector.get_current_prices(symbols)
            
            for symbol, price_data in prices.items():
                spread_pips = price_data['spread'] * (10000 if 'JPY' not in symbol and 'XAU' not in symbol else 100 if 'JPY' in symbol else 10)
                decimals = 5 if 'JPY' not in symbol and 'XAU' not in symbol else 3 if 'JPY' in symbol else 2
                print(f"   {symbol}: {price_data['bid']:.{decimals}f} / {price_data['ask']:.{decimals}f} (Spread: {spread_pips:.1f} pips)")
            
            # ทดสอบดูโพซิชั่น
            positions = connector.get_positions()
            print(f"\n🎯 Current Positions: {len(positions)}")
            
            if positions:
                for pos in positions:
                    direction = "📈 BUY" if pos.type == 0 else "📉 SELL"
                    profit_color = "🟢" if pos.profit >= 0 else "🔴"
                    print(f"   {direction} {pos.symbol} {pos.volume} lots")
                    print(f"   Entry: {pos.price_open:.5f} | Current: {pos.price_current:.5f}")
                    print(f"   Profit: {profit_color} ${pos.profit:.2f}")
                    print(f"   Comment: {pos.comment}")
                    print()
            else:
                print("   No open positions")
            
            # ตรวจสอบเวลาตลาด
            market_open = connector.check_market_hours()
            print(f"\n🕐 Market Status: {'🟢 Open' if market_open else '🔴 Closed'}")
            
        else:
            print("❌ Auto-Connection Failed!")
            print()
            print("💡 Troubleshooting Tips:")
            print("   1. Make sure MT5 is running")
            print("   2. Login to your MT5 account first")
            print("   3. Enable 'Allow automated trading' in MT5 settings:")
            print("      - Tools → Options → Expert Advisors")
            print("      - ✅ Allow automated trading")
            print("      - ✅ Allow DLL imports")
            print("   4. Check if Python can access MT5 (run as administrator if needed)")
            print("   5. Make sure MT5 terminal is not busy (close other EAs)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Common Solutions:")
        print("   - Restart MT5 terminal")
        print("   - Check Windows Defender/Antivirus")
        print("   - Run Python as Administrator")
        print("   - Update MetaTrader5 Python package: pip install --upgrade MetaTrader5")
    
    finally:
        connector.disconnect()

if __name__ == "__main__":
    # รันทดสอบแบบ Auto Connect
    print("🚀 MT5 Connector Test Suite")
    print("=" * 50)
    asyncio.run(test_auto_connection())