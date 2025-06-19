import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    """การตั้งค่าเชื่อมต่อ MT5"""
    login: int = 0
    password: str = ""
    server: str = ""
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
    """คลาสหลักสำหรับเชื่อมต่อและควบคุม MT5 - Thread Safe"""
    
    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config
        self.is_connected = False
        self.account_info = None
        self.symbols_info = {}
        self.last_prices = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._connection_attempts = 0
        self._max_attempts = 3
        
    async def auto_connect(self) -> bool:
        """เชื่อมต่อ MT5 อัตโนมัติ - Thread Safe"""
        with self._lock:
            try:
                self._connection_attempts += 1
                logger.info(f"🔄 MT5 Connection Attempt {self._connection_attempts}/{self._max_attempts}")
                
                # รัน MT5 initialization ใน thread pool
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(self._executor, self._sync_connect)
                
                if success:
                    self.is_connected = True
                    self._connection_attempts = 0
                    logger.info("🎉 MT5 Auto-Connected Successfully!")
                    await self._log_connection_info()
                    return True
                else:
                    if self._connection_attempts >= self._max_attempts:
                        logger.error("❌ Max connection attempts reached")
                        return False
                    
                    # Retry with exponential backoff
                    retry_delay = min(2 ** self._connection_attempts, 30)
                    logger.warning(f"⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    return await self.auto_connect()
                    
            except Exception as e:
                logger.error(f"❌ Error auto-connecting to MT5: {e}")
                return False
    
    def _sync_connect(self) -> bool:
        """Synchronous MT5 connection for thread pool"""
        try:
            # เริ่มต้น MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialize failed: {error_code}")
                return False
            
            # ตรวจสอบบัญชี
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ No active MT5 account found")
                mt5.shutdown()
                return False
            
            # ตรวจสอบ terminal
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("❌ Cannot get MT5 terminal info")
                mt5.shutdown()
                return False
            
            # เก็บข้อมูลบัญชี
            self.account_info = account_info
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync connect error: {e}")
            return False
    
    async def _log_connection_info(self):
        """Log connection information"""
        if self.account_info:
            logger.info(f"📊 Account: {self.account_info.login}")
            logger.info(f"💰 Balance: ${self.account_info.balance:,.2f}")
            logger.info(f"📈 Equity: ${self.account_info.equity:,.2f}")
            logger.info(f"🏢 Broker: {self.account_info.company}")
    
    async def disconnect(self):
        """ตัดการเชื่อมต่อ MT5 - Async Safe"""
        with self._lock:
            try:
                if self.is_connected:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self._executor, mt5.shutdown)
                    self.is_connected = False
                    logger.info("🔌 Disconnected from MT5")
            except Exception as e:
                logger.error(f"Error disconnecting from MT5: {e}")
            finally:
                self._executor.shutdown(wait=False)
    
    async def get_account_info(self) -> Optional[Dict]:
        """ได้ข้อมูลบัญชีแบบ Async"""
        if not self.is_connected:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            account = await loop.run_in_executor(self._executor, mt5.account_info)
            
            if account is None:
                logger.warning("Cannot retrieve account info - connection lost?")
                self.is_connected = False
                return None
            
            # อัพเดทข้อมูลใน cache
            self.account_info = account
            
            # คำนวณข้อมูลเพิ่มเติม
            margin_level = (account.equity / account.margin * 100) if account.margin > 0 else 0
            profit_percentage = (account.profit / account.balance * 100) if account.balance > 0 else 0
            
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
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """ได้ราคาปัจจุบันแบบ Async"""
        if not self.is_connected:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            prices = await loop.run_in_executor(
                self._executor, 
                self._get_prices_sync, 
                symbols
            )
            return prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def _get_prices_sync(self, symbols: List[str]) -> Dict[str, Dict]:
        """Synchronous price fetching"""
        prices = {}
        for symbol in symbols:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    prices[symbol] = {
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": tick.ask - tick.bid,
                        "time": datetime.fromtimestamp(tick.time),
                        "volume": getattr(tick, 'volume', 0)
                    }
                    self.last_prices[symbol] = prices[symbol]
            except Exception as e:
                logger.warning(f"Cannot get tick data for {symbol}: {e}")
        return prices
    
    async def get_positions(self) -> List[MT5Position]:
        """ได้รายการโพซิชั่นแบบ Async"""
        if not self.is_connected:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(self._executor, self._get_positions_sync)
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _get_positions_sync(self) -> List[MT5Position]:
        """Synchronous positions fetching"""
        try:
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
            logger.error(f"Error in _get_positions_sync: {e}")
            return []
    
    async def place_market_order(self, symbol: str, order_type: str, volume: float, 
                                comment: str = "AI_Recovery_Bot", magic: int = 234000) -> Optional[Dict]:
        """วางออเดอร์ Market Order แบบ Async"""
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._place_order_sync,
                symbol, order_type, volume, comment, magic
            )
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _place_order_sync(self, symbol: str, order_type: str, volume: float, 
                         comment: str, magic: int) -> Optional[Dict]:
        """Synchronous order placement"""
        try:
            # ตรวจสอบ symbol
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # เปิดใช้งาน symbol ถ้าจำเป็น
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to enable symbol {symbol}")
                    return None
            
            # กำหนดประเภทออเดอร์และราคา
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get tick data for {symbol}")
                return None
            
            if order_type.upper() == "BUY":
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif order_type.upper() == "SELL":
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None
            
            # ปรับปรุงขนาด volume
            volume = self._normalize_volume_sync(symbol, volume)
            
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
            
            logger.info(f"✅ Order placed: {order_type} {volume} {symbol} at {result.price}")
            
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
            logger.error(f"Error in _place_order_sync: {e}")
            return None
    
    async def close_position(self, ticket: int) -> bool:
        """ปิดโพซิชั่นแบบ Async"""
        if not self.is_connected:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, self._close_position_sync, ticket)
            return result
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def _close_position_sync(self, ticket: int) -> bool:
        """Synchronous position closing"""
        try:
            # หาข้อมูลโพซิชั่น
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            
            # กำหนดประเภทออเดอร์ปิดและราคา
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logger.error(f"Cannot get tick data for {pos.symbol}")
                return False
            
            if pos.type == mt5.POSITION_TYPE_BUY:
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
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
            logger.error(f"Error in _close_position_sync: {e}")
            return False
    
    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """ปิดโพซิชั่นทั้งหมดแบบ Async"""
        try:
            positions = await self.get_positions()
            closed_count = 0
            
            for pos in positions:
                # ถ้าระบุ symbol แล้วปิดเฉพาะ symbol นั้น
                if symbol and pos.symbol != symbol:
                    continue
                
                if await self.close_position(pos.ticket):
                    closed_count += 1
                    # รอสักพักก่อนปิดตัวต่อไป
                    await asyncio.sleep(0.1)
            
            logger.info(f"✅ Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    def _normalize_volume_sync(self, symbol: str, volume: float) -> float:
        """ปรับปรุงขนาด volume ให้ถูกต้อง - Sync version"""
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
    
    async def check_market_hours(self) -> bool:
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
    
    async def health_check(self) -> Dict[str, Union[bool, str]]:
        """ตรวจสอบสุขภาพของการเชื่อมต่อ"""
        try:
            if not self.is_connected:
                return {"connected": False, "status": "Not connected"}
            
            # ทดสอบดึงข้อมูลบัญชี
            account_info = await self.get_account_info()
            if account_info is None:
                self.is_connected = False
                return {"connected": False, "status": "Connection lost"}
            
            # ทดสอบดึงราคา
            test_symbols = ['EURUSD']
            prices = await self.get_current_prices(test_symbols)
            
            return {
                "connected": True,
                "status": "Healthy",
                "account_login": account_info.get('login'),
                "prices_available": len(prices) > 0,
                "market_open": await self.check_market_hours()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"connected": False, "status": f"Error: {str(e)}"}

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

# ตัวอย่างการใช้งาน
async def test_improved_connection():
    """ทดสอบการเชื่อมต่อแบบใหม่"""
    connector = MT5Connector()
    
    print("🔄 Testing Improved MT5 Connection...")
    print("-" * 50)
    
    try:
        # เชื่อมต่อ
        if await connector.auto_connect():
            print("✅ Connected Successfully!")
            
            # ทดสอบ health check
            health = await connector.health_check()
            print(f"🔍 Health Check: {health}")
            
            # ทดสอบดึงข้อมูลแบบ concurrent
            tasks = [
                connector.get_account_info(),
                connector.get_current_prices(['EURUSD', 'GBPUSD']),
                connector.get_positions()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            account_info, prices, positions = results
            
            print(f"📊 Account: {account_info.get('login') if account_info else 'Error'}")
            print(f"📈 Prices: {len(prices) if isinstance(prices, dict) else 'Error'} symbols")
            print(f"🎯 Positions: {len(positions) if isinstance(positions, list) else 'Error'}")
            
        else:
            print("❌ Connection Failed!")
            
    except Exception as e:
        print(f"❌ Test Error: {e}")
    
    finally:
        await connector.disconnect()

if __name__ == "__main__":
    asyncio.run(test_improved_connection())