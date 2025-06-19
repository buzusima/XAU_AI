import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod
from mt5_connector import MT5Connector, MT5Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    RISK_OFF = "risk_off"

class RecoveryAction(Enum):
    INITIAL_ENTRY = "initial_entry"
    CORRELATION_RECOVERY = "correlation_recovery"
    GRID_RECOVERY = "grid_recovery"
    BASKET_RECOVERY = "basket_recovery"
    ARBITRAGE_RECOVERY = "arbitrage_recovery"
    HOLD_POSITION = "hold_position"
    CLOSE_ALL = "close_all"

@dataclass
class TradingSettings:
    account_balance: float = 5000.0
    daily_target_pct: float = 2.0
    monthly_target_pct: float = 20.0
    max_recovery_levels: int = 5
    recovery_multiplier: float = 1.5
    max_portfolio_risk: float = 30.0
    initial_lot_size: float = 0.01
    enabled_pairs: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    use_correlation_recovery: bool = True
    use_grid_recovery: bool = True
    use_basket_recovery: bool = True
    use_arbitrage_recovery: bool = True

@dataclass
class Position:
    symbol: str
    direction: str  # "BUY" or "SELL"
    size: float
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    recovery_level: int = 0
    is_recovery: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    parent_position_id: Optional[str] = None
    position_id: str = field(default_factory=lambda: f"pos_{datetime.now().timestamp()}")
    mt5_ticket: Optional[int] = None  # MT5 ticket number

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: float = 0.0

class MT5CurrencyPairs:
    """คลาสสำหรับจัดการคู่สกุลเงิน MT5"""
    
    MAJORS = {
        'EURUSD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'GBPUSD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'USDJPY': {'pip_value': 0.01, 'digits': 3, 'min_lot': 0.01},
        'USDCHF': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'AUDUSD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'NZDUSD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'USDCAD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
    }
    
    CROSSES = {
        'EURGBP': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'EURJPY': {'pip_value': 0.01, 'digits': 3, 'min_lot': 0.01},
        'EURCHF': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'EURAUD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'EURCAD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'GBPJPY': {'pip_value': 0.01, 'digits': 3, 'min_lot': 0.01},
        'GBPCHF': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'GBPAUD': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'AUDJPY': {'pip_value': 0.01, 'digits': 3, 'min_lot': 0.01},
        'AUDCHF': {'pip_value': 0.0001, 'digits': 5, 'min_lot': 0.01},
        'CHFJPY': {'pip_value': 0.01, 'digits': 3, 'min_lot': 0.01},
    }
    
    METALS = {
        'XAUUSD': {'pip_value': 0.01, 'digits': 2, 'min_lot': 0.01},
        'XAGUSD': {'pip_value': 0.001, 'digits': 3, 'min_lot': 0.01},
    }
    
    @classmethod
    def get_all_pairs(cls) -> Dict[str, Dict]:
        """รวมคู่สกุลเงินทั้งหมด"""
        return {**cls.MAJORS, **cls.CROSSES, **cls.METALS}
    
    @classmethod
    def get_pip_value(cls, symbol: str) -> float:
        """หาค่า pip ของคู่สกุลเงิน"""
        all_pairs = cls.get_all_pairs()
        return all_pairs.get(symbol, {}).get('pip_value', 0.0001)

class CorrelationMatrix:
    """คลาสสำหรับจัดการ Correlation ระหว่างคู่สกุลเงิน"""
    
    def __init__(self):
        self.correlation_groups = {
            'USD_BASKET': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
            'EUR_BASKET': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD'],
            'GBP_BASKET': ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF', 'GBPAUD'],
            'JPY_BASKET': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY'],
            'COMMODITY_BASKET': ['AUDUSD', 'NZDUSD', 'USDCAD', 'XAUUSD'],
            'SAFE_HAVEN': ['USDJPY', 'USDCHF', 'CHFJPY', 'XAUUSD']
        }
        
        # Correlation coefficients (simplified - in real system would be calculated dynamically)
        self.correlation_matrix = {
            'EURUSD': {'GBPUSD': 0.7, 'USDCHF': -0.8, 'USDJPY': -0.6},
            'GBPUSD': {'EURUSD': 0.7, 'GBPJPY': 0.8, 'EURGBP': -0.5},
            'USDJPY': {'EURJPY': 0.9, 'GBPJPY': 0.9, 'AUDJPY': 0.8},
            'XAUUSD': {'AUDUSD': 0.6, 'USDCHF': -0.4, 'USDJPY': -0.3}
        }
    
    def get_correlated_pairs(self, base_pair: str, min_correlation: float = 0.5) -> List[str]:
        """หาคู่สกุลเงินที่มี correlation สูง"""
        correlated = []
        if base_pair in self.correlation_matrix:
            for pair, corr in self.correlation_matrix[base_pair].items():
                if abs(corr) >= min_correlation:
                    correlated.append(pair)
        return correlated
    
    def get_hedge_pairs(self, base_pair: str, min_negative_correlation: float = -0.5) -> List[str]:
        """หาคู่สกุลเงินสำหรับ hedging (negative correlation)"""
        hedge_pairs = []
        if base_pair in self.correlation_matrix:
            for pair, corr in self.correlation_matrix[base_pair].items():
                if corr <= min_negative_correlation:
                    hedge_pairs.append(pair)
        return hedge_pairs

class RiskManager:
    """คลาสสำหรับจัดการความเสี่ยง"""
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.monthly_pnl = 0.0
    
    def calculate_position_size(self, symbol: str, recovery_level: int = 0) -> float:
        """คำนวณขนาดโพซิชั่น"""
        base_size = self.settings.initial_lot_size
        
        if recovery_level > 0:
            # เพิ่มขนาดตาม recovery multiplier
            multiplier = self.settings.recovery_multiplier ** recovery_level
            return base_size * multiplier
        
        return base_size
    
    def check_risk_limits(self, total_exposure: float, unrealized_pnl: float) -> bool:
        """ตรวจสอบขีดจำกัดความเสี่ยง"""
        # ตรวจสอบ portfolio risk
        risk_pct = (abs(unrealized_pnl) / self.settings.account_balance) * 100
        if risk_pct > self.settings.max_portfolio_risk:
            logger.warning(f"Portfolio risk {risk_pct:.2f}% exceeds limit {self.settings.max_portfolio_risk}%")
            return False
        
        # ตรวจสอบเป้าหมายกำไรรายวัน
        daily_target_amount = (self.settings.account_balance * self.settings.daily_target_pct) / 100
        if self.daily_pnl >= daily_target_amount:
            logger.info(f"Daily target reached: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    def update_pnl(self, realized_pnl: float):
        """อัพเดท P&L"""
        self.daily_pnl += realized_pnl
        self.monthly_pnl += realized_pnl

class RecoveryEngine:
    """เครื่องยนต์หลักสำหรับ Recovery Trading"""
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.correlation_matrix = CorrelationMatrix()
        self.risk_manager = RiskManager(settings)
        self.current_regime = MarketRegime.RANGING
        self.is_running = False
        
        # MT5 Integration
        self.mt5_connector = None
        self.mt5_positions = {}  # Map position_id to MT5 ticket
        
    async def initialize_mt5(self):
        """เริ่มต้นการเชื่อมต่อ MT5"""
        try:
            self.mt5_connector = MT5Connector()
            if await self.mt5_connector.auto_connect():
                logger.info("✅ MT5 Auto-Connected for AI Engine")
                return True
            else:
                logger.error("❌ Failed to connect MT5")
                return False
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False
        
    async def start_engine(self):
        """เริ่มต้น AI Engine"""
        # เชื่อมต่อ MT5 ก่อน
        if not await self.initialize_mt5():
            logger.error("Cannot start AI Engine without MT5 connection")
            return False
            
        self.is_running = True
        logger.info("🚀 AI Recovery Engine Started with Live MT5 Connection")
        
        # เริ่ม main trading loop
        await self.main_trading_loop()
        return True
    
    async def stop_engine(self):
        """หยุด AI Engine"""
        self.is_running = False
        if self.mt5_connector:
            self.mt5_connector.disconnect()
        logger.info("⏹️ AI Recovery Engine Stopped")
    
    async def main_trading_loop(self):
        """Loop หลักของการเทรด"""
        while self.is_running:
            try:
                # 1. อัพเดทข้อมูลตลาดจาก MT5
                await self.update_market_data_from_mt5()
                
                # 2. อัพเดท P&L ของโพซิชั่นจาก MT5
                await self.update_positions_from_mt5()
                
                # 3. วิเคราะห์สถานการณ์ตลาด
                await self.analyze_market_regime()
                
                # 4. ตรวจสอบโอกาสในการเทรด
                await self.check_trading_opportunities()
                
                # 5. ดำเนินการแก้ไม้ถ้าจำเป็น
                await self.execute_recovery_strategy()
                
                # 6. ตรวจสอบเงื่อนไขปิดโพซิชั่น
                await self.check_exit_conditions()
                
                # รอ 2 วินาทีก่อนรอบต่อไป
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)
    
    async def update_market_data_from_mt5(self):
        """อัพเดทข้อมูลตลาดจาก MT5"""
        if not self.mt5_connector or not self.mt5_connector.is_connected:
            return
            
        try:
            # ดึงราคาปัจจุบันจาก MT5
            prices = self.mt5_connector.get_current_prices(self.settings.enabled_pairs)
            
            for symbol, price_data in prices.items():
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    bid=price_data['bid'],
                    ask=price_data['ask'],
                    spread=price_data['spread'],
                    timestamp=price_data['time'],
                    volume=price_data.get('volume', 0)
                )
                
        except Exception as e:
            logger.error(f"Error updating market data from MT5: {e}")
    
    async def update_positions_from_mt5(self):
        """อัพเดทโพซิชั่นจาก MT5"""
        if not self.mt5_connector or not self.mt5_connector.is_connected:
            return
            
        try:
            # ดึงโพซิชั่นจาก MT5
            mt5_positions = self.mt5_connector.get_positions()
            
            # อัพเดทโพซิชั่นที่มีอยู่
            for pos_id, position in self.positions.items():
                if position.mt5_ticket:
                    # หาโพซิชั่นที่ตรงกันใน MT5
                    mt5_pos = next((p for p in mt5_positions if p.ticket == position.mt5_ticket), None)
                    
                    if mt5_pos:
                        # อัพเดทข้อมูลจาก MT5
                        position.current_price = mt5_pos.price_current
                        position.pnl = mt5_pos.profit
                        
                        # คำนวณ pips
                        pip_value = MT5CurrencyPairs.get_pip_value(position.symbol)
                        if position.direction == "BUY":
                            position.pnl_pips = (position.current_price - position.entry_price) / pip_value
                        else:
                            position.pnl_pips = (position.entry_price - position.current_price) / pip_value
                    else:
                        # โพซิชั่นถูกปิดใน MT5 แล้ว
                        logger.info(f"Position {position.symbol} closed in MT5")
                        # ลบออกจากระบบ
                        del self.positions[pos_id]
                        
        except Exception as e:
            logger.error(f"Error updating positions from MT5: {e}")
    
    async def analyze_market_regime(self):
        """วิเคราะห์สถานการณ์ตลาด"""
        if not self.market_data:
            return
            
        try:
            # คำนวณ volatility จาก spread และ volume
            volatility_scores = []
            for symbol, data in self.market_data.items():
                volatility = data.spread / data.bid if data.bid > 0 else 0
                volatility_scores.append(volatility)
            
            if not volatility_scores:
                return
                
            avg_volatility = np.mean(volatility_scores)
            
            if avg_volatility > 0.0008:
                self.current_regime = MarketRegime.VOLATILE
            elif avg_volatility > 0.0005:
                self.current_regime = MarketRegime.TRENDING
            else:
                self.current_regime = MarketRegime.RANGING
            
            logger.debug(f"Market regime: {self.current_regime.value}")
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
    
    async def check_trading_opportunities(self):
        """ตรวจสอบโอกาสในการเทรด"""
        if not self.risk_manager.check_risk_limits(self._calculate_total_exposure(), self._calculate_unrealized_pnl()):
            return
        
        # ตรวจสอบสัญญาณการเทรดสำหรับแต่ละคู่เงิน
        for symbol in self.settings.enabled_pairs:
            if symbol in self.market_data and not self._has_initial_position(symbol):
                signal = await self._analyze_trading_signal(symbol)
                
                if signal:
                    await self._place_initial_order_mt5(symbol, signal)
    
    async def _analyze_trading_signal(self, symbol: str) -> Optional[str]:
        """วิเคราะห์สัญญาณการเทรด"""
        # Simple signal based on market regime and spread
        try:
            market = self.market_data[symbol]
            
            # ตรวจสอบ spread ไม่กว้างเกินไป
            max_spread = 0.0005 if 'JPY' not in symbol else 0.05
            if market.spread > max_spread:
                return None
            
            # Simple random signal for demo (ในระบบจริงจะใช้ technical indicators)
            import random
            signal_strength = random.random()
            
            if signal_strength > 0.98:  # 2% chance for BUY
                return "BUY"
            elif signal_strength < 0.02:  # 2% chance for SELL  
                return "SELL"
                
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing trading signal for {symbol}: {e}")
            return None
    
    def _has_initial_position(self, symbol: str) -> bool:
        """ตรวจสอบว่ามีโพซิชั่นเริ่มต้นในคู่เงินนี้หรือไม่"""
        for position in self.positions.values():
            if position.symbol == symbol and position.recovery_level == 0:
                return True
        return False
    
    async def _place_initial_order_mt5(self, symbol: str, direction: str):
        """วางออเดอร์เริ่มต้นใน MT5"""
        if not self.mt5_connector:
            return
            
        try:
            lot_size = self.risk_manager.calculate_position_size(symbol, 0)
            
            # วางออเดอร์ใน MT5
            result = self.mt5_connector.place_market_order(
                symbol=symbol,
                order_type=direction,
                volume=lot_size,
                comment="AI_Initial_Entry"
            )
            
            if result:
                # สร้าง Position object
                market = self.market_data[symbol]
                entry_price = market.ask if direction == "BUY" else market.bid
                
                position = Position(
                    symbol=symbol,
                    direction=direction,
                    size=lot_size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    recovery_level=0,
                    is_recovery=False,
                    mt5_ticket=result['ticket']
                )
                
                self.positions[position.position_id] = position
                self.mt5_positions[position.position_id] = result['ticket']
                
                logger.info(f"✅ Placed {direction} order for {symbol}: {lot_size} lots at {entry_price}")
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error placing MT5 order for {symbol}: {e}")
    
    async def execute_recovery_strategy(self):
        """ดำเนินการกลยุทธ์แก้ไม้"""
        losing_positions = [pos for pos in self.positions.values() 
                          if pos.pnl < -10 and pos.recovery_level < self.settings.max_recovery_levels]
        
        for position in losing_positions:
            # ตรวจสอบว่าควรเริ่มแก้ไม้หรือไม่ (ขาดทุนมากกว่า $10)
            if abs(position.pnl) > 10:
                if self.settings.use_correlation_recovery:
                    await self._execute_correlation_recovery_mt5(position)
                elif self.settings.use_grid_recovery:
                    await self._execute_grid_recovery_mt5(position)
    
    async def _execute_correlation_recovery_mt5(self, losing_position: Position):
        """ดำเนินการแก้ไม้แบบ Correlation ใน MT5"""
        try:
            correlated_pairs = self.correlation_matrix.get_correlated_pairs(losing_position.symbol)
            
            for pair in correlated_pairs:
                if pair in self.settings.enabled_pairs and pair in self.market_data and not self._has_position(pair):
                    
                    recovery_size = self.risk_manager.calculate_position_size(pair, losing_position.recovery_level + 1)
                    
                    # วางออเดอร์แก้ไม้ใน MT5
                    result = self.mt5_connector.place_market_order(
                        symbol=pair,
                        order_type=losing_position.direction,
                        volume=recovery_size,
                        comment=f"AI_Correlation_Recovery_L{losing_position.recovery_level + 1}"
                    )
                    
                    if result:
                        market = self.market_data[pair]
                        entry_price = market.ask if losing_position.direction == "BUY" else market.bid
                        
                        recovery_position = Position(
                            symbol=pair,
                            direction=losing_position.direction,
                            size=recovery_size,
                            entry_price=entry_price,
                            current_price=entry_price,
                            recovery_level=losing_position.recovery_level + 1,
                            is_recovery=True,
                            parent_position_id=losing_position.position_id,
                            mt5_ticket=result['ticket']
                        )
                        
                        self.positions[recovery_position.position_id] = recovery_position
                        losing_position.recovery_level += 1
                        
                        logger.info(f"🔄 Correlation recovery: {losing_position.direction} {pair} "
                                  f"at {entry_price}, size: {recovery_size}, level: {recovery_position.recovery_level}")
                        break
                        
        except Exception as e:
            logger.error(f"Error executing correlation recovery: {e}")
    
    async def _execute_grid_recovery_mt5(self, losing_position: Position):
        """ดำเนินการแก้ไม้แบบ Grid ใน MT5"""
        try:
            grid_distance_pips = 20  # pips
            pip_value = MT5CurrencyPairs.get_pip_value(losing_position.symbol)
            
            # คำนวณราคาที่จะวาง recovery order
            if losing_position.direction == "BUY":
                new_entry = losing_position.entry_price - (grid_distance_pips * pip_value * (losing_position.recovery_level + 1))
            else:
                new_entry = losing_position.entry_price + (grid_distance_pips * pip_value * (losing_position.recovery_level + 1))
            
            # เช็คว่าราคาปัจจุบันผ่านจุดที่จะวาง order หรือไม่
            current_price = self.market_data[losing_position.symbol].bid if losing_position.direction == "BUY" else self.market_data[losing_position.symbol].ask
            
            should_place_order = False
            if losing_position.direction == "BUY" and current_price <= new_entry:
                should_place_order = True
            elif losing_position.direction == "SELL" and current_price >= new_entry:
                should_place_order = True
                
            if should_place_order:
                recovery_size = self.risk_manager.calculate_position_size(losing_position.symbol, losing_position.recovery_level + 1)
                
                result = self.mt5_connector.place_market_order(
                    symbol=losing_position.symbol,
                    order_type=losing_position.direction,
                    volume=recovery_size,
                    comment=f"AI_Grid_Recovery_L{losing_position.recovery_level + 1}"
                )
                
                if result:
                    recovery_position = Position(
                        symbol=losing_position.symbol,
                        direction=losing_position.direction,
                        size=recovery_size,
                        entry_price=result['price'],
                        current_price=result['price'],
                        recovery_level=losing_position.recovery_level + 1,
                        is_recovery=True,
                        parent_position_id=losing_position.position_id,
                        mt5_ticket=result['ticket']
                    )
                    
                    self.positions[recovery_position.position_id] = recovery_position
                    losing_position.recovery_level += 1
                    
                    logger.info(f"📊 Grid recovery: {losing_position.direction} {losing_position.symbol} "
                              f"at {result['price']}, size: {recovery_size}, level: {recovery_position.recovery_level}")
                    
        except Exception as e:
            logger.error(f"Error executing grid recovery: {e}")
    
    def _has_position(self, symbol: str) -> bool:
        """ตรวจสอบว่ามีโพซิชั่นในคู่เงินนี้หรือไม่"""
        for position in self.positions.values():
            if position.symbol == symbol:
                return True
        return False
    
    async def check_exit_conditions(self):
        """ตรวจสอบเงื่อนไขปิดโพซิชั่น"""
        # กลุ่มโพซิชั่นตาม parent_position_id
        position_groups = {}
        for pos in self.positions.values():
            parent_id = pos.parent_position_id or pos.position_id
            if parent_id not in position_groups:
                position_groups[parent_id] = []
            position_groups[parent_id].append(pos)
        
        # ตรวจสอบแต่ละกลุ่ม
        for group_id, positions in position_groups.items():
            total_pnl = sum(pos.pnl for pos in positions)
            
            # ปิดกลุ่มถ้ากำไรรวมเป็นบวก หรือขาดทุนเกิน 50%
            should_close = False
            reason = ""
            
            if total_pnl > 5:  # กำไรมากกว่า $5
                should_close = True
                reason = "profit target"
            elif total_pnl < -100:  # ขาดทุนเกิน $100
                should_close = True
                reason = "stop loss"
                
            if should_close:
                await self._close_position_group_mt5(positions, reason)
    
    async def _close_position_group_mt5(self, positions: List[Position], reason: str):
        """ปิดกลุ่มโพซิชั่นใน MT5"""
        if not self.mt5_connector:
            return
            
        try:
            total_pnl = sum(pos.pnl for pos in positions)
            closed_count = 0
            
            for pos in positions:
                if pos.mt5_ticket:
                    if self.mt5_connector.close_position(pos.mt5_ticket):
                        closed_count += 1
                        # ลบออกจากระบบ
                        if pos.position_id in self.positions:
                            del self.positions[pos.position_id]
                        if pos.position_id in self.mt5_positions:
                            del self.mt5_positions[pos.position_id]
                            
                        logger.info(f"✅ Closed {pos.symbol} {pos.direction} - P&L: ${pos.pnl:.2f}")
            
            if closed_count > 0:
                self.risk_manager.update_pnl(total_pnl)
                logger.info(f"🎯 Closed {closed_count} positions ({reason}) - Total P&L: ${total_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position group: {e}")
    
    def _calculate_total_exposure(self) -> float:
        """คำนวณ exposure รวม"""
        return sum(pos.size for pos in self.positions.values())
    
    def _calculate_unrealized_pnl(self) -> float:
        """คำนวณ unrealized P&L รวม"""
        return sum(pos.pnl for pos in self.positions.values())
    
    async def emergency_stop(self):
        """หยุดฉุกเฉินและปิดโพซิชั่นทั้งหมด"""
        if not self.mt5_connector:
            return
            
        try:
            logger.warning("🚨 EMERGENCY STOP ACTIVATED")
            
            # ปิดโพซิชั่นทั้งหมดใน MT5
            closed_count = self.mt5_connector.close_all_positions()
            
            # ล้างโพซิชั่นในระบบ
            total_pnl = self._calculate_unrealized_pnl()
            self.positions.clear()
            self.mt5_positions.clear()
            
            self.risk_manager.update_pnl(total_pnl)
            
            logger.warning(f"🚨 Emergency stop completed - {closed_count} positions closed, P&L: ${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
    
    def get_status(self) -> Dict:
        """ได้สถานะปัจจุบันของระบบ"""
        total_pnl = self._calculate_unrealized_pnl()
        active_positions = len(self.positions)
        
        # ได้ข้อมูลบัญชีจาก MT5
        account_info = None
        if self.mt5_connector:
            account_info = self.mt5_connector.get_account_info()
            if account_info:
                self.settings.account_balance = account_info['balance']
        
        return {
            'is_running': self.is_running,
            'market_regime': self.current_regime.value,
            'active_positions': active_positions,
            'total_pnl': total_pnl,
            'daily_pnl': self.risk_manager.daily_pnl,
            'monthly_pnl': self.risk_manager.monthly_pnl,
            'risk_level': self._calculate_risk_level(),
            'enabled_pairs': self.settings.enabled_pairs,
            'current_strategy': 'Live MT5 Trading' if self.is_running else 'Standby',
            'mt5_connected': self.mt5_connector.is_connected if self.mt5_connector else False,
            'account_balance': self.settings.account_balance,
            'positions': [
                {
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_pips': pos.pnl_pips,
                    'recovery_level': pos.recovery_level,
                    'is_recovery': pos.is_recovery,
                    'mt5_ticket': pos.mt5_ticket
                }
                for pos in self.positions.values()
            ]
        }
    
    def _calculate_risk_level(self) -> str:
        """คำนวณระดับความเสี่ยง"""
        unrealized_pnl = self._calculate_unrealized_pnl()
        risk_pct = (abs(unrealized_pnl) / self.settings.account_balance) * 100
        
        if risk_pct > 20:
            return "HIGH"
        elif risk_pct > 10:
            return "MEDIUM"
        else:
            return "LOW"

# ตัวอย่างการใช้งาน
async def main():
    """ฟังก์ชันหลักสำหรับทดสอบระบบ"""
    
    # สร้างการตั้งค่า
    settings = TradingSettings(
        account_balance=5000.0,
        daily_target_pct=2.0,
        enabled_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        initial_lot_size=0.01,
        max_recovery_levels=3
    )
    
    # สร้าง AI Engine
    engine = RecoveryEngine(settings)
    
    print("🚀 Starting AI Recovery Trading Engine with MT5...")
    print(f"💰 Account Balance: ${settings.account_balance}")
    print(f"📊 Enabled Pairs: {settings.enabled_pairs}")
    print(f"🎯 Daily Target: {settings.daily_target_pct}%")
    print("-" * 50)
    
    try:
        # เริ่มต้นระบบ
        if await engine.start_engine():
            print("✅ AI Engine started successfully with MT5 connection")
            
            # รันสำหรับ demo 60 วินาที
            await asyncio.sleep(60)
            
            # แสดงผลสุดท้าย
            status = engine.get_status()
            print("\n📊 Final Status:")
            print(f"   Active Positions: {status['active_positions']}")
            print(f"   Total P&L: ${status['total_pnl']:.2f}")
            print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
            print(f"   Risk Level: {status['risk_level']}")
            print(f"   MT5 Connected: {status['mt5_connected']}")
        else:
            print("❌ Failed to start AI Engine")
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping engine...")
    finally:
        await engine.stop_engine()

if __name__ == "__main__":
    asyncio.run(main())