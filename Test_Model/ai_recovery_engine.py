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

# Import ML Data Collection
from ml_data_logger import MLDataCollector, MLDataIntegration, MLTrainingRecord

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
    
    # ML Data Collection Settings
    enable_ml_logging: bool = True
    client_id: str = "client_001"
    collect_market_data: bool = True
    collect_technical_indicators: bool = True

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
    
    # ML Data Fields
    signal_strength: float = 0.0
    confidence: float = 0.0
    strategy_context: Dict = field(default_factory=dict)

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: float = 0.0
    
    # Extended market data for ML
    volatility_1h: Optional[float] = None
    volatility_4h: Optional[float] = None
    session_type: Optional[str] = None  # london, ny, asia

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

class TechnicalIndicators:
    """คลาสสำหรับคำนวณ Technical Indicators สำหรับ ML"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """คำนวณ RSI"""
        try:
            if len(prices) < period + 1:
                return None
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except:
            return None
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """คำนวณ Simple Moving Average"""
        try:
            if len(prices) < period:
                return None
            return float(np.mean(prices[-period:]))
        except:
            return None
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """คำนวณ Exponential Moving Average"""
        try:
            if len(prices) < period:
                return None
            
            multiplier = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return float(ema)
        except:
            return None

class RecoveryEngine:
    """เครื่องยนต์หลักสำหรับ Recovery Trading พร้อม ML Data Collection"""
    
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
        
        # ML Data Collection Integration
        self.ml_collector = None
        self.ml_integration = None
        self.price_history = {}  # เก็บประวัติราคาสำหรับ technical indicators
        
        # Initialize ML Data Collection
        if settings.enable_ml_logging:
            self._initialize_ml_collection()
    
    def _initialize_ml_collection(self):
        """เริ่มต้น ML Data Collection"""
        try:
            self.ml_collector = MLDataCollector(
                db_path=f"data/ml_training_{self.settings.client_id}.db"
            )
            logger.info("✅ ML Data Collection initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Data Collection: {e}")
    
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
        
        # Initialize ML Data Collection
        if self.ml_collector:
            await self.ml_collector.initialize()
            self.ml_integration = MLDataIntegration(self, self.ml_collector)
            logger.info("✅ ML Data Collection ready")
            
        self.is_running = True
        logger.info("🚀 AI Recovery Engine Started with Live MT5 Connection and ML Logging")
        
        # เริ่ม main trading loop
        asyncio.create_task(self.main_trading_loop())
        return True
    
    async def stop_engine(self):
        """หยุด AI Engine"""
        self.is_running = False
        if self.mt5_connector:
            await self.mt5_connector.disconnect()
        logger.info("⏹️ AI Recovery Engine Stopped")
    
    async def main_trading_loop(self):
        """Loop หลักของการเทรด พร้อม ML Data Collection"""
        while self.is_running:
            try:
                # 1. อัพเดทข้อมูลตลาดจาก MT5
                await self.update_market_data_from_mt5()
                
                # 2. เก็บข้อมูลตลาดสำหรับ ML (ทุก tick)
                if self.ml_integration:
                    await self._collect_market_data_for_ml()
                
                # 3. อัพเดท P&L ของโพซิชั่นจาก MT5
                await self.update_positions_from_mt5()
                
                # 4. วิเคราะห์สถานการณ์ตลาด
                await self.analyze_market_regime()
                
                # 5. ตรวจสอบโอกาสในการเทรด
                await self.check_trading_opportunities()
                
                # 6. ดำเนินการแก้ไม้ถ้าจำเป็น
                await self.execute_recovery_strategy()
                
                # 7. ตรวจสอบเงื่อนไขปิดโพซิชั่น
                await self.check_exit_conditions()
                
                # รอ 2 วินาทีก่อนรอบต่อไป
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_market_data_for_ml(self):
        """เก็บข้อมูลตลาดสำหรับ ML Training"""
        try:
            for symbol, market_data in self.market_data.items():
                # เก็บประวัติราคา
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                self.price_history[symbol].append({
                    'timestamp': market_data.timestamp,
                    'price': market_data.bid,
                    'volume': market_data.volume
                })
                
                # เก็บแค่ 200 records ล่าสุด
                if len(self.price_history[symbol]) > 200:
                    self.price_history[symbol] = self.price_history[symbol][-200:]
                
                # คำนวณ Technical Indicators
                prices = [item['price'] for item in self.price_history[symbol]]
                technical_data = self._calculate_technical_indicators(prices)
                
                # สร้าง ML Training Record
                now = datetime.now()
                record = MLTrainingRecord(
                    timestamp=now,
                    record_id=f"{symbol}_{now.timestamp()}",
                    client_id=self.settings.client_id,
                    
                    # Market Data
                    symbol=symbol,
                    timeframe="M1",
                    open_price=market_data.bid,  # Simplified for tick data
                    high_price=market_data.ask,
                    low_price=market_data.bid,
                    close_price=market_data.bid,
                    volume=market_data.volume,
                    spread=market_data.spread,
                    
                    # Technical Indicators
                    **technical_data,
                    
                    # Market Context
                    market_regime=self.current_regime.value,
                    hour_of_day=now.hour,
                    day_of_week=now.weekday(),
                    is_london_session=self._is_london_session(now),
                    is_ny_session=self._is_ny_session(now),
                    is_asia_session=self._is_asia_session(now),
                    
                    # Portfolio Context
                    account_equity=self.settings.account_balance,
                    portfolio_risk_pct=self._calculate_portfolio_risk(),
                    
                    # Data Quality
                    data_completeness=0.9,  # TODO: Calculate actual completeness
                    data_source="mt5_live"
                )
                
                # บันทึกข้อมูล (แต่ไม่ทุก tick เพื่อไม่ให้ database ใหญ่เกินไป)
                if now.second % 10 == 0:  # เก็บทุก 10 วินาที
                    await self.ml_collector.record_ml_data(record)
                
        except Exception as e:
            logger.error(f"Error collecting market data for ML: {e}")
    
    def _calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """คำนวณ Technical Indicators"""
        try:
            if len(prices) < 20:
                return {}
            
            return {
                'sma_5': TechnicalIndicators.calculate_sma(prices, 5),
                'sma_20': TechnicalIndicators.calculate_sma(prices, 20),
                'sma_50': TechnicalIndicators.calculate_sma(prices, 50),
                'ema_12': TechnicalIndicators.calculate_ema(prices, 12),
                'ema_26': TechnicalIndicators.calculate_ema(prices, 26),
                'rsi_14': TechnicalIndicators.calculate_rsi(prices, 14)
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _is_london_session(self, dt: datetime) -> bool:
        """ตรวจสอบว่าเป็น London Session หรือไม่"""
        return 8 <= dt.hour < 16
    
    def _is_ny_session(self, dt: datetime) -> bool:
        """ตรวจสอบว่าเป็น NY Session หรือไม่"""
        return 13 <= dt.hour < 21
    
    def _is_asia_session(self, dt: datetime) -> bool:
        """ตรวจสอบว่าเป็น Asia Session หรือไม่"""
        return 0 <= dt.hour < 8
    
    def _calculate_portfolio_risk(self) -> float:
        """คำนวณความเสี่ยงของพอร์ต"""
        try:
            unrealized_pnl = self._calculate_unrealized_pnl()
            return (abs(unrealized_pnl) / self.settings.account_balance) * 100
        except:
            return 0.0
    
    async def update_market_data_from_mt5(self):
        """อัพเดทข้อมูลตลาดจาก MT5"""
        if not self.mt5_connector or not self.mt5_connector.is_connected:
            return
            
        try:
            # ดึงราคาปัจจุบันจาก MT5 พร้อม await
            prices = await self.mt5_connector.get_current_prices(self.settings.enabled_pairs)
            
            for symbol, price_data in prices.items():
                # เพิ่มการคำนวณ session type
                now = datetime.now()
                session_type = "asia"
                if self._is_london_session(now):
                    session_type = "london"
                elif self._is_ny_session(now):
                    session_type = "ny"
                
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    bid=price_data['bid'],
                    ask=price_data['ask'],
                    spread=price_data['spread'],
                    timestamp=price_data['time'],
                    volume=price_data.get('volume', 0),
                    session_type=session_type
                )
                
        except Exception as e:
            logger.error(f"Error updating market data from MT5: {e}")
    
    async def update_positions_from_mt5(self):
        """อัพเดทโพซิชั่นจาก MT5 และเก็บข้อมูลสำหรับ ML"""
        if not self.mt5_connector or not self.mt5_connector.is_connected:
            return
            
        try:
            # ดึงโพซิชั่นจาก MT5 พร้อม await
            mt5_positions = await self.mt5_connector.get_positions()
            
            # อัพเดทโพซิชั่นที่มีอยู่
            for pos_id, position in list(self.positions.items()):
                if position.mt5_ticket:
                    # หาโพซิชั่นที่ตรงกันใน MT5
                    mt5_pos = next((p for p in mt5_positions if p.ticket == position.mt5_ticket), None)
                    
                    if mt5_pos:
                        # อัพเดทข้อมูลจาก MT5
                        old_pnl = position.pnl
                        position.current_price = mt5_pos.price_current
                        position.pnl = mt5_pos.profit
                        
                        # คำนวณ pips
                        pip_value = MT5CurrencyPairs.get_pip_value(position.symbol)
                        if position.direction == "BUY":
                            position.pnl_pips = (position.current_price - position.entry_price) / pip_value
                        else:
                            position.pnl_pips = (position.entry_price - position.current_price) / pip_value
                        
                        # เก็บข้อมูลการเปลี่ยนแปลง PnL สำหรับ ML (ถ้ามีการเปลี่ยนแปลงมาก)
                        if self.ml_integration and abs(position.pnl - old_pnl) > 1:  # เปลี่ยนแปลงมากกว่า $1
                            await self._record_position_update_for_ml(position, old_pnl)
                    else:
                        # โพซิชั่นถูกปิดใน MT5 แล้ว - บันทึก trade result สำหรับ ML
                        if self.ml_integration:
                            await self._record_trade_close_for_ml(position)
                        
                        logger.info(f"Position {position.symbol} closed in MT5")
                        del self.positions[pos_id]
                        
        except Exception as e:
            logger.error(f"Error updating positions from MT5: {e}")
    
    async def _record_position_update_for_ml(self, position: Position, old_pnl: float):
        """บันทึกการอัพเดทโพซิชั่นสำหรับ ML"""
        try:
            now = datetime.now()
            hold_duration = int((now - position.timestamp).total_seconds() / 60)
            
            # สร้าง partial ML record สำหรับ position update
            record = MLTrainingRecord(
                timestamp=now,
                record_id=f"update_{position.position_id}_{now.timestamp()}",
                client_id=self.settings.client_id,
                symbol=position.symbol,
                timeframe="POSITION_UPDATE",
                
                # ใช้ current price เป็น OHLC
                open_price=position.current_price,
                high_price=position.current_price,
                low_price=position.current_price,
                close_price=position.current_price,
                volume=0,
                spread=0,
                
                # Position context
                entry_price=position.entry_price,
                actual_pnl=position.pnl,
                actual_pips=position.pnl_pips,
                hold_duration_minutes=hold_duration,
                
                # AI context
                recovery_level=position.recovery_level,
                is_recovery_trade=position.is_recovery,
                position_size=position.size,
                
                # Portfolio context
                account_equity=self.settings.account_balance,
                portfolio_risk_pct=self._calculate_portfolio_risk(),
                
                data_source="position_update"
            )
            
            await self.ml_collector.record_ml_data(record)
            
        except Exception as e:
            logger.error(f"Error recording position update for ML: {e}")
    
    async def _record_trade_close_for_ml(self, position: Position):
        """บันทึกการปิดเทรดสำหรับ ML"""
        try:
            now = datetime.now()
            hold_duration = int((now - position.timestamp).total_seconds() / 60)
            
            # กำหนด trade result
            trade_result = "BREAKEVEN"
            win_quality = None
            loss_severity = None
            
            if position.pnl > 5:
                trade_result = "WIN"
                if position.pnl > 50:
                    win_quality = "BIG"
                elif position.pnl > 20:
                    win_quality = "MEDIUM"
                else:
                    win_quality = "SMALL"
            elif position.pnl < -5:
                trade_result = "LOSS"
                if position.pnl < -50:
                    loss_severity = "BIG"
                elif position.pnl < -20:
                    loss_severity = "MEDIUM"
                else:
                    loss_severity = "SMALL"
            
            # คำนวณ return percentage
            return_pct = (position.pnl / (position.size * 100000)) * 100  # Simplified calculation
            
            record = MLTrainingRecord(
                timestamp=now,
                record_id=f"close_{position.position_id}",
                client_id=self.settings.client_id,
                symbol=position.symbol,
                timeframe="TRADE_CLOSE",
                
                # Trade data
                open_price=position.entry_price,
                high_price=max(position.entry_price, position.current_price),
                low_price=min(position.entry_price, position.current_price),
                close_price=position.current_price,
                volume=0,
                spread=0,
                
                # AI Decision (from position context)
                ai_predicted_direction=position.direction,
                ai_signal_strength=position.signal_strength,
                ai_confidence=position.confidence,
                strategy_used=position.strategy_context.get('strategy', 'unknown'),
                
                # Actual Results
                actual_direction=position.direction,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                actual_pips=position.pnl_pips,
                actual_pnl=position.pnl,
                actual_return_pct=return_pct,
                hold_duration_minutes=hold_duration,
                trade_result=trade_result,
                win_quality=win_quality,
                loss_severity=loss_severity,
                
                # Context
                recovery_level=position.recovery_level,
                is_recovery_trade=position.is_recovery,
                position_size=position.size,
                account_equity=self.settings.account_balance,
                
                data_source="trade_close"
            )
            
            await self.ml_collector.record_ml_data(record)
            logger.info(f"📊 ML Trade Close recorded: {position.symbol} {trade_result} ${position.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade close for ML: {e}")
    
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
        """ตรวจสอบโอกาสในการเทรด พร้อมเก็บข้อมูล ML"""
        if not self.risk_manager.check_risk_limits(self._calculate_total_exposure(), self._calculate_unrealized_pnl()):
            return
        
        # ตรวจสอบสัญญาณการเทรดสำหรับแต่ละคู่เงิน
        for symbol in self.settings.enabled_pairs:
            if symbol in self.market_data and not self._has_initial_position(symbol):
                signal_data = await self._analyze_trading_signal(symbol)
                
                if signal_data and signal_data.get('signal'):
                    await self._place_initial_order_mt5(symbol, signal_data)
    
    async def _analyze_trading_signal(self, symbol: str) -> Optional[Dict]:
        """วิเคราะห์สัญญาณการเทรด พร้อมเก็บข้อมูล"""
        try:
            market = self.market_data[symbol]
            
            # ตรวจสอบ spread ไม่กว้างเกินไป
            max_spread = 0.0005 if 'JPY' not in symbol else 0.05
            if market.spread > max_spread:
                return None
            
            # Simple random signal for demo (ในระบบจริงจะใช้ technical indicators)
            import random
            signal_strength = random.random()
            confidence = random.uniform(0.4, 0.9)
            
            signal_data = {
                'signal': None,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'analysis_time': datetime.now(),
                'spread': market.spread,
                'session': market.session_type
            }
            
            if signal_strength > 0.98:  # 2% chance for BUY
                signal_data['signal'] = "BUY"
                signal_data['predicted_pips'] = random.uniform(10, 50)
            elif signal_strength < 0.02:  # 2% chance for SELL  
                signal_data['signal'] = "SELL"
                signal_data['predicted_pips'] = random.uniform(10, 50)
            
            # เก็บข้อมูลการวิเคราะห์สำหรับ ML (แม้ไม่มี signal)
            if self.ml_integration:
                await self._record_signal_analysis_for_ml(symbol, signal_data)
                
            return signal_data
            
        except Exception as e:
            logger.error(f"Error analyzing trading signal for {symbol}: {e}")
            return None
    
    async def _record_signal_analysis_for_ml(self, symbol: str, signal_data: Dict):
        """บันทึกการวิเคราะห์สัญญาณสำหรับ ML"""
        try:
            now = datetime.now()
            
            record = MLTrainingRecord(
                timestamp=now,
                record_id=f"signal_{symbol}_{now.timestamp()}",
                client_id=self.settings.client_id,
                symbol=symbol,
                timeframe="SIGNAL_ANALYSIS",
                
                # Market data
                open_price=self.market_data[symbol].bid,
                high_price=self.market_data[symbol].ask,
                low_price=self.market_data[symbol].bid,
                close_price=self.market_data[symbol].bid,
                volume=self.market_data[symbol].volume,
                spread=signal_data['spread'],
                
                # AI Analysis
                ai_predicted_direction=signal_data.get('signal'),
                ai_signal_strength=signal_data['signal_strength'],
                ai_confidence=signal_data['confidence'],
                ai_predicted_pips=signal_data.get('predicted_pips'),
                
                # Context
                market_regime=self.current_regime.value,
                hour_of_day=now.hour,
                day_of_week=now.weekday(),
                account_equity=self.settings.account_balance,
                
                data_source="signal_analysis"
            )
            
            # บันทึกเฉพาะตอนที่มี signal เท่านั้น (เพื่อไม่ให้ข้อมูลเยอะเกินไป)
            if signal_data.get('signal'):
                await self.ml_collector.record_ml_data(record)
                
        except Exception as e:
            logger.error(f"Error recording signal analysis for ML: {e}")
    
    def _has_initial_position(self, symbol: str) -> bool:
        """ตรวจสอบว่ามีโพซิชั่นเริ่มต้นในคู่เงินนี้หรือไม่"""
        for position in self.positions.values():
            if position.symbol == symbol and position.recovery_level == 0:
                return True
        return False
    
    async def _place_initial_order_mt5(self, symbol: str, signal_data: Dict):
        """วางออเดอร์เริ่มต้นใน MT5 พร้อมเก็บข้อมูล ML"""
        if not self.mt5_connector:
            return
            
        try:
            direction = signal_data['signal']
            lot_size = self.risk_manager.calculate_position_size(symbol, 0)
            
            # วางออเดอร์ใน MT5
            result = await self.mt5_connector.place_market_order(
                symbol=symbol,
                order_type=direction,
                volume=lot_size,
                comment="AI_Initial_Entry"
            )
            
            if result:
                # สร้าง Position object พร้อม ML context
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
                    mt5_ticket=result['ticket'],
                    signal_strength=signal_data['signal_strength'],
                    confidence=signal_data['confidence'],
                    strategy_context={
                        'strategy': 'initial_entry',
                        'predicted_pips': signal_data.get('predicted_pips'),
                        'analysis_time': signal_data['analysis_time'].isoformat()
                    }
                )
                
                self.positions[position.position_id] = position
                self.mt5_positions[position.position_id] = result['ticket']
                
                # บันทึกการเปิดเทรดสำหรับ ML
                if self.ml_integration:
                    await self._record_trade_open_for_ml(position, signal_data)
                
                logger.info(f"✅ Placed {direction} order for {symbol}: {lot_size} lots at {entry_price}")
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error placing MT5 order for {symbol}: {e}")
    
    async def _record_trade_open_for_ml(self, position: Position, signal_data: Dict):
        """บันทึกการเปิดเทรดสำหรับ ML"""
        try:
            record = MLTrainingRecord(
                timestamp=position.timestamp,
                record_id=f"open_{position.position_id}",
                client_id=self.settings.client_id,
                symbol=position.symbol,
                timeframe="TRADE_OPEN",
                
                # Market data at entry
                open_price=position.entry_price,
                high_price=position.entry_price,
                low_price=position.entry_price,
                close_price=position.entry_price,
                volume=0,
                spread=signal_data.get('spread', 0),
                
                # AI Decision
                ai_predicted_direction=position.direction,
                ai_signal_strength=position.signal_strength,
                ai_confidence=position.confidence,
                ai_predicted_pips=signal_data.get('predicted_pips'),
                strategy_used="initial_entry",
                
                # Trade context
                entry_price=position.entry_price,
                position_size=position.size,
                recovery_level=position.recovery_level,
                is_recovery_trade=position.is_recovery,
                
                # Market context
                market_regime=self.current_regime.value,
                hour_of_day=position.timestamp.hour,
                day_of_week=position.timestamp.weekday(),
                
                # Portfolio context
                account_equity=self.settings.account_balance,
                portfolio_risk_pct=self._calculate_portfolio_risk(),
                
                data_source="trade_open"
            )
            
            await self.ml_collector.record_ml_data(record)
            logger.info(f"📊 ML Trade Open recorded: {position.symbol} {position.direction}")
            
        except Exception as e:
            logger.error(f"Error recording trade open for ML: {e}")
    
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
                    result = await self.mt5_connector.place_market_order(
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
                            mt5_ticket=result['ticket'],
                            signal_strength=0.7,  # Recovery trades have different signal strength
                            confidence=0.6,
                            strategy_context={
                                'strategy': 'correlation_recovery',
                                'parent_symbol': losing_position.symbol,
                                'parent_loss': losing_position.pnl
                            }
                        )
                        
                        self.positions[recovery_position.position_id] = recovery_position
                        losing_position.recovery_level += 1
                        
                        # บันทึกการเปิด recovery trade สำหรับ ML
                        if self.ml_integration:
                            await self._record_recovery_trade_for_ml(recovery_position, losing_position)
                        
                        logger.info(f"🔄 Correlation recovery: {losing_position.direction} {pair} "
                                  f"at {entry_price}, size: {recovery_size}, level: {recovery_position.recovery_level}")
                        break
                        
        except Exception as e:
            logger.error(f"Error executing correlation recovery: {e}")
    
    async def _record_recovery_trade_for_ml(self, recovery_position: Position, losing_position: Position):
        """บันทึก Recovery Trade สำหรับ ML"""
        try:
            record = MLTrainingRecord(
                timestamp=recovery_position.timestamp,
                record_id=f"recovery_{recovery_position.position_id}",
                client_id=self.settings.client_id,
                symbol=recovery_position.symbol,
                timeframe="RECOVERY_TRADE",
                
                # Market data
                open_price=recovery_position.entry_price,
                high_price=recovery_position.entry_price,
                low_price=recovery_position.entry_price,
                close_price=recovery_position.entry_price,
                volume=0,
                spread=0,
                
                # Recovery context
                ai_predicted_direction=recovery_position.direction,
                ai_signal_strength=recovery_position.signal_strength,
                ai_confidence=recovery_position.confidence,
                strategy_used="correlation_recovery",
                recovery_level=recovery_position.recovery_level,
                is_recovery_trade=True,
                position_size=recovery_position.size,
                
                # Parent trade context
                entry_price=recovery_position.entry_price,
                
                # Market context
                market_regime=self.current_regime.value,
                account_equity=self.settings.account_balance,
                portfolio_risk_pct=self._calculate_portfolio_risk(),
                
                data_source="recovery_trade"
            )
            
            await self.ml_collector.record_ml_data(record)
            logger.info(f"📊 ML Recovery Trade recorded: {recovery_position.symbol} L{recovery_position.recovery_level}")
            
        except Exception as e:
            logger.error(f"Error recording recovery trade for ML: {e}")
    
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
                
                result = await self.mt5_connector.place_market_order(
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
                        mt5_ticket=result['ticket'],
                        signal_strength=0.6,
                        confidence=0.5,
                        strategy_context={
                            'strategy': 'grid_recovery',
                            'grid_distance_pips': grid_distance_pips,
                            'parent_loss': losing_position.pnl
                        }
                    )
                    
                    self.positions[recovery_position.position_id] = recovery_position
                    losing_position.recovery_level += 1
                    
                    # บันทึกสำหรับ ML
                    if self.ml_integration:
                        await self._record_recovery_trade_for_ml(recovery_position, losing_position)
                    
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
                    if await self.mt5_connector.close_position(pos.mt5_ticket):
                        closed_count += 1
                        
                        # บันทึกการปิดเทรดสำหรับ ML ก่อนลบ
                        if self.ml_integration:
                            await self._record_trade_close_for_ml(pos)
                        
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
            
            # บันทึกการปิดโพซิชั่นทั้งหมดสำหรับ ML ก่อน
            if self.ml_integration:
                for pos in self.positions.values():
                    await self._record_trade_close_for_ml(pos)
            
            # ปิดโพซิชั่นทั้งหมดใน MT5
            closed_count = await self.mt5_connector.close_all_positions()
            
            # ล้างโพซิชั่นในระบบ
            total_pnl = self._calculate_unrealized_pnl()
            self.positions.clear()
            self.mt5_positions.clear()
            
            self.risk_manager.update_pnl(total_pnl)
            
            logger.warning(f"🚨 Emergency stop completed - {closed_count} positions closed, P&L: ${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
    
    async def export_ml_data(self, days: int = 30) -> Dict:
        """ส่งออกข้อมูล ML ของลูกค้า"""
        try:
            if not self.ml_collector:
                return {"error": "ML collection not enabled"}
            
            start_date = datetime.now() - timedelta(days=days)
            
            result = await self.ml_collector.export_training_dataset(
                start_date=start_date,
                symbols=self.settings.enabled_pairs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting ML data: {e}")
            return {"error": str(e)}
    
    async def get_ml_stats(self) -> Dict:
        """ดึงสถิติข้อมูล ML ของลูกค้า"""
        try:
            if not self.ml_collector:
                return {"error": "ML collection not enabled"}
            
            stats = await self.ml_collector.get_ml_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ML stats: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict:
        """ได้สถานะปัจจุบันของระบบ"""
        total_pnl = self._calculate_unrealized_pnl()
        active_positions = len(self.positions)
        
        status = {
            'is_running': self.is_running,
            'market_regime': self.current_regime.value,
            'active_positions': active_positions,
            'total_pnl': total_pnl,
            'daily_pnl': self.risk_manager.daily_pnl,
            'monthly_pnl': self.risk_manager.monthly_pnl,
            'risk_level': self._calculate_risk_level(),
            'enabled_pairs': self.settings.enabled_pairs,
            'current_strategy': 'Live MT5 Trading with ML Logging' if self.is_running else 'Standby',
            'mt5_connected': self.mt5_connector.is_connected if self.mt5_connector else False,
            'account_balance': self.settings.account_balance,
            'ml_logging_enabled': self.settings.enable_ml_logging,
            'client_id': self.settings.client_id,
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
                    'mt5_ticket': pos.mt5_ticket,
                    'signal_strength': pos.signal_strength,
                    'confidence': pos.confidence
                }
                for pos in self.positions.values()
            ]
        }
        
        return status
    
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
    """ฟังก์ชันหลักสำหรับทดสอบระบบ พร้อม ML Logging"""
    
    # สร้างการตั้งค่า
    settings = TradingSettings(
        account_balance=5000.0,
        daily_target_pct=2.0,
        enabled_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        initial_lot_size=0.01,
        max_recovery_levels=3,
        enable_ml_logging=True,
        client_id="demo_client_001"
    )
    
    # สร้าง AI Engine
    engine = RecoveryEngine(settings)
    
    print("🚀 Starting AI Recovery Trading Engine with ML Data Collection...")
    print(f"💰 Account Balance: ${settings.account_balance}")
    print(f"📊 Enabled Pairs: {settings.enabled_pairs}")
    print(f"🎯 Daily Target: {settings.daily_target_pct}%")
    print(f"🧠 ML Logging: {'✅ Enabled' if settings.enable_ml_logging else '❌ Disabled'}")
    print(f"👤 Client ID: {settings.client_id}")
    print("-" * 60)
    
    try:
        # เริ่มต้นระบบ
        if await engine.start_engine():
            print("✅ AI Engine started successfully with MT5 connection and ML logging")
            
            # รันสำหรับ demo 120 วินาที
            await asyncio.sleep(120)
            
            # แสดงผลสุดท้าย
            status = engine.get_status()
            print("\n📊 Final Status:")
            print(f"   Active Positions: {status['active_positions']}")
            print(f"   Total P&L: ${status['total_pnl']:.2f}")
            print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
            print(f"   Risk Level: {status['risk_level']}")
            print(f"   MT5 Connected: {status['mt5_connected']}")
            print(f"   ML Logging: {status['ml_logging_enabled']}")
            
            # ดึงสถิติ ML
            if engine.ml_collector:
                ml_stats = await engine.get_ml_stats()
                print(f"\n🧠 ML Data Statistics:")
                print(f"   Total Records: {ml_stats.get('general', {}).get('total_records', 0)}")
                print(f"   Data Completeness: {ml_stats.get('general', {}).get('avg_completeness', 0):.2%}")
                
                # Export ML data
                export_result = await engine.export_ml_data(days=1)
                if export_result.get('status') == 'success':
                    print(f"   📁 ML Data Exported: {export_result['total_records']} records")
                    print(f"   📄 Export File: {export_result['export_dir']}")
        else:
            print("❌ Failed to start AI Engine")
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping engine...")
    finally:
        await engine.stop_engine()

if __name__ == "__main__":
    asyncio.run(main())