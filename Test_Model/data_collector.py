import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import sqlite3
import aiosqlite
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class TradeResult(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"

class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CORRUPTED = "corrupted"

@dataclass
class TradeRecord:
    """บันทึกการเทรดสำหรับ Active Learning"""
    # Trade Identification
    trade_id: str
    client_id: str
    mt5_ticket: Optional[int] = None
    
    # Market Data
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    
    # Trade Details
    direction: str  # BUY/SELL
    volume: float
    pnl: float = 0.0
    pnl_pips: float = 0.0
    
    # AI Context
    ai_signal_strength: float = 0.0
    market_regime: str = "unknown"
    recovery_level: int = 0
    is_recovery_trade: bool = False
    strategy_used: str = "unknown"
    
    # Market Context
    spread: float = 0.0
    volatility: float = 0.0
    volume_profile: Dict[str, float] = None
    correlation_data: Dict[str, float] = None
    
    # Results
    result: TradeResult = TradeResult.PENDING
    hold_duration_minutes: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # Metadata
    data_quality: DataQuality = DataQuality.HIGH
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        if not self.trade_id:
            self.trade_id = self.generate_trade_id()
    
    def generate_trade_id(self) -> str:
        """สร้าง unique trade ID"""
        data = f"{self.client_id}_{self.symbol}_{self.entry_time}_{self.entry_price}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def calculate_result(self):
        """คำนวณผลลัพธ์การเทรด"""
        if self.exit_price is None:
            self.result = TradeResult.PENDING
            return
        
        if abs(self.pnl) < 0.5:  # Breakeven threshold
            self.result = TradeResult.BREAKEVEN
        elif self.pnl > 0:
            self.result = TradeResult.WIN
        else:
            self.result = TradeResult.LOSS
    
    def calculate_hold_duration(self):
        """คำนวณระยะเวลาการถือโพซิชั่น"""
        if self.exit_time:
            delta = self.exit_time - self.entry_time
            self.hold_duration_minutes = int(delta.total_seconds() / 60)

@dataclass
class MarketSnapshot:
    """สแนปช็อตของสถานการณ์ตลาด"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    spread: float
    volume: float
    
    # Technical Indicators (สำหรับอนาคต)
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    # Market Context
    volatility_1h: Optional[float] = None
    volatility_4h: Optional[float] = None
    trend_direction: Optional[str] = None

class DataCollector:
    """ระบบเก็บข้อมูลสำหรับ Active Learning"""
    
    def __init__(self, db_path: str = "data/trading_data.db", 
                 data_dir: str = "data/collections"):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        
        # สร้างโฟลเดอร์ถ้าไม่มี
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_initialized = False
        self._buffer = []
        self._buffer_size = 100
        
    async def initialize(self):
        """เริ่มต้นระบบฐานข้อมูล"""
        try:
            await self._create_tables()
            self.is_initialized = True
            logger.info("✅ Data Collector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Collector: {e}")
            raise
    
    async def _create_tables(self):
        """สร้างตารางฐานข้อมูล"""
        async with aiosqlite.connect(self.db_path) as db:
            # ตาราง trades
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    mt5_ticket INTEGER,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_pips REAL DEFAULT 0,
                    ai_signal_strength REAL DEFAULT 0,
                    market_regime TEXT DEFAULT 'unknown',
                    recovery_level INTEGER DEFAULT 0,
                    is_recovery_trade BOOLEAN DEFAULT FALSE,
                    strategy_used TEXT DEFAULT 'unknown',
                    spread REAL DEFAULT 0,
                    volatility REAL DEFAULT 0,
                    result TEXT DEFAULT 'pending',
                    hold_duration_minutes INTEGER DEFAULT 0,
                    max_profit REAL DEFAULT 0,
                    max_loss REAL DEFAULT 0,
                    data_quality TEXT DEFAULT 'high',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ตาราง market_snapshots
            await db.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    spread REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    rsi REAL,
                    macd REAL,
                    bollinger_position REAL,
                    volatility_1h REAL,
                    volatility_4h REAL,
                    trend_direction TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ตาราง client_performance
            await db.execute("""
                CREATE TABLE IF NOT EXISTS client_performance (
                    client_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    total_volume REAL DEFAULT 0,
                    avg_hold_time_minutes INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (client_id, date)
                )
            """)
            
            # สร้าง indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_client_id ON trades(client_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_snapshots(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_snapshots(symbol)")
            
            await db.commit()
    
    async def record_trade_entry(self, trade_record: TradeRecord) -> bool:
        """บันทึกการเข้าเทรด"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, client_id, mt5_ticket, symbol, entry_time, entry_price,
                        direction, volume, ai_signal_strength, market_regime, recovery_level,
                        is_recovery_trade, strategy_used, spread, volatility, data_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record.trade_id, trade_record.client_id, trade_record.mt5_ticket,
                    trade_record.symbol, trade_record.entry_time, trade_record.entry_price,
                    trade_record.direction, trade_record.volume, trade_record.ai_signal_strength,
                    trade_record.market_regime, trade_record.recovery_level,
                    trade_record.is_recovery_trade, trade_record.strategy_used,
                    trade_record.spread, trade_record.volatility, trade_record.data_quality.value
                ))
                await db.commit()
            
            logger.info(f"✅ Trade entry recorded: {trade_record.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade entry: {e}")
            return False
    
    async def record_trade_exit(self, trade_id: str, exit_price: float, 
                               exit_time: datetime, final_pnl: float, final_pnl_pips: float) -> bool:
        """บันทึกการออกจากเทรด"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # อัพเดทข้อมูลการออกจากเทรด
                await db.execute("""
                    UPDATE trades SET 
                        exit_time = ?, exit_price = ?, pnl = ?, pnl_pips = ?, updated_at = ?
                    WHERE trade_id = ?
                """, (exit_time, exit_price, final_pnl, final_pnl_pips, datetime.now(), trade_id))
                
                # ดึงข้อมูลเทรดเพื่อคำนวณผลลัพธ์
                cursor = await db.execute("""
                    SELECT entry_time, exit_time, pnl FROM trades WHERE trade_id = ?
                """, (trade_id,))
                trade_data = await cursor.fetchone()
                
                if trade_data:
                    entry_time, exit_time, pnl = trade_data
                    
                    # คำนวณระยะเวลาการถือ
                    if exit_time and entry_time:
                        entry_dt = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
                        exit_dt = datetime.fromisoformat(exit_time) if isinstance(exit_time, str) else exit_time
                        hold_duration = int((exit_dt - entry_dt).total_seconds() / 60)
                    else:
                        hold_duration = 0
                    
                    # คำนวณผลลัพธ์
                    if abs(pnl) < 0.5:
                        result = TradeResult.BREAKEVEN.value
                    elif pnl > 0:
                        result = TradeResult.WIN.value
                    else:
                        result = TradeResult.LOSS.value
                    
                    # อัพเดทผลลัพธ์และระยะเวลา
                    await db.execute("""
                        UPDATE trades SET 
                            result = ?, hold_duration_minutes = ?, updated_at = ?
                        WHERE trade_id = ?
                    """, (result, hold_duration, datetime.now(), trade_id))
                
                await db.commit()
            
            logger.info(f"✅ Trade exit recorded: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade exit: {e}")
            return False
    
    async def record_market_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """บันทึกสแนปช็อตตลาด"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO market_snapshots (
                        timestamp, symbol, bid, ask, spread, volume,
                        rsi, macd, bollinger_position, volatility_1h, volatility_4h, trend_direction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.timestamp, snapshot.symbol, snapshot.bid, snapshot.ask,
                    snapshot.spread, snapshot.volume, snapshot.rsi, snapshot.macd,
                    snapshot.bollinger_position, snapshot.volatility_1h,
                    snapshot.volatility_4h, snapshot.trend_direction
                ))
                await db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record market snapshot: {e}")
            return False
    
    async def get_client_trades(self, client_id: str, days: int = 30) -> List[Dict]:
        """ดึงข้อมูลเทรดของลูกค้า"""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT * FROM trades 
                    WHERE client_id = ? AND entry_time >= ?
                    ORDER BY entry_time DESC
                """, (client_id, since_date))
                
                trades = []
                async for row in cursor:
                    trade_dict = dict(row)
                    trades.append(trade_dict)
                
                return trades
                
        except Exception as e:
            logger.error(f"❌ Failed to get client trades: {e}")
            return []
    
    async def calculate_client_performance(self, client_id: str, date: datetime = None) -> Dict:
        """คำนวณผลการดำเนินงานของลูกค้า"""
        try:
            if date is None:
                date = datetime.now().date()
            
            start_date = datetime.combine(date, datetime.min.time())
            end_date = start_date + timedelta(days=1)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        SUM(volume) as total_volume,
                        AVG(hold_duration_minutes) as avg_hold_time,
                        MIN(pnl) as max_loss
                    FROM trades 
                    WHERE client_id = ? AND entry_time >= ? AND entry_time < ?
                        AND result != 'pending'
                """, (client_id, start_date, end_date))
                
                row = await cursor.fetchone()
                
                if row:
                    data = dict(row)
                    
                    # คำนวณเพิ่มเติม
                    total_trades = data['total_trades'] or 0
                    winning_trades = data['winning_trades'] or 0
                    losing_trades = data['losing_trades'] or 0
                    total_pnl = data['total_pnl'] or 0
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    # คำนวณ Profit Factor
                    profit_cursor = await db.execute("""
                        SELECT SUM(pnl) as gross_profit FROM trades 
                        WHERE client_id = ? AND entry_time >= ? AND entry_time < ?
                            AND result = 'win'
                    """, (client_id, start_date, end_date))
                    gross_profit = (await profit_cursor.fetchone())[0] or 0
                    
                    loss_cursor = await db.execute("""
                        SELECT ABS(SUM(pnl)) as gross_loss FROM trades 
                        WHERE client_id = ? AND entry_time >= ? AND entry_time < ?
                            AND result = 'loss'
                    """, (client_id, start_date, end_date))
                    gross_loss = (await loss_cursor.fetchone())[0] or 0
                    
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
                    
                    performance = {
                        'client_id': client_id,
                        'date': date.isoformat(),
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'total_pnl': round(total_pnl, 2),
                        'total_volume': round(data['total_volume'] or 0, 2),
                        'avg_hold_time_minutes': int(data['avg_hold_time'] or 0),
                        'win_rate': round(win_rate, 2),
                        'profit_factor': round(profit_factor, 2),
                        'max_drawdown': abs(data['max_loss'] or 0)
                    }
                    
                    # บันทึกลงฐานข้อมูล
                    await self._save_client_performance(performance)
                    
                    return performance
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate client performance: {e}")
            return {}
    
    async def _save_client_performance(self, performance: Dict):
        """บันทึกผลการดำเนินงานของลูกค้า"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO client_performance (
                        client_id, date, total_trades, winning_trades, losing_trades,
                        total_pnl, total_volume, avg_hold_time_minutes, win_rate,
                        profit_factor, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance['client_id'], performance['date'],
                    performance['total_trades'], performance['winning_trades'],
                    performance['losing_trades'], performance['total_pnl'],
                    performance['total_volume'], performance['avg_hold_time_minutes'],
                    performance['win_rate'], performance['profit_factor'],
                    performance['max_drawdown']
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save client performance: {e}")
    
    async def export_training_data(self, min_trades: int = 100, quality_filter: str = "high") -> Dict:
        """ส่งออกข้อมูลสำหรับฝึก AI Model"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # ดึงข้อมูลเทรดที่มีคุณภาพสูง
                cursor = await db.execute("""
                    SELECT * FROM trades 
                    WHERE result != 'pending' 
                        AND data_quality = ? 
                        AND ai_signal_strength > 0
                    ORDER BY entry_time DESC
                """, (quality_filter,))
                
                trades = []
                async for row in cursor:
                    trades.append(dict(row))
                
                if len(trades) < min_trades:
                    logger.warning(f"⚠️ Insufficient training data: {len(trades)}/{min_trades}")
                    return {"status": "insufficient_data", "count": len(trades)}
                
                # จัดกลุ่มข้อมูลตามประเภท
                training_data = {
                    "metadata": {
                        "total_records": len(trades),
                        "export_time": datetime.now().isoformat(),
                        "quality_filter": quality_filter,
                        "min_trades": min_trades
                    },
                    "features": [],
                    "targets": [],
                    "symbols": list(set(trade['symbol'] for trade in trades)),
                    "strategies": list(set(trade['strategy_used'] for trade in trades)),
                    "clients": list(set(trade['client_id'] for trade in trades))
                }
                
                # แปลงข้อมูลเป็น features และ targets
                for trade in trades:
                    features = {
                        'symbol': trade['symbol'],
                        'direction': trade['direction'],
                        'ai_signal_strength': trade['ai_signal_strength'],
                        'market_regime': trade['market_regime'],
                        'spread': trade['spread'],
                        'volatility': trade['volatility'],
                        'recovery_level': trade['recovery_level'],
                        'is_recovery_trade': trade['is_recovery_trade'],
                        'strategy_used': trade['strategy_used'],
                        'volume': trade['volume']
                    }
                    
                    target = {
                        'result': trade['result'],
                        'pnl': trade['pnl'],
                        'pnl_pips': trade['pnl_pips'],
                        'hold_duration_minutes': trade['hold_duration_minutes']
                    }
                    
                    training_data['features'].append(features)
                    training_data['targets'].append(target)
                
                # บันทึกลงไฟล์
                export_file = self.data_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                async with aiofiles.open(export_file, 'w') as f:
                    await f.write(json.dumps(training_data, indent=2, default=str))
                
                logger.info(f"✅ Training data exported: {export_file} ({len(trades)} records)")
                
                return {
                    "status": "success",
                    "export_file": str(export_file),
                    "record_count": len(trades),
                    "metadata": training_data['metadata']
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to export training data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_data_stats(self) -> Dict:
        """ดึงสถิติข้อมูลที่เก็บรวบรวม"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # สถิติเทรดทั้งหมด
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(DISTINCT client_id) as unique_clients,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as total_wins,
                        SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as total_losses,
                        SUM(CASE WHEN result = 'pending' THEN 1 ELSE 0 END) as pending_trades,
                        AVG(pnl) as avg_pnl,
                        MIN(entry_time) as first_trade,
                        MAX(entry_time) as last_trade
                    FROM trades
                """)
                stats = dict(await cursor.fetchone())
                
                # สถิติตาม data quality
                quality_cursor = await db.execute("""
                    SELECT data_quality, COUNT(*) as count 
                    FROM trades 
                    GROUP BY data_quality
                """)
                quality_stats = {}
                async for row in quality_cursor:
                    quality_stats[row[0]] = row[1]
                
                # สถิติตามกลยุทธ์
                strategy_cursor = await db.execute("""
                    SELECT strategy_used, COUNT(*) as count, AVG(pnl) as avg_pnl
                    FROM trades 
                    WHERE result != 'pending'
                    GROUP BY strategy_used
                """)
                strategy_stats = {}
                async for row in strategy_cursor:
                    strategy_stats[row[0]] = {
                        'count': row[1],
                        'avg_pnl': round(row[2] or 0, 2)
                    }
                
                return {
                    "overview": stats,
                    "quality_distribution": quality_stats,
                    "strategy_performance": strategy_stats,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get data stats: {e}")
            return {}

# Integration กับระบบหลัก
class ActiveLearningIntegration:
    """Integration ระหว่าง AI Engine และ Data Collection"""
    
    def __init__(self, ai_engine, data_collector: DataCollector):
        self.ai_engine = ai_engine
        self.data_collector = data_collector
        self.client_id = "default_client"  # TODO: Dynamic client ID
    
    async def on_trade_opened(self, position):
        """Callback เมื่อเปิดเทรด"""
        try:
            trade_record = TradeRecord(
                trade_id="",  # Will be auto-generated
                client_id=self.client_id,
                mt5_ticket=position.mt5_ticket,
                symbol=position.symbol,
                entry_time=position.timestamp,
                entry_price=position.entry_price,
                direction=position.direction,
                volume=position.size,
                ai_signal_strength=0.8,  # TODO: Get from AI
                market_regime=self.ai_engine.current_regime.value,
                recovery_level=position.recovery_level,
                is_recovery_trade=position.is_recovery,
                strategy_used="correlation_recovery" if position.is_recovery else "initial_entry",
                spread=0.0,  # TODO: Get from market data
                volatility=0.0  # TODO: Calculate volatility
            )
            
            await self.data_collector.record_trade_entry(trade_record)
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade opening: {e}")
    
    async def on_trade_closed(self, position, exit_price: float):
        """Callback เมื่อปิดเทรด"""
        try:
            await self.data_collector.record_trade_exit(
                trade_id=position.position_id,
                exit_price=exit_price,
                exit_time=datetime.now(),
                final_pnl=position.pnl,
                final_pnl_pips=position.pnl_pips
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade closing: {e}")
    
    async def record_market_data(self, market_data: Dict):
        """บันทึกข้อมูลตลาด"""
        try:
            for symbol, data in market_data.items():
                snapshot = MarketSnapshot(
                    timestamp=data['timestamp'],
                    symbol=symbol,
                    bid=data['bid'],
                    ask=data['ask'],
                    spread=data['spread'],
                    volume=data.get('volume', 0)
                )
                
                await self.data_collector.record_market_snapshot(snapshot)
                
        except Exception as e:
            logger.error(f"❌ Failed to record market data: {e}")

# ตัวอย่างการใช้งาน
async def test_data_collection():
    """ทดสอบระบบเก็บข้อมูล"""
    collector = DataCollector()
    await collector.initialize()
    
    # สร้างข้อมูลทดสอบ
    test_trade = TradeRecord(
        trade_id="",
        client_id="test_client_001",
        symbol="EURUSD",
        entry_time=datetime.now(),
        entry_price=1.1050,
        direction="BUY",
        volume=0.1,
        ai_signal_strength=0.85,
        market_regime="trending",
        strategy_used="initial_entry"
    )
    
    # บันทึกการเข้าเทรด
    await collector.record_trade_entry(test_trade)
    
    # จำลองการปิดเทรด
    await asyncio.sleep(1)
    await collector.record_trade_exit(
        trade_id=test_trade.trade_id,
        exit_price=1.1075,
        exit_time=datetime.now(),
        final_pnl=25.0,
        final_pnl_pips=25.0
    )
    
    # ดูสถิติ
    stats = await collector.get_data_stats()
    print("📊 Data Statistics:")
    print(json.dumps(stats, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(test_data_collection())