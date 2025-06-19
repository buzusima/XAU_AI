import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
import asyncio
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MLTrainingRecord:
    """บันทึกข้อมูลสำหรับ ML Training - Fixed Order"""
    
    # === REQUIRED FIELDS (ไม่มี default) ===
    timestamp: datetime
    record_id: str
    client_id: str
    symbol: str
    timeframe: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    spread: float
    market_regime: str
    
    # === OPTIONAL FIELDS (มี default) ===
    # Technical Indicators
    sma_5: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Oscillators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None
    
    # Market Context
    volatility_1h: Optional[float] = None
    volatility_4h: Optional[float] = None
    volatility_daily: Optional[float] = None
    
    # Time Features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_london_session: bool = False
    is_ny_session: bool = False
    is_asia_session: bool = False
    
    # Correlation Data
    correlation_eur: Optional[float] = None
    correlation_gbp: Optional[float] = None
    correlation_jpy: Optional[float] = None
    correlation_gold: Optional[float] = None
    
    # Sentiment Data
    news_sentiment: Optional[float] = None
    economic_impact: Optional[str] = None
    
    # AI Decision Data
    ai_signal_strength: float = 0.0
    ai_confidence: float = 0.0
    ai_predicted_direction: Optional[str] = None
    ai_predicted_pips: Optional[float] = None
    
    # Strategy Context
    recovery_level: int = 0
    is_recovery_trade: bool = False
    strategy_used: str = "initial_entry"
    position_size: float = 0.0
    
    # Actual Results (Labels)
    actual_direction: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    actual_pips: Optional[float] = None
    hold_duration_minutes: Optional[int] = None
    
    # Profit/Loss
    actual_pnl: Optional[float] = None
    actual_return_pct: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    
    # Trade Quality
    trade_result: Optional[str] = None
    win_quality: Optional[str] = None
    loss_severity: Optional[str] = None
    
    # Risk Metrics
    portfolio_risk_pct: float = 0.0
    position_risk_pct: float = 0.0
    account_equity: float = 0.0
    
    # Data Quality
    data_completeness: float = 1.0
    data_source: str = "mt5"
    
    def to_dict(self) -> Dict:
        """แปลงเป็น dictionary สำหรับบันทึก"""
        data = asdict(self)
        # แปลง datetime เป็น string
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MLTrainingRecord':
        """สร้างจาก dictionary"""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class MLDataCollector:
    """เก็บข้อมูลสำหรับ Machine Learning Training"""
    
    def __init__(self, db_path: str = "data/ml_training.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_initialized = False
        
        # CSV Export paths
        self.csv_dir = Path("data/ml_exports")
        self.csv_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """เริ่มต้นฐานข้อมูล"""
        try:
            await self._create_ml_tables()
            self.is_initialized = True
            logger.info("✅ ML Data Collector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Data Collector: {e}")
            raise
    
    async def _create_ml_tables(self):
        """สร้างตารางสำหรับ ML Data"""
        try:
            import aiosqlite
        except ImportError:
            logger.error("❌ aiosqlite not installed. Run: pip install aiosqlite")
            raise
        
        async with aiosqlite.connect(self.db_path) as db:
            # ตาราง ML Training Records
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    client_id TEXT NOT NULL,
                    
                    -- Market Data
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    spread REAL NOT NULL,
                    
                    -- Technical Indicators
                    sma_5 REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    stochastic_k REAL,
                    stochastic_d REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    bb_width REAL,
                    bb_position REAL,
                    
                    -- Market Context
                    market_regime TEXT NOT NULL,
                    volatility_1h REAL,
                    volatility_4h REAL,
                    volatility_daily REAL,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    is_london_session BOOLEAN,
                    is_ny_session BOOLEAN,
                    is_asia_session BOOLEAN,
                    
                    -- Correlation Data
                    correlation_eur REAL,
                    correlation_gbp REAL,
                    correlation_jpy REAL,
                    correlation_gold REAL,
                    
                    -- Sentiment Data
                    news_sentiment REAL,
                    economic_impact TEXT,
                    
                    -- AI Decision
                    ai_signal_strength REAL,
                    ai_confidence REAL,
                    ai_predicted_direction TEXT,
                    ai_predicted_pips REAL,
                    recovery_level INTEGER,
                    is_recovery_trade BOOLEAN,
                    strategy_used TEXT,
                    position_size REAL,
                    
                    -- Actual Results (Labels)
                    actual_direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    actual_pips REAL,
                    hold_duration_minutes INTEGER,
                    actual_pnl REAL,
                    actual_return_pct REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    trade_result TEXT,
                    win_quality TEXT,
                    loss_severity TEXT,
                    
                    -- Risk Metrics
                    portfolio_risk_pct REAL,
                    position_risk_pct REAL,
                    account_equity REAL,
                    
                    -- Data Quality
                    data_completeness REAL,
                    data_source TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ตาราง Feature Importance
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    training_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ตาราง Model Performance
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    auc_score REAL,
                    profit_factor REAL,
                    win_rate REAL,
                    avg_trade_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    training_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # สร้าง indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ml_timestamp ON ml_training_records(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ml_symbol ON ml_training_records(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ml_client ON ml_training_records(client_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ml_result ON ml_training_records(trade_result)")
            
            await db.commit()
    
    async def record_ml_data(self, record: MLTrainingRecord) -> bool:
        """บันทึกข้อมูล ML Training"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                # แปลงเป็น dict
                data = record.to_dict()
                
                # สร้าง SQL แบบ Dynamic
                columns = []
                values = []
                placeholders = []
                
                for key, value in data.items():
                    if value is not None:  # เฉพาะ field ที่มีค่า
                        columns.append(key)
                        values.append(value)
                        placeholders.append('?')
                
                sql = f"""
                    INSERT OR REPLACE INTO ml_training_records 
                    ({', '.join(columns)}) 
                    VALUES ({', '.join(placeholders)})
                """
                
                await db.execute(sql, values)
                await db.commit()
            
            logger.debug(f"✅ ML record saved: {record.record_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save ML record: {e}")
            return False
    
    async def export_training_dataset(self, 
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    symbols: Optional[List[str]] = None,
                                    min_data_completeness: float = 0.8) -> Dict:
        """ส่งออกข้อมูลสำหรับ Training ML Model"""
        try:
            # สร้าง query conditions
            conditions = ["data_completeness >= ?"]
            params = [min_data_completeness]
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date.isoformat())
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(symbols)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                # ดึงข้อมูลทั้งหมด
                cursor = await db.execute(f"""
                    SELECT * FROM ml_training_records 
                    {where_clause}
                    ORDER BY timestamp DESC
                """, params)
                
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                # แปลงเป็น DataFrame
                df = pd.DataFrame(rows, columns=columns)
                
                if df.empty:
                    return {"status": "no_data", "message": "No data found for specified criteria"}
                
                # แยกข้อมูลเป็นส่วนๆ
                feature_columns = [
                    # Price features
                    'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'spread',
                    
                    # Technical indicators
                    'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                    'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                    'stochastic_k', 'stochastic_d',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                    
                    # Market context
                    'volatility_1h', 'volatility_4h', 'volatility_daily',
                    'hour_of_day', 'day_of_week',
                    'is_london_session', 'is_ny_session', 'is_asia_session',
                    
                    # Correlations
                    'correlation_eur', 'correlation_gbp', 'correlation_jpy', 'correlation_gold',
                    
                    # Sentiment
                    'news_sentiment',
                    
                    # AI context
                    'ai_signal_strength', 'ai_confidence', 'recovery_level', 'position_size',
                    'portfolio_risk_pct', 'position_risk_pct'
                ]
                
                target_columns = [
                    'actual_pnl', 'actual_return_pct', 'actual_pips',
                    'trade_result', 'win_quality', 'loss_severity',
                    'max_favorable_excursion', 'max_adverse_excursion'
                ]
                
                # กรองเฉพาะ columns ที่มีอยู่จริง
                available_features = [col for col in feature_columns if col in df.columns]
                available_targets = [col for col in target_columns if col in df.columns]
                
                # สร้าง datasets
                features_df = df[available_features].copy()
                targets_df = df[available_targets].copy()
                metadata_df = df[['record_id', 'timestamp', 'symbol', 'client_id']].copy()
                
                # บันทึกเป็นไฟล์
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = self.csv_dir / f"export_{timestamp}"
                export_dir.mkdir(exist_ok=True)
                
                features_file = export_dir / "features.csv"
                targets_file = export_dir / "targets.csv"
                metadata_file = export_dir / "metadata.csv"
                full_dataset_file = export_dir / "full_dataset.csv"
                
                features_df.to_csv(features_file, index=False)
                targets_df.to_csv(targets_file, index=False)
                metadata_df.to_csv(metadata_file, index=False)
                df.to_csv(full_dataset_file, index=False)
                
                # สร้าง summary
                summary = {
                    "status": "success",
                    "export_timestamp": timestamp,
                    "export_dir": str(export_dir),
                    "total_records": len(df),
                    "date_range": {
                        "start": df['timestamp'].min(),
                        "end": df['timestamp'].max()
                    },
                    "symbols": df['symbol'].unique().tolist(),
                    "files": {
                        "features": str(features_file),
                        "targets": str(targets_file),
                        "metadata": str(metadata_file),
                        "full_dataset": str(full_dataset_file)
                    },
                    "feature_stats": {
                        "total_features": len(available_features),
                        "missing_data": features_df.isnull().sum().to_dict()
                    },
                    "target_distribution": {}
                }
                
                # เพิ่มสถิติ targets
                if 'trade_result' in targets_df.columns:
                    summary["target_distribution"]["win_rate"] = (targets_df['trade_result'] == 'WIN').mean()
                if 'actual_pnl' in targets_df.columns:
                    summary["target_distribution"]["avg_pnl"] = targets_df['actual_pnl'].mean()
                if 'actual_return_pct' in targets_df.columns:
                    summary["target_distribution"]["avg_return"] = targets_df['actual_return_pct'].mean()
                
                # บันทึก summary
                summary_file = export_dir / "summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                logger.info(f"✅ ML dataset exported: {export_dir}")
                return summary
                
        except Exception as e:
            logger.error(f"❌ Failed to export ML dataset: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_ml_stats(self) -> Dict:
        """ดึงสถิติข้อมูล ML"""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                # ข้อมูลทั่วไป
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        COUNT(DISTINCT client_id) as unique_clients,
                        MIN(timestamp) as earliest_record,
                        MAX(timestamp) as latest_record,
                        AVG(data_completeness) as avg_completeness
                    FROM ml_training_records
                """)
                general_stats = dict(await cursor.fetchone())
                
                # สถิติ trade results
                cursor = await db.execute("""
                    SELECT 
                        trade_result,
                        COUNT(*) as count,
                        AVG(actual_pnl) as avg_pnl,
                        AVG(actual_return_pct) as avg_return
                    FROM ml_training_records 
                    WHERE trade_result IS NOT NULL
                    GROUP BY trade_result
                """)
                trade_stats = {}
                async for row in cursor:
                    if row[0]:  # ตรวจสอบว่าไม่ใช่ NULL
                        trade_stats[row[0]] = {
                            'count': row[1],
                            'avg_pnl': row[2] or 0,
                            'avg_return': row[3] or 0
                        }
                
                # สถิติตาม symbol
                cursor = await db.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as count,
                        AVG(actual_pnl) as avg_pnl
                    FROM ml_training_records 
                    WHERE actual_pnl IS NOT NULL
                    GROUP BY symbol
                    ORDER BY count DESC
                """)
                symbol_stats = {}
                async for row in cursor:
                    symbol_stats[row[0]] = {
                        'count': row[1],
                        'avg_pnl': row[2] or 0
                    }
                
                return {
                    "general": general_stats,
                    "trade_results": trade_stats,
                    "symbols": symbol_stats,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get ML stats: {e}")
            return {"error": str(e)}

# Integration กับ AI Engine
class MLDataIntegration:
    """Integration ระหว่าง AI Engine และ ML Data Collector"""
    
    def __init__(self, ai_engine, ml_collector: MLDataCollector):
        self.ai_engine = ai_engine
        self.ml_collector = ml_collector
        self.client_id = getattr(ai_engine.settings, 'client_id', 'default_client')
    
    async def record_market_analysis(self, symbol: str, market_data: Dict):
        """บันทึกการวิเคราะห์ตลาดสำหรับ ML"""
        try:
            now = datetime.now()
            
            record = MLTrainingRecord(
                timestamp=now,
                record_id=f"{symbol}_{now.timestamp()}",
                client_id=self.client_id,
                
                # Required fields
                symbol=symbol,
                timeframe="M1",
                open_price=market_data.get('open', market_data['bid']),
                high_price=market_data.get('high', market_data['ask']),
                low_price=market_data.get('low', market_data['bid']),
                close_price=market_data['bid'],
                volume=market_data.get('volume', 0),
                spread=market_data['spread'],
                market_regime=self.ai_engine.current_regime.value,
                
                # Optional fields
                hour_of_day=now.hour,
                day_of_week=now.weekday(),
                ai_signal_strength=0.5,
                ai_confidence=0.5,
                account_equity=self.ai_engine.settings.account_balance,
                data_source="mt5_live"
            )
            
            await self.ml_collector.record_ml_data(record)
            
        except Exception as e:
            logger.error(f"❌ Failed to record market analysis: {e}")

# ตัวอย่างการใช้งาน
async def test_ml_data_collection():
    """ทดสอบระบบเก็บข้อมูล ML"""
    
    # สร้าง ML Data Collector
    ml_collector = MLDataCollector()
    await ml_collector.initialize()
    
    # สร้างข้อมูลทดสอบ
    test_record = MLTrainingRecord(
        timestamp=datetime.now(),
        record_id="test_001",
        client_id="test_client",
        symbol="EURUSD",
        timeframe="M1",
        open_price=1.1050,
        high_price=1.1055,
        low_price=1.1048,
        close_price=1.1052,
        volume=1000,
        spread=0.0002,
        market_regime="trending",
        ai_signal_strength=0.8,
        ai_confidence=0.75,
        ai_predicted_direction="BUY",
        hour_of_day=10,
        day_of_week=1,
        account_equity=5000.0
    )
    
    # บันทึกข้อมูล
    success = await ml_collector.record_ml_data(test_record)
    print(f"✅ Record saved: {success}")
    
    # ดึงสถิติ
    stats = await ml_collector.get_ml_stats()
    print("📊 ML Data Stats:")
    print(json.dumps(stats, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(test_ml_data_collection())