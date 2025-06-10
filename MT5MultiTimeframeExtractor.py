import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class MT5MultiTimeframeExtractor:
    """
    Multi-Timeframe MT5 Data Extraction System for SMC AI Trading
    Created by อาจารย์ฟินิกซ์ - Optimized for Gold (XAUUSD.c) Analysis
    🔧 FIXED VERSION - แก้ไข range error และ syntax issues
    """

    def __init__(self, account: int = None, password: str = None, server: str = None):
        """Initialize MT5 connection"""
        self.timezone = pytz.timezone("Etc/UTC")
        self.is_connected = False
        self.account_info = None

        # Essential SMC Timeframes (เรียงจากเล็กไปใหญ่)
        self.smc_timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        # All available timeframes for flexibility
        self.all_timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }

        # 🥇 Gold-specific optimal data periods
        self.gold_data_periods = {
            "M5": timedelta(days=90),    # 3 months for Gold M5
            "M15": timedelta(days=180),  # 6 months for Gold M15
            "H1": timedelta(days=365),   # 1 year for Gold H1
            "H4": timedelta(days=730),   # 2 years for Gold H4
            "D1": timedelta(days=1825),  # 5 years for Gold D1
        }

        # Connect to MT5
        self.connect_mt5(account, password, server)

    def connect_mt5(
        self, account: int = None, password: str = None, server: str = None
    ):
        """Establish connection to MT5"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False

            # Login if credentials provided
            if account and password and server:
                if not mt5.login(account, password=password, server=server):
                    print(f"❌ Login failed: {mt5.last_error()}")
                    return False

            self.account_info = mt5.account_info()
            self.is_connected = True
            print(f"✅ MT5 Connected Successfully!")
            if self.account_info:
                print(f"📊 Account: {self.account_info.login}")
                print(f"🏦 Server: {self.account_info.server}")

            return True

        except Exception as e:
            print(f"❌ Connection Error: {str(e)}")
            return False

    def find_symbol_variations(self, base_symbol: str) -> List[str]:
        """หาชื่อ Symbol ที่แตกต่างกันในโบรกเกอร์ (รองรับทั้ง Forex และ Gold)"""
        print(f"🔍 Searching for {base_symbol} variations...")

        symbols = mt5.symbols_get()
        if symbols is None:
            print("❌ Cannot get symbols list")
            return []

        # 🥇 Gold-specific variations if XAUUSD
        if "XAU" in base_symbol.upper() or "GOLD" in base_symbol.upper():
            possible_variations = [
                base_symbol,
                f"{base_symbol}.c",
                f"{base_symbol}.v",
                f"{base_symbol}.m",
                f"{base_symbol}.raw",
                f"{base_symbol}.pro",
                f"{base_symbol}_",
                f"{base_symbol}#",
                "GOLD",
                "GOLD.c",
                "GOLD.v",
                "XAU/USD",
                "XAUUSD",
                "XAUUSD.c",
                "XAUUSD.v",
                f"{base_symbol}.a",
                f"{base_symbol}cash",
                f"{base_symbol}.cash"
            ]
        else:
            # Standard Forex variations
            possible_variations = [
                base_symbol,
                f"{base_symbol}.c",
                f"{base_symbol}.v",
                f"{base_symbol}.m",
                f"{base_symbol}.raw",
                f"{base_symbol}.pro",
                f"{base_symbol}_",
                f"{base_symbol}#",
            ]

        found_symbols = []

        for variation in possible_variations:
            try:
                tick = mt5.symbol_info_tick(variation)
                if tick is not None:
                    symbol_info = mt5.symbol_info(variation)
                    if symbol_info and symbol_info.trade_mode != 0:  # Trading allowed
                        symbol_data = {
                            "symbol": variation,
                            "description": symbol_info.description,
                            "digits": symbol_info.digits,
                            "spread": symbol_info.spread,
                            "trade_mode": symbol_info.trade_mode,
                        }
                        
                        # 🥇 Add Gold-specific information
                        if "XAU" in variation.upper() or "GOLD" in variation.upper():
                            symbol_data.update({
                                "is_gold": True,
                                "point_value": 0.01,
                                "pip_value": 0.1,
                                "contract_size": getattr(symbol_info, 'trade_contract_size', 100)
                            })
                        
                        found_symbols.append(symbol_data)
                        print(f"✅ Found: {variation} - {symbol_info.description}")
            except:
                continue

        # Search in all symbols for partial matches
        search_keywords = []
        if "XAU" in base_symbol.upper() or "GOLD" in base_symbol.upper():
            search_keywords = ["xau", "gold", "au"]
        else:
            search_keywords = [base_symbol.lower()]

        for symbol in symbols:
            symbol_name_lower = symbol.name.lower()
            if any(keyword in symbol_name_lower for keyword in search_keywords):
                if symbol.name not in [s["symbol"] for s in found_symbols]:
                    # Exclude obvious non-Gold symbols for Gold search
                    if "XAU" in base_symbol.upper():
                        if any(currency in symbol_name_lower for currency in ['aud', 'cad', 'eur', 'gbp', 'jpy']) and 'xau' not in symbol_name_lower:
                            continue
                    
                    try:
                        symbol_data = {
                            "symbol": symbol.name,
                            "description": symbol.description,
                            "digits": symbol.digits,
                            "spread": symbol.spread,
                            "trade_mode": symbol.trade_mode,
                        }
                        
                        # Add Gold info if applicable
                        if any(keyword in symbol_name_lower for keyword in ["xau", "gold"]):
                            symbol_data.update({
                                "is_gold": True,
                                "point_value": 0.01,
                                "pip_value": 0.1,
                                "contract_size": getattr(symbol, 'trade_contract_size', 100)
                            })
                        
                        found_symbols.append(symbol_data)
                        print(f"🔍 Also found: {symbol.name} - {symbol.description}")
                    except:
                        continue

        if not found_symbols:
            print(f"❌ No variations of {base_symbol} found")

        return found_symbols

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information with Gold-specific enhancements"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        symbol_data = {
            "symbol": symbol,
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "tick_size": info.trade_tick_size,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "margin_initial": info.margin_initial,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
            "currency_margin": info.currency_margin,
        }

        # 🥇 Add Gold-specific information
        if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
            symbol_data.update({
                "is_gold": True,
                "point_value": 0.01,  # $0.01 per point
                "pip_value": 0.1,     # $0.10 per pip
                "contract_size": getattr(info, 'trade_contract_size', 100),
                "calculation_mode": "Gold CFD",
                "trading_sessions": {
                    "asian": "22:00-07:00 UTC (Low liquidity)",
                    "london": "08:00-16:00 UTC (High volatility)",
                    "us": "13:00-21:00 UTC (High impact)"
                }
            })

        return symbol_data

    def get_optimal_data_period(
        self, symbol: str, timeframe: str
    ) -> Tuple[datetime, int]:
        """
        หาช่วงข้อมูลที่เหมาะสมสำหรับแต่ละ timeframe
        🥇 ปรับสำหรับ Gold เพื่อรองรับความผันผวนสูง
        """
        if timeframe not in self.all_timeframes:
            return None, 0

        # 🥇 ใช้ Gold-specific periods ถ้าเป็น Gold symbol
        if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
            data_periods = self.gold_data_periods
            print(f"🥇 Using Gold-optimized data periods for {symbol}")
        else:
            # Original periods for Forex
            data_periods = {
                "M5": timedelta(days=180),  # 6 เดือน
                "M15": timedelta(days=365),  # 1 ปี
                "H1": timedelta(days=1095),  # 3 ปี
                "H4": timedelta(days=1825),  # 5 ปี
                "D1": timedelta(days=3650),  # 10 ปี
            }

        end_date = datetime.now(self.timezone)
        start_date = end_date - data_periods.get(timeframe, timedelta(days=365))

        print(
            f"📊 {timeframe} optimal period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            rates = mt5.copy_rates_range(
                symbol, self.all_timeframes[timeframe], start_date, end_date
            )

            if rates is not None and len(rates) > 0:
                count = len(rates)
                first_date = datetime.fromtimestamp(rates[0]["time"], tz=self.timezone)
                print(
                    f"✅ {timeframe}: {count:,} candles available from {first_date.strftime('%Y-%m-%d')}"
                )
                return first_date, count
            else:
                print(f"❌ {timeframe}: No data available")
                return None, 0

        except Exception as e:
            print(f"❌ {timeframe}: Error - {str(e)}")
            return None, 0

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Extract historical data with Gold-specific enhancements
        🔧 FIXED VERSION - แก้ไข range error
        """

        if not self.is_connected:
            print("❌ MT5 not connected!")
            return pd.DataFrame()

        if timeframe not in self.all_timeframes:
            print(f"❌ Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        # ตรวจสอบ symbol
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not found!")
            return pd.DataFrame()

        # ใช้ช่วงข้อมูลที่เหมาะสมถ้าไม่ได้ระบุ
        if start_date is None:
            start_date, expected_count = self.get_optimal_data_period(symbol, timeframe)
            if start_date is None:
                return pd.DataFrame()

        if end_date is None:
            end_date = datetime.now(self.timezone)

        # Convert to UTC if needed
        if start_date.tzinfo is None:
            start_date = self.timezone.localize(start_date)
        if end_date.tzinfo is None:
            end_date = self.timezone.localize(end_date)

        try:
            print(
                f"🔄 Extracting {timeframe} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
            )

            rates = mt5.copy_rates_range(
                symbol, self.all_timeframes[timeframe], start_date, end_date
            )

            if rates is None or len(rates) == 0:
                print(f"❌ No data received for {symbol} {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Add metadata
            df["symbol"] = symbol
            df["timeframe"] = timeframe
            df["digits"] = symbol_info["digits"]

            # Enhanced price metrics
            df["hl2"] = (df["high"] + df["low"]) / 2
            df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
            df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        # 🔧 FIX: Price ranges and volatility (moved up)
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100
            # 🔧 FIX: สร้าง range calculations ขึ้นมาก่อน
            df["range"] = df["high"] - df["low"]
            df["range_pct"] = (df["range"] / df["close"]) * 100

            # 🥇 Gold-specific price calculations
            is_gold = symbol_info.get("is_gold", False)
            if is_gold:
                point_value = symbol_info.get("point_value", 0.01)
                df["range_points"] = df["range"] / point_value
                
                # Gold session analysis
                df["hour"] = df.index.hour
                df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] <= 16)).astype(int)
                df["is_us_session"] = ((df["hour"] >= 13) & (df["hour"] <= 21)).astype(int)
                df["is_asian_session"] = ((df["hour"] >= 22) | (df["hour"] <= 7)).astype(int)
                df["is_high_impact_hour"] = df["hour"].isin([8, 13, 14, 15]).astype(int)
                
                print(f"🥇 Added Gold-specific features for {symbol}")
            else:
                # Standard Forex pip calculations
                pip_size = 0.0001 if "JPY" not in symbol else 0.01
                df["range_pips"] = df["range"] / pip_size

            # Candle patterns
            df["body"] = abs(df["close"] - df["open"])
            df["body_pct"] = np.where(df["range"] > 0, (df["body"] / df["range"]) * 100, 0)
            df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
            df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
            df["is_bullish"] = (df["close"] > df["open"]).astype(int)
            df["is_doji"] = (df["body_pct"] < 10).astype(int)

            # Price movements
            df["price_change"] = df["close"].diff()
            df["price_change_pct"] = df["close"].pct_change() * 100

            # 🥇 Gold-specific price change in points
            if is_gold:
                df["price_change_points"] = df["price_change"] / point_value

            # Volatility measures
            df["atr_14"] = df["range"].rolling(14).mean()
            df["volatility_pct"] = (
                df["close"].rolling(20).std() / df["close"].rolling(20).mean() * 100
            )

            # Basic indicators
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()
            df["ema_20"] = df["close"].ewm(span=20).mean()

            # Volume analysis
            df["volume_ma"] = df["tick_volume"].rolling(20).mean()
            df["volume_ratio"] = np.where(
                df["volume_ma"] > 0, df["tick_volume"] / df["volume_ma"], 1
            )

            # Higher highs, Lower lows for market structure
            df["prev_high"] = df["high"].shift(1)
            df["prev_low"] = df["low"].shift(1)
            
            # 🥇 Adjust rolling window for Gold (more volatile)
            rolling_window = 12 if is_gold else 10
            df["higher_high"] = (
                df["high"] > df["high"].rolling(rolling_window).max().shift(1)
            ).astype(int)
            df["lower_low"] = (
                df["low"] < df["low"].rolling(rolling_window).min().shift(1)
            ).astype(int)

            print(f"✅ {symbol} {timeframe}: {len(df):,} candles extracted")
            print(
                f"📅 Period: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"📊 Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
            
            if is_gold:
                print(f"🥇 Gold features added: sessions, points calculation, volatility analysis")

            return df

        except Exception as e:
            print(f"❌ Data extraction error: {str(e)}")
            print(f"🔧 Error type: {type(e).__name__}")
            return pd.DataFrame()

    def get_smc_dataset(
        self, symbol: str, custom_timeframes: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract complete SMC dataset with all essential timeframes
        🥇 Enhanced for Gold trading
        """

        timeframes = (
            custom_timeframes if custom_timeframes else list(self.smc_timeframes.keys())
        )

        # 🥇 Check if Gold symbol
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        symbol_type = "Gold" if is_gold else "Forex"

        print(f"🚀 Extracting {symbol_type} SMC Dataset for {symbol}")
        print(f"🎯 Timeframes: {', '.join(timeframes)}")
        print("=" * 60)

        smc_data = {}
        extraction_summary = []

        for tf in timeframes:
            if tf not in self.all_timeframes:
                print(f"❌ Invalid timeframe: {tf}")
                continue

            print(f"\n📊 Processing {tf}...")

            df = self.get_historical_data(symbol, tf)

            if not df.empty:
                smc_data[tf] = df

                # Calculate some stats
                candles = len(df)
                days = (df.index[-1] - df.index[0]).days
                avg_candles_per_day = candles / max(days, 1)

                extraction_summary.append(
                    {
                        "timeframe": tf,
                        "candles": candles,
                        "days": days,
                        "avg_per_day": avg_candles_per_day,
                        "start_date": df.index[0].strftime("%Y-%m-%d"),
                        "end_date": df.index[-1].strftime("%Y-%m-%d"),
                        "status": "success",
                        "symbol_type": symbol_type
                    }
                )

                print(f"✅ {tf}: {candles:,} candles ({days} days)")
                
                # 🥇 Gold-specific quality checks
                if is_gold:
                    london_sessions = df["is_london_session"].sum() if "is_london_session" in df.columns else 0
                    us_sessions = df["is_us_session"].sum() if "is_us_session" in df.columns else 0
                    print(f"   🥇 Gold sessions: {london_sessions} London, {us_sessions} US")

            else:
                extraction_summary.append({
                    "timeframe": tf, 
                    "status": "failed",
                    "symbol_type": symbol_type
                })
                print(f"❌ {tf}: Failed to extract data")

        # Print summary
        print("\n" + "=" * 60)
        print(f"📋 {symbol_type.upper()} EXTRACTION SUMMARY")
        print("=" * 60)

        total_candles = 0
        successful_tfs = 0

        for summary in extraction_summary:
            if summary["status"] == "success":
                print(
                    f"✅ {summary['timeframe']:>3}: {summary['candles']:>6,} candles | {summary['start_date']} to {summary['end_date']}"
                )
                total_candles += summary["candles"]
                successful_tfs += 1
            else:
                print(f"❌ {summary['timeframe']:>3}: Failed")

        print("-" * 60)
        print(
            f"🎯 Total: {successful_tfs}/{len(timeframes)} timeframes | {total_candles:,} total candles"
        )

        if successful_tfs == len(timeframes):
            print(f"🎉 Perfect! All {symbol_type} timeframes extracted successfully")
        elif successful_tfs > 0:
            print(f"⚠️ Partial success - some {symbol_type} timeframes missing")
        else:
            print(f"❌ Complete failure - no {symbol_type} data extracted")

        return smc_data

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Enhanced data quality validation with Gold-specific checks"""
        if df.empty:
            return {"status": "empty", "issues": ["No data available"], "score": 0}

        issues = []
        warnings = []
        score = 100

        # Missing values check
        missing_values = df.isnull().sum()
        if missing_values.any():
            missing_pct = (missing_values.sum() / len(df)) * 100
            if missing_pct > 5:
                issues.append(f"High missing values: {missing_pct:.2f}%")
                score -= 20
            else:
                warnings.append(f"Some missing values: {missing_pct:.2f}%")
                score -= 5

        # Duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            if dup_pct > 1:
                issues.append(f"High duplicate timestamps: {dup_pct:.2f}%")
                score -= 15
            else:
                warnings.append(f"Some duplicate timestamps: {dup_pct:.2f}%")
                score -= 5

        # 🥇 Gold-specific price consistency checks
        is_gold = "XAU" in str(df["symbol"].iloc[0]).upper() if "symbol" in df.columns else False
        
        if "close" in df.columns:
            price_changes = df["close"].pct_change().abs()
            
            # Different thresholds for Gold vs Forex
            extreme_threshold = 0.10 if is_gold else 0.05  # Gold can move 10% in extreme cases
            extreme_moves = (price_changes > extreme_threshold).sum()
            
            if extreme_moves > 0:
                extreme_pct = (extreme_moves / len(df)) * 100
                threshold_pct = 1.0 if is_gold else 0.5  # More lenient for Gold
                
                if extreme_pct > threshold_pct:
                    issues.append(
                        f"Many extreme moves: {extreme_moves} ({extreme_pct:.2f}%)"
                    )
                    score -= 10
                else:
                    warnings.append(f"Some extreme moves: {extreme_moves}")
                    score -= 2

        # OHLC consistency
        ohlc_issues = 0
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            ohlc_issues += (df["high"] < df[["open", "close"]].max(axis=1)).sum()
            ohlc_issues += (df["low"] > df[["open", "close"]].min(axis=1)).sum()

            if ohlc_issues > 0:
                ohlc_pct = (ohlc_issues / len(df)) * 100
                if ohlc_pct > 1:
                    issues.append(
                        f"OHLC inconsistencies: {ohlc_issues} ({ohlc_pct:.2f}%)"
                    )
                    score -= 15
                else:
                    warnings.append(f"Minor OHLC issues: {ohlc_issues}")
                    score -= 3

        # 🥇 Gold-specific session coverage check
        if is_gold and "is_london_session" in df.columns:
            london_coverage = df["is_london_session"].mean() * 100
            us_coverage = df["is_us_session"].mean() * 100 if "is_us_session" in df.columns else 0
            
            if london_coverage < 20:  # Less than 20% London session
                warnings.append(f"Low London session coverage: {london_coverage:.1f}%")
                score -= 5
            
            if us_coverage < 15:  # Less than 15% US session
                warnings.append(f"Low US session coverage: {us_coverage:.1f}%")
                score -= 3

        # Determine status
        if score >= 90:
            status = "excellent"
        elif score >= 80:
            status = "good"
        elif score >= 70:
            status = "acceptable"
        elif score >= 50:
            status = "poor"
        else:
            status = "very_poor"

        return {
            "status": status,
            "score": max(0, score),
            "total_candles": len(df),
            "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            "timespan_days": (df.index[-1] - df.index[0]).days,
            "issues": issues,
            "warnings": warnings,
            "is_gold": is_gold,
        }

    def export_smc_dataset(
        self, smc_data: Dict[str, pd.DataFrame], base_filename: str, format: str = "csv"
    ) -> bool:
        """Export complete SMC dataset with Gold-specific enhancements"""

        try:
            # 🥇 Detect if Gold dataset
            is_gold = False
            if smc_data:
                first_df = list(smc_data.values())[0]
                if "symbol" in first_df.columns:
                    symbol = first_df["symbol"].iloc[0]
                    is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()

            symbol_type = "Gold" if is_gold else "Forex"
            print(f"\n💾 Exporting {symbol_type} SMC Dataset...")
            print("-" * 40)

            exported_files = []
            total_rows = 0

            for timeframe, df in smc_data.items():
                if df.empty:
                    continue

                filename = f"{base_filename}_{timeframe}"

                if format.lower() == "csv":
                    file_path = f"{filename}.csv"
                    df.to_csv(file_path)
                elif format.lower() == "parquet":
                    file_path = f"{filename}.parquet"
                    df.to_parquet(file_path)

                exported_files.append(file_path)
                total_rows += len(df)
                print(f"✅ {timeframe}: {file_path} ({len(df):,} rows)")

            # Create enhanced summary
            summary = {
                "symbol": smc_data[list(smc_data.keys())[0]]["symbol"].iloc[0],
                "symbol_type": symbol_type,
                "is_gold": is_gold,
                "timeframes": list(smc_data.keys()),
                "total_files": len(exported_files),
                "total_candles": total_rows,
                "extraction_date": datetime.now().isoformat(),
                "files": exported_files,
            }

            # 🥇 Add Gold-specific summary info
            if is_gold:
                summary.update({
                    "gold_features": [
                        "range_points", "price_change_points", 
                        "is_london_session", "is_us_session", "is_asian_session",
                        "is_high_impact_hour"
                    ],
                    "trading_sessions": {
                        "london": "08:00-16:00 UTC (High volatility)",
                        "us": "13:00-21:00 UTC (High impact)", 
                        "asian": "22:00-07:00 UTC (Low liquidity)"
                    },
                    "point_value": 0.01,
                    "recommended_risk": "1.5% per trade (lower than Forex due to volatility)"
                })

            # Export summary
            import json

            summary_file = f"{base_filename}_SMC_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 {symbol_type} SMC Dataset Complete!")
            print(f"📊 {len(exported_files)} files | {total_rows:,} total candles")
            
            if is_gold:
                print("🥇 Gold-specific features included for enhanced analysis")

            return True

        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return False

    def disconnect(self):
        """Properly disconnect from MT5"""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            print("✅ MT5 disconnected successfully")


# Usage Example - Complete SMC Dataset for Gold
if __name__ == "__main__":
    print("🥇 MT5 Multi-Timeframe Gold SMC Data Extraction")
    print("=" * 50)

    # Initialize extractor
    extractor = MT5MultiTimeframeExtractor()

    # 🥇 Find XAUUSD variations
    print("\n🔍 Searching for XAUUSD variations...")
    xauusd_variants = extractor.find_symbol_variations("XAUUSD")

    if xauusd_variants:
        # Select .c or .v suffix if available (preferred for Gold)
        target_symbol = None
        for variant in xauusd_variants:
            if ".c" in variant["symbol"] or ".v" in variant["symbol"]:
                target_symbol = variant["symbol"]
                break

        if target_symbol is None:
            target_symbol = xauusd_variants[0]["symbol"]

        print(f"\n🎯 Selected Gold Symbol: {target_symbol}")

        # Show Gold symbol details
        gold_info = extractor.get_symbol_info(target_symbol)
        if gold_info and gold_info.get("is_gold"):
            print(f"💰 Point Value: ${gold_info['point_value']}")
            print(f"📊 Contract Size: {gold_info['contract_size']} oz")
            print(f"🎯 Digits: {gold_info['digits']}")

        # Extract complete Gold SMC dataset
        smc_data = extractor.get_smc_dataset(target_symbol)

        if smc_data:
            # Validate each timeframe
            print(f"\n🔍 Gold Data Quality Validation:")
            print("-" * 40)

            overall_quality = []
            for tf, df in smc_data.items():
                quality = extractor.validate_data_quality(df)
                overall_quality.append(quality["score"])
                print(
                    f"{tf:>3}: {quality['status']} (Score: {quality['score']}/100) | {quality['total_candles']:,} candles"
                )

                if quality["issues"]:
                    for issue in quality["issues"]:
                        print(f"     ❌ {issue}")
                if quality["warnings"]:
                    for warning in quality["warnings"]:
                        print(f"     ⚠️ {warning}")

            avg_quality = sum(overall_quality) / len(overall_quality)
            print(f"\n🎯 Average Gold Data Quality Score: {avg_quality:.1f}/100")

            # Export if quality is acceptable
            if avg_quality >= 70:
                print(f"\n💾 Quality acceptable - proceeding with Gold export...")
                base_filename = f"{target_symbol.replace('.', '_')}_SMC_dataset"
                extractor.export_smc_dataset(smc_data, base_filename, "csv")
                print(f"\n🎉 Gold SMC Dataset ready for AI training!")
                print(f"🥇 Files created: {base_filename}_[M5|M15|H1|H4|D1].csv")
                print(f"📊 Summary: {base_filename}_SMC_summary.json")
            else:
                print(
                    f"\n⚠️ Quality too low for AI training (Average: {avg_quality:.1f}/100)"
                )
                print("🔧 Consider checking broker data quality or adjusting timeframes")

        else:
            print("❌ No Gold SMC data extracted!")

    else:
        print("❌ No XAUUSD variants found!")
        print("🔧 Please check:")
        print("   - MT5 connection")
        print("   - Symbol availability in your broker")
        print("   - Market hours (Gold trading may be closed)")

    # Disconnect
    extractor.disconnect()