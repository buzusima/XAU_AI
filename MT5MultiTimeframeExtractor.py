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
    Created by อาจารย์ฟินิกซ์ - Optimized for SMC Analysis
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
            print(
                f"📊 Account: {self.account_info.login if self.account_info else 'Demo'}"
            )
            print(
                f"🏦 Server: {self.account_info.server if self.account_info else 'Default'}"
            )

            return True

        except Exception as e:
            print(f"❌ Connection Error: {str(e)}")
            return False

    def find_symbol_variations(self, base_symbol: str) -> List[str]:
        """หาชื่อ Symbol ที่แตกต่างกันในโบรกเกอร์"""
        print(f"🔍 Searching for {base_symbol} variations...")

        symbols = mt5.symbols_get()
        if symbols is None:
            print("❌ Cannot get symbols list")
            return []

        possible_variations = [
            base_symbol,
            f"{base_symbol}.c",
            f"{base_symbol}.m",
            f"{base_symbol}.raw",
            f"{base_symbol}.pro",
            f"{base_symbol}_",
            f"{base_symbol}#",
        ]

        found_symbols = []

        for variation in possible_variations:
            symbol_info = mt5.symbol_info(variation)
            if symbol_info is not None:
                found_symbols.append(
                    {
                        "symbol": variation,
                        "description": symbol_info.description,
                        "digits": symbol_info.digits,
                        "spread": symbol_info.spread,
                        "trade_mode": symbol_info.trade_mode,
                    }
                )
                print(f"✅ Found: {variation} - {symbol_info.description}")

        # Search in all symbols for partial matches
        for symbol in symbols:
            if base_symbol.lower() in symbol.name.lower() and symbol.name not in [
                s["symbol"] for s in found_symbols
            ]:
                found_symbols.append(
                    {
                        "symbol": symbol.name,
                        "description": symbol.description,
                        "digits": symbol.digits,
                        "spread": symbol.spread,
                        "trade_mode": symbol.trade_mode,
                    }
                )
                print(f"🔍 Also found: {symbol.name} - {symbol.description}")

        if not found_symbols:
            print(f"❌ No variations of {base_symbol} found")

        return found_symbols

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
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

    def get_optimal_data_period(
        self, symbol: str, timeframe: str
    ) -> Tuple[datetime, int]:
        """
        หาช่วงข้อมูลที่เหมาะสมสำหรับแต่ละ timeframe
        M5: 6 เดือน, M15: 1 ปี, H1: 3 ปี, H4: 5 ปี, D1: 10 ปี
        """
        if timeframe not in self.all_timeframes:
            return None, 0

        # กำหนดช่วงข้อมูลตาม timeframe
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
        """Extract historical data with enhanced features"""

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

            # Price ranges and volatility
            df["range"] = df["high"] - df["low"]
            df["range_pct"] = (df["range"] / df["close"]) * 100
            df["range_pips"] = df["range"] / symbol_info["point"]

            # Candle patterns
            df["body"] = abs(df["close"] - df["open"])
            df["body_pct"] = np.where(
                df["range"] > 0, (df["body"] / df["range"]) * 100, 0
            )
            df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
            df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
            df["is_bullish"] = (df["close"] > df["open"]).astype(int)
            df["is_doji"] = (df["body_pct"] < 10).astype(int)

            # Price movements
            df["price_change"] = df["close"].diff()
            df["price_change_pct"] = df["close"].pct_change() * 100
            df["price_change_pips"] = df["price_change"] / symbol_info["point"]

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
            df["higher_high"] = (
                df["high"] > df["high"].rolling(10).max().shift(1)
            ).astype(int)
            df["lower_low"] = (df["low"] < df["low"].rolling(10).min().shift(1)).astype(
                int
            )

            print(f"✅ {symbol} {timeframe}: {len(df):,} candles extracted")
            print(
                f"📅 Period: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"📊 Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

            return df

        except Exception as e:
            print(f"❌ Data extraction error: {str(e)}")
            return pd.DataFrame()

    def get_smc_dataset(
        self, symbol: str, custom_timeframes: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract complete SMC dataset with all essential timeframes
        """

        timeframes = (
            custom_timeframes if custom_timeframes else list(self.smc_timeframes.keys())
        )

        print(f"🚀 Extracting SMC Dataset for {symbol}")
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
                    }
                )

                print(f"✅ {tf}: {candles:,} candles ({days} days)")
            else:
                extraction_summary.append({"timeframe": tf, "status": "failed"})
                print(f"❌ {tf}: Failed to extract data")

        # Print summary
        print("\n" + "=" * 60)
        print("📋 EXTRACTION SUMMARY")
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
            print("🎉 Perfect! All timeframes extracted successfully")
        elif successful_tfs > 0:
            print("⚠️ Partial success - some timeframes missing")
        else:
            print("❌ Complete failure - no data extracted")

        return smc_data

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Enhanced data quality validation"""
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

        # Price consistency
        if "close" in df.columns:
            price_changes = df["close"].pct_change().abs()
            extreme_moves = (price_changes > 0.05).sum()
            if extreme_moves > 0:
                extreme_pct = (extreme_moves / len(df)) * 100
                if extreme_pct > 0.5:
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
        }

    def export_smc_dataset(
        self, smc_data: Dict[str, pd.DataFrame], base_filename: str, format: str = "csv"
    ) -> bool:
        """Export complete SMC dataset"""

        try:
            print(f"\n💾 Exporting SMC Dataset...")
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

            # Create combined summary
            summary = {
                "symbol": smc_data[list(smc_data.keys())[0]]["symbol"].iloc[0],
                "timeframes": list(smc_data.keys()),
                "total_files": len(exported_files),
                "total_candles": total_rows,
                "extraction_date": datetime.now().isoformat(),
                "files": exported_files,
            }

            # Export summary
            import json

            summary_file = f"{base_filename}_SMC_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 SMC Dataset Complete!")
            print(f"📊 {len(exported_files)} files | {total_rows:,} total candles")

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


# Usage Example - Complete SMC Dataset
if __name__ == "__main__":
    print("🚀 MT5 Multi-Timeframe SMC Data Extraction")
    print("=" * 50)

    # Initialize extractor
    extractor = MT5MultiTimeframeExtractor()

    # Find EURUSD variations
    print("\n🔍 Searching for EURUSD variations...")
    eurusd_variants = extractor.find_symbol_variations("EURUSD")

    if eurusd_variants:
        # Select .c suffix if available
        target_symbol = None
        for variant in eurusd_variants:
            if ".c" in variant["symbol"]:
                target_symbol = variant["symbol"]
                break

        if target_symbol is None:
            target_symbol = eurusd_variants[0]["symbol"]

        print(f"\n🎯 Selected Symbol: {target_symbol}")

        # Extract complete SMC dataset
        smc_data = extractor.get_smc_dataset(target_symbol)

        if smc_data:
            # Validate each timeframe
            print(f"\n🔍 Data Quality Validation:")
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
            print(f"\n🎯 Average Quality Score: {avg_quality:.1f}/100")

            # Export if quality is acceptable
            if avg_quality >= 70:
                print(f"\n💾 Quality acceptable - proceeding with export...")
                base_filename = f"{target_symbol.replace('.', '_')}_SMC_dataset"
                extractor.export_smc_dataset(smc_data, base_filename, "csv")
                print(f"\n🎉 SMC Dataset ready for AI training!")
            else:
                print(
                    f"\n⚠️ Quality too low for AI training (Average: {avg_quality:.1f}/100)"
                )

        else:
            print("❌ No SMC data extracted!")

    else:
        print("❌ No EURUSD variants found!")

    # Disconnect
    extractor.disconnect()
