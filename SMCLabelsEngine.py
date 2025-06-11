import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class SMCLabelsEngine:
    """
    SMC Training Labels Creation Engine for AI Trading
    Created by อาจารย์ฟินิกซ์ - Professional Label Engineering

    Creates high-quality training labels based on:
    - SMC Entry/Exit Logic
    - Risk Management Rules
    - Multiple Target/Stop Levels
    - Realistic Trading Scenarios
    """

    def __init__(self):
        """Initialize SMC Labels Engine"""
        # Risk Management Parameters
        self.default_risk_reward = 2.0  # Default R:R ratio
        self.max_holding_periods = {  # Max holding periods per TF
            "M5": 288,  # 24 hours
            "M15": 96,  # 24 hours
            "H1": 24,  # 24 hours
            "H4": 12,  # 48 hours
            "D1": 5,  # 5 days
        }

        # Entry Signal Thresholds
        self.min_confluence = 2  # Minimum SMC confluence for entry
        self.structure_break_threshold = 0.5  # Minimum structure break strength

        # Target/Stop Calculation Methods
        self.atr_multiplier_sl = 1.5  # ATR multiplier for Stop Loss
        self.atr_multiplier_tp = 3.0  # ATR multiplier for Take Profit

        print("🎯 SMC Labels Engine Initialized")
        print("📊 Ready for Professional Label Creation")

    def load_smc_features(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load SMC features dataset"""
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        smc_features = {}

        print("📂 Loading SMC Features Dataset...")
        print("-" * 40)

        for tf in timeframes:
            try:
                filename = f"{base_filename}_SMC_features_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                smc_features[tf] = df
                print(
                    f"✅ {tf:>3}: {len(df):,} candles with {len(df.columns)} features"
                )
            except FileNotFoundError:
                print(f"❌ {tf:>3}: File not found - {filename}")
            except Exception as e:
                print(f"❌ {tf:>3}: Error loading - {str(e)}")

        print("-" * 40)
        total_candles = sum(len(df) for df in smc_features.values())
        print(f"📊 Total: {len(smc_features)} timeframes | {total_candles:,} candles")

        return smc_features

    def identify_entry_signals(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Identify high-probability entry signals based on SMC confluence
        """
        df = df.copy()

        # Initialize entry signal columns
        df["long_entry"] = 0
        df["short_entry"] = 0
        df["entry_strength"] = 0
        df["entry_reason"] = ""

        for i in range(50, len(df)):  # Start after enough history
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]

            # Check for Long Entry Conditions
            long_conditions = []

            # 1. Bullish Structure Break
            if current_candle["structure_break"] > 0:
                long_conditions.append("BOS_Bull")

            # 2. CHoCH to Bullish
            if current_candle["choch"] > 0:
                long_conditions.append("CHoCH_Bull")

            # 3. Bullish Order Block Test
            if current_candle["bullish_ob"] == 1 or (
                not pd.isna(current_candle["ob_low"])
                and current_candle["low"] <= current_candle["ob_low"]
                and current_candle["close"] > current_candle["ob_low"]
            ):
                long_conditions.append("OB_Test")

            # 4. Fair Value Gap Fill (Bullish)
            if current_candle["bullish_fvg"] == 1 or (
                not pd.isna(current_candle["fvg_low"])
                and current_candle["low"] <= current_candle["fvg_low"]
                and current_candle["close"] > current_candle["fvg_low"]
            ):
                long_conditions.append("FVG_Fill")

            # 5. Market Structure is Bullish
            if current_candle["market_structure"] > 0:
                long_conditions.append("Bull_Structure")

            # 6. Liquidity Sweep (Buy Liquidity)
            if current_candle["buy_liquidity"] == 1:
                long_conditions.append("Liquidity_Sweep")

            # Check confluence for Long Entry
            if len(long_conditions) >= self.min_confluence:
                df.iloc[i, df.columns.get_loc("long_entry")] = 1
                df.iloc[i, df.columns.get_loc("entry_strength")] = len(long_conditions)
                df.iloc[i, df.columns.get_loc("entry_reason")] = "+".join(
                    long_conditions
                )

            # Check for Short Entry Conditions
            short_conditions = []

            # 1. Bearish Structure Break
            if current_candle["structure_break"] < 0:
                short_conditions.append("BOS_Bear")

            # 2. CHoCH to Bearish
            if current_candle["choch"] < 0:
                short_conditions.append("CHoCH_Bear")

            # 3. Bearish Order Block Test
            if current_candle["bearish_ob"] == 1 or (
                not pd.isna(current_candle["ob_high"])
                and current_candle["high"] >= current_candle["ob_high"]
                and current_candle["close"] < current_candle["ob_high"]
            ):
                short_conditions.append("OB_Test")

            # 4. Fair Value Gap Fill (Bearish)
            if current_candle["bearish_fvg"] == 1 or (
                not pd.isna(current_candle["fvg_high"])
                and current_candle["high"] >= current_candle["fvg_high"]
                and current_candle["close"] < current_candle["fvg_high"]
            ):
                short_conditions.append("FVG_Fill")

            # 5. Market Structure is Bearish
            if current_candle["market_structure"] < 0:
                short_conditions.append("Bear_Structure")

            # 6. Liquidity Sweep (Sell Liquidity)
            if current_candle["sell_liquidity"] == 1:
                short_conditions.append("Liquidity_Sweep")

            # Check confluence for Short Entry
            if len(short_conditions) >= self.min_confluence:
                df.iloc[i, df.columns.get_loc("short_entry")] = 1
                df.iloc[i, df.columns.get_loc("entry_strength")] = len(short_conditions)
                df.iloc[i, df.columns.get_loc("entry_reason")] = "+".join(
                    short_conditions
                )

        return df

    def calculate_stop_take_levels(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Calculate realistic Stop Loss and Take Profit levels using SMC logic
        """
        df = df.copy()

        # Initialize stop/take columns
        df["stop_loss"] = np.nan
        df["take_profit_1"] = np.nan
        df["take_profit_2"] = np.nan
        df["risk_reward"] = np.nan

        # Calculate ATR for dynamic stop/take levels
        if "atr_14" not in df.columns:
            df["atr_14"] = df["range"].rolling(14).mean()

        for i in range(len(df)):
            current_price = df["close"].iloc[i]
            atr = df["atr_14"].iloc[i]

            if pd.isna(atr) or atr == 0:
                continue

            # Long Entry Stop/Take Calculation
            if df["long_entry"].iloc[i] == 1:
                # Stop Loss: Below recent swing low or ATR-based
                recent_lows = df.iloc[max(0, i - 20) : i]["low"]
                if len(recent_lows) > 0:
                    swing_low = recent_lows.min()
                    atr_stop = current_price - (atr * self.atr_multiplier_sl)
                    stop_loss = min(swing_low, atr_stop)
                else:
                    stop_loss = current_price - (atr * self.atr_multiplier_sl)

                # Take Profit levels
                risk = current_price - stop_loss
                take_profit_1 = current_price + (risk * self.default_risk_reward)
                take_profit_2 = current_price + (risk * self.default_risk_reward * 2)

                df.iloc[i, df.columns.get_loc("stop_loss")] = stop_loss
                df.iloc[i, df.columns.get_loc("take_profit_1")] = take_profit_1
                df.iloc[i, df.columns.get_loc("take_profit_2")] = take_profit_2
                df.iloc[i, df.columns.get_loc("risk_reward")] = (
                    risk / (current_price - stop_loss) if risk > 0 else 0
                )

            # Short Entry Stop/Take Calculation
            if df["short_entry"].iloc[i] == 1:
                # Stop Loss: Above recent swing high or ATR-based
                recent_highs = df.iloc[max(0, i - 20) : i]["high"]
                if len(recent_highs) > 0:
                    swing_high = recent_highs.max()
                    atr_stop = current_price + (atr * self.atr_multiplier_sl)
                    stop_loss = max(swing_high, atr_stop)
                else:
                    stop_loss = current_price + (atr * self.atr_multiplier_sl)

                # Take Profit levels
                risk = stop_loss - current_price
                take_profit_1 = current_price - (risk * self.default_risk_reward)
                take_profit_2 = current_price - (risk * self.default_risk_reward * 2)

                df.iloc[i, df.columns.get_loc("stop_loss")] = stop_loss
                df.iloc[i, df.columns.get_loc("take_profit_1")] = take_profit_1
                df.iloc[i, df.columns.get_loc("take_profit_2")] = take_profit_2
                df.iloc[i, df.columns.get_loc("risk_reward")] = (
                    risk / (stop_loss - current_price) if risk > 0 else 0
                )

        return df

    def simulate_trade_outcomes(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Simulate realistic trade outcomes based on actual price movement
        """
        df = df.copy()

        # Initialize outcome columns
        df["trade_outcome"] = 0  # 1=Win, -1=Loss, 0=No Trade
        df["exit_price"] = np.nan
        df["exit_reason"] = ""
        df["holding_periods"] = 0
        df["pnl_pips"] = 0
        df["pnl_percent"] = 0

        max_holding = self.max_holding_periods.get(timeframe, 24)

        # Process each entry signal
        entry_indices = df[(df["long_entry"] == 1) | (df["short_entry"] == 1)].index

        for entry_idx in entry_indices:
            if entry_idx not in df.index:
                continue

            entry_row = df.loc[entry_idx]
            entry_price = entry_row["close"]
            stop_loss = entry_row["stop_loss"]
            take_profit_1 = entry_row["take_profit_1"]

            if pd.isna(stop_loss) or pd.isna(take_profit_1):
                continue

            is_long = entry_row["long_entry"] == 1

            # Look forward to simulate trade
            entry_pos = df.index.get_loc(entry_idx)
            max_look_ahead = min(entry_pos + max_holding, len(df))

            trade_closed = False

            for future_pos in range(entry_pos + 1, max_look_ahead):
                future_candle = df.iloc[future_pos]
                high_price = future_candle["high"]
                low_price = future_candle["low"]
                close_price = future_candle["close"]

                holding_periods = future_pos - entry_pos

                if is_long:
                    # Check Long Trade
                    if low_price <= stop_loss:
                        # Stop Loss Hit
                        df.loc[entry_idx, "trade_outcome"] = -1
                        df.loc[entry_idx, "exit_price"] = stop_loss
                        df.loc[entry_idx, "exit_reason"] = "Stop_Loss"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        df.loc[entry_idx, "pnl_pips"] = (
                            stop_loss - entry_price
                        ) / 0.00001
                        df.loc[entry_idx, "pnl_percent"] = (
                            (stop_loss / entry_price) - 1
                        ) * 100
                        trade_closed = True
                        break
                    elif high_price >= take_profit_1:
                        # Take Profit Hit
                        df.loc[entry_idx, "trade_outcome"] = 1
                        df.loc[entry_idx, "exit_price"] = take_profit_1
                        df.loc[entry_idx, "exit_reason"] = "Take_Profit"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        df.loc[entry_idx, "pnl_pips"] = (
                            take_profit_1 - entry_price
                        ) / 0.00001
                        df.loc[entry_idx, "pnl_percent"] = (
                            (take_profit_1 / entry_price) - 1
                        ) * 100
                        trade_closed = True
                        break

                else:
                    # Check Short Trade
                    if high_price >= stop_loss:
                        # Stop Loss Hit
                        df.loc[entry_idx, "trade_outcome"] = -1
                        df.loc[entry_idx, "exit_price"] = stop_loss
                        df.loc[entry_idx, "exit_reason"] = "Stop_Loss"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        df.loc[entry_idx, "pnl_pips"] = (
                            entry_price - stop_loss
                        ) / 0.00001
                        df.loc[entry_idx, "pnl_percent"] = (
                            (entry_price / stop_loss) - 1
                        ) * 100
                        trade_closed = True
                        break
                    elif low_price <= take_profit_1:
                        # Take Profit Hit
                        df.loc[entry_idx, "trade_outcome"] = 1
                        df.loc[entry_idx, "exit_price"] = take_profit_1
                        df.loc[entry_idx, "exit_reason"] = "Take_Profit"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        df.loc[entry_idx, "pnl_pips"] = (
                            entry_price - take_profit_1
                        ) / 0.00001
                        df.loc[entry_idx, "pnl_percent"] = (
                            (entry_price / take_profit_1) - 1
                        ) * 100
                        trade_closed = True
                        break

            # If trade not closed by Stop/Take, close at max holding period
            if not trade_closed and max_look_ahead > entry_pos + 1:
                final_candle = df.iloc[max_look_ahead - 1]
                exit_price = final_candle["close"]

                if is_long:
                    outcome = 1 if exit_price > entry_price else -1
                    pnl_pips = (exit_price - entry_price) / 0.00001
                    pnl_percent = ((exit_price / entry_price) - 1) * 100
                else:
                    outcome = 1 if exit_price < entry_price else -1
                    pnl_pips = (entry_price - exit_price) / 0.00001
                    pnl_percent = ((entry_price / exit_price) - 1) * 100

                df.loc[entry_idx, "trade_outcome"] = outcome
                df.loc[entry_idx, "exit_price"] = exit_price
                df.loc[entry_idx, "exit_reason"] = "Time_Exit"
                df.loc[entry_idx, "holding_periods"] = max_holding
                df.loc[entry_idx, "pnl_pips"] = pnl_pips
                df.loc[entry_idx, "pnl_percent"] = pnl_percent

        return df

    def create_classification_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create multiple types of classification labels for different AI models
        """
        df = df.copy()

        # 1. Simple Direction Labels (3-class)
        df["direction_label"] = 0  # 0=Hold, 1=Long, -1=Short
        df.loc[df["long_entry"] == 1, "direction_label"] = 1
        df.loc[df["short_entry"] == 1, "direction_label"] = -1

        # 2. Signal Quality Labels (4-class)
        df["signal_quality"] = 0  # 0=No Signal, 1=Weak, 2=Medium, 3=Strong
        df.loc[df["entry_strength"] == 2, "signal_quality"] = 1
        df.loc[df["entry_strength"] == 3, "signal_quality"] = 2
        df.loc[df["entry_strength"] >= 4, "signal_quality"] = 3

        # 3. Trade Outcome Labels (for supervised learning)
        df["outcome_label"] = 0  # 0=No Trade, 1=Profitable, -1=Loss
        df.loc[df["trade_outcome"] == 1, "outcome_label"] = 1
        df.loc[df["trade_outcome"] == -1, "outcome_label"] = -1

        # 4. Risk-Adjusted Labels (considering R:R)
        df["risk_adjusted_label"] = 0
        # Only mark as positive if trade is profitable AND has good R:R
        profitable_trades = (df["trade_outcome"] == 1) & (df["pnl_pips"] > 10)
        df.loc[profitable_trades, "risk_adjusted_label"] = 1

        # Mark as negative if loss exceeds certain threshold
        significant_losses = (df["trade_outcome"] == -1) & (df["pnl_pips"] < -20)
        df.loc[significant_losses, "risk_adjusted_label"] = -1

        return df

    def process_single_timeframe_labels(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Complete label creation process for single timeframe
        """
        print(f"🎯 Creating labels for {timeframe}...")

        # Step 1: Identify entry signals
        df = self.identify_entry_signals(df, timeframe)
        long_entries = df["long_entry"].sum()
        short_entries = df["short_entry"].sum()
        print(f"   📊 Entry Signals: {long_entries} long, {short_entries} short")

        # Step 2: Calculate stop/take levels
        df = self.calculate_stop_take_levels(df, timeframe)
        valid_levels = (~pd.isna(df["stop_loss"])).sum()
        print(f"   🎯 Valid Stop/Take Levels: {valid_levels}")

        # Step 3: Simulate trade outcomes
        df = self.simulate_trade_outcomes(df, timeframe)
        total_trades = (df["trade_outcome"] != 0).sum()
        winning_trades = (df["trade_outcome"] == 1).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        print(
            f"   💹 Trade Simulation: {total_trades} trades, {win_rate:.1f}% win rate"
        )

        # Step 4: Create classification labels
        df = self.create_classification_labels(df)
        direction_signals = (df["direction_label"] != 0).sum()
        quality_signals = (df["signal_quality"] > 0).sum()
        print(
            f"   🏷️ Labels Created: {direction_signals} direction, {quality_signals} quality signals"
        )

        print(f"✅ {timeframe} label creation complete!")

        return df

    def process_complete_labels_dataset(
        self, smc_features: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process complete dataset with trading labels
        """
        print("🎯 Creating SMC Trading Labels Dataset")
        print("=" * 50)

        labeled_data = {}

        for timeframe, df in smc_features.items():
            print(f"\n📊 Processing {timeframe} ({len(df):,} candles)...")
            labeled_df = self.process_single_timeframe_labels(df, timeframe)
            labeled_data[timeframe] = labeled_df

        print("\n" + "=" * 50)
        print("📋 TRADING LABELS SUMMARY")
        print("=" * 50)

        # Summary statistics
        total_signals = 0
        total_trades = 0
        total_winners = 0

        for timeframe, df in labeled_data.items():
            signals = (df["direction_label"] != 0).sum()
            trades = (df["trade_outcome"] != 0).sum()
            winners = (df["trade_outcome"] == 1).sum()
            win_rate = (winners / trades * 100) if trades > 0 else 0

            total_signals += signals
            total_trades += trades
            total_winners += winners

            print(
                f"{timeframe:>3}: {signals:>4} signals | {trades:>4} trades | {win_rate:>5.1f}% win rate"
            )

        overall_win_rate = (
            (total_winners / total_trades * 100) if total_trades > 0 else 0
        )
        print("-" * 50)
        print(
            f"📊 Total: {total_signals} signals | {total_trades} trades | {overall_win_rate:.1f}% overall win rate"
        )
        print("🎉 SMC Trading Labels Complete!")

        return labeled_data

    def export_labeled_dataset(
        self, labeled_data: Dict[str, pd.DataFrame], base_filename: str
    ) -> bool:
        """
        Export complete labeled dataset ready for AI training
        """
        try:
            print(f"\n💾 Exporting Labeled Training Dataset...")
            print("-" * 40)

            exported_files = []

            for timeframe, df in labeled_data.items():
                filename = f"{base_filename}_labeled_{timeframe}.csv"
                df.to_csv(filename)
                exported_files.append(filename)

                # Count different label types
                direction_labels = (df["direction_label"] != 0).sum()
                quality_labels = (df["signal_quality"] > 0).sum()
                outcome_labels = (df["outcome_label"] != 0).sum()

                print(f"✅ {timeframe}: {filename}")
                print(f"    📊 {len(df):,} rows, {len(df.columns)} features")
                print(
                    f"    🏷️ {direction_labels} direction, {quality_labels} quality, {outcome_labels} outcome labels"
                )

            # Create training summary
            training_summary = {
                "total_timeframes": len(labeled_data),
                "total_candles": sum(len(df) for df in labeled_data.values()),
                "total_signals": sum(
                    (df["direction_label"] != 0).sum() for df in labeled_data.values()
                ),
                "total_trades": sum(
                    (df["trade_outcome"] != 0).sum() for df in labeled_data.values()
                ),
                "overall_win_rate": sum(
                    (df["trade_outcome"] == 1).sum() for df in labeled_data.values()
                )
                / max(
                    1,
                    sum(
                        (df["trade_outcome"] != 0).sum() for df in labeled_data.values()
                    ),
                )
                * 100,
                "label_types": [
                    "direction_label",
                    "signal_quality",
                    "outcome_label",
                    "risk_adjusted_label",
                ],
                "export_date": pd.Timestamp.now().isoformat(),
                "files": exported_files,
            }

            import json

            summary_file = f"{base_filename}_training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(training_summary, f, indent=2, default=str)

            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 Labeled Training Dataset Complete!")
            print(f"📊 {len(exported_files)} files | Ready for AI Training!")

            return True

        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return False


# Usage Example
if __name__ == "__main__":
    print("🎯 SMC Trading Labels Creation System")
    print("=" * 50)

    # Initialize engine
    engine = SMCLabelsEngine()

    # Load SMC features
    print("\n📂 Loading SMC Features...")
    smc_features = engine.load_smc_features("EURUSD_c")

    if smc_features:
        # Create trading labels
        labeled_data = engine.process_complete_labels_dataset(smc_features)

        # Export labeled dataset
        engine.export_labeled_dataset(labeled_data, "EURUSD_c")

        print("\n🚀 AI Training Pipeline Ready!")
        print("=" * 50)
        print("✅ 1. Data Extracted from MT5")
        print("✅ 2. SMC Features Engineered")
        print("✅ 3. Trading Labels Created")
        print("🎯 4. Next: Train AI Models")
        print("🎯 5. Next: Backtest & Optimize")

    else:
        print("❌ No SMC features loaded. Please run SMC Features Engineering first.")
