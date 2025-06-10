import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SMCLabelsEngine:
    """
    SMC Training Labels Creation Engine for AI Trading
    Created by อาจารย์ฟินิกซ์ - Professional Label Engineering
    🥇 Optimized for Gold (XAUUSD.c) Trading

    Creates high-quality training labels based on:
    - SMC Entry/Exit Logic
    - Risk Management Rules
    - Multiple Target/Stop Levels
    - Realistic Trading Scenarios
    - Gold-specific volatility considerations
    """

    def __init__(self):
        """Initialize SMC Labels Engine with Gold optimizations"""
        # 🥇 Gold-optimized Risk Management Parameters
        self.default_risk_reward = 2.5  # Better R:R for Gold volatility
        
        # 🥇 Gold-specific max holding periods (adjusted for volatility)
        self.max_holding_periods = {
            "M5": 200,   # Reduced from 288 - Gold moves faster
            "M15": 80,   # Reduced from 96 - Quick Gold reactions
            "H1": 20,    # Reduced from 24 - Gold session-based
            "H4": 10,    # Reduced from 12 - Conservative for Gold
            "D1": 5,     # Same - Daily view
        }

        # 🥇 Gold-specific Entry Signal Thresholds
        self.min_confluence = 3  # Higher confluence for Gold (vs 2)
        self.structure_break_threshold = 1.0  # Larger threshold for Gold points

        # 🥇 Gold-specific Target/Stop Calculation Methods
        self.atr_multiplier_sl = 2.0   # Larger stops for Gold volatility
        self.atr_multiplier_tp = 5.0   # Larger targets for Gold moves

        # 🥇 Gold point value
        self.gold_point_value = 0.01   # $0.01 per point

        # 🥇 Session-based risk adjustments
        self.session_risk_multipliers = {
            "london": 1.2,    # Higher risk during London (high volatility)
            "us": 1.1,        # Slightly higher during US
            "asian": 0.8,     # Lower risk during Asian
            "transition": 0.6  # Lowest risk during transitions
        }

        print("🥇 SMC Labels Engine Initialized for Gold Trading")
        print("📊 Optimized for XAUUSD.c with enhanced volatility handling")

    def load_smc_features(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load SMC features dataset"""
        timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        smc_features = {}

        print("📂 Loading Gold SMC Features Dataset...")
        print("-" * 40)

        for tf in timeframes:
            try:
                filename = f"{base_filename}_SMC_features_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                smc_features[tf] = df
                
                # 🥇 Check if Gold data
                is_gold = False
                if 'symbol' in df.columns and len(df) > 0:
                    symbol = df['symbol'].iloc[0]
                    is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()
                
                symbol_type = "🥇 Gold" if is_gold else "📈 Forex"
                print(f"✅ {tf:>3}: {len(df):,} candles with {len(df.columns)} features ({symbol_type})")
                
            except FileNotFoundError:
                print(f"❌ {tf:>3}: File not found - {filename}")
            except Exception as e:
                print(f"❌ {tf:>3}: Error loading - {str(e)}")

        print("-" * 40)
        total_candles = sum(len(df) for df in smc_features.values())
        print(f"📊 Total: {len(smc_features)} timeframes | {total_candles:,} candles")

        return smc_features

    def get_current_session(self, timestamp) -> str:
        """Determine trading session for timestamp"""
        hour = timestamp.hour
        
        if 22 <= hour or hour <= 7:
            return "asian"
        elif 8 <= hour <= 16:
            return "london"
        elif 13 <= hour <= 21:
            return "us"
        else:
            return "transition"

    def identify_entry_signals(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Identify high-probability entry signals based on SMC confluence
        🥇 Enhanced for Gold with session-based weighting
        """
        df = df.copy()

        # Initialize entry signal columns
        df["long_entry"] = 0
        df["short_entry"] = 0
        df["entry_strength"] = 0
        df["entry_reason"] = ""
        df["session_risk_multiplier"] = 1.0  # 🥇 Session-based risk

        for i in range(50, len(df)):  # Start after enough history
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            
            # 🥇 Get session for current candle
            current_session = self.get_current_session(df.index[i])
            session_multiplier = self.session_risk_multipliers.get(current_session, 1.0)
            df.iloc[i, df.columns.get_loc("session_risk_multiplier")] = session_multiplier

            # Check for Long Entry Conditions
            long_conditions = []

            # 1. Bullish Structure Break (🥇 enhanced threshold for Gold)
            if current_candle.get("structure_break", 0) > 0:
                if current_candle.get("break_strength", 0) >= self.structure_break_threshold:
                    long_conditions.append("BOS_Bull_Strong")
                else:
                    long_conditions.append("BOS_Bull")

            # 2. CHoCH to Bullish
            if current_candle.get("choch", 0) > 0:
                long_conditions.append("CHoCH_Bull")

            # 3. Bullish Order Block Test (🥇 enhanced for Gold)
            if current_candle.get("bullish_ob", 0) == 1:
                ob_strength = current_candle.get("ob_strength", 0)
                if ob_strength > 1.0:  # Strong OB for Gold
                    long_conditions.append("OB_Test_Strong")
                else:
                    long_conditions.append("OB_Test")
            elif (not pd.isna(current_candle.get("ob_low", np.nan)) and
                  current_candle["low"] <= current_candle.get("ob_low", float('inf')) and
                  current_candle["close"] > current_candle.get("ob_low", 0)):
                long_conditions.append("OB_Retest")

            # 4. Fair Value Gap Fill (Bullish) (🥇 enhanced for Gold)
            if current_candle.get("bullish_fvg", 0) == 1:
                fvg_efficiency = current_candle.get("fvg_efficiency", 0)
                if fvg_efficiency > 0.5:  # Efficient FVG for Gold
                    long_conditions.append("FVG_Fill_Efficient")
                else:
                    long_conditions.append("FVG_Fill")
            elif (not pd.isna(current_candle.get("fvg_low", np.nan)) and
                  current_candle["low"] <= current_candle.get("fvg_low", float('inf')) and
                  current_candle["close"] > current_candle.get("fvg_low", 0)):
                long_conditions.append("FVG_Retest")

            # 5. Market Structure is Bullish (🥇 with structure strength)
            if current_candle.get("market_structure", 0) > 0:
                structure_strength = current_candle.get("structure_strength", 0)
                if structure_strength > 1.0:
                    long_conditions.append("Bull_Structure_Strong")
                else:
                    long_conditions.append("Bull_Structure")

            # 6. Liquidity Sweep (Buy Liquidity) (🥇 with proximity check)
            if current_candle.get("buy_liquidity", 0) == 1:
                liquidity_proximity = current_candle.get("liquidity_proximity", 1.0)
                if liquidity_proximity < 0.001:  # Very close to liquidity
                    long_conditions.append("Liquidity_Sweep_Precise")
                else:
                    long_conditions.append("Liquidity_Sweep")

            # 🥇 Session bonus for Gold
            if current_session in ["london", "us"]:
                if current_candle.get("is_high_impact_hour", 0) == 1:
                    long_conditions.append("High_Impact_Session")

            # Check confluence for Long Entry (🥇 adjusted threshold)
            adjusted_min_confluence = max(self.min_confluence, 
                                        int(self.min_confluence * session_multiplier))
            
            if len(long_conditions) >= adjusted_min_confluence:
                df.iloc[i, df.columns.get_loc("long_entry")] = 1
                df.iloc[i, df.columns.get_loc("entry_strength")] = len(long_conditions)
                df.iloc[i, df.columns.get_loc("entry_reason")] = "+".join(long_conditions)

            # Check for Short Entry Conditions
            short_conditions = []

            # 1. Bearish Structure Break (🥇 enhanced threshold for Gold)
            if current_candle.get("structure_break", 0) < 0:
                if abs(current_candle.get("break_strength", 0)) >= self.structure_break_threshold:
                    short_conditions.append("BOS_Bear_Strong")
                else:
                    short_conditions.append("BOS_Bear")

            # 2. CHoCH to Bearish
            if current_candle.get("choch", 0) < 0:
                short_conditions.append("CHoCH_Bear")

            # 3. Bearish Order Block Test (🥇 enhanced for Gold)
            if current_candle.get("bearish_ob", 0) == 1:
                ob_strength = current_candle.get("ob_strength", 0)
                if ob_strength > 1.0:
                    short_conditions.append("OB_Test_Strong")
                else:
                    short_conditions.append("OB_Test")
            elif (not pd.isna(current_candle.get("ob_high", np.nan)) and
                  current_candle["high"] >= current_candle.get("ob_high", 0) and
                  current_candle["close"] < current_candle.get("ob_high", float('inf'))):
                short_conditions.append("OB_Retest")

            # 4. Fair Value Gap Fill (Bearish) (🥇 enhanced for Gold)
            if current_candle.get("bearish_fvg", 0) == 1:
                fvg_efficiency = current_candle.get("fvg_efficiency", 0)
                if fvg_efficiency > 0.5:
                    short_conditions.append("FVG_Fill_Efficient")
                else:
                    short_conditions.append("FVG_Fill")
            elif (not pd.isna(current_candle.get("fvg_high", np.nan)) and
                  current_candle["high"] >= current_candle.get("fvg_high", 0) and
                  current_candle["close"] < current_candle.get("fvg_high", float('inf'))):
                short_conditions.append("FVG_Retest")

            # 5. Market Structure is Bearish (🥇 with structure strength)
            if current_candle.get("market_structure", 0) < 0:
                structure_strength = abs(current_candle.get("structure_strength", 0))
                if structure_strength > 1.0:
                    short_conditions.append("Bear_Structure_Strong")
                else:
                    short_conditions.append("Bear_Structure")

            # 6. Liquidity Sweep (Sell Liquidity) (🥇 with proximity check)
            if current_candle.get("sell_liquidity", 0) == 1:
                liquidity_proximity = current_candle.get("liquidity_proximity", 1.0)
                if liquidity_proximity < 0.001:
                    short_conditions.append("Liquidity_Sweep_Precise")
                else:
                    short_conditions.append("Liquidity_Sweep")

            # 🥇 Session bonus for Gold
            if current_session in ["london", "us"]:
                if current_candle.get("is_high_impact_hour", 0) == 1:
                    short_conditions.append("High_Impact_Session")

            # Check confluence for Short Entry (🥇 adjusted threshold)
            if len(short_conditions) >= adjusted_min_confluence:
                df.iloc[i, df.columns.get_loc("short_entry")] = 1
                df.iloc[i, df.columns.get_loc("entry_strength")] = len(short_conditions)
                df.iloc[i, df.columns.get_loc("entry_reason")] = "+".join(short_conditions)

        return df

    def calculate_stop_take_levels(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate realistic Stop Loss and Take Profit levels using SMC logic
        🥇 Enhanced for Gold with point-based calculations
        """
        df = df.copy()

        # Initialize stop/take columns
        df["stop_loss"] = np.nan
        df["take_profit_1"] = np.nan
        df["take_profit_2"] = np.nan
        df["risk_reward"] = np.nan
        df["stop_loss_points"] = np.nan  # 🥇 Gold points
        df["take_profit_points"] = np.nan  # 🥇 Gold points

        # 🥇 Calculate ATR for dynamic stop/take levels (Gold-adjusted)
        if "atr_14" not in df.columns:
            df["atr_14"] = df["range"].rolling(14).mean()

        # 🥇 Use Gold ATR in points if available
        if "atr_points" in df.columns:
            atr_column = "atr_points"
            point_multiplier = 1.0
        else:
            atr_column = "atr_14"
            point_multiplier = 1.0 / self.gold_point_value  # Convert to points

        for i in range(len(df)):
            current_price = df["close"].iloc[i]
            atr = df[atr_column].iloc[i]
            
            # 🥇 Session-based risk adjustment
            session_multiplier = df.get("session_risk_multiplier", pd.Series([1.0] * len(df))).iloc[i]

            if pd.isna(atr) or atr == 0:
                continue

            # Long Entry Stop/Take Calculation
            if df["long_entry"].iloc[i] == 1:
                # 🥇 Stop Loss: Below recent swing low or ATR-based (Gold-adjusted)
                recent_lows = df.iloc[max(0, i - 30) : i]["low"]  # Extended lookback for Gold
                if len(recent_lows) > 0:
                    swing_low = recent_lows.min()
                    atr_stop = current_price - (atr * self.atr_multiplier_sl * session_multiplier)
                    stop_loss = min(swing_low, atr_stop)
                else:
                    stop_loss = current_price - (atr * self.atr_multiplier_sl * session_multiplier)

                # 🥇 Take Profit levels (Gold-optimized)
                risk = current_price - stop_loss
                base_tp_ratio = self.default_risk_reward * session_multiplier
                
                take_profit_1 = current_price + (risk * base_tp_ratio)
                take_profit_2 = current_price + (risk * base_tp_ratio * 2)

                # 🥇 Calculate in Gold points
                stop_loss_points = (current_price - stop_loss) / self.gold_point_value
                take_profit_points = (take_profit_1 - current_price) / self.gold_point_value

                df.iloc[i, df.columns.get_loc("stop_loss")] = stop_loss
                df.iloc[i, df.columns.get_loc("take_profit_1")] = take_profit_1
                df.iloc[i, df.columns.get_loc("take_profit_2")] = take_profit_2
                df.iloc[i, df.columns.get_loc("stop_loss_points")] = stop_loss_points
                df.iloc[i, df.columns.get_loc("take_profit_points")] = take_profit_points
                df.iloc[i, df.columns.get_loc("risk_reward")] = base_tp_ratio

            # Short Entry Stop/Take Calculation
            if df["short_entry"].iloc[i] == 1:
                # 🥇 Stop Loss: Above recent swing high or ATR-based (Gold-adjusted)
                recent_highs = df.iloc[max(0, i - 30) : i]["high"]  # Extended lookback for Gold
                if len(recent_highs) > 0:
                    swing_high = recent_highs.max()
                    atr_stop = current_price + (atr * self.atr_multiplier_sl * session_multiplier)
                    stop_loss = max(swing_high, atr_stop)
                else:
                    stop_loss = current_price + (atr * self.atr_multiplier_sl * session_multiplier)

                # 🥇 Take Profit levels (Gold-optimized)
                risk = stop_loss - current_price
                base_tp_ratio = self.default_risk_reward * session_multiplier
                
                take_profit_1 = current_price - (risk * base_tp_ratio)
                take_profit_2 = current_price - (risk * base_tp_ratio * 2)

                # 🥇 Calculate in Gold points
                stop_loss_points = (stop_loss - current_price) / self.gold_point_value
                take_profit_points = (current_price - take_profit_1) / self.gold_point_value

                df.iloc[i, df.columns.get_loc("stop_loss")] = stop_loss
                df.iloc[i, df.columns.get_loc("take_profit_1")] = take_profit_1
                df.iloc[i, df.columns.get_loc("take_profit_2")] = take_profit_2
                df.iloc[i, df.columns.get_loc("stop_loss_points")] = stop_loss_points
                df.iloc[i, df.columns.get_loc("take_profit_points")] = take_profit_points
                df.iloc[i, df.columns.get_loc("risk_reward")] = base_tp_ratio

        return df

    def simulate_trade_outcomes(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Simulate realistic trade outcomes based on actual price movement
        🥇 Enhanced for Gold with session-based adjustments
        """
        df = df.copy()

        # Initialize outcome columns
        df["trade_outcome"] = 0  # 1=Win, -1=Loss, 0=No Trade
        df["exit_price"] = np.nan
        df["exit_reason"] = ""
        df["holding_periods"] = 0
        df["pnl_points"] = 0  # 🥇 Gold points P&L
        df["pnl_percent"] = 0
        df["max_favorable_excursion"] = 0  # 🥇 MFE in points
        df["max_adverse_excursion"] = 0    # 🥇 MAE in points

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
            
            # 🥇 Session-based max holding adjustment
            session_multiplier = entry_row.get("session_risk_multiplier", 1.0)
            adjusted_max_holding = int(max_holding * session_multiplier)

            # Look forward to simulate trade
            entry_pos = df.index.get_loc(entry_idx)
            max_look_ahead = min(entry_pos + adjusted_max_holding, len(df))

            trade_closed = False
            max_favorable = 0  # Track MFE
            max_adverse = 0    # Track MAE

            for future_pos in range(entry_pos + 1, max_look_ahead):
                future_candle = df.iloc[future_pos]
                high_price = future_candle["high"]
                low_price = future_candle["low"]
                close_price = future_candle["close"]

                holding_periods = future_pos - entry_pos

                if is_long:
                    # Track MFE/MAE for long position
                    favorable_move = (high_price - entry_price) / self.gold_point_value
                    adverse_move = (entry_price - low_price) / self.gold_point_value
                    
                    max_favorable = max(max_favorable, favorable_move)
                    max_adverse = max(max_adverse, adverse_move)

                    # Check Long Trade
                    if low_price <= stop_loss:
                        # Stop Loss Hit
                        df.loc[entry_idx, "trade_outcome"] = -1
                        df.loc[entry_idx, "exit_price"] = stop_loss
                        df.loc[entry_idx, "exit_reason"] = "Stop_Loss"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        
                        # 🥇 Calculate P&L in Gold points
                        pnl_points = (stop_loss - entry_price) / self.gold_point_value
                        df.loc[entry_idx, "pnl_points"] = pnl_points
                        df.loc[entry_idx, "pnl_percent"] = ((stop_loss / entry_price) - 1) * 100
                        
                        trade_closed = True
                        break
                    elif high_price >= take_profit_1:
                        # Take Profit Hit
                        df.loc[entry_idx, "trade_outcome"] = 1
                        df.loc[entry_idx, "exit_price"] = take_profit_1
                        df.loc[entry_idx, "exit_reason"] = "Take_Profit"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        
                        # 🥇 Calculate P&L in Gold points
                        pnl_points = (take_profit_1 - entry_price) / self.gold_point_value
                        df.loc[entry_idx, "pnl_points"] = pnl_points
                        df.loc[entry_idx, "pnl_percent"] = ((take_profit_1 / entry_price) - 1) * 100
                        
                        trade_closed = True
                        break

                else:  # Short position
                    # Track MFE/MAE for short position
                    favorable_move = (entry_price - low_price) / self.gold_point_value
                    adverse_move = (high_price - entry_price) / self.gold_point_value
                    
                    max_favorable = max(max_favorable, favorable_move)
                    max_adverse = max(max_adverse, adverse_move)

                    # Check Short Trade
                    if high_price >= stop_loss:
                        # Stop Loss Hit
                        df.loc[entry_idx, "trade_outcome"] = -1
                        df.loc[entry_idx, "exit_price"] = stop_loss
                        df.loc[entry_idx, "exit_reason"] = "Stop_Loss"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        
                        # 🥇 Calculate P&L in Gold points
                        pnl_points = (entry_price - stop_loss) / self.gold_point_value
                        df.loc[entry_idx, "pnl_points"] = pnl_points
                        df.loc[entry_idx, "pnl_percent"] = ((entry_price / stop_loss) - 1) * 100
                        
                        trade_closed = True
                        break
                    elif low_price <= take_profit_1:
                        # Take Profit Hit
                        df.loc[entry_idx, "trade_outcome"] = 1
                        df.loc[entry_idx, "exit_price"] = take_profit_1
                        df.loc[entry_idx, "exit_reason"] = "Take_Profit"
                        df.loc[entry_idx, "holding_periods"] = holding_periods
                        
                        # 🥇 Calculate P&L in Gold points
                        pnl_points = (entry_price - take_profit_1) / self.gold_point_value
                        df.loc[entry_idx, "pnl_points"] = pnl_points
                        df.loc[entry_idx, "pnl_percent"] = ((entry_price / take_profit_1) - 1) * 100
                        
                        trade_closed = True
                        break

            # Store MFE/MAE
            df.loc[entry_idx, "max_favorable_excursion"] = max_favorable
            df.loc[entry_idx, "max_adverse_excursion"] = max_adverse

            # If trade not closed by Stop/Take, close at max holding period
            if not trade_closed and max_look_ahead > entry_pos + 1:
                final_candle = df.iloc[max_look_ahead - 1]
                exit_price = final_candle["close"]

                if is_long:
                    outcome = 1 if exit_price > entry_price else -1
                    pnl_points = (exit_price - entry_price) / self.gold_point_value
                    pnl_percent = ((exit_price / entry_price) - 1) * 100
                else:
                    outcome = 1 if exit_price < entry_price else -1
                    pnl_points = (entry_price - exit_price) / self.gold_point_value
                    pnl_percent = ((entry_price / exit_price) - 1) * 100

                df.loc[entry_idx, "trade_outcome"] = outcome
                df.loc[entry_idx, "exit_price"] = exit_price
                df.loc[entry_idx, "exit_reason"] = "Time_Exit"
                df.loc[entry_idx, "holding_periods"] = adjusted_max_holding
                df.loc[entry_idx, "pnl_points"] = pnl_points
                df.loc[entry_idx, "pnl_percent"] = pnl_percent

        return df

    def create_classification_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create multiple types of classification labels for different AI models
        🥇 Enhanced for Gold with point-based thresholds
        """
        df = df.copy()

        # 1. Simple Direction Labels (3-class)
        df["direction_label"] = 0  # 0=Hold, 1=Long, -1=Short
        df.loc[df["long_entry"] == 1, "direction_label"] = 1
        df.loc[df["short_entry"] == 1, "direction_label"] = -1

        # 2. Signal Quality Labels (4-class) - 🥇 adjusted for Gold
        df["signal_quality"] = 0  # 0=No Signal, 1=Weak, 2=Medium, 3=Strong
        df.loc[df["entry_strength"] == 3, "signal_quality"] = 1  # Increased from 2
        df.loc[df["entry_strength"] == 4, "signal_quality"] = 2  # Increased from 3
        df.loc[df["entry_strength"] >= 5, "signal_quality"] = 3  # Increased from 4

        # 3. Trade Outcome Labels (for supervised learning)
        df["outcome_label"] = 0  # 0=No Trade, 1=Profitable, -1=Loss
        df.loc[df["trade_outcome"] == 1, "outcome_label"] = 1
        df.loc[df["trade_outcome"] == -1, "outcome_label"] = -1

        # 4. 🥇 Gold-specific Risk-Adjusted Labels (using points)
        df["risk_adjusted_label"] = 0
        
        # Only mark as positive if trade is profitable AND meets Gold point threshold
        profitable_gold_trades = (df["trade_outcome"] == 1) & (df["pnl_points"] > 20)  # 20 points profit
        df.loc[profitable_gold_trades, "risk_adjusted_label"] = 1

        # Mark as negative if loss exceeds Gold point threshold
        significant_gold_losses = (df["trade_outcome"] == -1) & (df["pnl_points"] < -30)  # 30 points loss
        df.loc[significant_gold_losses, "risk_adjusted_label"] = -1

        # 5. 🥇 Session-based Labels
        df["session_label"] = 0
        # Mark high-quality signals during good sessions
        good_session_signals = (
            (df["direction_label"] != 0) & 
            (df["session_risk_multiplier"] >= 1.1) &  # London/US sessions
            (df["signal_quality"] >= 2)
        )
        df.loc[good_session_signals, "session_label"] = 1

        return df

    def process_single_timeframe_labels(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Complete label creation process for single timeframe
        🥇 Enhanced for Gold with detailed progress tracking
        """
        print(f"🎯 Creating Gold labels for {timeframe}...")

        # Check if Gold data
        is_gold = False
        if 'symbol' in df.columns and len(df) > 0:
            symbol = df['symbol'].iloc[0]
            is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()

        if is_gold:
            print(f"   🥇 Gold-optimized label creation for {timeframe}")

        # Step 1: Identify entry signals
        df = self.identify_entry_signals(df, timeframe)
        long_entries = df["long_entry"].sum()
        short_entries = df["short_entry"].sum()
        print(f"   📊 Entry Signals: {long_entries} long, {short_entries} short")

        # Step 2: Calculate stop/take levels
        df = self.calculate_stop_take_levels(df, timeframe)
        valid_levels = (~pd.isna(df["stop_loss"])).sum()
        print(f"   🎯 Valid Stop/Take Levels: {valid_levels}")

        # 🥇 Show Gold-specific statistics
        if is_gold and valid_levels > 0:
            avg_sl_points = df["stop_loss_points"].mean() if "stop_loss_points" in df.columns else 0
            avg_tp_points = df["take_profit_points"].mean() if "take_profit_points" in df.columns else 0
            print(f"   🥇 Avg SL: {avg_sl_points:.0f} points, Avg TP: {avg_tp_points:.0f} points")

        # Step 3: Simulate trade outcomes
        df = self.simulate_trade_outcomes(df, timeframe)
        total_trades = (df["trade_outcome"] != 0).sum()
        winning_trades = (df["trade_outcome"] == 1).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 🥇 Gold-specific performance metrics
        if is_gold and total_trades > 0:
            avg_win_points = df[df["trade_outcome"] == 1]["pnl_points"].mean() if winning_trades > 0 else 0
            avg_loss_points = df[df["trade_outcome"] == -1]["pnl_points"].mean() if (total_trades - winning_trades) > 0 else 0
            print(f"   💹 Gold Performance: {total_trades} trades, {win_rate:.1f}% win rate")
            print(f"   🥇 Avg Win: {avg_win_points:.0f} points, Avg Loss: {avg_loss_points:.0f} points")
        else:
            print(f"   💹 Trade Simulation: {total_trades} trades, {win_rate:.1f}% win rate")

        # Step 4: Create classification labels
        df = self.create_classification_labels(df)
        direction_signals = (df["direction_label"] != 0).sum()
        quality_signals = (df["signal_quality"] > 0).sum()
        session_signals = (df["session_label"] == 1).sum() if "session_label" in df.columns else 0
        
        print(f"   🏷️ Labels Created: {direction_signals} direction, {quality_signals} quality")
        if is_gold:
            print(f"   🥇 Session Labels: {session_signals} high-quality session signals")

        print(f"✅ {timeframe} Gold label creation complete!")

        return df

    def process_complete_labels_dataset(self, smc_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process complete dataset with trading labels
        🥇 Enhanced for Gold with comprehensive reporting
        """
        print("🎯 Creating Gold SMC Trading Labels Dataset")
        print("=" * 50)

        labeled_data = {}

        for timeframe, df in smc_features.items():
            print(f"\n📊 Processing {timeframe} ({len(df):,} candles)...")
            labeled_df = self.process_single_timeframe_labels(df, timeframe)
            labeled_data[timeframe] = labeled_df

        print("\n" + "=" * 50)
        print("📋 GOLD TRADING LABELS SUMMARY")
        print("=" * 50)

        # Summary statistics
        total_signals = 0
        total_trades = 0
        total_winners = 0
        total_points_profit = 0

        for timeframe, df in labeled_data.items():
            signals = (df["direction_label"] != 0).sum()
            trades = (df["trade_outcome"] != 0).sum()
            winners = (df["trade_outcome"] == 1).sum()
            win_rate = (winners / trades * 100) if trades > 0 else 0
            
            # 🥇 Gold points profit calculation
            points_profit = df["pnl_points"].sum() if "pnl_points" in df.columns else 0

            total_signals += signals
            total_trades += trades
            total_winners += winners
            total_points_profit += points_profit

            print(f"{timeframe:>3}: {signals:>4} signals | {trades:>4} trades | {win_rate:>5.1f}% win rate | {points_profit:>6.0f} points")

        overall_win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
        print("-" * 50)
        print(f"📊 Total: {total_signals} signals | {total_trades} trades | {overall_win_rate:.1f}% overall win rate")
        print(f"🥇 Total Gold Profit: {total_points_profit:.0f} points (${total_points_profit * self.gold_point_value:.2f})")
        print("🎉 Gold SMC Trading Labels Complete!")

        return labeled_data

    def export_labeled_dataset(self, labeled_data: Dict[str, pd.DataFrame], base_filename: str) -> bool:
        """
        Export complete labeled dataset ready for AI training
        🥇 Enhanced for Gold with detailed documentation
        """
        try:
            print(f"\n💾 Exporting Gold Labeled Training Dataset...")
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
                session_labels = (df["session_label"] == 1).sum() if "session_label" in df.columns else 0

                print(f"✅ {timeframe}: {filename}")
                print(f"    📊 {len(df):,} rows, {len(df.columns)} features")
                print(f"    🏷️ {direction_labels} direction, {quality_labels} quality, {outcome_labels} outcome")
                if session_labels > 0:
                    print(f"    🥇 {session_labels} session-based labels")

            # 🥇 Create enhanced training summary for Gold
            # Check if Gold dataset
            is_gold_dataset = False
            if labeled_data:
                first_df = list(labeled_data.values())[0]
                if 'symbol' in first_df.columns and len(first_df) > 0:
                    symbol = first_df['symbol'].iloc[0]
                    is_gold_dataset = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()

            training_summary = {
                "dataset_type": "Gold SMC Trading Labels" if is_gold_dataset else "Forex SMC Trading Labels",
                "total_timeframes": len(labeled_data),
                "total_candles": sum(len(df) for df in labeled_data.values()),
                "total_signals": sum((df["direction_label"] != 0).sum() for df in labeled_data.values()),
                "total_trades": sum((df["trade_outcome"] != 0).sum() for df in labeled_data.values()),
                "overall_win_rate": sum((df["trade_outcome"] == 1).sum() for df in labeled_data.values()) / 
                                  max(1, sum((df["trade_outcome"] != 0).sum() for df in labeled_data.values())) * 100,
                "label_types": ["direction_label", "signal_quality", "outcome_label", "risk_adjusted_label"],
                "export_date": pd.Timestamp.now().isoformat(),
                "files": exported_files,
            }

            # 🥇 Add Gold-specific summary data
            if is_gold_dataset:
                total_points_profit = sum(df["pnl_points"].sum() if "pnl_points" in df.columns else 0 
                                        for df in labeled_data.values())
                total_dollar_profit = total_points_profit * self.gold_point_value
                
                training_summary.update({
                    "gold_specific_features": [
                        "stop_loss_points", "take_profit_points", "pnl_points",
                        "max_favorable_excursion", "max_adverse_excursion",
                        "session_risk_multiplier", "session_label"
                    ],
                    "gold_performance": {
                        "total_points_profit": float(total_points_profit),
                        "total_dollar_profit": float(total_dollar_profit),
                        "point_value": self.gold_point_value,
                        "avg_risk_reward": self.default_risk_reward
                    },
                    "gold_parameters": {
                        "min_confluence": self.min_confluence,
                        "atr_multiplier_sl": self.atr_multiplier_sl,
                        "atr_multiplier_tp": self.atr_multiplier_tp,
                        "max_holding_periods": self.max_holding_periods,
                        "session_risk_multipliers": self.session_risk_multipliers
                    },
                    "optimization_notes": "Optimized for XAUUSD.c with enhanced volatility handling and session-based risk management"
                })

                training_summary["label_types"].append("session_label")

            import json
            summary_file = f"{base_filename}_training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(training_summary, f, indent=2, default=str)

            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 {'Gold' if is_gold_dataset else 'Forex'} Labeled Training Dataset Complete!")
            print(f"📊 {len(exported_files)} files | Ready for AI Training!")
            
            if is_gold_dataset:
                print("🥇 Gold-specific optimizations included")
                print(f"💰 Projected profit: {total_points_profit:.0f} points (${total_dollar_profit:.2f})")

            return True

        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return False


# Usage Example for Gold
if __name__ == "__main__":
    print("🥇 SMC Trading Labels Creation System for Gold")
    print("=" * 50)

    # Initialize engine with Gold optimizations
    engine = SMCLabelsEngine()

    # Load Gold SMC features
    print("\n📂 Loading Gold SMC Features...")
    smc_features = engine.load_smc_features("XAUUSD_c")

    if smc_features:
        # Create Gold trading labels
        labeled_data = engine.process_complete_labels_dataset(smc_features)

        # Export Gold labeled dataset
        engine.export_labeled_dataset(labeled_data, "XAUUSD_c")

        print("\n🚀 Gold AI Training Pipeline Ready!")
        print("=" * 50)
        print("✅ 1. Gold Data Extracted from MT5")
        print("✅ 2. Gold SMC Features Engineered")
        print("✅ 3. Gold Trading Labels Created")
        print("🎯 4. Next: Train Gold AI Models")
        print("🎯 5. Next: Backtest & Optimize Gold Strategy")

    else:
        print("❌ No Gold SMC features loaded. Please run SMC Features Engineering first.")
        print("🔧 Expected files: XAUUSD_c_SMC_features_[M5|M15|H1|H4|D1].csv")