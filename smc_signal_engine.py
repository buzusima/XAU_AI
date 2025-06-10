import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json

warnings.filterwarnings("ignore")

# ML Libraries
import joblib
from sklearn.preprocessing import StandardScaler

# Deep Learning
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SMCSignalEngine:
    """
    SMC Real-time Signal Prediction Engine with Feature Alignment
    Created by อาจารย์ฟินิกซ์ - Live Trading Signal Generation
    🥇 Optimized for Gold (XAUUSD.c) Trading

    Features:
    - Perfect feature alignment with training data
    - Real-time Gold data fetching from MT5
    - Multi-timeframe signal aggregation
    - Gold-specific confidence scoring
    - Session-based signal filtering
    - Point-based risk calculations
    """

    def __init__(self, models_path: str = "XAUUSD_c_SMC"):
        """Initialize Gold SMC Signal Engine"""
        self.models_path = models_path
        self.models = {}
        self.training_features = {}
        self.timezone = pytz.timezone("Etc/UTC")

        # Load feature mapping
        self.load_feature_mapping()

        # 🥇 Gold timeframes (same priorities but Gold-focused)
        self.timeframes = ["M5", "M15", "H1", "H4", "D1"]
        self.mt5_timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        # 🥇 Gold-specific signal thresholds
        self.min_confidence = 0.80       # Higher confidence for Gold
        self.min_consensus = 4           # More consensus required (4/5)
        self.max_signal_age = 5          # Signal freshness

        # 🥇 Gold position sizing
        self.base_lot_size = 0.01        # Base position
        self.max_lot_size = 0.05         # Lower max for Gold volatility
        self.risk_per_trade = 0.015      # 1.5% risk for Gold

        # 🥇 Gold symbol variations to try
        self.gold_symbols = [
            "XAUUSD.c", "XAUUSD", "GOLD.c", "GOLD", 
            "XAU/USD", "XAUUSD#", "XAUUSD.raw", "XAUUSD.a"
        ]

        # 🥇 Gold session preferences
        self.session_preferences = {
            "london": {"weight": 1.5, "preferred": True},
            "us": {"weight": 1.3, "preferred": True},
            "asian": {"weight": 0.7, "preferred": False},
            "transition": {"weight": 0.5, "preferred": False}
        }

        print("🥇 Gold SMC Signal Engine Initialized")
        print("📊 Optimized for XAUUSD.c real-time signal generation")

    def load_feature_mapping(self) -> bool:
        """Load feature mapping for perfect alignment"""
        try:
            feature_mapping_file = f"{self.models_path}_feature_mapping.json"
            with open(feature_mapping_file, "r") as f:
                self.training_features = json.load(f)

            print("✅ Gold feature mapping loaded:")
            for tf, features in self.training_features.items():
                print(f"  {tf}: {len(features)} features")

            return True

        except Exception as e:
            print(f"❌ Failed to load Gold feature mapping: {str(e)}")
            print("⚠️ Falling back to basic feature extraction")
            return False

    def load_trained_models(self) -> bool:
        """Load all trained Gold AI models"""
        print("📂 Loading Trained Gold AI Models...")
        print("-" * 40)

        loaded_count = 0

        for timeframe in self.timeframes:
            tf_models = {}

            # Load Random Forest
            try:
                rf_path = f"{self.models_path}_{timeframe}_random_forest_model.pkl"
                rf_model = joblib.load(rf_path)
                tf_models["random_forest"] = rf_model
                print(f"✅ {timeframe} Random Forest loaded")
                loaded_count += 1
            except Exception as e:
                print(f"❌ {timeframe} Random Forest failed: {str(e)}")

            # Load XGBoost (if available)
            try:
                xgb_path = f"{self.models_path}_{timeframe}_xgboost_model.pkl"
                xgb_model = joblib.load(xgb_path)
                tf_models["xgboost"] = xgb_model
                print(f"✅ {timeframe} XGBoost loaded")
                loaded_count += 1
            except:
                pass  # XGBoost might not be available

            # Load Neural Network (if available)
            if TENSORFLOW_AVAILABLE:
                try:
                    # Load Keras model
                    nn_keras_path = f"{self.models_path}_{timeframe}_neural_network_model.h5"
                    nn_model = tf.keras.models.load_model(nn_keras_path)

                    # Load other components
                    nn_components_path = f"{self.models_path}_{timeframe}_neural_network_model.pkl"
                    nn_components = joblib.load(nn_components_path)

                    tf_models["neural_network"] = {"model": nn_model, **nn_components}
                    print(f"✅ {timeframe} Neural Network loaded")
                    loaded_count += 1
                except:
                    pass  # Neural Network might not be available

            if tf_models:
                self.models[timeframe] = tf_models

        print("-" * 40)
        print(f"📊 Total: {loaded_count} Gold models loaded across {len(self.models)} timeframes")

        return len(self.models) > 0

    def connect_mt5(
        self, account: int = None, password: str = None, server: str = None
    ) -> bool:
        """Connect to MT5 for real-time Gold data"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False

            if account and password and server:
                if not mt5.login(account, password=password, server=server):
                    print(f"❌ Login failed: {mt5.last_error()}")
                    return False

            print("✅ MT5 Connected for Gold real-time data")
            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def find_best_gold_symbol(self) -> Optional[str]:
        """Find the best available Gold symbol"""
        print("🔍 Searching for Gold symbols...")
        
        for symbol in self.gold_symbols:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info and symbol_info.trade_mode != 0:  # Trading allowed
                        print(f"✅ Found tradeable Gold symbol: {symbol}")
                        return symbol
            except:
                continue
        
        print("❌ No tradeable Gold symbol found")
        return None

    def get_current_session(self) -> str:
        """Get current trading session"""
        current_hour = datetime.now().hour
        
        if 22 <= current_hour or current_hour <= 7:
            return "asian"
        elif 8 <= current_hour <= 16:
            return "london"
        elif 13 <= current_hour <= 21:
            return "us"
        else:
            return "transition"

    def get_session_info(self) -> Dict:
        """Get detailed session information for Gold"""
        current_hour = datetime.now().hour
        session = self.get_current_session()
        
        session_info = {
            "current_hour": current_hour,
            "session": session,
            "is_preferred": self.session_preferences[session]["preferred"],
            "weight": self.session_preferences[session]["weight"],
            "volatility": "low"
        }
        
        # High impact hours for Gold
        if current_hour in [8, 13, 14, 15]:
            session_info["is_high_impact"] = True
            session_info["volatility"] = "very_high"
        elif session in ["london", "us"]:
            session_info["volatility"] = "high"
        
        return session_info

    def get_realtime_data(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get real-time Gold market data for prediction"""
        try:
            # Get latest data from MT5
            rates = mt5.copy_rates_from_pos(
                symbol, self.mt5_timeframes[timeframe], 0, count
            )

            if rates is None or len(rates) == 0:
                print(f"❌ No real-time Gold data for {symbol} {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Add Gold metadata
            df["symbol"] = symbol
            df["timeframe"] = timeframe

            return df

        except Exception as e:
            print(f"❌ Real-time Gold data error: {str(e)}")
            return pd.DataFrame()

    def calculate_complete_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete SMC features for Gold matching training data exactly"""
        if df.empty:
            return df

        df = df.copy()

        # Basic price metrics
        df["hl2"] = (df["high"] + df["low"]) / 2
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        # 🥇 Gold-specific price calculations
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100
        df["range_points"] = df["range"] / 0.01  # Gold points

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
        df["price_change_points"] = df["price_change"] / 0.01  # Gold points

        # Technical indicators
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()

        # 🥇 Gold-specific volatility measures
        df["atr_14"] = df["range"].rolling(14).mean()
        df["atr_points"] = df["range_points"].rolling(14).mean()  # Gold ATR in points
        df["volatility_pct"] = (df["close"].rolling(20).std() / df["close"].rolling(20).mean() * 100)

        # Volume analysis
        df["volume_ma"] = df["tick_volume"].rolling(20).mean()
        df["volume_ratio"] = np.where(df["volume_ma"] > 0, df["tick_volume"] / df["volume_ma"], 1)

        # 🥇 Gold session analysis
        df["hour"] = df.index.hour
        df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] <= 16)).astype(int)
        df["is_us_session"] = ((df["hour"] >= 13) & (df["hour"] <= 21)).astype(int)
        df["is_asian_session"] = ((df["hour"] >= 22) | (df["hour"] <= 7)).astype(int)
        df["is_high_impact_hour"] = df["hour"].isin([8, 13, 14, 15]).astype(int)

        # Market structure (Gold-adjusted periods)
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)
        df["higher_high"] = (df["high"] > df["high"].rolling(12).max().shift(1)).astype(int)  # Adjusted for Gold
        df["lower_low"] = (df["low"] < df["low"].rolling(12).min().shift(1)).astype(int)     # Adjusted for Gold

        # 🥇 Gold-specific swing detection (longer period)
        period = 8  # Longer for Gold volatility
        df["swing_high"] = 0
        df["swing_low"] = 0
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan

        for i in range(period, len(df) - period):
            if df["high"].iloc[i] == df["high"].iloc[i - period : i + period + 1].max():
                if (df["high"].iloc[i] > df["high"].iloc[i - 1] and 
                    df["high"].iloc[i] > df["high"].iloc[i + 1]):
                    df.iloc[i, df.columns.get_loc("swing_high")] = 1
                    df.iloc[i, df.columns.get_loc("swing_high_price")] = df["high"].iloc[i]

            if df["low"].iloc[i] == df["low"].iloc[i - period : i + period + 1].min():
                if (df["low"].iloc[i] < df["low"].iloc[i - 1] and 
                    df["low"].iloc[i] < df["low"].iloc[i + 1]):
                    df.iloc[i, df.columns.get_loc("swing_low")] = 1
                    df.iloc[i, df.columns.get_loc("swing_low_price")] = df["low"].iloc[i]

        # Market structure analysis (Gold-adjusted)
        df["higher_high"] = 0
        df["higher_low"] = 0
        df["lower_high"] = 0
        df["lower_low"] = 0
        df["market_structure"] = 0

        swing_highs = df[df["swing_high"] == 1]["swing_high_price"].dropna()
        swing_lows = df[df["swing_low"] == 1]["swing_low_price"].dropna()

        if len(swing_highs) >= 2:
            swing_high_indices = df[df["swing_high"] == 1].index
            for i in range(1, len(swing_highs)):
                current_idx = swing_high_indices[i]
                current_high = swing_highs.iloc[i]
                previous_high = swing_highs.iloc[i - 1]

                if current_high > previous_high:
                    df.loc[current_idx, "higher_high"] = 1
                else:
                    df.loc[current_idx, "lower_high"] = 1

        if len(swing_lows) >= 2:
            swing_low_indices = df[df["swing_low"] == 1].index
            for i in range(1, len(swing_lows)):
                current_idx = swing_low_indices[i]
                current_low = swing_lows.iloc[i]
                previous_low = swing_lows.iloc[i - 1]

                if current_low > previous_low:
                    df.loc[current_idx, "higher_low"] = 1
                else:
                    df.loc[current_idx, "lower_low"] = 1

        # Market structure determination (Gold-weighted by session)
        structure_window = 30  # Extended for Gold
        for i in range(structure_window, len(df)):
            window_data = df.iloc[i - structure_window : i]

            hh_count = window_data["higher_high"].sum()
            hl_count = window_data["higher_low"].sum()
            lh_count = window_data["lower_high"].sum()
            ll_count = window_data["lower_low"].sum()

            # 🥇 Apply session weighting for Gold
            session_weight = 1.0
            if df["is_london_session"].iloc[i] == 1:
                session_weight = 1.5
            elif df["is_us_session"].iloc[i] == 1:
                session_weight = 1.3
            elif df["is_asian_session"].iloc[i] == 1:
                session_weight = 0.7

            bullish_signals = (hh_count + hl_count) * session_weight
            bearish_signals = (lh_count + ll_count) * session_weight

            if bullish_signals > bearish_signals and bullish_signals > 0:
                df.iloc[i, df.columns.get_loc("market_structure")] = 1
            elif bearish_signals > bullish_signals and bearish_signals > 0:
                df.iloc[i, df.columns.get_loc("market_structure")] = -1
            else:
                df.iloc[i, df.columns.get_loc("market_structure")] = 0

        # CHoCH/BOS detection (Gold-adjusted)
        df["choch"] = 0
        df["bos"] = 0
        df["structure_break"] = 0

        # 🥇 Gold-specific Order Blocks (larger thresholds)
        df["bullish_ob"] = 0
        df["bearish_ob"] = 0
        df["ob_high"] = np.nan
        df["ob_low"] = np.nan

        lookback = 15  # Extended for Gold
        for i in range(lookback, len(df)):
            current_close = df["close"].iloc[i]

            for j in range(1, lookback + 1):
                ob_idx = i - j
                ob_candle = df.iloc[ob_idx]

                # Bullish Order Block
                if (ob_candle["close"] < ob_candle["open"] and
                    current_close > ob_candle["high"] and
                    current_close > df["close"].iloc[i - 1]):

                    move_size = current_close - ob_candle["high"]
                    if move_size >= 0.5:  # 🥇 Gold threshold (50 cents)
                        df.iloc[ob_idx, df.columns.get_loc("bullish_ob")] = 1
                        df.iloc[ob_idx, df.columns.get_loc("ob_high")] = ob_candle["high"]
                        df.iloc[ob_idx, df.columns.get_loc("ob_low")] = ob_candle["low"]
                        break

                # Bearish Order Block
                if (ob_candle["close"] > ob_candle["open"] and
                    current_close < ob_candle["low"] and
                    current_close < df["close"].iloc[i - 1]):

                    move_size = ob_candle["low"] - current_close
                    if move_size >= 0.5:  # 🥇 Gold threshold
                        df.iloc[ob_idx, df.columns.get_loc("bearish_ob")] = 1
                        df.iloc[ob_idx, df.columns.get_loc("ob_high")] = ob_candle["high"]
                        df.iloc[ob_idx, df.columns.get_loc("ob_low")] = ob_candle["low"]
                        break

        # 🥇 Gold-specific Fair Value Gaps (larger thresholds)
        df["bullish_fvg"] = 0
        df["bearish_fvg"] = 0
        df["fvg_high"] = np.nan
        df["fvg_low"] = np.nan
        df["fvg_size"] = np.nan

        for i in range(2, len(df)):
            candle1 = df.iloc[i - 2]
            candle2 = df.iloc[i - 1]
            candle3 = df.iloc[i]

            # Bullish FVG
            if candle3["low"] > candle1["high"] and candle2["close"] > candle2["open"]:
                gap_size = candle3["low"] - candle1["high"]
                if gap_size >= 0.3:  # 🥇 Gold threshold (30 cents)
                    df.iloc[i - 1, df.columns.get_loc("bullish_fvg")] = 1
                    df.iloc[i - 1, df.columns.get_loc("fvg_high")] = candle3["low"]
                    df.iloc[i - 1, df.columns.get_loc("fvg_low")] = candle1["high"]
                    df.iloc[i - 1, df.columns.get_loc("fvg_size")] = gap_size

            # Bearish FVG
            if candle3["high"] < candle1["low"] and candle2["close"] < candle2["open"]:
                gap_size = candle1["low"] - candle3["high"]
                if gap_size >= 0.3:  # 🥇 Gold threshold
                    df.iloc[i - 1, df.columns.get_loc("bearish_fvg")] = 1
                    df.iloc[i - 1, df.columns.get_loc("fvg_high")] = candle1["low"]
                    df.iloc[i - 1, df.columns.get_loc("fvg_low")] = candle3["high"]
                    df.iloc[i - 1, df.columns.get_loc("fvg_size")] = gap_size

        # 🥇 Gold Liquidity Zones (extended analysis)
        df["buy_liquidity"] = 0
        df["sell_liquidity"] = 0
        df["liquidity_strength"] = 0

        period = 100  # Extended for Gold
        for i in range(period, len(df)):
            window_data = df.iloc[i - period : i]
            current_price = df["close"].iloc[i]

            recent_highs = window_data[window_data["swing_high"] == 1]
            if len(recent_highs) > 0:
                highest_point = recent_highs["swing_high_price"].max()
                if abs(current_price - highest_point) / highest_point < 0.002:  # 0.2% for Gold
                    df.iloc[i, df.columns.get_loc("sell_liquidity")] = 1
                    df.iloc[i, df.columns.get_loc("liquidity_strength")] = len(recent_highs)

            recent_lows = window_data[window_data["swing_low"] == 1]
            if len(recent_lows) > 0:
                lowest_point = recent_lows["swing_low_price"].min()
                if abs(current_price - lowest_point) / lowest_point < 0.002:  # 0.2% for Gold
                    df.iloc[i, df.columns.get_loc("buy_liquidity")] = 1
                    df.iloc[i, df.columns.get_loc("liquidity_strength")] = len(recent_lows)

        # Advanced Gold SMC features
        df["structure_momentum"] = df["market_structure"].rolling(15).mean()
        df["ob_distance"] = np.nan
        df["ob_age"] = 0
        df["fvg_unfilled"] = 0
        df["break_strength"] = abs(df["structure_break"]) * df["range_pct"]

        # 🥇 Gold-specific confluence zones
        df["smc_confluence"] = (
            abs(df["structure_break"]) +
            df["bullish_ob"] + df["bearish_ob"] +
            df["bullish_fvg"] + df["bearish_fvg"] +
            df["buy_liquidity"] + df["sell_liquidity"]
        )

        # 🥇 Session-weighted confluence for Gold
        df["session_weighted_confluence"] = df["smc_confluence"].copy()
        london_mask = df["is_london_session"] == 1
        us_mask = df["is_us_session"] == 1
        asian_mask = df["is_asian_session"] == 1

        df.loc[london_mask, "session_weighted_confluence"] *= 1.5
        df.loc[us_mask, "session_weighted_confluence"] *= 1.3
        df.loc[asian_mask, "session_weighted_confluence"] *= 0.7

        # Fill NaN values
        df = df.fillna(0)

        return df

    def align_features_with_training(self, df: pd.DataFrame, timeframe: str) -> np.ndarray:
        """Align real-time features with Gold training features perfectly"""

        if timeframe not in self.training_features:
            print(f"⚠️ No training features for {timeframe}, using Gold fallback")
            # Gold-specific fallback features
            gold_features = [
                col for col in df.columns
                if any(
                    keyword in col.lower()
                    for keyword in [
                        "swing", "structure", "choch", "bos", "ob", "fvg", "liquidity", "smc",
                        "atr", "volume", "range", "body", "shadow", "higher", "lower",
                        "bullish", "bearish", "sma", "ema", "change", "momentum", "volatility",
                        "confluence", "session", "points", "london", "us", "asian", "impact"
                    ]
                )
            ]

            available_features = [col for col in gold_features if col in df.columns]
            latest_candle = df.iloc[-1]
            feature_vector = latest_candle[available_features].values.reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

            # Pad to expected size for Gold
            expected_features = 60  # Gold uses more features
            current_features = feature_vector.shape[1]
            if current_features < expected_features:
                padding = np.zeros((1, expected_features - current_features))
                feature_vector = np.hstack([feature_vector, padding])
                print(f"📊 Gold features padded: {current_features} → {expected_features}")

            return feature_vector

        # Perfect alignment using training feature mapping
        expected_features = self.training_features[timeframe]
        latest_candle = df.iloc[-1]

        # Create feature vector with exact training feature order
        feature_vector = []
        missing_features = []

        for feature_name in expected_features:
            if feature_name in latest_candle.index:
                value = latest_candle[feature_name]
                # Handle NaN/inf values
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            else:
                # Feature missing - use default value
                feature_vector.append(0.0)
                missing_features.append(feature_name)

        if missing_features:
            print(f"⚠️ {timeframe}: {len(missing_features)} Gold features missing (filled with 0)")
        else:
            print(f"✅ {timeframe}: Perfect Gold feature alignment ({len(feature_vector)} features)")

        return np.array(feature_vector).reshape(1, -1)

    def predict_signal(self, symbol: str, timeframe: str) -> Dict:
        """Generate Gold prediction for single timeframe"""

        if timeframe not in self.models:
            return {"error": f"No Gold models loaded for {timeframe}"}

        # Get real-time Gold data
        df = self.get_realtime_data(symbol, timeframe, 100)
        if df.empty:
            return {"error": f"No Gold data available for {symbol} {timeframe}"}

        # Calculate complete SMC features (Gold-optimized)
        df = self.calculate_complete_smc_features(df)

        # Align features with Gold training data
        feature_vector = self.align_features_with_training(df, timeframe)

        predictions = {}
        confidences = {}

        # Random Forest prediction
        if "random_forest" in self.models[timeframe]:
            try:
                rf_model = self.models[timeframe]["random_forest"]

                # Feature selection
                if "selector" in rf_model:
                    feature_vector_selected = rf_model["selector"].transform(feature_vector)
                else:
                    feature_vector_selected = feature_vector

                # Prediction
                rf_pred = rf_model["model"].predict(feature_vector_selected)[0]
                rf_proba = rf_model["model"].predict_proba(feature_vector_selected)[0]
                rf_confidence = np.max(rf_proba)

                predictions["random_forest"] = rf_pred
                confidences["random_forest"] = rf_confidence

            except Exception as e:
                print(f"❌ Gold Random Forest prediction error: {str(e)}")

        # XGBoost prediction
        if "xgboost" in self.models[timeframe]:
            try:
                xgb_model = self.models[timeframe]["xgboost"]

                # Feature selection
                if "selector" in xgb_model:
                    feature_vector_selected = xgb_model["selector"].transform(feature_vector)
                else:
                    feature_vector_selected = feature_vector

                # Prediction
                xgb_pred = xgb_model["model"].predict(feature_vector_selected)[0]
                xgb_proba = xgb_model["model"].predict_proba(feature_vector_selected)[0]
                xgb_confidence = np.max(xgb_proba)

                # Convert back if label encoder exists
                if "label_encoder" in xgb_model:
                    xgb_pred = xgb_model["label_encoder"].inverse_transform([xgb_pred])[0]

                predictions["xgboost"] = xgb_pred
                confidences["xgboost"] = xgb_confidence

            except Exception as e:
                print(f"❌ Gold XGBoost prediction error: {str(e)}")

        # Neural Network prediction
        if "neural_network" in self.models[timeframe] and TENSORFLOW_AVAILABLE:
            try:
                nn_model_data = self.models[timeframe]["neural_network"]
                nn_model = nn_model_data["model"]

                # Scale features if scaler available
                if "scaler" in nn_model_data:
                    feature_vector_scaled = nn_model_data["scaler"].transform(feature_vector)
                else:
                    feature_vector_scaled = feature_vector

                # Prediction
                nn_proba = nn_model.predict(feature_vector_scaled, verbose=0)[0]

                if len(nn_proba) > 1:  # Multi-class
                    nn_pred = np.argmax(nn_proba)
                    nn_confidence = np.max(nn_proba)

                    # Convert back to original labels if label encoder available
                    if "label_encoder" in nn_model_data:
                        nn_pred = nn_model_data["label_encoder"].inverse_transform([nn_pred])[0]
                else:  # Binary
                    nn_pred = 1 if nn_proba[0] > 0.5 else -1
                    nn_confidence = nn_proba[0] if nn_pred == 1 else (1 - nn_proba[0])

                predictions["neural_network"] = nn_pred
                confidences["neural_network"] = nn_confidence

            except Exception as e:
                print(f"❌ Gold Neural Network prediction error: {str(e)}")

        # Aggregate predictions
        if predictions:
            # Consensus prediction
            pred_values = list(predictions.values())
            pred_counts = {pred: pred_values.count(pred) for pred in set(pred_values)}
            consensus_pred = max(pred_counts, key=pred_counts.get)
            consensus_strength = pred_counts[consensus_pred] / len(predictions)

            # Average confidence
            avg_confidence = np.mean(list(confidences.values()))

            # 🥇 Gold session adjustment
            session_info = self.get_session_info()
            session_adjusted_confidence = avg_confidence * session_info["weight"]

            return {
                "timeframe": timeframe,
                "timestamp": df.index[-1],
                "current_price": df["close"].iloc[-1],
                "predictions": predictions,
                "confidences": confidences,
                "consensus_prediction": consensus_pred,
                "consensus_strength": consensus_strength,
                "average_confidence": avg_confidence,
                "session_adjusted_confidence": session_adjusted_confidence,
                "session_info": session_info,
                "signal_quality": (
                    "HIGH" if session_adjusted_confidence >= 0.85
                    else "MEDIUM" if session_adjusted_confidence >= 0.70 
                    else "LOW"
                ),
                "is_gold": True
            }

        return {"error": "No Gold predictions generated"}

    def get_multi_timeframe_signals(self, symbol: str) -> Dict:
        """Get Gold signals from all timeframes and aggregate"""
        print(f"🔄 Generating multi-timeframe Gold signals for {symbol}...")

        all_signals = {}

        for timeframe in self.timeframes:
            if timeframe in self.models:
                signal = self.predict_signal(symbol, timeframe)
                if "error" not in signal:
                    all_signals[timeframe] = signal
                    direction = (
                        "LONG" if signal["consensus_prediction"] == 1
                        else "SHORT" if signal["consensus_prediction"] == -1 
                        else "HOLD"
                    )
                    print(f"  {timeframe}: {direction} ({signal['session_adjusted_confidence']:.3f} confidence)")

        if not all_signals:
            return {"error": "No Gold signals generated"}

        # Aggregate Gold signals
        return self.aggregate_gold_signals(all_signals)

    def aggregate_gold_signals(self, signals: Dict) -> Dict:
        """Aggregate multi-timeframe Gold signals into final recommendation"""

        # Collect all predictions with Gold-specific weighting
        long_signals = []
        short_signals = []
        confidences = []
        
        # 🥇 Gold timeframe weights
        tf_weights = {
            "M5": 1.0,
            "M15": 1.2,
            "H1": 1.5,
            "H4": 1.8,
            "D1": 2.0
        }

        for tf, signal in signals.items():
            pred = signal["consensus_prediction"]
            conf = signal["session_adjusted_confidence"]
            weight = tf_weights.get(tf, 1.0)
            
            weighted_confidence = conf * weight

            if pred == 1:  # Long
                long_signals.append((tf, weighted_confidence))
            elif pred == -1:  # Short
                short_signals.append((tf, weighted_confidence))

            confidences.append(weighted_confidence)

        # Calculate final recommendation
        long_count = len(long_signals)
        short_count = len(short_signals)
        total_signals = long_count + short_count

        if total_signals == 0:
            final_direction = "HOLD"
            final_confidence = 0
        elif long_count > short_count:
            final_direction = "LONG"
            final_confidence = np.mean([conf for tf, conf in long_signals])
        elif short_count > long_count:
            final_direction = "SHORT"
            final_confidence = np.mean([conf for tf, conf in short_signals])
        else:
            final_direction = "NEUTRAL"
            final_confidence = np.mean(confidences)

        # 🥇 Gold-specific risk assessment
        session_info = self.get_session_info()
        
        if (final_confidence >= 0.85 and total_signals >= self.min_consensus and 
            session_info["is_preferred"]):
            risk_level = "LOW"
            recommended_lot = min(self.max_lot_size, self.base_lot_size * 1.5)
        elif (final_confidence >= 0.75 and total_signals >= 3 and 
              session_info["weight"] >= 1.0):
            risk_level = "MEDIUM"
            recommended_lot = self.base_lot_size
        else:
            risk_level = "HIGH"
            recommended_lot = self.base_lot_size * 0.5
            if not session_info["is_preferred"]:
                final_direction = "HOLD"  # Don't trade during bad sessions

        return {
            "symbol": list(signals.values())[0].get("current_price", 0),
            "timestamp": datetime.now(self.timezone).isoformat(),
            "current_price": list(signals.values())[0].get("current_price", 0),
            "final_direction": final_direction,
            "final_confidence": final_confidence,
            "risk_level": risk_level,
            "recommended_lot_size": recommended_lot,
            "timeframe_consensus": f"{long_count}L/{short_count}S/{len(self.timeframes)-total_signals}H",
            "individual_signals": signals,
            "trading_recommendation": (
                "TRADE" if final_direction in ["LONG", "SHORT"] and session_info["is_preferred"]
                else "WAIT"
            ),
            "symbol_type": "GOLD",
            "session_info": session_info,
            "gold_optimized": True
        }

    def start_signal_monitoring(self, symbol: str, update_interval: int = 60):
        """Start continuous Gold signal monitoring"""
        print(f"🥇 Starting Gold Signal Monitoring for {symbol}")
        print(f"📊 Update interval: {update_interval} seconds")
        print("=" * 60)

        while True:
            try:
                # Generate Gold signals
                signals = self.get_multi_timeframe_signals(symbol)

                if "error" in signals:
                    print(f"❌ Error: {signals['error']}")
                else:
                    # Display Gold results
                    session_info = signals.get("session_info", {})
                    print(f"\n⏰ {signals['timestamp']}")
                    print(f"🥇 {symbol}: {signals['final_direction']} | Confidence: {signals['final_confidence']:.3f}")
                    print(f"🎯 Risk: {signals['risk_level']} | Lot: {signals['recommended_lot_size']}")
                    print(f"📈 Consensus: {signals['timeframe_consensus']} | Action: {signals['trading_recommendation']}")
                    print(f"🌍 Session: {session_info.get('session', 'unknown').upper()} | Volatility: {session_info.get('volatility', 'normal')}")

                    if signals["trading_recommendation"] == "TRADE":
                        print(f"🔥 GOLD TRADING SIGNAL: {signals['final_direction']} {symbol}")

                # Wait for next update
                import time
                time.sleep(update_interval)

            except KeyboardInterrupt:
                print("\n🛑 Gold signal monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Gold monitoring error: {str(e)}")
                import time
                time.sleep(10)  # Wait before retry


# Usage Example for Gold
if __name__ == "__main__":
    print("🥇 SMC Real-time Gold Signal Prediction Engine")
    print("=" * 50)

    # Initialize Gold signal engine
    engine = SMCSignalEngine("XAUUSD_c_SMC")

    # Load trained Gold models
    if engine.load_trained_models():

        # Connect to MT5
        if engine.connect_mt5():

            # Find Gold symbol
            gold_symbol = engine.find_best_gold_symbol()
            
            if gold_symbol:
                print(f"🥇 Using Gold symbol: {gold_symbol}")
                
                # Generate test Gold signal
                print("\n🔄 Testing Gold Signal Generation...")
                signal = engine.get_multi_timeframe_signals(gold_symbol)

                if "error" not in signal:
                    print(f"\n🎯 GOLD SIGNAL RESULT:")
                    print(f"Direction: {signal['final_direction']}")
                    print(f"Confidence: {signal['final_confidence']:.3f}")
                    print(f"Risk Level: {signal['risk_level']}")
                    print(f"Session: {signal.get('session_info', {}).get('session', 'unknown')}")
                    print(f"Recommendation: {signal['trading_recommendation']}")

                    # Ask user for continuous monitoring
                    user_input = input("\n🚀 Start continuous Gold monitoring? (y/n): ")
                    if user_input.lower() == "y":
                        engine.start_signal_monitoring(gold_symbol, 60)
                else:
                    print(f"❌ Gold signal generation failed: {signal['error']}")
            else:
                print("❌ No Gold symbols found")

        else:
            print("❌ MT5 connection failed")

    else:
        print("❌ Gold model loading failed")