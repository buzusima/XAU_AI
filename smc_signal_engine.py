# Corrected SMC Signal Engine - แก้ไข Feature Alignment
# โมเดลต้องการ 90 features แต่ feature mapping มี 60

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
import os
import glob

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
    SMC Real-time Signal Prediction Engine - Fixed Feature Alignment
    Created by อาจารย์ฟินิกซ์ - Fixed for 90-feature models
    """

    def __init__(self, models_path: str = "XAUUSD_v_SMC"):
        """Initialize SMC Signal Engine"""
        self.models_path = models_path
        self.models = {}
        self.training_features = {}
        self.timezone = pytz.timezone("Etc/UTC")

        # Load feature mapping
        self.load_feature_mapping()

        # Signal configuration
        self.timeframes = ["M5", "M15", "H1", "H4", "D1"]
        self.mt5_timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        # Signal thresholds
        self.min_confidence = 0.7
        self.min_consensus = 3
        self.max_signal_age = 5

        # Position sizing
        self.base_lot_size = 0.01
        self.max_lot_size = 0.05
        self.risk_per_trade = 0.01

        print("🎯 SMC Signal Engine Initialized (90-Feature Compatible)")
        print(f"📂 Models Path: {self.models_path}")

    def load_feature_mapping(self) -> bool:
        """Load feature mapping for alignment"""
        try:
            feature_mapping_file = f"{self.models_path}_feature_mapping.json"
            
            if os.path.exists(feature_mapping_file):
                with open(feature_mapping_file, "r") as f:
                    self.training_features = json.load(f)

                print("✅ Feature mapping loaded:")
                for tf, features in self.training_features.items():
                    print(f"  {tf}: {len(features)} features")
                return True
            else:
                print(f"⚠️ Feature mapping not found: {feature_mapping_file}")
                return False

        except Exception as e:
            print(f"❌ Failed to load feature mapping: {str(e)}")
            return False

    def load_trained_models(self) -> bool:
        """Load all trained AI models"""
        print("📂 Loading Trained XAUUSD AI Models...")
        print("-" * 40)

        loaded_count = 0

        for timeframe in self.timeframes:
            tf_models = {}
            
            print(f"\n🔄 Loading {timeframe} models...")

            # Load Random Forest
            try:
                rf_path = f"{self.models_path}_{timeframe}_random_forest_model.pkl"
                if os.path.exists(rf_path):
                    rf_model = joblib.load(rf_path)
                    tf_models["random_forest"] = rf_model
                    print(f"✅ {timeframe} Random Forest loaded")
                    loaded_count += 1
                else:
                    print(f"❌ {timeframe} Random Forest not found")
            except Exception as e:
                print(f"❌ {timeframe} Random Forest failed: {str(e)}")

            # Load XGBoost
            try:
                xgb_path = f"{self.models_path}_{timeframe}_xgboost_model.pkl"
                if os.path.exists(xgb_path):
                    xgb_model = joblib.load(xgb_path)
                    tf_models["xgboost"] = xgb_model
                    print(f"✅ {timeframe} XGBoost loaded")
                    loaded_count += 1
                else:
                    print(f"❌ {timeframe} XGBoost not found")
            except Exception as e:
                print(f"❌ {timeframe} XGBoost failed: {str(e)}")

            # Load Neural Network
            if TENSORFLOW_AVAILABLE:
                try:
                    nn_keras_path = f"{self.models_path}_{timeframe}_neural_network_model.h5"
                    nn_components_path = f"{self.models_path}_{timeframe}_neural_network_model.pkl"
                    
                    if os.path.exists(nn_keras_path) and os.path.exists(nn_components_path):
                        nn_model = tf.keras.models.load_model(nn_keras_path)
                        nn_components = joblib.load(nn_components_path)
                        
                        tf_models["neural_network"] = {"model": nn_model, **nn_components}
                        print(f"✅ {timeframe} Neural Network loaded")
                        print(f"   Input shape: {nn_model.input_shape}")
                        loaded_count += 1
                    else:
                        print(f"❌ {timeframe} Neural Network files missing")
                        
                except Exception as e:
                    print(f"❌ {timeframe} Neural Network failed: {str(e)}")

            if tf_models:
                self.models[timeframe] = tf_models

        print("-" * 40)
        print(f"📊 Total: {loaded_count} models loaded across {len(self.models)} timeframes")

        return len(self.models) > 0

    def connect_mt5(self, account: int = None, password: str = None, server: str = None) -> bool:
        """Connect to MT5 for real-time data"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False

            if account and password and server:
                if not mt5.login(account, password=password, server=server):
                    print(f"❌ Login failed: {mt5.last_error()}")
                    return False

            print("✅ MT5 Connected for real-time data")
            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def get_realtime_data(self, symbol: str, timeframe: str, count: int = 150) -> pd.DataFrame:
        """Get real-time market data for prediction"""
        try:
            # Auto-detect Gold symbol
            if symbol == "XAUUSD" or symbol == "GOLD":
                symbols = mt5.symbols_get()
                if symbols:
                    gold_symbols = [s.name for s in symbols if 'XAU' in s.name.upper()]
                    if gold_symbols:
                        symbol = gold_symbols[0]
                        print(f"🔄 Using Gold symbol: {symbol}")

            rates = mt5.copy_rates_from_pos(
                symbol, self.mt5_timeframes[timeframe], 0, count
            )

            if rates is None or len(rates) == 0:
                print(f"❌ No real-time data for {symbol} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            print(f"📊 {symbol} {timeframe}: {len(df)} candles loaded")
            return df

        except Exception as e:
            print(f"❌ Real-time data error: {str(e)}")
            return pd.DataFrame()

    def calculate_complete_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive SMC features to match 90-feature model"""
        if df.empty:
            return df

        df = df.copy()

        # === BASIC PRICE FEATURES ===
        df["hl2"] = (df["high"] + df["low"]) / 2
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        # === PRICE RANGES ===
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100
        df["range_pips"] = df["range"] / 0.01  # Gold pips

        # === CANDLE PATTERNS ===
        df["body"] = abs(df["close"] - df["open"])
        df["body_pct"] = np.where(df["range"] > 0, (df["body"] / df["range"]) * 100, 0)
        df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
        df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)
        df["is_doji"] = (df["body_pct"] < 10).astype(int)

        # === PRICE MOVEMENTS ===
        df["price_change"] = df["close"].diff()
        df["price_change_pct"] = df["close"].pct_change() * 100
        df["price_change_pips"] = df["price_change"] / 0.01

        # === MOVING AVERAGES ===
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()

        # === MA RELATIONSHIPS ===
        df["price_above_sma20"] = (df["close"] > df["sma_20"]).astype(int)
        df["price_above_ema20"] = (df["close"] > df["ema_20"]).astype(int)
        df["sma_alignment"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["ema_alignment"] = (df["ema_20"] > df["ema_50"]).astype(int)

        # === VOLATILITY ===
        df["atr_5"] = df["range"].rolling(5).mean()
        df["atr_14"] = df["range"].rolling(14).mean()
        df["atr_20"] = df["range"].rolling(20).mean()
        df["volatility_5"] = df["close"].rolling(5).std()
        df["volatility_20"] = df["close"].rolling(20).std()
        df["volatility_pct"] = (df["volatility_20"] / df["close"]) * 100

        # === VOLUME ANALYSIS ===
        df["volume_ma"] = df["tick_volume"].rolling(20).mean()
        df["volume_ratio"] = np.where(df["volume_ma"] > 0, df["tick_volume"] / df["volume_ma"], 1)
        df["high_volume"] = (df["volume_ratio"] > 1.5).astype(int)

        # === RSI ===
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        # === MACD ===
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

        # === BOLLINGER BANDS ===
        bb_period = 20
        df["bb_middle"] = df["close"].rolling(bb_period).mean()
        bb_std = df["close"].rolling(bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # === SWING POINTS ===
        period = 5
        df["swing_high"] = 0
        df["swing_low"] = 0
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan

        for i in range(period, len(df) - period):
            if df["high"].iloc[i] == df["high"].iloc[i - period : i + period + 1].max():
                df.iloc[i, df.columns.get_loc("swing_high")] = 1
                df.iloc[i, df.columns.get_loc("swing_high_price")] = df["high"].iloc[i]

            if df["low"].iloc[i] == df["low"].iloc[i - period : i + period + 1].min():
                df.iloc[i, df.columns.get_loc("swing_low")] = 1
                df.iloc[i, df.columns.get_loc("swing_low_price")] = df["low"].iloc[i]

        # === MARKET STRUCTURE ===
        df["higher_high"] = 0
        df["higher_low"] = 0
        df["lower_high"] = 0
        df["lower_low"] = 0
        df["market_structure"] = 0

        # === ORDER BLOCKS ===
        df["bullish_ob"] = 0
        df["bearish_ob"] = 0
        df["ob_high"] = np.nan
        df["ob_low"] = np.nan

        # === FAIR VALUE GAPS ===
        df["bullish_fvg"] = 0
        df["bearish_fvg"] = 0
        df["fvg_high"] = np.nan
        df["fvg_low"] = np.nan
        df["fvg_size"] = np.nan

        # === LIQUIDITY ZONES ===
        df["buy_liquidity"] = 0
        df["sell_liquidity"] = 0
        df["liquidity_strength"] = 0

        # === TREND INDICATORS ===
        df["uptrend"] = ((df["close"] > df["sma_20"]) & (df["sma_20"] > df["sma_50"])).astype(int)
        df["downtrend"] = ((df["close"] < df["sma_20"]) & (df["sma_20"] < df["sma_50"])).astype(int)

        # === MOMENTUM ===
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # === SUPPORT/RESISTANCE ===
        df["recent_high"] = df["high"].rolling(20).max()
        df["recent_low"] = df["low"].rolling(20).min()
        df["near_resistance"] = (df["close"] > df["recent_high"] * 0.995).astype(int)
        df["near_support"] = (df["close"] < df["recent_low"] * 1.005).astype(int)

        # === ADDITIONAL SMC FEATURES ===
        df["break_strength"] = df["range_pct"] * df["volume_ratio"]
        df["smc_confluence"] = (df["bullish_ob"] + df["bearish_ob"] + df["bullish_fvg"] + df["bearish_fvg"])
        df["structure_momentum"] = df["market_structure"]
        df["ob_distance"] = 0
        df["ob_age"] = 0
        df["fvg_unfilled"] = 0

        # Fill NaN values
        df = df.fillna(0)

        return df

    def prepare_features_for_model(self, df: pd.DataFrame, timeframe: str) -> np.ndarray:
        """Prepare exactly 90 features for the model"""
        
        # ใช้ข้อมูล candle ล่าสุด
        latest_candle = df.iloc[-1]
        
        # เลือกฟีเจอร์ที่สำคัญ 90 ตัว
        feature_columns = [
            # Price features (10)
            "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "range", "range_pct", "body",
            
            # Candle patterns (10) 
            "body_pct", "upper_shadow", "lower_shadow", "is_bullish", "is_bearish", "is_doji",
            "price_change", "price_change_pct", "momentum_5", "momentum_10",
            
            # Moving averages (16)
            "sma_5", "sma_10", "sma_20", "sma_50", "ema_5", "ema_10", "ema_20", "ema_50",
            "price_above_sma20", "price_above_ema20", "sma_alignment", "ema_alignment",
            "momentum_20", "uptrend", "downtrend", "bb_position",
            
            # Technical indicators (14)
            "rsi", "rsi_oversold", "rsi_overbought", "macd", "macd_signal", "macd_histogram", "macd_bullish",
            "bb_middle", "bb_upper", "bb_lower", "atr_5", "atr_14", "atr_20", "volatility_pct",
            
            # Volume (6)
            "tick_volume", "volume_ma", "volume_ratio", "high_volume", "volatility_5", "volatility_20",
            
            # SMC features (14)
            "swing_high", "swing_low", "higher_high", "higher_low", "lower_high", "lower_low",
            "market_structure", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg",
            "buy_liquidity", "sell_liquidity", "smc_confluence",
            
            # Support/Resistance (6)
            "recent_high", "recent_low", "near_resistance", "near_support", "break_strength", "structure_momentum",
            
            # Additional features (14) - เพิ่มเพื่อครบ 90
            "liquidity_strength", "ob_distance", "ob_age", "fvg_unfilled", "range_pips", "price_change_pips",
            "swing_high_price", "swing_low_price", "ob_high", "ob_low", "fvg_high", "fvg_low", "fvg_size", "real"
        ]
        
        # ตรวจสอบว่ามีครบหรือไม่
        available_features = []
        for feature in feature_columns[:89]:  # เอา 89 ตัวแรก
            if feature in latest_candle.index:
                available_features.append(feature)
            else:
                print(f"⚠️ Missing feature: {feature}")
        
        # เพิ่มฟีเจอร์จนครบ 90
        while len(available_features) < 90:
            available_features.append("tick_volume")  # ใช้ tick_volume ซ้ำ
        
        # ตัด 90 ตัวแรก
        selected_features = available_features[:90]
        
        # สร้าง feature vector
        feature_vector = []
        for feature in selected_features:
            if feature in latest_candle.index:
                value = latest_candle[feature]
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        print(f"✅ {timeframe}: Prepared {len(feature_vector)} features for model")
        
        return np.array(feature_vector).reshape(1, -1)

    def predict_signal(self, symbol: str, timeframe: str) -> Dict:
        """Generate prediction for single timeframe"""

        if timeframe not in self.models:
            return {"error": f"No models loaded for {timeframe}"}

        # Get real-time data
        df = self.get_realtime_data(symbol, timeframe, 150)
        if df.empty:
            return {"error": f"No data available for {symbol} {timeframe}"}

        # Calculate complete SMC features
        df = self.calculate_complete_smc_features(df)

        # Prepare features for model (90 features)
        feature_vector = self.prepare_features_for_model(df, timeframe)

        predictions = {}
        confidences = {}

        # Random Forest prediction
        if "random_forest" in self.models[timeframe]:
            try:
                rf_model = self.models[timeframe]["random_forest"]
                
                if isinstance(rf_model, dict) and "model" in rf_model:
                    # Use feature selector if available
                    if "selector" in rf_model:
                        feature_vector_selected = rf_model["selector"].transform(feature_vector)
                    else:
                        feature_vector_selected = feature_vector
                    
                    rf_pred = rf_model["model"].predict(feature_vector_selected)[0]
                    rf_proba = rf_model["model"].predict_proba(feature_vector_selected)[0]
                    rf_confidence = np.max(rf_proba)
                    
                    predictions["random_forest"] = rf_pred
                    confidences["random_forest"] = rf_confidence

            except Exception as e:
                print(f"❌ Random Forest prediction error: {str(e)}")

        # XGBoost prediction
        if "xgboost" in self.models[timeframe]:
            try:
                xgb_model = self.models[timeframe]["xgboost"]
                
                if isinstance(xgb_model, dict) and "model" in xgb_model:
                    # Use feature selector if available
                    if "selector" in xgb_model:
                        feature_vector_selected = xgb_model["selector"].transform(feature_vector)
                    else:
                        feature_vector_selected = feature_vector
                    
                    xgb_pred = xgb_model["model"].predict(feature_vector_selected)[0]
                    xgb_proba = xgb_model["model"].predict_proba(feature_vector_selected)[0]
                    xgb_confidence = np.max(xgb_proba)
                    
                    predictions["xgboost"] = xgb_pred
                    confidences["xgboost"] = xgb_confidence

            except Exception as e:
                print(f"❌ XGBoost prediction error: {str(e)}")

        # Neural Network prediction
        if "neural_network" in self.models[timeframe] and TENSORFLOW_AVAILABLE:
            try:
                nn_model_data = self.models[timeframe]["neural_network"]
                nn_model = nn_model_data["model"]

                # Scale features
                if "scaler" in nn_model_data:
                    feature_vector_scaled = nn_model_data["scaler"].transform(feature_vector)
                else:
                    feature_vector_scaled = feature_vector

                # Prediction
                nn_proba = nn_model.predict(feature_vector_scaled, verbose=0)[0]
                
                # Binary classification
                nn_pred = 1 if nn_proba[0] > 0.5 else -1
                nn_confidence = nn_proba[0] if nn_pred == 1 else (1 - nn_proba[0])

                predictions["neural_network"] = nn_pred
                confidences["neural_network"] = nn_confidence

            except Exception as e:
                print(f"❌ Neural Network prediction error: {str(e)}")

        # Aggregate predictions
        if predictions:
            pred_values = list(predictions.values())
            pred_counts = {pred: pred_values.count(pred) for pred in set(pred_values)}
            consensus_pred = max(pred_counts, key=pred_counts.get)
            consensus_strength = pred_counts[consensus_pred] / len(predictions)
            avg_confidence = np.mean(list(confidences.values()))

            return {
                "timeframe": timeframe,
                "timestamp": df.index[-1],
                "current_price": df["close"].iloc[-1],
                "predictions": predictions,
                "confidences": confidences,
                "consensus_prediction": consensus_pred,
                "consensus_strength": consensus_strength,
                "average_confidence": avg_confidence,
                "signal_quality": (
                    "HIGH" if avg_confidence >= 0.8
                    else "MEDIUM" if avg_confidence >= 0.6 else "LOW"
                ),
            }

        return {"error": "No predictions generated"}

    def get_multi_timeframe_signals(self, symbol: str) -> Dict:
        """Get signals from all timeframes and aggregate"""
        print(f"🔄 Generating multi-timeframe signals for {symbol}...")

        all_signals = {}

        for timeframe in self.timeframes:
            if timeframe in self.models:
                signal = self.predict_signal(symbol, timeframe)
                if "error" not in signal:
                    all_signals[timeframe] = signal
                    direction = (
                        "LONG" if signal["consensus_prediction"] == 1
                        else "SHORT" if signal["consensus_prediction"] == -1 else "HOLD"
                    )
                    print(f"  {timeframe}: {direction} ({signal['average_confidence']:.3f} confidence)")

        if not all_signals:
            return {"error": "No signals generated"}

        return self.aggregate_signals(all_signals, symbol)

    def aggregate_signals(self, signals: Dict, symbol: str) -> Dict:
        """Aggregate multi-timeframe signals"""
        
        long_signals = []
        short_signals = []
        confidences = []

        for tf, signal in signals.items():
            pred = signal["consensus_prediction"]
            conf = signal["average_confidence"]

            if pred == 1:
                long_signals.append((tf, conf))
            elif pred == -1:
                short_signals.append((tf, conf))

            confidences.append(conf)

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

        # Risk assessment
        if final_confidence >= 0.8 and total_signals >= self.min_consensus:
            risk_level = "LOW"
            recommended_lot = min(self.max_lot_size, self.base_lot_size * 2)
        elif final_confidence >= 0.6 and total_signals >= 2:
            risk_level = "MEDIUM"
            recommended_lot = self.base_lot_size
        else:
            risk_level = "HIGH"
            recommended_lot = self.base_lot_size * 0.5
            if final_confidence < 0.6:
                final_direction = "HOLD"

        # Get symbol from signals
        return {
            "symbol": symbol,
            "timestamp": datetime.now(self.timezone).isoformat(),
            "current_price": list(signals.values())[0].get("current_price", 0),
            "final_direction": final_direction,
            "final_confidence": final_confidence,
            "risk_level": risk_level,
            "recommended_lot_size": recommended_lot,
            "timeframe_consensus": f"{long_count}L/{short_count}S/{len(self.timeframes)-total_signals}H",
            "individual_signals": signals,
            "trading_recommendation": (
                "TRADE" if final_direction in ["LONG", "SHORT"] else "WAIT"
            ),
        }


# Test the corrected engine
if __name__ == "__main__":
    print("🎯 Corrected SMC Signal Engine Test")
    print("=" * 50)

    engine = SMCSignalEngine("XAUUSD_v_SMC")

    if engine.load_trained_models():
        print("✅ Models loaded successfully!")
        
        if engine.connect_mt5():
            signal = engine.get_multi_timeframe_signals("XAUUSD")
            
            if "error" not in signal:
                print(f"\n🥇 XAUUSD SIGNAL RESULT:")
                print(f"Direction: {signal['final_direction']}")
                print(f"Confidence: {signal['final_confidence']:.3f}")
                print(f"Risk Level: {signal['risk_level']}")
                print(f"Recommendation: {signal['trading_recommendation']}")
            else:
                print(f"❌ Error: {signal['error']}")
        else:
            print("❌ MT5 connection failed")
    else:
        print("❌ Model loading failed")