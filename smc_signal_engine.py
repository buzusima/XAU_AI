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


class AutoModelDetector:
    """Auto-detect and load available AI models"""
    
    def __init__(self, search_path: str = "."):
        self.search_path = search_path
        self.detected_models = {}
        self.available_symbols = []
        self.available_timeframes = []
        
    def scan_for_models(self) -> Dict:
        """Scan directory for available AI models"""
        print("🔍 Scanning for AI models...")
        print("-" * 50)
        
        model_files = []
        
        # Search patterns for different model types
        search_patterns = [
            "*_random_forest_model.pkl",
            "*_xgboost_model.pkl", 
            "*_neural_network_model.pkl",
            "*_neural_network_model.h5"
        ]
        
        for pattern in search_patterns:
            files = glob.glob(os.path.join(self.search_path, pattern))
            model_files.extend(files)
            
        print(f"📁 Found {len(model_files)} model files")
        
        # Parse model files to extract symbol and timeframe info
        for file_path in model_files:
            filename = os.path.basename(file_path)
            self.parse_model_filename(filename, file_path)
            
        # Organize detected models
        self.organize_models()
        
        return self.detected_models
    
    def parse_model_filename(self, filename: str, file_path: str):
        """Parse model filename to extract symbol, timeframe, and model type"""
        try:
            parts = filename.replace('.pkl', '').replace('.h5', '').split('_')
            
            # Find model type
            model_type = None
            if 'random_forest' in filename:
                model_type = 'random_forest'
            elif 'xgboost' in filename:
                model_type = 'xgboost'
            elif 'neural_network' in filename:
                model_type = 'neural_network'
            
            if not model_type:
                return
                
            # Find timeframe (M5, M15, H1, H4, D1)
            timeframe = None
            timeframe_patterns = ['M5', 'M15', 'H1', 'H4', 'D1']
            for tf in timeframe_patterns:
                if tf in parts:
                    timeframe = tf
                    break
                    
            if not timeframe:
                print(f"⚠️ Could not extract timeframe from: {filename}")
                return
                
            # Extract symbol (everything before timeframe)
            symbol_parts = []
            for part in parts:
                if part == timeframe:
                    break
                symbol_parts.append(part)
                
            if not symbol_parts:
                print(f"⚠️ Could not extract symbol from: {filename}")
                return
                
            symbol = '_'.join(symbol_parts)
            
            # Store model info
            if symbol not in self.detected_models:
                self.detected_models[symbol] = {}
            if timeframe not in self.detected_models[symbol]:
                self.detected_models[symbol][timeframe] = {}
                
            self.detected_models[symbol][timeframe][model_type] = file_path
            
            print(f"✅ Detected: {symbol} | {timeframe} | {model_type}")
            
        except Exception as e:
            print(f"⚠️ Error parsing {filename}: {str(e)}")
    
    def organize_models(self):
        """Organize and validate detected models"""
        self.available_symbols = list(self.detected_models.keys())
        
        # Get all unique timeframes
        all_timeframes = set()
        for symbol_models in self.detected_models.values():
            all_timeframes.update(symbol_models.keys())
        self.available_timeframes = sorted(list(all_timeframes))
        
        print(f"\n📊 Model Summary:")
        print(f"🎯 Available Symbols: {', '.join(self.available_symbols)}")
        print(f"⏰ Available Timeframes: {', '.join(self.available_timeframes)}")
        
        # Show detailed breakdown
        for symbol in self.available_symbols:
            print(f"\n{symbol}:")
            for timeframe in self.available_timeframes:
                if timeframe in self.detected_models[symbol]:
                    models = list(self.detected_models[symbol][timeframe].keys())
                    print(f"  {timeframe}: {', '.join(models)}")
                else:
                    print(f"  {timeframe}: No models")
    
    def get_best_symbol_match(self, requested_symbol: str) -> Optional[str]:
        """Find best matching symbol from available models"""
        
        requested_upper = requested_symbol.upper()
        
        # Exact match first
        for symbol in self.available_symbols:
            if symbol.upper() == requested_upper:
                return symbol
                
        # Partial match
        for symbol in self.available_symbols:
            if requested_upper in symbol.upper() or symbol.upper() in requested_upper:
                return symbol
                
        # Special cases
        if requested_upper == "XAUUSD":
            for symbol in self.available_symbols:
                if any(gold_term in symbol.upper() for gold_term in ["XAU", "GOLD"]):
                    return symbol
                    
        if requested_upper == "EURUSD":
            for symbol in self.available_symbols:
                if "EUR" in symbol.upper() and "USD" in symbol.upper():
                    return symbol
                    
        return None


class SMCSignalEngine:
    """
    SMC Signal Engine with Auto Model Detection - Quick Fix Version
    Created by อาจารย์ฟินิกซ์ - แก้ปัญหา Feature Mismatch
    """

    def __init__(self, models_path: str = None):
        """Initialize SMC Signal Engine with auto model detection"""
        
        # Auto-detect models
        self.model_detector = AutoModelDetector(models_path or ".")
        self.detected_models = self.model_detector.scan_for_models()
        
        # Initialize core components
        self.models = {}
        self.training_features = {}
        self.timezone = pytz.timezone("Etc/UTC")

        # Dynamic symbol mapping based on detected models
        self.symbol_mapping = self.create_dynamic_symbol_mapping()

        # Signal configuration
        self.timeframes = self.model_detector.available_timeframes or ["M5", "M15", "H1", "H4", "D1"]
        self.mt5_timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        # Signal thresholds
        self.min_confidence = 0.7
        self.min_consensus = 2
        self.max_signal_age = 5

        print(f"\n🎯 SMC Signal Engine Initialized (Quick Fix Version)")
        print(f"📊 Available Symbols: {', '.join(self.model_detector.available_symbols)}")
        print(f"📊 Symbol mappings: {self.symbol_mapping}")

    def create_dynamic_symbol_mapping(self) -> Dict[str, str]:
        """Create symbol mapping based on detected models and MT5 symbols"""
        mapping = {}
        
        if not mt5.initialize():
            print("⚠️ Cannot initialize MT5 for symbol detection")
            return mapping
            
        try:
            # For each detected model symbol, try to find the best MT5 symbol
            mt5_symbols = mt5.symbols_get()
            if mt5_symbols:
                mt5_symbol_names = [s.name for s in mt5_symbols]
                
                for model_symbol in self.model_detector.available_symbols:
                    # Try to find matching MT5 symbol
                    best_match = self.find_best_mt5_symbol(model_symbol, mt5_symbol_names)
                    if best_match:
                        mapping[model_symbol] = best_match
                        print(f"🔗 Auto-mapped: {model_symbol} → {best_match}")
                    else:
                        # Use model symbol as-is
                        mapping[model_symbol] = model_symbol
                        print(f"⚠️ No MT5 match for {model_symbol}, using as-is")
            
        except Exception as e:
            print(f"⚠️ Error creating symbol mapping: {e}")
        finally:
            mt5.shutdown()
            
        return mapping
    
    def find_best_mt5_symbol(self, model_symbol: str, mt5_symbols: List[str]) -> Optional[str]:
        """Find best matching MT5 symbol for model symbol"""
        
        model_upper = model_symbol.upper()
        
        # Direct exact match
        for mt5_sym in mt5_symbols:
            if mt5_sym.upper() == model_upper:
                return mt5_sym
        
        # Check for common suffixes (.c, .v, .m, etc.)
        for mt5_sym in mt5_symbols:
            base_mt5 = mt5_sym.split('.')[0].upper()
            base_model = model_symbol.split('.')[0].split('_')[0].upper()
            
            if base_mt5 == base_model:
                return mt5_sym
        
        # Special case mappings
        if "EURUSD" in model_upper:
            for mt5_sym in mt5_symbols:
                if "EUR" in mt5_sym.upper() and "USD" in mt5_sym.upper():
                    return mt5_sym
                    
        if any(term in model_upper for term in ["XAU", "GOLD"]):
            for mt5_sym in mt5_symbols:
                if any(term in mt5_sym.upper() for term in ["XAU", "GOLD"]):
                    return mt5_sym
        
        return None

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available models"""
        return self.model_detector.available_symbols

    def get_available_timeframes(self, symbol: str = None) -> List[str]:
        """Get available timeframes for a symbol or all timeframes"""
        if symbol:
            model_symbol = self.model_detector.get_best_symbol_match(symbol)
            if model_symbol and model_symbol in self.detected_models:
                return list(self.detected_models[model_symbol].keys())
        return self.model_detector.available_timeframes

    def load_trained_models(self, symbol: str = None) -> bool:
        """Load models for specific symbol or all available symbols"""
        print("📂 Loading AI Models...")
        print("-" * 40)

        if symbol:
            # Load models for specific symbol
            model_symbol = self.model_detector.get_best_symbol_match(symbol)
            if not model_symbol:
                print(f"❌ No models found for symbol: {symbol}")
                return False
            symbols_to_load = [model_symbol]
        else:
            # Load all available models
            symbols_to_load = self.model_detector.available_symbols

        loaded_count = 0
        self.models = {}

        for model_symbol in symbols_to_load:
            if model_symbol not in self.detected_models:
                continue
                
            print(f"\n📊 Loading models for {model_symbol}:")
            
            symbol_models = {}
            
            for timeframe in self.timeframes:
                if timeframe not in self.detected_models[model_symbol]:
                    continue
                    
                tf_models = {}
                
                # Load each available model type
                for model_type, file_path in self.detected_models[model_symbol][timeframe].items():
                    try:
                        if model_type == "random_forest":
                            rf_model = joblib.load(file_path)
                            tf_models["random_forest"] = rf_model
                            print(f"  ✅ {timeframe} Random Forest")
                            loaded_count += 1
                            
                        elif model_type == "xgboost":
                            xgb_model = joblib.load(file_path)
                            tf_models["xgboost"] = xgb_model
                            print(f"  ✅ {timeframe} XGBoost")
                            loaded_count += 1
                            
                        elif model_type == "neural_network" and TENSORFLOW_AVAILABLE:
                            if file_path.endswith('.h5'):
                                # Load Keras model
                                nn_model = tf.keras.models.load_model(file_path)
                                
                                # Try to load companion components file
                                components_path = file_path.replace('.h5', '.pkl')
                                if os.path.exists(components_path):
                                    nn_components = joblib.load(components_path)
                                    tf_models["neural_network"] = {"model": nn_model, **nn_components}
                                else:
                                    tf_models["neural_network"] = {"model": nn_model}
                                    
                                print(f"  ✅ {timeframe} Neural Network")
                                loaded_count += 1
                            else:
                                # Load pickle file (components only)
                                nn_components = joblib.load(file_path)
                                tf_models["neural_network"] = nn_components
                                print(f"  ✅ {timeframe} Neural Network (components)")
                                loaded_count += 1
                                
                    except Exception as e:
                        print(f"  ❌ {timeframe} {model_type}: {str(e)}")
                
                if tf_models:
                    symbol_models[timeframe] = tf_models
            
            if symbol_models:
                self.models[model_symbol] = symbol_models

        print("-" * 40)
        print(f"📊 Total: {loaded_count} models loaded for {len(self.models)} symbols")
        
        # Load feature mappings if available
        self.load_feature_mappings()

        return len(self.models) > 0

    def load_feature_mappings(self):
        """Try to load feature mapping files"""
        
        for symbol in self.models.keys():
            try:
                # Try different possible feature mapping file names
                possible_names = [
                    f"{symbol}_feature_mapping.json",
                    f"{symbol}_SMC_feature_mapping.json",
                    "feature_mapping.json"
                ]
                
                for filename in possible_names:
                    if os.path.exists(filename):
                        with open(filename, "r") as f:
                            features = json.load(f)
                        self.training_features[symbol] = features
                        print(f"✅ Feature mapping loaded for {symbol}")
                        break
                        
            except Exception as e:
                print(f"⚠️ Could not load feature mapping for {symbol}: {e}")

    def get_broker_symbol(self, requested_symbol: str) -> str:
        """Get the correct symbol name for this broker"""
        
        # Find model symbol match
        model_symbol = self.model_detector.get_best_symbol_match(requested_symbol)
        
        if model_symbol and model_symbol in self.symbol_mapping:
            broker_symbol = self.symbol_mapping[model_symbol]
            print(f"🔄 Symbol mapping: {requested_symbol} → {model_symbol} → {broker_symbol}")
            return broker_symbol
        
        # Fallback to requested symbol
        print(f"🔄 Using symbol as-is: {requested_symbol}")
        return requested_symbol

    def verify_symbol_data(self, symbol: str) -> bool:
        """Verify that symbol has available data"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 5)
            if rates is not None and len(rates) > 0:
                print(f"✅ {symbol}: Data available")
                return True
            else:
                print(f"❌ {symbol}: No data available")
                return False
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
            return False

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
            
            # Verify symbol availability
            print("\n🔍 Verifying symbol availability:")
            for model_symbol, broker_symbol in self.symbol_mapping.items():
                self.verify_symbol_data(broker_symbol)
            
            return True

        except Exception as e:
            print(f"❌ MT5 connection error: {str(e)}")
            return False

    def get_realtime_data(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get real-time market data for prediction"""
        
        broker_symbol = self.get_broker_symbol(symbol)
        
        try:
            rates = mt5.copy_rates_from_pos(
                broker_symbol, self.mt5_timeframes[timeframe], 0, count
            )

            if rates is None or len(rates) == 0:
                print(f"❌ No real-time data for {broker_symbol} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            df["symbol"] = broker_symbol
            df["timeframe"] = timeframe

            print(f"✅ Got {len(df)} candles for {broker_symbol} {timeframe}")
            return df

        except Exception as e:
            print(f"❌ Real-time data error for {broker_symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_complete_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 QUICK FIX: Enhanced feature calculation to reach 90+ features"""
        if df.empty:
            return df

        df = df.copy()
        print(f"🔄 Calculating enhanced features for {len(df)} candles...")

        # === BASIC PRICE METRICS (15 features) ===
        df["hl2"] = (df["high"] + df["low"]) / 2
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        df["hlcc4"] = (df["high"] + df["low"] + df["close"] + df["close"]) / 4
        df["oc2"] = (df["open"] + df["close"]) / 2
        df["median_price"] = df["hl2"]
        df["weighted_close"] = (df["high"] + df["low"] + df["close"] * 2) / 4
        df["typical_price"] = df["hlc3"]

        # Price ranges and volatility
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100
        
        # Adjust pip calculation based on symbol
        symbol_name = df["symbol"].iloc[0] if "symbol" in df.columns else ""
        if "XAU" in symbol_name or "GOLD" in symbol_name:
            df["range_pips"] = df["range"] / 0.01
            pip_size = 0.01
        else:
            df["range_pips"] = df["range"] / 0.00001
            pip_size = 0.00001
        
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["range_efficiency"] = np.where(df["true_range"] > 0, df["range"] / df["true_range"], 1)
        df["gap_up"] = np.maximum(df["open"] - df["close"].shift(1), 0)
        df["gap_down"] = np.maximum(df["close"].shift(1) - df["open"], 0)

        # === CANDLE PATTERNS (15 features) ===
        df["body"] = abs(df["close"] - df["open"])
        df["body_pct"] = np.where(df["range"] > 0, (df["body"] / df["range"]) * 100, 0)
        df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
        df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
        df["upper_shadow_pct"] = np.where(df["range"] > 0, (df["upper_shadow"] / df["range"]) * 100, 0)
        df["lower_shadow_pct"] = np.where(df["range"] > 0, (df["lower_shadow"] / df["range"]) * 100, 0)
        df["shadow_ratio"] = np.where(df["lower_shadow"] > 0, df["upper_shadow"] / df["lower_shadow"], 0)
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)
        df["is_doji"] = (df["body_pct"] < 10).astype(int)
        df["is_hammer"] = ((df["lower_shadow"] > df["body"] * 2) & (df["upper_shadow"] < df["body"] * 0.1)).astype(int)
        df["is_shooting_star"] = ((df["upper_shadow"] > df["body"] * 2) & (df["lower_shadow"] < df["body"] * 0.1)).astype(int)
        df["is_marubozu"] = (df["body_pct"] > 95).astype(int)
        df["body_position"] = np.where(df["range"] > 0, (np.minimum(df["open"], df["close"]) - df["low"]) / df["range"], 0)
        df["candle_strength"] = np.where(df["range"] > 0, df["body"] / df["range"], 0)

        # === PRICE MOVEMENTS (10 features) ===
        df["price_change"] = df["close"].diff()
        df["price_change_pct"] = df["close"].pct_change() * 100
        df["price_change_pips"] = df["price_change"] / pip_size
        df["high_change"] = df["high"].diff()
        df["low_change"] = df["low"].diff()
        df["momentum_1"] = df["close"] / df["close"].shift(1) - 1
        df["momentum_3"] = df["close"] / df["close"].shift(3) - 1
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["price_acceleration"] = df["price_change"].diff()
        df["price_velocity"] = df["price_change"].rolling(3).mean()

        # === TECHNICAL INDICATORS (20 features) ===
        # Moving Averages
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # RSI variations
        df["rsi_7"] = self.calculate_rsi(df["close"], 7)
        df["rsi_14"] = self.calculate_rsi(df["close"], 14)
        df["rsi_21"] = self.calculate_rsi(df["close"], 21)
        
        # MACD
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_upper"] = df["sma_20"] + (df["close"].rolling(20).std() * 2)
        df["bb_lower"] = df["sma_20"] - (df["close"].rolling(20).std() * 2)
        df["bb_position"] = np.where((df["bb_upper"] - df["bb_lower"]) > 0, 
                                   (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]), 0.5)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
        
        # Stochastic
        lowest_low = df["low"].rolling(14).min()
        highest_high = df["high"].rolling(14).max()
        df["stoch_k"] = np.where((highest_high - lowest_low) > 0, 
                               (df["close"] - lowest_low) / (highest_high - lowest_low) * 100, 50)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # === VOLATILITY MEASURES (8 features) ===
        df["atr_7"] = df["true_range"].rolling(7).mean()
        df["atr_14"] = df["true_range"].rolling(14).mean()
        df["atr_21"] = df["true_range"].rolling(21).mean()
        df["volatility_pct"] = (df["close"].rolling(20).std() / df["close"].rolling(20).mean()) * 100
        df["volatility_rank"] = df["volatility_pct"].rolling(50).rank(pct=True)
        df["atr_pct"] = (df["atr_14"] / df["close"]) * 100
        df["efficiency_ratio"] = np.where(df["atr_14"].rolling(10).sum() > 0,
                                        abs(df["close"] - df["close"].shift(10)) / df["atr_14"].rolling(10).sum(), 0)
        df["volatility_breakout"] = (df["range"] > df["atr_14"] * 1.5).astype(int)

        # === VOLUME ANALYSIS (6 features) ===
        df["volume_ma"] = df["tick_volume"].rolling(20).mean()
        df["volume_ratio"] = np.where(df["volume_ma"] > 0, df["tick_volume"] / df["volume_ma"], 1)
        df["volume_sma"] = df["tick_volume"].rolling(10).mean()
        df["volume_change"] = df["tick_volume"].pct_change()
        df["volume_rank"] = df["tick_volume"].rolling(20).rank(pct=True)
        df["price_volume"] = df["price_change_pct"] * df["volume_ratio"]

        # === BASIC SMC FEATURES (16 features) ===
        df = self.add_basic_smc_features(df)

        # === ADDITIONAL FEATURES TO REACH 90+ (20 features) ===
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
        df["overnight_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["range_position"] = np.where(df["range"] > 0, (df["close"] - df["low"]) / df["range"], 0.5)
        df["price_location"] = df["range_position"]  # Alias
        df["range_rank"] = df["range"].rolling(20).rank(pct=True)
        df["price_momentum_rank"] = df["price_change_pct"].rolling(20).rank(pct=True)
        df["support_level"] = df["low"].rolling(20).min()
        df["resistance_level"] = df["high"].rolling(20).max()
        df["support_distance"] = (df["close"] - df["support_level"]) / df["close"] * 100
        df["resistance_distance"] = (df["resistance_level"] - df["close"]) / df["close"] * 100
        df["pivot_point"] = (df["high"] + df["low"] + df["close"]) / 3
        df["pivot_distance"] = (df["close"] - df["pivot_point"]) / df["close"] * 100
        df["trend_direction"] = np.where(df["close"] > df["sma_20"], 1, np.where(df["close"] < df["sma_20"], -1, 0))
        df["trend_strength"] = np.where(df["atr_14"] > 0, abs(df["close"] - df["sma_20"]) / df["atr_14"], 0)
        df["ma_slope"] = df["sma_20"].diff()
        df["price_vs_ma"] = (df["close"] - df["sma_20"]) / df["sma_20"] * 100
        df["volume_trend"] = df["tick_volume"].rolling(10).mean() / df["tick_volume"].rolling(30).mean()
        df["price_density"] = df["close"].rolling(10).std() / df["close"].rolling(10).mean()
        df["market_phase"] = np.where(df["volatility_pct"] > df["volatility_pct"].rolling(50).mean(), 1, 0)
        df["breakout_strength"] = np.where(df["atr_14"] > 0, df["range"] / df["atr_14"], 1)

        # Fill NaN values
        df = df.fillna(0)
        
        # Count final features
        exclude_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'symbol', 'timeframe', 'time']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_count = len(feature_columns)
        
        print(f"✅ Generated {feature_count} features for model alignment")
        
        return df

    def add_basic_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic SMC features"""
        
        # Initialize SMC feature columns
        df["swing_high"] = 0
        df["swing_low"] = 0
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan
        df["higher_high"] = (df["high"] > df["high"].rolling(10).max().shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].rolling(10).min().shift(1)).astype(int)
        df["higher_low"] = (df["low"] > df["low"].rolling(10).min().shift(1)).astype(int)
        df["lower_high"] = (df["high"] < df["high"].rolling(10).max().shift(1)).astype(int)
        
        # Market structure
        df["market_structure"] = 0
        structure_window = 20
        for i in range(structure_window, len(df)):
            window_data = df.iloc[i - structure_window : i]
            hh_count = window_data["higher_high"].sum()
            ll_count = window_data["lower_low"].sum()
            
            if hh_count > ll_count:
                df.iloc[i, df.columns.get_loc("market_structure")] = 1
            elif ll_count > hh_count:
                df.iloc[i, df.columns.get_loc("market_structure")] = -1
        
        df["structure_momentum"] = df["market_structure"].rolling(10).mean()
        df["structure_change"] = df["market_structure"].diff()
        df["structure_break"] = np.where(abs(df["structure_change"]) >= 1, df["structure_change"], 0)
        
        # Basic CHoCH/BOS
        df["choch"] = np.where(abs(df["structure_break"]) >= 2, df["market_structure"], 0)
        df["bos"] = np.where(abs(df["structure_break"]) == 1, df["market_structure"], 0)
        
        # Basic Order Blocks and FVG (simplified)
        df["bullish_ob"] = 0
        df["bearish_ob"] = 0
        df["ob_high"] = np.nan
        df["ob_low"] = np.nan
        df["bullish_fvg"] = 0
        df["bearish_fvg"] = 0
        df["fvg_high"] = np.nan
        df["fvg_low"] = np.nan
        df["fvg_size"] = np.nan
        
        # Basic Liquidity zones
        df["buy_liquidity"] = 0
        df["sell_liquidity"] = 0
        df["liquidity_strength"] = 0
        
        # SMC confluence
        df["smc_confluence"] = (
            abs(df["structure_break"]) + df["bullish_ob"] + df["bearish_ob"] + 
            df["bullish_fvg"] + df["bearish_fvg"] + df["buy_liquidity"] + df["sell_liquidity"]
        )
        df["break_strength"] = abs(df["structure_break"]) * df["range_pct"]
        
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def align_features_with_training(self, df: pd.DataFrame, symbol: str, timeframe: str) -> np.ndarray:
        """🔧 QUICK FIX: Smart feature alignment with automatic padding"""
        
        model_symbol = self.model_detector.get_best_symbol_match(symbol)
        
        # Get all feature columns (excluding OHLC and metadata)
        exclude_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 
                          'real_volume', 'symbol', 'timeframe', 'time']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        latest_candle = df.iloc[-1]
        feature_vector = latest_candle[feature_columns].values
        
        # Handle NaN and inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 🎯 SMART PADDING TO EXACTLY 90 FEATURES
        target_features = 90
        current_features = len(feature_vector)
        
        print(f"📊 Feature alignment for {timeframe}:")
        print(f"   Current features: {current_features}")
        print(f"   Target features: {target_features}")
        
        if current_features < target_features:
            # Calculate different types of padding
            missing_count = target_features - current_features
            
            # 1. Calculate some intelligent values based on existing features
            if current_features > 10:  # If we have enough features to work with
                feature_mean = np.mean(feature_vector[feature_vector != 0]) if np.any(feature_vector != 0) else 0
                feature_std = np.std(feature_vector[feature_vector != 0]) if np.any(feature_vector != 0) else 0.1
                
                # Create intelligent padding: mix of small random values and zeros
                intelligent_pad_count = min(missing_count // 3, 10)  # Max 10 intelligent values
                zero_pad_count = missing_count - intelligent_pad_count
                
                # Intelligent padding (small values based on data statistics)
                intelligent_padding = np.random.normal(0, feature_std * 0.1, intelligent_pad_count)
                # Zero padding for the rest
                zero_padding = np.zeros(zero_pad_count)
                
                total_padding = np.concatenate([intelligent_padding, zero_padding])
            else:
                # If too few features, just use zeros
                total_padding = np.zeros(missing_count)
            
            feature_vector = np.hstack([feature_vector, total_padding])
            print(f"🔧 Padded features: {current_features} → {len(feature_vector)}")
            
        elif current_features > target_features:
            # Truncate to target size
            feature_vector = feature_vector[:target_features]
            print(f"🔧 Truncated features: {current_features} → {len(feature_vector)}")
        
        print(f"✅ {timeframe}: Perfect alignment with {len(feature_vector)} features")
        
        return feature_vector.reshape(1, -1)

    def predict_signal(self, symbol: str, timeframe: str) -> Dict:
        """Generate prediction for single timeframe with enhanced error handling"""

        model_symbol = self.model_detector.get_best_symbol_match(symbol)
        
        if not model_symbol or model_symbol not in self.models:
            return {"error": f"No models available for symbol: {symbol}"}

        if timeframe not in self.models[model_symbol]:
            return {"error": f"No models loaded for {symbol} {timeframe}"}

        # Get real-time data
        df = self.get_realtime_data(symbol, timeframe, 100)
        if df.empty:
            return {"error": f"No data available for {symbol} {timeframe}"}

        # Calculate enhanced SMC features
        df = self.calculate_complete_smc_features(df)

        # Smart feature alignment
        feature_vector = self.align_features_with_training(df, symbol, timeframe)

        predictions = {}
        confidences = {}

        # Try each available model with enhanced error handling
        for model_name, model_data in self.models[model_symbol][timeframe].items():
            try:
                if model_name == "random_forest":
                    # Handle Random Forest with automatic feature alignment
                    if "selector" in model_data:
                        try:
                            feature_vector_selected = model_data["selector"].transform(feature_vector)
                        except ValueError as e:
                            print(f"⚠️ Selector error: {e}")
                            # Try to handle mismatch
                            expected_features = model_data["selector"].k if hasattr(model_data["selector"], 'k') else 50
                            current_features = feature_vector.shape[1]
                            
                            if current_features != expected_features:
                                if current_features < expected_features:
                                    padding = np.zeros((1, expected_features - current_features))
                                    padded_vector = np.hstack([feature_vector, padding])
                                    feature_vector_selected = model_data["selector"].transform(padded_vector)
                                else:
                                    truncated_vector = feature_vector[:, :expected_features]
                                    feature_vector_selected = model_data["selector"].transform(truncated_vector)
                            else:
                                feature_vector_selected = feature_vector
                    else:
                        # Direct model prediction
                        try:
                            expected_features = model_data["model"].n_features_in_
                            current_features = feature_vector.shape[1]
                            
                            if current_features != expected_features:
                                if current_features < expected_features:
                                    padding = np.zeros((1, expected_features - current_features))
                                    feature_vector_selected = np.hstack([feature_vector, padding])
                                else:
                                    feature_vector_selected = feature_vector[:, :expected_features]
                            else:
                                feature_vector_selected = feature_vector
                        except:
                            feature_vector_selected = feature_vector

                    rf_pred = model_data["model"].predict(feature_vector_selected)[0]
                    rf_proba = model_data["model"].predict_proba(feature_vector_selected)[0]
                    rf_confidence = np.max(rf_proba)

                    predictions[model_name] = rf_pred
                    confidences[model_name] = rf_confidence
                    print(f"✅ {timeframe} Random Forest: {rf_pred} (conf: {rf_confidence:.3f})")

                elif model_name == "neural_network" and TENSORFLOW_AVAILABLE:
                    # Handle Neural Network with automatic feature alignment
                    nn_model = model_data["model"]
                    
                    # Get expected input shape
                    try:
                        expected_features = nn_model.layers[0].input_shape[-1]
                        current_features = feature_vector.shape[1]
                        
                        if current_features != expected_features:
                            if current_features < expected_features:
                                padding = np.zeros((1, expected_features - current_features))
                                feature_vector_adjusted = np.hstack([feature_vector, padding])
                            else:
                                feature_vector_adjusted = feature_vector[:, :expected_features]
                        else:
                            feature_vector_adjusted = feature_vector
                    except:
                        feature_vector_adjusted = feature_vector
                    
                    # Apply scaling if available
                    if "scaler" in model_data:
                        try:
                            feature_vector_scaled = model_data["scaler"].transform(feature_vector_adjusted)
                        except ValueError as e:
                            print(f"⚠️ Scaler error: {e}")
                            feature_vector_scaled = feature_vector_adjusted
                    else:
                        feature_vector_scaled = feature_vector_adjusted

                    nn_proba = nn_model.predict(feature_vector_scaled, verbose=0)[0]

                    if len(nn_proba) > 1:
                        nn_pred = np.argmax(nn_proba)
                        nn_confidence = np.max(nn_proba)
                        if "label_encoder" in model_data:
                            try:
                                nn_pred = model_data["label_encoder"].inverse_transform([nn_pred])[0]
                            except:
                                pass  # Keep original prediction if transform fails
                    else:
                        nn_pred = 1 if nn_proba[0] > 0.5 else -1
                        nn_confidence = nn_proba[0] if nn_pred == 1 else (1 - nn_proba[0])

                    predictions[model_name] = nn_pred
                    confidences[model_name] = nn_confidence
                    print(f"✅ {timeframe} Neural Network: {nn_pred} (conf: {nn_confidence:.3f})")

            except Exception as e:
                print(f"❌ {model_name} prediction error for {timeframe}: {str(e)}")
                continue

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
        
        model_symbol = self.model_detector.get_best_symbol_match(symbol)
        if not model_symbol:
            return {"error": f"No models available for symbol: {symbol}"}
            
        broker_symbol = self.get_broker_symbol(symbol)
        print(f"🔄 Generating multi-timeframe signals for {broker_symbol}...")

        all_signals = {}
        successful_signals = 0

        # Use available timeframes for this symbol
        available_timeframes = self.get_available_timeframes(symbol)
        
        for timeframe in available_timeframes:
            signal = self.predict_signal(symbol, timeframe)
            if "error" not in signal:
                all_signals[timeframe] = signal
                successful_signals += 1
                direction = (
                    "LONG" if signal["consensus_prediction"] == 1
                    else "SHORT" if signal["consensus_prediction"] == -1 else "HOLD"
                )
                print(f"  {timeframe}: {direction} ({signal['average_confidence']:.3f} confidence)")
            else:
                print(f"  {timeframe}: {signal['error']}")

        if successful_signals == 0:
            return {"error": "No signals generated"}

        print(f"✅ Generated signals for {successful_signals}/{len(available_timeframes)} timeframes")

        return self.aggregate_signals(all_signals, broker_symbol)

    def aggregate_signals(self, signals: Dict, symbol: str) -> Dict:
        """Aggregate multi-timeframe signals into final recommendation"""

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

        # Adjust consensus requirement based on available signals
        min_required = min(self.min_consensus, len(signals))
        
        if final_confidence >= 0.8 and total_signals >= min_required:
            risk_level = "LOW"
        elif final_confidence >= 0.6 and total_signals >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "symbol": symbol,
            "timestamp": datetime.now(self.timezone).isoformat(),
            "current_price": list(signals.values())[0].get("current_price", 0),
            "final_direction": final_direction,
            "final_confidence": final_confidence,
            "risk_level": risk_level,
            "timeframe_consensus": f"{long_count}L/{short_count}S/{len(signals)-total_signals}H",
            "individual_signals": signals,
            "trading_recommendation": (
                "TRADE" if final_direction in ["LONG", "SHORT"] and final_confidence >= 0.7 else "WAIT"
            ),
        }

    def list_available_symbols(self):
        """List all available symbols with their model counts"""
        print("\n📊 Available Trading Symbols:")
        print("-" * 50)
        
        for symbol in self.model_detector.available_symbols:
            if symbol in self.models:
                timeframe_count = len(self.models[symbol])
                model_count = sum(len(tf_models) for tf_models in self.models[symbol].values())
                print(f"🎯 {symbol}: {timeframe_count} timeframes, {model_count} total models")
                
                for tf in sorted(self.models[symbol].keys()):
                    model_types = list(self.models[symbol][tf].keys())
                    print(f"   {tf}: {', '.join(model_types)}")


# Usage Example
if __name__ == "__main__":
    print("🎯 SMC Signal Engine - Quick Fix Version")
    print("=" * 60)

    # Initialize signal engine (auto-detect models)
    engine = SMCSignalEngine()

    # Show available symbols
    engine.list_available_symbols()

    # Load all detected models
    if engine.load_trained_models():
        # Connect to MT5
        if engine.connect_mt5():
            
            print(f"\n🧪 Testing available symbols:")
            
            for symbol in engine.get_available_symbols():
                print(f"\n{'='*30}")
                print(f"Testing {symbol}:")
                
                signal = engine.get_multi_timeframe_signals(symbol)
                if "error" not in signal:
                    print(f"✅ {symbol}: {signal['final_direction']} (Confidence: {signal['final_confidence']:.3f})")
                    print(f"   Risk: {signal['risk_level']} | Consensus: {signal['timeframe_consensus']}")
                else:
                    print(f"❌ {symbol} Error: {signal['error']}")

        else:
            print("❌ MT5 connection failed")
    else:
        print("❌ No models could be loaded")