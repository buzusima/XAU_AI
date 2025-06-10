import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# XGBoost (separate package)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense,
        LSTM,
        Dropout,
        Conv1D,
        MaxPooling1D,
        Flatten,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Deep Learning models will be skipped.")


class SMCAITrainer:
    """
    SMC AI Training System for Multi-Model Trading AI
    Created by อาจารย์ฟินิกซ์ - Professional AI Trading Models
    🥇 Optimized for Gold (XAUUSD.c) Trading

    Supports:
    - Random Forest (Ensemble)
    - XGBoost (Gradient Boosting)
    - Neural Networks (Deep Learning)
    - LSTM (Sequential Learning)
    - Multi-Timeframe Models
    - Gold-specific optimizations
    """

    def __init__(self):
        """Initialize SMC AI Trainer with Gold optimizations"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

        # 🥇 Gold-optimized Random Forest configuration
        self.random_forest_config = {
            "n_estimators": [150, 250, 350],     # More trees for Gold complexity
            "max_depth": [15, 25, None],         # Deeper trees for Gold patterns
            "min_samples_split": [2, 3, 5],     # Conservative splitting
            "min_samples_leaf": [1, 2, 3],      # Smaller leaves for precision
            "max_features": ["sqrt", "log2", 0.8], # Feature selection options
            "class_weight": ["balanced", None]   # Handle class imbalance
        }

        # 🥇 Gold-optimized XGBoost configuration
        self.xgboost_config = {
            "n_estimators": [150, 250, 350],     # More estimators
            "max_depth": [8, 12, 16],            # Deeper for Gold complexity
            "learning_rate": [0.05, 0.1, 0.15], # Conservative learning rates
            "subsample": [0.8, 0.9, 1.0],       # Sample ratios
            "colsample_bytree": [0.8, 0.9, 1.0], # Feature sampling
            "reg_alpha": [0, 0.1, 0.5],         # L1 regularization
            "reg_lambda": [1, 1.5, 2]           # L2 regularization
        }

        # 🥇 Enhanced feature selection for Gold
        self.feature_selection_k = 60  # More features for Gold (vs 50)

        # 🥇 Gold-specific performance thresholds
        self.min_accuracy_threshold = 0.68      # Slightly higher for Gold
        self.min_cv_score = 0.65                # Cross-validation threshold
        self.max_overfitting_gap = 0.15         # Train vs validation gap

        print("🥇 SMC AI Training System Initialized for Gold")
        print("🎯 Optimized for XAUUSD.c multi-model training")

    def load_labeled_dataset(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load complete labeled dataset"""
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        labeled_data = {}

        print("📂 Loading Gold Labeled Training Dataset...")
        print("-" * 40)

        for tf in timeframes:
            try:
                filename = f"{base_filename}_labeled_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                labeled_data[tf] = df

                # 🥇 Check if Gold data and get Gold-specific stats
                is_gold = False
                if 'symbol' in df.columns and len(df) > 0:
                    symbol = df['symbol'].iloc[0]
                    is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()

                # Calculate statistics
                signals = (df["direction_label"] != 0).sum()
                trades = (df["trade_outcome"] != 0).sum() if "trade_outcome" in df.columns else 0
                
                if trades > 0:
                    win_rate = (df["trade_outcome"] == 1).sum() / trades * 100
                else:
                    win_rate = 0

                # 🥇 Gold-specific metrics
                if is_gold and "pnl_points" in df.columns:
                    avg_profit_points = df[df["trade_outcome"] == 1]["pnl_points"].mean() if (df["trade_outcome"] == 1).sum() > 0 else 0
                    avg_loss_points = df[df["trade_outcome"] == -1]["pnl_points"].mean() if (df["trade_outcome"] == -1).sum() > 0 else 0
                    total_points = df["pnl_points"].sum()
                    
                    symbol_type = "🥇 Gold"
                    print(f"✅ {tf:>3}: {len(df):,} candles, {signals:,} signals, {win_rate:.1f}% win rate ({symbol_type})")
                    print(f"    🥇 {total_points:.0f} total points | Avg Win: {avg_profit_points:.0f}pts | Avg Loss: {avg_loss_points:.0f}pts")
                else:
                    symbol_type = "📈 Forex"
                    print(f"✅ {tf:>3}: {len(df):,} candles, {signals:,} signals, {win_rate:.1f}% win rate ({symbol_type})")

            except FileNotFoundError:
                print(f"❌ {tf:>3}: File not found - {filename}")
            except Exception as e:
                print(f"❌ {tf:>3}: Error loading - {str(e)}")

        print("-" * 40)
        total_candles = sum(len(df) for df in labeled_data.values())
        total_signals = sum(
            (df["direction_label"] != 0).sum() for df in labeled_data.values()
        )
        print(
            f"📊 Total: {len(labeled_data)} timeframes | {total_candles:,} candles | {total_signals:,} signals"
        )

        return labeled_data

    def prepare_features_labels(
        self, df: pd.DataFrame, label_type: str = "direction_label"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and labels for ML training
        🥇 Enhanced for Gold with better feature selection
        """
        # Select only trading signals (non-zero labels)
        signal_data = df[df[label_type] != 0].copy()

        if len(signal_data) == 0:
            print(f"❌ No signals found for {label_type}")
            return np.array([]), np.array([]), []

        # 🥇 Enhanced SMC feature selection for Gold
        gold_smc_features = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "swing", "structure", "choch", "bos", "ob", "fvg", "liquidity", "smc",
                    "break_strength", "ob_strength", "fvg_efficiency", "liquidity_proximity",
                    "session", "confluence", "gold", "points"
                ]
            )
        ]

        # Technical indicator features
        tech_features = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["sma", "ema", "rsi", "macd", "bb", "stoch", "atr", "volatility"]
            )
        ]

        # Price action features
        price_features = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["change", "momentum", "range", "body", "shadow", 
                              "bullish", "bearish", "higher", "lower"]
            )
        ]

        # 🥇 Session and time-based features for Gold
        session_features = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["london", "us", "asian", "session", "hour", "impact"]
            )
        ]

        # Combine all relevant features
        feature_columns = list(set(gold_smc_features + tech_features + price_features + session_features))

        # Remove non-numeric columns and labels
        exclude_columns = [
            "symbol", "timeframe", "entry_reason", "exit_reason",
            "direction_label", "signal_quality", "outcome_label", 
            "risk_adjusted_label", "session_label"
        ]
        feature_columns = [
            col for col in feature_columns
            if col in df.columns and col not in exclude_columns
        ]

        # 🥇 Check if Gold data for feature prioritization
        is_gold = False
        if 'symbol' in df.columns and len(df) > 0:
            symbol = df['symbol'].iloc[0]
            is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()

        # Prepare feature matrix
        X = signal_data[feature_columns].values
        y = signal_data[label_type].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        symbol_type = "🥇 Gold" if is_gold else "📈 Forex"
        print(f"📊 Prepared {len(X)} {symbol_type} samples with {len(feature_columns)} features")
        
        # 🥇 Show label distribution with Gold context
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        print(f"🎯 Label distribution: {label_dist}")
        
        if is_gold:
            print(f"🥇 Gold-specific features included: {len([f for f in feature_columns if any(k in f.lower() for k in ['gold', 'session', 'points'])])}")

        return X, y, feature_columns

    def train_random_forest(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train Random Forest model with Gold-optimized hyperparameter tuning
        """
        print(f"🌲 Training Gold-optimized Random Forest for {timeframe}...")

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 🥇 Enhanced feature selection for Gold
        selector = SelectKBest(
            score_func=f_classif, k=min(self.feature_selection_k, X.shape[1])
        )
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_features = [
            feature_names[i] for i in selector.get_support(indices=True)
        ]

        # 🥇 Gold-optimized Grid search with more comprehensive parameters
        rf = RandomForestClassifier(
            random_state=42, 
            n_jobs=-1,
            oob_score=True  # Out-of-bag scoring
        )
        
        grid_search = GridSearchCV(
            rf, 
            self.random_forest_config, 
            cv=5,  # 5-fold CV for better reliability
            scoring="accuracy", 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_selected, y_train)

        # Best model
        best_rf = grid_search.best_estimator_

        # Train and evaluate
        y_pred = best_rf.predict(X_test_selected)
        y_pred_proba = best_rf.predict_proba(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        # 🥇 Enhanced cross-validation with multiple metrics
        cv_scores = cross_val_score(best_rf, X_train_selected, y_train, cv=5)
        
        # 🥇 Calculate additional Gold-specific metrics
        train_accuracy = best_rf.score(X_train_selected, y_train)
        overfitting_gap = train_accuracy - accuracy
        
        # Feature importance analysis
        importance_df = pd.DataFrame({
            "feature": selected_features, 
            "importance": best_rf.feature_importances_
        }).sort_values("importance", ascending=False)

        # 🥇 Identify Gold-specific important features
        gold_features = importance_df[
            importance_df['feature'].str.contains('gold|session|points|strength|efficiency|confluence', case=False, na=False)
        ]

        results = {
            "model": best_rf,
            "selector": selector,
            "accuracy": accuracy,
            "train_accuracy": train_accuracy,
            "overfitting_gap": overfitting_gap,
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std(),
            "oob_score": getattr(best_rf, 'oob_score_', None),
            "best_params": grid_search.best_params_,
            "feature_importance": importance_df,
            "selected_features": selected_features,
            "gold_features_importance": gold_features if len(gold_features) > 0 else None,
            "classification_report": classification_report(y_test, y_pred),
            "prediction_probabilities": y_pred_proba
        }

        print(f"✅ Random Forest {timeframe}: {accuracy:.3f} accuracy, {cv_scores.mean():.3f}±{cv_scores.std():.3f} CV")
        if overfitting_gap > self.max_overfitting_gap:
            print(f"⚠️ Potential overfitting detected: {overfitting_gap:.3f} gap")

        return results

    def train_xgboost(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train XGBoost model with Gold-optimized hyperparameter tuning
        """
        if not XGBOOST_AVAILABLE:
            print(f"⚠️ Skipping XGBoost for {timeframe} - XGBoost not available")
            return {}

        print(f"🚀 Training Gold-optimized XGBoost for {timeframe}...")

        # Encode labels for XGBoost
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Feature selection
        selector = SelectKBest(
            score_func=f_classif, k=min(self.feature_selection_k, X.shape[1])
        )
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_features = [
            feature_names[i] for i in selector.get_support(indices=True)
        ]

        # 🥇 Gold-optimized XGBoost with enhanced parameters
        xgb = XGBClassifier(
            random_state=42, 
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method='hist'  # Faster training
        )
        
        grid_search = GridSearchCV(
            xgb, 
            self.xgboost_config, 
            cv=5, 
            scoring="accuracy", 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_selected, y_train)

        # Best model
        best_xgb = grid_search.best_estimator_

        # Train and evaluate
        y_pred = best_xgb.predict(X_test_selected)
        y_pred_proba = best_xgb.predict_proba(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(best_xgb, X_train_selected, y_train, cv=5)
        
        # 🥇 Additional metrics
        train_accuracy = best_xgb.score(X_train_selected, y_train)
        overfitting_gap = train_accuracy - accuracy

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": selected_features, 
            "importance": best_xgb.feature_importances_
        }).sort_values("importance", ascending=False)

        # 🥇 Gold-specific feature analysis
        gold_features = importance_df[
            importance_df['feature'].str.contains('gold|session|points|strength|efficiency|confluence', case=False, na=False)
        ]

        results = {
            "model": best_xgb,
            "selector": selector,
            "label_encoder": le,
            "accuracy": accuracy,
            "train_accuracy": train_accuracy,
            "overfitting_gap": overfitting_gap,
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std(),
            "best_params": grid_search.best_params_,
            "feature_importance": importance_df,
            "selected_features": selected_features,
            "gold_features_importance": gold_features if len(gold_features) > 0 else None,
            "classification_report": classification_report(y_test, y_pred),
            "prediction_probabilities": y_pred_proba
        }

        print(f"✅ XGBoost {timeframe}: {accuracy:.3f} accuracy, {cv_scores.mean():.3f}±{cv_scores.std():.3f} CV")
        if overfitting_gap > self.max_overfitting_gap:
            print(f"⚠️ Potential overfitting detected: {overfitting_gap:.3f} gap")

        return results

    def train_neural_network(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train Neural Network model for Gold SMC trading
        """
        if not TENSORFLOW_AVAILABLE:
            print(f"⚠️ Skipping Neural Network for {timeframe} - TensorFlow not available")
            return {}

        print(f"🧠 Training Gold-optimized Neural Network for {timeframe}...")

        # Encode labels for neural network
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(np.unique(y_encoded))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to categorical if multi-class
        if n_classes > 2:
            y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)
        else:
            y_train_cat = y_train
            y_test_cat = y_test

        # 🥇 Build Gold-optimized neural network
        model = Sequential([
            Dense(512, activation="relu", input_shape=(X_train_scaled.shape[1],)),  # Larger first layer
            Dropout(0.4),  # Higher dropout for Gold volatility
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(
                n_classes if n_classes > 2 else 1,
                activation="softmax" if n_classes > 2 else "sigmoid",
            ),
        ])

        # 🥇 Compile with Gold-optimized settings
        optimizer = Adam(learning_rate=0.0005)  # Slower learning for stability
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy" if n_classes > 2 else "binary_crossentropy",
            metrics=["accuracy"],
        )

        # 🥇 Enhanced callbacks for Gold
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            patience=15,  # More patience for Gold
            restore_best_weights=True,
            min_delta=0.001
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.3,  # More aggressive LR reduction
            patience=8, 
            min_lr=1e-7
        )

        # Train model
        history = model.fit(
            X_train_scaled,
            y_train_cat,
            epochs=150,  # More epochs for Gold
            batch_size=64,  # Larger batch size
            validation_split=0.25,  # More validation data
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate model
        if n_classes > 2:
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # 🥇 Calculate training accuracy and overfitting
        if n_classes > 2:
            train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
        else:
            train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
        
        overfitting_gap = train_accuracy - test_accuracy

        results = {
            "model": model,
            "scaler": scaler,
            "label_encoder": le,
            "accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "overfitting_gap": overfitting_gap,
            "test_loss": test_loss,
            "train_loss": train_loss,
            "history": history.history,
            "n_classes": n_classes,
            "classification_report": classification_report(y_test, y_pred),
            "prediction_probabilities": y_pred_proba
        }

        print(f"✅ Neural Network {timeframe}: {test_accuracy:.3f} accuracy, {test_loss:.3f} loss")
        if overfitting_gap > self.max_overfitting_gap:
            print(f"⚠️ Potential overfitting detected: {overfitting_gap:.3f} gap")

        return results

    def train_single_timeframe(
        self, df: pd.DataFrame, timeframe: str, models_to_train: List[str] = None
    ) -> Dict:
        """
        Train all models for a single timeframe
        🥇 Enhanced for Gold with quality validation
        """
        if models_to_train is None:
            models_to_train = ["random_forest", "xgboost", "neural_network"]

        print(f"\n🎯 Training Gold AI Models for {timeframe}")
        print("-" * 40)

        results = {}

        # Prepare data
        X, y, feature_names = self.prepare_features_labels(df, "direction_label")

        if len(X) == 0:
            print(f"❌ No training data available for {timeframe}")
            return {}

        # 🥇 Check data quality for Gold
        signal_ratio = len(X) / len(df) * 100
        print(f"📊 Signal density: {signal_ratio:.1f}% ({len(X)}/{len(df)} samples)")
        
        if signal_ratio < 1.0:  # Less than 1% signals
            print(f"⚠️ Low signal density for {timeframe}, consider adjusting parameters")

        # Train Random Forest
        if "random_forest" in models_to_train:
            try:
                rf_results = self.train_random_forest(X, y, feature_names, timeframe)
                if rf_results and rf_results.get("accuracy", 0) >= self.min_accuracy_threshold:
                    results["random_forest"] = rf_results
                else:
                    print(f"⚠️ Random Forest accuracy below threshold for {timeframe}")
            except Exception as e:
                print(f"❌ Random Forest training failed: {str(e)}")

        # Train XGBoost
        if "xgboost" in models_to_train and XGBOOST_AVAILABLE:
            try:
                xgb_results = self.train_xgboost(X, y, feature_names, timeframe)
                if xgb_results and xgb_results.get("accuracy", 0) >= self.min_accuracy_threshold:
                    results["xgboost"] = xgb_results
                else:
                    print(f"⚠️ XGBoost accuracy below threshold for {timeframe}")
            except Exception as e:
                print(f"❌ XGBoost training failed: {str(e)}")

        # Train Neural Network
        if "neural_network" in models_to_train and TENSORFLOW_AVAILABLE:
            try:
                nn_results = self.train_neural_network(X, y, feature_names, timeframe)
                if nn_results and nn_results.get("accuracy", 0) >= self.min_accuracy_threshold:
                    results["neural_network"] = nn_results
                else:
                    print(f"⚠️ Neural Network accuracy below threshold for {timeframe}")
            except Exception as e:
                print(f"❌ Neural Network training failed: {str(e)}")

        return results

    def train_all_timeframes(
        self, labeled_data: Dict[str, pd.DataFrame], models_to_train: List[str] = None
    ) -> Dict:
        """
        Train models for all timeframes
        🥇 Enhanced for Gold with comprehensive validation
        """
        print("🚀 Training Gold SMC AI Models - All Timeframes")
        print("=" * 60)

        all_results = {}

        for timeframe, df in labeled_data.items():
            timeframe_results = self.train_single_timeframe(
                df, timeframe, models_to_train
            )
            all_results[timeframe] = timeframe_results

        # 🥇 Enhanced summary with Gold-specific metrics
        print("\n" + "=" * 60)
        print("📋 GOLD TRAINING SUMMARY")
        print("=" * 60)

        total_models = 0
        successful_models = 0
        accuracy_scores = []

        for timeframe, tf_results in all_results.items():
            print(f"\n🥇 {timeframe} Gold Models:")
            for model_name, model_results in tf_results.items():
                if "accuracy" in model_results:
                    acc = model_results["accuracy"]
                    cv_score = model_results.get("cv_score_mean", 0)
                    overfitting = model_results.get("overfitting_gap", 0)
                    
                    total_models += 1
                    if acc >= self.min_accuracy_threshold:
                        successful_models += 1
                        accuracy_scores.append(acc)
                    
                    print(f"  {model_name:>15}: {acc:.3f} accuracy | CV: {cv_score:.3f} | Gap: {overfitting:.3f}")
                    
                    # 🥇 Show Gold-specific feature importance
                    if "gold_features_importance" in model_results and model_results["gold_features_importance"] is not None:
                        top_gold_features = model_results["gold_features_importance"].head(3)
                        if len(top_gold_features) > 0:
                            print(f"    🥇 Top Gold features: {', '.join(top_gold_features['feature'].tolist())}")

        # 🥇 Overall performance summary
        print(f"\n📊 Overall Gold Model Performance:")
        print(f"   Successful models: {successful_models}/{total_models}")
        if accuracy_scores:
            print(f"   Average accuracy: {np.mean(accuracy_scores):.3f}")
            print(f"   Best accuracy: {np.max(accuracy_scores):.3f}")
            print(f"   Worst accuracy: {np.min(accuracy_scores):.3f}")

        success_rate = successful_models / total_models * 100 if total_models > 0 else 0
        if success_rate >= 70:
            print("🎉 Gold model training successful!")
        else:
            print("⚠️ Some Gold models need improvement")

        self.models = all_results
        print("\n🥇 All Gold Models Training Complete!")

        return all_results

    def save_models(self, base_filename: str) -> bool:
        """
        Save all trained models with Gold-specific enhancements
        """
        try:
            print(f"\n💾 Saving Gold Trained Models...")
            print("-" * 40)

            saved_files = []

            for timeframe, tf_models in self.models.items():
                for model_name, model_data in tf_models.items():
                    if not model_data:
                        continue

                    # Save model
                    model_filename = f"{base_filename}_{timeframe}_{model_name}_model.pkl"

                    if model_name == "neural_network" and TENSORFLOW_AVAILABLE:
                        # Save Keras model separately
                        keras_filename = f"{base_filename}_{timeframe}_{model_name}_model.h5"
                        model_data["model"].save(keras_filename)

                        # Save other components
                        model_components = {
                            k: v for k, v in model_data.items() if k != "model"
                        }
                        joblib.dump(model_components, model_filename)
                        saved_files.extend([keras_filename, model_filename])

                    else:
                        # Save scikit-learn models
                        joblib.dump(model_data, model_filename)
                        saved_files.append(model_filename)

                    print(f"✅ {timeframe} {model_name}: {model_filename}")

            # 🥇 Save Gold-specific feature mapping
            feature_mapping = {}
            for timeframe, tf_models in self.models.items():
                for model_name, model_data in tf_models.items():
                    if "selected_features" in model_data:
                        key = f"{timeframe}_{model_name}"
                        feature_mapping[key] = model_data["selected_features"]
                
                # Use first available model's features as timeframe default
                if timeframe not in feature_mapping:
                    for model_name, model_data in tf_models.items():
                        if "selected_features" in model_data:
                            feature_mapping[timeframe] = model_data["selected_features"]
                            break

            # Save feature mapping
            import json
            mapping_file = f"{base_filename}_feature_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(feature_mapping, f, indent=2)

            saved_files.append(mapping_file)
            print(f"✅ Feature mapping: {mapping_file}")

            print(f"\n✅ {len(saved_files)} Gold model files saved!")

            return True

        except Exception as e:
            print(f"❌ Save error: {str(e)}")
            return False

    def generate_performance_report(self, base_filename: str) -> bool:
        """
        Generate comprehensive performance report for Gold models
        """
        try:
            print(f"\n📊 Generating Gold Performance Report...")

            report_data = {
                "model_type": "Gold SMC Trading Models",
                "training_date": pd.Timestamp.now().isoformat(),
                "timeframes": list(self.models.keys()),
                "model_types": [],
                "performance_summary": {},
                "gold_optimizations": {
                    "feature_selection_k": self.feature_selection_k,
                    "min_accuracy_threshold": self.min_accuracy_threshold,
                    "min_cv_score": self.min_cv_score,
                    "max_overfitting_gap": self.max_overfitting_gap
                }
            }

            # Collect performance data
            for timeframe, tf_models in self.models.items():
                report_data["performance_summary"][timeframe] = {}

                for model_name, model_data in tf_models.items():
                    if not model_data or "accuracy" not in model_data:
                        continue

                    if model_name not in report_data["model_types"]:
                        report_data["model_types"].append(model_name)

                    perf_data = {
                        "accuracy": float(model_data["accuracy"]),
                        "train_accuracy": float(model_data.get("train_accuracy", 0)),
                        "overfitting_gap": float(model_data.get("overfitting_gap", 0)),
                        "cv_score_mean": float(model_data.get("cv_score_mean", 0)),
                        "cv_score_std": float(model_data.get("cv_score_std", 0)),
                    }

                    if "best_params" in model_data:
                        perf_data["best_params"] = model_data["best_params"]

                    if "feature_importance" in model_data:
                        # Top 10 features
                        top_features = (
                            model_data["feature_importance"].head(10).to_dict("records")
                        )
                        perf_data["top_features"] = top_features

                    # 🥇 Gold-specific feature analysis
                    if "gold_features_importance" in model_data and model_data["gold_features_importance"] is not None:
                        gold_features = model_data["gold_features_importance"].head(5).to_dict("records")
                        perf_data["top_gold_features"] = gold_features

                    # 🥇 Model quality assessment
                    quality_score = self._assess_model_quality(model_data)
                    perf_data["quality_score"] = quality_score
                    perf_data["quality_rating"] = self._get_quality_rating(quality_score)

                    report_data["performance_summary"][timeframe][model_name] = perf_data

            # 🥇 Overall Gold model statistics
            all_accuracies = []
            all_cv_scores = []
            for tf_data in report_data["performance_summary"].values():
                for model_data in tf_data.values():
                    all_accuracies.append(model_data["accuracy"])
                    all_cv_scores.append(model_data["cv_score_mean"])

            if all_accuracies:
                report_data["overall_statistics"] = {
                    "average_accuracy": float(np.mean(all_accuracies)),
                    "best_accuracy": float(np.max(all_accuracies)),
                    "worst_accuracy": float(np.min(all_accuracies)),
                    "average_cv_score": float(np.mean(all_cv_scores)),
                    "total_models": len(all_accuracies),
                    "models_above_threshold": sum(1 for acc in all_accuracies if acc >= self.min_accuracy_threshold)
                }

            # Save report
            import json
            report_filename = f"{base_filename}_performance_report.json"
            with open(report_filename, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            print(f"✅ Gold performance report: {report_filename}")

            return True

        except Exception as e:
            print(f"❌ Report generation error: {str(e)}")
            return False

    def _assess_model_quality(self, model_data: Dict) -> float:
        """🥇 Assess overall model quality for Gold trading"""
        score = 0.0
        
        # Accuracy component (40%)
        accuracy = model_data.get("accuracy", 0)
        score += (accuracy - 0.5) * 0.8  # Normalized from 0.5-1.0 to 0-0.4
        
        # Cross-validation stability (30%)
        cv_mean = model_data.get("cv_score_mean", 0)
        cv_std = model_data.get("cv_score_std", 1)
        cv_stability = cv_mean - cv_std  # Penalize high variance
        score += cv_stability * 0.3
        
        # Overfitting penalty (30%)
        overfitting_gap = model_data.get("overfitting_gap", 0)
        overfitting_penalty = max(0, overfitting_gap - 0.05) * 2  # Penalty above 5% gap
        score -= overfitting_penalty * 0.3
        
        return max(0.0, min(1.0, score))

    def _get_quality_rating(self, quality_score: float) -> str:
        """🥇 Get quality rating for Gold models"""
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.7:
            return "Good"
        elif quality_score >= 0.6:
            return "Acceptable"
        elif quality_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"


# Usage Example for Gold
if __name__ == "__main__":
    print("🥇 SMC AI Training System for Gold")
    print("=" * 50)

    # Initialize trainer with Gold optimizations
    trainer = SMCAITrainer()

    # Load Gold labeled dataset
    print("\n📂 Loading Gold Labeled Dataset...")
    labeled_data = trainer.load_labeled_dataset("XAUUSD_c")

    if labeled_data:
        # Train Gold models
        print("\n🎯 Starting Gold AI Training...")
        models_to_train = ["random_forest"]  # Start with Random Forest

        if XGBOOST_AVAILABLE:
            models_to_train.append("xgboost")

        if TENSORFLOW_AVAILABLE:
            models_to_train.append("neural_network")

        print(f"🎯 Training models: {models_to_train}")

        all_results = trainer.train_all_timeframes(labeled_data, models_to_train)

        # Save Gold models
        trainer.save_models("XAUUSD_c_SMC")

        # Generate Gold performance report
        trainer.generate_performance_report("XAUUSD_c_SMC")

        print("\n🎉 Gold SMC AI Training Pipeline Complete!")
        print("=" * 50)
        print("✅ 1. Gold Data Extracted from MT5")
        print("✅ 2. Gold SMC Features Engineered")
        print("✅ 3. Gold Trading Labels Created")
        print("✅ 4. Gold AI Models Trained")
        print("🎯 5. Next: Gold Live Trading Implementation")

    else:
        print("❌ No Gold labeled data loaded. Please run label creation first.")
        print("🔧 Expected files: XAUUSD_c_labeled_[M5|M15|H1|H4|D1].csv")