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

    Supports:
    - Random Forest (Ensemble)
    - XGBoost (Gradient Boosting)
    - Neural Networks (Deep Learning)
    - LSTM (Sequential Learning)
    - Multi-Timeframe Models
    """

    def __init__(self):
        """Initialize SMC AI Trainer"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

        # Model configurations
        self.random_forest_config = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        self.xgboost_config = {
            "n_estimators": [100, 200, 300],
            "max_depth": [6, 10, 15],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
        }

        # Feature selection settings
        self.feature_selection_k = 50  # Top K features to select

        print("🚀 SMC AI Training System Initialized")
        print("🎯 Ready for Multi-Model Training")

    def load_labeled_dataset(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load complete labeled dataset"""
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        labeled_data = {}

        print("📂 Loading Labeled Training Dataset...")
        print("-" * 40)

        for tf in timeframes:
            try:
                filename = f"{base_filename}_labeled_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                labeled_data[tf] = df

                # Quick stats
                signals = (df["direction_label"] != 0).sum()
                win_rate = (
                    (df["trade_outcome"] == 1).sum()
                    / max(1, (df["trade_outcome"] != 0).sum())
                    * 100
                )

                print(
                    f"✅ {tf:>3}: {len(df):,} candles, {signals:,} signals, {win_rate:.1f}% win rate"
                )
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
        """
        # Select only trading signals (non-zero labels)
        signal_data = df[df[label_type] != 0].copy()

        if len(signal_data) == 0:
            print(f"❌ No signals found for {label_type}")
            return np.array([]), np.array([]), []

        # SMC feature columns (exclude basic OHLC and metadata)
        smc_features = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "swing",
                    "structure",
                    "choch",
                    "bos",
                    "ob",
                    "fvg",
                    "liquidity",
                    "smc",
                    "atr",
                    "volume",
                    "range",
                    "body",
                    "shadow",
                    "higher",
                    "lower",
                    "bullish",
                    "bearish",
                ]
            )
        ]

        # Technical indicator features
        tech_features = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["sma", "ema", "rsi", "macd", "bb", "stoch"]
            )
        ]

        # Price action features
        price_features = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["change", "momentum", "volatility", "confluence"]
            )
        ]

        # Combine all relevant features
        feature_columns = list(set(smc_features + tech_features + price_features))

        # Remove non-numeric columns and labels
        exclude_columns = [
            "symbol",
            "timeframe",
            "entry_reason",
            "exit_reason",
            "direction_label",
            "signal_quality",
            "outcome_label",
            "risk_adjusted_label",
        ]
        feature_columns = [
            col
            for col in feature_columns
            if col in df.columns and col not in exclude_columns
        ]

        # Prepare feature matrix
        X = signal_data[feature_columns].values
        y = signal_data[label_type].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        print(f"📊 Prepared {len(X)} samples with {len(feature_columns)} features")
        print(
            f"🎯 Label distribution: {np.bincount(y + 1)}"
        )  # +1 to handle negative labels

        return X, y, feature_columns

    def train_random_forest(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train Random Forest model with hyperparameter tuning
        """
        print(f"🌲 Training Random Forest for {timeframe}...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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

        # Grid search for best parameters
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, self.random_forest_config, cv=3, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train_selected, y_train)

        # Best model
        best_rf = grid_search.best_estimator_

        # Train and evaluate
        y_pred = best_rf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(best_rf, X_train_selected, y_train, cv=5)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": selected_features, "importance": best_rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        results = {
            "model": best_rf,
            "selector": selector,
            "accuracy": accuracy,
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std(),
            "best_params": grid_search.best_params_,
            "feature_importance": importance_df,
            "selected_features": selected_features,
            "classification_report": classification_report(y_test, y_pred),
        }

        print(
            f"✅ Random Forest {timeframe}: {accuracy:.3f} accuracy, {cv_scores.mean():.3f}±{cv_scores.std():.3f} CV"
        )

        return results

    def train_xgboost(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train XGBoost model with hyperparameter tuning
        """
        if not XGBOOST_AVAILABLE:
            print(f"⚠️ Skipping XGBoost for {timeframe} - XGBoost not available")
            return {}

        print(f"🚀 Training XGBoost for {timeframe}...")

        # Encode labels for XGBoost (needs 0-based labels)
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

        # Grid search for best parameters
        xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        grid_search = GridSearchCV(
            xgb, self.xgboost_config, cv=3, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train_selected, y_train)

        # Best model
        best_xgb = grid_search.best_estimator_

        # Train and evaluate
        y_pred = best_xgb.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(best_xgb, X_train_selected, y_train, cv=5)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": selected_features, "importance": best_xgb.feature_importances_}
        ).sort_values("importance", ascending=False)

        results = {
            "model": best_xgb,
            "selector": selector,
            "label_encoder": le,
            "accuracy": accuracy,
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std(),
            "best_params": grid_search.best_params_,
            "feature_importance": importance_df,
            "selected_features": selected_features,
            "classification_report": classification_report(y_test, y_pred),
        }

        print(
            f"✅ XGBoost {timeframe}: {accuracy:.3f} accuracy, {cv_scores.mean():.3f}±{cv_scores.std():.3f} CV"
        )

        return results

    def train_neural_network(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], timeframe: str
    ) -> Dict:
        """
        Train Neural Network model for SMC trading
        """
        if not TENSORFLOW_AVAILABLE:
            print(
                f"⚠️ Skipping Neural Network for {timeframe} - TensorFlow not available"
            )
            return {}

        print(f"🧠 Training Neural Network for {timeframe}...")

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

        # Build neural network
        model = Sequential(
            [
                Dense(256, activation="relu", input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.3),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(
                    n_classes if n_classes > 2 else 1,
                    activation="softmax" if n_classes > 2 else "sigmoid",
                ),
            ]
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy" if n_classes > 2 else "binary_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7
        )

        # Train model
        history = model.fit(
            X_train_scaled,
            y_train_cat,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        )

        # Evaluate model
        if n_classes > 2:
            test_loss, test_accuracy = model.evaluate(
                X_test_scaled, y_test_cat, verbose=0
            )
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            test_loss, test_accuracy = model.evaluate(
                X_test_scaled, y_test_cat, verbose=0
            )
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        results = {
            "model": model,
            "scaler": scaler,
            "label_encoder": le,
            "accuracy": test_accuracy,
            "test_loss": test_loss,
            "history": history.history,
            "n_classes": n_classes,
            "classification_report": classification_report(y_test, y_pred),
        }

        print(
            f"✅ Neural Network {timeframe}: {test_accuracy:.3f} accuracy, {test_loss:.3f} loss"
        )

        return results

    def train_single_timeframe(
        self, df: pd.DataFrame, timeframe: str, models_to_train: List[str] = None
    ) -> Dict:
        """
        Train all models for a single timeframe
        """
        if models_to_train is None:
            models_to_train = ["random_forest", "xgboost", "neural_network"]

        print(f"\n🎯 Training AI Models for {timeframe}")
        print("-" * 40)

        results = {}

        # Prepare data
        X, y, feature_names = self.prepare_features_labels(df, "direction_label")

        if len(X) == 0:
            print(f"❌ No training data available for {timeframe}")
            return {}

        # Train Random Forest
        if "random_forest" in models_to_train:
            try:
                rf_results = self.train_random_forest(X, y, feature_names, timeframe)
                results["random_forest"] = rf_results
            except Exception as e:
                print(f"❌ Random Forest training failed: {str(e)}")

        # Train XGBoost
        if "xgboost" in models_to_train and XGBOOST_AVAILABLE:
            try:
                xgb_results = self.train_xgboost(X, y, feature_names, timeframe)
                results["xgboost"] = xgb_results
            except Exception as e:
                print(f"❌ XGBoost training failed: {str(e)}")

        # Train Neural Network
        if "neural_network" in models_to_train and TENSORFLOW_AVAILABLE:
            try:
                nn_results = self.train_neural_network(X, y, feature_names, timeframe)
                results["neural_network"] = nn_results
            except Exception as e:
                print(f"❌ Neural Network training failed: {str(e)}")

        return results

    def train_all_timeframes(
        self, labeled_data: Dict[str, pd.DataFrame], models_to_train: List[str] = None
    ) -> Dict:
        """
        Train models for all timeframes
        """
        print("🚀 Training SMC AI Models - All Timeframes")
        print("=" * 60)

        all_results = {}

        for timeframe, df in labeled_data.items():
            timeframe_results = self.train_single_timeframe(
                df, timeframe, models_to_train
            )
            all_results[timeframe] = timeframe_results

        # Summary
        print("\n" + "=" * 60)
        print("📋 TRAINING SUMMARY")
        print("=" * 60)

        for timeframe, tf_results in all_results.items():
            print(f"\n{timeframe} Models:")
            for model_name, model_results in tf_results.items():
                if "accuracy" in model_results:
                    acc = model_results["accuracy"]
                    print(f"  {model_name:>15}: {acc:.3f} accuracy")

        self.models = all_results
        print("\n🎉 All Models Training Complete!")

        return all_results

    def save_models(self, base_filename: str) -> bool:
        """
        Save all trained models
        """
        try:
            print(f"\n💾 Saving Trained Models...")
            print("-" * 40)

            saved_files = []

            for timeframe, tf_models in self.models.items():
                for model_name, model_data in tf_models.items():
                    if not model_data:
                        continue

                    # Save model
                    model_filename = (
                        f"{base_filename}_{timeframe}_{model_name}_model.pkl"
                    )

                    if model_name == "neural_network" and TENSORFLOW_AVAILABLE:
                        # Save Keras model separately
                        keras_filename = (
                            f"{base_filename}_{timeframe}_{model_name}_model.h5"
                        )
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

            print(f"\n✅ {len(saved_files)} model files saved!")

            return True

        except Exception as e:
            print(f"❌ Save error: {str(e)}")
            return False

    def generate_performance_report(self, base_filename: str) -> bool:
        """
        Generate comprehensive performance report
        """
        try:
            print(f"\n📊 Generating Performance Report...")

            report_data = {
                "training_date": pd.Timestamp.now().isoformat(),
                "timeframes": list(self.models.keys()),
                "model_types": [],
                "performance_summary": {},
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

                    report_data["performance_summary"][timeframe][
                        model_name
                    ] = perf_data

            # Save report
            import json

            report_filename = f"{base_filename}_performance_report.json"
            with open(report_filename, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            print(f"✅ Performance report: {report_filename}")

            return True

        except Exception as e:
            print(f"❌ Report generation error: {str(e)}")
            return False


# Usage Example
if __name__ == "__main__":
    print("🚀 SMC AI Training System")
    print("=" * 50)

    # Initialize trainer
    trainer = SMCAITrainer()

    # Load labeled dataset
    print("\n📂 Loading Labeled Dataset...")
    labeled_data = trainer.load_labeled_dataset("EURUSD_c")

    if labeled_data:
        # Train all models
        print("\n🎯 Starting AI Training...")
        models_to_train = [
            "random_forest"
        ]  # Start with Random Forest (always available)

        if XGBOOST_AVAILABLE:
            models_to_train.append("xgboost")

        if TENSORFLOW_AVAILABLE:
            models_to_train.append("neural_network")

        all_results = trainer.train_all_timeframes(labeled_data, models_to_train)

        # Save models
        trainer.save_models("EURUSD_c_SMC")

        # Generate report
        trainer.generate_performance_report("EURUSD_c_SMC")

        print("\n🎉 SMC AI Training Pipeline Complete!")
        print("=" * 50)
        print("✅ 1. Data Extracted from MT5")
        print("✅ 2. SMC Features Engineered")
        print("✅ 3. Trading Labels Created")
        print("✅ 4. AI Models Trained")
        print("🎯 5. Next: Live Trading Implementation")

    else:
        print("❌ No labeled data loaded. Please run label creation first.")
