#!/usr/bin/env python3
"""
🥇 Complete Gold Trading Setup Script
อาจารย์ฟินิกซ์ - XAUUSD.c Auto Trading System

This script sets up the complete Gold trading pipeline:
1. Data Extraction from MT5
2. SMC Features Engineering
3. Trading Labels Creation
4. AI Model Training
5. Live Signal Generation
6. Automated Trading

Usage:
python gold_complete_setup.py
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import all required modules
from MT5MultiTimeframeExtractor import MT5MultiTimeframeExtractor
from SMCFeaturesEngine import SMCFeaturesEngine
from SMCLabelsEngine import SMCLabelsEngine
from SMCAITrainer import SMCAITrainer

class GoldTradingSystemSetup:
    """
    Complete Gold Trading System Setup
    """
    
    def __init__(self):
        self.base_symbol = "XAUUSD"
        self.target_symbol = None
        self.setup_config = {
            "data_extraction": True,
            "features_engineering": True,
            "labels_creation": True,
            "model_training": True,
            "performance_validation": True
        }
        
        # Gold-specific configurations
        self.gold_config = {
            "timeframes": ["M5", "M15", "H1", "H4", "D1"],
            "point_size": 0.01,
            "pip_size": 0.1,
            "min_data_quality": 75,  # Minimum data quality score
            "min_win_rate": 55,      # Minimum win rate for training
            "min_trades": 100        # Minimum trades for validation
        }
        
        print("🥇 Gold Trading System Setup Initialized")
        print(f"🎯 Target Symbol: {self.base_symbol}")
    
    def step_1_extract_gold_data(self) -> bool:
        """Step 1: Extract Gold market data from MT5"""
        print("\n" + "="*60)
        print("1️⃣ STEP 1: GOLD DATA EXTRACTION")
        print("="*60)
        
        try:
            # Initialize MT5 extractor
            extractor = MT5MultiTimeframeExtractor()
            
            # Find Gold symbol variations
            print(f"🔍 Searching for {self.base_symbol} variations...")
            variations = extractor.find_symbol_variations(self.base_symbol)
            
            if not variations:
                print(f"❌ No {self.base_symbol} symbols found")
                return False
            
            # Select best symbol (prefer .c suffix)
            self.target_symbol = None
            for var in variations:
                if ".c" in var["symbol"]:
                    self.target_symbol = var["symbol"]
                    break
            
            if self.target_symbol is None:
                self.target_symbol = variations[0]["symbol"]
            
            print(f"✅ Selected Gold symbol: {self.target_symbol}")
            
            # Extract complete dataset
            print(f"📊 Extracting complete dataset for {self.target_symbol}...")
            gold_data = extractor.get_smc_dataset(self.target_symbol)
            
            if not gold_data:
                print("❌ Failed to extract Gold data")
                return False
            
            # Validate data quality
            print("🔍 Validating Gold data quality...")
            overall_quality = []
            
            for tf, df in gold_data.items():
                quality = extractor.validate_data_quality(df)
                overall_quality.append(quality["score"])
                print(f"   {tf}: {quality['status']} (Score: {quality['score']}/100)")
                
                if quality["score"] < self.gold_config["min_data_quality"]:
                    print(f"⚠️ {tf} quality below threshold: {quality['score']}")
            
            avg_quality = sum(overall_quality) / len(overall_quality)
            print(f"📊 Average data quality: {avg_quality:.1f}/100")
            
            if avg_quality < self.gold_config["min_data_quality"]:
                print(f"❌ Overall data quality too low: {avg_quality:.1f}")
                return False
            
            # Export raw data
            base_filename = self.target_symbol.replace(".", "_") + "_SMC_dataset"
            success = extractor.export_smc_dataset(gold_data, base_filename, "csv")
            
            if success:
                print("✅ Step 1 completed: Gold data extracted successfully")
                return True
            else:
                print("❌ Step 1 failed: Data export error")
                return False
                
        except Exception as e:
            print(f"❌ Step 1 error: {str(e)}")
            return False
    
    def step_2_engineer_gold_features(self) -> bool:
        """Step 2: Engineer SMC features for Gold"""
        print("\n" + "="*60)
        print("2️⃣ STEP 2: GOLD SMC FEATURES ENGINEERING")
        print("="*60)
        
        try:
            # Initialize features engine
            features_engine = SMCFeaturesEngine()
            
            # Load extracted data
            base_filename = self.target_symbol.replace(".", "_") + "_SMC_dataset"
            print(f"📂 Loading Gold dataset: {base_filename}")
            
            smc_data = features_engine.load_smc_dataset(base_filename)
            
            if not smc_data:
                print("❌ Failed to load Gold dataset")
                return False
            
            # Process SMC features with Gold optimizations
            print("🔧 Processing Gold SMC features...")
            
            # Enhanced parameters for Gold
            features_engine.swing_period = 8          # Longer swings for Gold
            features_engine.structure_period = 30     # Extended structure
            features_engine.ob_lookback = 15         # More OB lookback
            features_engine.ob_min_size = 0.5        # Larger OB threshold
            features_engine.fvg_min_size = 0.3       # Larger FVG threshold
            features_engine.liquidity_period = 100   # Extended liquidity
            
            processed_data = features_engine.process_complete_dataset(smc_data)
            
            # Add Gold-specific features
            print("🥇 Adding Gold-specific features...")
            for timeframe, df in processed_data.items():
                # Session-based features
                df["hour"] = df.index.hour
                df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] <= 16)).astype(int)
                df["is_us_session"] = ((df["hour"] >= 13) & (df["hour"] <= 21)).astype(int)
                df["is_asian_session"] = ((df["hour"] >= 22) | (df["hour"] <= 7)).astype(int)
                
                # Gold volatility features
                df["gold_range_points"] = df["range"] / 0.01  # Points calculation
                df["gold_atr_points"] = df["gold_range_points"].rolling(14).mean()
                df["gold_volatility"] = df["gold_range_points"].rolling(20).std()
                
                # High impact hours
                df["is_high_impact_hour"] = (
                    (df["hour"] == 8) |   # London open
                    (df["hour"] == 13) |  # US open
                    (df["hour"] == 14) |  # US continuation
                    (df["hour"] == 15)    # Overlap
                ).astype(int)
                
                # Gold momentum features
                df["gold_momentum_5"] = df["close"].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
                df["gold_momentum_10"] = df["close"].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
                
                # News impact simulation
                df["potential_news_impact"] = (
                    df["gold_range_points"] > df["gold_atr_points"] * 2
                ).astype(int)
                
                processed_data[timeframe] = df
            
            # Export enhanced features
            export_base = self.target_symbol.replace(".", "_")
            success = features_engine.export_smc_features(processed_data, export_base)
            
            if success:
                print("✅ Step 2 completed: Gold SMC features engineered successfully")
                return True
            else:
                print("❌ Step 2 failed: Features export error")
                return False
                
        except Exception as e:
            print(f"❌ Step 2 error: {str(e)}")
            return False
    
    def step_3_create_gold_labels(self) -> bool:
        """Step 3: Create trading labels for Gold"""
        print("\n" + "="*60)
        print("3️⃣ STEP 3: GOLD TRADING LABELS CREATION")
        print("="*60)
        
        try:
            # Initialize labels engine
            labels_engine = SMCLabelsEngine()
            
            # Gold-specific label parameters
            labels_engine.default_risk_reward = 2.5  # Better R:R for Gold
            labels_engine.atr_multiplier_sl = 2.0    # Wider stops for Gold
            labels_engine.atr_multiplier_tp = 5.0    # Wider targets
            labels_engine.min_confluence = 3         # Higher confluence
            
            # Adjust max holding periods for Gold volatility
            labels_engine.max_holding_periods = {
                "M5": 200,   # Shorter for Gold M5
                "M15": 80,   # Adjusted for Gold
                "H1": 20,    # Conservative for Gold
                "H4": 10,    # Very conservative
                "D1": 5,     # Same as before
            }
            
            # Load SMC features
            base_filename = self.target_symbol.replace(".", "_")
            print(f"📂 Loading Gold SMC features: {base_filename}")
            
            smc_features = labels_engine.load_smc_features(base_filename)
            
            if not smc_features:
                print("❌ Failed to load Gold SMC features")
                return False
            
            # Create trading labels
            print("🏷️ Creating Gold trading labels...")
            labeled_data = labels_engine.process_complete_labels_dataset(smc_features)
            
            # Validate label quality
            print("🔍 Validating Gold label quality...")
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
                
                print(f"   {timeframe}: {signals} signals, {trades} trades, {win_rate:.1f}% win rate")
                
                if trades < 20:  # Minimum trades threshold
                    print(f"⚠️ {timeframe} has insufficient trades: {trades}")
            
            overall_win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
            print(f"📊 Overall Gold performance: {total_trades} trades, {overall_win_rate:.1f}% win rate")
            
            if overall_win_rate < self.gold_config["min_win_rate"]:
                print(f"⚠️ Win rate below threshold: {overall_win_rate:.1f}% < {self.gold_config['min_win_rate']}%")
                print("🔧 Consider adjusting label parameters")
            
            if total_trades < self.gold_config["min_trades"]:
                print(f"⚠️ Insufficient total trades: {total_trades} < {self.gold_config['min_trades']}")
                print("🔧 Consider lowering confluence requirements")
            
            # Export labels
            success = labels_engine.export_labeled_dataset(labeled_data, base_filename)
            
            if success:
                print("✅ Step 3 completed: Gold trading labels created successfully")
                return True
            else:
                print("❌ Step 3 failed: Labels export error")
                return False
                
        except Exception as e:
            print(f"❌ Step 3 error: {str(e)}")
            return False
    
    def step_4_train_gold_models(self) -> bool:
        """Step 4: Train AI models for Gold"""
        print("\n" + "="*60)
        print("4️⃣ STEP 4: GOLD AI MODEL TRAINING")
        print("="*60)
        
        try:
            # Initialize trainer
            trainer = SMCAITrainer()
            
            # Gold-specific training parameters
            trainer.feature_selection_k = 60  # More features for Gold
            
            # Enhanced Random Forest for Gold
            trainer.random_forest_config = {
                "n_estimators": [150, 250, 350],     # More trees
                "max_depth": [15, 25, None],         # Deeper trees
                "min_samples_split": [2, 3, 5],     # More conservative
                "min_samples_leaf": [1, 2, 3],      # Smaller leaves
                "max_features": ["sqrt", "log2", 0.8] # Feature selection
            }
            
            # Enhanced XGBoost for Gold
            trainer.xgboost_config = {
                "n_estimators": [150, 250, 350],
                "max_depth": [8, 12, 16],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
            
            # Load labeled data
            base_filename = self.target_symbol.replace(".", "_")
            print(f"📂 Loading Gold labeled dataset: {base_filename}")
            
            labeled_data = trainer.load_labeled_dataset(base_filename)
            
            if not labeled_data:
                print("❌ Failed to load Gold labeled dataset")
                return False
            
            # Train models
            print("🤖 Training Gold AI models...")
            models_to_train = ["random_forest"]
            
            # Check for optional libraries
            try:
                import xgboost
                models_to_train.append("xgboost")
                print("✅ XGBoost available")
            except ImportError:
                print("⚠️ XGBoost not available")
            
            try:
                import tensorflow
                models_to_train.append("neural_network")
                print("✅ TensorFlow available")
            except ImportError:
                print("⚠️ TensorFlow not available")
            
            print(f"🎯 Training models: {models_to_train}")
            
            all_results = trainer.train_all_timeframes(labeled_data, models_to_train)
            
            # Validate model performance
            print("🔍 Validating Gold model performance...")
            performance_summary = {}
            
            for timeframe, tf_results in all_results.items():
                tf_performance = {}
                for model_name, model_results in tf_results.items():
                    if "accuracy" in model_results:
                        accuracy = model_results["accuracy"]
                        cv_score = model_results.get("cv_score_mean", 0)
                        tf_performance[model_name] = {
                            "accuracy": accuracy,
                            "cv_score": cv_score
                        }
                        print(f"   {timeframe} {model_name}: {accuracy:.3f} accuracy, {cv_score:.3f} CV")
                
                performance_summary[timeframe] = tf_performance
            
            # Calculate overall performance
            all_accuracies = []
            for tf_perf in performance_summary.values():
                for model_perf in tf_perf.values():
                    all_accuracies.append(model_perf["accuracy"])
            
            avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
            print(f"📊 Average model accuracy: {avg_accuracy:.3f}")
            
            if avg_accuracy < 0.65:  # 65% minimum threshold
                print(f"⚠️ Model performance below threshold: {avg_accuracy:.3f}")
                print("🔧 Consider feature engineering or parameter tuning")
            
            # Save models
            model_base = self.target_symbol.replace(".", "_") + "_SMC"
            success = trainer.save_models(model_base)
            
            if success:
                # Generate performance report
                trainer.generate_performance_report(model_base)
                
                # Save feature mapping for prediction alignment
                self.save_gold_feature_mapping(all_results, model_base)
                
                print("✅ Step 4 completed: Gold AI models trained successfully")
                return True
            else:
                print("❌ Step 4 failed: Model saving error")
                return False
                
        except Exception as e:
            print(f"❌ Step 4 error: {str(e)}")
            return False
    
    def save_gold_feature_mapping(self, model_results: dict, base_filename: str):
        """Save feature mapping for prediction alignment"""
        try:
            feature_mapping = {}
            
            for timeframe, tf_models in model_results.items():
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
            
            # Save mapping
            mapping_file = f"{base_filename}_feature_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(feature_mapping, f, indent=2)
            
            print(f"💾 Feature mapping saved: {mapping_file}")
            
        except Exception as e:
            print(f"⚠️ Feature mapping save error: {str(e)}")
    
    def step_5_validate_system(self) -> bool:
        """Step 5: Validate complete Gold trading system"""
        print("\n" + "="*60)
        print("5️⃣ STEP 5: GOLD SYSTEM VALIDATION")
        print("="*60)
        
        try:
            base_filename = self.target_symbol.replace(".", "_")
            
            # Check all required files exist
            required_files = [
                f"{base_filename}_SMC_dataset_M5.csv",
                f"{base_filename}_SMC_features_M5.csv",
                f"{base_filename}_labeled_M5.csv",
                f"{base_filename}_SMC_M5_random_forest_model.pkl",
                f"{base_filename}_SMC_feature_mapping.json",
                f"{base_filename}_SMC_performance_report.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                print("❌ Missing required files:")
                for file in missing_files:
                    print(f"   - {file}")
                return False
            
            print("✅ All required files present")
            
            # Validate model performance
            try:
                with open(f"{base_filename}_SMC_performance_report.json", 'r') as f:
                    performance = json.load(f)
                
                print("📊 Model Performance Summary:")
                timeframes = performance.get("timeframes", [])
                
                for tf in timeframes:
                    tf_perf = performance.get("performance_summary", {}).get(tf, {})
                    for model, metrics in tf_perf.items():
                        accuracy = metrics.get("accuracy", 0)
                        cv_score = metrics.get("cv_score_mean", 0)
                        print(f"   {tf} {model}: {accuracy:.3f} accuracy, {cv_score:.3f} CV")
                
            except Exception as e:
                print(f"⚠️ Could not load performance report: {str(e)}")
            
            # Test signal generation (dry run)
            print("🧪 Testing signal generation...")
            try:
                from smc_signal_engine import SMCSignalEngine
                
                # Initialize signal engine
                signal_engine = SMCSignalEngine(f"{base_filename}_SMC")
                
                # Try to load models
                models_loaded = signal_engine.load_trained_models()
                
                if models_loaded:
                    print("✅ Signal engine models loaded successfully")
                    
                    # Test feature alignment
                    if hasattr(signal_engine, 'training_features') and signal_engine.training_features:
                        print("✅ Feature mapping loaded successfully")
                        for tf, features in signal_engine.training_features.items():
                            print(f"   {tf}: {len(features)} features mapped")
                    else:
                        print("⚠️ Feature mapping not available")
                    
                else:
                    print("❌ Failed to load signal engine models")
                    return False
                
            except Exception as e:
                print(f"❌ Signal engine test failed: {str(e)}")
                return False
            
            print("✅ Step 5 completed: Gold system validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Step 5 error: {str(e)}")
            return False
    
    def run_complete_setup(self):
        """Run complete Gold trading system setup"""
        print("🥇 GOLD TRADING SYSTEM COMPLETE SETUP")
        print("=" * 70)
        print("🎯 Target: XAUUSD.c Auto Trading System")
        print("👨‍🏫 Created by: อาจารย์ฟินิกซ์")
        print("=" * 70)
        
        setup_steps = [
            ("Data Extraction", self.step_1_extract_gold_data),
            ("SMC Features Engineering", self.step_2_engineer_gold_features),
            ("Trading Labels Creation", self.step_3_create_gold_labels),
            ("AI Model Training", self.step_4_train_gold_models),
            ("System Validation", self.step_5_validate_system)
        ]
        
        completed_steps = 0
        
        for step_name, step_function in setup_steps:
            print(f"\n🚀 Starting: {step_name}")
            
            if not self.setup_config.get(step_name.lower().replace(" ", "_"), True):
                print(f"⏭️ Skipping {step_name} (disabled in config)")
                continue
            
            try:
                success = step_function()
                if success:
                    completed_steps += 1
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    
                    user_input = input(f"\n❓ Continue with next step? (y/n): ").lower()
                    if user_input != 'y':
                        print("🛑 Setup interrupted by user")
                        break
                        
            except KeyboardInterrupt:
                print(f"\n🛑 Setup interrupted during {step_name}")
                break
            except Exception as e:
                print(f"❌ {step_name} error: {str(e)}")
                
                user_input = input(f"\n❓ Continue despite error? (y/n): ").lower()
                if user_input != 'y':
                    break
        
        # Final summary
        print(f"\n" + "="*70)
        print("📋 GOLD TRADING SYSTEM SETUP SUMMARY")
        print("="*70)
        print(f"✅ Completed steps: {completed_steps}/{len(setup_steps)}")
        
        if completed_steps == len(setup_steps):
            print("🎉 COMPLETE SUCCESS! Gold trading system ready for live trading")
            print(f"🥇 Symbol: {self.target_symbol}")
            print("📂 Files generated:")
            
            base_name = self.target_symbol.replace(".", "_")
            files_to_list = [
                f"{base_name}_SMC_dataset_*.csv",
                f"{base_name}_SMC_features_*.csv", 
                f"{base_name}_labeled_*.csv",
                f"{base_name}_SMC_*_model.pkl",
                f"{base_name}_SMC_performance_report.json"
            ]
            
            for file_pattern in files_to_list:
                print(f"   📄 {file_pattern}")
            
            print(f"\n🚀 Next Steps:")
            print(f"1. Test signal generation: python smc_signal_engine.py")
            print(f"2. Start auto trading: python gold_auto_trader.py")
            print(f"3. Monitor performance in: gold_trading_logs/")
            
        else:
            print("⚠️ PARTIAL COMPLETION - Some steps failed or were skipped")
            print("🔧 Review error messages and retry failed steps")
        
        print("="*70)

def main():
    """Main execution function"""
    print("🥇 Gold Trading System Setup")
    print("="*50)
    
    # Initialize setup
    setup = GoldTradingSystemSetup()
    
    # Ask user for confirmation
    print(f"🎯 This will setup complete AI trading system for {setup.base_symbol}")
    print("📊 Steps: Data → Features → Labels → Training → Validation")
    print("⏱️ Estimated time: 30-60 minutes depending on data size")
    
    confirm = input("\n🚀 Proceed with complete setup? (yes/no): ").lower().strip()
    
    if confirm == "yes":
        setup.run_complete_setup()
    else:
        print("🛑 Setup cancelled by user")
        
        # Offer step-by-step option
        step_by_step = input("❓ Run step-by-step instead? (yes/no): ").lower().strip()
        if step_by_step == "yes":
            print("🔧 Step-by-step mode - you can skip individual steps")
            setup.run_complete_setup()

if __name__ == "__main__":
    main()