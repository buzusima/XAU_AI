#!/usr/bin/env python3
"""
🥇 Gold Trading System - Main Setup Script
อาจารย์ฟินิกซ์ - Complete XAUUSD.c Trading Solution

This is the main script to run the complete Gold trading setup.
It will guide you through all steps needed for Gold trading.

Usage:
python gold_main_setup.py
"""

import os
import sys
from datetime import datetime

def print_header():
    """Print application header"""
    print("=" * 70)
    print("🥇 GOLD TRADING SYSTEM - COMPLETE SETUP")
    print("=" * 70)
    print("🎯 Target: XAUUSD.c (Gold) Auto Trading")
    print("👨‍🏫 Created by: อาจารย์ฟินิกซ์")
    print("🚀 AI-Powered SMC Trading Bot")
    print("=" * 70)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n🔍 Checking Dependencies...")
    
    required_packages = [
        'MetaTrader5',
        'pandas', 
        'numpy',
        'scikit-learn',
        'joblib'
    ]
    
    optional_packages = [
        'xgboost',
        'tensorflow'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("\n✅ Dependencies check complete")
    return True

def check_mt5_files():
    """Check if all required MT5 files exist"""
    print("\n📂 Checking Required Files...")
    
    required_files = [
        'MT5MultiTimeframeExtractor.py',
        'SMCFeaturesEngine.py', 
        'SMCLabelsEngine.py',
        'SMCAITrainer.py',
        'smc_signal_engine.py',
        'SMCAutoTrader_new.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    print("\n✅ All required files present")
    return True

def show_setup_options():
    """Show available setup options"""
    print("\n🎯 GOLD TRADING SETUP OPTIONS")
    print("=" * 50)
    print("1️⃣ Complete Setup (Recommended)")
    print("   📊 Data Extraction → Features → Labels → Training → Validation")
    print("   ⏱️ Time: 30-60 minutes")
    print()
    print("2️⃣ Quick Start (Use Existing Models)")
    print("   🚀 Jump straight to live trading")
    print("   ⏱️ Time: 2-5 minutes")
    print()
    print("3️⃣ Step-by-Step Setup")
    print("   🔧 Run individual steps with control")
    print("   ⏱️ Time: Variable")
    print()
    print("4️⃣ Signal Testing Only")
    print("   🧪 Test signal generation without trading")
    print("   ⏱️ Time: 2-3 minutes")
    print("=" * 50)

def run_complete_setup():
    """Run complete Gold trading setup"""
    print("\n🚀 Starting Complete Gold Setup...")
    
    try:
        # Import and run complete setup
        from gold_complete_setup import GoldTradingSystemSetup
        
        setup = GoldTradingSystemSetup()
        setup.run_complete_setup()
        
    except ImportError:
        print("❌ gold_complete_setup.py not found")
        print("🔧 Running individual setup steps...")
        run_individual_steps()
    except Exception as e:
        print(f"❌ Setup error: {str(e)}")

def run_individual_steps():
    """Run individual setup steps"""
    print("\n🔧 Running Individual Setup Steps...")
    
    steps = [
        ("Data Extraction", run_data_extraction),
        ("SMC Features", run_smc_features),
        ("Trading Labels", run_trading_labels),
        ("AI Training", run_ai_training),
        ("Validation", run_validation)
    ]
    
    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        proceed = input(f"Run {step_name}? (y/n/skip): ").lower()
        
        if proceed == 'y':
            try:
                step_function()
                print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {str(e)}")
                
                continue_anyway = input(f"Continue to next step? (y/n): ").lower()
                if continue_anyway != 'y':
                    break
        elif proceed == 'skip':
            print(f"⏭️ Skipped {step_name}")
        else:
            print("🛑 Setup stopped by user")
            break

def run_data_extraction():
    """Run data extraction step"""
    print("📊 Extracting Gold data from MT5...")
    
    # Import and run data extraction
    from MT5MultiTimeframeExtractor import MT5MultiTimeframeExtractor
    
    extractor = MT5MultiTimeframeExtractor()
    
    # Find Gold symbols
    gold_variations = extractor.find_symbol_variations("XAUUSD")
    
    if not gold_variations:
        print("❌ No Gold symbols found")
        return
    
    # Select symbol
    target_symbol = None
    for var in gold_variations:
        if ".c" in var["symbol"]:
            target_symbol = var["symbol"]
            break
    
    if target_symbol is None:
        target_symbol = gold_variations[0]["symbol"]
    
    print(f"🥇 Using Gold symbol: {target_symbol}")
    
    # Extract data
    gold_data = extractor.get_smc_dataset(target_symbol)
    
    if gold_data:
        # Export data
        base_filename = target_symbol.replace(".", "_") + "_SMC_dataset"
        extractor.export_smc_dataset(gold_data, base_filename, "csv")
        print("✅ Gold data extracted successfully")
    else:
        print("❌ Gold data extraction failed")

def run_smc_features():
    """Run SMC features engineering"""
    print("🔧 Engineering SMC features for Gold...")
    
    from SMCFeaturesEngine import SMCFeaturesEngine
    
    engine = SMCFeaturesEngine()
    
    # Adjust parameters for Gold
    engine.swing_period = 8
    engine.structure_period = 30
    engine.ob_lookback = 15
    engine.ob_min_size = 0.5
    engine.fvg_min_size = 0.3
    engine.liquidity_period = 100
    
    # Load and process data
    smc_data = engine.load_smc_dataset("XAUUSD_c_SMC_dataset")
    
    if smc_data:
        processed_data = engine.process_complete_dataset(smc_data)
        engine.export_smc_features(processed_data, "XAUUSD_c")
        print("✅ SMC features engineered successfully")
    else:
        print("❌ SMC features engineering failed")

def run_trading_labels():
    """Run trading labels creation"""
    print("🏷️ Creating trading labels for Gold...")
    
    from SMCLabelsEngine import SMCLabelsEngine
    
    engine = SMCLabelsEngine()
    
    # Adjust parameters for Gold
    engine.default_risk_reward = 2.5
    engine.atr_multiplier_sl = 2.0
    engine.atr_multiplier_tp = 5.0
    engine.min_confluence = 3
    
    # Load and process
    smc_features = engine.load_smc_features("XAUUSD_c")
    
    if smc_features:
        labeled_data = engine.process_complete_labels_dataset(smc_features)
        engine.export_labeled_dataset(labeled_data, "XAUUSD_c")
        print("✅ Trading labels created successfully")
    else:
        print("❌ Trading labels creation failed")

def run_ai_training():
    """Run AI model training"""
    print("🤖 Training AI models for Gold...")
    
    from SMCAITrainer import SMCAITrainer
    
    trainer = SMCAITrainer()
    
    # Load labeled data
    labeled_data = trainer.load_labeled_dataset("XAUUSD_c")
    
    if labeled_data:
        # Train models
        models_to_train = ["random_forest"]
        
        try:
            import xgboost
            models_to_train.append("xgboost")
        except ImportError:
            pass
        
        try:
            import tensorflow
            models_to_train.append("neural_network")
        except ImportError:
            pass
        
        all_results = trainer.train_all_timeframes(labeled_data, models_to_train)
        trainer.save_models("XAUUSD_c_SMC")
        trainer.generate_performance_report("XAUUSD_c_SMC")
        print("✅ AI models trained successfully")
    else:
        print("❌ AI training failed")

def run_validation():
    """Run system validation"""
    print("🧪 Validating Gold trading system...")
    
    # Check required files exist
    required_files = [
        "XAUUSD_c_SMC_M5_random_forest_model.pkl",
        "XAUUSD_c_SMC_performance_report.json"
    ]
    
    all_exist = True
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing: {file}")
            all_exist = False
        else:
            print(f"✅ Found: {file}")
    
    if all_exist:
        print("✅ System validation successful")
    else:
        print("❌ System validation failed")

def run_quick_start():
    """Run quick start for existing models"""
    print("\n🚀 Quick Start - Using Existing Models...")
    
    # Check if models exist
    model_files = [
        "XAUUSD_c_SMC_M5_random_forest_model.pkl",
        "XAUUSD_c_SMC_H1_random_forest_model.pkl"
    ]
    
    models_exist = any(os.path.exists(file) for file in model_files)
    
    if not models_exist:
        print("❌ No existing Gold models found")
        print("🔧 You need to run Complete Setup first")
        return
    
    print("✅ Found existing Gold models")
    
    # Test signal generation
    try:
        from smc_signal_engine import SMCSignalEngine
        
        print("🧪 Testing Gold signal generation...")
        engine = SMCSignalEngine("XAUUSD_c_SMC")
        
        if engine.load_trained_models():
            print("✅ Models loaded successfully")
            
            # Test MT5 connection
            if engine.connect_mt5():
                print("✅ MT5 connected")
                
                # Find Gold symbol
                gold_symbols = engine.find_gold_symbol_variations("XAUUSD")
                
                if gold_symbols:
                    target_symbol = gold_symbols[0]["symbol"]
                    print(f"🥇 Testing with: {target_symbol}")
                    
                    # Generate test signal
                    signal = engine.get_multi_timeframe_signals(target_symbol)
                    
                    if "error" not in signal:
                        print("✅ Signal generation successful")
                        print(f"   Direction: {signal['final_direction']}")
                        print(f"   Confidence: {signal['final_confidence']:.3f}")
                        print("🚀 Ready for live trading!")
                    else:
                        print(f"❌ Signal generation failed: {signal['error']}")
                else:
                    print("❌ No Gold symbols found")
            else:
                print("❌ MT5 connection failed")
        else:
            print("❌ Model loading failed")
            
    except Exception as e:
        print(f"❌ Quick start error: {str(e)}")

def run_signal_testing():
    """Run signal testing only"""
    print("\n🧪 Testing Gold Signal Generation...")
    
    try:
        from smc_signal_engine import SMCSignalEngine
        
        engine = SMCSignalEngine("XAUUSD_c_SMC")
        
        if engine.load_trained_models():
            if engine.connect_mt5():
                # Find Gold symbol
                gold_symbols = engine.find_gold_symbol_variations("XAUUSD")
                
                if gold_symbols:
                    target_symbol = gold_symbols[0]["symbol"]
                    print(f"🥇 Testing signals for: {target_symbol}")
                    
                    # Run continuous signal testing
                    print("🔄 Starting signal monitoring (Ctrl+C to stop)...")
                    engine.start_signal_monitoring(target_symbol, 30)
                else:
                    print("❌ No Gold symbols found")
            else:
                print("❌ MT5 connection failed")
        else:
            print("❌ Model loading failed")
            
    except KeyboardInterrupt:
        print("\n✅ Signal testing stopped")
    except Exception as e:
        print(f"❌ Signal testing error: {str(e)}")

def run_live_trading():
    """Start live Gold trading"""
    print("\n🚀 Starting Live Gold Trading...")
    
    # Check models exist
    if not os.path.exists("XAUUSD_c_SMC_M5_random_forest_model.pkl"):
        print("❌ No trained models found")
        print("🔧 Run Complete Setup first")
        return
    
    try:
        from SMCAutoTrader_new import SMCAutoTrader
        
        # Initialize Gold trader
        trader = SMCAutoTrader(
            models_path="XAUUSD_c_SMC",
            max_risk_per_trade=0.015,
            max_daily_loss=0.04,
            min_confidence=0.80,
            min_consensus=4,
            default_sl_pips=500,
            default_tp_ratio=2.5,
            max_lot_size=0.05
        )
        
        print("🥇 Gold Auto Trader initialized")
        trader.print_current_settings()
        
        if trader.connect_mt5():
            if trader.load_models():
                print("✅ Gold trading system ready!")
                
                # Ask for trading confirmation
                enable_trading = input("\n⚠️ Enable LIVE GOLD TRADING? (yes/no): ").lower()
                
                if enable_trading == "yes":
                    trader.enable_trading(True)
                    print("🔥 LIVE GOLD TRADING ENABLED!")
                else:
                    trader.enable_trading(False)
                    print("📊 Demo mode - signals only")
                
                # Start trading
                trader.start_auto_trading("XAUUSD.c", 60)
                
            else:
                print("❌ Failed to load models")
        else:
            print("❌ Failed to connect to MT5")
            
    except KeyboardInterrupt:
        print("\n✅ Gold trading stopped")
    except Exception as e:
        print(f"❌ Trading error: {str(e)}")

def show_status_report():
    """Show current system status"""
    print("\n📊 GOLD TRADING SYSTEM STATUS")
    print("=" * 50)
    
    # Check data files
    data_files = [
        "XAUUSD_c_SMC_dataset_M5.csv",
        "XAUUSD_c_SMC_dataset_H1.csv",
        "XAUUSD_c_SMC_dataset_D1.csv"
    ]
    
    print("📂 Data Files:")
    for file in data_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # Check feature files
    feature_files = [
        "XAUUSD_c_SMC_features_M5.csv",
        "XAUUSD_c_SMC_features_H1.csv",
        "XAUUSD_c_SMC_features_D1.csv"
    ]
    
    print("\n🔧 Feature Files:")
    for file in feature_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # Check label files
    label_files = [
        "XAUUSD_c_labeled_M5.csv",
        "XAUUSD_c_labeled_H1.csv",
        "XAUUSD_c_labeled_D1.csv"
    ]
    
    print("\n🏷️ Label Files:")
    for file in label_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # Check model files
    model_files = [
        "XAUUSD_c_SMC_M5_random_forest_model.pkl",
        "XAUUSD_c_SMC_H1_random_forest_model.pkl",
        "XAUUSD_c_SMC_D1_random_forest_model.pkl"
    ]
    
    print("\n🤖 Model Files:")
    for file in model_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # Check performance report
    report_files = [
        "XAUUSD_c_SMC_performance_report.json",
        "XAUUSD_c_SMC_feature_mapping.json"
    ]
    
    print("\n📊 Report Files:")
    for file in report_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    # Check log directory
    log_dir = "gold_trading_logs"
    print(f"\n📝 Log Directory:")
    print(f"   {'✅' if os.path.exists(log_dir) else '❌'} {log_dir}/")
    
    print("=" * 50)

def main():
    """Main function"""
    print_header()
    
    # Check dependencies and files
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    if not check_mt5_files():
        print("\n❌ Please ensure all required files are present")
        return
    
    while True:
        show_setup_options()
        
        choice = input("\n🎯 Select option (1-4) or 'status'/'quit': ").strip().lower()
        
        if choice == '1':
            run_complete_setup()
            
        elif choice == '2':
            run_quick_start()
            
        elif choice == '3':
            run_individual_steps()
            
        elif choice == '4':
            run_signal_testing()
            
        elif choice == 'status':
            show_status_report()
            
        elif choice in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        elif choice == 'trade':
            run_live_trading()
            
        else:
            print("❌ Invalid option. Please try again.")
        
        # Ask if continue
        if choice not in ['quit', 'exit', 'q', 'status']:
            continue_choice = input("\n❓ Return to main menu? (y/n): ").lower()
            if continue_choice != 'y':
                print("👋 Goodbye!")
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("🔧 Please check your setup and try again")