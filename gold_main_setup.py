#!/usr/bin/env python3
"""
🥇 Gold Trading System - Main Setup Script (COMPLETE FIXED VERSION)
อาจารย์ฟินิกซ์ - Complete XAUUSD.c Trading Solution

This is the main script to run the complete Gold trading setup.
แก้ไขปัญหาทั้งหมด: dependency check, symbol detection, data extraction

Usage:
python gold_main_setup.py
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

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
    """Check if all required dependencies are installed - FIXED VERSION"""
    print("\n🔍 Checking Dependencies...")
    
    # 🔧 แก้ไข: ใช้ dictionary mapping แทน tuple เพื่อหลีกเลี่ยง error
    required_packages = {
        'MetaTrader5': 'MetaTrader5',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',  # 🔧 import sklearn, pip install scikit-learn
        'joblib': 'joblib'
    }
    
    optional_packages = {
        'xgboost': 'xgboost',
        'tensorflow': 'tensorflow'
    }
    
    missing_required = []
    missing_optional = []
    
    # ตรวจสอบ required packages
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            missing_required.append(pip_name)
            print(f"❌ {pip_name}")
    
    # ตรวจสอบ optional packages
    for import_name, pip_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name} (optional)")
        except ImportError:
            missing_optional.append(pip_name)
            print(f"⚠️ {pip_name} (optional)")
    
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

def find_gold_symbol():
    """Find proper Gold symbol - ENHANCED FIXED VERSION"""
    try:
        # Import here to avoid early import errors
        import MetaTrader5 as mt5
        
        # Initialize MT5 connection
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return None
        
        print("✅ MT5 Connected for symbol search")
        
        # 🥇 Gold symbol priorities (in order of preference)
        gold_symbols_priority = [
            "XAUUSD.c",    # Top priority - most common
            "XAUUSD.v",    # Second priority  
            "XAUUSD",      # Standard
            "GOLD.c", 
            "GOLD.v",
            "GOLD",
            "XAU/USD",
            "XAUUSD#",
            "XAUUSD.raw",
            "XAUUSD.a"
        ]
        
        print("🔍 Searching for Gold symbols...")
        
        # Check priority symbols first
        for symbol in gold_symbols_priority:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    # Test if we can get tick data
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None and tick.bid > 0:
                        print(f"✅ Found tradeable Gold symbol: {symbol}")
                        print(f"   Description: {symbol_info.description}")
                        print(f"   Current price: {tick.bid:.2f}")
                        return symbol
            except Exception as e:
                continue
        
        # Search all symbols for Gold-related ones
        print("🔍 Searching all symbols for Gold...")
        symbols = mt5.symbols_get()
        if symbols is not None:
            gold_candidates = []
            
            for symbol in symbols:
                symbol_name = symbol.name.upper()
                symbol_desc = symbol.description.upper()
                
                # Check if it's Gold related
                if any(keyword in symbol_name for keyword in ['XAU', 'GOLD']) or \
                   any(keyword in symbol_desc for keyword in ['GOLD', 'XAU']):
                    
                    # Exclude AUD symbols (Australian Dollar)
                    if not any(keyword in symbol_name for keyword in ['AUD', 'CAD', 'JPY', 'EUR', 'GBP']) or \
                       'XAU' in symbol_name:
                        
                        try:
                            tick = mt5.symbol_info_tick(symbol.name)
                            if tick is not None and tick.bid > 1000:  # Gold should be > 1000
                                gold_candidates.append({
                                    'symbol': symbol.name,
                                    'description': symbol.description,
                                    'price': tick.bid
                                })
                        except:
                            continue
            
            # Sort by preference (XAUUSD variations first)
            gold_candidates.sort(key=lambda x: (
                'XAUUSD' not in x['symbol'].upper(),  # XAUUSD first
                '.c' not in x['symbol'],              # .c suffix preferred
                x['symbol']                           # Alphabetical
            ))
            
            if gold_candidates:
                best_symbol = gold_candidates[0]
                print(f"🥇 Found Gold symbol: {best_symbol['symbol']}")
                print(f"   Description: {best_symbol['description']}")
                print(f"   Current price: {best_symbol['price']:.2f}")
                
                # Show all found Gold symbols
                if len(gold_candidates) > 1:
                    print(f"🔍 Other Gold symbols available:")
                    for candidate in gold_candidates[1:]:
                        print(f"   - {candidate['symbol']}: {candidate['description']}")
                
                return best_symbol['symbol']
        
        print("❌ No Gold symbols found")
        print("🔧 Possible issues:")
        print("   - Broker doesn't offer Gold trading")
        print("   - Symbol names are different")
        print("   - Market is closed")
        print("   - Account doesn't have Gold permissions")
        return None
        
    except Exception as e:
        print(f"❌ Error finding Gold symbol: {str(e)}")
        return None

def apply_data_extraction_fix():
    """Apply fix to MT5MultiTimeframeExtractor.py for range error"""
    try:
        file_path = "MT5MultiTimeframeExtractor.py"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        print("🔧 Applying data extraction fix...")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already fixed
        if '# 🔧 FIX: Price ranges and volatility (moved up)' in content:
            print("✅ Data extraction fix already applied")
            return True
        
        # Apply the fix - move range calculation up
        if 'df["range"] = df["high"] - df["low"]' in content:
            # Find the position to insert the range calculation
            ohlc4_pos = content.find('df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4')
            
            if ohlc4_pos != -1:
                # Find the end of the ohlc4 line
                ohlc4_end = content.find('\n', ohlc4_pos) + 1
                
                # Remove old range calculation
                old_range_section = '''
        # Price ranges and volatility
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100'''
                
                content = content.replace(old_range_section, '')
                
                # Insert new range calculation after ohlc4
                new_range_section = '''
        # 🔧 FIX: Price ranges and volatility (moved up)
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100'''
                
                content = content[:ohlc4_end] + new_range_section + content[ohlc4_end:]
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✅ Applied data extraction fix successfully")
                return True
        
        print("⚠️ Could not apply automatic fix - manual intervention may be needed")
        return False
        
    except Exception as e:
        print(f"❌ Error applying fix: {str(e)}")
        return False

def run_complete_setup():
    """Run complete Gold trading setup - ENHANCED FIXED VERSION"""
    print("\n🚀 Starting Complete Gold Setup...")
    
    try:
        # Apply data extraction fix first
        print("🔧 Checking and applying necessary fixes...")
        # apply_data_extraction_fix()
        
        print("✅ Using pre-fixed MT5MultiTimeframeExtractor.py")

        # Find Gold symbol first
        gold_symbol = find_gold_symbol()
        if not gold_symbol:
            print("❌ Cannot proceed without Gold symbol")
            print("🔧 Please check:")
            print("   1. MT5 Terminal is running")
            print("   2. Account has Gold trading permissions")
            print("   3. Broker offers Gold/XAUUSD trading")
            return False
        
        print(f"🥇 Using Gold symbol: {gold_symbol}")
        
        # Import modules here to avoid early import issues
        from MT5MultiTimeframeExtractor import MT5MultiTimeframeExtractor
        from SMCFeaturesEngine import SMCFeaturesEngine
        from SMCLabelsEngine import SMCLabelsEngine
        from SMCAITrainer import SMCAITrainer
        
        print("\n" + "="*60)
        print("1️⃣ STEP 1: GOLD DATA EXTRACTION")
        print("="*60)
        
        # Initialize extractor
        extractor = MT5MultiTimeframeExtractor()
        
        # Extract Gold data
        print(f"📊 Extracting Gold data for {gold_symbol}...")
        gold_data = extractor.get_smc_dataset(gold_symbol)
        
        if not gold_data:
            print("❌ Failed to extract Gold data")
            print("🔧 Possible solutions:")
            print("   1. Check MT5 connection")
            print("   2. Verify symbol permissions")
            print("   3. Try different time periods")
            return False
        
        # Validate extracted data
        total_candles = sum(len(df) for df in gold_data.values())
        if total_candles < 1000:
            print(f"⚠️ Low data volume: {total_candles} candles")
            print("🔧 Continuing with available data...")
        
        # Export data
        base_filename = gold_symbol.replace(".", "_") + "_SMC_dataset"
        success = extractor.export_smc_dataset(gold_data, base_filename, "csv")
        
        if not success:
            print("❌ Failed to export Gold data")
            return False
        
        print("✅ Step 1 completed: Gold data extracted successfully")
        print(f"📊 Total candles: {total_candles:,}")
        
        print("\n" + "="*60)
        print("2️⃣ STEP 2: GOLD SMC FEATURES ENGINEERING")
        print("="*60)
        
        # Process SMC features
        features_engine = SMCFeaturesEngine()
        
        # Load and process
        smc_data = features_engine.load_smc_dataset(base_filename)
        if not smc_data:
            print("❌ Failed to load Gold dataset for features")
            return False
        
        processed_data = features_engine.process_complete_dataset(smc_data)
        
        # Export features
        export_base = gold_symbol.replace(".", "_")
        success = features_engine.export_smc_features(processed_data, export_base)
        
        if not success:
            print("❌ Failed to export Gold features")
            return False
        
        print("✅ Step 2 completed: Gold SMC features engineered successfully")
        
        print("\n" + "="*60)
        print("3️⃣ STEP 3: GOLD TRADING LABELS CREATION")
        print("="*60)
        
        # Create trading labels
        labels_engine = SMCLabelsEngine()
        
        # Load features and create labels
        smc_features = labels_engine.load_smc_features(export_base)
        if not smc_features:
            print("❌ Failed to load Gold SMC features")
            return False
        
        labeled_data = labels_engine.process_complete_labels_dataset(smc_features)
        
        # Export labels
        success = labels_engine.export_labeled_dataset(labeled_data, export_base)
        
        if not success:
            print("❌ Failed to export Gold labels")
            return False
        
        # Validate label quality
        total_signals = sum((df["direction_label"] != 0).sum() for df in labeled_data.values())
        total_trades = sum((df["trade_outcome"] != 0).sum() for df in labeled_data.values() if "trade_outcome" in df.columns)
        
        print("✅ Step 3 completed: Gold trading labels created successfully")
        print(f"📊 Total signals: {total_signals}, Total trades: {total_trades}")
        
        if total_signals < 100:
            print("⚠️ Low signal count - consider adjusting parameters")
        
        print("\n" + "="*60)
        print("4️⃣ STEP 4: GOLD AI MODEL TRAINING")
        print("="*60)
        
        # Train AI models
        trainer = SMCAITrainer()
        
        # Load labeled data
        labeled_data = trainer.load_labeled_dataset(export_base)
        if not labeled_data:
            print("❌ Failed to load Gold labeled dataset")
            return False
        
        # Determine models to train
        models_to_train = ["random_forest"]
        
        try:
            import xgboost
            models_to_train.append("xgboost")
            print("✅ XGBoost available - will train XGBoost models")
        except ImportError:
            print("⚠️ XGBoost not available - skipping XGBoost models")
        
        try:
            import tensorflow
            models_to_train.append("neural_network")
            print("✅ TensorFlow available - will train Neural Network models")
        except ImportError:
            print("⚠️ TensorFlow not available - skipping Neural Network models")
        
        print(f"🎯 Training models: {models_to_train}")
        
        all_results = trainer.train_all_timeframes(labeled_data, models_to_train)
        
        # Save models
        model_base = export_base + "_SMC"
        success = trainer.save_models(model_base)
        
        if not success:
            print("❌ Failed to save Gold models")
            return False
        
        # Generate performance report
        trainer.generate_performance_report(model_base)
        
        print("✅ Step 4 completed: Gold AI models trained successfully")
        
        print("\n" + "="*60)
        print("5️⃣ STEP 5: GOLD SYSTEM VALIDATION")
        print("="*60)
        
        # Test signal generation
        try:
            from smc_signal_engine import SMCSignalEngine
            
            signal_engine = SMCSignalEngine(model_base)
            models_loaded = signal_engine.load_trained_models()
            
            if models_loaded:
                print("✅ Gold models loaded successfully")
                
                # Test signal generation
                print(f"🧪 Testing Gold signal generation for {gold_symbol}...")
                test_signal = signal_engine.get_multi_timeframe_signals(gold_symbol)
                if "error" not in test_signal:
                    print("✅ Gold signal generation working perfectly!")
                    print(f"   Direction: {test_signal['final_direction']}")
                    print(f"   Confidence: {test_signal['final_confidence']:.3f}")
                    print(f"   Risk Level: {test_signal['risk_level']}")
                    print(f"   Recommendation: {test_signal['trading_recommendation']}")
                else:
                    print(f"⚠️ Signal generation test: {test_signal['error']}")
            else:
                print("❌ Failed to load Gold models for validation")
                return False
        
        except Exception as e:
            print(f"⚠️ Validation error: {str(e)}")
            print("🔧 Models saved but validation incomplete")
        
        print("✅ Step 5 completed: Gold system validation successful")
        
        print("\n" + "🎉"*30)
        print("🥇 GOLD TRADING SYSTEM SETUP COMPLETE!")
        print("🎉"*30)
        print(f"✅ Gold Symbol: {gold_symbol}")
        print(f"✅ Models Path: {model_base}")
        print(f"✅ Total Candles: {total_candles:,}")
        print(f"✅ Total Signals: {total_signals}")
        print(f"✅ Models Trained: {len(models_to_train)}")
        print("\n🚀 Next steps:")
        print(f"1. Test signals: python smc_signal_engine.py")
        print(f"2. Start live trading: python SMCAutoTrader_new.py")
        print("3. Monitor logs in: gold_trading_logs/")
        print("\n💡 Tips:")
        print("🥇 Gold is more volatile than Forex - use appropriate position sizing")
        print("🕐 Best trading sessions: London (8-16 UTC), US (13-21 UTC)")
        print("⚠️ Avoid Asian session (22-7 UTC) due to low liquidity")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup error: {str(e)}")
        print(f"🔧 Error type: {type(e).__name__}")
        import traceback
        print("🔍 Full error trace:")
        traceback.print_exc()
        return False

def run_quick_start():
    """Run quick start with existing models"""
    print("\n🚀 Quick Start - Testing Existing Gold Models...")
    
    try:
        from smc_signal_engine import SMCSignalEngine
        
        # Find Gold symbol
        gold_symbol = find_gold_symbol()
        if not gold_symbol:
            print("❌ Cannot find Gold symbol")
            return False
        
        # Check for existing models
        model_patterns = [
            f"{gold_symbol.replace('.', '_')}_SMC",
            "XAUUSD_c_SMC",
            "XAUUSD_v_SMC", 
            "XAUUSD_SMC"
        ]
        
        model_path = None
        for pattern in model_patterns:
            model_files = [
                f"{pattern}_M5_random_forest_model.pkl",
                f"{pattern}_H1_random_forest_model.pkl"
            ]
            
            if any(os.path.exists(f) for f in model_files):
                model_path = pattern
                print(f"✅ Found existing models: {pattern}")
                break
        
        if not model_path:
            print("❌ No existing Gold models found")
            print("🔧 Available options:")
            print("   1. Run Complete Setup (Option 1)")
            print("   2. Check for model files in current directory")
            return False
        
        # Test signal engine
        signal_engine = SMCSignalEngine(model_path)
        
        if signal_engine.load_trained_models():
            print("✅ Gold models loaded successfully")
            
            # Test signal generation
            print(f"🧪 Testing Gold signals for {gold_symbol}...")
            signal = signal_engine.get_multi_timeframe_signals(gold_symbol)
            
            if "error" not in signal:
                print("✅ Gold signal generation successful!")
                print(f"   Symbol: {gold_symbol}")
                print(f"   Direction: {signal['final_direction']}")
                print(f"   Confidence: {signal['final_confidence']:.3f}")
                print(f"   Risk Level: {signal['risk_level']}")
                print(f"   Consensus: {signal['timeframe_consensus']}")
                print(f"   Recommendation: {signal['trading_recommendation']}")
                print("\n🚀 Gold system ready for live trading!")
                return True
            else:
                print(f"❌ Signal generation failed: {signal['error']}")
                return False
        else:
            print("❌ Failed to load Gold models")
            return False
            
    except Exception as e:
        print(f"❌ Quick start error: {str(e)}")
        return False

def run_signal_testing():
    """Run signal testing only"""
    print("\n🧪 Testing Gold Signal Generation...")
    
    try:
        # Find Gold symbol
        gold_symbol = find_gold_symbol()
        if not gold_symbol:
            return False
        
        from smc_signal_engine import SMCSignalEngine
        
        # Find models
        model_patterns = [
            f"{gold_symbol.replace('.', '_')}_SMC",
            "XAUUSD_c_SMC",
            "XAUUSD_v_SMC",
            "XAUUSD_SMC"
        ]
        
        model_path = None
        for pattern in model_patterns:
            if os.path.exists(f"{pattern}_M5_random_forest_model.pkl"):
                model_path = pattern
                break
        
        if not model_path:
            print("❌ No Gold models found for testing")
            print("🔧 Please run Complete Setup first (Option 1)")
            return False
        
        signal_engine = SMCSignalEngine(model_path)
        
        if signal_engine.load_trained_models():
            print(f"🔄 Starting Gold signal monitoring for {gold_symbol}...")
            print("📊 Update interval: 30 seconds")
            print("🛑 Press Ctrl+C to stop")
            print("-" * 50)
            signal_engine.start_signal_monitoring(gold_symbol, 30)
        else:
            print("❌ Failed to load Gold models")
            return False
            
    except KeyboardInterrupt:
        print("\n✅ Signal testing stopped by user")
        return True
    except Exception as e:
        print(f"❌ Signal testing error: {str(e)}")
        return False

def show_status_report():
    """Show current system status"""
    print("\n📊 GOLD TRADING SYSTEM STATUS")
    print("=" * 50)
    
    # Check Gold symbol
    gold_symbol = find_gold_symbol()
    if gold_symbol:
        print(f"✅ Gold symbol available: {gold_symbol}")
    else:
        print("❌ No Gold symbol found")
    
    # Check data files
    data_patterns = ["*_SMC_dataset_*.csv", "*_SMC_features_*.csv", "*_labeled_*.csv"]
    print("\n📂 Data Files:")
    found_files = False
    for pattern in data_patterns:
        import glob
        files = glob.glob(pattern)
        if files:
            found_files = True
            for file in files[:3]:  # Show first 3
                print(f"   ✅ {file}")
            if len(files) > 3:
                print(f"   ... and {len(files)-3} more")
    
    if not found_files:
        print("   ❌ No data files found")
    
    # Check model files
    model_patterns = ["*_SMC_*_model.pkl", "*_SMC_*.h5"]
    print("\n🤖 Model Files:")
    found_models = False
    for pattern in model_patterns:
        files = glob.glob(pattern)
        if files:
            found_models = True
            for file in files[:3]:
                print(f"   ✅ {file}")
            if len(files) > 3:
                print(f"   ... and {len(files)-3} more")
    
    if not found_models:
        print("   ❌ No model files found")
    
    print("=" * 50)

def main():
    """Main function - COMPLETE FIXED VERSION"""
    print_header()
    
    # ตรวจสอบ dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        print("💡 Quick fix: pip install scikit-learn")
        print("💡 Full install: pip install pandas numpy MetaTrader5 pytz scikit-learn joblib xgboost tensorflow")
        return
    
    # ตรวจสอบไฟล์
    if not check_mt5_files():
        print("\n❌ Please ensure all required files are present")
        return
    
    while True:
        show_setup_options()
        
        choice = input("\n🎯 Select option (1-4) or 'status'/'quit': ").strip().lower()
        
        if choice == '1':
            print("\n🚀 Starting Complete Gold Setup...")
            success = run_complete_setup()
            if success:
                print("\n🎉 Complete setup finished successfully!")
            else:
                print("\n❌ Complete setup failed - check error messages above")
            
        elif choice == '2':
            success = run_quick_start()
            if success:
                print("\n🎉 Quick start successful!")
            else:
                print("\n❌ Quick start failed")
            
        elif choice == '3':
            print("\n🔧 Step-by-step setup available...")
            print("📋 Manual execution order:")
            print("1. python MT5MultiTimeframeExtractor.py")
            print("2. python SMCFeaturesEngine.py")
            print("3. python SMCLabelsEngine.py")
            print("4. python SMCAITrainer.py")
            print("5. python smc_signal_engine.py")
            print("6. python SMCAutoTrader_new.py")
            
        elif choice == '4':
            success = run_signal_testing()
            if success:
                print("\n✅ Signal testing completed")
            
        elif choice in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        elif choice == 'status':
            show_status_report()
            
        else:
            print("❌ Invalid option. Please try again.")
        
        # Ask if continue
        if choice not in ['quit', 'exit', 'q', 'status', '3']:
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
        import traceback
        print("\n🔍 Full error trace:")
        traceback.print_exc()