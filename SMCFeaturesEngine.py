import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SMCFeaturesEngine:
    """
    Smart Money Concepts Features Engineering System
    Created by อาจารย์ฟินิกซ์ - Professional SMC Analysis for AI Trading
    
    Features:
    - Market Structure Analysis (HH, HL, LH, LL)
    - Change of Character (CHoCH) / Break of Structure (BOS)
    - Order Blocks (OB) Detection
    - Fair Value Gaps (FVG) Identification
    - Liquidity Zones
    - Multi-Timeframe Context
    """
    
    def __init__(self):
        """Initialize SMC Features Engine"""
        # Swing detection parameters
        self.swing_period = 5          # Period for swing highs/lows
        self.structure_period = 20     # Period for market structure
        
        # Order Block parameters
        self.ob_lookback = 10          # Lookback for OB detection
        self.ob_min_size = 0.0001      # Minimum OB size in price
        
        # Fair Value Gap parameters
        self.fvg_min_size = 0.0001     # Minimum FVG size
        
        # Liquidity parameters
        self.liquidity_period = 50     # Period for liquidity zones
        
        print("🚀 SMC Features Engine Initialized")
        print("📊 Ready for Professional SMC Analysis")
    
    def load_smc_dataset(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load complete SMC dataset from CSV files"""
        timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        smc_data = {}
        
        print("📂 Loading SMC Dataset...")
        print("-" * 40)
        
        for tf in timeframes:
            try:
                filename = f"{base_filename}_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                smc_data[tf] = df
                print(f"✅ {tf:>3}: {len(df):,} candles loaded from {filename}")
            except FileNotFoundError:
                print(f"❌ {tf:>3}: File not found - {filename}")
            except Exception as e:
                print(f"❌ {tf:>3}: Error loading - {str(e)}")
        
        print("-" * 40)
        total_candles = sum(len(df) for df in smc_data.values())
        print(f"📊 Total: {len(smc_data)} timeframes | {total_candles:,} candles")
        
        return smc_data
    
    def detect_swing_points(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Detect Swing Highs and Swing Lows
        Foundation for Market Structure Analysis
        """
        if period is None:
            period = self.swing_period
        
        df = df.copy()
        
        # Initialize swing columns
        df['swing_high'] = 0
        df['swing_low'] = 0
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        # Detect swing highs
        for i in range(period, len(df) - period):
            # Check if current high is highest in the period
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                # Ensure it's higher than neighbors
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i+1]):
                    df.iloc[i, df.columns.get_loc('swing_high')] = 1
                    df.iloc[i, df.columns.get_loc('swing_high_price')] = df['high'].iloc[i]
        
        # Detect swing lows
        for i in range(period, len(df) - period):
            # Check if current low is lowest in the period
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                # Ensure it's lower than neighbors
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i+1]):
                    df.iloc[i, df.columns.get_loc('swing_low')] = 1
                    df.iloc[i, df.columns.get_loc('swing_low_price')] = df['low'].iloc[i]
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Market Structure: Higher Highs, Higher Lows, Lower Highs, Lower Lows
        """
        df = df.copy()
        
        # Get swing points
        swing_highs = df[df['swing_high'] == 1]['swing_high_price'].dropna()
        swing_lows = df[df['swing_low'] == 1]['swing_low_price'].dropna()
        
        # Initialize structure columns
        df['higher_high'] = 0
        df['higher_low'] = 0
        df['lower_high'] = 0
        df['lower_low'] = 0
        df['market_structure'] = 0  # 1=Bullish, -1=Bearish, 0=Neutral
        
        # Analyze swing highs for HH/LH
        if len(swing_highs) >= 2:
            swing_high_indices = df[df['swing_high'] == 1].index
            for i in range(1, len(swing_highs)):
                current_idx = swing_high_indices[i]
                current_high = swing_highs.iloc[i]
                previous_high = swing_highs.iloc[i-1]
                
                if current_high > previous_high:
                    df.loc[current_idx, 'higher_high'] = 1
                else:
                    df.loc[current_idx, 'lower_high'] = 1
        
        # Analyze swing lows for HL/LL
        if len(swing_lows) >= 2:
            swing_low_indices = df[df['swing_low'] == 1].index
            for i in range(1, len(swing_lows)):
                current_idx = swing_low_indices[i]
                current_low = swing_lows.iloc[i]
                previous_low = swing_lows.iloc[i-1]
                
                if current_low > previous_low:
                    df.loc[current_idx, 'higher_low'] = 1
                else:
                    df.loc[current_idx, 'lower_low'] = 1
        
        # Determine overall market structure
        structure_window = 20
        for i in range(structure_window, len(df)):
            window_data = df.iloc[i-structure_window:i]
            
            hh_count = window_data['higher_high'].sum()
            hl_count = window_data['higher_low'].sum()
            lh_count = window_data['lower_high'].sum()
            ll_count = window_data['lower_low'].sum()
            
            bullish_signals = hh_count + hl_count
            bearish_signals = lh_count + ll_count
            
            if bullish_signals > bearish_signals and bullish_signals > 0:
                df.iloc[i, df.columns.get_loc('market_structure')] = 1
            elif bearish_signals > bullish_signals and bearish_signals > 0:
                df.iloc[i, df.columns.get_loc('market_structure')] = -1
            else:
                df.iloc[i, df.columns.get_loc('market_structure')] = 0
        
        return df
    
    def detect_choch_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Change of Character (CHoCH) and Break of Structure (BOS)
        """
        df = df.copy()
        
        # Initialize CHoCH/BOS columns
        df['choch'] = 0
        df['bos'] = 0
        df['structure_break'] = 0  # Combined signal
        
        # Get significant swing points
        significant_highs = df[df['swing_high'] == 1].copy()
        significant_lows = df[df['swing_low'] == 1].copy()
        
        # Detect BOS (Break of Structure)
        # Bullish BOS: Price breaks above recent swing high
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            
            # Look for recent swing high to break
            recent_highs = significant_highs[significant_highs.index < df.index[i]]
            if len(recent_highs) > 0:
                recent_high = recent_highs['swing_high_price'].iloc[-1]
                if current_price > recent_high:
                    # Check if this is a new high break
                    prev_breaks = df.iloc[:i]['bos'].sum()
                    if prev_breaks == 0 or current_price > df.iloc[:i]['high'].max():
                        df.iloc[i, df.columns.get_loc('bos')] = 1
                        df.iloc[i, df.columns.get_loc('structure_break')] = 1
            
            # Look for recent swing low to break
            recent_lows = significant_lows[significant_lows.index < df.index[i]]
            if len(recent_lows) > 0:
                recent_low = recent_lows['swing_low_price'].iloc[-1]
                if current_price < recent_low:
                    # Check if this is a new low break
                    prev_breaks = df.iloc[:i]['bos'].sum()
                    if prev_breaks == 0 or current_price < df.iloc[:i]['low'].min():
                        df.iloc[i, df.columns.get_loc('bos')] = -1
                        df.iloc[i, df.columns.get_loc('structure_break')] = -1
        
        # Detect CHoCH (Change of Character)
        # This occurs when market structure changes from bullish to bearish or vice versa
        structure_changes = df['market_structure'].diff()
        choch_points = structure_changes[abs(structure_changes) >= 2].index
        
        for idx in choch_points:
            if idx in df.index:
                df.loc[idx, 'choch'] = df.loc[idx, 'market_structure']
        
        return df
    
    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks (OB) - Candles before significant moves
        """
        df = df.copy()
        
        # Initialize OB columns
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        df['ob_high'] = np.nan
        df['ob_low'] = np.nan
        
        lookback = self.ob_lookback
        
        for i in range(lookback, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Look back for potential OB candles
            for j in range(1, lookback + 1):
                ob_idx = i - j
                ob_candle = df.iloc[ob_idx]
                
                # Check for Bullish Order Block
                # Last bearish candle before significant bullish move
                if (ob_candle['close'] < ob_candle['open'] and  # Bearish candle
                    current_close > ob_candle['high'] and       # Price broke above OB high
                    current_close > df['close'].iloc[i-1]):     # Upward movement
                    
                    # Ensure it's a significant move
                    move_size = current_close - ob_candle['high']
                    if move_size >= self.ob_min_size:
                        df.iloc[ob_idx, df.columns.get_loc('bullish_ob')] = 1
                        df.iloc[ob_idx, df.columns.get_loc('ob_high')] = ob_candle['high']
                        df.iloc[ob_idx, df.columns.get_loc('ob_low')] = ob_candle['low']
                        break
                
                # Check for Bearish Order Block
                # Last bullish candle before significant bearish move
                if (ob_candle['close'] > ob_candle['open'] and  # Bullish candle
                    current_close < ob_candle['low'] and        # Price broke below OB low
                    current_close < df['close'].iloc[i-1]):     # Downward movement
                    
                    # Ensure it's a significant move
                    move_size = ob_candle['low'] - current_close
                    if move_size >= self.ob_min_size:
                        df.iloc[ob_idx, df.columns.get_loc('bearish_ob')] = 1
                        df.iloc[ob_idx, df.columns.get_loc('ob_high')] = ob_candle['high']
                        df.iloc[ob_idx, df.columns.get_loc('ob_low')] = ob_candle['low']
                        break
        
        return df
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) - Imbalances in price action
        """
        df = df.copy()
        
        # Initialize FVG columns
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan
        df['fvg_size'] = np.nan
        
        for i in range(2, len(df)):
            # Get three consecutive candles
            candle1 = df.iloc[i-2]  # First candle
            candle2 = df.iloc[i-1]  # Middle candle (gap candle)
            candle3 = df.iloc[i]    # Third candle
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if (candle3['low'] > candle1['high'] and 
                candle2['close'] > candle2['open']):  # Middle candle is bullish
                
                gap_size = candle3['low'] - candle1['high']
                if gap_size >= self.fvg_min_size:
                    df.iloc[i-1, df.columns.get_loc('bullish_fvg')] = 1
                    df.iloc[i-1, df.columns.get_loc('fvg_high')] = candle3['low']
                    df.iloc[i-1, df.columns.get_loc('fvg_low')] = candle1['high']
                    df.iloc[i-1, df.columns.get_loc('fvg_size')] = gap_size
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            if (candle3['high'] < candle1['low'] and 
                candle2['close'] < candle2['open']):  # Middle candle is bearish
                
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= self.fvg_min_size:
                    df.iloc[i-1, df.columns.get_loc('bearish_fvg')] = 1
                    df.iloc[i-1, df.columns.get_loc('fvg_high')] = candle1['low']
                    df.iloc[i-1, df.columns.get_loc('fvg_low')] = candle3['high']
                    df.iloc[i-1, df.columns.get_loc('fvg_size')] = gap_size
        
        return df
    
    def detect_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Liquidity Zones - Areas where stops are likely placed
        """
        df = df.copy()
        
        # Initialize liquidity columns
        df['buy_liquidity'] = 0
        df['sell_liquidity'] = 0
        df['liquidity_strength'] = 0
        
        period = self.liquidity_period
        
        for i in range(period, len(df)):
            window_data = df.iloc[i-period:i]
            current_price = df['close'].iloc[i]
            
            # Find recent highs (potential sell liquidity)
            recent_highs = window_data[window_data['swing_high'] == 1]
            if len(recent_highs) > 0:
                highest_point = recent_highs['swing_high_price'].max()
                # If price is near recent high, mark as sell liquidity zone
                if abs(current_price - highest_point) / highest_point < 0.001:  # Within 0.1%
                    df.iloc[i, df.columns.get_loc('sell_liquidity')] = 1
                    df.iloc[i, df.columns.get_loc('liquidity_strength')] = len(recent_highs)
            
            # Find recent lows (potential buy liquidity)
            recent_lows = window_data[window_data['swing_low'] == 1]
            if len(recent_lows) > 0:
                lowest_point = recent_lows['swing_low_price'].min()
                # If price is near recent low, mark as buy liquidity zone
                if abs(current_price - lowest_point) / lowest_point < 0.001:  # Within 0.1%
                    df.iloc[i, df.columns.get_loc('buy_liquidity')] = 1
                    df.iloc[i, df.columns.get_loc('liquidity_strength')] = len(recent_lows)
        
        return df
    
    def add_advanced_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced SMC-derived features for ML training
        """
        df = df.copy()
        
        # Market structure momentum
        df['structure_momentum'] = df['market_structure'].rolling(10).mean()
        
        # Order block strength (distance and time)
        df['ob_distance'] = np.nan
        df['ob_age'] = 0
        
        # Track OB distances and age
        for i in range(len(df)):
            if df['bullish_ob'].iloc[i] == 1 or df['bearish_ob'].iloc[i] == 1:
                ob_price = df['ob_high'].iloc[i] if not pd.isna(df['ob_high'].iloc[i]) else df['ob_low'].iloc[i]
                current_price = df['close'].iloc[i]
                df.iloc[i, df.columns.get_loc('ob_distance')] = abs(current_price - ob_price) / current_price
        
        # FVG fill rate (how often FVGs get filled)
        df['fvg_unfilled'] = 0
        for i in range(len(df)):
            if df['bullish_fvg'].iloc[i] == 1:
                fvg_high = df['fvg_high'].iloc[i]
                fvg_low = df['fvg_low'].iloc[i]
                # Check if FVG gets filled in future candles
                filled = False
                for j in range(i+1, min(i+50, len(df))):  # Check next 50 candles
                    if df['low'].iloc[j] <= fvg_high and df['high'].iloc[j] >= fvg_low:
                        filled = True
                        break
                if not filled:
                    df.iloc[i, df.columns.get_loc('fvg_unfilled')] = 1
        
        # Structure break strength
        df['break_strength'] = abs(df['structure_break']) * df['range_pct']
        
        # Confluence zones (multiple SMC factors)
        df['smc_confluence'] = (
            abs(df['structure_break']) +
            df['bullish_ob'] + df['bearish_ob'] +
            df['bullish_fvg'] + df['bearish_fvg'] +
            df['buy_liquidity'] + df['sell_liquidity']
        )
        
        return df
    
    def process_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Complete SMC analysis for a single timeframe
        """
        print(f"🔄 Processing {timeframe} SMC features...")
        
        # Step 1: Detect swing points
        df = self.detect_swing_points(df)
        swing_highs = df['swing_high'].sum()
        swing_lows = df['swing_low'].sum()
        print(f"   📊 Swing Points: {swing_highs} highs, {swing_lows} lows")
        
        # Step 2: Analyze market structure
        df = self.analyze_market_structure(df)
        structure_changes = abs(df['market_structure'].diff()).sum()
        print(f"   📈 Market Structure: {structure_changes} changes detected")
        
        # Step 3: Detect CHoCH/BOS
        df = self.detect_choch_bos(df)
        choch_count = abs(df['choch']).sum()
        bos_count = abs(df['bos']).sum()
        print(f"   🔄 CHoCH: {choch_count}, BOS: {bos_count}")
        
        # Step 4: Detect Order Blocks
        df = self.detect_order_blocks(df)
        bullish_ob = df['bullish_ob'].sum()
        bearish_ob = df['bearish_ob'].sum()
        print(f"   📦 Order Blocks: {bullish_ob} bullish, {bearish_ob} bearish")
        
        # Step 5: Detect Fair Value Gaps
        df = self.detect_fair_value_gaps(df)
        bullish_fvg = df['bullish_fvg'].sum()
        bearish_fvg = df['bearish_fvg'].sum()
        print(f"   🕳️ Fair Value Gaps: {bullish_fvg} bullish, {bearish_fvg} bearish")
        
        # Step 6: Detect Liquidity Zones
        df = self.detect_liquidity_zones(df)
        buy_liquidity = df['buy_liquidity'].sum()
        sell_liquidity = df['sell_liquidity'].sum()
        print(f"   💧 Liquidity Zones: {buy_liquidity} buy, {sell_liquidity} sell")
        
        # Step 7: Add advanced features
        df = self.add_advanced_smc_features(df)
        confluence_zones = (df['smc_confluence'] >= 2).sum()
        print(f"   🎯 Confluence Zones: {confluence_zones}")
        
        print(f"✅ {timeframe} SMC analysis complete!")
        
        return df
    
    def process_complete_dataset(self, smc_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process complete SMC dataset with all timeframes
        """
        print("🚀 Processing Complete SMC Dataset")
        print("=" * 50)
        
        processed_data = {}
        
        for timeframe, df in smc_data.items():
            print(f"\n📊 Processing {timeframe} ({len(df):,} candles)...")
            processed_df = self.process_single_timeframe(df, timeframe)
            processed_data[timeframe] = processed_df
        
        print("\n" + "=" * 50)
        print("📋 SMC FEATURES SUMMARY")
        print("=" * 50)
        
        # Summary statistics
        for timeframe, df in processed_data.items():
            total_signals = (
                df['swing_high'].sum() + df['swing_low'].sum() +
                abs(df['choch']).sum() + abs(df['bos']).sum() +
                df['bullish_ob'].sum() + df['bearish_ob'].sum() +
                df['bullish_fvg'].sum() + df['bearish_fvg'].sum() +
                df['buy_liquidity'].sum() + df['sell_liquidity'].sum()
            )
            signal_density = (total_signals / len(df)) * 100
            
            print(f"{timeframe:>3}: {total_signals:>4} total signals ({signal_density:.1f}% density)")
        
        print("🎉 Complete SMC Analysis Finished!")
        
        return processed_data
    
    def export_smc_features(self, processed_data: Dict[str, pd.DataFrame], base_filename: str) -> bool:
        """
        Export processed SMC features dataset
        """
        try:
            print(f"\n💾 Exporting SMC Features Dataset...")
            print("-" * 40)
            
            exported_files = []
            
            for timeframe, df in processed_data.items():
                filename = f"{base_filename}_SMC_features_{timeframe}.csv"
                df.to_csv(filename)
                exported_files.append(filename)
                print(f"✅ {timeframe}: {filename} ({len(df):,} rows, {len(df.columns)} features)")
            
            # Create features summary
            feature_summary = {}
            sample_df = list(processed_data.values())[0]
            
            smc_features = [col for col in sample_df.columns if any(keyword in col.lower() 
                           for keyword in ['swing', 'structure', 'choch', 'bos', 'ob', 'fvg', 'liquidity', 'smc'])]
            
            feature_summary = {
                'total_features': len(sample_df.columns),
                'smc_features': len(smc_features),
                'smc_feature_list': smc_features,
                'timeframes': list(processed_data.keys()),
                'export_date': pd.Timestamp.now().isoformat()
            }
            
            import json
            summary_file = f"{base_filename}_SMC_features_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(feature_summary, f, indent=2, default=str)
            
            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 SMC Features Dataset Complete!")
            print(f"📊 {len(exported_files)} files | {len(smc_features)} SMC features")
            
            return True
            
        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return False

# Usage Example
if __name__ == "__main__":
    print("🚀 SMC Features Engineering System")
    print("=" * 50)
    
    # Initialize engine
    engine = SMCFeaturesEngine()
    
    # Load dataset
    print("\n📂 Loading SMC Dataset...")
    smc_data = engine.load_smc_dataset("EURUSD_c_SMC_dataset")
    
    if smc_data:
        # Process complete dataset
        processed_data = engine.process_complete_dataset(smc_data)
        
        # Export features
        engine.export_smc_features(processed_data, "EURUSD_c")
        
        print("\n🎯 Next Steps:")
        print("1. ✅ SMC Features Ready")
        print("2. 🎯 Create Training Labels") 
        print("3. 🚀 Train AI Models")
        print("4. 📊 Backtest & Optimize")
        
    else:
        print("❌ No SMC data loaded. Please check file paths.")