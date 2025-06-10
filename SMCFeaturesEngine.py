import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SMCFeaturesEngine:
    """
    Smart Money Concepts Features Engineering System
    Created by อาจารย์ฟินิกซ์ - Professional SMC Analysis for AI Trading
    🥇 Optimized for Gold (XAUUSD.c) Trading
    
    Features:
    - Market Structure Analysis (HH, HL, LH, LL)
    - Change of Character (CHoCH) / Break of Structure (BOS)
    - Order Blocks (OB) Detection
    - Fair Value Gaps (FVG) Identification
    - Liquidity Zones
    - Multi-Timeframe Context
    - Gold-specific optimizations
    """
    
    def __init__(self):
        """Initialize SMC Features Engine with Gold-optimized parameters"""
        # 🥇 Gold-optimized swing detection parameters
        self.swing_period = 8          # Longer period for Gold volatility
        self.structure_period = 30     # Extended structure analysis
        
        # 🥇 Gold-optimized Order Block parameters
        self.ob_lookback = 15          # More lookback for Gold
        self.ob_min_size = 0.5         # Larger minimum size (Gold points)
        
        # 🥇 Gold-optimized Fair Value Gap parameters
        self.fvg_min_size = 0.3        # Larger minimum size (Gold points)
        
        # 🥇 Gold-optimized Liquidity parameters
        self.liquidity_period = 100    # Extended period for Gold
        
        # 🥇 Gold-specific thresholds
        self.gold_volatility_threshold = 2.0  # Higher volatility threshold
        self.gold_session_weight = {
            "london": 1.5,     # Higher weight for London session
            "us": 1.3,         # Higher weight for US session
            "asian": 0.7       # Lower weight for Asian session
        }
        
        print("🥇 SMC Features Engine Initialized for Gold Trading")
        print("📊 Optimized parameters for XAUUSD.c analysis")
    
    def load_smc_dataset(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """Load complete SMC dataset from CSV files"""
        timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        smc_data = {}
        
        print("📂 Loading Gold SMC Dataset...")
        print("-" * 40)
        
        for tf in timeframes:
            try:
                filename = f"{base_filename}_{tf}.csv"
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                smc_data[tf] = df
                
                # 🥇 Check if Gold data
                is_gold = False
                if 'symbol' in df.columns:
                    symbol = df['symbol'].iloc[0] if len(df) > 0 else ""
                    is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()
                
                symbol_type = "🥇 Gold" if is_gold else "📈 Forex"
                print(f"✅ {tf:>3}: {len(df):,} candles loaded from {filename} ({symbol_type})")
                
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
        🥇 Enhanced for Gold volatility
        """
        if period is None:
            period = self.swing_period
        
        df = df.copy()
        
        # Initialize swing columns
        df['swing_high'] = 0
        df['swing_low'] = 0
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        # 🥇 Gold-specific swing strength
        df['swing_strength'] = 0
        
        # Detect swing highs
        for i in range(period, len(df) - period):
            # Check if current high is highest in the period
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                # Ensure it's higher than neighbors
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i+1]):
                    
                    df.iloc[i, df.columns.get_loc('swing_high')] = 1
                    df.iloc[i, df.columns.get_loc('swing_high_price')] = df['high'].iloc[i]
                    
                    # 🥇 Calculate swing strength for Gold
                    swing_range = df['high'].iloc[i] - df['low'].iloc[i-period:i+period+1].min()
                    df.iloc[i, df.columns.get_loc('swing_strength')] = swing_range
        
        # Detect swing lows
        for i in range(period, len(df) - period):
            # Check if current low is lowest in the period
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                # Ensure it's lower than neighbors
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i+1]):
                    
                    df.iloc[i, df.columns.get_loc('swing_low')] = 1
                    df.iloc[i, df.columns.get_loc('swing_low_price')] = df['low'].iloc[i]
                    
                    # 🥇 Calculate swing strength for Gold
                    swing_range = df['high'].iloc[i-period:i+period+1].max() - df['low'].iloc[i]
                    df.iloc[i, df.columns.get_loc('swing_strength')] = swing_range
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Market Structure: Higher Highs, Higher Lows, Lower Highs, Lower Lows
        🥇 Enhanced for Gold market behavior
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
        
        # 🥇 Gold-specific structure strength
        df['structure_strength'] = 0
        
        # Analyze swing highs for HH/LH
        if len(swing_highs) >= 2:
            swing_high_indices = df[df['swing_high'] == 1].index
            for i in range(1, len(swing_highs)):
                current_idx = swing_high_indices[i]
                current_high = swing_highs.iloc[i]
                previous_high = swing_highs.iloc[i-1]
                
                if current_high > previous_high:
                    df.loc[current_idx, 'higher_high'] = 1
                    # 🥇 Structure strength for Gold
                    strength = (current_high - previous_high) / previous_high * 100
                    df.loc[current_idx, 'structure_strength'] = strength
                else:
                    df.loc[current_idx, 'lower_high'] = 1
                    # 🥇 Negative strength for lower high
                    strength = (previous_high - current_high) / previous_high * 100
                    df.loc[current_idx, 'structure_strength'] = -strength
        
        # Analyze swing lows for HL/LL
        if len(swing_lows) >= 2:
            swing_low_indices = df[df['swing_low'] == 1].index
            for i in range(1, len(swing_lows)):
                current_idx = swing_low_indices[i]
                current_low = swing_lows.iloc[i]
                previous_low = swing_lows.iloc[i-1]
                
                if current_low > previous_low:
                    df.loc[current_idx, 'higher_low'] = 1
                    # 🥇 Structure strength for Gold
                    strength = (current_low - previous_low) / previous_low * 100
                    df.loc[current_idx, 'structure_strength'] = strength
                else:
                    df.loc[current_idx, 'lower_low'] = 1
                    # 🥇 Negative strength for lower low
                    strength = (previous_low - current_low) / previous_low * 100
                    df.loc[current_idx, 'structure_strength'] = -strength
        
        # 🥇 Determine overall market structure with Gold-specific weighting
        structure_window = self.structure_period
        for i in range(structure_window, len(df)):
            window_data = df.iloc[i-structure_window:i]
            
            hh_count = window_data['higher_high'].sum()
            hl_count = window_data['higher_low'].sum()
            lh_count = window_data['lower_high'].sum()
            ll_count = window_data['lower_low'].sum()
            
            # 🥇 Weight by session for Gold
            session_weight = 1.0
            if 'is_london_session' in df.columns:
                if df['is_london_session'].iloc[i] == 1:
                    session_weight = self.gold_session_weight['london']
                elif df.get('is_us_session', pd.Series([0])).iloc[i] == 1:
                    session_weight = self.gold_session_weight['us']
                elif df.get('is_asian_session', pd.Series([0])).iloc[i] == 1:
                    session_weight = self.gold_session_weight['asian']
            
            bullish_signals = (hh_count + hl_count) * session_weight
            bearish_signals = (lh_count + ll_count) * session_weight
            
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
        🥇 Enhanced for Gold volatility patterns
        """
        df = df.copy()
        
        # Initialize CHoCH/BOS columns
        df['choch'] = 0
        df['bos'] = 0
        df['structure_break'] = 0  # Combined signal
        df['break_strength'] = 0   # 🥇 Strength of the break
        
        # Get significant swing points
        significant_highs = df[df['swing_high'] == 1].copy()
        significant_lows = df[df['swing_low'] == 1].copy()
        
        # 🥇 Gold-specific break detection with volatility consideration
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            previous_price = df['close'].iloc[i-1]
            price_change = abs(current_price - previous_price)
            
            # 🥇 Dynamic threshold based on Gold volatility
            volatility = df.get('atr_14', pd.Series([0])).iloc[i]
            min_break_size = max(0.5, volatility * 0.5) if volatility > 0 else 0.5
            
            # Look for recent swing high to break
            recent_highs = significant_highs[significant_highs.index < df.index[i]]
            if len(recent_highs) > 0:
                recent_high = recent_highs['swing_high_price'].iloc[-1]
                if current_price > recent_high and price_change >= min_break_size:
                    # Check if this is a new high break
                    prev_breaks = df.iloc[:i]['bos'].sum()
                    if prev_breaks == 0 or current_price > df.iloc[:i]['high'].max():
                        df.iloc[i, df.columns.get_loc('bos')] = 1
                        df.iloc[i, df.columns.get_loc('structure_break')] = 1
                        # 🥇 Calculate break strength
                        break_strength = (current_price - recent_high) / recent_high * 100
                        df.iloc[i, df.columns.get_loc('break_strength')] = break_strength
            
            # Look for recent swing low to break
            recent_lows = significant_lows[significant_lows.index < df.index[i]]
            if len(recent_lows) > 0:
                recent_low = recent_lows['swing_low_price'].iloc[-1]
                if current_price < recent_low and price_change >= min_break_size:
                    # Check if this is a new low break
                    prev_breaks = df.iloc[:i]['bos'].sum()
                    if prev_breaks == 0 or current_price < df.iloc[:i]['low'].min():
                        df.iloc[i, df.columns.get_loc('bos')] = -1
                        df.iloc[i, df.columns.get_loc('structure_break')] = -1
                        # 🥇 Calculate break strength
                        break_strength = (recent_low - current_price) / recent_low * 100
                        df.iloc[i, df.columns.get_loc('break_strength')] = break_strength
        
        # Detect CHoCH (Change of Character)
        structure_changes = df['market_structure'].diff()
        choch_points = structure_changes[abs(structure_changes) >= 2].index
        
        for idx in choch_points:
            if idx in df.index:
                df.loc[idx, 'choch'] = df.loc[idx, 'market_structure']
        
        return df
    
    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks (OB) - Candles before significant moves
        🥇 Enhanced for Gold with larger thresholds
        """
        df = df.copy()
        
        # Initialize OB columns
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        df['ob_high'] = np.nan
        df['ob_low'] = np.nan
        df['ob_strength'] = 0  # 🥇 OB strength measure
        
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
                if (ob_candle['close'] < ob_candle['open'] and  # Bearish candle
                    current_close > ob_candle['high'] and       # Price broke above OB high
                    current_close > df['close'].iloc[i-1]):     # Upward movement
                    
                    # 🥇 Ensure it's a significant move for Gold
                    move_size = current_close - ob_candle['high']
                    if move_size >= self.ob_min_size:
                        df.iloc[ob_idx, df.columns.get_loc('bullish_ob')] = 1
                        df.iloc[ob_idx, df.columns.get_loc('ob_high')] = ob_candle['high']
                        df.iloc[ob_idx, df.columns.get_loc('ob_low')] = ob_candle['low']
                        
                        # 🥇 Calculate OB strength
                        ob_strength = move_size / ob_candle['high'] * 100
                        df.iloc[ob_idx, df.columns.get_loc('ob_strength')] = ob_strength
                        break
                
                # Check for Bearish Order Block
                if (ob_candle['close'] > ob_candle['open'] and  # Bullish candle
                    current_close < ob_candle['low'] and        # Price broke below OB low
                    current_close < df['close'].iloc[i-1]):     # Downward movement
                    
                    # 🥇 Ensure it's a significant move for Gold
                    move_size = ob_candle['low'] - current_close
                    if move_size >= self.ob_min_size:
                        df.iloc[ob_idx, df.columns.get_loc('bearish_ob')] = 1
                        df.iloc[ob_idx, df.columns.get_loc('ob_high')] = ob_candle['high']
                        df.iloc[ob_idx, df.columns.get_loc('ob_low')] = ob_candle['low']
                        
                        # 🥇 Calculate OB strength
                        ob_strength = move_size / ob_candle['low'] * 100
                        df.iloc[ob_idx, df.columns.get_loc('ob_strength')] = ob_strength
                        break
        
        return df
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) - Imbalances in price action
        🥇 Enhanced for Gold with larger gap requirements
        """
        df = df.copy()
        
        # Initialize FVG columns
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan
        df['fvg_size'] = np.nan
        df['fvg_efficiency'] = np.nan  # 🥇 Gap efficiency measure
        
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
                    
                    # 🥇 Calculate FVG efficiency for Gold
                    efficiency = gap_size / candle2['range'] if candle2['range'] > 0 else 0
                    df.iloc[i-1, df.columns.get_loc('fvg_efficiency')] = efficiency
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            if (candle3['high'] < candle1['low'] and 
                candle2['close'] < candle2['open']):  # Middle candle is bearish
                
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= self.fvg_min_size:
                    df.iloc[i-1, df.columns.get_loc('bearish_fvg')] = 1
                    df.iloc[i-1, df.columns.get_loc('fvg_high')] = candle1['low']
                    df.iloc[i-1, df.columns.get_loc('fvg_low')] = candle3['high']
                    df.iloc[i-1, df.columns.get_loc('fvg_size')] = gap_size
                    
                    # 🥇 Calculate FVG efficiency for Gold
                    efficiency = gap_size / candle2['range'] if candle2['range'] > 0 else 0
                    df.iloc[i-1, df.columns.get_loc('fvg_efficiency')] = efficiency
        
        return df
    
    def detect_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Liquidity Zones - Areas where stops are likely placed
        🥇 Enhanced for Gold with extended analysis period
        """
        df = df.copy()
        
        # Initialize liquidity columns
        df['buy_liquidity'] = 0
        df['sell_liquidity'] = 0
        df['liquidity_strength'] = 0
        df['liquidity_proximity'] = np.nan  # 🥇 Distance to liquidity
        
        period = self.liquidity_period
        
        for i in range(period, len(df)):
            window_data = df.iloc[i-period:i]
            current_price = df['close'].iloc[i]
            
            # Find recent highs (potential sell liquidity)
            recent_highs = window_data[window_data['swing_high'] == 1]
            if len(recent_highs) > 0:
                highest_point = recent_highs['swing_high_price'].max()
                # 🥇 More sensitive proximity for Gold (0.2% instead of 0.1%)
                price_diff = abs(current_price - highest_point) / highest_point
                if price_diff < 0.002:
                    df.iloc[i, df.columns.get_loc('sell_liquidity')] = 1
                    df.iloc[i, df.columns.get_loc('liquidity_strength')] = len(recent_highs)
                    df.iloc[i, df.columns.get_loc('liquidity_proximity')] = price_diff
            
            # Find recent lows (potential buy liquidity)
            recent_lows = window_data[window_data['swing_low'] == 1]
            if len(recent_lows) > 0:
                lowest_point = recent_lows['swing_low_price'].min()
                # 🥇 More sensitive proximity for Gold
                price_diff = abs(current_price - lowest_point) / lowest_point
                if price_diff < 0.002:
                    df.iloc[i, df.columns.get_loc('buy_liquidity')] = 1
                    df.iloc[i, df.columns.get_loc('liquidity_strength')] = len(recent_lows)
                    df.iloc[i, df.columns.get_loc('liquidity_proximity')] = price_diff
        
        return df
    
    def add_advanced_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced SMC-derived features for ML training
        🥇 Enhanced with Gold-specific features
        """
        df = df.copy()
        
        # Market structure momentum (extended for Gold)
        df['structure_momentum'] = df['market_structure'].rolling(15).mean()
        
        # 🥇 Gold-specific volatility features
        if 'range_points' in df.columns:
            # Use Gold points if available
            df['gold_volatility'] = df['range_points'].rolling(20).std()
            df['gold_momentum'] = df['range_points'].rolling(10).mean()
        else:
            # Fallback to standard range
            df['gold_volatility'] = df['range'].rolling(20).std()
            df['gold_momentum'] = df['range'].rolling(10).mean()
        
        # Order block strength and aging
        df['ob_distance'] = np.nan
        df['ob_age'] = 0
        
        # Track OB distances and age
        for i in range(len(df)):
            if df['bullish_ob'].iloc[i] == 1 or df['bearish_ob'].iloc[i] == 1:
                ob_price = df['ob_high'].iloc[i] if not pd.isna(df['ob_high'].iloc[i]) else df['ob_low'].iloc[i]
                current_price = df['close'].iloc[i]
                if not pd.isna(ob_price) and ob_price > 0:
                    df.iloc[i, df.columns.get_loc('ob_distance')] = abs(current_price - ob_price) / current_price
        
        # FVG fill rate (how often FVGs get filled)
        df['fvg_unfilled'] = 0
        for i in range(len(df)):
            if df['bullish_fvg'].iloc[i] == 1:
                fvg_high = df['fvg_high'].iloc[i]
                fvg_low = df['fvg_low'].iloc[i]
                if not pd.isna(fvg_high) and not pd.isna(fvg_low):
                    # Check if FVG gets filled in future candles (reduced window for Gold)
                    filled = False
                    for j in range(i+1, min(i+30, len(df))):  # Reduced from 50 to 30 for Gold
                        if df['low'].iloc[j] <= fvg_high and df['high'].iloc[j] >= fvg_low:
                            filled = True
                            break
                    if not filled:
                        df.iloc[i, df.columns.get_loc('fvg_unfilled')] = 1
        
        # 🥇 Enhanced structure break strength for Gold
        df['break_strength'] = abs(df.get('structure_break', 0)) * df['range_pct']
        
        # 🥇 Gold-specific confluence zones
        df['smc_confluence'] = (
            abs(df.get('structure_break', 0)) +
            df['bullish_ob'] + df['bearish_ob'] +
            df['bullish_fvg'] + df['bearish_fvg'] +
            df['buy_liquidity'] + df['sell_liquidity']
        )
        
        # 🥇 Session-based feature weighting
        if 'is_london_session' in df.columns:
            df['session_weighted_confluence'] = df['smc_confluence'].copy()
            london_mask = df['is_london_session'] == 1
            us_mask = df.get('is_us_session', pd.Series([0] * len(df))) == 1
            asian_mask = df.get('is_asian_session', pd.Series([0] * len(df))) == 1
            
            df.loc[london_mask, 'session_weighted_confluence'] *= self.gold_session_weight['london']
            df.loc[us_mask, 'session_weighted_confluence'] *= self.gold_session_weight['us']
            df.loc[asian_mask, 'session_weighted_confluence'] *= self.gold_session_weight['asian']
        
        return df
    
    def process_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Complete SMC analysis for a single timeframe
        🥇 Enhanced for Gold with progress tracking
        """
        print(f"🔄 Processing {timeframe} Gold SMC features...")
        
        # Check if Gold data
        is_gold = False
        if 'symbol' in df.columns and len(df) > 0:
            symbol = df['symbol'].iloc[0]
            is_gold = "XAU" in str(symbol).upper() or "GOLD" in str(symbol).upper()
        
        if is_gold:
            print(f"   🥇 Gold-optimized parameters active for {timeframe}")
        
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
        
        # 🥇 Gold-specific feature summary
        if is_gold:
            if 'session_weighted_confluence' in df.columns:
                weighted_zones = (df['session_weighted_confluence'] >= 3).sum()
                print(f"   🥇 Session-Weighted Zones: {weighted_zones}")
        
        print(f"✅ {timeframe} SMC analysis complete!")
        
        return df
    
    def process_complete_dataset(self, smc_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process complete SMC dataset with all timeframes
        🥇 Enhanced for Gold with comprehensive reporting
        """
        print("🚀 Processing Complete Gold SMC Dataset")
        print("=" * 50)
        
        processed_data = {}
        
        for timeframe, df in smc_data.items():
            print(f"\n📊 Processing {timeframe} ({len(df):,} candles)...")
            processed_df = self.process_single_timeframe(df, timeframe)
            processed_data[timeframe] = processed_df
        
        print("\n" + "=" * 50)
        print("📋 GOLD SMC FEATURES SUMMARY")
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
            
            # 🥇 Check for Gold-specific features
            has_gold_features = 'session_weighted_confluence' in df.columns
            gold_indicator = "🥇" if has_gold_features else "📈"
            
            print(f"{gold_indicator} {timeframe:>3}: {total_signals:>4} total signals ({signal_density:.1f}% density)")
            
            if has_gold_features:
                session_signals = (df['session_weighted_confluence'] >= 3).sum()
                print(f"    Session-weighted signals: {session_signals}")
        
        print("🎉 Complete Gold SMC Analysis Finished!")
        
        return processed_data
    
    def export_smc_features(self, processed_data: Dict[str, pd.DataFrame], base_filename: str) -> bool:
        """
        Export processed SMC features dataset
        🥇 Enhanced for Gold with detailed feature documentation
        """
        try:
            print(f"\n💾 Exporting Gold SMC Features Dataset...")
            print("-" * 40)
            
            exported_files = []
            
            for timeframe, df in processed_data.items():
                filename = f"{base_filename}_SMC_features_{timeframe}.csv"
                df.to_csv(filename)
                exported_files.append(filename)
                
                # 🥇 Check for Gold features
                has_gold_features = 'session_weighted_confluence' in df.columns
                gold_indicator = "🥇 Gold" if has_gold_features else "📈 Forex"
                
                print(f"✅ {timeframe}: {filename} ({len(df):,} rows, {len(df.columns)} features) [{gold_indicator}]")
            
            # Create enhanced features summary
            sample_df = list(processed_data.values())[0]
            
            # Standard SMC features
            smc_features = [col for col in sample_df.columns if any(keyword in col.lower() 
                           for keyword in ['swing', 'structure', 'choch', 'bos', 'ob', 'fvg', 'liquidity', 'smc'])]
            
            # 🥇 Gold-specific features
            gold_features = [col for col in sample_df.columns if any(keyword in col.lower()
                            for keyword in ['gold', 'session', 'weighted', 'proximity', 'efficiency', 'strength'])]
            
            # 🥇 Check if this is Gold data
            is_gold_dataset = 'session_weighted_confluence' in sample_df.columns
            
            feature_summary = {
                'dataset_type': 'Gold SMC Features' if is_gold_dataset else 'Forex SMC Features',
                'total_features': len(sample_df.columns),
                'smc_features': len(smc_features),
                'gold_specific_features': len(gold_features) if is_gold_dataset else 0,
                'smc_feature_list': smc_features,
                'timeframes': list(processed_data.keys()),
                'export_date': pd.Timestamp.now().isoformat(),
                'optimization': 'Gold (XAUUSD.c)' if is_gold_dataset else 'General Forex'
            }
            
            # 🥇 Add Gold-specific documentation
            if is_gold_dataset:
                feature_summary.update({
                    'gold_feature_list': gold_features,
                    'session_weights': self.gold_session_weight,
                    'gold_parameters': {
                        'swing_period': self.swing_period,
                        'ob_lookback': self.ob_lookback,
                        'ob_min_size': self.ob_min_size,
                        'fvg_min_size': self.fvg_min_size,
                        'liquidity_period': self.liquidity_period
                    },
                    'trading_recommendations': {
                        'preferred_sessions': ['London', 'US'],
                        'avoid_sessions': ['Asian'],
                        'volatility_consideration': 'High - use larger stops and targets',
                        'point_value': '$0.01 per point movement'
                    }
                })
            
            import json
            summary_file = f"{base_filename}_SMC_features_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(feature_summary, f, indent=2, default=str)
            
            print(f"✅ Summary: {summary_file}")
            print("-" * 40)
            print(f"🎉 {'Gold' if is_gold_dataset else 'Forex'} SMC Features Dataset Complete!")
            print(f"📊 {len(exported_files)} files | {len(smc_features)} SMC features")
            
            if is_gold_dataset:
                print(f"🥇 {len(gold_features)} Gold-specific features included")
                print("🎯 Optimized for XAUUSD.c trading")
            
            return True
            
        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return False

# Usage Example for Gold
if __name__ == "__main__":
    print("🥇 SMC Features Engineering System for Gold")
    print("=" * 50)
    
    # Initialize engine with Gold optimizations
    engine = SMCFeaturesEngine()
    
    # Load Gold dataset
    print("\n📂 Loading Gold SMC Dataset...")
    smc_data = engine.load_smc_dataset("XAUUSD_c_SMC_dataset")
    
    if smc_data:
        # Process complete Gold dataset
        processed_data = engine.process_complete_dataset(smc_data)
        
        # Export Gold features
        engine.export_smc_features(processed_data, "XAUUSD_c")
        
        print("\n🎯 Next Steps for Gold Trading:")
        print("1. ✅ Gold SMC Features Ready")
        print("2. 🎯 Create Gold Trading Labels") 
        print("3. 🚀 Train Gold AI Models")
        print("4. 📊 Backtest & Optimize for Gold")
        print("5. 🥇 Deploy Gold Auto Trading")
        
    else:
        print("❌ No Gold SMC data loaded. Please check file paths.")
        print("🔧 Expected files: XAUUSD_c_SMC_dataset_[M5|M15|H1|H4|D1].csv")