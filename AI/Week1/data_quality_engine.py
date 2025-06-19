import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataQualityEngine:
    """
    Institutional-Grade Data Quality Control System
    ใช้ตรวจสอบและทำความสะอาดข้อมูล tick data แบบธนาคาร
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.setup_logger()
        
        # Quality Control Thresholds (Based on XAUUSD characteristics)
        self.max_spread_pips = 50  # Maximum acceptable spread
        self.max_price_jump = 0.005  # 0.5% maximum price jump
        self.min_volume = 0.01  # Minimum volume threshold
        self.stale_data_threshold = 60  # seconds
        
        # Quality Metrics
        self.quality_stats = {
            'total_ticks': 0,
            'invalid_ticks': 0,
            'gap_fills': 0,
            'spread_outliers': 0,
            'price_outliers': 0,
            'duplicate_removed': 0
        }
    
    def setup_logger(self):
        """Setup institutional-grade logging"""
        self.logger = logging.getLogger(f'DataQuality_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def validate_tick_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main validation pipeline - ตรวจสอบข้อมูล tick แบบครอบคลุม
        """
        self.logger.info(f"Starting validation for {len(df)} ticks")
        original_count = len(df)
        
        # Step 1: Basic data validation
        df = self._validate_basic_structure(df)
        
        # Step 2: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 3: Price validation
        df = self._validate_prices(df)
        
        # Step 4: Spread validation
        df = self._validate_spreads(df)
        
        # Step 5: Volume validation
        df = self._validate_volume(df)
        
        # Step 6: Time gap analysis
        df = self._detect_time_gaps(df)
        
        # Step 7: Statistical outlier detection
        df = self._detect_statistical_outliers(df)
        
        # Update quality stats
        self.quality_stats['total_ticks'] = original_count
        self.quality_stats['invalid_ticks'] = original_count - len(df)
        
        quality_score = self._calculate_quality_score()
        
        self.logger.info(f"Validation complete. Quality Score: {quality_score:.2f}")
        
        return df, self.quality_stats
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบโครงสร้างข้อมูลพื้นฐาน"""
        required_columns = ['timestamp', 'bid', 'ask', 'volume']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with null values
        df = df.dropna(subset=required_columns)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """ลบข้อมูลซ้ำซ้อน"""
        before_count = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'bid', 'ask'])
        duplicates_removed = before_count - len(df)
        
        self.quality_stats['duplicate_removed'] = duplicates_removed
        if duplicates_removed > 0:
            self.logger.warning(f"Removed {duplicates_removed} duplicate ticks")
        
        return df
    
    def _validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบความผิดปกติของราคา"""
        # Calculate mid price
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Check for zero or negative prices
        invalid_prices = (df['bid'] <= 0) | (df['ask'] <= 0) | (df['ask'] <= df['bid'])
        
        if invalid_prices.sum() > 0:
            self.logger.warning(f"Found {invalid_prices.sum()} invalid price relationships")
            df = df[~invalid_prices]
        
        # Price jump detection
        if len(df) > 1:
            df['price_change'] = df['mid_price'].pct_change().abs()
            price_outliers = df['price_change'] > self.max_price_jump
            
            if price_outliers.sum() > 0:
                self.quality_stats['price_outliers'] = price_outliers.sum()
                self.logger.warning(f"Found {price_outliers.sum()} price jump outliers")
                df = df[~price_outliers]
        
        return df
    
    def _validate_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบ spread ผิดปกติ"""
        df['spread'] = df['ask'] - df['bid']
        df['spread_pips'] = df['spread'] * 10  # Assuming XAUUSD pip value
        
        # Identify spread outliers
        spread_outliers = df['spread_pips'] > self.max_spread_pips
        
        if spread_outliers.sum() > 0:
            self.quality_stats['spread_outliers'] = spread_outliers.sum()
            self.logger.warning(f"Found {spread_outliers.sum()} spread outliers")
            df = df[~spread_outliers]
        
        return df
    
    def _validate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบ volume"""
        if 'volume' in df.columns:
            # Remove zero or negative volume
            invalid_volume = df['volume'] <= 0
            if invalid_volume.sum() > 0:
                self.logger.warning(f"Removed {invalid_volume.sum()} invalid volume ticks")
                df = df[~invalid_volume]
        
        return df
    
    def _detect_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจหา gap ในข้อมูล"""
        if len(df) > 1:
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
            
            # Identify large time gaps (> 5 minutes during market hours)
            large_gaps = df['time_diff'] > 300  # 5 minutes
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                self.logger.info(f"Detected {gap_count} time gaps > 5 minutes")
                # Mark gaps for potential filling
                df['has_gap'] = large_gaps
            else:
                df['has_gap'] = False
        
        return df
    
    def _detect_statistical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """ใช้ Statistical methods หา outliers"""
        if len(df) > 50:  # Need sufficient data
            # Z-score method for price changes
            if 'price_change' in df.columns:
                z_scores = np.abs((df['price_change'] - df['price_change'].mean()) / df['price_change'].std())
                statistical_outliers = z_scores > 4  # 4 standard deviations
                
                if statistical_outliers.sum() > 0:
                    self.logger.info(f"Removed {statistical_outliers.sum()} statistical outliers")
                    df = df[~statistical_outliers]
        
        return df
    
    def _calculate_quality_score(self) -> float:
        """คำนวณ Quality Score (0-100)"""
        if self.quality_stats['total_ticks'] == 0:
            return 0.0
        
        valid_ratio = (self.quality_stats['total_ticks'] - self.quality_stats['invalid_ticks']) / self.quality_stats['total_ticks']
        
        # Penalty for various issues
        penalty = 0
        if self.quality_stats['spread_outliers'] > 0:
            penalty += 5
        if self.quality_stats['price_outliers'] > 0:
            penalty += 10
        if self.quality_stats['duplicate_removed'] > 10:
            penalty += 5
        
        quality_score = max(0, (valid_ratio * 100) - penalty)
        return min(100, quality_score)
    
    def generate_quality_report(self) -> str:
        """สร้างรายงานคุณภาพข้อมูล"""
        quality_score = self._calculate_quality_score()
        
        report = f"""
========== DATA QUALITY REPORT ==========
Symbol: {self.symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUALITY METRICS:
- Quality Score: {quality_score:.2f}/100
- Total Ticks: {self.quality_stats['total_ticks']:,}
- Valid Ticks: {self.quality_stats['total_ticks'] - self.quality_stats['invalid_ticks']:,}
- Invalid Ticks: {self.quality_stats['invalid_ticks']:,}

ISSUES DETECTED:
- Duplicates Removed: {self.quality_stats['duplicate_removed']:,}
- Price Outliers: {self.quality_stats['price_outliers']:,}
- Spread Outliers: {self.quality_stats['spread_outliers']:,}
- Gap Fills: {self.quality_stats['gap_fills']:,}

QUALITY ASSESSMENT:
{self._get_quality_assessment(quality_score)}
==========================================
        """
        
        return report
    
    def _get_quality_assessment(self, score: float) -> str:
        """ประเมินคุณภาพข้อมูล"""
        if score >= 95:
            return "EXCELLENT - Institutional Grade Quality"
        elif score >= 85:
            return "GOOD - Acceptable for Production"
        elif score >= 70:
            return "FAIR - Requires Monitoring"
        else:
            return "POOR - Not Suitable for Trading"

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample XAUUSD tick data
    dates = pd.date_range(start='2024-01-01 09:00:00', periods=1000, freq='1s')
    base_price = 2000.0
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'bid': base_price + np.random.normal(0, 0.5, 1000),
        'ask': base_price + np.random.normal(0.2, 0.5, 1000),  # Ask slightly higher
        'volume': np.random.exponential(1, 1000)
    })
    
    # Add some intentional bad data for testing
    sample_data.loc[100, 'bid'] = 0  # Invalid price
    sample_data.loc[200, 'ask'] = sample_data.loc[200, 'bid'] - 1  # Invalid spread
    sample_data.loc[300:305] = sample_data.loc[300:305]  # Duplicates
    
    # Initialize and test the quality engine
    quality_engine = DataQualityEngine("XAUUSD")
    
    print("Testing Data Quality Engine...")
    cleaned_data, stats = quality_engine.validate_tick_data(sample_data)
    
    print(f"\nOriginal data: {len(sample_data)} ticks")
    print(f"Cleaned data: {len(cleaned_data)} ticks")
    print(f"Removed: {len(sample_data) - len(cleaned_data)} ticks")
    
    print(quality_engine.generate_quality_report())