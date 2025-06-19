import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
from scipy import interpolate
from enum import Enum

class GapType(Enum):
    """Types of data gaps"""
    MINOR_GAP = "MINOR_GAP"           # < 5 minutes
    MODERATE_GAP = "MODERATE_GAP"     # 5-30 minutes  
    MAJOR_GAP = "MAJOR_GAP"           # 30 minutes - 2 hours
    SESSION_GAP = "SESSION_GAP"       # Between sessions
    WEEKEND_GAP = "WEEKEND_GAP"       # Weekend closure
    HOLIDAY_GAP = "HOLIDAY_GAP"       # Holiday closure
    DATA_OUTAGE = "DATA_OUTAGE"       # Technical issues

class FillMethod(Enum):
    """Data filling methods"""
    FORWARD_FILL = "FORWARD_FILL"
    INTERPOLATION = "INTERPOLATION"
    LAST_KNOWN = "LAST_KNOWN"
    SESSION_OPEN = "SESSION_OPEN"
    NO_FILL = "NO_FILL"

class DataGapHandler:
    """
    Institutional-Grade Data Gap Detection and Filling System
    จัดการ missing data และ price gaps แบบมืออาชีพ
    """
    
    def __init__(self, symbol: str = "XAUUSD", expected_frequency: str = "1s"):
        self.symbol = symbol
        self.expected_frequency = expected_frequency
        self.setup_logger()
        
        # Gap detection thresholds (in seconds)
        self.gap_thresholds = {
            GapType.MINOR_GAP: 60,         # 1 minute
            GapType.MODERATE_GAP: 300,     # 5 minutes
            GapType.MAJOR_GAP: 1800,       # 30 minutes
            GapType.SESSION_GAP: 7200,     # 2 hours
            GapType.WEEKEND_GAP: 172800,   # 48 hours
            GapType.HOLIDAY_GAP: 86400     # 24 hours
        }
        
        # Filling strategies by gap type
        self.fill_strategies = {
            GapType.MINOR_GAP: FillMethod.FORWARD_FILL,
            GapType.MODERATE_GAP: FillMethod.INTERPOLATION,
            GapType.MAJOR_GAP: FillMethod.LAST_KNOWN,
            GapType.SESSION_GAP: FillMethod.SESSION_OPEN,
            GapType.WEEKEND_GAP: FillMethod.LAST_KNOWN,
            GapType.HOLIDAY_GAP: FillMethod.LAST_KNOWN,
            GapType.DATA_OUTAGE: FillMethod.INTERPOLATION
        }
        
        # Statistics tracking
        self.gap_stats = {
            'total_gaps_detected': 0,
            'gaps_filled': 0,
            'gaps_by_type': {gap_type: 0 for gap_type in GapType},
            'fill_methods_used': {method: 0 for method in FillMethod},
            'largest_gap_duration': timedelta(0),
            'data_completeness': 100.0
        }
        
        # Market hours for gap classification
        self.market_hours = {
            'weekdays': [(21, 0), (21, 0)],  # Sunday 21:00 to Friday 21:00 UTC
            'weekend_start': (5, 21, 0),     # Friday 21:00 UTC
            'weekend_end': (0, 21, 0)        # Sunday 21:00 UTC
        }
        
        # Price movement limits for validation
        self.max_price_movement = {
            'per_minute': 0.001,    # 0.1% per minute
            'per_hour': 0.005,      # 0.5% per hour
            'session_gap': 0.02     # 2% for session gaps
        }
        
    def setup_logger(self):
        """Setup logging system"""
        self.logger = logging.getLogger(f'GapHandler_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def detect_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        ตรวจหา data gaps ในข้อมูล
        """
        if len(df) < 2:
            return []
        
        gaps_detected = []
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate time differences
        df['time_diff'] = df['timestamp'].diff()
        
        # Expected frequency in seconds
        expected_seconds = self._parse_frequency_to_seconds(self.expected_frequency)
        
        for idx in range(1, len(df)):
            time_diff = df.loc[idx, 'time_diff']
            
            if pd.isna(time_diff):
                continue
                
            gap_seconds = time_diff.total_seconds()
            
            # Only consider as gap if significantly larger than expected
            if gap_seconds > expected_seconds * 2:  # Allow some tolerance
                gap_info = self._classify_gap(
                    gap_seconds, 
                    df.loc[idx-1, 'timestamp'], 
                    df.loc[idx, 'timestamp']
                )
                
                gap_info.update({
                    'start_idx': idx - 1,
                    'end_idx': idx,
                    'duration': time_diff,
                    'start_price': df.loc[idx-1, 'bid'] if 'bid' in df.columns else None,
                    'end_price': df.loc[idx, 'bid'] if 'bid' in df.columns else None
                })
                
                gaps_detected.append(gap_info)
                self.gap_stats['gaps_by_type'][gap_info['type']] += 1
        
        self.gap_stats['total_gaps_detected'] = len(gaps_detected)
        
        if gaps_detected:
            largest_gap = max(gaps_detected, key=lambda x: x['duration'])
            self.gap_stats['largest_gap_duration'] = largest_gap['duration']
            
            self.logger.info(f"Detected {len(gaps_detected)} gaps, largest: {largest_gap['duration']}")
        
        return gaps_detected
    
    def _parse_frequency_to_seconds(self, frequency: str) -> int:
        """แปลง frequency string เป็น seconds"""
        freq_map = {
            '1s': 1, '5s': 5, '10s': 10, '15s': 15, '30s': 30,
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return freq_map.get(frequency, 1)
    
    def _classify_gap(self, gap_seconds: float, start_time: datetime, end_time: datetime) -> Dict:
        """จำแนกประเภทของ gap"""
        
        # Check if it's a weekend gap
        if self._is_weekend_gap(start_time, end_time):
            return {
                'type': GapType.WEEKEND_GAP,
                'severity': 'NORMAL',
                'description': 'Weekend market closure'
            }
        
        # Check if it's a session gap
        if self._is_session_gap(start_time, end_time):
            return {
                'type': GapType.SESSION_GAP,
                'severity': 'NORMAL',
                'description': 'Session transition'
            }
        
        # Classify by duration
        if gap_seconds < self.gap_thresholds[GapType.MINOR_GAP]:
            gap_type = GapType.MINOR_GAP
            severity = 'LOW'
        elif gap_seconds < self.gap_thresholds[GapType.MODERATE_GAP]:
            gap_type = GapType.MODERATE_GAP
            severity = 'MEDIUM'
        elif gap_seconds < self.gap_thresholds[GapType.MAJOR_GAP]:
            gap_type = GapType.MAJOR_GAP
            severity = 'HIGH'
        else:
            gap_type = GapType.DATA_OUTAGE
            severity = 'CRITICAL'
        
        return {
            'type': gap_type,
            'severity': severity,
            'description': f'{gap_type.value} - {gap_seconds:.0f} seconds'
        }
    
    def _is_weekend_gap(self, start_time: datetime, end_time: datetime) -> bool:
        """ตรวจสอบว่าเป็น weekend gap หรือไม่"""
        # Friday after 21:00 UTC to Sunday before 21:00 UTC
        if start_time.weekday() == 4 and start_time.hour >= 21:  # Friday after 21:00
            if end_time.weekday() == 6 and end_time.hour >= 21:  # Sunday after 21:00
                return True
        return False
    
    def _is_session_gap(self, start_time: datetime, end_time: datetime) -> bool:
        """ตรวจสอบว่าเป็น session gap หรือไม่"""
        # Simplified session gap detection
        # Gaps between major sessions (e.g., NY close to Asian open)
        gap_hours = [21, 22, 23, 0, 1, 2, 3, 6, 7]  # Common gap hours
        return start_time.hour in gap_hours or end_time.hour in gap_hours
    
    def fill_gaps(self, df: pd.DataFrame, gaps: List[Dict] = None) -> pd.DataFrame:
        """
        เติมข้อมูลในช่วง gaps
        """
        if gaps is None:
            gaps = self.detect_gaps(df)
        
        if not gaps:
            self.logger.info("No gaps to fill")
            return df
        
        filled_df = df.copy()
        total_filled = 0
        
        # Sort gaps by start time to process in order
        gaps_sorted = sorted(gaps, key=lambda x: x['start_idx'])
        
        for gap in gaps_sorted:
            fill_method = self.fill_strategies.get(gap['type'], FillMethod.FORWARD_FILL)
            
            if fill_method == FillMethod.NO_FILL:
                self.logger.info(f"Skipping gap of type {gap['type'].value}")
                continue
            
            filled_data = self._fill_gap_with_method(filled_df, gap, fill_method)
            
            if filled_data is not None:
                # Insert filled data
                filled_df = self._insert_filled_data(filled_df, gap, filled_data)
                total_filled += len(filled_data)
                self.gap_stats['fill_methods_used'][fill_method] += 1
        
        self.gap_stats['gaps_filled'] = total_filled
        self._calculate_data_completeness(filled_df, len(df))
        
        self.logger.info(f"Filled {total_filled} data points across {len(gaps_sorted)} gaps")
        
        return filled_df
    
    def _fill_gap_with_method(self, df: pd.DataFrame, gap: Dict, method: FillMethod) -> Optional[pd.DataFrame]:
        """เติมข้อมูลด้วยวิธีที่กำหนด"""
        
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']
        start_time = df.loc[start_idx, 'timestamp']
        end_time = df.loc[end_idx, 'timestamp']
        
        # Generate time range for filling
        time_range = pd.date_range(
            start=start_time + timedelta(seconds=1),
            end=end_time - timedelta(seconds=1),
            freq=self.expected_frequency
        )
        
        if len(time_range) == 0:
            return None
        
        if method == FillMethod.FORWARD_FILL:
            return self._forward_fill(df, start_idx, time_range)
        
        elif method == FillMethod.INTERPOLATION:
            return self._interpolate_fill(df, start_idx, end_idx, time_range)
        
        elif method == FillMethod.LAST_KNOWN:
            return self._last_known_fill(df, start_idx, time_range)
        
        elif method == FillMethod.SESSION_OPEN:
            return self._session_open_fill(df, start_idx, end_idx, time_range)
        
        else:
            self.logger.warning(f"Unknown fill method: {method}")
            return None
    
    def _forward_fill(self, df: pd.DataFrame, start_idx: int, time_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Forward fill method - ใช้ราคาล่าสุด"""
        last_row = df.loc[start_idx].copy()
        
        filled_data = []
        for timestamp in time_range:
            new_row = last_row.copy()
            new_row['timestamp'] = timestamp
            filled_data.append(new_row)
        
        return pd.DataFrame(filled_data)
    
    def _interpolate_fill(self, df: pd.DataFrame, start_idx: int, end_idx: int, time_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Linear interpolation method"""
        start_row = df.loc[start_idx]
        end_row = df.loc[end_idx]
        
        filled_data = []
        total_points = len(time_range) + 1  # Include start and end
        
        for i, timestamp in enumerate(time_range):
            weight = (i + 1) / total_points  # Linear weight
            
            new_row = start_row.copy()
            new_row['timestamp'] = timestamp
            
            # Interpolate numeric columns
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    if not pd.isna(start_row[col]) and not pd.isna(end_row[col]):
                        new_row[col] = start_row[col] + weight * (end_row[col] - start_row[col])
            
            filled_data.append(new_row)
        
        return pd.DataFrame(filled_data)
    
    def _last_known_fill(self, df: pd.DataFrame, start_idx: int, time_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Last known value method - similar to forward fill but with gap awareness"""
        return self._forward_fill(df, start_idx, time_range)
    
    def _session_open_fill(self, df: pd.DataFrame, start_idx: int, end_idx: int, time_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Session open method - gradual transition to next session"""
        # For session gaps, use interpolation towards expected opening price
        return self._interpolate_fill(df, start_idx, end_idx, time_range)
    
    def _insert_filled_data(self, df: pd.DataFrame, gap: Dict, filled_data: pd.DataFrame) -> pd.DataFrame:
        """แทรกข้อมูลที่เติมแล้วเข้าไปใน DataFrame"""
        if filled_data.empty:
            return df
        
        # Split original data at gap point
        before_gap = df.iloc[:gap['start_idx'] + 1].copy()
        after_gap = df.iloc[gap['end_idx']:].copy()
        
        # Mark filled data
        filled_data['is_filled'] = True
        if 'is_filled' not in before_gap.columns:
            before_gap['is_filled'] = False
        if 'is_filled' not in after_gap.columns:
            after_gap['is_filled'] = False
        
        # Combine all data
        result = pd.concat([before_gap, filled_data, after_gap], ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        return result
    
    def _calculate_data_completeness(self, filled_df: pd.DataFrame, original_length: int):
        """คำนวณ data completeness percentage"""
        if 'is_filled' in filled_df.columns:
            filled_points = filled_df['is_filled'].sum()
            total_points = len(filled_df)
            self.gap_stats['data_completeness'] = ((total_points - filled_points) / total_points) * 100
        else:
            self.gap_stats['data_completeness'] = 100.0
    
    def validate_filled_data(self, df: pd.DataFrame) -> Dict:
        """ตรวจสอบคุณภาพของข้อมูลที่เติมแล้ว"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if 'is_filled' not in df.columns:
            return validation_results
        
        filled_mask = df['is_filled'] == True
        
        if not filled_mask.any():
            return validation_results
        
        # Check price movements in filled sections
        if 'bid' in df.columns:
            filled_data = df[filled_mask]
            price_changes = filled_data['bid'].pct_change().abs()
            
            # Check for unrealistic price movements
            large_moves = price_changes > self.max_price_movement['per_minute']
            if large_moves.any():
                validation_results['warnings'].append(
                    f"Found {large_moves.sum()} large price movements in filled data"
                )
        
        # Check for data consistency
        filled_count = filled_mask.sum()
        total_count = len(df)
        fill_percentage = (filled_count / total_count) * 100
        
        validation_results['statistics'] = {
            'filled_points': int(filled_count),
            'total_points': int(total_count),
            'fill_percentage': round(fill_percentage, 2)
        }
        
        if fill_percentage > 50:
            validation_results['warnings'].append(
                f"High percentage of filled data: {fill_percentage:.1f}%"
            )
        
        return validation_results
    
    def generate_gap_report(self) -> str:
        """สร้างรายงาน gap analysis"""
        report = f"""
========== GAP ANALYSIS REPORT ==========
Symbol: {self.symbol}
Expected Frequency: {self.expected_frequency}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

GAP STATISTICS:
- Total Gaps Detected: {self.gap_stats['total_gaps_detected']}
- Gaps Filled: {self.gap_stats['gaps_filled']}
- Data Completeness: {self.gap_stats['data_completeness']:.2f}%
- Largest Gap: {self.gap_stats['largest_gap_duration']}

GAPS BY TYPE:
"""
        
        for gap_type, count in self.gap_stats['gaps_by_type'].items():
            if count > 0:
                report += f"- {gap_type.value}: {count}\n"
        
        report += "\nFILL METHODS USED:\n"
        for method, count in self.gap_stats['fill_methods_used'].items():
            if count > 0:
                report += f"- {method.value}: {count}\n"
        
        report += "\nQUALITY ASSESSMENT:\n"
        if self.gap_stats['data_completeness'] >= 95:
            report += "- EXCELLENT: High data completeness\n"
        elif self.gap_stats['data_completeness'] >= 85:
            report += "- GOOD: Acceptable data completeness\n"
        elif self.gap_stats['data_completeness'] >= 70:
            report += "- FAIR: Moderate gaps present\n"
        else:
            report += "- POOR: Significant data gaps\n"
        
        report += "=" * 42
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample data with intentional gaps
    np.random.seed(42)
    
    # Generate base time series
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    times = []
    current_time = base_time
    
    # Add normal data
    for i in range(100):
        times.append(current_time)
        current_time += timedelta(seconds=1)
    
    # Create a gap (5 minutes)
    current_time += timedelta(minutes=5)
    
    # Add more data after gap
    for i in range(50):
        times.append(current_time)
        current_time += timedelta(seconds=1)
    
    # Create another gap (30 minutes - major gap)
    current_time += timedelta(minutes=30)
    
    # Add final data
    for i in range(30):
        times.append(current_time)
        current_time += timedelta(seconds=1)
    
    # Create sample price data
    sample_data = pd.DataFrame({
        'timestamp': times,
        'bid': 2000 + np.random.normal(0, 0.5, len(times)),
        'ask': 2000.3 + np.random.normal(0, 0.5, len(times)),
        'volume': np.random.exponential(1, len(times))
    })
    
    # Initialize gap handler
    gap_handler = DataGapHandler("XAUUSD", "1s")
    
    print("Testing Data Gap Handler...")
    print(f"Original data points: {len(sample_data)}")
    
    # Detect gaps
    gaps = gap_handler.detect_gaps(sample_data)
    print(f"Gaps detected: {len(gaps)}")
    
    for gap in gaps:
        print(f"- {gap['type'].value}: {gap['duration']} ({gap['severity']})")
    
    # Fill gaps
    filled_data = gap_handler.fill_gaps(sample_data, gaps)
    print(f"Data points after filling: {len(filled_data)}")
    
    # Validate filled data
    validation = gap_handler.validate_filled_data(filled_data)
    print(f"Validation: {validation['statistics']}")
    
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
    
    # Generate report
    print(gap_handler.generate_gap_report())