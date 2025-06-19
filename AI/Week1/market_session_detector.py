import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
import pytz

class TradingSession(Enum):
    """Trading Session Classifications"""
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NY = "NY"
    OVERLAP_ASIAN_LONDON = "OVERLAP_ASIAN_LONDON"
    OVERLAP_LONDON_NY = "OVERLAP_LONDON_NY"
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"

class MarketSessionDetector:
    """
    Institutional-Grade Market Session Detection System
    ตรวจจับ trading sessions และวิเคราะห์ลักษณะเฉพาะแต่ละ session
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.setup_logger()
        
        # Timezone definitions
        self.timezones = {
            'UTC': pytz.UTC,
            'TOKYO': pytz.timezone('Asia/Tokyo'),
            'LONDON': pytz.timezone('Europe/London'),
            'NY': pytz.timezone('America/New_York'),
            'SYDNEY': pytz.timezone('Australia/Sydney')
        }
        
        # Session time definitions (UTC)
        self.session_times = {
            TradingSession.ASIAN: {
                'start': 23,  # 23:00 UTC (Sydney open)
                'end': 8,     # 08:00 UTC
                'major_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
                'characteristics': {
                    'volatility': 'LOW_TO_MEDIUM',
                    'liquidity': 'MEDIUM',
                    'trend_strength': 'WEAK_TO_MEDIUM'
                }
            },
            TradingSession.LONDON: {
                'start': 7,   # 07:00 UTC (London open)
                'end': 16,    # 16:00 UTC
                'major_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
                'characteristics': {
                    'volatility': 'HIGH',
                    'liquidity': 'HIGH',
                    'trend_strength': 'STRONG'
                }
            },
            TradingSession.NY: {
                'start': 12,  # 12:00 UTC (NY open)
                'end': 21,    # 21:00 UTC
                'major_pairs': ['EURUSD', 'GBPUSD', 'USDCAD'],
                'characteristics': {
                    'volatility': 'HIGH',
                    'liquidity': 'HIGH',
                    'trend_strength': 'STRONG'
                }
            }
        }
        
        # Overlap periods (highest liquidity)
        self.overlap_periods = {
            TradingSession.OVERLAP_ASIAN_LONDON: {
                'start': 7, 'end': 8,
                'description': 'Asian-London Overlap'
            },
            TradingSession.OVERLAP_LONDON_NY: {
                'start': 12, 'end': 16,
                'description': 'London-NY Overlap (Highest Liquidity)'
            }
        }
        
        # Session statistics tracking
        self.session_stats = {
            session: {
                'price_data': [],
                'volume_data': [],
                'spread_data': [],
                'volatility_data': [],
                'last_update': None
            }
            for session in TradingSession
        }
        
        # Current session tracking
        self.current_session = None
        self.session_start_time = None
        self.session_characteristics = {}
        
        # Major news/economic events time windows
        self.high_impact_hours = [8, 9, 12, 13, 14, 15]  # UTC hours for major news
        
    def setup_logger(self):
        """Setup logging system"""
        self.logger = logging.getLogger(f'SessionDetector_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def detect_session(self, timestamp: datetime) -> Tuple[TradingSession, Dict]:
        """
        ตรวจจับ trading session ปัจจุบัน
        """
        # Convert to UTC if needed
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        elif timestamp.tzinfo != pytz.UTC:
            timestamp = timestamp.astimezone(pytz.UTC)
        
        hour_utc = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Check for weekend
        if weekday >= 5:  # Saturday or Sunday
            if weekday == 6 and hour_utc >= 21:  # Sunday after 21:00 UTC
                pass  # Market opening soon
            elif weekday == 5 and hour_utc < 21:  # Friday before 21:00 UTC
                pass  # Market still open
            else:
                return TradingSession.WEEKEND, self._get_weekend_characteristics()
        
        # Detect main sessions and overlaps
        session, characteristics = self._classify_session_by_time(hour_utc)
        
        # Update current session if changed
        if session != self.current_session:
            self._on_session_change(session, timestamp)
        
        # Add additional characteristics
        characteristics.update(self._get_additional_characteristics(timestamp, hour_utc))
        
        return session, characteristics
    
    def _classify_session_by_time(self, hour_utc: int) -> Tuple[TradingSession, Dict]:
        """จำแนก session ตามเวลา UTC"""
        
        # Check for overlaps first (higher priority)
        for overlap_session, times in self.overlap_periods.items():
            if self._time_in_range(hour_utc, times['start'], times['end']):
                characteristics = self._get_overlap_characteristics(overlap_session)
                return overlap_session, characteristics
        
        # Check main sessions
        london_active = self._time_in_range(hour_utc, 7, 16)
        ny_active = self._time_in_range(hour_utc, 12, 21)
        asian_active = self._time_in_range(hour_utc, 23, 8, crosses_midnight=True)
        
        if london_active and ny_active:
            # This should be caught by overlap detection, but double check
            return TradingSession.OVERLAP_LONDON_NY, self._get_overlap_characteristics(TradingSession.OVERLAP_LONDON_NY)
        elif london_active:
            return TradingSession.LONDON, self.session_times[TradingSession.LONDON]['characteristics'].copy()
        elif ny_active:
            return TradingSession.NY, self.session_times[TradingSession.NY]['characteristics'].copy()
        elif asian_active:
            return TradingSession.ASIAN, self.session_times[TradingSession.ASIAN]['characteristics'].copy()
        else:
            # Low liquidity period
            return TradingSession.LOW_LIQUIDITY, {
                'volatility': 'VERY_LOW',
                'liquidity': 'LOW',
                'trend_strength': 'VERY_WEAK',
                'recommendation': 'AVOID_TRADING'
            }
    
    def _time_in_range(self, hour: int, start: int, end: int, crosses_midnight: bool = False) -> bool:
        """ตรวจสอบว่าเวลาอยู่ในช่วงที่กำหนดหรือไม่"""
        if crosses_midnight:
            return hour >= start or hour < end
        else:
            return start <= hour < end
    
    def _get_overlap_characteristics(self, overlap_session: TradingSession) -> Dict:
        """ลักษณะเฉพาะของ overlap periods"""
        base_characteristics = {
            'volatility': 'VERY_HIGH',
            'liquidity': 'VERY_HIGH',
            'trend_strength': 'VERY_STRONG',
            'pip_movement_expected': 'HIGH',
            'breakout_probability': 'HIGH'
        }
        
        if overlap_session == TradingSession.OVERLAP_LONDON_NY:
            base_characteristics.update({
                'description': 'Highest liquidity period - Best for scalping and breakouts',
                'major_currency_focus': ['EUR', 'GBP', 'USD'],
                'news_impact': 'MAXIMUM',
                'recommended_strategies': ['SCALPING', 'BREAKOUT', 'NEWS_TRADING']
            })
        elif overlap_session == TradingSession.OVERLAP_ASIAN_LONDON:
            base_characteristics.update({
                'description': 'Moderate overlap - Good transition period',
                'major_currency_focus': ['EUR', 'JPY', 'GBP'],
                'news_impact': 'MEDIUM',
                'recommended_strategies': ['RANGE_TRADING', 'EARLY_BREAKOUT']
            })
        
        return base_characteristics
    
    def _get_weekend_characteristics(self) -> Dict:
        """ลักษณะเฉพาะของ weekend"""
        return {
            'volatility': 'NONE',
            'liquidity': 'NONE',
            'trend_strength': 'NONE',
            'market_status': 'CLOSED',
            'recommendation': 'NO_TRADING',
            'next_open': self._calculate_next_market_open()
        }
    
    def _get_additional_characteristics(self, timestamp: datetime, hour_utc: int) -> Dict:
        """ลักษณะเฉพาะเพิ่มเติมตามสถานการณ์"""
        additional = {}
        
        # High impact news hours
        if hour_utc in self.high_impact_hours:
            additional['news_risk'] = 'HIGH'
            additional['volatility_modifier'] = 'ELEVATED'
        
        # First/Last hour of major sessions
        if hour_utc in [7, 8, 15, 16, 12, 13, 20, 21]:
            additional['session_transition'] = True
            additional['volatility_modifier'] = 'ELEVATED'
        
        # Month/Week end effects
        if timestamp.day >= 28 or timestamp.weekday() == 4:  # Month end or Friday
            additional['institutional_flows'] = 'HIGH'
            additional['rebalancing_risk'] = 'ELEVATED'
        
        return additional
    
    def _on_session_change(self, new_session: TradingSession, timestamp: datetime):
        """เมื่อมีการเปลี่ยน session"""
        old_session = self.current_session
        self.current_session = new_session
        self.session_start_time = timestamp
        
        self.logger.info(f"Session changed: {old_session} → {new_session} at {timestamp}")
        
        # Reset session-specific counters
        if new_session in self.session_stats:
            self.session_stats[new_session]['last_update'] = timestamp
    
    def _calculate_next_market_open(self) -> datetime:
        """คำนวณเวลาเปิดตลาดครั้งถัดไป"""
        # Simple calculation - next Monday 21:00 UTC (Sydney open)
        now = datetime.now(pytz.UTC)
        days_until_monday = 7 - now.weekday() if now.weekday() != 6 else 1
        next_open = now + timedelta(days=days_until_monday)
        next_open = next_open.replace(hour=21, minute=0, second=0, microsecond=0)
        return next_open
    
    def update_session_data(self, timestamp: datetime, price: float, volume: float, spread: float):
        """อัพเดตข้อมูลสถิติของแต่ละ session"""
        session, _ = self.detect_session(timestamp)
        
        if session in self.session_stats:
            stats = self.session_stats[session]
            stats['price_data'].append(price)
            stats['volume_data'].append(volume)
            stats['spread_data'].append(spread)
            
            # Calculate volatility (rolling 20-period)
            if len(stats['price_data']) >= 20:
                recent_prices = stats['price_data'][-20:]
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns) * np.sqrt(1440)  # Annualized
                stats['volatility_data'].append(volatility)
            
            stats['last_update'] = timestamp
    
    def get_session_statistics(self, session: TradingSession) -> Optional[Dict]:
        """ดึงสถิติของ session ที่กำหนด"""
        if session not in self.session_stats:
            return None
        
        stats = self.session_stats[session]
        
        if not stats['price_data']:
            return None
        
        return {
            'avg_price': np.mean(stats['price_data']),
            'price_range': np.max(stats['price_data']) - np.min(stats['price_data']),
            'avg_volume': np.mean(stats['volume_data']) if stats['volume_data'] else 0,
            'avg_spread': np.mean(stats['spread_data']) if stats['spread_data'] else 0,
            'avg_volatility': np.mean(stats['volatility_data']) if stats['volatility_data'] else 0,
            'data_points': len(stats['price_data']),
            'last_update': stats['last_update']
        }
    
    def get_all_session_comparison(self) -> Dict:
        """เปรียบเทียบสถิติของทุก sessions"""
        comparison = {}
        
        for session in TradingSession:
            stats = self.get_session_statistics(session)
            if stats:
                comparison[session.value] = stats
        
        return comparison
    
    def get_current_session_info(self, timestamp: datetime = None) -> Dict:
        """ดึงข้อมูลของ session ปัจจุบัน"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        session, characteristics = self.detect_session(timestamp)
        
        info = {
            'current_session': session.value,
            'session_start': self.session_start_time,
            'characteristics': characteristics,
            'session_statistics': self.get_session_statistics(session),
            'time_in_session': None,
            'time_until_next_session': self._calculate_time_until_next_session(timestamp)
        }
        
        if self.session_start_time:
            info['time_in_session'] = timestamp - self.session_start_time
        
        return info
    
    def _calculate_time_until_next_session(self, timestamp: datetime) -> timedelta:
        """คำนวณเวลาจนกว่าจะถึง session ถัดไป"""
        hour_utc = timestamp.hour
        
        # Simple calculation for major session changes
        next_session_hours = [7, 12, 16, 21, 23]  # Major session start times
        
        for next_hour in next_session_hours:
            if next_hour > hour_utc:
                next_session_time = timestamp.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                return next_session_time - timestamp
        
        # If no session found today, next session is tomorrow at 21:00 (Sunday open)
        next_session_time = (timestamp + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
        return next_session_time - timestamp
    
    def generate_session_report(self, timestamp: datetime = None) -> str:
        """สร้างรายงาน session analysis"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        current_info = self.get_current_session_info(timestamp)
        session_comparison = self.get_all_session_comparison()
        
        report = f"""
========== SESSION ANALYSIS REPORT ==========
Symbol: {self.symbol}
Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

CURRENT SESSION:
- Session: {current_info['current_session']}
- Volatility: {current_info['characteristics'].get('volatility', 'N/A')}
- Liquidity: {current_info['characteristics'].get('liquidity', 'N/A')}
- Trend Strength: {current_info['characteristics'].get('trend_strength', 'N/A')}
"""
        
        if current_info['time_in_session']:
            report += f"- Time in Session: {current_info['time_in_session']}\n"
        
        report += f"- Time to Next Session: {current_info['time_until_next_session']}\n"
        
        if current_info['session_statistics']:
            stats = current_info['session_statistics']
            report += f"""
SESSION STATISTICS:
- Average Price: {stats['avg_price']:.5f}
- Price Range: {stats['price_range']:.5f}
- Average Spread: {stats['avg_spread']:.2f}
- Data Points: {stats['data_points']:,}
"""
        
        report += "\nSESSION COMPARISON:\n"
        for session_name, stats in session_comparison.items():
            if stats['data_points'] > 0:
                report += f"- {session_name}: Range {stats['price_range']:.5f}, Spread {stats['avg_spread']:.2f}\n"
        
        report += "=" * 45
        
        return report
    
    def get_trading_recommendation(self, timestamp: datetime = None) -> Dict:
        """ให้คำแนะนำการเทรดตาม session"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        session, characteristics = self.detect_session(timestamp)
        
        # Base recommendation from characteristics
        if 'recommendation' in characteristics:
            action = characteristics['recommendation']
        elif characteristics.get('liquidity') == 'VERY_HIGH':
            action = 'AGGRESSIVE_TRADING'
        elif characteristics.get('liquidity') == 'HIGH':
            action = 'NORMAL_TRADING'
        elif characteristics.get('liquidity') == 'MEDIUM':
            action = 'CAUTIOUS_TRADING'
        else:
            action = 'AVOID_TRADING'
        
        recommendation = {
            'action': action,
            'session': session.value,
            'volatility_expected': characteristics.get('volatility', 'UNKNOWN'),
            'liquidity_level': characteristics.get('liquidity', 'UNKNOWN'),
            'recommended_strategies': characteristics.get('recommended_strategies', []),
            'risk_factors': []
        }
        
        # Add risk factors
        if characteristics.get('news_risk') == 'HIGH':
            recommendation['risk_factors'].append('High impact news expected')
        
        if characteristics.get('session_transition'):
            recommendation['risk_factors'].append('Session transition period')
        
        if characteristics.get('rebalancing_risk') == 'ELEVATED':
            recommendation['risk_factors'].append('Month/week end flows')
        
        return recommendation

# Example usage and testing
if __name__ == "__main__":
    # Initialize session detector
    detector = MarketSessionDetector("XAUUSD")
    
    print("Testing Market Session Detector...")
    
    # Test different times
    test_times = [
        datetime(2024, 1, 15, 2, 0, 0, tzinfo=pytz.UTC),   # Asian session
        datetime(2024, 1, 15, 9, 0, 0, tzinfo=pytz.UTC),   # London session
        datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC),  # London-NY overlap
        datetime(2024, 1, 15, 18, 0, 0, tzinfo=pytz.UTC),  # NY session
        datetime(2024, 1, 13, 12, 0, 0, tzinfo=pytz.UTC),  # Weekend
    ]
    
    for test_time in test_times:
        session, characteristics = detector.detect_session(test_time)
        print(f"\nTime: {test_time}")
        print(f"Session: {session.value}")
        print(f"Characteristics: {characteristics}")
        
        # Get trading recommendation
        recommendation = detector.get_trading_recommendation(test_time)
        print(f"Recommendation: {recommendation['action']}")
    
    # Simulate session data updates
    print("\nSimulating session data updates...")
    current_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
    
    for i in range(100):
        timestamp = current_time + timedelta(minutes=i)
        price = 2000 + np.random.normal(0, 1)
        volume = np.random.exponential(1)
        spread = np.random.normal(0.3, 0.05)
        
        detector.update_session_data(timestamp, price, volume, spread)
    
    # Generate comprehensive report
    print(detector.generate_session_report(current_time))
    
    # Session comparison
    comparison = detector.get_all_session_comparison()
    print(f"\nAvailable session data: {list(comparison.keys())}")