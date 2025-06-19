import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Set
import logging
import pytz
from enum import Enum
import calendar

class MarketRegion(Enum):
    """Major market regions"""
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    
class MarketStatus(Enum):
    """Market status types"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"
    HOLIDAY = "HOLIDAY"
    PARTIAL_TRADING = "PARTIAL_TRADING"

class MarketCalendar:
    """
    Institutional-Grade Market Calendar and Hours System
    จัดการเวลาเปิด-ปิดตลาดและวันหยุดทั่วโลก
    """
    
    def __init__(self):
        self.setup_logger()
        
        # Timezone definitions
        self.timezones = {
            MarketRegion.SYDNEY: pytz.timezone('Australia/Sydney'),
            MarketRegion.TOKYO: pytz.timezone('Asia/Tokyo'),
            MarketRegion.LONDON: pytz.timezone('Europe/London'),
            MarketRegion.NEW_YORK: pytz.timezone('America/New_York')
        }
        
        # Market hours (local time) - 24-hour format
        self.market_hours = {
            MarketRegion.SYDNEY: {'open': (17, 0), 'close': (1, 0)},  # 5:00 PM - 1:00 AM next day
            MarketRegion.TOKYO: {'open': (9, 0), 'close': (15, 0)},   # 9:00 AM - 3:00 PM
            MarketRegion.LONDON: {'open': (8, 0), 'close': (16, 30)}, # 8:00 AM - 4:30 PM
            MarketRegion.NEW_YORK: {'open': (8, 0), 'close': (17, 0)} # 8:00 AM - 5:00 PM
        }
        
        # Forex market is 24/5, but individual centers have different hours
        # XAUUSD typically follows NY precious metals market hours
        self.forex_active_regions = {
            'ASIAN_SESSION': [MarketRegion.SYDNEY, MarketRegion.TOKYO],
            'LONDON_SESSION': [MarketRegion.LONDON],
            'NY_SESSION': [MarketRegion.NEW_YORK],
            'OVERLAP_LONDON_NY': [MarketRegion.LONDON, MarketRegion.NEW_YORK]
        }
        
        # Holiday calendars (2024-2025)
        self.holidays = {
            MarketRegion.NEW_YORK: [
                date(2024, 1, 1),   # New Year's Day
                date(2024, 1, 15),  # Martin Luther King Jr. Day
                date(2024, 2, 19),  # Presidents' Day
                date(2024, 3, 29),  # Good Friday
                date(2024, 5, 27),  # Memorial Day
                date(2024, 6, 19),  # Juneteenth
                date(2024, 7, 4),   # Independence Day
                date(2024, 9, 2),   # Labor Day
                date(2024, 10, 14), # Columbus Day
                date(2024, 11, 11), # Veterans Day
                date(2024, 11, 28), # Thanksgiving
                date(2024, 12, 25), # Christmas
                date(2025, 1, 1),   # New Year's Day 2025
            ],
            MarketRegion.LONDON: [
                date(2024, 1, 1),   # New Year's Day
                date(2024, 3, 29),  # Good Friday
                date(2024, 4, 1),   # Easter Monday
                date(2024, 5, 6),   # Early May Bank Holiday
                date(2024, 5, 27),  # Spring Bank Holiday
                date(2024, 8, 26),  # Summer Bank Holiday
                date(2024, 12, 25), # Christmas Day
                date(2024, 12, 26), # Boxing Day
                date(2025, 1, 1),   # New Year's Day 2025
            ],
            MarketRegion.TOKYO: [
                date(2024, 1, 1),   # New Year's Day
                date(2024, 1, 2),   # Bank Holiday
                date(2024, 1, 8),   # Coming of Age Day
                date(2024, 2, 11),  # National Foundation Day
                date(2024, 2, 12),  # National Foundation Day (observed)
                date(2024, 2, 23),  # Emperor's Birthday
                date(2024, 3, 20),  # Vernal Equinox Day
                date(2024, 4, 29),  # Showa Day
                date(2024, 5, 3),   # Constitution Memorial Day
                date(2024, 5, 4),   # Greenery Day
                date(2024, 5, 5),   # Children's Day
                date(2024, 5, 6),   # Children's Day (observed)
                date(2024, 7, 15),  # Marine Day
                date(2024, 8, 11),  # Mountain Day
                date(2024, 8, 12),  # Mountain Day (observed)
                date(2024, 9, 16),  # Respect for the Aged Day
                date(2024, 9, 22),  # Autumnal Equinox Day
                date(2024, 9, 23),  # Autumnal Equinox Day (observed)
                date(2024, 10, 14), # Health and Sports Day
                date(2024, 11, 3),  # Culture Day
                date(2024, 11, 4),  # Culture Day (observed)
                date(2024, 11, 23), # Labor Thanksgiving Day
                date(2024, 12, 31), # New Year's Eve
                date(2025, 1, 1),   # New Year's Day 2025
            ],
            MarketRegion.SYDNEY: [
                date(2024, 1, 1),   # New Year's Day
                date(2024, 1, 26),  # Australia Day
                date(2024, 3, 29),  # Good Friday
                date(2024, 4, 1),   # Easter Monday
                date(2024, 4, 25),  # Anzac Day
                date(2024, 6, 10),  # Queen's Birthday
                date(2024, 10, 7),  # Labour Day
                date(2024, 12, 25), # Christmas Day
                date(2024, 12, 26), # Boxing Day
                date(2025, 1, 1),   # New Year's Day 2025
            ]
        }
        
        # Market impact levels for different events
        self.market_impact_events = {
            'HIGH_IMPACT': [
                'NFP',           # Non-Farm Payrolls
                'FOMC',          # Federal Open Market Committee
                'CPI',           # Consumer Price Index
                'GDP',           # Gross Domestic Product
                'UNEMPLOYMENT',  # Unemployment Rate
            ],
            'MEDIUM_IMPACT': [
                'PMI',           # Purchasing Managers' Index
                'RETAIL_SALES',  # Retail Sales
                'INDUSTRIAL_PRODUCTION',
                'HOUSING_DATA',
            ],
            'LOW_IMPACT': [
                'BUILDING_PERMITS',
                'CONSUMER_CONFIDENCE',
                'BUSINESS_INVENTORIES',
            ]
        }
        
        # Current market status cache
        self.status_cache = {}
        self.cache_expiry = None
        
    def setup_logger(self):
        """Setup logging system"""
        self.logger = logging.getLogger('MarketCalendar')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def is_market_open(self, region: MarketRegion, timestamp: datetime = None) -> bool:
        """ตรวจสอบว่าตลาดเปิดหรือไม่"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        # Convert to market local time
        local_tz = self.timezones[region]
        local_time = timestamp.astimezone(local_tz)
        
        # Check if it's a holiday
        if self.is_holiday(region, local_time.date()):
            return False
        
        # Check if it's weekend
        if local_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours
        market_hours = self.market_hours[region]
        open_hour, open_minute = market_hours['open']
        close_hour, close_minute = market_hours['close']
        
        current_time = local_time.time()
        open_time = local_time.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0).time()
        
        # Handle markets that close next day
        if close_hour < open_hour:  # Crosses midnight
            close_time = (local_time + timedelta(days=1)).replace(
                hour=close_hour, minute=close_minute, second=0, microsecond=0
            ).time()
            return current_time >= open_time or current_time <= close_time
        else:
            close_time = local_time.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0).time()
            return open_time <= current_time <= close_time
    
    def is_holiday(self, region: MarketRegion, check_date: date) -> bool:
        """ตรวจสอบว่าเป็นวันหยุดหรือไม่"""
        return check_date in self.holidays.get(region, [])
    
    def get_market_status(self, region: MarketRegion, timestamp: datetime = None) -> MarketStatus:
        """ดึงสถานะของตลาด"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        local_tz = self.timezones[region]
        local_time = timestamp.astimezone(local_tz)
        
        # Check holiday first
        if self.is_holiday(region, local_time.date()):
            return MarketStatus.HOLIDAY
        
        # Check weekend
        if local_time.weekday() >= 5:
            return MarketStatus.CLOSED
        
        # Check if market is open
        if self.is_market_open(region, timestamp):
            return MarketStatus.OPEN
        else:
            return MarketStatus.CLOSED
    
    def get_next_market_open(self, region: MarketRegion, timestamp: datetime = None) -> datetime:
        """คำนวณเวลาเปิดตลาดครั้งถัดไป"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        local_tz = self.timezones[region]
        local_time = timestamp.astimezone(local_tz)
        
        # Start from tomorrow if market is closed today
        check_date = local_time.date()
        if not self.is_market_open(region, timestamp):
            check_date = local_time.date() + timedelta(days=1)
        
        # Find next business day that's not a holiday
        for i in range(10):  # Check up to 10 days ahead
            candidate_date = check_date + timedelta(days=i)
            
            # Skip weekends
            if candidate_date.weekday() >= 5:
                continue
            
            # Skip holidays
            if self.is_holiday(region, candidate_date):
                continue
            
            # Found next open day
            market_hours = self.market_hours[region]
            open_hour, open_minute = market_hours['open']
            
            next_open = local_tz.localize(
                datetime.combine(candidate_date, datetime.min.time()).replace(
                    hour=open_hour, minute=open_minute
                )
            )
            
            return next_open.astimezone(pytz.UTC)
        
        # If no open day found in 10 days, return None
        return None
    
    def get_next_market_close(self, region: MarketRegion, timestamp: datetime = None) -> datetime:
        """คำนวณเวลาปิดตลาดครั้งถัดไป"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        if not self.is_market_open(region, timestamp):
            return None  # Market is already closed
        
        local_tz = self.timezones[region]
        local_time = timestamp.astimezone(local_tz)
        
        market_hours = self.market_hours[region]
        close_hour, close_minute = market_hours['close']
        
        # Calculate close time
        if close_hour < market_hours['open'][0]:  # Closes next day
            close_time = local_tz.localize(
                datetime.combine(local_time.date() + timedelta(days=1), datetime.min.time()).replace(
                    hour=close_hour, minute=close_minute
                )
            )
        else:
            close_time = local_tz.localize(
                datetime.combine(local_time.date(), datetime.min.time()).replace(
                    hour=close_hour, minute=close_minute
                )
            )
        
        return close_time.astimezone(pytz.UTC)
    
    def get_forex_market_status(self, timestamp: datetime = None) -> Dict:
        """ดึงสถานะของตลาด Forex โดยรวม"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        status = {}
        
        # Check each major center
        for region in MarketRegion:
            status[region.value] = {
                'is_open': self.is_market_open(region, timestamp),
                'status': self.get_market_status(region, timestamp).value,
                'next_open': self.get_next_market_open(region, timestamp),
                'next_close': self.get_next_market_close(region, timestamp)
            }
        
        # Determine overall forex market activity
        open_markets = [region for region in MarketRegion if status[region.value]['is_open']]
        
        status['OVERALL'] = {
            'active_sessions': len(open_markets),
            'open_markets': [market.value for market in open_markets],
            'market_activity': self._classify_market_activity(open_markets),
            'next_major_event': self._find_next_major_event(timestamp)
        }
        
        return status
    
    def _classify_market_activity(self, open_markets: List[MarketRegion]) -> str:
        """จำแนกระดับกิจกรรมของตลาด"""
        if not open_markets:
            return 'CLOSED'
        elif len(open_markets) == 1:
            return 'LOW_ACTIVITY'
        elif len(open_markets) == 2:
            # Check for major overlaps
            if (MarketRegion.LONDON in open_markets and MarketRegion.NEW_YORK in open_markets):
                return 'PEAK_ACTIVITY'  # London-NY overlap
            else:
                return 'MEDIUM_ACTIVITY'
        else:
            return 'HIGH_ACTIVITY'
    
    def _find_next_major_event(self, timestamp: datetime) -> Optional[Dict]:
        """หาเหตุการณ์สำคัญครั้งถัดไป"""
        # Simplified - in real implementation, this would check economic calendar
        next_events = []
        
        # Check for market opens/closes in next 24 hours
        for region in MarketRegion:
            next_open = self.get_next_market_open(region, timestamp)
            next_close = self.get_next_market_close(region, timestamp)
            
            if next_open and (next_open - timestamp).total_seconds() < 86400:  # Within 24 hours
                next_events.append({
                    'time': next_open,
                    'event': f'{region.value} Market Open',
                    'impact': 'MEDIUM'
                })
            
            if next_close and (next_close - timestamp).total_seconds() < 86400:
                next_events.append({
                    'time': next_close,
                    'event': f'{region.value} Market Close',
                    'impact': 'MEDIUM'
                })
        
        if next_events:
            # Return the nearest event
            next_events.sort(key=lambda x: x['time'])
            return next_events[0]
        
        return None
    
    def get_trading_week_info(self, timestamp: datetime = None) -> Dict:
        """ดึงข้อมูลสัปดาห์การเทรด"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        # Forex week typically starts Sunday 21:00 UTC (Sydney open)
        # and ends Friday 21:00 UTC (NY close)
        
        # Find the start of current trading week
        days_since_sunday = timestamp.weekday() + 1 if timestamp.weekday() != 6 else 0
        if timestamp.weekday() == 6 and timestamp.hour >= 21:  # Sunday after 21:00
            days_since_sunday = 0
        
        week_start = timestamp - timedelta(days=days_since_sunday)
        week_start = week_start.replace(hour=21, minute=0, second=0, microsecond=0)
        
        # Find end of trading week (Friday 21:00 UTC)
        days_until_friday = 4 - timestamp.weekday() if timestamp.weekday() < 5 else 7 - timestamp.weekday() + 4
        week_end = timestamp + timedelta(days=days_until_friday)
        week_end = week_end.replace(hour=21, minute=0, second=0, microsecond=0)
        
        return {
            'week_start': week_start,
            'week_end': week_end,
            'days_into_week': (timestamp - week_start).days,
            'hours_remaining': (week_end - timestamp).total_seconds() / 3600,
            'is_week_end_approaching': (week_end - timestamp).total_seconds() < 7200,  # Last 2 hours
            'week_progress': min(1.0, (timestamp - week_start).total_seconds() / (week_end - week_start).total_seconds())
        }
    
    def get_holiday_calendar(self, region: MarketRegion, year: int = None) -> List[date]:
        """ดึงปฏิทินวันหยุดของภูมิภาค"""
        if year is None:
            year = datetime.now().year
        
        region_holidays = self.holidays.get(region, [])
        year_holidays = [h for h in region_holidays if h.year == year]
        
        return sorted(year_holidays)
    
    def generate_market_schedule(self, start_date: date, end_date: date, region: MarketRegion = None) -> pd.DataFrame:
        """สร้างตารางเวลาตลาด"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if region:
            regions = [region]
        else:
            regions = list(MarketRegion)
        
        schedule_data = []
        
        for single_date in date_range:
            for region in regions:
                is_holiday = self.is_holiday(region, single_date.date())
                is_weekend = single_date.weekday() >= 5
                
                schedule_data.append({
                    'date': single_date.date(),
                    'region': region.value,
                    'is_trading_day': not (is_holiday or is_weekend),
                    'is_holiday': is_holiday,
                    'is_weekend': is_weekend,
                    'day_of_week': single_date.strftime('%A')
                })
        
        return pd.DataFrame(schedule_data)
    
    def generate_calendar_report(self, timestamp: datetime = None) -> str:
        """สร้างรายงานปฏิทินตลาด"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        forex_status = self.get_forex_market_status(timestamp)
        week_info = self.get_trading_week_info(timestamp)
        
        report = f"""
========== MARKET CALENDAR REPORT ==========
Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Day of Week: {timestamp.strftime('%A')}

FOREX MARKET STATUS:
- Active Sessions: {forex_status['OVERALL']['active_sessions']}
- Open Markets: {', '.join(forex_status['OVERALL']['open_markets'])}
- Market Activity: {forex_status['OVERALL']['market_activity']}

REGIONAL MARKET STATUS:
"""
        
        for region_name, status in forex_status.items():
            if region_name != 'OVERALL':
                report += f"- {region_name}: {status['status']}"
                if status['next_open']:
                    time_to_open = status['next_open'] - timestamp
                    report += f" (Next open in {time_to_open})"
                report += "\n"
        
        report += f"""
TRADING WEEK INFO:
- Week Progress: {week_info['week_progress']:.1%}
- Days into Week: {week_info['days_into_week']}
- Hours Remaining: {week_info['hours_remaining']:.1f}
- Week End Approaching: {'Yes' if week_info['is_week_end_approaching'] else 'No'}

NEXT MAJOR EVENT:
"""
        
        next_event = forex_status['OVERALL']['next_major_event']
        if next_event:
            time_to_event = next_event['time'] - timestamp
            report += f"- {next_event['event']} in {time_to_event}"
        else:
            report += "- No major events in next 24 hours"
        
        # Add upcoming holidays
        report += "\n\nUPCOMING HOLIDAYS (Next 30 days):\n"
        for region in MarketRegion:
            upcoming_holidays = [
                h for h in self.holidays.get(region, [])
                if timestamp.date() <= h <= timestamp.date() + timedelta(days=30)
            ]
            if upcoming_holidays:
                report += f"- {region.value}: {', '.join(h.strftime('%Y-%m-%d') for h in upcoming_holidays)}\n"
        
        report += "=" * 44
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize market calendar
    calendar = MarketCalendar()
    
    print("Testing Market Calendar...")
    
    # Test current market status
    current_time = datetime.now(pytz.UTC)
    print(f"Current time: {current_time}")
    
    # Check each market
    for region in MarketRegion:
        is_open = calendar.is_market_open(region, current_time)
        status = calendar.get_market_status(region, current_time)
        print(f"{region.value}: {'OPEN' if is_open else 'CLOSED'} ({status.value})")
    
    # Test forex market status
    forex_status = calendar.get_forex_market_status(current_time)
    print(f"\nForex Market Activity: {forex_status['OVERALL']['market_activity']}")
    print(f"Active Sessions: {forex_status['OVERALL']['active_sessions']}")
    
    # Test next market events
    for region in MarketRegion:
        next_open = calendar.get_next_market_open(region, current_time)
        if next_open:
            time_to_open = next_open - current_time
            print(f"Next {region.value} open: {next_open} (in {time_to_open})")
    
    # Test trading week info
    week_info = calendar.get_trading_week_info(current_time)
    print(f"\nTrading Week Progress: {week_info['week_progress']:.1%}")
    print(f"Hours remaining this week: {week_info['hours_remaining']:.1f}")
    
    # Test holiday checking
    test_dates = [
        date(2024, 12, 25),  # Christmas
        date(2024, 1, 1),    # New Year
        date(2024, 7, 4),    # Independence Day
    ]
    
    print(f"\nHoliday Testing:")
    for test_date in test_dates:
        for region in MarketRegion:
            is_holiday = calendar.is_holiday(region, test_date)
            if is_holiday:
                print(f"- {test_date} is a holiday in {region.value}")
    
    # Generate market schedule for next 7 days
    schedule = calendar.generate_market_schedule(
        start_date=current_time.date(),
        end_date=current_time.date() + timedelta(days=7),
        region=MarketRegion.NEW_YORK
    )
    
    trading_days = schedule[schedule['is_trading_day'] == True]
    print(f"\nTrading days for NY in next 7 days: {len(trading_days)}")
    
    # Generate comprehensive report
    print(calendar.generate_calendar_report(current_time))