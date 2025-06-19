import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import statistics

class SpreadMonitor:
    """
    Real-time Spread Monitoring System
    ติดตาม spread patterns, liquidity conditions และ market stress
    """
    
    def __init__(self, symbol: str = "XAUUSD", window_size: int = 100):
        self.symbol = symbol
        self.window_size = window_size
        self.setup_logger()
        
        # Spread data storage (rolling window)
        self.spread_data = deque(maxlen=window_size)
        self.timestamp_data = deque(maxlen=window_size)
        
        # Alert thresholds (in pips for XAUUSD)
        self.normal_spread_max = 5.0  # Normal market conditions
        self.high_spread_threshold = 10.0  # High volatility warning
        self.extreme_spread_threshold = 20.0  # Market stress alert
        
        # Statistics tracking
        self.spread_stats = {
            'current_spread': 0.0,
            'avg_spread_5min': 0.0,
            'avg_spread_1hr': 0.0,
            'min_spread_today': float('inf'),
            'max_spread_today': 0.0,
            'spread_volatility': 0.0,
            'liquidity_score': 100.0,
            'market_stress_level': 'NORMAL'
        }
        
        # Alert system
        self.alerts = []
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes between same alerts
        
        # Session-based spread tracking
        self.session_spreads = {
            'ASIAN': deque(maxlen=500),
            'LONDON': deque(maxlen=500),
            'NY': deque(maxlen=500),
            'OVERLAP_LONDON_NY': deque(maxlen=500)
        }
    
    def setup_logger(self):
        """Setup logging system"""
        self.logger = logging.getLogger(f'SpreadMonitor_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def update_spread(self, timestamp: datetime, bid: float, ask: float) -> Dict:
        """
        Update spread data และคำนวณ metrics แบบ real-time
        """
        spread = ask - bid
        spread_pips = spread * 10  # Convert to pips for XAUUSD
        
        # Store data
        self.spread_data.append(spread_pips)
        self.timestamp_data.append(timestamp)
        
        # Update current statistics
        self._update_statistics(spread_pips, timestamp)
        
        # Check for alerts
        self._check_alerts(spread_pips, timestamp)
        
        # Update session-based tracking
        self._update_session_spreads(timestamp, spread_pips)
        
        return self.get_current_status()
    
    def _update_statistics(self, spread_pips: float, timestamp: datetime):
        """อัพเดต statistics แบบ real-time"""
        self.spread_stats['current_spread'] = spread_pips
        
        if len(self.spread_data) > 0:
            # 5-minute average (assuming 1-second data)
            recent_300 = list(self.spread_data)[-300:]
            self.spread_stats['avg_spread_5min'] = statistics.mean(recent_300)
            
            # 1-hour average
            recent_3600 = list(self.spread_data)[-3600:]
            if len(recent_3600) > 10:
                self.spread_stats['avg_spread_1hr'] = statistics.mean(recent_3600)
            
            # Daily min/max
            self.spread_stats['min_spread_today'] = min(
                self.spread_stats['min_spread_today'], 
                spread_pips
            )
            self.spread_stats['max_spread_today'] = max(
                self.spread_stats['max_spread_today'], 
                spread_pips
            )
            
            # Spread volatility (rolling standard deviation)
            if len(self.spread_data) > 20:
                self.spread_stats['spread_volatility'] = statistics.stdev(
                    list(self.spread_data)[-60:]  # Last 60 seconds
                )
            
            # Liquidity score (inverse relationship with spread)
            self.spread_stats['liquidity_score'] = max(
                0, 100 - (spread_pips - 2) * 10
            )
            
            # Market stress level
            self.spread_stats['market_stress_level'] = self._calculate_stress_level(spread_pips)
    
    def _calculate_stress_level(self, spread_pips: float) -> str:
        """คำนวณระดับความเครียดของตลาด"""
        if spread_pips <= self.normal_spread_max:
            return 'NORMAL'
        elif spread_pips <= self.high_spread_threshold:
            return 'ELEVATED'
        elif spread_pips <= self.extreme_spread_threshold:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    def _check_alerts(self, spread_pips: float, timestamp: datetime):
        """ตรวจสอบและส่ง alerts"""
        current_time = timestamp.timestamp()
        
        # High spread alert
        if spread_pips > self.high_spread_threshold:
            alert_key = 'HIGH_SPREAD'
            if self._should_send_alert(alert_key, current_time):
                self._send_alert(
                    'HIGH_SPREAD',
                    f'Spread elevated to {spread_pips:.1f} pips (threshold: {self.high_spread_threshold})',
                    timestamp,
                    'WARNING'
                )
        
        # Extreme spread alert
        if spread_pips > self.extreme_spread_threshold:
            alert_key = 'EXTREME_SPREAD'
            if self._should_send_alert(alert_key, current_time):
                self._send_alert(
                    'EXTREME_SPREAD',
                    f'CRITICAL: Spread reached {spread_pips:.1f} pips! Market stress detected.',
                    timestamp,
                    'CRITICAL'
                )
        
        # Spread volatility alert
        if self.spread_stats['spread_volatility'] > 2.0:
            alert_key = 'HIGH_VOLATILITY'
            if self._should_send_alert(alert_key, current_time):
                self._send_alert(
                    'HIGH_VOLATILITY',
                    f'High spread volatility: {self.spread_stats["spread_volatility"]:.2f}',
                    timestamp,
                    'WARNING'
                )
    
    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """ตรวจสอบว่าควรส่ง alert หรือไม่ (cooldown system)"""
        if alert_key not in self.last_alert_time:
            return True
        
        time_since_last = current_time - self.last_alert_time[alert_key]
        return time_since_last > self.alert_cooldown
    
    def _send_alert(self, alert_type: str, message: str, timestamp: datetime, severity: str):
        """ส่ง alert และบันทึกลง log"""
        alert = {
            'timestamp': timestamp,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'spread_pips': self.spread_stats['current_spread']
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = timestamp.timestamp()
        
        # Log based on severity
        if severity == 'CRITICAL':
            self.logger.critical(f"[{alert_type}] {message}")
        elif severity == 'WARNING':
            self.logger.warning(f"[{alert_type}] {message}")
        else:
            self.logger.info(f"[{alert_type}] {message}")
    
    def _update_session_spreads(self, timestamp: datetime, spread_pips: float):
        """อัพเดต spread ตาม trading sessions"""
        hour_utc = timestamp.hour
        
        # Determine trading session
        if 0 <= hour_utc < 7:  # Asian session
            self.session_spreads['ASIAN'].append(spread_pips)
        elif 7 <= hour_utc < 12:  # London session
            self.session_spreads['LONDON'].append(spread_pips)
        elif 12 <= hour_utc < 17:  # London-NY overlap
            self.session_spreads['OVERLAP_LONDON_NY'].append(spread_pips)
        else:  # NY session
            self.session_spreads['NY'].append(spread_pips)
    
    def get_session_analysis(self) -> Dict:
        """วิเคราะห์ spread ตาม trading sessions"""
        session_stats = {}
        
        for session, spreads in self.session_spreads.items():
            if len(spreads) > 0:
                session_stats[session] = {
                    'avg_spread': statistics.mean(spreads),
                    'min_spread': min(spreads),
                    'max_spread': max(spreads),
                    'volatility': statistics.stdev(spreads) if len(spreads) > 1 else 0,
                    'sample_count': len(spreads)
                }
            else:
                session_stats[session] = None
        
        return session_stats
    
    def get_current_status(self) -> Dict:
        """ส่งคืนสถานะปัจจุบันของ spread"""
        return {
            'timestamp': datetime.now(),
            'spread_stats': self.spread_stats.copy(),
            'recent_alerts': self.alerts[-5:],  # Last 5 alerts
            'data_points': len(self.spread_data)
        }
    
    def generate_spread_report(self) -> str:
        """สร้างรายงาน spread analysis"""
        session_stats = self.get_session_analysis()
        
        report = f"""
========== SPREAD ANALYSIS REPORT ==========
Symbol: {self.symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Points: {len(self.spread_data)}

CURRENT METRICS:
- Current Spread: {self.spread_stats['current_spread']:.2f} pips
- 5-min Average: {self.spread_stats['avg_spread_5min']:.2f} pips
- 1-hour Average: {self.spread_stats['avg_spread_1hr']:.2f} pips
- Today's Range: {self.spread_stats['min_spread_today']:.2f} - {self.spread_stats['max_spread_today']:.2f} pips
- Spread Volatility: {self.spread_stats['spread_volatility']:.2f}
- Liquidity Score: {self.spread_stats['liquidity_score']:.1f}/100
- Market Stress: {self.spread_stats['market_stress_level']}

SESSION ANALYSIS:
"""
        
        for session, stats in session_stats.items():
            if stats:
                report += f"- {session}: Avg {stats['avg_spread']:.2f} pips (Vol: {stats['volatility']:.2f})\n"
            else:
                report += f"- {session}: No data\n"
        
        report += f"""
RECENT ALERTS: {len(self.alerts)}
"""
        
        for alert in self.alerts[-3:]:
            report += f"- {alert['timestamp'].strftime('%H:%M:%S')} [{alert['severity']}] {alert['message']}\n"
        
        report += "=" * 45
        
        return report
    
    def get_trading_recommendation(self) -> Dict:
        """ให้คำแนะนำสำหรับการเทรดตาม spread conditions"""
        current_spread = self.spread_stats['current_spread']
        stress_level = self.spread_stats['market_stress_level']
        liquidity_score = self.spread_stats['liquidity_score']
        
        if stress_level == 'EXTREME':
            recommendation = {
                'action': 'AVOID_TRADING',
                'reason': 'Extreme spread conditions - high transaction costs',
                'risk_level': 'HIGH',
                'suggested_wait': '15-30 minutes'
            }
        elif stress_level == 'HIGH':
            recommendation = {
                'action': 'REDUCE_SIZE',
                'reason': 'Elevated spreads - reduce position sizes',
                'risk_level': 'MEDIUM',
                'size_multiplier': 0.5
            }
        elif current_spread <= self.normal_spread_max:
            recommendation = {
                'action': 'NORMAL_TRADING',
                'reason': 'Good liquidity conditions',
                'risk_level': 'LOW',
                'size_multiplier': 1.0
            }
        else:
            recommendation = {
                'action': 'CAUTIOUS_TRADING',
                'reason': 'Slightly elevated spreads',
                'risk_level': 'LOW_MEDIUM',
                'size_multiplier': 0.8
            }
        
        recommendation['current_spread'] = current_spread
        recommendation['liquidity_score'] = liquidity_score
        
        return recommendation

# Example usage and testing
if __name__ == "__main__":
    # Initialize spread monitor
    spread_monitor = SpreadMonitor("XAUUSD")
    
    print("Testing Spread Monitor...")
    
    # Simulate real-time spread data
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    base_price = 2000.0
    
    # Normal conditions
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        spread = np.random.normal(0.3, 0.05)  # Normal spread ~3 pips
        bid = base_price + np.random.normal(0, 0.1)
        ask = bid + spread
        
        status = spread_monitor.update_spread(timestamp, bid, ask)
    
    print("Normal conditions processed...")
    
    # Simulate market stress (high spreads)
    for i in range(50):
        timestamp = base_time + timedelta(seconds=100 + i)
        spread = np.random.normal(1.5, 0.3)  # High spread ~15 pips
        bid = base_price + np.random.normal(0, 0.2)
        ask = bid + spread
        
        status = spread_monitor.update_spread(timestamp, bid, ask)
    
    print("Market stress conditions processed...")
    
    # Generate reports
    print("\n" + spread_monitor.generate_spread_report())
    
    # Get trading recommendation
    recommendation = spread_monitor.get_trading_recommendation()
    print(f"\nTrading Recommendation: {recommendation['action']}")
    print(f"Reason: {recommendation['reason']}")
    print(f"Risk Level: {recommendation['risk_level']}")
    
    # Session analysis
    session_analysis = spread_monitor.get_session_analysis()
    print(f"\nSession Analysis Available: {list(session_analysis.keys())}")