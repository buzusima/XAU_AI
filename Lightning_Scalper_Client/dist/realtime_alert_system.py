"""
⚡ REAL-TIME TRADING ALERT SYSTEM
================================
Live Signal Monitoring & Notification System
Integration with Optimized Trading System V2.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlertSignal:
    """Signal data structure for alerts"""
    timestamp: datetime
    symbol: str
    signal_direction: str
    strength_score: float
    confidence: str
    current_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    alert_type: str  # 'NEW_SIGNAL', 'SIGNAL_UPDATE', 'SIGNAL_CANCEL'
    timeframe_summary: str
    risk_factors: List[str]

class RealTimeAlertSystem:
    """Real-time alert system for trading signals"""
    
    def __init__(self, 
                 symbol: str = "XAUUSD.c",
                 monitoring_interval: int = 60,  # seconds
                 alert_cooldown: int = 300):     # 5 minutes
        """Initialize real-time alert system"""
        self.symbol = symbol
        self.monitoring_interval = monitoring_interval
        self.alert_cooldown = alert_cooldown
        
        # Alert configuration
        self.email_alerts = False
        self.console_alerts = True
        self.log_alerts = True
        self.telegram_alerts = False
        
        # Email settings (configure if needed)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'recipient_email': ''
        }
        
        # System state
        self.is_running = False
        self.last_signal = None
        self.last_alert_time = {}
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'high_confidence_signals': 0,
            'filtered_signals': 0,
            'uptime_start': None
        }
        
        # Threading
        self.alert_queue = queue.Queue()
        self.monitor_thread = None
        self.alert_thread = None
        
        self.setup_logging()
        self.initialize_optimized_system()
        
    def setup_logging(self):
        """Setup enhanced logging for alerts"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('realtime_alerts.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_optimized_system(self):
        """Initialize the optimized trading system"""
        try:
            # Import optimized system components (simplified for demonstration)
            from datetime import datetime
            
            # Optimized system parameters
            self.timeframes = {
                'H4': mt5.TIMEFRAME_H4,
                'H1': mt5.TIMEFRAME_H1,
                'M15': mt5.TIMEFRAME_M15,
                'M5': mt5.TIMEFRAME_M5,
                'M1': mt5.TIMEFRAME_M1
            }
            
            # Enhanced filtering thresholds
            self.min_confluence_score = 7.5
            self.min_timeframes_agreement = 4
            self.max_rsi_extreme = 20
            self.min_volume_confirmation = 1.3
            self.min_atr_threshold = 0.001
            self.max_atr_threshold = 0.020
            self.trend_strength_min = 0.75
            
            self.logger.info("Optimized trading system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing optimized system: {str(e)}")
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {self.symbol} not found")
                return False
                
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to select symbol {self.symbol}")
                    return False
                    
            self.logger.info(f"Successfully connected to MT5 for real-time monitoring")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def get_current_signal(self) -> Optional[Dict]:
        """Get current trading signal (simplified version)"""
        try:
            # This is a simplified version of the optimized system
            # In production, you would import and use the full OptimizedTradingSystemV2
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
                
            current_price = tick.bid
            
            # Simplified signal generation for demonstration
            # In reality, this would call the full optimized system
            
            # Get basic market data
            rates_h1 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 50)
            if rates_h1 is None:
                return None
                
            df = pd.DataFrame(rates_h1)
            
            # Calculate basic indicators
            close = df['close']
            ema_9 = close.ewm(span=9).mean().iloc[-1]
            ema_21 = close.ewm(span=21).mean().iloc[-1]
            
            # Simple RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Simple signal logic for demonstration
            signal_direction = "NONE"
            confidence = "LOW"
            strength_score = 0.0
            
            # Basic trend analysis
            if current_price > ema_9 > ema_21 and 40 < rsi < 60:
                signal_direction = "BUY"
                strength_score = 6.5
                confidence = "MEDIUM"
            elif current_price < ema_9 < ema_21 and 40 < rsi < 60:
                signal_direction = "SELL"
                strength_score = 6.5
                confidence = "MEDIUM"
            
            # Apply enhanced filtering (simplified)
            if rsi < 25 or rsi > 75:
                signal_direction = "NONE"
                confidence = "FILTERED"
                strength_score = 0.0
            
            # Risk/reward calculation
            atr_approx = current_price * 0.005
            
            if signal_direction == "BUY":
                stop_loss = current_price - (atr_approx * 1.5)
                take_profit_1 = current_price + (atr_approx * 2.5)
                take_profit_2 = current_price + (atr_approx * 4.0)
                take_profit_3 = current_price + (atr_approx * 6.0)
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr_approx * 1.5)
                take_profit_1 = current_price - (atr_approx * 2.5)
                take_profit_2 = current_price - (atr_approx * 4.0)
                take_profit_3 = current_price - (atr_approx * 6.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = current_price
            
            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_direction': signal_direction,
                'strength_score': round(strength_score, 2),
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'take_profit_3': round(take_profit_3, 2),
                'rsi': round(rsi, 1),
                'ema_9': round(ema_9, 2),
                'ema_21': round(ema_21, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current signal: {str(e)}")
            return None
    
    def should_send_alert(self, signal: Dict) -> Tuple[bool, str]:
        """Determine if alert should be sent"""
        try:
            current_time = datetime.now()
            signal_direction = signal['signal_direction']
            confidence = signal['confidence']
            
            # Alert criteria
            alert_reasons = []
            
            # 1. High confidence signals always alert
            if confidence in ['VERY_HIGH', 'HIGH']:
                alert_reasons.append("High confidence signal detected")
                
            # 2. Signal direction change
            if self.last_signal and self.last_signal['signal_direction'] != signal_direction:
                if signal_direction != 'NONE':
                    alert_reasons.append(f"Signal direction changed to {signal_direction}")
                    
            # 3. Strength score improvement
            if (self.last_signal and 
                signal['strength_score'] > self.last_signal['strength_score'] + 1.0 and
                signal['strength_score'] >= 7.0):
                alert_reasons.append("Signal strength significantly improved")
                
            # 4. First valid signal of the session
            if (signal_direction != 'NONE' and 
                signal['confidence'] != 'FILTERED' and
                not self.last_signal):
                alert_reasons.append("First valid signal of session")
            
            # Check cooldown
            alert_key = f"{signal_direction}_{confidence}"
            if alert_key in self.last_alert_time:
                time_since_last = (current_time - self.last_alert_time[alert_key]).seconds
                if time_since_last < self.alert_cooldown:
                    return False, "Alert cooldown active"
            
            # Send alert if any criteria met
            if alert_reasons:
                self.last_alert_time[alert_key] = current_time
                return True, "; ".join(alert_reasons)
                
            return False, "No alert criteria met"
            
        except Exception as e:
            self.logger.error(f"Error checking alert criteria: {str(e)}")
            return False, "Error checking criteria"
    
    def create_alert_signal(self, signal: Dict, alert_reason: str) -> AlertSignal:
        """Create structured alert signal"""
        try:
            # Determine alert type
            if not self.last_signal:
                alert_type = 'NEW_SIGNAL'
            elif self.last_signal['signal_direction'] != signal['signal_direction']:
                alert_type = 'SIGNAL_UPDATE'
            else:
                alert_type = 'SIGNAL_UPDATE'
            
            # Create timeframe summary
            timeframe_summary = f"RSI: {signal['rsi']}, EMA9: {signal['ema_9']}, EMA21: {signal['ema_21']}"
            
            # Risk factors
            risk_factors = []
            if signal['rsi'] < 30:
                risk_factors.append("RSI Oversold")
            elif signal['rsi'] > 70:
                risk_factors.append("RSI Overbought")
                
            if signal['confidence'] == 'FILTERED':
                risk_factors.append("Signal Filtered")
            
            return AlertSignal(
                timestamp=signal['timestamp'],
                symbol=signal['symbol'],
                signal_direction=signal['signal_direction'],
                strength_score=signal['strength_score'],
                confidence=signal['confidence'],
                current_price=signal['current_price'],
                stop_loss=signal['stop_loss'],
                take_profit_1=signal['take_profit_1'],
                take_profit_2=signal['take_profit_2'],
                take_profit_3=signal['take_profit_3'],
                alert_type=alert_type,
                timeframe_summary=timeframe_summary,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error creating alert signal: {str(e)}")
            return None
    
    def send_console_alert(self, alert: AlertSignal):
        """Send console alert"""
        try:
            direction_emoji = "🟢" if alert.signal_direction == "BUY" else "🔴" if alert.signal_direction == "SELL" else "⚪"
            
            print("\n" + "="*70)
            print("⚡ REAL-TIME TRADING ALERT")
            print("="*70)
            print(f"📅 Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💰 Symbol: {alert.symbol}")
            print(f"💵 Current Price: ${alert.current_price}")
            print(f"📊 Strength Score: {alert.strength_score}/10")
            print(f"🎯 Confidence: {alert.confidence}")
            print(f"{direction_emoji} SIGNAL: {alert.signal_direction}")
            print(f"🔔 Alert Type: {alert.alert_type}")
            
            if alert.signal_direction != "NONE":
                print(f"\n🎯 TRADING LEVELS:")
                print(f"   Stop Loss: ${alert.stop_loss}")
                print(f"   Take Profit 1: ${alert.take_profit_1}")
                print(f"   Take Profit 2: ${alert.take_profit_2}")
                print(f"   Take Profit 3: ${alert.take_profit_3}")
            
            print(f"\n📊 Market Data: {alert.timeframe_summary}")
            
            if alert.risk_factors:
                print(f"⚠️ Risk Factors: {', '.join(alert.risk_factors)}")
            
            print("="*70)
            
        except Exception as e:
            self.logger.error(f"Error sending console alert: {str(e)}")
    
    def send_email_alert(self, alert: AlertSignal):
        """Send email alert (configure email settings first)"""
        try:
            if not self.email_alerts or not self.email_config['sender_email']:
                return
                
            # Create email content
            subject = f"Trading Alert: {alert.symbol} - {alert.signal_direction} Signal"
            
            body = f"""
            REAL-TIME TRADING ALERT
            
            Time: {alert.timestamp}
            Symbol: {alert.symbol}
            Signal: {alert.signal_direction}
            Confidence: {alert.confidence}
            Strength Score: {alert.strength_score}/10
            Current Price: ${alert.current_price}
            
            Trading Levels:
            Stop Loss: ${alert.stop_loss}
            Take Profit 1: ${alert.take_profit_1}
            Take Profit 2: ${alert.take_profit_2}
            Take Profit 3: ${alert.take_profit_3}
            
            Market Data: {alert.timeframe_summary}
            Risk Factors: {', '.join(alert.risk_factors) if alert.risk_factors else 'None'}
            
            Alert Type: {alert.alert_type}
            """
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
    
    def log_alert(self, alert: AlertSignal):
        """Log alert to file"""
        try:
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            
            # Log to file
            with open('trading_alerts.json', 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
                
            # Add to history
            self.signal_history.append(alert)
            
            # Keep only last 100 alerts
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error logging alert: {str(e)}")
    
    def process_alerts(self):
        """Process alerts from queue"""
        while self.is_running:
            try:
                # Get alert from queue (with timeout)
                alert = self.alert_queue.get(timeout=1)
                
                # Send alerts through all configured channels
                if self.console_alerts:
                    self.send_console_alert(alert)
                    
                if self.email_alerts:
                    self.send_email_alert(alert)
                    
                if self.log_alerts:
                    self.log_alert(alert)
                
                # Update performance stats
                self.performance_stats['total_signals'] += 1
                if alert.confidence in ['HIGH', 'VERY_HIGH']:
                    self.performance_stats['high_confidence_signals'] += 1
                if alert.confidence == 'FILTERED':
                    self.performance_stats['filtered_signals'] += 1
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {str(e)}")
    
    def monitor_signals(self):
        """Main monitoring loop"""
        self.logger.info(f"Starting signal monitoring for {self.symbol}")
        
        while self.is_running:
            try:
                # Get current signal
                current_signal = self.get_current_signal()
                
                if current_signal:
                    # Check if alert should be sent
                    should_alert, reason = self.should_send_alert(current_signal)
                    
                    if should_alert:
                        # Create and queue alert
                        alert = self.create_alert_signal(current_signal, reason)
                        if alert:
                            self.alert_queue.put(alert)
                            self.logger.info(f"Alert queued: {alert.signal_direction} - {reason}")
                    
                    # Update last signal
                    self.last_signal = current_signal
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        try:
            if self.is_running:
                self.logger.warning("Monitoring is already running")
                return
                
            # Connect to MT5
            if not self.connect_mt5():
                self.logger.error("Failed to connect to MT5")
                return
                
            self.is_running = True
            self.performance_stats['uptime_start'] = datetime.now()
            
            # Start threads
            self.monitor_thread = threading.Thread(target=self.monitor_signals, daemon=True)
            self.alert_thread = threading.Thread(target=self.process_alerts, daemon=True)
            
            self.monitor_thread.start()
            self.alert_thread.start()
            
            self.logger.info(f"Real-time monitoring started for {self.symbol}")
            self.logger.info(f"Monitoring interval: {self.monitoring_interval} seconds")
            self.logger.info(f"Alert cooldown: {self.alert_cooldown} seconds")
            
            # Print initial status
            print("\n🚀 REAL-TIME ALERT SYSTEM STARTED")
            print("="*50)
            print(f"📊 Symbol: {self.symbol}")
            print(f"⏱️ Monitoring Interval: {self.monitoring_interval} seconds")
            print(f"🔔 Alert Channels: Console={'✅' if self.console_alerts else '❌'}, "
                  f"Email={'✅' if self.email_alerts else '❌'}, "
                  f"Log={'✅' if self.log_alerts else '❌'}")
            print("📡 Status: MONITORING ACTIVE")
            print("⏹️ Press Ctrl+C to stop monitoring")
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        try:
            self.logger.info("Stopping real-time monitoring...")
            self.is_running = False
            
            # Wait for threads to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
            if self.alert_thread and self.alert_thread.is_alive():
                self.alert_thread.join(timeout=5)
            
            # Print final statistics
            uptime = datetime.now() - self.performance_stats['uptime_start']
            
            print("\n📊 MONITORING SESSION SUMMARY")
            print("="*50)
            print(f"⏱️ Session Duration: {uptime}")
            print(f"📊 Total Signals: {self.performance_stats['total_signals']}")
            print(f"🎯 High Confidence Signals: {self.performance_stats['high_confidence_signals']}")
            print(f"🛡️ Filtered Signals: {self.performance_stats['filtered_signals']}")
            print("="*50)
            
            mt5.shutdown()
            self.logger.info("Real-time monitoring stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'monitoring_interval': self.monitoring_interval,
            'performance_stats': self.performance_stats,
            'last_signal': self.last_signal,
            'alert_queue_size': self.alert_queue.qsize()
        }

def main():
    """Main execution for real-time monitoring"""
    print("⚡ Starting Real-Time Trading Alert System...")
    
    # Initialize alert system
    alert_system = RealTimeAlertSystem(
        symbol="XAUUSD.c",
        monitoring_interval=60,  # Check every minute
        alert_cooldown=300       # 5 minute cooldown
    )
    
    # Configure alerts (customize as needed)
    alert_system.console_alerts = True
    alert_system.log_alerts = True
    alert_system.email_alerts = False  # Set to True and configure email if needed
    
    try:
        # Start monitoring
        alert_system.start_monitoring()
        
        # Keep running until interrupted
        while alert_system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitoring system...")
        alert_system.stop_monitoring()
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        alert_system.stop_monitoring()

if __name__ == "__main__":
    main()