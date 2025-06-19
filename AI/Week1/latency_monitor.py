import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import logging
import threading
from collections import deque
import statistics
import psutil
import tracemalloc

class LatencyMonitor:
    """
    Institutional-Grade Latency and Performance Monitoring System
    ติดตาม latency, throughput และ system performance แบบ real-time
    """
    
    def __init__(self, symbol: str = "XAUUSD", max_samples: int = 10000):
        self.symbol = symbol
        self.max_samples = max_samples
        self.setup_logger()
        
        # Performance metrics storage
        self.latency_data = deque(maxlen=max_samples)
        self.throughput_data = deque(maxlen=1000)
        self.processing_times = deque(maxlen=max_samples)
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        
        # Latency tracking
        self.pending_operations = {}
        self.operation_start_times = {}
        
        # Performance thresholds (milliseconds)
        self.latency_thresholds = {
            'excellent': 1.0,     # < 1ms
            'good': 5.0,          # < 5ms
            'acceptable': 20.0,   # < 20ms
            'poor': 100.0,        # < 100ms
            'critical': 1000.0    # < 1000ms
        }
        
        # Throughput tracking
        self.message_count = 0
        self.start_time = time.time()
        self.last_throughput_check = time.time()
        
        # Performance statistics
        self.performance_stats = {
            'avg_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': float('inf'),
            'messages_per_second': 0.0,
            'total_messages': 0,
            'avg_cpu_usage': 0.0,
            'avg_memory_usage_mb': 0.0,
            'performance_grade': 'UNKNOWN'
        }
        
        # System monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            'high_latency_ms': 50.0,
            'low_throughput_mps': 100.0,  # messages per second
            'high_cpu_percent': 80.0,
            'high_memory_mb': 1000.0
        }
        
        # Start memory tracking
        tracemalloc.start()
    
    def setup_logger(self):
        """Setup logging system"""
        self.logger = logging.getLogger(f'LatencyMonitor_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start_monitoring(self):
        """เริ่มระบบ monitoring แบบ background"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """หยุดระบบ monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _system_monitor_loop(self):
        """Background loop สำหรับติดตาม system metrics"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                self.memory_usage.append(memory_mb)
                
                # Update performance stats
                self._update_system_stats()
                
                # Check for alerts
                self._check_performance_alerts()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(5)  # Wait longer on error
    
    def start_operation(self, operation_id: str, operation_type: str = "data_processing") -> str:
        """เริ่มติดตาม operation"""
        start_time = time.perf_counter()
        self.operation_start_times[operation_id] = {
            'start_time': start_time,
            'type': operation_type,
            'timestamp': datetime.now()
        }
        return operation_id
    
    def end_operation(self, operation_id: str) -> Optional[float]:
        """จบการติดตาม operation และคำนวณ latency"""
        if operation_id not in self.operation_start_times:
            self.logger.warning(f"Operation {operation_id} not found")
            return None
        
        end_time = time.perf_counter()
        start_info = self.operation_start_times.pop(operation_id)
        
        latency_seconds = end_time - start_info['start_time']
        latency_ms = latency_seconds * 1000
        
        # Store latency data
        self.latency_data.append({
            'operation_id': operation_id,
            'type': start_info['type'],
            'latency_ms': latency_ms,
            'timestamp': start_info['timestamp']
        })
        
        # Update message count
        self.message_count += 1
        self.performance_stats['total_messages'] = self.message_count
        
        # Update statistics
        self._update_latency_stats()
        
        return latency_ms
    
    def measure_function(self, func: Callable, *args, **kwargs) -> Tuple[any, float]:
        """วัดประสิทธิภาพของ function"""
        operation_id = f"func_{func.__name__}_{int(time.time() * 1000000)}"
        
        self.start_operation(operation_id, f"function_{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            latency = self.end_operation(operation_id)
            return result, latency
        except Exception as e:
            self.end_operation(operation_id)
            raise e
    
    def record_tick_latency(self, tick_timestamp: datetime, received_timestamp: datetime = None):
        """บันทึก latency ของ tick data"""
        if received_timestamp is None:
            received_timestamp = datetime.now()
        
        # Calculate data latency (time from tick to receipt)
        if tick_timestamp.tzinfo is None:
            tick_timestamp = tick_timestamp.replace(tzinfo=received_timestamp.tzinfo)
        
        data_latency = (received_timestamp - tick_timestamp).total_seconds() * 1000
        
        # Store tick latency
        operation_id = f"tick_{int(time.time() * 1000000)}"
        self.latency_data.append({
            'operation_id': operation_id,
            'type': 'tick_data',
            'latency_ms': data_latency,
            'timestamp': received_timestamp
        })
        
        self.message_count += 1
        self._update_latency_stats()
    
    def _update_latency_stats(self):
        """อัพเดต latency statistics"""
        if not self.latency_data:
            return
        
        latencies = [item['latency_ms'] for item in self.latency_data]
        
        self.performance_stats['avg_latency_ms'] = statistics.mean(latencies)
        self.performance_stats['min_latency_ms'] = min(latencies)
        self.performance_stats['max_latency_ms'] = max(latencies)
        
        if len(latencies) >= 20:  # Need sufficient samples for percentiles
            self.performance_stats['p95_latency_ms'] = np.percentile(latencies, 95)
            self.performance_stats['p99_latency_ms'] = np.percentile(latencies, 99)
        
        # Calculate throughput
        current_time = time.time()
        time_elapsed = current_time - self.start_time
        if time_elapsed > 0:
            self.performance_stats['messages_per_second'] = self.message_count / time_elapsed
        
        # Update performance grade
        self.performance_stats['performance_grade'] = self._calculate_performance_grade()
    
    def _update_system_stats(self):
        """อัพเดต system statistics"""
        if self.cpu_usage:
            self.performance_stats['avg_cpu_usage'] = statistics.mean(list(self.cpu_usage)[-60:])  # Last 60 seconds
        
        if self.memory_usage:
            self.performance_stats['avg_memory_usage_mb'] = statistics.mean(list(self.memory_usage)[-60:])
    
    def _calculate_performance_grade(self) -> str:
        """คำนวณ performance grade"""
        avg_latency = self.performance_stats['avg_latency_ms']
        p99_latency = self.performance_stats['p99_latency_ms']
        throughput = self.performance_stats['messages_per_second']
        
        # Grade based on latency
        if avg_latency <= self.latency_thresholds['excellent'] and p99_latency <= 5.0:
            latency_grade = 'EXCELLENT'
        elif avg_latency <= self.latency_thresholds['good'] and p99_latency <= 20.0:
            latency_grade = 'GOOD'
        elif avg_latency <= self.latency_thresholds['acceptable']:
            latency_grade = 'ACCEPTABLE'
        elif avg_latency <= self.latency_thresholds['poor']:
            latency_grade = 'POOR'
        else:
            latency_grade = 'CRITICAL'
        
        # Consider throughput
        if throughput < 50:
            return f"{latency_grade}_LOW_THROUGHPUT"
        elif throughput > 1000:
            return f"{latency_grade}_HIGH_THROUGHPUT"
        else:
            return latency_grade
    
    def _check_performance_alerts(self):
        """ตรวจสอบและส่ง performance alerts"""
        current_time = datetime.now()
        
        # High latency alert
        if (self.performance_stats['avg_latency_ms'] > self.alert_thresholds['high_latency_ms']):
            self._send_alert(
                'HIGH_LATENCY',
                f"Average latency: {self.performance_stats['avg_latency_ms']:.2f}ms",
                'WARNING'
            )
        
        # Low throughput alert
        if (self.performance_stats['messages_per_second'] < self.alert_thresholds['low_throughput_mps'] and
            self.message_count > 100):  # Only after sufficient data
            self._send_alert(
                'LOW_THROUGHPUT',
                f"Throughput: {self.performance_stats['messages_per_second']:.1f} msg/s",
                'WARNING'
            )
        
        # High CPU usage alert
        if self.performance_stats['avg_cpu_usage'] > self.alert_thresholds['high_cpu_percent']:
            self._send_alert(
                'HIGH_CPU',
                f"CPU usage: {self.performance_stats['avg_cpu_usage']:.1f}%",
                'WARNING'
            )
        
        # High memory usage alert
        if self.performance_stats['avg_memory_usage_mb'] > self.alert_thresholds['high_memory_mb']:
            self._send_alert(
                'HIGH_MEMORY',
                f"Memory usage: {self.performance_stats['avg_memory_usage_mb']:.1f}MB",
                'WARNING'
            )
    
    def _send_alert(self, alert_type: str, message: str, severity: str):
        """ส่ง performance alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'performance_stats': self.performance_stats.copy()
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        if severity == 'CRITICAL':
            self.logger.critical(f"[{alert_type}] {message}")
        elif severity == 'WARNING':
            self.logger.warning(f"[{alert_type}] {message}")
        else:
            self.logger.info(f"[{alert_type}] {message}")
    
    def get_latency_distribution(self) -> Dict:
        """ดึงการกระจายของ latency"""
        if not self.latency_data:
            return {}
        
        latencies = [item['latency_ms'] for item in self.latency_data]
        
        distribution = {}
        for threshold_name, threshold_value in self.latency_thresholds.items():
            count = sum(1 for lat in latencies if lat <= threshold_value)
            percentage = (count / len(latencies)) * 100
            distribution[threshold_name] = {
                'count': count,
                'percentage': percentage,
                'threshold_ms': threshold_value
            }
        
        return distribution
    
    def get_throughput_analysis(self) -> Dict:
        """วิเคราะห์ throughput ตามเวลา"""
        if not self.latency_data:
            return {}
        
        # Group by minute
        current_time = datetime.now()
        minute_counts = {}
        
        for item in self.latency_data:
            minute_key = item['timestamp'].replace(second=0, microsecond=0)
            minute_counts[minute_key] = minute_counts.get(minute_key, 0) + 1
        
        if minute_counts:
            avg_per_minute = statistics.mean(minute_counts.values())
            max_per_minute = max(minute_counts.values())
            min_per_minute = min(minute_counts.values())
        else:
            avg_per_minute = max_per_minute = min_per_minute = 0
        
        return {
            'avg_messages_per_minute': avg_per_minute,
            'max_messages_per_minute': max_per_minute,
            'min_messages_per_minute': min_per_minute,
            'current_messages_per_second': self.performance_stats['messages_per_second'],
            'total_minutes_tracked': len(minute_counts)
        }
    
    def generate_performance_report(self) -> str:
        """สร้างรายงาน performance analysis"""
        latency_dist = self.get_latency_distribution()
        throughput_analysis = self.get_throughput_analysis()
        
        report = f"""
========== PERFORMANCE ANALYSIS REPORT ==========
Symbol: {self.symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Monitoring Duration: {time.time() - self.start_time:.1f} seconds

LATENCY METRICS:
- Average Latency: {self.performance_stats['avg_latency_ms']:.2f}ms
- P95 Latency: {self.performance_stats['p95_latency_ms']:.2f}ms
- P99 Latency: {self.performance_stats['p99_latency_ms']:.2f}ms
- Min/Max Latency: {self.performance_stats['min_latency_ms']:.2f}/{self.performance_stats['max_latency_ms']:.2f}ms

THROUGHPUT METRICS:
- Messages/Second: {self.performance_stats['messages_per_second']:.1f}
- Total Messages: {self.performance_stats['total_messages']:,}
- Avg Messages/Minute: {throughput_analysis.get('avg_messages_per_minute', 0):.1f}

SYSTEM METRICS:
- Average CPU Usage: {self.performance_stats['avg_cpu_usage']:.1f}%
- Average Memory Usage: {self.performance_stats['avg_memory_usage_mb']:.1f}MB

LATENCY DISTRIBUTION:
"""
        
        for grade, stats in latency_dist.items():
            report += f"- {grade.upper()} (<{stats['threshold_ms']}ms): {stats['percentage']:.1f}%\n"
        
        report += f"""
PERFORMANCE GRADE: {self.performance_stats['performance_grade']}

RECENT ALERTS: {len(self.alerts)}
"""
        
        for alert in self.alerts[-3:]:
            report += f"- {alert['timestamp'].strftime('%H:%M:%S')} [{alert['severity']}] {alert['message']}\n"
        
        report += "=" * 50
        
        return report
    
    def get_performance_summary(self) -> Dict:
        """ดึงสรุปประสิทธิภาพ"""
        return {
            'performance_grade': self.performance_stats['performance_grade'],
            'avg_latency_ms': self.performance_stats['avg_latency_ms'],
            'messages_per_second': self.performance_stats['messages_per_second'],
            'system_health': {
                'cpu_usage': self.performance_stats['avg_cpu_usage'],
                'memory_usage_mb': self.performance_stats['avg_memory_usage_mb']
            },
            'recent_alerts_count': len([a for a in self.alerts if 
                                     (datetime.now() - a['timestamp']).total_seconds() < 300]),  # Last 5 minutes
            'data_points': len(self.latency_data)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize latency monitor
    monitor = LatencyMonitor("XAUUSD")
    monitor.start_monitoring()
    
    print("Testing Latency Monitor...")
    
    try:
        # Simulate various operations
        for i in range(100):
            # Simulate data processing operation
            op_id = f"data_proc_{i}"
            monitor.start_operation(op_id, "data_processing")
            
            # Simulate processing time
            time.sleep(np.random.uniform(0.001, 0.02))  # 1-20ms
            
            latency = monitor.end_operation(op_id)
            
            # Simulate tick data latency
            tick_time = datetime.now() - timedelta(milliseconds=np.random.uniform(1, 50))
            monitor.record_tick_latency(tick_time)
            
            if i % 20 == 0:
                print(f"Processed {i+1} operations...")
        
        # Test function measurement
        def test_function(n):
            """Test function to measure"""
            return sum(range(n))
        
        result, func_latency = monitor.measure_function(test_function, 1000)
        print(f"Function result: {result}, latency: {func_latency:.2f}ms")
        
        # Wait a bit for system monitoring
        time.sleep(2)
        
        # Generate reports
        print("\n" + monitor.generate_performance_report())
        
        # Performance summary
        summary = monitor.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"- Grade: {summary['performance_grade']}")
        print(f"- Avg Latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"- Throughput: {summary['messages_per_second']:.1f} msg/s")
        print(f"- CPU Usage: {summary['system_health']['cpu_usage']:.1f}%")
        
        # Latency distribution
        distribution = monitor.get_latency_distribution()
        print(f"\nLatency Distribution:")
        for grade, stats in distribution.items():
            print(f"- {grade}: {stats['percentage']:.1f}%")
    
    finally:
        monitor.stop_monitoring()
        print("\nMonitoring stopped.")