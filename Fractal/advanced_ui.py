import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import queue
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from collections import deque
import sqlite3
import os
from pathlib import Path

# Fix circular import with TYPE_CHECKING
if TYPE_CHECKING:
    from strategy_engine import StrategyEngine, EngineState
    from trading_core import TradingConfig
    from risk_manager import RiskLevel
    from position_manager import Position

class AlertType(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"
    SIGNAL = "signal"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    auto_dismiss: bool = True
    dismiss_after: int = 5000  # milliseconds
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }

class DataStore:
    """SQLite data store for historical data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    metadata TEXT
                )
            ''')
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    ticket INTEGER,
                    symbol TEXT,
                    type TEXT,
                    volume REAL,
                    open_price REAL,
                    close_price REAL,
                    profit REAL,
                    duration_seconds INTEGER
                )
            ''')
            
            # Signal history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    signal_type TEXT,
                    strength REAL,
                    confidence REAL,
                    rsi_value REAL,
                    executed BOOLEAN,
                    result TEXT
                )
            ''')
            
            # Account snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    balance REAL,
                    equity REAL,
                    margin REAL,
                    free_margin REAL,
                    margin_level REAL,
                    total_positions INTEGER
                )
            ''')
            
            conn.commit()
    
    def store_performance_metric(self, metric_name: str, value: float, unit: str = "", metadata: Dict = None):
        """Store performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), metric_name, value, unit, json.dumps(metadata or {})))
    
    def store_trade(self, trade_data: Dict):
        """Store trade data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trade_history (timestamp, ticket, symbol, type, volume, open_price, close_price, profit, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                trade_data.get('ticket', 0),
                trade_data.get('symbol', ''),
                trade_data.get('type', ''),
                trade_data.get('volume', 0),
                trade_data.get('open_price', 0),
                trade_data.get('close_price', 0),
                trade_data.get('profit', 0),
                trade_data.get('duration_seconds', 0)
            ))
    
    def store_signal(self, signal_data: Dict):
        """Store signal data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_history (timestamp, signal_type, strength, confidence, rsi_value, executed, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                signal_data.get('signal_type', ''),
                signal_data.get('strength', 0),
                signal_data.get('confidence', 0),
                signal_data.get('rsi_value', 0),
                signal_data.get('executed', False),
                signal_data.get('result', '')
            ))
    
    def store_account_snapshot(self, account_data: Dict):
        """Store account snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO account_snapshots (timestamp, balance, equity, margin, free_margin, margin_level, total_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                account_data.get('balance', 0),
                account_data.get('equity', 0),
                account_data.get('margin', 0),
                account_data.get('free_margin', 0),
                account_data.get('margin_level', 0),
                account_data.get('total_positions', 0)
            ))
    
    def get_performance_data(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get performance data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, value, unit, metadata FROM performance_metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (metric_name, cutoff_time))
            
            return [
                {
                    "timestamp": row[0],
                    "value": row[1],
                    "unit": row[2],
                    "metadata": json.loads(row[3])
                }
                for row in cursor.fetchall()
            ]
    
    def get_account_history(self, hours: int = 24) -> List[Dict]:
        """Get account history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, balance, equity, margin_level FROM account_snapshots
                WHERE timestamp > ?
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            return [
                {
                    "timestamp": row[0],
                    "balance": row[1],
                    "equity": row[2],
                    "margin_level": row[3]
                }
                for row in cursor.fetchall()
            ]

class AlertManager:
    """Alert management system"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.alerts = {}
        self.alert_counter = 0
        self.max_alerts = 10
        
        # Alert display frame
        self.alerts_frame = ttk.Frame(parent_widget)
        self.alerts_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Sound alerts (optional)
        self.sound_enabled = True
        
    def add_alert(self, alert_type: AlertType, title: str, message: str, auto_dismiss: bool = True):
        """Add new alert"""
        alert_id = f"alert_{self.alert_counter}"
        self.alert_counter += 1
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            auto_dismiss=auto_dismiss
        )
        
        # Remove oldest alerts if at max capacity
        if len(self.alerts) >= self.max_alerts:
            oldest_id = min(self.alerts.keys(), key=lambda x: self.alerts[x].timestamp)
            self.remove_alert(oldest_id)
        
        self.alerts[alert_id] = alert
        self._display_alert(alert)
        
        # Auto-dismiss if configured
        if auto_dismiss:
            self.parent.after(alert.dismiss_after, lambda: self.remove_alert(alert_id))
        
        # Play sound if enabled
        if self.sound_enabled and alert_type in [AlertType.ERROR, AlertType.WARNING, AlertType.TRADE]:
            self._play_alert_sound(alert_type)
    
    def _display_alert(self, alert: Alert):
        """Display alert in UI"""
        # Create alert frame
        alert_frame = ttk.Frame(self.alerts_frame, relief="raised", borderwidth=1)
        alert_frame.pack(fill=tk.X, pady=1)
        
        # Color coding
        colors = {
            AlertType.INFO: "#E3F2FD",
            AlertType.SUCCESS: "#E8F5E8", 
            AlertType.WARNING: "#FFF3E0",
            AlertType.ERROR: "#FFEBEE",
            AlertType.TRADE: "#E0F2F1",
            AlertType.SIGNAL: "#F3E5F5"
        }
        
        alert_frame.configure(style=f"{alert.type.value.title()}.TFrame")
        
        # Alert content
        content_frame = ttk.Frame(alert_frame)
        content_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Icon and title
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill=tk.X)
        
        # Icon
        icons = {
            AlertType.INFO: "ℹ️",
            AlertType.SUCCESS: "✅",
            AlertType.WARNING: "⚠️", 
            AlertType.ERROR: "❌",
            AlertType.TRADE: "💰",
            AlertType.SIGNAL: "📊"
        }
        
        icon_label = ttk.Label(header_frame, text=icons.get(alert.type, "•"), font=("Arial", 12))
        icon_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Title
        title_label = ttk.Label(header_frame, text=alert.title, font=("Arial", 10, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Timestamp
        time_label = ttk.Label(header_frame, text=alert.timestamp.strftime("%H:%M:%S"), 
                              font=("Arial", 8), foreground="gray")
        time_label.pack(side=tk.RIGHT)
        
        # Message
        if alert.message:
            message_label = ttk.Label(content_frame, text=alert.message, font=("Arial", 9))
            message_label.pack(fill=tk.X, pady=(2, 0))
        
        # Close button
        close_btn = ttk.Button(content_frame, text="×", width=3,
                              command=lambda: self.remove_alert(alert.id))
        close_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Store reference to frame for removal
        alert.frame = alert_frame
    
    def remove_alert(self, alert_id: str):
        """Remove alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            if hasattr(alert, 'frame'):
                alert.frame.destroy()
            del self.alerts[alert_id]
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        for alert_id in list(self.alerts.keys()):
            self.remove_alert(alert_id)
    
    def _play_alert_sound(self, alert_type: AlertType):
        """Play alert sound (placeholder)"""
        # This would play system sounds or custom audio files
        # For now, just use system bell
        try:
            if alert_type in [AlertType.ERROR, AlertType.WARNING]:
                self.parent.bell()
        except:
            pass

class LiveChartWidget:
    """Live chart widget for real-time data visualization"""
    
    def __init__(self, parent, title: str = "Live Chart", max_points: int = 100):
        self.parent = parent
        self.title = title
        self.max_points = max_points
        
        # Data storage
        self.data_series = {}
        self.timestamps = deque(maxlen=max_points)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        
        # Animation
        self.animation = None
        self.is_animating = False
        
        # Colors for different series
        self.colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.color_index = 0
    
    def add_data_series(self, name: str, color: str = None):
        """Add a new data series"""
        if name not in self.data_series:
            if color is None:
                color = self.colors[self.color_index % len(self.colors)]
                self.color_index += 1
            
            self.data_series[name] = {
                'data': deque(maxlen=self.max_points),
                'color': color,
                'line': None
            }
    
    def update_data(self, series_name: str, value: float, timestamp: datetime = None):
        """Update data for a series"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add series if it doesn't exist
        if series_name not in self.data_series:
            self.add_data_series(series_name)
        
        # Add data point
        self.data_series[series_name]['data'].append(value)
        
        # Update timestamps (only need one timeline)
        if len(self.timestamps) == 0 or timestamp > self.timestamps[-1]:
            self.timestamps.append(timestamp)
    
    def start_animation(self, interval: int = 1000):
        """Start real-time animation"""
        if not self.is_animating:
            self.animation = FuncAnimation(
                self.fig, self._animate, interval=interval, blit=False
            )
            self.is_animating = True
            self.canvas.draw()
    
    def stop_animation(self):
        """Stop animation"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_animating = False
    
    def _animate(self, frame):
        """Animation function"""
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)
        
        if len(self.timestamps) < 2:
            return
        
        # Plot each data series
        for name, series in self.data_series.items():
            if len(series['data']) > 0:
                # Ensure data and timestamps have same length
                data_len = len(series['data'])
                time_data = list(self.timestamps)[-data_len:]
                
                self.ax.plot(time_data, list(series['data']), 
                           label=name, color=series['color'], linewidth=2)
        
        # Format x-axis for timestamps
        self.ax.tick_params(axis='x', rotation=45)
        
        # Legend
        if self.data_series:
            self.ax.legend()
        
        # Adjust layout
        self.fig.tight_layout()
    
    def manual_refresh(self):
        """Manually refresh the chart"""
        self._animate(None)
        self.canvas.draw()

class PerformanceDashboard:
    """Performance dashboard with multiple charts and metrics"""
    
    def __init__(self, parent):
        self.parent = parent
        self.data_store = DataStore()
        
        # Create dashboard frame
        self.dashboard_frame = ttk.Frame(parent)
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different chart types
        self.chart_notebook = ttk.Notebook(self.dashboard_frame)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Account Performance Tab
        self.account_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.account_frame, text="Account")
        self.account_chart = LiveChartWidget(self.account_frame, "Account Performance")
        self.account_chart.add_data_series("Balance", "blue")
        self.account_chart.add_data_series("Equity", "green")
        
        # Signal Analysis Tab
        self.signals_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.signals_frame, text="Signals")
        self.signals_chart = LiveChartWidget(self.signals_frame, "Signal Strength")
        self.signals_chart.add_data_series("Signal Strength", "purple")
        self.signals_chart.add_data_series("Confidence", "orange")
        
        # Performance Metrics Tab
        self.performance_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.performance_frame, text="Performance")
        self.performance_chart = LiveChartWidget(self.performance_frame, "System Performance")
        self.performance_chart.add_data_series("Loop Time (ms)", "red")
        self.performance_chart.add_data_series("Memory Usage", "brown")
        
        # Risk Metrics Tab
        self.risk_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.risk_frame, text="Risk")
        self.risk_chart = LiveChartWidget(self.risk_frame, "Risk Metrics")
        self.risk_chart.add_data_series("Drawdown %", "red")
        self.risk_chart.add_data_series("Margin Level", "blue")
        
        # Control panel
        self.create_control_panel()
        
        # Auto-refresh timer
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 seconds
        self.start_auto_refresh()
    
    def create_control_panel(self):
        """Create dashboard control panel"""
        control_frame = ttk.Frame(self.dashboard_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto Refresh", 
                       variable=self.auto_refresh_var,
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Manual refresh button
        ttk.Button(control_frame, text="Refresh Now", 
                  command=self.manual_refresh).pack(side=tk.LEFT, padx=5)
        
        # Export button
        ttk.Button(control_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Time range selector
        ttk.Label(control_frame, text="Time Range:").pack(side=tk.LEFT, padx=(20, 5))
        self.time_range_var = tk.StringVar(value="1h")
        time_combo = ttk.Combobox(control_frame, textvariable=self.time_range_var,
                                values=["15m", "30m", "1h", "4h", "1d", "1w"],
                                state="readonly", width=8)
        time_combo.pack(side=tk.LEFT, padx=5)
        time_combo.bind("<<ComboboxSelected>>", self.on_time_range_changed)
        
        # Chart type selector
        ttk.Label(control_frame, text="Chart:").pack(side=tk.LEFT, padx=(20, 5))
        chart_types = ["Account", "Signals", "Performance", "Risk"]
        for chart_type in chart_types:
            ttk.Button(control_frame, text=chart_type, width=10,
                      command=lambda ct=chart_type: self.switch_chart(ct)).pack(side=tk.LEFT, padx=2)
    
    def update_account_data(self, balance: float, equity: float):
        """Update account performance chart"""
        timestamp = datetime.now()
        self.account_chart.update_data("Balance", balance, timestamp)
        self.account_chart.update_data("Equity", equity, timestamp)
        
        # Store in database
        self.data_store.store_account_snapshot({
            "balance": balance,
            "equity": equity,
            "margin": 0,
            "free_margin": 0,
            "margin_level": 999.99,
            "total_positions": 0
        })
    
    def update_signal_data(self, strength: float, confidence: float):
        """Update signal analysis chart"""
        timestamp = datetime.now()
        self.signals_chart.update_data("Signal Strength", strength, timestamp)
        self.signals_chart.update_data("Confidence", confidence, timestamp)
    
    def update_performance_data(self, loop_time_ms: float, memory_usage: float = 0):
        """Update performance metrics chart"""
        timestamp = datetime.now()
        self.performance_chart.update_data("Loop Time (ms)", loop_time_ms, timestamp)
        if memory_usage > 0:
            self.performance_chart.update_data("Memory Usage", memory_usage, timestamp)
        
        # Store in database
        self.data_store.store_performance_metric("loop_time", loop_time_ms, "ms")
    
    def update_risk_data(self, drawdown_percent: float, margin_level: float):
        """Update risk metrics chart"""
        timestamp = datetime.now()
        self.risk_chart.update_data("Drawdown %", drawdown_percent, timestamp)
        self.risk_chart.update_data("Margin Level", margin_level, timestamp)
    
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        if self.auto_refresh:
            self.refresh_charts()
            self.parent.after(self.refresh_interval, self.start_auto_refresh)
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.start_auto_refresh()
    
    def manual_refresh(self):
        """Manually refresh all charts"""
        self.refresh_charts()
    
    def refresh_charts(self):
        """Refresh all charts"""
        try:
            self.account_chart.manual_refresh()
            self.signals_chart.manual_refresh()
            self.performance_chart.manual_refresh()
            self.risk_chart.manual_refresh()
        except Exception as e:
            print(f"Chart refresh error: {e}")
    
    def switch_chart(self, chart_type: str):
        """Switch to specific chart tab"""
        chart_map = {
            "Account": 0,
            "Signals": 1,
            "Performance": 2,
            "Risk": 3
        }
        
        if chart_type in chart_map:
            self.chart_notebook.select(chart_map[chart_type])
    
    def on_time_range_changed(self, event=None):
        """Handle time range change"""
        # This would reload data for the selected time range
        # For now, just refresh the charts
        self.manual_refresh()
    
    def export_data(self):
        """Export chart data to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Export current chart data
                current_tab = self.chart_notebook.select()
                tab_index = self.chart_notebook.index(current_tab)
                
                chart_map = {
                    0: self.account_chart,
                    1: self.signals_chart,
                    2: self.performance_chart,
                    3: self.risk_chart
                }
                
                chart = chart_map.get(tab_index)
                if chart:
                    self._export_chart_data(chart, filename)
                    messagebox.showinfo("Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export_chart_data(self, chart: LiveChartWidget, filename: str):
        """Export chart data to CSV file"""
        data_rows = []
        
        # Get timestamps
        timestamps = list(chart.timestamps)
        
        # Get all series data
        for name, series in chart.data_series.items():
            data = list(series['data'])
            
            # Align data with timestamps
            for i, (timestamp, value) in enumerate(zip(timestamps[-len(data):], data)):
                if len(data_rows) <= i:
                    data_rows.append({"timestamp": timestamp})
                data_rows[i][name] = value
        
        # Write to CSV
        if data_rows:
            import csv
            
            fieldnames = ["timestamp"] + list(chart.data_series.keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_rows)

class ConfigurationManager:
    """Advanced configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Current configuration
        self.current_config = {}
        
        # Configuration history
        self.config_history = []
        self.max_history = 50
    
    def save_config(self, config: Dict, name: str = None, description: str = ""):
        """Save configuration with metadata"""
        if name is None:
            name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_data = {
            "name": name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "version": "1.0"
        }
        
        # Save to file
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Add to history
        self.config_history.append(config_data)
        if len(self.config_history) > self.max_history:
            self.config_history.pop(0)
        
        return str(config_file)
    
    def load_config(self, name: str) -> Dict:
        """Load configuration by name"""
        config_file = self.config_dir / f"{name}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get("config", {})
        
        return {}
    
    def list_configs(self) -> List[Dict]:
        """List all saved configurations"""
        configs = []
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    configs.append({
                        "name": config_data.get("name", config_file.stem),
                        "description": config_data.get("description", ""),
                        "timestamp": config_data.get("timestamp", ""),
                        "file": str(config_file)
                    })
            except:
                continue
        
        return sorted(configs, key=lambda x: x["timestamp"], reverse=True)
    
    def delete_config(self, name: str) -> bool:
        """Delete configuration"""
        config_file = self.config_dir / f"{name}.json"
        
        if config_file.exists():
            config_file.unlink()
            return True
        
        return False
    
    def backup_configs(self, backup_path: str = None) -> str:
        """Backup all configurations"""
        if backup_path is None:
            backup_path = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        import zipfile
        
        with zipfile.ZipFile(backup_path, 'w') as zipf:
            for config_file in self.config_dir.glob("*.json"):
                zipf.write(config_file, config_file.name)
        
        return backup_path
    
    def restore_configs(self, backup_path: str) -> bool:
        """Restore configurations from backup"""
        try:
            import zipfile
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(self.config_dir)
            
            return True
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

# Enhanced UI Controller with advanced features
class AdvancedUIController:
    """Enhanced UI controller with advanced features"""
    
    def __init__(self, main_ui):
        self.main_ui = main_ui
        self.root = main_ui.root
        
        # Advanced components
        self.alert_manager = None
        self.performance_dashboard = None
        self.config_manager = ConfigurationManager()
        self.data_store = DataStore()
        
        # Initialize advanced features
        self.setup_advanced_features()
    
    def setup_advanced_features(self):
        """Setup advanced UI features"""
        # Add alert manager to main window
        alert_container = ttk.Frame(self.root)
        alert_container.pack(side=tk.TOP, fill=tk.X, before=self.main_ui.notebook)
        self.alert_manager = AlertManager(alert_container)
        
        # Add performance dashboard tab
        self.performance_dashboard = PerformanceDashboard(self.main_ui.advanced_frame)
        
        # Add advanced menu bar
        self.create_advanced_menu()
        
        # Add status bar
        self.create_status_bar()
    
    def create_advanced_menu(self):
        """Create advanced menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.save_configuration)
        file_menu.add_command(label="Load Configuration", command=self.load_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Export Logs", command=self.export_logs)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.main_ui.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="System Diagnostics", command=self.show_diagnostics)
        tools_menu.add_command(label="Performance Report", command=self.show_performance_report)
        tools_menu.add_command(label="Clear All Data", command=self.clear_all_data)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Alerts", command=self.show_alerts)
        view_menu.add_command(label="Performance Dashboard", command=self.show_performance_dashboard)
        view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status sections
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(self.status_bar, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        
        # Connection status
        self.connection_status = tk.StringVar(value="Disconnected")
        ttk.Label(self.status_bar, textvariable=self.connection_status).pack(side=tk.RIGHT, padx=5)
        
        # Memory usage
        self.memory_status = tk.StringVar(value="Memory: 0 MB")
        ttk.Label(self.status_bar, textvariable=self.memory_status).pack(side=tk.RIGHT, padx=5)
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_text.set(message)
    
    def update_connection_status(self, connected: bool):
        """Update connection status"""
        status = "Connected" if connected else "Disconnected"
        self.connection_status.set(f"MT5: {status}")
    
    def update_memory_usage(self, memory_mb: float):
        """Update memory usage display"""
        self.memory_status.set(f"Memory: {memory_mb:.1f} MB")
    
    # Alert methods
    def show_info_alert(self, title: str, message: str):
        """Show info alert"""
        self.alert_manager.add_alert(AlertType.INFO, title, message)
    
    def show_success_alert(self, title: str, message: str):
        """Show success alert"""
        self.alert_manager.add_alert(AlertType.SUCCESS, title, message)
    
    def show_warning_alert(self, title: str, message: str):
        """Show warning alert"""
        self.alert_manager.add_alert(AlertType.WARNING, title, message, auto_dismiss=False)
    
    def show_error_alert(self, title: str, message: str):
        """Show error alert"""
        self.alert_manager.add_alert(AlertType.ERROR, title, message, auto_dismiss=False)
    
    def show_trade_alert(self, title: str, message: str):
        """Show trade alert"""
        self.alert_manager.add_alert(AlertType.TRADE, title, message)
    
    def show_signal_alert(self, title: str, message: str):
        """Show signal alert"""
        self.alert_manager.add_alert(AlertType.SIGNAL, title, message)
    
    # Configuration management
    def save_configuration(self):
        """Save current configuration"""
        try:
            # Get current config from main UI
            config = self.main_ui._collect_parameters() if hasattr(self.main_ui, '_collect_parameters') else {}
            
            # Ask for name and description
            dialog = ConfigSaveDialog(self.root)
            result = dialog.show()
            
            if result:
                name, description = result
                file_path = self.config_manager.save_config(config, name, description)
                self.show_success_alert("Configuration Saved", f"Saved to {file_path}")
            
        except Exception as e:
            self.show_error_alert("Save Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        """Load configuration"""
        try:
            configs = self.config_manager.list_configs()
            
            if not configs:
                self.show_info_alert("No Configurations", "No saved configurations found")
                return