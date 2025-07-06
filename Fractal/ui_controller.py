import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from strategy_engine import StrategyEngine, EngineState
from trading_core import TradingConfig
from risk_manager import RiskLevel

class UITheme(Enum):
    DARK = "dark"
    LIGHT = "light"
    CUSTOM = "custom"

@dataclass
class UIConfig:
    """UI Configuration"""
    theme: UITheme = UITheme.DARK
    update_interval: float = 1.0
    chart_history_bars: int = 100
    log_max_lines: int = 1000
    auto_scroll_logs: bool = True
    show_advanced_controls: bool = False
    position_in_title: bool = True
    sound_alerts: bool = True
    
class PresetManager:
    """Trading preset configurations"""
    
    PRESETS = {
        "Scalping": {
            "lot_size": 0.01,
            "rsi_up": 60,
            "rsi_down": 40,
            "tp_first": 150,
            "exit_speed": 0,  # FAST
            "recovery_price": 80,
            "martingale": 1.5,
            "max_recovery": 2,
            "primary_tf": "M5"
        },
        "Intraday": {
            "lot_size": 0.02,
            "rsi_up": 55,
            "rsi_down": 45,
            "tp_first": 200,
            "exit_speed": 1,  # MEDIUM
            "recovery_price": 100,
            "martingale": 2.0,
            "max_recovery": 3,
            "primary_tf": "M15"
        },
        "Swing": {
            "lot_size": 0.05,
            "rsi_up": 50,
            "rsi_down": 50,
            "tp_first": 300,
            "exit_speed": 2,  # SLOW
            "recovery_price": 150,
            "martingale": 2.5,
            "max_recovery": 4,
            "primary_tf": "H1"
        },
        "Conservative": {
            "lot_size": 0.01,
            "rsi_up": 65,
            "rsi_down": 35,
            "tp_first": 250,
            "exit_speed": 1,
            "recovery_price": 120,
            "martingale": 1.8,
            "max_recovery": 2,
            "primary_tf": "H1"
        }
    }

class XAUUSDTradingUI:
    def __init__(self):
        # Initialize components
        self.engine = None
        self.ui_config = UIConfig()
        self.preset_manager = PresetManager()
        
        # UI state
        self.running = False
        self.update_thread = None
        self.last_update = None
        
        # Data for UI
        self.status_data = {}
        self.position_data = []
        self.recovery_data = []
        self.performance_data = {}
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("XAUUSD Multi-Timeframe EA - Professional Trading System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure styles
        self.setup_styles()
        
        # Create UI components
        self.create_menu()
        self.create_main_layout()
        self.create_control_panel()
        self.create_status_panel()
        self.create_trading_panel()
        self.create_risk_panel()
        self.create_positions_panel()
        self.create_logs_panel()
        
        # Create logs panel first so log_text exists
        self.create_logs_panel()
        
        # Now setup logging with UI handler
        self.setup_ui_logging()
        
        # Setup event bindings
        self.setup_event_bindings()
        
        # Initialize engine
        # self.initialize_engine()
    
    def setup_styles(self):
        """Setup UI styles and themes"""
        self.style = ttk.Style()
        
        if self.ui_config.theme == UITheme.DARK:
            # Dark theme colors
            self.colors = {
                'bg': '#1e1e1e',
                'fg': '#ffffff',
                'select_bg': '#404040',
                'button_bg': '#404040',
                'success': '#00ff00',
                'warning': '#ffaa00',
                'error': '#ff4444',
                'profit': '#00aa00',
                'loss': '#aa0000'
            }
        else:
            # Light theme colors
            self.colors = {
                'bg': '#ffffff',
                'fg': '#000000',
                'select_bg': '#e0e0e0',
                'button_bg': '#f0f0f0',
                'success': '#008800',
                'warning': '#cc8800',
                'error': '#cc0000',
                'profit': '#006600',
                'loss': '#cc0000'
            }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg'])
    
    def create_menu(self):
        """Create menu bar"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Logs", command=self.export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Presets menu
        presets_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Presets", menu=presets_menu)
        for preset_name in self.preset_manager.PRESETS.keys():
            presets_menu.add_command(
                label=preset_name,
                command=lambda name=preset_name: self.load_preset(name)
            )
        
        # Tools menu
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Risk Calculator", command=self.open_risk_calculator)
        tools_menu.add_command(label="Performance Report", command=self.open_performance_report)
        tools_menu.add_command(label="Settings", command=self.open_settings)
        
        # Help menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create main layout with panels"""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main trading tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Trading")
        
        # Advanced tab
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced")
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Configure main frame layout
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(1, weight=1)
    
    def create_control_panel(self):
        """Create main control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Engine Control", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Engine state display
        self.state_var = tk.StringVar(value="STOPPED")
        self.state_label = ttk.Label(control_frame, textvariable=self.state_var, 
                                   font=("Arial", 12, "bold"))
        self.state_label.grid(row=0, column=0, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=1, padx=20)
        
        self.start_btn = ttk.Button(button_frame, text="START", command=self.start_engine,
                                  style="Success.TButton")
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="STOP", command=self.stop_engine,
                                 style="Warning.TButton")
        self.stop_btn.grid(row=0, column=1, padx=2)
        
        self.pause_btn = ttk.Button(button_frame, text="PAUSE", command=self.pause_engine)
        self.pause_btn.grid(row=0, column=2, padx=2)
        
        self.emergency_btn = ttk.Button(button_frame, text="EMERGENCY STOP", 
                                      command=self.emergency_stop,
                                      style="Danger.TButton")
        self.emergency_btn.grid(row=0, column=3, padx=10)
        
        # Quick preset selector
        preset_frame = ttk.Frame(control_frame)
        preset_frame.grid(row=0, column=2, padx=20)
        
        ttk.Label(preset_frame, text="Quick Preset:").grid(row=0, column=0)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                  values=list(self.preset_manager.PRESETS.keys()),
                                  state="readonly", width=12)
        preset_combo.grid(row=0, column=1, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        # Status indicators
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=3, padx=20)
        
        # Connection status
        self.connection_var = tk.StringVar(value="Disconnected")
        ttk.Label(status_frame, text="MT5:").grid(row=0, column=0)
        self.connection_label = ttk.Label(status_frame, textvariable=self.connection_var)
        self.connection_label.grid(row=0, column=1, padx=5)
        
        # Uptime
        self.uptime_var = tk.StringVar(value="00:00:00")
        ttk.Label(status_frame, text="Uptime:").grid(row=1, column=0)
        ttk.Label(status_frame, textvariable=self.uptime_var).grid(row=1, column=1, padx=5)
    
    def create_status_panel(self):
        """Create status and metrics panel"""
        status_frame = ttk.LabelFrame(self.main_frame, text="Live Status", padding="5")
        status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Real-time metrics
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable text widget for status
        status_text_frame = ttk.Frame(metrics_frame)
        status_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_text_frame, height=15, width=40, 
                                 bg=self.colors['bg'], fg=self.colors['fg'])
        status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", 
                                       command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_trading_panel(self):
        """Create trading parameters panel"""
        trading_frame = ttk.LabelFrame(self.main_frame, text="Trading Parameters", padding="5")
        trading_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create notebook for parameter categories
        param_notebook = ttk.Notebook(trading_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Entry Settings Tab
        entry_frame = ttk.Frame(param_notebook)
        param_notebook.add(entry_frame, text="Entry")
        self.create_entry_parameters(entry_frame)
        
        # Exit Settings Tab
        exit_frame = ttk.Frame(param_notebook)
        param_notebook.add(exit_frame, text="Exit")
        self.create_exit_parameters(exit_frame)
        
        # Recovery Settings Tab
        recovery_frame = ttk.Frame(param_notebook)
        param_notebook.add(recovery_frame, text="Recovery")
        self.create_recovery_parameters(recovery_frame)
        
        # Risk Settings Tab
        risk_frame = ttk.Frame(param_notebook)
        param_notebook.add(risk_frame, text="Risk")
        self.create_risk_parameters(risk_frame)
    
    def create_entry_parameters(self, parent):
        """Create entry parameter controls"""
        # Lot Size
        row = 0
        ttk.Label(parent, text="Lot Size:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lot_size_var = tk.DoubleVar(value=0.01)
        lot_spin = ttk.Spinbox(parent, from_=0.01, to=10.0, increment=0.01, 
                              textvariable=self.lot_size_var, width=10)
        lot_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # RSI Upper
        row += 1
        ttk.Label(parent, text="RSI Upper:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.rsi_up_var = tk.IntVar(value=55)
        rsi_up_spin = ttk.Spinbox(parent, from_=50, to=80, increment=1, 
                                 textvariable=self.rsi_up_var, width=10)
        rsi_up_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # RSI Lower
        row += 1
        ttk.Label(parent, text="RSI Lower:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.rsi_down_var = tk.IntVar(value=45)
        rsi_down_spin = ttk.Spinbox(parent, from_=20, to=50, increment=1, 
                                   textvariable=self.rsi_down_var, width=10)
        rsi_down_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Trading Direction
        row += 1
        ttk.Label(parent, text="Direction:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.direction_var = tk.StringVar(value="BOTH")
        direction_combo = ttk.Combobox(parent, textvariable=self.direction_var,
                                     values=["BOTH", "BUY_ONLY", "SELL_ONLY", "STOP"],
                                     state="readonly", width=12)
        direction_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Primary Timeframe
        row += 1
        ttk.Label(parent, text="Timeframe:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.timeframe_var = tk.StringVar(value="M15")
        tf_combo = ttk.Combobox(parent, textvariable=self.timeframe_var,
                               values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                               state="readonly", width=12)
        tf_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Apply button
        row += 1
        apply_btn = ttk.Button(parent, text="Apply Changes", command=self.apply_parameters)
        apply_btn.grid(row=row, column=0, columnspan=2, pady=10)
    
    def create_exit_parameters(self, parent):
        """Create exit parameter controls"""
        # Take Profit
        row = 0
        ttk.Label(parent, text="TP Points:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.tp_first_var = tk.IntVar(value=200)
        tp_spin = ttk.Spinbox(parent, from_=50, to=1000, increment=10, 
                             textvariable=self.tp_first_var, width=10)
        tp_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Exit Speed
        row += 1
        ttk.Label(parent, text="Exit Speed:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.exit_speed_var = tk.StringVar(value="MEDIUM")
        speed_combo = ttk.Combobox(parent, textvariable=self.exit_speed_var,
                                  values=["FAST", "MEDIUM", "SLOW"],
                                  state="readonly", width=12)
        speed_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Dynamic TP
        row += 1
        self.dynamic_tp_var = tk.BooleanVar(value=True)
        dynamic_check = ttk.Checkbutton(parent, text="Dynamic TP for Recovery",
                                       variable=self.dynamic_tp_var)
        dynamic_check.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
    
    def create_recovery_parameters(self, parent):
        """Create recovery parameter controls"""
        # Recovery Price
        row = 0
        ttk.Label(parent, text="Recovery at Loss:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.recovery_price_var = tk.IntVar(value=100)
        recovery_spin = ttk.Spinbox(parent, from_=50, to=500, increment=10, 
                                   textvariable=self.recovery_price_var, width=10)
        recovery_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="points").grid(row=row, column=2, sticky="w", padx=5)
        
        # Martingale
        row += 1
        ttk.Label(parent, text="Martingale:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.martingale_var = tk.DoubleVar(value=2.0)
        martingale_spin = ttk.Spinbox(parent, from_=1.1, to=5.0, increment=0.1, 
                                     textvariable=self.martingale_var, width=10)
        martingale_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Max Recovery
        row += 1
        ttk.Label(parent, text="Max Recovery:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_recovery_var = tk.IntVar(value=3)
        max_recovery_spin = ttk.Spinbox(parent, from_=1, to=10, increment=1, 
                                       textvariable=self.max_recovery_var, width=10)
        max_recovery_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Smart Recovery
        row += 1
        self.smart_recovery_var = tk.BooleanVar(value=True)
        smart_check = ttk.Checkbutton(parent, text="Smart Recovery (Wait for same signal)",
                                     variable=self.smart_recovery_var)
        smart_check.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
    
    def create_risk_parameters(self, parent):
        """Create risk parameter controls"""
        # Daily Loss Limit
        row = 0
        ttk.Label(parent, text="Daily Loss Limit:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.daily_loss_var = tk.DoubleVar(value=100.0)
        daily_spin = ttk.Spinbox(parent, from_=10, to=1000, increment=10, 
                                textvariable=self.daily_loss_var, width=10)
        daily_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="USD").grid(row=row, column=2, sticky="w", padx=5)
        
        # Max Drawdown
        row += 1
        ttk.Label(parent, text="Max Drawdown:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_drawdown_var = tk.DoubleVar(value=10.0)
        drawdown_spin = ttk.Spinbox(parent, from_=1, to=50, increment=1, 
                                   textvariable=self.max_drawdown_var, width=10)
        drawdown_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="%").grid(row=row, column=2, sticky="w", padx=5)
        
        # Max Positions
        row += 1
        ttk.Label(parent, text="Max Positions:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_positions_var = tk.IntVar(value=5)
        positions_spin = ttk.Spinbox(parent, from_=1, to=20, increment=1, 
                                    textvariable=self.max_positions_var, width=10)
        positions_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Max Spread
        row += 1
        ttk.Label(parent, text="Max Spread:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_spread_var = tk.IntVar(value=30)
        spread_spin = ttk.Spinbox(parent, from_=5, to=100, increment=5, 
                                 textvariable=self.max_spread_var, width=10)
        spread_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="points").grid(row=row, column=2, sticky="w", padx=5)
    
    def create_risk_panel(self):
        """Create risk monitoring panel"""
        # This would be in the advanced tab
        risk_frame = ttk.LabelFrame(self.advanced_frame, text="Risk Monitor", padding="5")
        risk_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Risk level indicator
        level_frame = ttk.Frame(risk_frame)
        level_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(level_frame, text="Risk Level:").pack(side=tk.LEFT)
        self.risk_level_var = tk.StringVar(value="LOW")
        self.risk_level_label = ttk.Label(level_frame, textvariable=self.risk_level_var,
                                         font=("Arial", 12, "bold"))
        self.risk_level_label.pack(side=tk.LEFT, padx=10)
        
        # Risk metrics
        metrics_frame = ttk.Frame(risk_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.risk_text = tk.Text(metrics_frame, height=20, 
                               bg=self.colors['bg'], fg=self.colors['fg'])
        risk_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", 
                                     command=self.risk_text.yview)
        self.risk_text.configure(yscrollcommand=risk_scrollbar.set)
        
        self.risk_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        risk_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_positions_panel(self):
        """Create positions monitoring panel"""
        # This would be in the analysis tab
        positions_frame = ttk.LabelFrame(self.analysis_frame, text="Active Positions", padding="5")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Position tree view
        columns = ("Ticket", "Type", "Volume", "Open Price", "Current Price", "Profit", "Recovery")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", 
                                          command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Position control buttons
        button_frame = ttk.Frame(positions_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Close Selected", 
                  command=self.close_selected_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close All", 
                  command=self.close_all_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh", 
                  command=self.refresh_positions).pack(side=tk.LEFT, padx=5)
    
    def create_logs_panel(self):
        """Create logging panel"""
        logs_frame = ttk.LabelFrame(self.analysis_frame, text="System Logs", padding="5")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text widget
        self.log_text = tk.Text(logs_frame, height=15, 
                              bg=self.colors['bg'], fg=self.colors['fg'])
        log_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", 
                                    command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_controls, text="Clear", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Export", command=self.export_logs).pack(side=tk.LEFT, padx=5)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto Scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=10)
    
    def setup_event_bindings(self):
        """Setup event bindings"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind parameter changes to auto-apply (with delay)
        self.lot_size_var.trace_add("write", self.on_parameter_changed)
        self.rsi_up_var.trace_add("write", self.on_parameter_changed)
        self.rsi_down_var.trace_add("write", self.on_parameter_changed)
    
    def initialize_engine(self):
        """Initialize trading engine"""
        try:
            config = TradingConfig()
            self.engine = StrategyEngine(config)
            
            # Setup event handlers
            self.engine.add_event_handler('on_trade_opened', self.on_trade_opened)
            self.engine.add_event_handler('on_trade_closed', self.on_trade_closed)
            self.engine.add_event_handler('on_state_changed', self.on_engine_state_changed)
            self.engine.add_event_handler('on_error', self.on_engine_error)
            
            self.logger.info("Engine initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize engine: {e}"
            print(error_msg)  # Fallback to print if logger fails
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def setup_ui_logging(self):
        """Setup UI logging handler after log_text is created"""
        # Create logger
        self.logger = logging.getLogger(f"{__name__}.UI")
        
        class UILogHandler(logging.Handler):
            def __init__(self, text_widget, auto_scroll_var):
                super().__init__()
                self.text_widget = text_widget
                self.auto_scroll_var = auto_scroll_var
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.text_widget.insert(tk.END, msg + "\n")
                    
                    # Auto scroll if enabled
                    if self.auto_scroll_var.get():
                        self.text_widget.see(tk.END)
                    
                    # Limit log lines
                    lines = int(self.text_widget.index('end-1c').split('.')[0])
                    if lines > 1000:
                        self.text_widget.delete('1.0', '100.0')
                        
                except:
                    pass
        
        # Add UI log handler
        ui_handler = UILogHandler(self.log_text, self.auto_scroll_var)
        ui_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add to root logger
        logging.getLogger().addHandler(ui_handler)
        logging.getLogger().setLevel(logging.INFO)
    
    def start_ui_updates(self):
        """Start UI update thread"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.ui_update_loop, daemon=True)
            self.update_thread.start()
    
    def stop_ui_updates(self):
        """Stop UI update thread"""
        self.running = False
    
    def ui_update_loop(self):
        """UI update loop"""
        while self.running:
            try:
                self.update_ui_data()
                time.sleep(self.ui_config.update_interval)
            except Exception as e:
                self.logger.error(f"UI update error: {e}")
    
    def update_ui_data(self):
        """Update UI with latest data"""
        if not self.engine:
            return
        
        try:
            # Get engine status
            status = self.engine.get_detailed_status()
            
            # Update UI elements in main thread
            self.root.after(0, self.update_status_display, status)
            
        except Exception as e:
            self.logger.error(f"Failed to update UI data: {e}")
    
    def update_status_display(self, status):
        """Update status display (called in main thread)"""
        try:
            # Update engine state
            engine_state = status.get('engine', {}).get('state', 'UNKNOWN')
            self.state_var.set(engine_state)
            
            # Update connection status
            self.connection_var.set("Connected" if engine_state != "STOPPED" else "Disconnected")
            
            # Update uptime
            uptime_seconds = status.get('engine', {}).get('uptime', 0)
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            self.uptime_var.set(uptime_str)
            
            # Update status text
            self.update_status_text(status)
            
            # Update risk display
            self.update_risk_display(status.get('risk', {}))
            
            # Update positions
            self.update_positions_display(status.get('positions', {}))
            
        except Exception as e:
            self.logger.error(f"Status display update error: {e}")
    
    def update_status_text(self, status):
        """Update status text widget"""
        self.status_text.delete('1.0', tk.END)
        
        lines = [
            f"Engine State: {status.get('engine', {}).get('state', 'UNKNOWN')}",
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}",
            "",
            "TRADING METRICS:",
            f"Total Trades: {status.get('trading', {}).get('total_trades', 0)}",
            f"Successful: {status.get('trading', {}).get('successful_trades', 0)}",
            f"Current Positions: {status.get('trading', {}).get('current_positions', 0)}",
            f"Total P&L: ${status.get('trading', {}).get('total_pnl', 0):.2f}",
            "",
            "RISK STATUS:",
            f"Risk Level: {status.get('risk', {}).get('risk_level', 'unknown').upper()}",
            f"Trading Allowed: {status.get('risk', {}).get('trading_allowed', False)}",
        ]
        
        # Add restrictions if any
        restrictions = status.get('risk', {}).get('restrictions', [])
        if restrictions:
            lines.append("")
            lines.append("RESTRICTIONS:")
            for restriction in restrictions:
                lines.append(f"• {restriction}")
        
        # Add market conditions
        market = status.get('risk', {}).get('market_condition', {})
        if market:
            lines.extend([
                "",
                "MARKET CONDITIONS:",
                f"Volatility: {market.get('volatility', 0):.2f}%",
                f"Session: {market.get('session', 'unknown').title()}",
                f"High Spread: {market.get('high_spread', False)}",
                f"Low Liquidity: {market.get('low_liquidity', False)}"
            ])
        
        self.status_text.insert('1.0', '\n'.join(lines))
    
    def update_risk_display(self, risk_data):
        """Update risk display"""
        risk_level = risk_data.get('risk_level', 'unknown')
        self.risk_level_var.set(risk_level.upper())
        
        # Update risk level label color
        colors = {
            'low': self.colors['success'],
            'medium': self.colors['warning'],
            'high': self.colors['error'],
            'critical': self.colors['error']
        }
        color = colors.get(risk_level.lower(), self.colors['fg'])
        self.risk_level_label.config(foreground=color)
        
        # Update risk text
        if hasattr(self, 'risk_text'):
            self.risk_text.delete('1.0', tk.END)
            
            metrics = risk_data.get('metrics', {})
            lines = [
                f"Daily P&L: ${metrics.get('daily_pnl', 0):.2f}",
                f"Weekly P&L: ${metrics.get('weekly_pnl', 0):.2f}",
                f"Monthly P&L: ${metrics.get('monthly_pnl', 0):.2f}",
                f"Current Drawdown: {metrics.get('current_drawdown', 0):.2f}%",
                f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%",
                f"Exposure: {metrics.get('exposure', 0):.2f}%",
                f"Win Rate: {metrics.get('win_rate', 0):.1f}%",
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
                "",
                "LIMITS:",
                f"Daily Loss Limit: ${risk_data.get('limits', {}).get('daily_loss_limit', 0):.2f}",
                f"Max Drawdown: {risk_data.get('limits', {}).get('max_drawdown_percent', 0):.1f}%",
                f"Max Positions: {risk_data.get('limits', {}).get('max_positions', 0)}",
                f"Max Exposure: {risk_data.get('limits', {}).get('max_exposure_percent', 0):.1f}%"
            ]
            
            self.risk_text.insert('1.0', '\n'.join(lines))
    
    def update_positions_display(self, positions_data):
        """Update positions display"""
        if not hasattr(self, 'positions_tree'):
            return
        
        # Clear existing items
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Add current positions (this would need to be implemented with actual position data)
        # For now, showing placeholder
        if positions_data.get('total_positions', 0) > 0:
            self.positions_tree.insert('', 'end', values=(
                "Loading...", "positions", "data", "from", "engine", "...", ""
            ))
    
    # Event handlers
    def start_engine(self):
        """Start trading engine"""
        try:
            if self.engine and self.engine.start():
                self.start_ui_updates()
                self.logger.info("Engine started successfully")
            else:
                messagebox.showerror("Error", "Failed to start engine")
        except Exception as e:
            self.logger.error(f"Start engine error: {e}")
            messagebox.showerror("Error", f"Failed to start engine: {e}")
    
    def stop_engine(self):
        """Stop trading engine"""
        try:
            if self.engine:
                self.engine.stop()
                self.stop_ui_updates()
                self.logger.info("Engine stopped")
        except Exception as e:
            self.logger.error(f"Stop engine error: {e}")
    
    def pause_engine(self):
        """Pause trading engine"""
        try:
            if self.engine:
                self.engine.pause()
                self.logger.info("Engine paused")
        except Exception as e:
            self.logger.error(f"Pause engine error: {e}")
    
    def emergency_stop(self):
        """Emergency stop"""
        if messagebox.askyesno("Emergency Stop", 
                              "This will close all positions and stop trading. Continue?"):
            try:
                if self.engine:
                    self.engine.emergency_stop()
                    self.stop_ui_updates()
                    self.logger.critical("EMERGENCY STOP EXECUTED")
            except Exception as e:
                self.logger.error(f"Emergency stop error: {e}")
    
    def apply_parameters(self):
        """Apply parameter changes to engine"""
        try:
            if not self.engine:
                return
             # เพิ่มการตรวจสอบนี้ก่อน apply
            rsi_up = self.rsi_up_var.get()
            rsi_down = self.rsi_down_var.get()
            
            if rsi_down >= rsi_up:
                messagebox.showerror("Parameter Error", 
                                f"RSI Lower ({rsi_down}) must be less than RSI Upper ({rsi_up})")
                return
            
            # Get direction mapping
            direction_map = {"BOTH": 0, "BUY_ONLY": 1, "SELL_ONLY": 2, "STOP": 3}
            exit_speed_map = {"FAST": 0, "MEDIUM": 1, "SLOW": 2}
            
            # Collect parameters
            params = {
                "lot_size": self.lot_size_var.get(),
                "rsi_up": self.rsi_up_var.get(),
                "rsi_down": self.rsi_down_var.get(),
                "trading_direction": direction_map.get(self.direction_var.get(), 0),
                "primary_tf": self.timeframe_var.get(),
                "tp_first": self.tp_first_var.get(),
                "exit_speed": exit_speed_map.get(self.exit_speed_var.get(), 1),
                "dynamic_tp": self.dynamic_tp_var.get(),
                "recovery_price": self.recovery_price_var.get(),
                "martingale": self.martingale_var.get(),
                "max_recovery": self.max_recovery_var.get(),
                "smart_recovery": self.smart_recovery_var.get(),
                "daily_loss_limit": self.daily_loss_var.get(),
                "max_drawdown_percent": self.max_drawdown_var.get(),
                "max_positions": self.max_positions_var.get(),
                "max_spread_alert": self.max_spread_var.get()
            }
            
            # Apply to engine
            self.engine.update_config(params)
            self.logger.info("Parameters applied successfully")
            
        except Exception as e:
            self.logger.error(f"Apply parameters error: {e}")
            messagebox.showerror("Error", f"Failed to apply parameters: {e}")
    
    def on_parameter_changed(self, *args):
        """Handle parameter change (for auto-apply)"""
        # Add delay to avoid too frequent updates
        self.root.after(2000, self.apply_parameters)
    
    def on_preset_selected(self, event=None):
        """Handle preset selection"""
        preset_name = self.preset_var.get()
        if preset_name:
            self.load_preset(preset_name)
    
    def load_preset(self, preset_name):
        """Load trading preset"""
        try:
            if preset_name not in self.preset_manager.PRESETS:
                return
            
            preset = self.preset_manager.PRESETS[preset_name]
            
            # Update UI variables
            self.lot_size_var.set(preset["lot_size"])
            self.rsi_up_var.set(preset["rsi_up"])
            self.rsi_down_var.set(preset["rsi_down"])
            self.tp_first_var.set(preset["tp_first"])
            self.recovery_price_var.set(preset["recovery_price"])
            self.martingale_var.set(preset["martingale"])
            self.max_recovery_var.set(preset["max_recovery"])
            self.timeframe_var.set(preset["primary_tf"])
            
            # Map exit speed
            speed_names = ["FAST", "MEDIUM", "SLOW"]
            self.exit_speed_var.set(speed_names[preset["exit_speed"]])
            
            # Apply changes
            self.apply_parameters()
            
            self.logger.info(f"Loaded preset: {preset_name}")
            
        except Exception as e:
            self.logger.error(f"Load preset error: {e}")
            messagebox.showerror("Error", f"Failed to load preset: {e}")
    
    def save_config(self):
        """Save current configuration"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename and self.engine:
                config = self.engine.config.to_dict()
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info(f"Configuration saved to {filename}")
                messagebox.showinfo("Success", "Configuration saved successfully")
                
        except Exception as e:
            self.logger.error(f"Save config error: {e}")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                if self.engine:
                    self.engine.update_config(config)
                
                # Update UI variables
                self.update_ui_from_config(config)
                
                self.logger.info(f"Configuration loaded from {filename}")
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Load config error: {e}")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def update_ui_from_config(self, config):
        """Update UI variables from config"""
        try:
            if "lot_size" in config:
                self.lot_size_var.set(config["lot_size"])
            if "rsi_up" in config:
                self.rsi_up_var.set(config["rsi_up"])
            # ... update other variables
            
        except Exception as e:
            self.logger.error(f"Update UI from config error: {e}")
    
    # Engine event handlers
    def on_trade_opened(self, trade_info):
        """Handle trade opened event"""
        self.logger.info(f"Trade opened: {trade_info}")
        
        # Update title if enabled
        if self.ui_config.position_in_title:
            positions = len(self.position_data) if hasattr(self, 'position_data') else 0
            self.root.title(f"XAUUSD EA - {positions} Positions")
    
    def on_trade_closed(self, trade_info):
        """Handle trade closed event"""
        self.logger.info(f"Trade closed: {trade_info}")
    
    def on_engine_state_changed(self, new_state):
        """Handle engine state change"""
        self.state_var.set(new_state.value.upper())
    
    def on_engine_error(self, error_msg):
        """Handle engine error"""
        self.logger.error(f"Engine error: {error_msg}")
        
        # Show error in UI if critical
        if "EMERGENCY" in error_msg.upper() or "CRITICAL" in error_msg.upper():
            messagebox.showerror("Critical Error", error_msg)
    
    # Utility methods
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete('1.0', tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                logs = self.log_text.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(logs)
                
                messagebox.showinfo("Success", "Logs exported successfully")
                
        except Exception as e:
            self.logger.error(f"Export logs error: {e}")
            messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def close_selected_position(self):
        """Close selected position"""
        # Implementation would depend on position data structure
        pass
    
    def close_all_positions(self):
        """Close all positions"""
        if messagebox.askyesno("Confirm", "Close all positions?"):
            try:
                if self.engine and self.engine.order_executor:
                    self.engine.order_executor.emergency_close_all()
                    self.logger.info("All positions closed")
            except Exception as e:
                self.logger.error(f"Close all positions error: {e}")
    
    def refresh_positions(self):
        """Refresh positions display"""
        if self.engine:
            self.engine.position_manager.update_positions()
    
    def open_risk_calculator(self):
        """Open risk calculator window"""
        # Placeholder for risk calculator
        messagebox.showinfo("Risk Calculator", "Risk calculator feature coming soon")
    
    def open_performance_report(self):
        """Open performance report window"""
        # Placeholder for performance report
        messagebox.showinfo("Performance Report", "Performance report feature coming soon")
    
    def open_settings(self):
        """Open settings window"""
        # Placeholder for settings
        messagebox.showinfo("Settings", "Settings window coming soon")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
XAUUSD Multi-Timeframe EA - User Guide

QUICK START:
1. Configure parameters in Trading tab
2. Select a preset for quick setup
3. Click START to begin trading

TRADING LOGIC:
- BUY: Fractal Down + RSI > RSI_UP
- SELL: Fractal Up + RSI < RSI_DOWN
- Recovery: Martingale system with smart entry

RISK MANAGEMENT:
- Daily/Weekly/Monthly loss limits
- Maximum drawdown protection
- Position size validation
- Market condition monitoring

For detailed documentation, visit our website.
        """
        messagebox.showinfo("User Guide", help_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
XAUUSD Multi-Timeframe EA
Professional Trading System

Version: 1.0
Copyright: Senior Forex Developer
License: Professional

Features:
• Advanced Fractal + RSI strategy
• Smart recovery system
• Multi-timeframe analysis
• Comprehensive risk management
• Real-time monitoring
• Copy trading ready

Contact: support@seniorforex.com
        """
        messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """Handle window closing"""
        if self.engine and self.engine.state != EngineState.STOPPED:
            if messagebox.askyesno("Confirm Exit", 
                                  "Trading engine is running. Stop and exit?"):
                self.stop_engine()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the UI"""
        try:
            self.logger.info("Starting XAUUSD Trading UI")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"UI error: {e}")
        finally:
            if self.engine:
                self.engine.stop()

# Main execution
if __name__ == "__main__":
    app = XAUUSDTradingUI()
    app.run()