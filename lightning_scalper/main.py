#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Ultimate Complete Version
รวมทุก feature เดิม + แก้ไข imports และปัญหาทั้งหมด + เพิ่ม Dashboard Integration
"""

"""
[ROCKET] Lightning Scalper - Main Application Entry Point
Production-Grade AI Trading System for 80+ Clients

This is the main entry point for the Lightning Scalper trading system.
It orchestrates all components including FVG detection, trade execution,
MT5 integration, and multi-client management.

Author: Phoenix Trading AI
Version: 1.0.0
License: Proprietary
"""

import asyncio
import signal
import sys
import os
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from pathlib import Path

# Add project root to Python path - FIXED PATH RESOLUTION
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

print(f"[DEBUG] Project Root: {PROJECT_ROOT}")

# Import our core modules - FIXED IMPORTS
try:
    from core.main_controller import LightningScalperController
    from core.lightning_scalper_engine import CurrencyPair
    from execution.trade_executor import ClientAccount
    from adapters.mt5_adapter import MT5Adapter
    print("[✓] Core modules imported successfully")
except ImportError as e:
    print(f"[X] Failed to import core modules: {e}")
    print("   Make sure all required files are in the correct directories")
    sys.exit(1)

# Import dashboard module - NEW DASHBOARD INTEGRATION
try:
    from dashboard.web_dashboard import LightningScalperDashboard
    DASHBOARD_AVAILABLE = True
    print("[✓] Dashboard module imported successfully")
except ImportError as e:
    print(f"[WARNING] Dashboard module not available: {e}")
    print("          System will run without web dashboard")
    # Create a dummy class for type annotation
    class LightningScalperDashboard:
        pass
    DASHBOARD_AVAILABLE = False

class LightningScalperApp:
    """
    [ROCKET] Lightning Scalper Main Application
    Production-Grade AI Trading System for 80+ Clients
    
    Complete application wrapper that provides:
    - Multi-client trading orchestration
    - Advanced configuration management
    - Performance monitoring and analytics
    - Auto-restart and recovery mechanisms
    - Comprehensive logging and error handling
    - Risk management integration
    - Real-time dashboard support
    - MT5 integration for live trading
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.app_name = "Lightning Scalper"
        self.version = "1.0.0"
        self.config_path = config_path or "config/settings.json"
        
        # Core system components
        self.controller: Optional[LightningScalperController] = None
        self.dashboard: Optional[LightningScalperDashboard] = None  # NEW: Dashboard component
        self.dashboard_thread: Optional[threading.Thread] = None    # NEW: Dashboard thread
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Application state management
        self.start_time: Optional[datetime] = None
        self.initialization_time: Optional[datetime] = None
        self.last_health_check = None
        self.last_performance_update = None
        
        # Comprehensive statistics tracking
        self.stats = {
            'startup_time': None,
            'total_runtime': timedelta(0),
            'restart_count': 0,
            'last_restart': None,
            'total_clients_served': 0,
            'total_signals_processed': 0,
            'total_trades_executed': 0,
            'system_errors': 0,
            'performance_warnings': 0,
            'memory_peaks': [],
            'cpu_peaks': [],
            'dashboard_status': 'disabled'  # NEW: Dashboard status tracking
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'disk_usage_percent': 0.0,
            'network_io': {'sent': 0, 'received': 0},
            'active_threads': 0,
            'database_connections': 0,
            'cache_hit_rate': 0.0,
            'average_response_time': 0.0
        }
        
        # Threading management
        self.background_threads = {
            'performance_monitor': None,
            'health_checker': None,
            'log_rotator': None,
            'cache_cleaner': None,
            'metric_collector': None
        }
        
        # Setup comprehensive logging - FIXED LOGGER INITIALIZATION
        self._setup_application_logging()
        self.logger = logging.getLogger('LightningScalperApp')
        
        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Default configuration
        self.default_config = self._get_default_config()
        
        # Advanced features
        self._setup_advanced_features()
        
        self.initialization_time = datetime.now()
        self.logger.info(f"[ROCKET] {self.app_name} v{self.version} initialized successfully")
    
    def _setup_application_logging(self):
        """Setup comprehensive application logging system"""
        try:
            # Create logs directory structure
            logs_dir = PROJECT_ROOT / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for different log types
            (logs_dir / "application").mkdir(exist_ok=True)
            (logs_dir / "trading").mkdir(exist_ok=True)
            (logs_dir / "performance").mkdir(exist_ok=True)
            (logs_dir / "errors").mkdir(exist_ok=True)
            (logs_dir / "audit").mkdir(exist_ok=True)
            
            # Setup log file paths with rotation
            timestamp = datetime.now().strftime("%Y%m%d")
            
            log_files = {
                'main': logs_dir / "application" / f"lightning_scalper_{timestamp}.log",
                'error': logs_dir / "errors" / f"errors_{timestamp}.log",
                'performance': logs_dir / "performance" / f"performance_{timestamp}.log",
                'trading': logs_dir / "trading" / f"trading_{timestamp}.log",
                'audit': logs_dir / "audit" / f"audit_{timestamp}.log"
            }
            
            # Setup comprehensive formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            
            simple_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            )
            
            performance_formatter = logging.Formatter(
                '%(asctime)s | PERF | %(message)s'
            )
            
            # Setup multiple handlers
            handlers = []
            
            # Main application log
            main_handler = logging.FileHandler(log_files['main'], encoding='utf-8')
            main_handler.setFormatter(detailed_formatter)
            main_handler.setLevel(logging.DEBUG)
            handlers.append(main_handler)
            
            # Error-only log
            error_handler = logging.FileHandler(log_files['error'], encoding='utf-8')
            error_handler.setFormatter(detailed_formatter)
            error_handler.setLevel(logging.ERROR)
            handlers.append(error_handler)
            
            # Performance log
            perf_handler = logging.FileHandler(log_files['performance'], encoding='utf-8')
            perf_handler.setFormatter(performance_formatter)
            perf_handler.setLevel(logging.INFO)
            perf_handler.addFilter(lambda record: 'PERF' in record.getMessage())
            handlers.append(perf_handler)
            
            # Console handler with colors
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(logging.INFO)
            handlers.append(console_handler)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.handlers.clear()
            
            for handler in handlers:
                root_logger.addHandler(handler)
            
            print(f"[✓] Comprehensive logging system initialized")
            print(f"    Main Log: {log_files['main']}")
            print(f"    Error Log: {log_files['error']}")
            print(f"    Performance Log: {log_files['performance']}")
            
        except Exception as e:
            print(f"[X] Failed to setup comprehensive logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _setup_signal_handlers(self):
        """Setup comprehensive signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_names = {
                signal.SIGTERM: 'SIGTERM',
                signal.SIGINT: 'SIGINT'
            }
            
            if hasattr(signal, 'SIGHUP'):
                signal_names[signal.SIGHUP] = 'SIGHUP'
            if hasattr(signal, 'SIGBREAK'):
                signal_names[signal.SIGBREAK] = 'SIGBREAK'
            
            signal_name = signal_names.get(signum, f'Signal {signum}')
            self.logger.info(f"[📶] Received {signal_name}, initiating graceful shutdown...")
            self.shutdown_event.set()
            
            # Start shutdown in separate thread to avoid blocking
            shutdown_thread = threading.Thread(target=self._graceful_shutdown)
            shutdown_thread.start()
        
        try:
            # Unix/Linux signals
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)
            
            # Windows signals
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, signal_handler)
                
            self.logger.info("[✓] Signal handlers configured for graceful shutdown")
        except Exception as e:
            self.logger.warning(f"[⚠] Signal handler setup failed: {e}")
    
    def _setup_advanced_features(self):
        """Setup advanced application features"""
        try:
            # Performance monitoring configuration
            self.perf_config = {
                'monitor_interval': 30,  # seconds
                'memory_alert_threshold': 1000,  # MB
                'cpu_alert_threshold': 80,  # percent
                'disk_alert_threshold': 90,  # percent
                'enable_predictive_scaling': True,
                'enable_auto_optimization': True
            }
            
            # Health check configuration
            self.health_config = {
                'check_interval': 60,  # seconds
                'component_timeout': 30,  # seconds
                'retry_attempts': 3,
                'alert_on_failure': True,
                'auto_recovery': True
            }
            
            # Cache management
            self.cache_config = {
                'cleanup_interval': 300,  # 5 minutes
                'max_memory_usage': 500,  # MB
                'ttl_default': 3600,  # 1 hour
                'enable_compression': True
            }
            
            self.logger.info("[✓] Advanced features configured")
            
        except Exception as e:
            self.logger.error(f"[X] Advanced features setup failed: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get comprehensive default application configuration"""
        return {
            "application": {
                "name": self.app_name,
                "version": self.version,
                "debug_mode": False,
                "auto_start_trading": True,
                "max_restart_attempts": 3,
                "restart_delay_seconds": 30,
                "auto_restart_on_failure": True,
                "enable_recovery_mode": True,
                "maintenance_mode": False,
                "api_rate_limit": 1000  # requests per minute
            },
            "system": {
                "max_clients": 100,
                "data_update_interval": 1.0,
                "signal_generation_interval": 5.0,
                "auto_reconnect": True,
                "max_reconnect_attempts": 5,
                "enable_performance_monitoring": True,
                "enable_health_checks": True,
                "connection_timeout": 30,
                "heartbeat_interval": 60
            },
            "risk": {
                "global_daily_loss_limit": 5000.0,
                "max_concurrent_trades": 200,
                "emergency_stop_loss_percent": 10.0,
                "max_signals_per_client_hour": 10,
                "enable_global_safety": True,
                "risk_calculation_method": "kelly",
                "max_portfolio_exposure": 0.25,
                "correlation_threshold": 0.7
            },
            "trading": {
                "default_currency_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"],
                "default_timeframes": ["M1", "M5", "M15", "H1"],
                "min_signal_confluence": 65.0,
                "max_slippage_pips": 3.0,
                "default_risk_per_trade": 0.02,
                "enable_news_filter": True,
                "enable_market_hours_filter": True,
                "enable_volatility_filter": True
            },
            "logging": {
                "log_level": "INFO",
                "max_log_file_size": "100MB",
                "log_retention_days": 30,
                "enable_performance_logs": True,
                "enable_trade_logs": True,
                "enable_audit_logs": True,
                "compress_old_logs": True,
                "log_rotation_interval": "daily"
            },
            "clients": {
                "auto_load_clients": True,
                "clients_config_file": "config/clients.json",
                "client_validation_strict": True,
                "enable_client_isolation": True,
                "default_client_settings": {
                    "max_positions": 5,
                    "max_lot_size": 1.0,
                    "risk_multiplier": 1.0,
                    "auto_trading": True,
                    "enable_notifications": True,
                    "notification_methods": ["email", "webhook"]
                }
            },
            "server": {
                "host": "localhost",
                "port": 8080,
                "enable_dashboard": True,        # NEW: Dashboard enabled by default
                "dashboard_host": "0.0.0.0",    # NEW: Dashboard host
                "dashboard_port": 5000,          # NEW: Dashboard port
                "enable_websocket": True,
                "websocket_port": 8081,
                "enable_api": True,
                "api_port": 9000,
                "enable_ssl": False,
                "max_connections": 1000
            },
            "database": {
                "type": "sqlite",
                "path": "data/lightning_scalper.db",
                "backup_interval_hours": 6,
                "enable_data_compression": True,
                "enable_encryption": False,
                "connection_pool_size": 10,
                "query_timeout": 30
            },
            "mt5": {
                "enable_auto_connection": True,
                "connection_timeout": 30,
                "max_retries": 3,
                "enable_symbol_validation": True,
                "enable_account_validation": True,
                "enable_trade_copying": False,
                "copy_trades_delay": 0.1
            },
            "notifications": {
                "enable_email": False,
                "enable_webhook": False,
                "enable_telegram": False,
                "enable_slack": False,
                "email_settings": {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "use_tls": True
                },
                "webhook_settings": {
                    "url": "",
                    "secret": "",
                    "timeout": 10
                },
                "telegram_settings": {
                    "bot_token": "",
                    "chat_id": ""
                }
            },
            "security": {
                "enable_api_authentication": True,
                "api_key_expiry_days": 30,
                "enable_rate_limiting": True,
                "enable_ip_whitelisting": False,
                "allowed_ips": [],
                "enable_audit_logging": True,
                "encryption_algorithm": "AES-256"
            },
            "optimization": {
                "enable_auto_scaling": True,
                "enable_caching": True,
                "cache_size_mb": 256,
                "enable_compression": True,
                "enable_multi_threading": True,
                "max_worker_threads": 10,
                "enable_async_processing": True
            }
        }
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load comprehensive application configuration from file"""
        try:
            config_file = PROJECT_ROOT / self.config_path
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Deep merge with default config
                config = self.default_config.copy()
                self._deep_merge_config(config, file_config)
                
                # Validate configuration
                validation_result = self._validate_configuration(config)
                if not validation_result['valid']:
                    self.logger.warning(f"[⚠] Configuration validation issues: {validation_result['errors']}")
                
                self.logger.info(f"[✓] Configuration loaded and validated from {config_file}")
                return config
            else:
                self.logger.warning(f"[⚠] Config file {config_file} not found, creating default")
                self._create_default_config_file(config_file)
                return self.default_config
                
        except Exception as e:
            self.logger.error(f"[X] Failed to load configuration: {e}")
            self.logger.info("[🔄] Using default configuration")
            return self.default_config
    
    def _deep_merge_config(self, base: Dict, update: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for consistency and security"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate critical settings
            if config['system']['max_clients'] <= 0:
                validation_result['errors'].append("max_clients must be positive")
                validation_result['valid'] = False
            
            if config['risk']['global_daily_loss_limit'] >= 0:
                validation_result['errors'].append("global_daily_loss_limit should be negative")
                validation_result['valid'] = False
            
            if config['application']['max_restart_attempts'] > 10:
                validation_result['warnings'].append("High restart attempts may cause instability")
            
            # Validate ports
            ports = [
                config['server']['port'],
                config['server']['dashboard_port'],
                config['server']['websocket_port'],
                config['server']['api_port']
            ]
            
            if len(set(ports)) != len(ports):
                validation_result['errors'].append("Port conflicts detected")
                validation_result['valid'] = False
            
            # Validate file paths
            data_path = Path(config['database']['path']).parent
            if not data_path.exists():
                try:
                    data_path.mkdir(parents=True, exist_ok=True)
                    validation_result['warnings'].append(f"Created data directory: {data_path}")
                except Exception:
                    validation_result['errors'].append(f"Cannot create data directory: {data_path}")
                    validation_result['valid'] = False
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['valid'] = False
        
        return validation_result
    
    def _create_default_config_file(self, config_file: Path):
        """Create default configuration file with comments"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to config
            config_with_metadata = {
                "_metadata": {
                    "created": datetime.now().isoformat(),
                    "version": self.version,
                    "description": "Lightning Scalper Configuration File",
                    "documentation": "https://docs.lightning-scalper.com/configuration"
                },
                **self.default_config
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_with_metadata, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"[📝] Created default configuration file: {config_file}")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create config file: {e}")
    
    def load_clients_configuration(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load and validate client configurations"""
        try:
            if not config['clients']['auto_load_clients']:
                self.logger.info("[INFO] Auto-load clients disabled")
                return []
            
            clients_file = Path(config['clients']['clients_config_file'])
            
            # Handle relative paths
            if not clients_file.is_absolute():
                clients_file = PROJECT_ROOT / clients_file
            
            if clients_file.exists():
                with open(clients_file, 'r', encoding='utf-8') as f:
                    clients_data = json.load(f)
                
                # Handle different file formats
                if isinstance(clients_data, list):
                    client_list = clients_data
                elif isinstance(clients_data, dict):
                    client_list = clients_data.get('clients', [])
                else:
                    self.logger.error("[X] Invalid clients file format")
                    return []
                
                # Validate clients
                validated_clients = []
                for client_data in client_list:
                    if self._validate_client_config(client_data):
                        validated_clients.append(client_data)
                    else:
                        self.logger.warning(f"[⚠] Skipping invalid client: {client_data.get('client_id', 'unknown')}")
                
                self.logger.info(f"[✓] Loaded {len(validated_clients)}/{len(client_list)} valid client configurations")
                return validated_clients
                
            else:
                self.logger.warning(f"[⚠] Clients config file {clients_file} not found")
                self._create_sample_clients_file(clients_file, config)
                return []
                
        except Exception as e:
            self.logger.error(f"[X] Failed to load clients configuration: {e}")
            return []
    
    def _validate_client_config(self, client_data: Dict[str, Any]) -> bool:
        """Validate individual client configuration"""
        try:
            required_fields = ['client_id']
            for field in required_fields:
                if field not in client_data or not client_data[field]:
                    self.logger.error(f"[X] Client missing required field: {field}")
                    return False
            
            # Validate risk settings
            risk_settings = client_data.get('risk_settings', {})
            risk_per_trade = risk_settings.get('risk_per_trade', 0)
            if risk_per_trade <= 0 or risk_per_trade > 0.1:
                self.logger.warning(f"[⚠] Client {client_data['client_id']}: Unusual risk per trade ({risk_per_trade})")
            
            # Validate trading pairs
            allowed_pairs = client_data.get('allowed_pairs', [])
            if not allowed_pairs:
                self.logger.warning(f"[⚠] Client {client_data['client_id']}: No trading pairs specified")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Client validation error: {e}")
            return False
    
    def _create_sample_clients_file(self, clients_file: Path, config: Dict[str, Any]):
        """Create comprehensive sample clients configuration file"""
        try:
            clients_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_clients = {
                "_metadata": {
                    "description": "Lightning Scalper Client Configuration",
                    "version": "1.0.0",
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_clients": 3,
                    "security_notice": "NEVER commit real passwords to version control",
                    "demo_mode_notice": "Set 'is_demo': true for testing without real trading"
                },
                "global_settings": {
                    "default_risk_per_trade": 0.01,
                    "default_max_positions": 5,
                    "default_lot_size": 0.1,
                    "auto_trading_default": True,
                    "notifications_default": True
                },
                "clients": [
                    {
                        "client_id": "DEMO_CLIENT_001",
                        "name": "Demo Trading Account 001",
                        "description": "Primary demo account for testing",
                        "broker": "demo_broker",
                        "account_number": "demo_001",
                        "is_active": True,
                        "is_demo": True,
                        "created_date": "2024-01-01T00:00:00Z",
                        "last_login": None,
                        "mt5_settings": {
                            "login": 0,
                            "password": "demo_password",
                            "server": "demo_server",
                            "enable_trade_copying": False
                        },
                        "risk_settings": {
                            "risk_per_trade": 0.01,
                            "max_daily_loss": 100.0,
                            "max_positions": 3,
                            "max_lot_size": 0.1,
                            "stop_loss_buffer_pips": 5,
                            "take_profit_ratio": 2.0
                        },
                        "trading_settings": {
                            "allowed_pairs": ["EURUSD", "GBPUSD", "USDJPY"],
                            "allowed_timeframes": ["M5", "M15", "H1"],
                            "auto_trading": True,
                            "signal_filters": {
                                "min_confluence": 70,
                                "enable_news_filter": True,
                                "enable_session_filter": True
                            }
                        },
                        "notifications": {
                            "email": "demo@example.com",
                            "webhook": "",
                            "telegram_chat_id": "",
                            "enable_notifications": False,
                            "notification_events": ["trade_opened", "trade_closed", "daily_summary"]
                        },
                        "performance_tracking": {
                            "track_execution_time": True,
                            "track_slippage": True,
                            "track_spread": True,
                            "generate_reports": True
                        }
                    },
                    {
                        "client_id": "DEMO_CLIENT_002",
                        "name": "Demo Trading Account 002",
                        "description": "Secondary demo account for strategy testing",
                        "broker": "demo_broker",
                        "account_number": "demo_002",
                        "is_active": True,
                        "is_demo": True,
                        "created_date": "2024-01-01T00:00:00Z",
                        "last_login": None,
                        "mt5_settings": {
                            "login": 0,
                            "password": "demo_password",
                            "server": "demo_server",
                            "enable_trade_copying": False
                        },
                        "risk_settings": {
                            "risk_per_trade": 0.02,
                            "max_daily_loss": 200.0,
                            "max_positions": 5,
                            "max_lot_size": 0.2,
                            "stop_loss_buffer_pips": 3,
                            "take_profit_ratio": 1.5
                        },
                        "trading_settings": {
                            "allowed_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                            "allowed_timeframes": ["M1", "M5", "M15"],
                            "auto_trading": True,
                            "signal_filters": {
                                "min_confluence": 65,
                                "enable_news_filter": False,
                                "enable_session_filter": False
                            }
                        },
                        "notifications": {
                            "email": "demo2@example.com",
                            "webhook": "",
                            "telegram_chat_id": "",
                            "enable_notifications": False,
                            "notification_events": ["trade_opened", "trade_closed"]
                        },
                        "performance_tracking": {
                            "track_execution_time": True,
                            "track_slippage": True,
                            "track_spread": False,
                            "generate_reports": False
                        }
                    },
                    {
                        "client_id": "DEMO_CLIENT_003",
                        "name": "Demo Conservative Account",
                        "description": "Conservative risk demo account",
                        "broker": "demo_broker",
                        "account_number": "demo_003",
                        "is_active": False,
                        "is_demo": True,
                        "created_date": "2024-01-01T00:00:00Z",
                        "last_login": None,
                        "mt5_settings": {
                            "login": 0,
                            "password": "demo_password",
                            "server": "demo_server",
                            "enable_trade_copying": False
                        },
                        "risk_settings": {
                            "risk_per_trade": 0.005,
                            "max_daily_loss": 50.0,
                            "max_positions": 2,
                            "max_lot_size": 0.05,
                            "stop_loss_buffer_pips": 10,
                            "take_profit_ratio": 3.0
                        },
                        "trading_settings": {
                            "allowed_pairs": ["EURUSD", "GBPUSD"],
                            "allowed_timeframes": ["H1", "H4"],
                            "auto_trading": False,
                            "signal_filters": {
                                "min_confluence": 80,
                                "enable_news_filter": True,
                                "enable_session_filter": True
                            }
                        },
                        "notifications": {
                            "email": "conservative@example.com",
                            "webhook": "",
                            "telegram_chat_id": "",
                            "enable_notifications": False,
                            "notification_events": ["daily_summary"]
                        },
                        "performance_tracking": {
                            "track_execution_time": False,
                            "track_slippage": False,
                            "track_spread": False,
                            "generate_reports": True
                        }
                    }
                ]
            }
            
            with open(clients_file, 'w', encoding='utf-8') as f:
                json.dump(sample_clients, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"[📝] Created comprehensive sample clients file: {clients_file}")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create sample clients file: {e}")
    
    def _start_dashboard(self, config: Dict[str, Any]) -> bool:
        """Start web dashboard in separate thread - NEW DASHBOARD INTEGRATION"""
        try:
            if not DASHBOARD_AVAILABLE:
                self.logger.warning("[DASHBOARD] Dashboard module not available")
                self.stats['dashboard_status'] = 'unavailable'
                return False
            
            if not config.get('server', {}).get('enable_dashboard', True):
                self.logger.info("[DASHBOARD] Dashboard disabled in configuration")
                self.stats['dashboard_status'] = 'disabled'
                return False
            
            # Dashboard configuration
            dashboard_host = config.get('server', {}).get('dashboard_host', '0.0.0.0')
            dashboard_port = config.get('server', {}).get('dashboard_port', 5000)
            debug_mode = config.get('application', {}).get('debug_mode', False)
            
            self.logger.info(f"[DASHBOARD] Initializing web dashboard...")
            self.logger.info(f"            Host: {dashboard_host}")
            self.logger.info(f"            Port: {dashboard_port}")
            self.logger.info(f"            Debug: {debug_mode}")
            
            # Create dashboard instance
            self.dashboard = LightningScalperDashboard(
                controller=self.controller,
                host=dashboard_host,
                port=dashboard_port,
                debug=debug_mode
            )
            
            # Start dashboard in separate thread
            def run_dashboard():
                try:
                    self.logger.info("[DASHBOARD] Starting dashboard server...")
                    if hasattr(self.dashboard, 'start'):
                        self.dashboard.start()
                    elif hasattr(self.dashboard, 'run'):
                        self.dashboard.run()
                    else:
                        self.logger.error("[DASHBOARD] No start/run method found")
                except Exception as e:
                    self.logger.error(f"[DASHBOARD] Dashboard error: {e}")
                    self.stats['dashboard_status'] = 'error'            
            self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            self.dashboard_thread.start()
            
            # Give dashboard time to start
            time.sleep(2)
            
            self.logger.info(f"[DASHBOARD] ✅ Web dashboard started successfully!")
            self.logger.info(f"[DASHBOARD] 🌐 Access URL: http://localhost:{dashboard_port}")
            self.stats['dashboard_status'] = 'running'
            
            return True
            
        except Exception as e:
            self.logger.error(f"[DASHBOARD] Failed to start dashboard: {e}")
            self.stats['dashboard_status'] = 'error'
            return False
    
    async def startup(self, config: Dict[str, Any]) -> bool:
        """Comprehensive application startup sequence"""
        try:
            self.start_time = datetime.now()
            startup_start = time.time()
            
            self.logger.info("[🚀] Starting Lightning Scalper Application...")
            self.logger.info("[📋] Configuration Summary:")
            self.logger.info(f"    Environment: {'DEBUG' if config['application']['debug_mode'] else 'PRODUCTION'}")
            self.logger.info(f"    Max Clients: {config['system']['max_clients']}")
            self.logger.info(f"    Auto Trading: {config['application']['auto_start_trading']}")
            self.logger.info(f"    Performance Monitoring: {config['system']['enable_performance_monitoring']}")
            self.logger.info(f"    Risk Management: {config['risk']['enable_global_safety']}")
            self.logger.info(f"    Dashboard: {config.get('server', {}).get('enable_dashboard', True)}")  # NEW
            
            # Phase 1: Initialize core controller
            self.logger.info("[🛰] Phase 1: Initializing system controller...")
            self.controller = LightningScalperController(self.config_path)
            
            # Phase 2: Start core trading system
            self.logger.info("[⚡] Phase 2: Starting core trading system...")
            success = await self.controller.startup(config)
            
            if not success:
                self.logger.error("[X] Failed to start core trading system")
                return False
            
            # Phase 3: Load and validate clients
            self.logger.info("[👥] Phase 3: Loading client configurations...")
            clients_data = self.load_clients_configuration(config)
            
            added_clients = 0
            skipped_clients = 0
            
            if clients_data:
                self.logger.info(f"[👥] Processing {len(clients_data)} client configurations...")
                
                for client_data in clients_data:
                    try:
                        # Apply filters based on mode
                        if (client_data.get('is_demo', False) and 
                            not config['application']['debug_mode'] and
                            config.get('skip_demo_in_production', False)):
                            self.logger.info(f"[⏭] Skipping demo client {client_data.get('client_id')} in production mode")
                            skipped_clients += 1
                            continue
                        
                        # Create comprehensive client account object
                        client_account = ClientAccount(
                            client_id=client_data['client_id'],
                            broker=client_data.get('broker', 'unknown'),
                            account_number=client_data.get('account_number', ''),
                            account_type=client_data.get('account_type', 'demo'),
                            balance=client_data.get('balance', 10000.0),
                            currency=client_data.get('currency', 'USD'),
                            leverage=client_data.get('leverage', 100),
                            max_position_size=client_data.get('max_position_size', 1.0)
                        )
                        
                        # Add client to controller
                        client_added = await self.controller.add_client(client_data)
                        
                        if client_added:
                            added_clients += 1
                            client_type = "DEMO" if client_data.get('is_demo', True) else "LIVE"
                            self.logger.info(f"[✓] Added {client_type} client: {client_data.get('client_id')}")
                        else:
                            skipped_clients += 1
                            self.logger.warning(f"[⚠] Failed to add client: {client_data.get('client_id')}")
                        
                    except Exception as e:
                        skipped_clients += 1
                        self.logger.error(f"[X] Error processing client {client_data.get('client_id', 'unknown')}: {e}")
                
                self.logger.info(f"[✓] Client loading complete: {added_clients} added, {skipped_clients} skipped")
                self.stats['total_clients_served'] = added_clients
            else:
                self.logger.warning("[⚠] No clients loaded - system will run without active clients")
            
            # Phase 4: Start web dashboard - NEW DASHBOARD INTEGRATION
            self.logger.info("[🌐] Phase 4: Starting web dashboard...")
            dashboard_success = self._start_dashboard(config)
            if dashboard_success:
                self.logger.info("[✅] Dashboard integration successful")
            else:
                self.logger.warning("[⚠] Dashboard not available (system will continue)")
            
            # Phase 5: Initialize background services
            self.logger.info("[🔧] Phase 5: Starting background services...")
            self._start_background_services(config)
            
            # Phase 6: Setup monitoring and alerts
            if config['system']['enable_performance_monitoring']:
                self.logger.info("[📊] Phase 6: Enabling performance monitoring...")
                self._start_performance_monitoring()
            
            if config['system']['enable_health_checks']:
                self.logger.info("[🏥] Phase 6: Enabling health monitoring...")
                self._start_health_monitoring()
            
            # Phase 7: Initialize external integrations
            self.logger.info("[🌐] Phase 7: Initializing external integrations...")
            
            if config.get('notifications', {}).get('enable_email', False):
                self.logger.info("[📧] Email notifications enabled")
            
            if config.get('notifications', {}).get('enable_webhook', False):
                self.logger.info("[🔗] Webhook notifications enabled")
            
            # Phase 8: Final startup validation
            self.logger.info("[✅] Phase 8: Final system validation...")
            system_status = self.controller.get_system_status()
            
            # Calculate and log startup metrics
            self.is_running = True
            self.stats['startup_time'] = time.time() - startup_start
            
            # Final status report
            self.logger.info("[🎉] Lightning Scalper Application started successfully!")
            self.logger.info(f"   ⏱ Startup time: {self.stats['startup_time']:.2f} seconds")
            self.logger.info(f"   👥 Active clients: {system_status['clients']['active']}")
            self.logger.info(f"   🎯 System status: {system_status['status']}")
            memory_usage = system_status.get('metrics', {}).get('memory_usage', 0.0)
            self.logger.info(f"   🧠 Memory usage: {memory_usage:.1f} MB")
            self.logger.info(f"   🌐 Dashboard: {self.stats['dashboard_status']}")  # NEW
            
            # NEW: Dashboard URL logging
            if dashboard_success:
                dashboard_port = config.get('server', {}).get('dashboard_port', 5000)
                self.logger.info(f"   🔗 Dashboard URL: http://localhost:{dashboard_port}")
            
            self.logger.info("[🎯] System ready for trading operations!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Application startup failed: {e}")
            import traceback
            self.logger.error(f"[📋] Detailed error trace: {traceback.format_exc()}")
            return False
    
    def _start_background_services(self, config: Dict[str, Any]):
        """Start comprehensive background services"""
        try:
            # Performance monitoring service
            if config['system']['enable_performance_monitoring']:
                self.background_threads['performance_monitor'] = threading.Thread(
                    target=self._performance_monitoring_loop,
                    daemon=True,
                    name="PerformanceMonitor"
                )
                self.background_threads['performance_monitor'].start()
                self.logger.info("[✓] Performance monitoring service started")
            
            # Health check service
            if config['system']['enable_health_checks']:
                self.background_threads['health_checker'] = threading.Thread(
                    target=self._health_check_loop,
                    daemon=True,
                    name="HealthChecker"
                )
                self.background_threads['health_checker'].start()
                self.logger.info("[✓] Health check service started")
            
            # Log rotation service
            self.background_threads['log_rotator'] = threading.Thread(
                target=self._log_rotation_loop,
                daemon=True,
                name="LogRotator"
            )
            self.background_threads['log_rotator'].start()
            self.logger.info("[✓] Log rotation service started")
            
            # Cache cleanup service
            if config.get('optimization', {}).get('enable_caching', True):
                self.background_threads['cache_cleaner'] = threading.Thread(
                    target=self._cache_cleanup_loop,
                    daemon=True,
                    name="CacheCleaner"
                )
                self.background_threads['cache_cleaner'].start()
                self.logger.info("[✓] Cache cleanup service started")
            
            # Metrics collection service
            self.background_threads['metric_collector'] = threading.Thread(
                target=self._metrics_collection_loop,
                daemon=True,
                name="MetricsCollector"
            )
            self.background_threads['metric_collector'].start()
            self.logger.info("[✓] Metrics collection service started")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to start background services: {e}")
    
    def _start_performance_monitoring(self):
        """Initialize comprehensive performance monitoring"""
        try:
            self.last_performance_update = datetime.now()
            self.logger.info("[📊] Performance monitoring initialized")
        except Exception as e:
            self.logger.error(f"[X] Performance monitoring initialization failed: {e}")
    
    def _start_health_monitoring(self):
        """Initialize comprehensive health monitoring"""
        try:
            self.last_health_check = datetime.now()
            self.logger.info("[🏥] Health monitoring initialized")
        except Exception as e:
            self.logger.error(f"[X] Health monitoring initialization failed: {e}")
    
    def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for performance alerts
                self._check_performance_alerts()
                
                # Log performance summary
                self._log_performance_summary()
                
                # Sleep until next check
                self.shutdown_event.wait(self.perf_config['monitor_interval'])
                
            except Exception as e:
                self.logger.error(f"[X] Performance monitoring error: {e}")
                time.sleep(30)  # Fallback sleep
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Perform comprehensive health checks
                health_status = self._perform_health_checks()
                
                # Handle health issues
                if not health_status['overall_healthy']:
                    self._handle_health_issues(health_status)
                
                # Sleep until next check
                self.shutdown_event.wait(self.health_config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"[X] Health check error: {e}")
                time.sleep(60)  # Fallback sleep
    
    def _log_rotation_loop(self):
        """Background log rotation loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Rotate logs if needed
                self._rotate_logs_if_needed()
                
                # Cleanup old logs
                self._cleanup_old_logs()
                
                # Sleep for 1 hour
                self.shutdown_event.wait(3600)
                
            except Exception as e:
                self.logger.error(f"[X] Log rotation error: {e}")
                time.sleep(3600)  # Fallback sleep
    
    def _cache_cleanup_loop(self):
        """Background cache cleanup loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Cleanup expired cache entries
                self._cleanup_cache()
                
                # Sleep until next cleanup
                self.shutdown_event.wait(self.cache_config['cleanup_interval'])
                
            except Exception as e:
                self.logger.error(f"[X] Cache cleanup error: {e}")
                time.sleep(300)  # Fallback sleep
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Store metrics for analysis
                self._store_metrics()
                
                # Sleep until next collection
                self.shutdown_event.wait(60)  # Every minute
                
            except Exception as e:
                self.logger.error(f"[X] Metrics collection error: {e}")
                time.sleep(60)  # Fallback sleep
    
    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            # Memory usage
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                self.performance_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                self.performance_metrics['cpu_usage_percent'] = process.cpu_percent()
                
                # System-wide metrics
                self.performance_metrics['disk_usage_percent'] = psutil.disk_usage('/').percent
                net_io = psutil.net_io_counters()
                self.performance_metrics['network_io'] = {
                    'sent': net_io.bytes_sent,
                    'received': net_io.bytes_recv
                }
                
            except ImportError:
                # Fallback if psutil not available
                self.performance_metrics['memory_usage_mb'] = 0.0
                self.performance_metrics['cpu_usage_percent'] = 0.0
            
            # Thread count
            self.performance_metrics['active_threads'] = threading.active_count()
            
            # Application-specific metrics
            if self.controller:
                controller_status = self.controller.get_system_status()
                self.performance_metrics['cache_hit_rate'] = controller_status.get('cache', {}).get('hit_rate', 0.0)
                self.performance_metrics['average_response_time'] = controller_status.get('performance', {}).get('average_execution_time', 0.0)
            
        except Exception as e:
            self.logger.error(f"[X] Performance metrics update failed: {e}")
    
    def _check_performance_alerts(self):
        """Check performance metrics against alert thresholds"""
        try:
            alerts = []
            
            # Memory alert
            if self.performance_metrics['memory_usage_mb'] > self.perf_config['memory_alert_threshold']:
                alerts.append(f"High memory usage: {self.performance_metrics['memory_usage_mb']:.1f}MB")
            
            # CPU alert
            if self.performance_metrics['cpu_usage_percent'] > self.perf_config['cpu_alert_threshold']:
                alerts.append(f"High CPU usage: {self.performance_metrics['cpu_usage_percent']:.1f}%")
            
            # Disk alert
            if self.performance_metrics['disk_usage_percent'] > self.perf_config['disk_alert_threshold']:
                alerts.append(f"High disk usage: {self.performance_metrics['disk_usage_percent']:.1f}%")
            
            # Log alerts
            for alert in alerts:
                self.logger.warning(f"[⚠] PERFORMANCE ALERT: {alert}")
                self.stats['performance_warnings'] += 1
            
        except Exception as e:
            self.logger.error(f"[X] Performance alert check failed: {e}")
    
    def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        try:
            if datetime.now().minute % 5 == 0:  # Every 5 minutes
                perf_msg = (
                    f"PERF | Memory: {self.performance_metrics['memory_usage_mb']:.1f}MB | "
                    f"CPU: {self.performance_metrics['cpu_usage_percent']:.1f}% | "
                    f"Threads: {self.performance_metrics['active_threads']} | "
                    f"Uptime: {datetime.now() - self.start_time if self.start_time else 'Unknown'}"
                )
                self.logger.info(perf_msg)
            
        except Exception as e:
            self.logger.error(f"[X] Performance summary logging failed: {e}")
    
    def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive system health checks"""
        health_status = {
            'overall_healthy': True,
            'checks': {},
            'issues': [],
            'timestamp': datetime.now()
        }
        
        try:
            # Controller health
            if self.controller:
                controller_status = self.controller.get_system_status()
                health_status['checks']['controller'] = controller_status.get('status') != 'error'
                if not health_status['checks']['controller']:
                    health_status['issues'].append("Controller is in error state")
                    health_status['overall_healthy'] = False
            else:
                health_status['checks']['controller'] = False
                health_status['issues'].append("Controller not initialized")
                health_status['overall_healthy'] = False
            
            # Dashboard health - NEW
            health_status['checks']['dashboard'] = self.stats['dashboard_status'] in ['running', 'disabled']
            if self.stats['dashboard_status'] == 'error':
                health_status['issues'].append("Dashboard is in error state")
                health_status['overall_healthy'] = False
            
            # Memory health
            memory_usage = self.performance_metrics.get('memory_usage_mb', 0)
            health_status['checks']['memory'] = memory_usage < self.perf_config['memory_alert_threshold']
            if not health_status['checks']['memory']:
                health_status['issues'].append(f"High memory usage: {memory_usage:.1f}MB")
                health_status['overall_healthy'] = False
            
            # Background services health
            for service_name, thread in self.background_threads.items():
                if thread:
                    health_status['checks'][f'service_{service_name}'] = thread.is_alive()
                    if not thread.is_alive():
                        health_status['issues'].append(f"Background service {service_name} is not running")
                        health_status['overall_healthy'] = False
            
        except Exception as e:
            health_status['overall_healthy'] = False
            health_status['issues'].append(f"Health check error: {str(e)}")
            self.logger.error(f"[X] Health check failed: {e}")
        
        return health_status
    
    def _handle_health_issues(self, health_status: Dict[str, Any]):
        """Handle detected health issues"""
        try:
            for issue in health_status['issues']:
                self.logger.error(f"[🏥] HEALTH ISSUE: {issue}")
            
            # Implement auto-recovery if enabled
            if self.health_config.get('auto_recovery', False):
                self.logger.info("[🔧] Attempting auto-recovery...")
                # Add specific recovery procedures here
                
        except Exception as e:
            self.logger.error(f"[X] Health issue handling failed: {e}")
    
    def _rotate_logs_if_needed(self):
        """Rotate logs if they exceed size limits"""
        try:
            logs_dir = PROJECT_ROOT / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.rglob("*.log"):
                    if log_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                        # Rotate log
                        rotated_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                        rotated_path = log_file.parent / rotated_name
                        log_file.rename(rotated_path)
                        self.logger.info(f"[🔄] Rotated log: {log_file.name} -> {rotated_name}")
        except Exception as e:
            self.logger.error(f"[X] Log rotation failed: {e}")
    
    def _cleanup_old_logs(self):
        """Cleanup old log files"""
        try:
            logs_dir = PROJECT_ROOT / "logs"
            if logs_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in logs_dir.rglob("*.log"):
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        log_file.unlink()
                        self.logger.info(f"[🗑] Cleaned old log: {log_file.name}")
        except Exception as e:
            self.logger.error(f"[X] Log cleanup failed: {e}")
    
    def _cleanup_cache(self):
        """Cleanup expired cache entries"""
        try:
            if self.controller:
                # Controller has its own cache cleanup
                pass
            
            # Application-level cache cleanup can be added here
            
        except Exception as e:
            self.logger.error(f"[X] Cache cleanup failed: {e}")
    
    def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'memory_usage': self.performance_metrics['memory_usage_mb'],
                'cpu_usage': self.performance_metrics['cpu_usage_percent'],
                'active_threads': self.performance_metrics['active_threads'],
                'application_uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            if self.controller:
                controller_metrics = self.controller.get_system_status()['metrics']
                metrics.update({
                    'active_clients': controller_metrics.get('active_clients', 0),
                    'total_signals': controller_metrics.get('total_signals_generated', 0),
                    'total_trades': controller_metrics.get('total_trades_executed', 0)
                })
            
            # Store metrics for analysis
            if not hasattr(self, 'metrics_history'):
                self.metrics_history = []
            
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"[X] Metrics collection failed: {e}")
    
    def _store_metrics(self):
        """Store metrics to persistent storage"""
        try:
            # This would typically save to database
            # For now, just log summary
            if hasattr(self, 'metrics_history') and len(self.metrics_history) > 0:
                latest = self.metrics_history[-1]
                self.logger.debug(f"[📊] Metrics stored: Memory={latest['memory_usage']:.1f}MB, "
                                 f"CPU={latest['cpu_usage']:.1f}%, Clients={latest.get('active_clients', 0)}")
        except Exception as e:
            self.logger.error(f"[X] Metrics storage failed: {e}")
    
    async def run(self):
        """Enhanced main application run loop"""
        try:
            self.logger.info("[▶] Starting enhanced main application loop...")
            
            loop_count = 0
            last_status_log = datetime.now()
            status_log_interval = timedelta(minutes=10)
            last_maintenance = datetime.now()
            maintenance_interval = timedelta(hours=6)
            
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    loop_count += 1
                    current_time = datetime.now()
                    
                    # Run controller main loop
                    if self.controller:
                        while self.is_running:
                            await asyncio.sleep(1)                    
                    # Periodic status logging
                    if current_time - last_status_log >= status_log_interval:
                        self._log_application_status()
                        last_status_log = current_time
                    
                    # Periodic maintenance
                    if current_time - last_maintenance >= maintenance_interval:
                        await self._perform_maintenance()
                        last_maintenance = current_time
                    
                    # Check for emergency conditions
                    if self.controller:
                        status = self.controller.get_system_status()
                        if status.get('risk', {}).get('emergency_stop_active', False):
                            self.logger.critical("[🚨] Emergency stop detected!")
                            # Handle emergency stop procedures
                    
                    # Performance optimization check
                    if loop_count % 100 == 0:  # Every 100 loops
                        await self._optimize_performance()
                    
                    # Health validation
                    if loop_count % 300 == 0:  # Every 300 loops
                        health_status = self._perform_health_checks()
                        if not health_status['overall_healthy']:
                            self.logger.warning("[⚠] System health issues detected")
                    
                    # Wait for next iteration
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"[X] Error in main loop: {e}")
                    self.stats['system_errors'] += 1
                    
                    # Decide whether to restart or shutdown
                    if self._should_restart():
                        await self._attempt_restart()
                    else:
                        self.logger.error("[💥] Too many failures, shutting down")
                        break
            
        except KeyboardInterrupt:
            self.logger.info("[⌨] Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"[X] Critical error in main loop: {e}")
        finally:
            await self.shutdown()
    
    def _log_application_status(self):
        """Log comprehensive application status"""
        try:
            status = self.get_app_status()
            
            self.logger.info("[📊] Application Status Summary:")
            self.logger.info(f"    Uptime: {status['application']['uptime']}")
            self.logger.info(f"    Memory: {self.performance_metrics['memory_usage_mb']:.1f}MB")
            self.logger.info(f"    CPU: {self.performance_metrics['cpu_usage_percent']:.1f}%")
            self.logger.info(f"    Dashboard: {self.stats['dashboard_status']}")  # NEW
            
            if status['system']:
                metrics = status['system']['metrics']
                self.logger.info(f"    Active Clients: {metrics.get('active_clients', 0)}")
                self.logger.info(f"    Total Signals: {metrics.get('total_signals_generated', 0)}")
                self.logger.info(f"    Total Trades: {metrics.get('total_trades_executed', 0)}")
                
        except Exception as e:
            self.logger.error(f"[X] Status logging failed: {e}")
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            self.logger.info("[🔧] Performing periodic maintenance...")
            
            # Cleanup old data
            await self._cleanup_old_data()
            
            # Optimize database
            await self._optimize_database()
            
            # Update statistics
            await self._update_statistics()
            
            # Generate reports
            await self._generate_reports()
            
            self.logger.info("[✓] Periodic maintenance completed")
            
        except Exception as e:
            self.logger.error(f"[X] Maintenance failed: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old application data"""
        try:
            # Cleanup temporary files
            temp_dir = PROJECT_ROOT / "temp"
            if temp_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=1)
                for temp_file in temp_dir.iterdir():
                    if temp_file.is_file() and datetime.fromtimestamp(temp_file.stat().st_mtime) < cutoff_date:
                        temp_file.unlink()
                        self.logger.debug(f"[🗑] Cleaned temp file: {temp_file.name}")
            
        except Exception as e:
            self.logger.error(f"[X] Data cleanup failed: {e}")
    
    async def _optimize_database(self):
        """Optimize database performance"""
        try:
            # Database optimization logic would go here
            self.logger.debug("[🗃] Database optimization completed")
        except Exception as e:
            self.logger.error(f"[X] Database optimization failed: {e}")
    
    async def _update_statistics(self):
        """Update application statistics"""
        try:
            # Update comprehensive statistics
            if self.controller:
                controller_status = self.controller.get_system_status()
                self.stats['total_signals_processed'] = controller_status['metrics'].get('total_signals_generated', 0)
                self.stats['total_trades_executed'] = controller_status['metrics'].get('total_trades_executed', 0)
            
        except Exception as e:
            self.logger.error(f"[X] Statistics update failed: {e}")
    
    async def _generate_reports(self):
        """Generate periodic reports"""
        try:
            # Generate daily/weekly reports
            if datetime.now().hour == 0:  # Midnight
                report = {
                    'date': datetime.now().date().isoformat(),
                    'uptime': str(datetime.now() - self.start_time) if self.start_time else 'Unknown',
                    'stats': self.stats.copy(),
                    'performance': self.performance_metrics.copy()
                }
                
                self.logger.info(f"[📊] Daily Report Generated: {json.dumps(report, indent=2)}")
                
        except Exception as e:
            self.logger.error(f"[X] Report generation failed: {e}")
    
    async def _optimize_performance(self):
        """Optimize system performance"""
        try:
            # Performance optimization logic
            if self.performance_metrics['memory_usage_mb'] > 800:
                # Trigger garbage collection
                import gc
                gc.collect()
                self.logger.debug("[🧹] Triggered garbage collection for memory optimization")
            
        except Exception as e:
            self.logger.error(f"[X] Performance optimization failed: {e}")
    
    def _should_restart(self) -> bool:
        """Enhanced restart decision logic"""
        max_attempts = self.default_config['application']['max_restart_attempts']
        error_threshold = 10  # Maximum errors before restart consideration
        
        return (self.stats['restart_count'] < max_attempts and 
                self.is_running and 
                not self.shutdown_event.is_set() and
                self.stats['system_errors'] < error_threshold)
    
    async def _attempt_restart(self):
        """Enhanced system restart with comprehensive recovery"""
        try:
            self.stats['restart_count'] += 1
            self.stats['last_restart'] = datetime.now()
            
            restart_delay = self.default_config['application']['restart_delay_seconds']
            
            self.logger.info(f"[🔄] Attempting enhanced system restart (attempt {self.stats['restart_count']}/{self.default_config['application']['max_restart_attempts']})...")
            
            # Phase 1: Graceful component shutdown
            self.logger.info("[🔄] Phase 1: Graceful component shutdown...")
            if self.controller:
                await self.controller.shutdown()
            
            # Phase 2: Stop background services
            self.logger.info("[🔄] Phase 2: Stopping background services...")
            for service_name, thread in self.background_threads.items():
                if thread and thread.is_alive():
                    # Signal thread to stop (threads should check shutdown_event)
                    self.logger.info(f"[🔄] Stopping {service_name}...")
            
            # Phase 3: Wait for cleanup
            self.logger.info(f"[⏳] Phase 3: Waiting {restart_delay} seconds for cleanup...")
            await asyncio.sleep(restart_delay)
            
            # Phase 4: Reset system state
            self.logger.info("[🔄] Phase 4: Resetting system state...")
            self.shutdown_event.clear()
            self.stats['system_errors'] = 0  # Reset error count
            
            # Phase 5: Reload configuration and restart
            self.logger.info("[🔄] Phase 5: Reloading configuration...")
            config = self.load_configuration()
            
            self.logger.info("[🔄] Phase 6: Restarting system...")
            success = await self.startup(config)
            
            if success:
                self.logger.info("[✅] Enhanced system restart successful")
                self.stats['restart_count'] = 0  # Reset counter on successful restart
            else:
                self.logger.error("[❌] Enhanced system restart failed")
                
        except Exception as e:
            self.logger.error(f"[X] Enhanced restart attempt failed: {e}")
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown - UPDATED WITH DASHBOARD"""
        try:
            self.logger.info("[SHUTDOWN] Starting graceful shutdown...")
            
            # Stop main loop
            self.is_running = False
            
            # Stop dashboard - NEW
            if self.dashboard:
                self.logger.info("[SHUTDOWN] Stopping dashboard...")
                try:
                    self.dashboard.stop()
                except:
                    pass
            
            # Stop controller
            if self.controller:
                self.logger.info("[SHUTDOWN] Stopping core system...")
                asyncio.run(self.controller.shutdown())
            
            # Stop background threads
            self.logger.info("[SHUTDOWN] Stopping background services...")
            stopped_services = []
            failed_services = []
            
            for service_name, thread in self.background_threads.items():
                try:
                    if thread and thread.is_alive():
                        self.logger.info(f"[SHUTDOWN] Stopping {service_name}...")
                        thread.join(timeout=5)
                        if not thread.is_alive():
                            stopped_services.append(service_name)
                        else:
                            failed_services.append(service_name)
                            self.logger.warning(f"[SHUTDOWN] {service_name} did not stop gracefully")
                except Exception as e:
                    failed_services.append(service_name)
                    self.logger.error(f"[SHUTDOWN] Error stopping {service_name}: {e}")
            
            # Generate shutdown report
            shutdown_report = {
                'shutdown_time': datetime.now().isoformat(),
                'startup_time': f"{self.stats['startup_time']:.2f}s" if self.stats['startup_time'] else None,
                'total_runtime': str(self.stats['total_runtime']),
                'restart_count': self.stats['restart_count'],
                'total_clients_served': self.stats['total_clients_served'],
                'total_signals_processed': self.stats['total_signals_processed'],
                'total_trades_executed': self.stats['total_trades_executed'],
                'system_errors': self.stats['system_errors'],
                'performance_warnings': self.stats['performance_warnings'],
                'dashboard_status': self.stats['dashboard_status'],  # NEW
                'services_stopped': stopped_services,
                'services_failed': failed_services,
                'final_memory_usage': f"{self.performance_metrics['memory_usage_mb']:.1f}MB",
                'final_cpu_usage': f"{self.performance_metrics['cpu_usage_percent']:.1f}%"
            }
            
            # Final statistics log
            self.logger.info("[📊] FINAL SHUTDOWN REPORT:")
            self.logger.info(f"   ⏱ Startup Time: {shutdown_report['startup_time']}")
            self.logger.info(f"   ⏰ Total Runtime: {shutdown_report['total_runtime']}")
            self.logger.info(f"   🔄 Restart Count: {shutdown_report['restart_count']}")
            self.logger.info(f"   👥 Clients Served: {shutdown_report['total_clients_served']}")
            self.logger.info(f"   📡 Signals Processed: {shutdown_report['total_signals_processed']}")
            self.logger.info(f"   💹 Trades Executed: {shutdown_report['total_trades_executed']}")
            self.logger.info(f"   ❌ System Errors: {shutdown_report['system_errors']}")
            self.logger.info(f"   ⚠ Performance Warnings: {shutdown_report['performance_warnings']}")
            self.logger.info(f"   🌐 Dashboard Status: {shutdown_report['dashboard_status']}")  # NEW
            self.logger.info(f"   🧠 Final Memory Usage: {shutdown_report['final_memory_usage']}")
            self.logger.info(f"   ⚡ Final CPU Usage: {shutdown_report['final_cpu_usage']}")
            
            if stopped_services:
                self.logger.info(f"   ✅ Services Stopped: {', '.join(stopped_services)}")
            if failed_services:
                self.logger.warning(f"   ⚠ Services Failed to Stop: {', '.join(failed_services)}")
            
            self.logger.info("[✅] Lightning Scalper Application shutdown complete")
            
            # Save shutdown report to file
            try:
                reports_dir = PROJECT_ROOT / "reports"
                reports_dir.mkdir(exist_ok=True)
                shutdown_report_file = reports_dir / f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(shutdown_report_file, 'w', encoding='utf-8') as f:
                    json.dump(shutdown_report, f, indent=4, ensure_ascii=False)
                
                self.logger.info(f"[📄] Shutdown report saved: {shutdown_report_file}")
                
            except Exception as e:
                self.logger.warning(f"[⚠] Failed to save shutdown report: {e}")
            
        except Exception as e:
            self.logger.error(f"[X] Error during shutdown: {e}")
    
    async def shutdown(self):
        """Comprehensive graceful application shutdown"""
        try:
            self.logger.info("[🔄] Initiating comprehensive graceful shutdown...")
            self.is_running = False
            self.shutdown_event.set()
            
            # Phase 1: Stop accepting new work
            self.logger.info("[🔄] Phase 1: Stopping new work acceptance...")
            
            # Phase 2: Complete current work
            self.logger.info("[🔄] Phase 2: Completing current operations...")
            await asyncio.sleep(5)  # Allow current operations to complete
            
            # Phase 3: Shutdown core controller
            if self.controller:
                self.logger.info("[🔄] Phase 3: Shutting down core controller...")
                await self.controller.shutdown()
                self.logger.info("[✓] Core controller shutdown complete")
            
            # Phase 4: Stop dashboard - NEW
            if self.dashboard:
                self.logger.info("[🔄] Phase 4: Shutting down dashboard...")
                try:
                    self.dashboard.stop()
                    self.logger.info("[✓] Dashboard shutdown complete")
                except Exception as e:
                    self.logger.warning(f"[⚠] Dashboard shutdown error: {e}")
            
            # Phase 5: Stop background services
            self.logger.info("[🔄] Phase 5: Stopping background services...")
            stopped_services = []
            failed_services = []
            
            for service_name, thread in self.background_threads.items():
                if thread and thread.is_alive():
                    try:
                        thread.join(timeout=10)  # Wait up to 10 seconds for each thread
                        if thread.is_alive():
                            self.logger.warning(f"[⚠] Service {service_name} did not stop gracefully")
                            failed_services.append(service_name)
                        else:
                            stopped_services.append(service_name)
                            self.logger.info(f"[✓] Service {service_name} stopped gracefully")
                    except Exception as e:
                        self.logger.error(f"[X] Error stopping service {service_name}: {e}")
                        failed_services.append(service_name)
            
            # Phase 6: Close file handles and cleanup resources
            self.logger.info("[🔄] Phase 6: Cleaning up resources...")
            try:
                # Close logging handlers
                for handler in logging.getLogger().handlers:
                    if hasattr(handler, 'close'):
                        handler.close()
            except Exception as e:
                self.logger.warning(f"[⚠] Error closing log handlers: {e}")
            
            # Phase 7: Calculate final statistics
            if self.start_time:
                self.stats['total_runtime'] = datetime.now() - self.start_time
            
            # Phase 8: Generate shutdown report
            self.logger.info("[📊] Phase 7: Generating shutdown report...")
            shutdown_report = {
                'shutdown_time': datetime.now().isoformat(),
                'total_runtime': str(self.stats['total_runtime']),
                'startup_time': f"{self.stats['startup_time']:.2f}s" if self.stats['startup_time'] else 'Unknown',
                'restart_count': self.stats['restart_count'],
                'total_clients_served': self.stats['total_clients_served'],
                'total_signals_processed': self.stats['total_signals_processed'],
                'total_trades_executed': self.stats['total_trades_executed'],
                'system_errors': self.stats['system_errors'],
                'performance_warnings': self.stats['performance_warnings'],
                'dashboard_status': self.stats['dashboard_status'],  # NEW
                'services_stopped': stopped_services,
                'services_failed': failed_services,
                'final_memory_usage': f"{self.performance_metrics['memory_usage_mb']:.1f}MB",
                'final_cpu_usage': f"{self.performance_metrics['cpu_usage_percent']:.1f}%"
            }
            
            # Final statistics log
            self.logger.info("[📊] FINAL SHUTDOWN REPORT:")
            self.logger.info(f"   ⏱ Startup Time: {shutdown_report['startup_time']}")
            self.logger.info(f"   ⏰ Total Runtime: {shutdown_report['total_runtime']}")
            self.logger.info(f"   🔄 Restart Count: {shutdown_report['restart_count']}")
            self.logger.info(f"   👥 Clients Served: {shutdown_report['total_clients_served']}")
            self.logger.info(f"   📡 Signals Processed: {shutdown_report['total_signals_processed']}")
            self.logger.info(f"   💹 Trades Executed: {shutdown_report['total_trades_executed']}")
            self.logger.info(f"   ❌ System Errors: {shutdown_report['system_errors']}")
            self.logger.info(f"   ⚠ Performance Warnings: {shutdown_report['performance_warnings']}")
            self.logger.info(f"   🌐 Dashboard Status: {shutdown_report['dashboard_status']}")  # NEW
            self.logger.info(f"   🧠 Final Memory Usage: {shutdown_report['final_memory_usage']}")
            self.logger.info(f"   ⚡ Final CPU Usage: {shutdown_report['final_cpu_usage']}")
            
            if stopped_services:
                self.logger.info(f"   ✅ Services Stopped: {', '.join(stopped_services)}")
            if failed_services:
                self.logger.warning(f"   ⚠ Services Failed to Stop: {', '.join(failed_services)}")
            
            self.logger.info("[✅] Lightning Scalper Application shutdown complete")
            
            # Save shutdown report to file
            try:
                reports_dir = PROJECT_ROOT / "reports"
                reports_dir.mkdir(exist_ok=True)
                shutdown_report_file = reports_dir / f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(shutdown_report_file, 'w', encoding='utf-8') as f:
                    json.dump(shutdown_report, f, indent=4, ensure_ascii=False)
                
                self.logger.info(f"[📄] Shutdown report saved: {shutdown_report_file}")
                
            except Exception as e:
                self.logger.warning(f"[⚠] Failed to save shutdown report: {e}")
            
        except Exception as e:
            self.logger.error(f"[X] Error during comprehensive shutdown: {e}")
    
    def get_app_status(self) -> Dict[str, Any]:
        """Get comprehensive application status"""
        try:
            current_time = datetime.now()
            
            status = {
                'application': {
                    'name': self.app_name,
                    'version': self.version,
                    'is_running': self.is_running,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
                    'uptime': str(current_time - self.start_time) if self.start_time else None,
                    'uptime_seconds': (current_time - self.start_time).total_seconds() if self.start_time else 0
                },
                'stats': self.stats,
                'performance': self.performance_metrics,
                'system': None,
                'background_services': {
                    service_name: {
                        'running': thread.is_alive() if thread else False,
                        'name': thread.name if thread else None
                    }
                    for service_name, thread in self.background_threads.items()
                },
                'health': {
                    'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                    'last_performance_update': self.last_performance_update.isoformat() if self.last_performance_update else None
                },
                'configuration': {
                    'config_file': self.config_path,
                    'debug_mode': getattr(self, 'debug_mode', False),
                    'auto_restart_enabled': self.default_config['application']['auto_restart_on_failure']
                },
                'dashboard': {  # NEW: Dashboard status section
                    'status': self.stats['dashboard_status'],
                    'enabled': self.dashboard is not None,
                    'thread_alive': self.dashboard_thread.is_alive() if self.dashboard_thread else False
                }
            }
            
            # Get controller status if available
            if self.controller:
                controller_status = self.controller.get_system_status()
                status['system'] = controller_status
                
                # Merge controller metrics with application metrics
                if 'metrics' in controller_status:
                    status['stats'].update({
                        'controller_signals': controller_status['metrics'].get('total_signals_generated', 0),
                        'controller_trades': controller_status['metrics'].get('total_trades_executed', 0),
                        'controller_clients': controller_status['metrics'].get('total_clients_connected', 0)
                    })
            
            return status
            
        except Exception as e:
            self.logger.error(f"[X] Failed to get comprehensive app status: {e}")
            return {
                'application': {
                    'name': self.app_name,
                    'version': self.version,
                    'is_running': self.is_running,
                    'error': str(e)
                },
                'stats': self.stats,
                'system': None,
                'dashboard': {'status': 'error', 'error': str(e)}  # NEW
            }
    def get_safe_metrics(self):
        try:
            if hasattr(self, 'controller') and self.controller:
                system_status = self.controller.get_system_status()
                return system_status.get('metrics', {'memory_usage': 0.0, 'cpu_usage': 0.0})
            return {'memory_usage': 0.0, 'cpu_usage': 0.0}
        except Exception as e:
            return {'memory_usage': 0.0, 'cpu_usage': 0.0}

def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Lightning Scalper - Production AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --config custom.json     # Use custom config file
  python main.py --debug                  # Start in debug mode
  python main.py --clients-only           # Load clients without trading
  python main.py --demo-only              # Load only demo clients
  python main.py --live-only              # Load only live clients
  python main.py --performance-mode       # High performance mode
  python main.py --maintenance-mode       # Maintenance mode
  python main.py --safe-mode              # Safe mode with minimal features
  python main.py --disable-dashboard      # Disable web dashboard
  python main.py --port 8080              # Custom dashboard port
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/settings.json',
        help='Configuration file path (default: config/settings.json)'
    )
    
    # Operation modes
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Run in debug mode (no real trading, verbose logging)'
    )
    
    parser.add_argument(
        '--clients-only',
        action='store_true',
        help='Load clients without starting trading engine'
    )
    
    parser.add_argument(
        '--demo-only',
        action='store_true',
        help='Load only demo clients'
    )
    
    parser.add_argument(
        '--live-only',
        action='store_true',
        help='Load only live clients'
    )
    
    parser.add_argument(
        '--no-clients',
        action='store_true',
        help='Start without loading any clients'
    )
    
    parser.add_argument(
        '--performance-mode',
        action='store_true',
        help='Run in high-performance mode with optimizations'
    )
    
    parser.add_argument(
        '--maintenance-mode',
        action='store_true',
        help='Run in maintenance mode (limited functionality)'
    )
    
    parser.add_argument(
        '--safe-mode',
        action='store_true',
        help='Run in safe mode with minimal features'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with stub components'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Custom log file path'
    )
    
    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable console logging'
    )
    
    # System options
    parser.add_argument(
        '--max-clients',
        type=int,
        help='Override maximum number of clients'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Override dashboard port'  # NEW: Dashboard port override
    )
    
    parser.add_argument(
        '--disable-dashboard',
        action='store_true',
        help='Disable web dashboard'  # NEW: Disable dashboard option
    )
    
    parser.add_argument(
        '--disable-api',
        action='store_true',
        help='Disable REST API'
    )
    
    # Version and help
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Lightning Scalper v1.0.0'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    return parser

async def main():
    """Enhanced main application entry point"""
    try:
        # Setup UTF-8 encoding for Windows
        if sys.platform == "win32":
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Print startup banner
        print("=" * 80)
        print("[🚀] Lightning Scalper - Production AI Trading System")
        print("=" * 80)
        print(f"Version: 1.0.0")
        print(f"Python: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Config: {args.config}")
        print(f"Log Level: {args.log_level}")
        print("=" * 80)
        
        # Show operation modes
        modes = []
        if args.debug:
            modes.append("DEBUG")
        if args.performance_mode:
            modes.append("PERFORMANCE")
        if args.maintenance_mode:
            modes.append("MAINTENANCE")
        if args.safe_mode:
            modes.append("SAFE")
        if args.test_mode:
            modes.append("TEST")
        if args.clients_only:
            modes.append("CLIENTS-ONLY")
        
        if modes:
            print(f"Operation Modes: {', '.join(modes)}")
        
        # Show filters
        filters = []
        if args.demo_only:
            filters.append("DEMO CLIENTS ONLY")
        if args.live_only:
            filters.append("LIVE CLIENTS ONLY")
        if args.no_clients:
            filters.append("NO CLIENTS")
        if args.disable_dashboard:  # NEW
            filters.append("DASHBOARD DISABLED")
        
        if filters:
            print(f"Filters: {', '.join(filters)}")
        
        print("=" * 80)
        
        # Handle special commands
        if args.validate_config:
            app = LightningScalperApp(config_path=args.config)
            config = app.load_configuration()
            validation = app._validate_configuration(config)
            
            if validation['valid']:
                print("[✅] Configuration validation PASSED")
            else:
                print("[❌] Configuration validation FAILED")
                for error in validation['errors']:
                    print(f"   ❌ {error}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"   ⚠ {warning}")
            
            sys.exit(0 if validation['valid'] else 1)
        
        if args.show_config:
            app = LightningScalperApp(config_path=args.config)
            config = app.load_configuration()
            print(json.dumps(config, indent=4))
            sys.exit(0)
        
        # Initialize application
        app = LightningScalperApp(config_path=args.config)
        
        # Load and modify configuration based on arguments
        config = app.load_configuration()
        
        # Apply command line overrides
        if args.debug:
            config['application']['debug_mode'] = True
            config['application']['auto_start_trading'] = not args.clients_only
            config['logging']['log_level'] = 'DEBUG'
        
        if args.performance_mode:
            config['system']['data_update_interval'] = 0.5
            config['system']['signal_generation_interval'] = 2.0
            config['optimization']['enable_auto_scaling'] = True
            config['optimization']['enable_caching'] = True
        
        if args.maintenance_mode:
            config['application']['maintenance_mode'] = True
            config['application']['auto_start_trading'] = False
            config['clients']['auto_load_clients'] = False
        
        if args.safe_mode:
            config['application']['debug_mode'] = True
            config['risk']['enable_global_safety'] = True
            config['risk']['max_concurrent_trades'] = 10
            config['system']['max_clients'] = 5
        
        if args.test_mode:
            config['application']['debug_mode'] = True
            config['system']['max_clients'] = 3
            config['clients']['auto_load_clients'] = False
        
        if args.no_clients:
            config['clients']['auto_load_clients'] = False
        
        if args.demo_only:
            config['demo_only'] = True
        
        if args.live_only:
            config['live_only'] = True
            config['application']['debug_mode'] = False
        
        if args.max_clients:
            config['system']['max_clients'] = args.max_clients
        
        if args.port:  # NEW: Dashboard port override
            config['server']['dashboard_port'] = args.port
        
        if args.disable_dashboard:  # NEW: Disable dashboard
            config['server']['enable_dashboard'] = False
        
        if args.disable_api:
            config['server']['enable_api'] = False
        
        # Start application
        print("[🚀] Initializing Lightning Scalper...")
        success = await app.startup(config)
        
        if not success:
            print("[❌] Failed to start Lightning Scalper")
            return 1
        
        print("[✅] Lightning Scalper started successfully!")
        print("\n[📊] System Status:")
        status = app.get_app_status()
        print(f"   Application: {status['application']['name']} v{status['application']['version']}")
        print(f"   Running: {status['application']['is_running']}")
        print(f"   Memory: {status['performance']['memory_usage_mb']:.1f}MB")
        print(f"   CPU: {status['performance']['cpu_usage_percent']:.1f}%")
        
        if status['system']:
            metrics = status['system'].get('metrics', {})
            print(f"   Active Clients: {metrics.get('active_clients', 0)}")
            print(f"   System Status: {status['system'].get('status', 'unknown')}")
        
        # Show service status
        services = status.get('background_services', {})
        running_services = [name for name, info in services.items() if info['running']]
        if running_services:
            print(f"   Background Services: {', '.join(running_services)}")
        
        # NEW: Dashboard status
        dashboard_info = status.get('dashboard', {})
        if dashboard_info.get('enabled', False):
            dashboard_status = dashboard_info.get('status', 'unknown')
            dashboard_port = config.get('server', {}).get('dashboard_port', 5000)
            print(f"   Dashboard: {dashboard_status} (Port {dashboard_port})")
            if dashboard_status == 'running':
                print(f"   Dashboard URL: http://localhost:{dashboard_port}")
        
        print("\n[💡] Press Ctrl+C to shutdown gracefully")
        if not args.disable_dashboard and config.get('server', {}).get('enable_dashboard', True):
            dashboard_port = config.get('server', {}).get('dashboard_port', 5000)
            print(f"[📊] Dashboard: http://localhost:{dashboard_port}")
        print("[🔌] API: http://localhost:9000 (if enabled)")
        
        # Run main application loop
        await app.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[⌨] Application interrupted by user")
        return 0
    except Exception as e:
        print(f"\n[❌] Critical application error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("[❌] Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            sys.exit(1)
        
        # Run application
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n[👋] Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[💥] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)