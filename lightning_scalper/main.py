#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Windows Safe Version
Auto-fixed for Unicode compatibility
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

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
LIGHTNING_SCALPER_PATH = PROJECT_ROOT / "lightning_scalper"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIGHTNING_SCALPER_PATH))

# Import our core modules
try:
    from lightning_scalper.core.main_controller import LightningScalperController
    from lightning_scalper.core.lightning_scalper_engine import CurrencyPair
    from lightning_scalper.execution.trade_executor import ClientAccount
    from lightning_scalper.adapters.mt5_adapter import MT5Adapter
except ImportError as e:
    print(f"[X] Failed to import core modules: {e}")
    print("   Make sure all required files are in the correct directories")
    sys.exit(1)

class LightningScalperApp:
    """
    [ROCKET] Lightning Scalper Main Application
    Production application wrapper for the trading system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.app_name = "Lightning Scalper"
        self.version = "1.0.0"
        self.config_path = config_path or "lightning_scalper/config/settings.json"
        
        # Core system
        self.controller: Optional[LightningScalperController] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Application state
        self.start_time: Optional[datetime] = None
        self.stats = {
            'startup_time': None,
            'total_runtime': timedelta(0),
            'restart_count': 0,
            'last_restart': None
        }
        
        # Setup logging
        self._setup_application_logging()
        self.logger = logging.getLogger('LightningScalperApp')
        
        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Default configuration
        self.default_config = self._get_default_config()
        
        self.logger.info(f"[ROCKET] {self.app_name} v{self.version} initialized")
    
    def _setup_application_logging(self):
        """Setup comprehensive application logging"""
        # Create logs directory if it doesn't exist
        logs_dir = LIGHTNING_SCALPER_PATH / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(logs_dir / 'lightning_scalper_main.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Create separate loggers for different components
        component_loggers = [
            'LightningScalperController',
            'EnhancedFVGDetector', 
            'TradeExecutor',
            'MT5Adapter'
        ]
        
        for logger_name in component_loggers:
            logger = logging.getLogger(logger_name)
            handler = logging.FileHandler(logs_dir / f'{logger_name.lower()}.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"[SATELLITE] Received {signal_name} signal, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        if hasattr(signal, 'SIGHUP'):  # Unix systems only
            signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default application configuration"""
        return {
            "application": {
                "name": self.app_name,
                "version": self.version,
                "debug_mode": True,
                "auto_start_trading": False,
                "max_startup_time": 60,  # seconds
                "health_check_interval": 30,  # seconds
                "auto_restart_on_failure": True,
                "max_restart_attempts": 3
            },
            "system": {
                "max_clients": 100,
                "data_update_interval": 1.0,
                "signal_generation_interval": 5.0,
                "auto_reconnect": True,
                "max_reconnect_attempts": 5,
                "enable_performance_monitoring": True
            },
            "risk": {
                "global_daily_loss_limit": 5000.0,
                "max_concurrent_trades": 200,
                "emergency_stop_loss_percent": 10.0,
                "max_signals_per_client_hour": 10,
                "enable_global_safety": True
            },
            "trading": {
                "default_currency_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"],
                "default_timeframes": ["M1", "M5", "M15", "H1"],
                "min_signal_confluence": 65.0,
                "max_slippage_pips": 3.0,
                "default_risk_per_trade": 0.02,
                "allow_demo_trading": True
            },
            "logging": {
                "log_level": "INFO",
                "max_log_file_size": "100MB",
                "log_retention_days": 30,
                "enable_performance_logs": True,
                "enable_trade_logs": True
            },
            "clients": {
                "auto_load_clients": True,
                "clients_config_file": "lightning_scalper/config/clients.json",
                "default_client_settings": {
                    "max_positions": 5,
                    "max_lot_size": 1.0,
                    "risk_multiplier": 1.0,
                    "auto_trading": True
                }
            }
        }
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load application configuration from file"""
        try:
            config_file = PROJECT_ROOT / self.config_path
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Merge with default config
                config = self.default_config.copy()
                self._deep_merge_config(config, file_config)
                
                self.logger.info(f"[CHECK] Configuration loaded from {config_file}")
                return config
            else:
                self.logger.warning(f"[WARNING] Config file {config_file} not found, using defaults")
                
                # Create default config file
                self._create_default_config_file(config_file)
                return self.default_config
                
        except Exception as e:
            self.logger.error(f"[X] Failed to load configuration: {e}")
            self.logger.info("[REFRESH] Using default configuration")
            return self.default_config
    
    def _deep_merge_config(self, base: Dict, update: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _create_default_config_file(self, config_file: Path):
        """Create default configuration file"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.default_config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"[MEMO] Created default config file: {config_file}")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create config file: {e}")
    
    def load_clients_configuration(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load client configurations"""
        try:
            if not config['clients']['auto_load_clients']:
                return []
            
            clients_file = PROJECT_ROOT / config['clients']['clients_config_file']
            
            if clients_file.exists():
                with open(clients_file, 'r', encoding='utf-8') as f:
                    clients_data = json.load(f)
                
                self.logger.info(f"[CHECK] Loaded {len(clients_data)} client configurations")
                
                # Handle both list and dict formats
                if isinstance(clients_data, list):
                    return clients_data
                else:
                    return clients_data.get('clients', [])
            else:
                self.logger.warning(f"[WARNING] Clients config file {clients_file} not found")
                
                # Create sample clients file
                self._create_sample_clients_file(clients_file, config)
                return []
                
        except Exception as e:
            self.logger.error(f"[X] Failed to load clients configuration: {e}")
            return []
    
    def _create_sample_clients_file(self, clients_file: Path, config: Dict[str, Any]):
        """Create sample clients configuration file"""
        try:
            clients_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_clients = [
                {
                    "client_id": "DEMO_CLIENT_001",
                    "is_demo": True,
                    "mt5_login": 12345001,
                    "mt5_password": "REPLACE_WITH_REAL_PASSWORD",
                    "mt5_server": "REPLACE_WITH_REAL_SERVER",
                    "account_info": {
                        "client_id": "DEMO_CLIENT_001",
                        "account_number": "12345001",
                        "broker": "MetaTrader5",
                        "currency": "USD",
                        "balance": 10000.0,
                        "equity": 10000.0,
                        "margin": 0.0,
                        "free_margin": 10000.0,
                        "margin_level": 0.0,
                        "max_daily_loss": 200.0,
                        "max_weekly_loss": 500.0,
                        "max_monthly_loss": 1500.0,
                        "max_positions": 5,
                        "max_lot_size": 1.0,
                        "preferred_pairs": ["EURUSD", "GBPUSD", "USDJPY"],
                        "trading_sessions": ["London", "NewYork"]
                    },
                    "preferred_pairs": ["EURUSD", "GBPUSD", "USDJPY"],
                    "risk_multiplier": 1.0,
                    "max_signals_per_hour": 8,
                    "auto_trading": True
                }
            ]
            
            with open(clients_file, 'w', encoding='utf-8') as f:
                json.dump(sample_clients, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"[MEMO] Created sample clients file: {clients_file}")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to create sample clients file: {e}")
    
    async def startup(self, config: Dict[str, Any]) -> bool:
        """Application startup sequence"""
        try:
            self.start_time = datetime.now()
            startup_start = time.time()
            
            self.logger.info("[ROCKET] Starting Lightning Scalper Application...")
            
            # 1. Initialize controller
            self.logger.info("[SATELLITE] Initializing system controller...")
            self.controller = LightningScalperController(self.config_path)
            
            # 2. Start core system
            self.logger.info("[LIGHTNING] Starting core trading system...")
            success = await self.controller.start_system()
            
            if not success:
                self.logger.error("[X] Failed to start core trading system")
                return False
            
            # 3. Load and add clients
            clients_data = self.load_clients_configuration(config)
            
            if clients_data:
                self.logger.info(f"[USERS] Loading {len(clients_data)} clients...")
                
                for client_data in clients_data:
                    try:
                        # Skip demo clients in production mode
                        if client_data.get('is_demo', False) and not config['application']['debug_mode']:
                            self.logger.info(f"[SKIP] Skipping demo client {client_data['client_id']} (production mode)")
                            continue
                        
                        # Add client to system
                        if config['application']['debug_mode']:
                            # In debug mode, just register without MT5 connection
                            client_account = ClientAccount(**client_data['account_info'])
                            success = self.controller.trade_executor.register_client(client_account)
                            
                            if success:
                                # Also add to controller's active_clients
                                self.controller.add_client(client_account)
                                self.logger.info(f"[CHECK] Client {client_data['client_id']} registered (Debug Mode)")
                            else:
                                self.logger.warning(f"[WARNING] Failed to register client {client_data['client_id']}")
                        else:
                            # Production mode - full MT5 integration
                            client_account = ClientAccount(**client_data['account_info'])
                            success = self.controller.add_client(client_account)
                            if success:
                                self.logger.info(f"[CHECK] Client {client_data['client_id']} connected successfully")
                            else:
                                self.logger.warning(f"[WARNING] Failed to connect client {client_data['client_id']}")
                    
                    except Exception as e:
                        self.logger.error(f"[X] Error loading client {client_data.get('client_id', 'Unknown')}: {e}")
            else:
                self.logger.info("[INFO] No clients to load")
            
            # 4. Setup application monitoring
            if config['system']['enable_performance_monitoring']:
                self._start_performance_monitoring()
            
            # 5. Calculate startup time
            startup_time = time.time() - startup_start
            self.stats['startup_time'] = startup_time
            
            # 6. Final status check
            system_status = self.controller.get_system_status()
            
            self.logger.info("[CHECK] Lightning Scalper Application started successfully!")
            self.logger.info(f"   [TIME] Startup time: {startup_time:.2f} seconds")
            self.logger.info(f"   [STATUS] System status: {system_status.get('status', 'Unknown')}")
            self.logger.info(f"   [USERS] Active clients: {system_status.get('metrics', {}).get('active_clients', 0)}")
            
            self.is_running = True
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Startup failed: {e}")
            return False
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        def performance_monitor():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    if self.controller:
                        status = self.controller.get_system_status()
                        
                        # Log performance metrics every 5 minutes
                        metrics = status.get('metrics', {})
                        self.logger.debug(f"[PERFORMANCE] "
                                       f"Signals: {metrics.get('total_signals_today', 0)}, "
                                       f"Trades: {metrics.get('executed_trades_today', 0)}, "
                                       f"P&L: ${metrics.get('total_pnl_today', 0.0):.2f}")
                    
                    # Sleep for monitoring interval
                    self.shutdown_event.wait(300)  # 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"[X] Performance monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=performance_monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("[CHECK] Performance monitoring started")
    
    async def run(self):
        """Main application run loop"""
        try:
            # Load configuration
            config = self.load_configuration()
            
            # Startup
            success = await self.startup(config)
            if not success:
                self.logger.error("[X] Failed to start Lightning Scalper")
                return
            
            self.logger.info("[ROCKET] Lightning Scalper is running...")
            
            # Main application loop
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Check system health
                    if self.controller:
                        status = self.controller.get_system_status()
                        
                        # Check for emergency conditions
                        safety = status.get('safety', {})
                        if safety.get('emergency_stop', False):
                            self.logger.critical("[ALERT] Emergency stop detected!")
                            # Could implement auto-restart logic here
                    
                    # Wait for shutdown signal or periodic check
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"[X] Error in main loop: {e}")
                    
                    # Decide whether to restart or shutdown
                    if self._should_restart():
                        await self._attempt_restart()
                    else:
                        self.logger.error("[X] Too many failures, shutting down")
                        break
            
        except KeyboardInterrupt:
            self.logger.info("[INTERRUPT] Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"[X] Critical error in main loop: {e}")
        finally:
            await self.shutdown()
    
    def _should_restart(self) -> bool:
        """Determine if system should attempt restart"""
        return (self.stats['restart_count'] < 3 and 
                self.is_running and 
                not self.shutdown_event.is_set())
    
    async def _attempt_restart(self):
        """Attempt to restart the system"""
        try:
            self.stats['restart_count'] += 1
            self.stats['last_restart'] = datetime.now()
            
            self.logger.info(f"[REFRESH] Attempting system restart (attempt {self.stats['restart_count']}/3)...")
            
            # Shutdown current system
            if self.controller:
                await self.controller.shutdown()
            
            # Wait before restart
            await asyncio.sleep(10)
            
            # Reload configuration and restart
            config = self.load_configuration()
            success = await self.startup(config)
            
            if success:
                self.logger.info("[CHECK] System restart successful")
                self.stats['restart_count'] = 0  # Reset counter on successful restart
            else:
                self.logger.error("[X] System restart failed")
                
        except Exception as e:
            self.logger.error(f"[X] Restart attempt failed: {e}")
    
    async def shutdown(self):
        """Graceful application shutdown"""
        try:
            self.logger.info("[SATELLITE] Initiating graceful shutdown...")
            self.is_running = False
            self.shutdown_event.set()
            
            # Calculate total runtime
            if self.start_time:
                self.stats['total_runtime'] = datetime.now() - self.start_time
                self.logger.info(f"[TIME] Total runtime: {self.stats['total_runtime']}")
            
            # Shutdown controller
            if self.controller:
                await self.controller.shutdown()
                self.logger.info("[CHECK] Controller shutdown complete")
            
            # Final statistics
            self.logger.info("[CHART] Final Statistics:")
            self.logger.info(f"   [TIME] Startup time: {self.stats['startup_time']:.2f}s")
            self.logger.info(f"   [TIMER] Total runtime: {self.stats['total_runtime']}")
            self.logger.info(f"   [REFRESH] Restart count: {self.stats['restart_count']}")
            
            self.logger.info("[CHECK] Lightning Scalper Application shutdown complete")
            
        except Exception as e:
            self.logger.error(f"[X] Error during shutdown: {e}")
    
    def get_app_status(self) -> Dict[str, Any]:
        """Get application status"""
        status = {
            'application': {
                'name': self.app_name,
                'version': self.version,
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': str(datetime.now() - self.start_time) if self.start_time else None
            },
            'stats': self.stats,
            'system': None
        }
        
        if self.controller:
            status['system'] = self.controller.get_system_status()
        
        return status

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Lightning Scalper - Production AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --config custom.json     # Use custom config file
  python main.py --debug                  # Start in debug mode
  python main.py --clients-only           # Load clients without trading
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lightning_scalper/config/settings.json',
        help='Configuration file path (default: lightning_scalper/config/settings.json)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Run in debug mode (no real trading)'
    )
    
    parser.add_argument(
        '--clients-only',
        action='store_true',
        help='Load clients without starting trading engine'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Lightning Scalper v1.0.0'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    return parser

def main():
    """Main application entry point"""
    # Setup UTF-8 encoding for Windows
    if sys.platform == "win32":
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("[ROCKET] Lightning Scalper - Production AI Trading System")
    print("=" * 60)
    print(f"Version: 1.0.0")
    print(f"Config: {args.config}")
    print(f"Debug Mode: {args.debug}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    
    # Initialize application
    app = LightningScalperApp(config_path=args.config)
    
    try:
        # Run the application
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Application interrupted by user")
    except Exception as e:
        print(f"[X] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()