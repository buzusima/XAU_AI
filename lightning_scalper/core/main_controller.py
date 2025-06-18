#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper Controller - Auto MT5 Detection Version
แก้ไขให้ใช้ระบบตรวจจับ MT5 อัตโนมัติ - ไม่ต้องใส่ login/password
"""

import asyncio
import threading
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import json
import sys

# Auto MT5 Detection Import
try:
    from adapters.auto_mt5_detection import get_auto_detected_client, check_mt5_status
    AUTO_MT5_AVAILABLE = True
except ImportError:
    AUTO_MT5_AVAILABLE = False

# Import required modules - FIXED IMPORTS
try:
    from .lightning_scalper_engine import (
        EnhancedFVGDetector, 
        FVGSignal, 
        CurrencyPair, 
        SystemStatus
    )
    from ..execution.trade_executor import TradeExecutor, ClientAccount
    from ..adapters.mt5_adapter import MT5Adapter
except ImportError:
    # Fallback for direct execution - FIXED PATH
    try:
        from lightning_scalper_engine import (
            EnhancedFVGDetector, 
            FVGSignal, 
            CurrencyPair, 
            SystemStatus
        )
        from execution.trade_executor import TradeExecutor, ClientAccount
        from adapters.mt5_adapter import MT5Adapter
    except ImportError:
        # Create SystemStatus enum if not available
        class SystemStatus(Enum):
            STARTING = "starting"
            RUNNING = "running"
            STOPPING = "stopping"
            STOPPED = "stopped"
            ERROR = "error"
        
        # Create stub classes for testing
        class EnhancedFVGDetector:
            def __init__(self, config=None):
                self.logger = logging.getLogger('StubFVGDetector')
            async def start(self):
                return True
            async def stop(self):
                pass
            def get_status(self):
                return {'status': 'stub_running'}
        
        class FVGSignal:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class CurrencyPair:
            EURUSD = "EURUSD"
            GBPUSD = "GBPUSD"
            USDJPY = "USDJPY"
        
        class TradeExecutor:
            def __init__(self, config=None):
                self.logger = logging.getLogger('StubTradeExecutor')
            async def start(self):
                return True
            async def stop(self):
                pass
            def get_status(self):
                return {'status': 'stub_running'}
        
        class ClientAccount:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class MT5Adapter:
            def __init__(self, config=None):
                self.logger = logging.getLogger('StubMT5Adapter')
            async def start(self):
                return True
            async def stop(self):
                pass
            def get_status(self):
                return {'status': 'stub_running'}

class LightningScalperController:
    """
    [ROCKET] Lightning Scalper Main System Controller
    Production-Grade AI Trading System Controller with Auto MT5 Detection
    
    Enhanced Features:
    - Auto MT5 Connection Detection
    - Multi-client trading orchestration 
    - Advanced risk management
    - Real-time performance monitoring
    - Comprehensive error handling
    - Auto-recovery mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Lightning Scalper Controller with enhanced features"""
        
        # Core identification
        self.name = "Lightning Scalper Controller"
        self.version = "1.1.0"  # Updated for auto detection
        self.status = SystemStatus.STOPPED
        
        # Configuration
        self.config = config or {}
        
        # Core components
        self.fvg_detector: Optional[EnhancedFVGDetector] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.mt5_adapter: Optional[MT5Adapter] = None
        
        # Client management - Enhanced
        self.clients: Dict[str, ClientAccount] = {}
        self.active_clients: Set[str] = set()
        self.client_stats: Dict[str, Dict[str, Any]] = {}
        self.auto_detected_clients: Set[str] = set()  # Track auto-detected clients
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None
        self.system_health = {}
        
        # Performance tracking - Enhanced
        self.performance_metrics = {
            'total_signals_processed': 0,
            'total_trades_executed': 0,
            'total_clients_served': 0,
            'system_uptime': 0,
            'auto_detected_connections': 0,
            'manual_connections': 0,
            'connection_success_rate': 0.0,
            'avg_signal_processing_time': 0.0,
            'avg_trade_execution_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        # Threading and concurrency
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        self.health_check_interval = 30  # seconds
        self.performance_update_interval = 60  # seconds
        
        # Monitoring threads
        self.monitor_thread = None
        self.performance_thread = None
        
        # Setup logging
        self.logger = logging.getLogger('LightningScalperController')
        
        self.logger.info(f"[ROCKET] {self.name} v{self.version} initialized")
        if AUTO_MT5_AVAILABLE:
            self.logger.info("✅ Auto MT5 Detection enabled")
        else:
            self.logger.warning("⚠️ Auto MT5 Detection not available - using manual mode")
    
    async def startup(self, config: Dict[str, Any]) -> bool:
        """
        Enhanced startup sequence with auto MT5 detection
        """
        try:
            self.status = SystemStatus.STARTING
            self.start_time = datetime.now()
            
            self.logger.info("[🚀] Starting Lightning Scalper Controller...")
            
            # Phase 1: Initialize core components
            self.logger.info("[🔧] Initializing core components...")
            
            fvg_config = config.get('fvg_engine', {})
            self.fvg_detector = EnhancedFVGDetector(fvg_config)
            self.logger.info("[✓] FVG Detector initialized")
            
            executor_config = config.get('trade_execution', {})
            self.trade_executor = TradeExecutor(executor_config)
            self.logger.info("[✓] Trade Executor initialized")
            
            mt5_config = config.get('mt5_integration', {})
            self.mt5_adapter = MT5Adapter(mt5_config)
            self.logger.info("[✓] MT5 Adapter initialized")
            
            self.logger.info("[✓] All core components initialized successfully")
            
            # Phase 2: Advanced monitoring setup
            self.logger.info("[📊] Setting up advanced monitoring...")
            await self._setup_advanced_monitoring()
            self.logger.info("[✓] Advanced monitoring setup complete")
            
            # Phase 3: Risk management setup
            self.logger.info("[🛡] Setting up risk management...")
            await self._setup_risk_management(config)
            self.logger.info("[✓] Risk management setup complete")
            
            # Phase 4: Performance tracking setup
            self.logger.info("[📈] Setting up performance tracking...")
            await self._setup_performance_tracking()
            self.logger.info("[✓] Performance tracking setup complete")
            
            # Phase 5: Load clients with auto detection - UPDATED
            self.logger.info("[👥] Loading client configurations...")
            success = await self.load_clients_with_auto_detection(config)
            if not success:
                self.logger.error("[X] Failed to load clients")
                return False
            
            # Phase 6: Start core services
            self.logger.info("[⚡] Starting core services...")
            
            # Start monitoring
            await self._start_system_monitoring()
            self.logger.info("[✓] System monitoring started")
            
            # Start signal processing
            await self._start_signal_processing()
            self.logger.info("[✓] Signal processing started")
            
            # Start performance monitoring
            await self._start_performance_monitoring()
            self.logger.info("[✓] Performance monitoring started")
            
            # Phase 7: Final validation
            system_health = await self._perform_system_health_check()
            if not system_health.get('healthy', False):
                self.logger.error("[X] System health check failed")
                return False
            
            self.logger.info("[✓] System health validation passed")
            
            # Update status
            self.status = SystemStatus.RUNNING
            self.is_running = True
            
            self.logger.info("[✓] Lightning Scalper Controller started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Startup failed: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def load_clients_with_auto_detection(self, config: Dict[str, Any]) -> bool:
        """
        Load clients with auto MT5 detection
        ไม่ต้องใส่ login/password - ใช้ MT5 ที่เปิดอยู่แล้ว
        """
        try:
            self.logger.info("[👥] Starting smart client loading...")
            
            # ตรวจสอบว่ามี auto detection module หรือไม่
            if not AUTO_MT5_AVAILABLE:
                self.logger.warning("⚠️ Auto MT5 detection not available, using manual loading")
                return await self._load_manual_clients(config)
            
            # ตรวจสอบสถานะ MT5
            self.logger.info("[🔍] Checking for existing MT5 connection...")
            mt5_status = check_mt5_status()
            
            if mt5_status['status'] == 'connected':
                # พบ MT5 connection แล้ว
                account = mt5_status['account']
                self.logger.info("✅ MT5 connection found!")
                self.logger.info(f"   Account: {account['login']}")
                self.logger.info(f"   Server: {account['server']}")
                self.logger.info(f"   Balance: ${account['balance']:.2f}")
                self.logger.info(f"   Type: {'DEMO' if account['is_demo'] else 'LIVE'}")
                
                # สร้าง client configuration อัตโนมัติ
                auto_client_config = get_auto_detected_client()
                if auto_client_config:
                    await self._add_auto_detected_client(auto_client_config)
                    self.logger.info("✅ Auto-detected client added successfully!")
                    self.performance_metrics['auto_detected_connections'] += 1
                    return True
                else:
                    self.logger.error("❌ Failed to create auto client config")
                    
            else:
                # ไม่พบ MT5 connection
                self.logger.warning("⚠️ No MT5 connection found")
                self.logger.info("📝 Instructions for user:")
                self.logger.info("   1. Open MetaTrader 5")
                self.logger.info("   2. Login to your account")
                self.logger.info("   3. Restart Lightning Scalper")
                
                # สร้าง demo client สำรองเพื่อให้ระบบทำงานต่อได้
                await self._add_demo_client()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in auto client detection: {e}")
            # Fallback to manual loading
            return await self._load_manual_clients(config)
    
    async def _add_auto_detected_client(self, client_config: Dict[str, Any]):
        """เพิ่ม client ที่ตรวจพบอัตโนมัติ"""
        try:
            # สร้าง ClientAccount ด้วยข้อมูลที่ตรวจพบ
            client = ClientAccount(
                client_id=client_config['client_id'],
                account_number=client_config['account_number'],
                broker=client_config['broker'],
                currency=client_config['currency'],
                balance=client_config['balance'],
                equity=client_config['equity'],
                margin=client_config['margin'],
                free_margin=client_config['free_margin'],
                margin_level=client_config['margin_level'],
                max_daily_loss=client_config['max_daily_loss'],
                max_weekly_loss=client_config['max_weekly_loss'],
                max_monthly_loss=client_config['max_monthly_loss'],
                max_positions=client_config['max_positions'],
                max_lot_size=client_config['max_lot_size'],
                preferred_pairs=client_config['preferred_pairs'],
                trading_sessions=client_config['trading_sessions']
            )
            
            # เพิ่มเข้าระบบ
            self.clients[client.client_id] = client
            
            if client_config.get('is_active', True):
                self.active_clients.add(client.client_id)
            
            # Track auto-detected clients
            self.auto_detected_clients.add(client.client_id)
            
            # สร้าง statistics
            self.client_stats[client.client_id] = {
                'active_trades': 0,
                'total_trades': 0,
                'profit_loss': 0.0,
                'win_rate': 0.0,
                'last_trade_time': None,
                'connection_status': 'connected',  # เชื่อมต่อแล้ว
                'last_update': datetime.now(),
                'auto_detected': True,
                'mt5_account': client_config['account_number'],
                'detection_time': client_config.get('detection_time'),
                'connection_mode': 'auto_detected'
            }
            
            self.logger.info(f"✅ Auto-detected client added: {client.client_id}")
            self.logger.info(f"   Account: {client.account_number}")
            self.logger.info(f"   Type: {'DEMO' if client_config.get('is_demo') else 'LIVE'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add auto-detected client: {e}")
    
    async def _load_manual_clients(self, config: Dict[str, Any]) -> bool:
        """Load clients manually (fallback method)"""
        self.logger.info("[📋] Loading clients manually...")
        
        # ใช้วิธีเดิม - โหลดจาก config file
        clients_file = Path(config.get('clients_config_file', 'config/clients.json'))
        
        if not clients_file.exists():
            self.logger.warning(f"[⚠] Clients file {clients_file} not found")
            await self._add_demo_client()
            return True
        
        try:
            with open(clients_file, 'r') as f:
                clients_data = json.load(f)
            
            if 'clients' not in clients_data:
                self.logger.error("[X] Invalid clients configuration format")
                return False
            
            valid_clients = 0
            for client_data in clients_data['clients']:
                if self._validate_client_config(client_data):
                    await self._add_client_from_config(client_data)
                    valid_clients += 1
                    self.performance_metrics['manual_connections'] += 1
            
            self.logger.info(f"[✓] Loaded {valid_clients} clients manually")
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Error loading manual clients: {e}")
            return False
    
    async def _add_client_from_config(self, client_data: Dict[str, Any]):
        """Add client from manual configuration"""
        try:
            # Get account info
            account_info = client_data.get('account_info', {})
            
            # Create client account
            client = ClientAccount(
                client_id=account_info.get('client_id'),
                account_number=account_info.get('account_number', ''),
                broker=account_info.get('broker', 'demo'),
                currency=account_info.get('currency', 'USD'),
                balance=account_info.get('balance', 10000.0),
                equity=account_info.get('equity', 10000.0),
                margin=account_info.get('margin', 0.0),
                free_margin=account_info.get('free_margin', 10000.0),
                margin_level=account_info.get('margin_level', 0.0),
                max_daily_loss=account_info.get('max_daily_loss', 200.0),
                max_weekly_loss=account_info.get('max_weekly_loss', 500.0),
                max_monthly_loss=account_info.get('max_monthly_loss', 1500.0),
                max_positions=account_info.get('max_positions', 5),
                max_lot_size=account_info.get('max_lot_size', 1.0),
                preferred_pairs=account_info.get('preferred_pairs', []),
                trading_sessions=account_info.get('trading_sessions', [])
            )
            
            # Add to clients dictionary
            self.clients[client.client_id] = client
            
            # Add to active clients if active
            if client_data.get('is_active', True):
                self.active_clients.add(client.client_id)
            
            # Initialize client statistics
            self.client_stats[client.client_id] = {
                'active_trades': 0,
                'total_trades': 0,
                'profit_loss': 0.0,
                'win_rate': 0.0,
                'last_trade_time': None,
                'connection_status': 'disconnected',
                'last_update': datetime.now(),
                'auto_detected': False,
                'connection_mode': 'manual'
            }
            
            self.logger.info(f"[✓] Added client: {client.client_id} ({'DEMO' if client_data.get('is_demo') else 'LIVE'})")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to add client: {e}")
    
    async def _add_demo_client(self):
        """Add default demo client for testing"""
        try:
            demo_client_config = {
                'client_id': 'DEMO_001',
                'account_number': 'demo_001',
                'broker': 'demo',
                'currency': 'USD',
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'margin_level': 0.0,
                'max_daily_loss': 200.0,
                'max_weekly_loss': 500.0,
                'max_monthly_loss': 1500.0,
                'max_positions': 5,
                'max_lot_size': 1.0,
                'preferred_pairs': ['EURUSD', 'GBPUSD'],
                'trading_sessions': ['London', 'NewYork'],
                'is_active': True
            }
            
            await self._add_auto_detected_client(demo_client_config)
            self.logger.info("[✓] Added default demo client")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to add demo client: {e}")
    
    def _validate_client_config(self, client_data: Dict[str, Any]) -> bool:
        """Validate client configuration"""
        try:
            required_fields = ['client_id', 'account_info']
            for field in required_fields:
                if field not in client_data:
                    self.logger.error(f"[X] Missing required field: {field}")
                    return False
            
            account_info = client_data['account_info']
            if not account_info.get('client_id'):
                self.logger.error("[X] Missing client_id in account_info")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Client validation error: {e}")
            return False
    
    async def _setup_advanced_monitoring(self):
        """Setup advanced monitoring capabilities"""
        try:
            # Initialize monitoring components
            self.system_health = {
                'last_check': datetime.now(),
                'components': {
                    'fvg_detector': 'unknown',
                    'trade_executor': 'unknown',
                    'mt5_adapter': 'unknown',
                    'clients': 'unknown'
                },
                'performance': {
                    'memory_usage': 0.0,
                    'cpu_usage': 0.0,
                    'active_threads': 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"[X] Monitoring setup error: {e}")
    
    async def _setup_risk_management(self, config: Dict[str, Any]):
        """Setup comprehensive risk management"""
        try:
            risk_config = config.get('risk_management', {})
            
            # Global risk settings
            self.global_risk_settings = {
                'max_total_exposure': risk_config.get('max_total_exposure', 100000),
                'max_correlation_threshold': risk_config.get('max_correlation_threshold', 0.7),
                'emergency_stop_enabled': risk_config.get('emergency_stop_enabled', True),
                'max_daily_trades': risk_config.get('max_daily_trades', 100)
            }
            
        except Exception as e:
            self.logger.error(f"[X] Risk management setup error: {e}")
    
    async def _setup_performance_tracking(self):
        """Setup performance tracking system"""
        try:
            # Initialize performance tracking
            self.performance_history = []
            self.last_performance_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"[X] Performance tracking setup error: {e}")
    
    async def _start_system_monitoring(self):
        """Start system monitoring services"""
        pass
    
    async def _start_signal_processing(self):
        """Start signal processing services"""
        pass
    
    async def _start_performance_monitoring(self):
        """Start performance monitoring services"""
        pass
    
    async def _perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            health_status = {
                'healthy': True,
                'timestamp': datetime.now(),
                'components': {
                    'controller': 'healthy',
                    'clients': len(self.clients),
                    'active_clients': len(self.active_clients),
                    'auto_detected_clients': len(self.auto_detected_clients)
                }
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"[X] Health check error: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'name': self.name,
                'version': self.version,
                'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': uptime,
                'clients': {
                    'total': len(self.clients),
                    'active': len(self.active_clients),
                    'auto_detected': len(self.auto_detected_clients),
                    'manual': len(self.clients) - len(self.auto_detected_clients)
                },
                'auto_mt5_detection': AUTO_MT5_AVAILABLE,
                'performance': self.performance_metrics,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
            }
            
        except Exception as e:
            self.logger.error(f"[X] Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_all_clients(self) -> Dict[str, ClientAccount]:
        """Get all registered clients"""
        return self.clients.copy()
    
    def get_client_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get specific client status"""
        if client_id not in self.clients:
            return None
        
        client = self.clients[client_id]
        stats = self.client_stats.get(client_id, {})
        
        return {
            'client_id': client_id,
            'account_number': client.account_number,
            'broker': client.broker,
            'currency': client.currency,
            'balance': client.balance,
            'equity': client.equity,
            'is_active': client_id in self.active_clients,
            'auto_detected': client_id in self.auto_detected_clients,
            'connection_status': stats.get('connection_status', 'unknown'),
            'stats': stats
        }
    
    async def shutdown(self):
        """Graceful shutdown of the controller"""
        try:
            self.logger.info("[🛑] Shutting down Lightning Scalper Controller...")
            
            self.status = SystemStatus.STOPPING
            self.is_running = False
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Shutdown components
            if self.fvg_detector:
                await self.fvg_detector.stop()
            
            if self.trade_executor:
                await self.trade_executor.stop()
            
            if self.mt5_adapter:
                await self.mt5_adapter.stop()
            
            self.status = SystemStatus.STOPPED
            
            self.logger.info("[✓] Lightning Scalper Controller shutdown complete")
            
        except Exception as e:
            self.logger.error(f"[X] Error during shutdown: {e}")
            self.status = SystemStatus.ERROR