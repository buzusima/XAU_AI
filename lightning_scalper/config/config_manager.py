#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Windows Safe Version
Auto-fixed for Unicode compatibility
"""

"""
[ROCKET] Lightning Scalper - Configuration Manager
Production-Grade Configuration Management System

Handles all system configurations including:
- Multi-broker settings
- Client configurations
- Trading parameters
- Security management
- Environment-specific configs
- Dynamic configuration updates

Author: Phoenix Trading AI
Version: 1.0.0
License: Proprietary
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sys
import copy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

class ConfigType(Enum):
    SYSTEM = "system"
    CLIENT = "client"
    BROKER = "broker"
    TRADING = "trading"
    SECURITY = "security"
    DASHBOARD = "dashboard"
    LOGGING = "logging"

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class BrokerConfig:
    """Broker-specific configuration"""
    broker_id: str
    name: str
    broker_type: str  # MT4, MT5, cTrader, etc.
    server_list: List[str]
    api_config: Dict[str, Any] = field(default_factory=dict)
    trading_hours: Dict[str, Any] = field(default_factory=dict)
    symbol_config: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, Any] = field(default_factory=dict)
    commission_structure: Dict[str, Any] = field(default_factory=dict)
    connection_params: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ClientConfig:
    """Individual client configuration"""
    client_id: str
    name: str
    email: str
    broker_id: str
    account_number: str
    
    # Trading settings
    auto_trading: bool = True
    preferred_pairs: List[str] = field(default_factory=list)
    risk_multiplier: float = 1.0
    max_signals_per_hour: int = 10
    max_positions: int = 5
    max_lot_size: float = 1.0
    
    # Risk management
    max_daily_loss: float = 200.0
    max_weekly_loss: float = 500.0
    max_monthly_loss: float = 1500.0
    
    # Account information
    account_currency: str = "USD"
    initial_balance: float = 10000.0
    leverage: int = 100
    
    # Notification settings
    email_notifications: bool = True
    telegram_notifications: bool = False
    telegram_chat_id: Optional[str] = None
    
    # Security
    api_key: Optional[str] = None
    encrypted_password: Optional[str] = None
    
    # Status
    is_active: bool = True
    is_demo: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    strategy_name: str = "Lightning Scalper FVG"
    
    # Signal parameters
    min_confluence_score: float = 65.0
    max_signals_per_session: int = 20
    signal_expiry_minutes: int = 30
    
    # Risk management
    default_risk_per_trade: float = 0.02
    max_slippage_pips: float = 3.0
    emergency_stop_loss_percent: float = 10.0
    
    # Timeframes and pairs
    active_timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "H1"])
    active_currency_pairs: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"])
    
    # Session settings
    trading_sessions: Dict[str, Dict] = field(default_factory=dict)
    
    # Advanced settings
    use_advanced_confluence: bool = True
    enable_news_filter: bool = True
    enable_spread_filter: bool = True
    max_spread_pips: float = 2.0

@dataclass
class SystemConfig:
    """Core system configuration"""
    # Application settings
    app_name: str = "Lightning Scalper"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    
    # Performance settings
    max_clients: int = 100
    max_concurrent_trades: int = 200
    data_update_interval: float = 1.0
    signal_generation_interval: float = 5.0
    health_check_interval: float = 30.0
    
    # Database settings
    database_path: str = "data/lightning_scalper.db"
    backup_interval_hours: int = 6
    data_retention_days: int = 365
    
    # Logging settings
    log_level: str = "INFO"
    log_file_path: str = "logs/lightning_scalper.log"
    max_log_file_size_mb: int = 100
    log_retention_days: int = 30
    
    # Security settings
    encryption_enabled: bool = True
    session_timeout_minutes: int = 60
    max_failed_logins: int = 3
    
    # API settings
    api_rate_limit: int = 1000  # requests per hour
    api_timeout_seconds: int = 30

class LightningScalperConfigManager:
    """
    [TOOL] Lightning Scalper Configuration Manager
    Centralized configuration management for production deployment
    """
    
    def __init__(self, config_dir: str = "config", encryption_key: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration storage
        self.configs: Dict[ConfigType, Any] = {}
        self.clients: Dict[str, ClientConfig] = {}
        self.brokers: Dict[str, BrokerConfig] = {}
        
        # File paths
        self.system_config_file = self.config_dir / "system.json"
        self.trading_config_file = self.config_dir / "trading.json"
        self.clients_config_file = self.config_dir / "clients.json"
        self.brokers_config_file = self.config_dir / "brokers.json"
        self.security_config_file = self.config_dir / "security.json"
        
        # Environment file
        self.env_file = self.config_dir / f".env.{self._detect_environment()}"
        
        # Encryption setup
        self.encryption_key = encryption_key
        self.cipher_suite = None
        if encryption_key:
            self._setup_encryption(encryption_key)
        
        # Change tracking
        self.config_watchers: Dict[str, List[Callable]] = {}
        self.file_timestamps: Dict[Path, float] = {}
        self.watch_thread: Optional[threading.Thread] = None
        self.is_watching = False
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger('ConfigManager')
        
        # Initialize configurations
        self._initialize_default_configs()
        self.load_all_configurations()
        
        self.logger.info("[TOOL] Configuration Manager initialized")
    
    def _detect_environment(self) -> str:
        """Detect current environment"""
        env = os.getenv('LIGHTNING_ENVIRONMENT', 'development').lower()
        return env if env in ['development', 'staging', 'production', 'testing'] else 'development'
    
    def _setup_encryption(self, password: str):
        """Setup encryption for sensitive data"""
        try:
            password_bytes = password.encode()
            salt = b'lightning_scalper_salt_2024'  # In production, use random salt
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self.cipher_suite = Fernet(key)
            
            self.logger.info("? Encryption enabled for sensitive data")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to setup encryption: {e}")
            self.cipher_suite = None
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.cipher_suite:
            try:
                encrypted = self.cipher_suite.encrypt(data.encode())
                return base64.urlsafe_b64encode(encrypted).decode()
            except Exception as e:
                self.logger.error(f"Encryption error: {e}")
                return data
        return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.cipher_suite:
            try:
                encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
                decrypted = self.cipher_suite.decrypt(encrypted_bytes)
                return decrypted.decode()
            except Exception as e:
                self.logger.error(f"Decryption error: {e}")
                return encrypted_data
        return encrypted_data
    
    def _initialize_default_configs(self):
        """Initialize default configurations"""
        # System configuration
        self.configs[ConfigType.SYSTEM] = SystemConfig()
        
        # Trading configuration
        self.configs[ConfigType.TRADING] = TradingConfig()
        
        # Set trading sessions
        self.configs[ConfigType.TRADING].trading_sessions = {
            'Sydney': {
                'start_hour': 22, 'end_hour': 6,
                'volatility_multiplier': 0.8, 'is_major': False
            },
            'Tokyo': {
                'start_hour': 0, 'end_hour': 8,
                'volatility_multiplier': 1.1, 'is_major': True
            },
            'London': {
                'start_hour': 8, 'end_hour': 16,
                'volatility_multiplier': 1.3, 'is_major': True
            },
            'NewYork': {
                'start_hour': 13, 'end_hour': 21,
                'volatility_multiplier': 1.2, 'is_major': True
            }
        }
    
    def load_all_configurations(self):
        """Load all configuration files"""
        try:
            # Load system config
            self.load_system_config()
            
            # Load trading config
            self.load_trading_config()
            
            # Load clients and brokers
            self.load_clients_config()
            self.load_brokers_config()
            
            # Load environment variables
            self.load_environment_config()
            
            self.logger.info("[CHECK] All configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"[X] Error loading configurations: {e}")
    
    def load_system_config(self) -> SystemConfig:
        """Load system configuration"""
        try:
            if self.system_config_file.exists():
                with open(self.system_config_file, 'r') as f:
                    data = json.load(f)
                
                # Convert environment string to enum
                if 'environment' in data:
                    data['environment'] = Environment(data['environment'])
                
                self.configs[ConfigType.SYSTEM] = SystemConfig(**data)
                self.logger.info("[CHECK] System configuration loaded")
            else:
                # Create default config file
                self.save_system_config()
                self.logger.info("[MEMO] Created default system configuration")
            
            return self.configs[ConfigType.SYSTEM]
            
        except Exception as e:
            self.logger.error(f"[X] Error loading system config: {e}")
            return self.configs[ConfigType.SYSTEM]
    
    def save_system_config(self):
        """Save system configuration"""
        try:
            config = self.configs[ConfigType.SYSTEM]
            data = asdict(config)
            
            # Convert enum to string
            data['environment'] = config.environment.value
            
            with open(self.system_config_file, 'w') as f:
                json.dump(data, f, indent=4, default=str)
            
            self.logger.info("? System configuration saved")
            
        except Exception as e:
            self.logger.error(f"[X] Error saving system config: {e}")
    
    def load_trading_config(self) -> TradingConfig:
        """Load trading configuration"""
        try:
            if self.trading_config_file.exists():
                with open(self.trading_config_file, 'r') as f:
                    data = json.load(f)
                
                self.configs[ConfigType.TRADING] = TradingConfig(**data)
                self.logger.info("[CHECK] Trading configuration loaded")
            else:
                # Create default config file
                self.save_trading_config()
                self.logger.info("[MEMO] Created default trading configuration")
            
            return self.configs[ConfigType.TRADING]
            
        except Exception as e:
            self.logger.error(f"[X] Error loading trading config: {e}")
            return self.configs[ConfigType.TRADING]
    
    def save_trading_config(self):
        """Save trading configuration"""
        try:
            config = self.configs[ConfigType.TRADING]
            data = asdict(config)
            
            with open(self.trading_config_file, 'w') as f:
                json.dump(data, f, indent=4, default=str)
            
            self.logger.info("? Trading configuration saved")
            
        except Exception as e:
            self.logger.error(f"[X] Error saving trading config: {e}")
    
    def load_clients_config(self):
        """Load clients configuration"""
        try:
            if self.clients_config_file.exists():
                with open(self.clients_config_file, 'r') as f:
                    data = json.load(f)
                
                self.clients.clear()
                
                for client_data in data.get('clients', []):
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'updated_at', 'last_login']:
                        if field in client_data and client_data[field]:
                            client_data[field] = datetime.fromisoformat(client_data[field])
                    
                    # Decrypt sensitive data
                    if client_data.get('encrypted_password'):
                        client_data['encrypted_password'] = self._decrypt_data(
                            client_data['encrypted_password']
                        )
                    
                    client = ClientConfig(**client_data)
                    self.clients[client.client_id] = client
                
                self.logger.info(f"[CHECK] Loaded {len(self.clients)} client configurations")
            else:
                # Create sample clients file
                self._create_sample_clients_config()
                
        except Exception as e:
            self.logger.error(f"[X] Error loading clients config: {e}")
    
    def save_clients_config(self):
        """Save clients configuration"""
        try:
            clients_data = []
            
            for client in self.clients.values():
                data = asdict(client)
                
                # Convert datetime objects to strings
                for field in ['created_at', 'updated_at', 'last_login']:
                    if data[field]:
                        data[field] = data[field].isoformat()
                
                # Encrypt sensitive data
                if data.get('encrypted_password'):
                    data['encrypted_password'] = self._encrypt_data(data['encrypted_password'])
                
                clients_data.append(data)
            
            config_data = {
                'clients': clients_data,
                'metadata': {
                    'total_clients': len(clients_data),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
            }
            
            with open(self.clients_config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.logger.info(f"? Saved {len(clients_data)} client configurations")
            
        except Exception as e:
            self.logger.error(f"[X] Error saving clients config: {e}")
    
    def load_brokers_config(self):
        """Load brokers configuration"""
        try:
            if self.brokers_config_file.exists():
                with open(self.brokers_config_file, 'r') as f:
                    data = json.load(f)
                
                self.brokers.clear()
                
                for broker_data in data.get('brokers', []):
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'updated_at']:
                        if field in broker_data and broker_data[field]:
                            broker_data[field] = datetime.fromisoformat(broker_data[field])
                    
                    broker = BrokerConfig(**broker_data)
                    self.brokers[broker.broker_id] = broker
                
                self.logger.info(f"[CHECK] Loaded {len(self.brokers)} broker configurations")
            else:
                # Create sample brokers file
                self._create_sample_brokers_config()
                
        except Exception as e:
            self.logger.error(f"[X] Error loading brokers config: {e}")
    
    def save_brokers_config(self):
        """Save brokers configuration"""
        try:
            brokers_data = []
            
            for broker in self.brokers.values():
                data = asdict(broker)
                
                # Convert datetime objects to strings
                for field in ['created_at', 'updated_at']:
                    if data[field]:
                        data[field] = data[field].isoformat()
                
                brokers_data.append(data)
            
            config_data = {
                'brokers': brokers_data,
                'metadata': {
                    'total_brokers': len(brokers_data),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
            }
            
            with open(self.brokers_config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.logger.info(f"? Saved {len(brokers_data)} broker configurations")
            
        except Exception as e:
            self.logger.error(f"[X] Error saving brokers config: {e}")
    
    def load_environment_config(self):
        """Load environment-specific configuration"""
        try:
            if self.env_file.exists():
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                
                self.logger.info(f"[CHECK] Environment configuration loaded from {self.env_file.name}")
            else:
                # Create default environment file
                self._create_default_env_file()
                
        except Exception as e:
            self.logger.error(f"[X] Error loading environment config: {e}")
    
    def _create_sample_clients_config(self):
        """Create sample clients configuration"""
        try:
            sample_clients = [
                ClientConfig(
                    client_id="SAMPLE_CLIENT_001",
                    name="Demo Client 001",
                    email="demo1@example.com",
                    broker_id="BROKER_MT5_DEMO",
                    account_number="12345001",
                    preferred_pairs=["EURUSD", "GBPUSD", "USDJPY"],
                    risk_multiplier=1.0,
                    max_daily_loss=200.0,
                    is_demo=True
                ),
                ClientConfig(
                    client_id="SAMPLE_CLIENT_002",
                    name="Demo Client 002",
                    email="demo2@example.com",
                    broker_id="BROKER_MT5_DEMO",
                    account_number="12345002",
                    preferred_pairs=["EURUSD", "XAUUSD"],
                    risk_multiplier=1.5,
                    max_daily_loss=500.0,
                    initial_balance=25000.0,
                    is_demo=True
                )
            ]
            
            for client in sample_clients:
                self.clients[client.client_id] = client
            
            self.save_clients_config()
            self.logger.info("[MEMO] Created sample clients configuration")
            
        except Exception as e:
            self.logger.error(f"[X] Error creating sample clients config: {e}")
    
    def _create_sample_brokers_config(self):
        """Create sample brokers configuration"""
        try:
            sample_brokers = [
                BrokerConfig(
                    broker_id="BROKER_MT5_DEMO",
                    name="MetaTrader 5 Demo",
                    broker_type="MT5",
                    server_list=["DemoServer-MT5", "BackupDemo-MT5"],
                    symbol_config={
                        "EURUSD": {"min_lot": 0.01, "max_lot": 100, "step": 0.01},
                        "GBPUSD": {"min_lot": 0.01, "max_lot": 100, "step": 0.01},
                        "USDJPY": {"min_lot": 0.01, "max_lot": 100, "step": 0.01},
                        "XAUUSD": {"min_lot": 0.01, "max_lot": 10, "step": 0.01}
                    },
                    commission_structure={
                        "type": "spread_only",
                        "typical_spreads": {
                            "EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 0.9, "XAUUSD": 3.5
                        }
                    }
                ),
                BrokerConfig(
                    broker_id="BROKER_MT5_LIVE",
                    name="MetaTrader 5 Live",
                    broker_type="MT5",
                    server_list=["LiveServer-MT5", "BackupLive-MT5"],
                    symbol_config={
                        "EURUSD": {"min_lot": 0.01, "max_lot": 500, "step": 0.01},
                        "GBPUSD": {"min_lot": 0.01, "max_lot": 500, "step": 0.01},
                        "USDJPY": {"min_lot": 0.01, "max_lot": 500, "step": 0.01},
                        "XAUUSD": {"min_lot": 0.01, "max_lot": 50, "step": 0.01}
                    },
                    commission_structure={
                        "type": "commission_plus_spread",
                        "commission_per_lot": 7.0,
                        "typical_spreads": {
                            "EURUSD": 0.6, "GBPUSD": 0.9, "USDJPY": 0.7, "XAUUSD": 2.8
                        }
                    }
                )
            ]
            
            for broker in sample_brokers:
                self.brokers[broker.broker_id] = broker
            
            self.save_brokers_config()
            self.logger.info("[MEMO] Created sample brokers configuration")
            
        except Exception as e:
            self.logger.error(f"[X] Error creating sample brokers config: {e}")
    
    def _create_default_env_file(self):
        """Create default environment file"""
        try:
            env_content = f"""# Lightning Scalper Environment Configuration
# Environment: {self._detect_environment()}

# Database
DATABASE_URL=sqlite:///data/lightning_scalper.db
DATABASE_POOL_SIZE=20

# Security
ENCRYPTION_ENABLED=true
SESSION_TIMEOUT=3600
MAX_FAILED_LOGINS=3

# API Settings
API_RATE_LIMIT=1000
API_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE_MAX_SIZE=100MB
LOG_RETENTION_DAYS=30

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_DEBUG=false

# Trading
MAX_CONCURRENT_TRADES=200
SIGNAL_GENERATION_INTERVAL=5
DATA_UPDATE_INTERVAL=1

# Notifications
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ALERTS_ENABLED=false

# Monitoring
HEALTH_CHECK_INTERVAL=30
PERFORMANCE_MONITORING=true
BACKUP_INTERVAL_HOURS=6

# Development settings (only for development environment)
DEBUG_MODE=false
MOCK_TRADING=false
FAST_SIGNALS=false
"""
            
            with open(self.env_file, 'w') as f:
                f.write(env_content)
            
            self.logger.info(f"[MEMO] Created default environment file: {self.env_file.name}")
            
        except Exception as e:
            self.logger.error(f"[X] Error creating environment file: {e}")
    
    def add_client(self, client_config: ClientConfig) -> bool:
        """Add new client configuration"""
        try:
            with self.lock:
                # Validate client data
                if not self._validate_client_config(client_config):
                    return False
                
                # Check if client already exists
                if client_config.client_id in self.clients:
                    self.logger.warning(f"Client {client_config.client_id} already exists")
                    return False
                
                # Check if broker exists
                if client_config.broker_id not in self.brokers:
                    self.logger.error(f"Broker {client_config.broker_id} not found")
                    return False
                
                # Add client
                self.clients[client_config.client_id] = client_config
                
                # Save to file
                self.save_clients_config()
                
                self.logger.info(f"[CHECK] Added client: {client_config.client_id}")
                
                # Trigger watchers
                self._trigger_config_watchers('client_added', client_config)
                
                return True
                
        except Exception as e:
            self.logger.error(f"[X] Error adding client: {e}")
            return False
    
    def update_client(self, client_id: str, updates: Dict[str, Any]) -> bool:
        """Update client configuration"""
        try:
            with self.lock:
                if client_id not in self.clients:
                    self.logger.error(f"Client {client_id} not found")
                    return False
                
                client = self.clients[client_id]
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(client, key):
                        setattr(client, key, value)
                
                # Update timestamp
                client.updated_at = datetime.now()
                
                # Save to file
                self.save_clients_config()
                
                self.logger.info(f"[CHECK] Updated client: {client_id}")
                
                # Trigger watchers
                self._trigger_config_watchers('client_updated', client)
                
                return True
                
        except Exception as e:
            self.logger.error(f"[X] Error updating client: {e}")
            return False
    
    def remove_client(self, client_id: str) -> bool:
        """Remove client configuration"""
        try:
            with self.lock:
                if client_id not in self.clients:
                    self.logger.error(f"Client {client_id} not found")
                    return False
                
                client = self.clients[client_id]
                del self.clients[client_id]
                
                # Save to file
                self.save_clients_config()
                
                self.logger.info(f"[CHECK] Removed client: {client_id}")
                
                # Trigger watchers
                self._trigger_config_watchers('client_removed', client)
                
                return True
                
        except Exception as e:
            self.logger.error(f"[X] Error removing client: {e}")
            return False
    
    def add_broker(self, broker_config: BrokerConfig) -> bool:
        """Add new broker configuration"""
        try:
            with self.lock:
                # Validate broker data
                if not self._validate_broker_config(broker_config):
                    return False
                
                # Check if broker already exists
                if broker_config.broker_id in self.brokers:
                    self.logger.warning(f"Broker {broker_config.broker_id} already exists")
                    return False
                
                # Add broker
                self.brokers[broker_config.broker_id] = broker_config
                
                # Save to file
                self.save_brokers_config()
                
                self.logger.info(f"[CHECK] Added broker: {broker_config.broker_id}")
                
                # Trigger watchers
                self._trigger_config_watchers('broker_added', broker_config)
                
                return True
                
        except Exception as e:
            self.logger.error(f"[X] Error adding broker: {e}")
            return False
    
    def _validate_client_config(self, client: ClientConfig) -> bool:
        """Validate client configuration"""
        if not client.client_id or not client.name or not client.email:
            self.logger.error("Client ID, name, and email are required")
            return False
        
        if not client.broker_id or not client.account_number:
            self.logger.error("Broker ID and account number are required")
            return False
        
        if client.risk_multiplier <= 0 or client.risk_multiplier > 5:
            self.logger.error("Risk multiplier must be between 0 and 5")
            return False
        
        return True
    
    def _validate_broker_config(self, broker: BrokerConfig) -> bool:
        """Validate broker configuration"""
        if not broker.broker_id or not broker.name:
            self.logger.error("Broker ID and name are required")
            return False
        
        if not broker.server_list:
            self.logger.error("At least one server must be specified")
            return False
        
        return True
    
    def get_client_config(self, client_id: str) -> Optional[ClientConfig]:
        """Get client configuration"""
        return self.clients.get(client_id)
    
    def get_broker_config(self, broker_id: str) -> Optional[BrokerConfig]:
        """Get broker configuration"""
        return self.brokers.get(broker_id)
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.configs[ConfigType.SYSTEM]
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.configs[ConfigType.TRADING]
    
    def get_all_clients(self) -> Dict[str, ClientConfig]:
        """Get all client configurations"""
        return self.clients.copy()
    
    def get_all_brokers(self) -> Dict[str, BrokerConfig]:
        """Get all broker configurations"""
        return self.brokers.copy()
    
    def get_active_clients(self) -> Dict[str, ClientConfig]:
        """Get only active client configurations"""
        return {cid: client for cid, client in self.clients.items() if client.is_active}
    
    def get_clients_by_broker(self, broker_id: str) -> Dict[str, ClientConfig]:
        """Get clients for specific broker"""
        return {cid: client for cid, client in self.clients.items() 
                if client.broker_id == broker_id}
    
    def start_config_watcher(self):
        """Start watching configuration files for changes"""
        if not self.is_watching:
            self.is_watching = True
            self.watch_thread = threading.Thread(target=self._config_watch_loop, daemon=True)
            self.watch_thread.start()
            self.logger.info("?? Configuration file watcher started")
    
    def stop_config_watcher(self):
        """Stop watching configuration files"""
        if self.is_watching:
            self.is_watching = False
            if self.watch_thread:
                self.watch_thread.join(timeout=5)
            self.logger.info("? Configuration file watcher stopped")
    
    def _config_watch_loop(self):
        """Background loop for watching config file changes"""
        config_files = [
            self.system_config_file,
            self.trading_config_file,
            self.clients_config_file,
            self.brokers_config_file
        ]
        
        # Initialize timestamps
        for file_path in config_files:
            if file_path.exists():
                self.file_timestamps[file_path] = file_path.stat().st_mtime
        
        while self.is_watching:
            try:
                for file_path in config_files:
                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        
                        if (file_path not in self.file_timestamps or 
                            current_mtime > self.file_timestamps[file_path]):
                            
                            self.file_timestamps[file_path] = current_mtime
                            self._reload_config_file(file_path)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in config watch loop: {e}")
                time.sleep(10)
    
    def _reload_config_file(self, file_path: Path):
        """Reload specific configuration file"""
        try:
            if file_path == self.system_config_file:
                self.load_system_config()
                self._trigger_config_watchers('system_config_reloaded', None)
                
            elif file_path == self.trading_config_file:
                self.load_trading_config()
                self._trigger_config_watchers('trading_config_reloaded', None)
                
            elif file_path == self.clients_config_file:
                self.load_clients_config()
                self._trigger_config_watchers('clients_config_reloaded', None)
                
            elif file_path == self.brokers_config_file:
                self.load_brokers_config()
                self._trigger_config_watchers('brokers_config_reloaded', None)
            
            self.logger.info(f"[REFRESH] Reloaded configuration: {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"[X] Error reloading {file_path.name}: {e}")
    
    def add_config_watcher(self, event_type: str, callback: Callable):
        """Add callback for configuration changes"""
        if event_type not in self.config_watchers:
            self.config_watchers[event_type] = []
        
        self.config_watchers[event_type].append(callback)
        self.logger.debug(f"Added config watcher for {event_type}")
    
    def _trigger_config_watchers(self, event_type: str, data: Any):
        """Trigger configuration change callbacks"""
        try:
            callbacks = self.config_watchers.get(event_type, [])
            for callback in callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"Error in config watcher callback: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering config watchers: {e}")
    
    def create_backup(self) -> str:
        """Create backup of all configuration files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.config_dir / "backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all config files
            config_files = [
                self.system_config_file,
                self.trading_config_file,
                self.clients_config_file,
                self.brokers_config_file,
                self.env_file
            ]
            
            for file_path in config_files:
                if file_path.exists():
                    backup_file = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_file)
            
            self.logger.info(f"? Configuration backup created: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            self.logger.error(f"[X] Error creating backup: {e}")
            return ""
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore configuration from backup"""
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_path}")
                return False
            
            # Restore config files
            config_files = [
                ("system.json", self.system_config_file),
                ("trading.json", self.trading_config_file),
                ("clients.json", self.clients_config_file),
                ("brokers.json", self.brokers_config_file)
            ]
            
            for backup_name, target_path in config_files:
                backup_file = backup_dir / backup_name
                if backup_file.exists():
                    shutil.copy2(backup_file, target_path)
            
            # Reload all configurations
            self.load_all_configurations()
            
            self.logger.info(f"[CHECK] Configuration restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Error restoring backup: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        return {
            'system': {
                'environment': self.configs[ConfigType.SYSTEM].environment.value,
                'version': self.configs[ConfigType.SYSTEM].version,
                'max_clients': self.configs[ConfigType.SYSTEM].max_clients,
                'debug_mode': self.configs[ConfigType.SYSTEM].debug_mode
            },
            'trading': {
                'strategy': self.configs[ConfigType.TRADING].strategy_name,
                'min_confluence': self.configs[ConfigType.TRADING].min_confluence_score,
                'active_pairs': len(self.configs[ConfigType.TRADING].active_currency_pairs),
                'active_timeframes': len(self.configs[ConfigType.TRADING].active_timeframes)
            },
            'clients': {
                'total': len(self.clients),
                'active': len(self.get_active_clients()),
                'demo': len([c for c in self.clients.values() if c.is_demo]),
                'live': len([c for c in self.clients.values() if not c.is_demo])
            },
            'brokers': {
                'total': len(self.brokers),
                'active': len([b for b in self.brokers.values() if b.is_active]),
                'types': list(set(b.broker_type for b in self.brokers.values()))
            }
        }

# Demo and testing
def run_demo():
    """Run demo of the configuration manager"""
    print("[TOOL] Lightning Scalper Configuration Manager - Demo")
    print("=" * 60)
    
    # Initialize config manager
    config_manager = LightningScalperConfigManager(config_dir="demo_config")
    
    try:
        # Show config summary
        summary = config_manager.get_config_summary()
        print(f"[CHART] Configuration Summary:")
        print(f"   System Environment: {summary['system']['environment']}")
        print(f"   Total Clients: {summary['clients']['total']}")
        print(f"   Active Clients: {summary['clients']['active']}")
        print(f"   Total Brokers: {summary['brokers']['total']}")
        print(f"   Trading Strategy: {summary['trading']['strategy']}")
        
        # Show client configurations
        print(f"\n[USERS] Client Configurations:")
        for client_id, client in config_manager.get_all_clients().items():
            print(f"   {client_id}: {client.name} ({client.broker_id})")
            print(f"      Active: {client.is_active}, Demo: {client.is_demo}")
            print(f"      Risk Multiplier: {client.risk_multiplier}")
            print(f"      Max Daily Loss: ${client.max_daily_loss}")
        
        # Show broker configurations
        print(f"\n? Broker Configurations:")
        for broker_id, broker in config_manager.get_all_brokers().items():
            print(f"   {broker_id}: {broker.name} ({broker.broker_type})")
            print(f"      Servers: {', '.join(broker.server_list)}")
            print(f"      Active: {broker.is_active}")
        
        # Test adding new client
        new_client = ClientConfig(
            client_id="TEST_CLIENT_999",
            name="Test Client",
            email="test@example.com",
            broker_id="BROKER_MT5_DEMO",
            account_number="99999999",
            risk_multiplier=0.5,
            max_daily_loss=100.0,
            is_demo=True
        )
        
        if config_manager.add_client(new_client):
            print(f"\n[CHECK] Successfully added test client: {new_client.client_id}")
        
        # Create backup
        backup_path = config_manager.create_backup()
        if backup_path:
            print(f"\n? Backup created: {backup_path}")
        
        print(f"\n[CHECK] Demo completed successfully!")
        
    except Exception as e:
        print(f"\n[X] Demo error: {e}")
    
    finally:
        config_manager.stop_config_watcher()

if __name__ == "__main__":
    run_demo()