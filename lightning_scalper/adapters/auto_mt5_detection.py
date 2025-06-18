#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[ROCKET] Lightning Scalper - Auto MT5 Detection
ตรวจจับการเชื่อมต่อ MT5 อัตโนมัติ - ไม่ต้องใส่ login/password

แนวคิดใหม่:
- ✅ ตรวจสอบ MT5 ที่เปิดอยู่แล้ว
- ✅ ใช้ connection ที่มีอยู่
- ✅ ไม่ต้องเก็บ credentials
- ✅ ปลอดภัยกว่า
- ✅ ใช้งานง่ายกว่า

Author: Phoenix Trading AI
Version: 1.0.0
License: Proprietary
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[⚠️] MetaTrader5 package not available")

class ConnectionMode(Enum):
    """MT5 Connection modes"""
    AUTO_DETECT = "auto_detect"
    EXISTING_CONNECTION = "existing_connection"
    NEW_CONNECTION = "new_connection"

@dataclass
class MT5AccountDetected:
    """Auto-detected MT5 account information"""
    login: int
    server: str
    name: str
    company: str
    currency: str
    balance: float
    equity: float
    is_demo: bool
    trade_allowed: bool
    connection_status: str

class AutoMT5Detector:
    """
    [🔍] Auto MT5 Connection Detector
    ตรวจจับและใช้การเชื่อมต่อ MT5 ที่มีอยู่แล้ว
    """
    
    def __init__(self):
        self.logger = logging.getLogger('AutoMT5Detector')
        self.detected_account: Optional[MT5AccountDetected] = None
        self.is_connected = False
        
    def detect_existing_connection(self) -> Optional[MT5AccountDetected]:
        """ตรวจจับการเชื่อมต่อ MT5 ที่มีอยู่แล้ว - FIXED VERSION"""
        
        if not MT5_AVAILABLE:
            self.logger.error("❌ MetaTrader5 package not available")
            return None
        
        try:
            self.logger.info("🔍 Checking for existing MT5 connection...")
            
            # Force initialize (แก้ไขปัญหาการตรวจจับ)
            if not mt5.initialize():
                self.logger.warning("⚠️ No MT5 connection found")
                return None
            
            # ดึงข้อมูลบัญชี
            account_info = mt5.account_info()
            if not account_info:
                self.logger.warning("⚠️ Could not retrieve account information")
                mt5.shutdown()
                return None
            
            # ดึงข้อมูล terminal
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                self.logger.warning("⚠️ Could not retrieve terminal information")
                mt5.shutdown()
                return None
            
            # สร้าง detected account object
            detected_account = MT5AccountDetected(
                login=account_info.login,
                server=account_info.server,
                name=account_info.name,
                company=account_info.company,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                is_demo="demo" in account_info.server.lower() or "demo" in account_info.company.lower(),
                trade_allowed=account_info.trade_allowed,
                connection_status="connected"
            )
            
            self.detected_account = detected_account
            self.is_connected = True
            
            self.logger.info("✅ MT5 Connection detected successfully!")
            self.logger.info(f"   Account: {detected_account.login}")
            self.logger.info(f"   Server: {detected_account.server}")
            self.logger.info(f"   Balance: ${detected_account.balance:.2f}")
            self.logger.info(f"   Type: {'DEMO' if detected_account.is_demo else 'LIVE'}")
            
            # Don't shutdown here - keep connection alive
            
            return detected_account
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting MT5 connection: {e}")
            try:
                mt5.shutdown()
            except:
                pass
            return None
        
    def verify_connection(self) -> bool:
        """ตรวจสอบว่าการเชื่อมต่อยังทำงานอยู่"""
        
        if not self.is_connected or not MT5_AVAILABLE:
            return False
        
        try:
            # ตรวจสอบ account info
            account_info = mt5.account_info()
            if not account_info:
                self.logger.warning("⚠️ Connection lost - account info unavailable")
                self.is_connected = False
                return False
            
            # ตรวจสอบ terminal info
            terminal_info = mt5.terminal_info()
            if not terminal_info or not terminal_info.connected:
                self.logger.warning("⚠️ Connection lost - terminal disconnected")
                self.is_connected = False
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection verification failed: {e}")
            self.is_connected = False
            return False
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """ดึงสรุปข้อมูลการเชื่อมต่อ"""
        
        if not self.detected_account:
            return {
                'status': 'not_connected',
                'message': 'No MT5 connection detected'
            }
        
        return {
            'status': 'connected',
            'account': {
                'login': self.detected_account.login,
                'server': self.detected_account.server,
                'name': self.detected_account.name,
                'currency': self.detected_account.currency,
                'balance': self.detected_account.balance,
                'equity': self.detected_account.equity,
                'is_demo': self.detected_account.is_demo,
                'trade_allowed': self.detected_account.trade_allowed
            },
            'connection_time': datetime.now().isoformat(),
            'verification_status': self.verify_connection()
        }
    
    def wait_for_mt5_connection(self, timeout_seconds: int = 60, check_interval: float = 2.0) -> Optional[MT5AccountDetected]:
        """รอให้ MT5 เชื่อมต่อ (สำหรับ auto-start)"""
        
        self.logger.info(f"⏱️ Waiting for MT5 connection (timeout: {timeout_seconds}s)...")
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            detected = self.detect_existing_connection()
            if detected:
                self.logger.info("✅ MT5 connection established!")
                return detected
            
            self.logger.debug(f"🔄 Checking again in {check_interval}s...")
            time.sleep(check_interval)
        
        self.logger.warning(f"⏱️ Timeout waiting for MT5 connection ({timeout_seconds}s)")
        return None

class SmartClientManager:
    """
    [👤] Smart Client Manager
    จัดการ client โดยใช้ MT5 connection ที่ตรวจพบ
    """
    
    def __init__(self):
        self.logger = logging.getLogger('SmartClientManager')
        self.mt5_detector = AutoMT5Detector()
        self.current_client: Optional[Dict[str, Any]] = None
    
    def create_client_from_mt5(self) -> Optional[Dict[str, Any]]:
        """สร้าง client configuration จาก MT5 ที่ตรวจพบ"""
        
        # ตรวจจับ MT5 connection
        detected_account = self.mt5_detector.detect_existing_connection()
        if not detected_account:
            self.logger.error("❌ No MT5 connection found")
            return None
        
        # สร้าง client configuration
        client_config = {
            'client_id': f"AUTO_{detected_account.login}",
            'name': f"Auto Client ({detected_account.name})",
            'account_number': str(detected_account.login),
            'broker': detected_account.company,
            'currency': detected_account.currency,
            'balance': detected_account.balance,
            'equity': detected_account.equity,
            'margin': 0.0,
            'free_margin': detected_account.equity,
            'margin_level': 0.0,
            'is_demo': detected_account.is_demo,
            'is_active': True,
            'auto_detected': True,
            'server': detected_account.server,
            'trade_allowed': detected_account.trade_allowed,
            
            # Default trading settings
            'max_daily_loss': 200.0,
            'max_weekly_loss': 500.0,
            'max_monthly_loss': 1500.0,
            'max_positions': 5,
            'max_lot_size': 1.0,
            'preferred_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
            'trading_sessions': ['London', 'NewYork'],
            'risk_per_trade': 0.02,  # 2%
            
            # Auto-detection metadata
            'detection_time': datetime.now().isoformat(),
            'connection_mode': 'auto_detected'
        }
        
        self.current_client = client_config
        
        self.logger.info("✅ Client created from MT5 connection!")
        self.logger.info(f"   Client ID: {client_config['client_id']}")
        self.logger.info(f"   Account: {client_config['account_number']}")
        self.logger.info(f"   Type: {'DEMO' if client_config['is_demo'] else 'LIVE'}")
        
        return client_config
    
    def get_client_for_trade_executor(self) -> Optional[Dict[str, Any]]:
        """ดึง client config ในรูปแบบที่ TradeExecutor ต้องการ"""
        
        if not self.current_client:
            client = self.create_client_from_mt5()
            if not client:
                return None
        else:
            client = self.current_client
        
        # แปลงเป็นรูปแบบที่ TradeExecutor ต้องการ
        trade_executor_config = {
            'client_id': client['client_id'],
            'account_number': client['account_number'],
            'broker': client['broker'],
            'currency': client['currency'],
            'balance': client['balance'],
            'equity': client['equity'],
            'margin': client['margin'],
            'free_margin': client['free_margin'],
            'margin_level': client['margin_level'],
            'max_daily_loss': client['max_daily_loss'],
            'max_weekly_loss': client['max_weekly_loss'],
            'max_monthly_loss': client['max_monthly_loss'],
            'max_positions': client['max_positions'],
            'max_lot_size': client['max_lot_size'],
            'preferred_pairs': client['preferred_pairs'],
            'trading_sessions': client['trading_sessions']
        }
        
        return trade_executor_config

# ================================
# INTEGRATION FUNCTIONS
# ================================

def get_auto_detected_client() -> Optional[Dict[str, Any]]:
    """หน้าที่หลักในการดึง client configuration อัตโนมัติ"""
    
    manager = SmartClientManager()
    return manager.get_client_for_trade_executor()

def check_mt5_status() -> Dict[str, Any]:
    """ตรวจสอบสถานะ MT5 connection - SIMPLE VERSION"""
    
    if not MT5_AVAILABLE:
        return {
            'status': 'not_connected',
            'message': 'MetaTrader5 package not available'
        }
    
    try:
        # ทดสอบ connection โดยตรง
        if not mt5.initialize():
            return {
                'status': 'not_connected',
                'message': 'MT5 initialize failed'
            }
        
        account_info = mt5.account_info()
        if not account_info:
            mt5.shutdown()
            return {
                'status': 'not_connected',
                'message': 'No account info available'
            }
        
        # Success!
        return {
            'status': 'connected',
            'account': {
                'login': account_info.login,
                'server': account_info.server,
                'name': account_info.name,
                'currency': account_info.currency,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'is_demo': 'demo' in account_info.server.lower(),
                'trade_allowed': account_info.trade_allowed
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error: {str(e)}'
        }
    
def wait_for_mt5_ready(timeout: int = 60) -> bool:
    """รอให้ MT5 พร้อมใช้งาน"""
    
    detector = AutoMT5Detector()
    result = detector.wait_for_mt5_connection(timeout)
    return result is not None

# ================================
# TESTING FUNCTIONS
# ================================

def test_auto_detection():
    """ทดสอบระบบตรวจจับอัตโนมัติ"""
    
    print("🧪 Testing Auto MT5 Detection")
    print("=" * 50)
    
    # Test 1: Check MT5 status
    print("\n1. Checking MT5 status...")
    status = check_mt5_status()
    print(f"   Status: {status['status']}")
    if status['status'] == 'connected':
        account = status['account']
        print(f"   Account: {account['login']}")
        print(f"   Server: {account['server']}")
        print(f"   Balance: ${account['balance']:.2f}")
        print(f"   Type: {'DEMO' if account['is_demo'] else 'LIVE'}")
    
    # Test 2: Create client config
    print("\n2. Creating client configuration...")
    client_config = get_auto_detected_client()
    if client_config:
        print("   ✅ Client config created successfully!")
        print(f"   Client ID: {client_config['client_id']}")
        print(f"   Currency: {client_config['currency']}")
        print(f"   Max Positions: {client_config['max_positions']}")
    else:
        print("   ❌ Failed to create client config")
    
    # Test 3: Connection verification
    print("\n3. Verifying connection...")
    detector = AutoMT5Detector()
    detector.detect_existing_connection()
    is_connected = detector.verify_connection()
    print(f"   Connection verified: {'✅' if is_connected else '❌'}")
    
    print("\n" + "=" * 50)
    print("🎯 Auto detection test completed!")

if __name__ == "__main__":
    test_auto_detection()