#!/usr/bin/env python3
"""
🚀 Lightning Scalper - Simple Client Example
Easy-to-use example for beginners

📍 ไฟล์นี้: client_tools/simple_client_example.py

This is a simplified example showing how to use Lightning Scalper
for beginners who want to get started quickly.

Features:
- Simple configuration
- Automatic signal handling
- Basic risk management
- Easy setup and run

Author: Phoenix Trading AI (อาจารย์ฟินิกซ์)
Version: 1.0.0
License: Proprietary

Usage:
1. Edit the configuration section below
2. Run: python simple_client_example.py
3. Let the system trade automatically
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

# Add client tools to path
CLIENT_TOOLS_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CLIENT_TOOLS_DIR))

try:
    from lightning_scalper_sdk import LightningScalperClient, TradingSignal, TradeResult, SignalType
    from lightning_scalper_mt5_ea import LightningScalperMT5EA
except ImportError as e:
    print("❌ Missing Lightning Scalper modules!")
    print("   Make sure lightning_scalper_sdk.py and lightning_scalper_mt5_ea.py are in the same directory")
    sys.exit(1)

# 🔧 CONFIGURATION - แก้ไขตรงนี้
# ================================================
CONFIG = {
    # ข้อมูลลูกค้า Lightning Scalper
    'client_id': 'YOUR_CLIENT_ID',          # ได้รับจาก IB
    'api_key': 'YOUR_API_KEY',              # ได้รับจาก IB  
    'api_secret': 'YOUR_API_SECRET',        # ได้รับจาก IB
    
    # ข้อมูล MetaTrader 5
    'mt5_login': 12345678,                  # เลข Account MT5
    'mt5_password': 'your_mt5_password',    # รหัสผ่าน MT5
    'mt5_server': 'YourBroker-Demo',        # Server MT5
    
    # การตั้งค่าการเทรด
    'risk_per_trade': 2.0,                  # ความเสี่ยงต่อออเดอร์ (%)
    'max_positions': 3,                     # จำนวนออเดอร์สูงสุดที่เปิดพร้อมกัน
    'max_daily_trades': 10,                 # จำนวนเทรดสูงสุดต่อวัน
    'daily_loss_limit': 200.0,              # ขาดทุนสูงสุดต่อวัน ($)
    
    # เซิร์ฟเวอร์ (Demo/Live)
    'use_demo': True,                       # True = Demo, False = Live
}

class SimpleLightningClient:
    """
    🚀 Lightning Scalper Simple Client
    ลูกค้าแบบง่าย ๆ สำหรับมือใหม่
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lightning_scalper_simple.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SimpleLightningClient')
        
        # Determine server URLs
        if config['use_demo']:
            server_url = "ws://demo.lightning-scalper.com:8080"
            http_url = "https://demo.lightning-scalper.com"
        else:
            server_url = "wss://live.lightning-scalper.com:8080"  
            http_url = "https://live.lightning-scalper.com"
        
        # Create MT5 EA
        self.ea = LightningScalperMT5EA(
            client_id=config['client_id'],
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            mt5_login=config['mt5_login'],
            mt5_password=config['mt5_password'],
            mt5_server=config['mt5_server'],
            server_url=server_url,
            http_url=http_url
        )
        
        # Apply user settings
        self.ea.risk_per_trade = config['risk_per_trade'] / 100.0
        self.ea.max_positions = config['max_positions'] 
        self.ea.max_daily_trades = config['max_daily_trades']
        self.ea.max_daily_loss = -abs(config['daily_loss_limit'])
        
        # Statistics
        self.start_time = None
        self.stats = {
            'runtime_hours': 0,
            'signals_received': 0,
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }
        
        self.logger.info("🚀 Simple Lightning Scalper Client initialized")
        self.logger.info(f"   Client ID: {config['client_id']}")
        self.logger.info(f"   Environment: {'DEMO' if config['use_demo'] else 'LIVE'}")
        self.logger.info(f"   Risk per trade: {config['risk_per_trade']}%")
        self.logger.info(f"   Max positions: {config['max_positions']}")
    
    def start(self):
        """เริ่มต้นระบบเทรด"""
        try:
            self.logger.info("🚀 Starting Lightning Scalper...")
            
            # Validate configuration
            if not self._validate_config():
                return False
            
            # Start EA
            if not self.ea.start():
                self.logger.error("❌ Failed to start trading system")
                return False
            
            self.start_time = datetime.now()
            self.logger.info("✅ Lightning Scalper started successfully!")
            self.logger.info("🎯 System is now listening for signals...")
            
            # Show initial status
            self._show_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start: {e}")
            return False
    
    def stop(self):
        """หยุดระบบเทรด"""
        try:
            self.logger.info("🛑 Stopping Lightning Scalper...")
            self.ea.stop()
            self.logger.info("✅ Lightning Scalper stopped")
            
            # Show final summary
            self._show_summary()
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping: {e}")
    
    def _validate_config(self) -> bool:
        """ตรวจสอบการตั้งค่า"""
        required_fields = ['client_id', 'api_key', 'api_secret', 'mt5_login', 'mt5_password']
        
        for field in required_fields:
            if not self.config.get(field) or self.config[field] in ['YOUR_CLIENT_ID', 'YOUR_API_KEY', 'YOUR_API_SECRET']:
                self.logger.error(f"❌ Please configure {field} in the CONFIG section")
                return False
        
        if self.config['mt5_login'] == 12345678:
            self.logger.error("❌ Please configure your real MT5 login number")
            return False
        
        if self.config['mt5_password'] == 'your_mt5_password':
            self.logger.error("❌ Please configure your real MT5 password")
            return False
        
        return True
    
    def _show_status(self):
        """แสดงสถานะปัจจุบัน"""
        status = self.ea.get_status()
        
        print("\n" + "="*60)
        print("📊 LIGHTNING SCALPER STATUS")
        print("="*60)
        print(f"🟢 System Running: {status['ea_running']}")
        print(f"🔗 MT5 Connected: {status['mt5_connected']}")
        print(f"🌐 Server Connected: {status['lightning_connected']}")
        print(f"🤖 Auto Trading: {status['auto_trading']}")
        print(f"📈 Account Balance: ${status['account_balance']:.2f}")
        print(f"📊 Account Equity: ${status['account_equity']:.2f}")
        print(f"📝 Active Positions: {status['active_positions']}")
        print(f"💼 Daily Trades: {status['daily_trades']}")
        print(f"💰 Daily P&L: ${status['daily_profit']:.2f}")
        print("="*60)
        
        if status['emergency_stop']:
            print("🚨 EMERGENCY STOP ACTIVE")
            print("="*60)
    
    def _show_summary(self):
        """แสดงสรุปผลการเทรด"""
        if not self.start_time:
            return
        
        runtime = datetime.now() - self.start_time
        status = self.ea.get_status()
        stats = status['stats']
        
        print("\n" + "="*60)
        print("📈 LIGHTNING SCALPER SUMMARY")
        print("="*60)
        print(f"⏱️  Runtime: {runtime}")
        print(f"📡 Signals Received: {stats['signals_received']}")
        print(f"🎯 Trades Executed: {stats['trades_executed']}")
        print(f"✅ Winning Trades: {stats['winning_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        print(f"📊 Win Rate: {stats['win_rate']:.1%}")
        print(f"💰 Total Profit: ${stats['total_profit']:.2f}")
        print(f"📊 Total Pips: {stats['total_pips']:.1f}")
        
        if stats['winning_trades'] > 0:
            print(f"🎉 Largest Win: ${stats['largest_win']:.2f}")
        if stats['losing_trades'] > 0:
            print(f"😔 Largest Loss: ${stats['largest_loss']:.2f}")
        
        print("="*60)
    
    def run(self):
        """รันระบบหลัก"""
        try:
            if not self.start():
                return
            
            print("\n💡 Tips:")
            print("   - Press 's' + Enter to show status")
            print("   - Press 'q' + Enter to quit")
            print("   - Press 'e' + Enter to toggle emergency stop")
            print("   - Press 'a' + Enter to toggle auto trading")
            print("\n🎯 System is running... (Press Ctrl+C to stop)")
            
            # Main loop
            while True:
                try:
                    # Show periodic status (every 5 minutes)
                    if int(time.time()) % 300 == 0:
                        self._show_status()
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    print("\n⌨️ Keyboard interrupt received")
                    break
                    
        except Exception as e:
            self.logger.error(f"💀 Critical error: {e}")
        
        finally:
            self.stop()

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 Lightning Scalper - Simple Client")
    print("=" * 60)
    print("📋 Easy setup for beginners")
    print("🎯 Automated FVG trading system")
    print("📈 Professional risk management") 
    print("=" * 60)
    
    # Validate configuration
    if CONFIG['client_id'] == 'YOUR_CLIENT_ID':
        print("\n❌ CONFIGURATION REQUIRED!")
        print("Please edit the CONFIG section at the top of this file:")
        print("1. Set your client_id (received from your IB)")
        print("2. Set your api_key (received from your IB)")  
        print("3. Set your api_secret (received from your IB)")
        print("4. Set your MT5 login credentials")
        print("5. Save the file and run again")
        input("\nPress Enter to exit...")
        return
    
    # Create and run client
    client = SimpleLightningClient(CONFIG)
    client.run()

if __name__ == "__main__":
    main()