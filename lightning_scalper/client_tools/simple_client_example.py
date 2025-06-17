#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[ROCKET] Lightning Scalper - Simple Client Example
Easy-to-use example for beginners

? ???????: client_tools/simple_client_example.py

This is a simplified example showing how to use Lightning Scalper
for beginners who want to get started quickly.

Features:
- Simple configuration
- Automatic signal handling
- Basic risk management
- Easy setup and run

Author: Phoenix Trading AI (??????????????)
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
    print("[X] Missing Lightning Scalper modules!")
    print("   Make sure lightning_scalper_sdk.py and lightning_scalper_mt5_ea.py are in the same directory")
    sys.exit(1)

# [TOOL] CONFIGURATION - ???????????
# ================================================
CONFIG = {
    # ???????????? Lightning Scalper
    'client_id': 'YOUR_CLIENT_ID',          # ????????? IB
    'api_key': 'YOUR_API_KEY',              # ????????? IB  
    'api_secret': 'YOUR_API_SECRET',        # ????????? IB
    
    # ?????? MetaTrader 5
    'mt5_login': 12345678,                  # ??? Account MT5
    'mt5_password': 'your_mt5_password',    # ???????? MT5
    'mt5_server': 'YourBroker-Demo',        # Server MT5
    
    # ?????????????????
    'risk_per_trade': 2.0,                  # ???????????????????? (%)
    'max_positions': 3,                     # ?????????????????????????????????
    'max_daily_trades': 10,                 # ?????????????????????
    'daily_loss_limit': 200.0,              # ?????????????????? ($)
    
    # ??????????? (Demo/Live)
    'use_demo': True,                       # True = Demo, False = Live
}

class SimpleLightningClient:
    """
    [ROCKET] Lightning Scalper Simple Client
    ????????????? ? ?????????????
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
        
        self.logger.info("[ROCKET] Simple Lightning Scalper Client initialized")
        self.logger.info(f"   Client ID: {config['client_id']}")
        self.logger.info(f"   Environment: {'DEMO' if config['use_demo'] else 'LIVE'}")
        self.logger.info(f"   Risk per trade: {config['risk_per_trade']}%")
        self.logger.info(f"   Max positions: {config['max_positions']}")
    
    def start(self):
        """????????????????"""
        try:
            self.logger.info("[ROCKET] Starting Lightning Scalper...")
            
            # Validate configuration
            if not self._validate_config():
                return False
            
            # Start EA
            if not self.ea.start():
                self.logger.error("[X] Failed to start trading system")
                return False
            
            self.start_time = datetime.now()
            self.logger.info("[CHECK] Lightning Scalper started successfully!")
            self.logger.info("[TARGET] System is now listening for signals...")
            
            # Show initial status
            self._show_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"[X] Failed to start: {e}")
            return False
    
    def stop(self):
        """????????????"""
        try:
            self.logger.info("? Stopping Lightning Scalper...")
            self.ea.stop()
            self.logger.info("[CHECK] Lightning Scalper stopped")
            
            # Show final summary
            self._show_summary()
            
        except Exception as e:
            self.logger.error(f"[X] Error stopping: {e}")
    
    def _validate_config(self) -> bool:
        """?????????????????"""
        required_fields = ['client_id', 'api_key', 'api_secret', 'mt5_login', 'mt5_password']
        
        for field in required_fields:
            if not self.config.get(field) or self.config[field] in ['YOUR_CLIENT_ID', 'YOUR_API_KEY', 'YOUR_API_SECRET']:
                self.logger.error(f"[X] Please configure {field} in the CONFIG section")
                return False
        
        if self.config['mt5_login'] == 12345678:
            self.logger.error("[X] Please configure your real MT5 login number")
            return False
        
        if self.config['mt5_password'] == 'your_mt5_password':
            self.logger.error("[X] Please configure your real MT5 password")
            return False
        
        return True
    
    def _show_status(self):
        """?????????????????"""
        status = self.ea.get_status()
        
        print("\n" + "="*60)
        print("[CHART] LIGHTNING SCALPER STATUS")
        print("="*60)
        print(f"? System Running: {status['ea_running']}")
        print(f"? MT5 Connected: {status['mt5_connected']}")
        print(f"[GLOBE] Server Connected: {status['lightning_connected']}")
        print(f"? Auto Trading: {status['auto_trading']}")
        print(f"[TRENDING_UP] Account Balance: ${status['account_balance']:.2f}")
        print(f"[CHART] Account Equity: ${status['account_equity']:.2f}")
        print(f"[MEMO] Active Positions: {status['active_positions']}")
        print(f"? Daily Trades: {status['daily_trades']}")
        print(f"[MONEY] Daily P&L: ${status['daily_profit']:.2f}")
        print("="*60)
        
        if status['emergency_stop']:
            print("[SIREN] EMERGENCY STOP ACTIVE")
            print("="*60)
    
    def _show_summary(self):
        """?????????????????"""
        if not self.start_time:
            return
        
        runtime = datetime.now() - self.start_time
        status = self.ea.get_status()
        stats = status['stats']
        
        print("\n" + "="*60)
        print("[TRENDING_UP] LIGHTNING SCALPER SUMMARY")
        print("="*60)
        print(f"[TIMER]  Runtime: {runtime}")
        print(f"[SATELLITE] Signals Received: {stats['signals_received']}")
        print(f"[TARGET] Trades Executed: {stats['trades_executed']}")
        print(f"[CHECK] Winning Trades: {stats['winning_trades']}")
        print(f"[X] Losing Trades: {stats['losing_trades']}")
        print(f"[CHART] Win Rate: {stats['win_rate']:.1%}")
        print(f"[MONEY] Total Profit: ${stats['total_profit']:.2f}")
        print(f"[CHART] Total Pips: {stats['total_pips']:.1f}")
        
        if stats['winning_trades'] > 0:
            print(f"[PARTY] Largest Win: ${stats['largest_win']:.2f}")
        if stats['losing_trades'] > 0:
            print(f"? Largest Loss: ${stats['largest_loss']:.2f}")
        
        print("="*60)
    
    def run(self):
        """???????????"""
        try:
            if not self.start():
                return
            
            print("\n[BULB] Tips:")
            print("   - Press 's' + Enter to show status")
            print("   - Press 'q' + Enter to quit")
            print("   - Press 'e' + Enter to toggle emergency stop")
            print("   - Press 'a' + Enter to toggle auto trading")
            print("\n[TARGET] System is running... (Press Ctrl+C to stop)")
            
            # Main loop
            while True:
                try:
                    # Show periodic status (every 5 minutes)
                    if int(time.time()) % 300 == 0:
                        self._show_status()
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    print("\n?? Keyboard interrupt received")
                    break
                    
        except Exception as e:
            self.logger.error(f"? Critical error: {e}")
        
        finally:
            self.stop()

def main():
    """????????????"""
    print("[ROCKET] Lightning Scalper - Simple Client")
    print("=" * 60)
    print("[CLIPBOARD] Easy setup for beginners")
    print("[TARGET] Automated FVG trading system")
    print("[TRENDING_UP] Professional risk management") 
    print("=" * 60)
    
    # Validate configuration
    if CONFIG['client_id'] == 'YOUR_CLIENT_ID':
        print("\n[X] CONFIGURATION REQUIRED!")
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