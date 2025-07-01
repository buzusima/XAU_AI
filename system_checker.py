#!/usr/bin/env python3
"""
🔍 FOREX TRADING SYSTEM HEALTH CHECKER
=====================================
ตรวจสอบสถานะของระบบ Trading ทั้งหมด
"""

import requests
import json
import time
from datetime import datetime
import sys

class SystemHealthChecker:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = {}
        
    def print_header(self, title):
        """พิมพ์หัวข้อสวยๆ"""
        print("\n" + "="*60)
        print(f"🔍 {title}")
        print("="*60)
    
    def print_result(self, component, status, details=""):
        """พิมพ์ผลลัพธ์"""
        status_icon = "✅" if status == "OK" else "❌" if status == "ERROR" else "⚠️"
        print(f"{status_icon} {component:<30} {status:<10} {details}")
    
    def test_api_endpoint(self, endpoint, description):
        """ทดสอบ API endpoint"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success', True):
                    return "OK", f"Status: {response.status_code}"
                else:
                    return "ERROR", f"API Error: {data.get('error', 'Unknown')}"
            else:
                return "ERROR", f"HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "ERROR", "Connection refused - Server not running"
        except requests.exceptions.Timeout:
            return "ERROR", "Request timeout"
        except Exception as e:
            return "ERROR", f"Exception: {str(e)}"
    
    def check_core_apis(self):
        """ตรวจสอบ Core APIs"""
        self.print_header("CORE API ENDPOINTS")
        
        endpoints = [
            ("/api/market-data", "Market Data API"),
            ("/api/account-info", "Account Information"),
            ("/api/system-status", "System Status"),
            ("/api/symbol-data/EURUSD.c", "Symbol Data API"),
            ("/", "Main Dashboard"),
        ]
        
        for endpoint, description in endpoints:
            status, details = self.test_api_endpoint(endpoint, description)
            self.print_result(description, status, details)
            self.results[f"api_{endpoint.replace('/', '_')}"] = status == "OK"
    
    def check_trading_systems(self):
        """ตรวจสอบ Trading Systems"""
        self.print_header("TRADING SYSTEMS")
        
        # Test Enhanced Signal Engine
        status, details = self.test_api_endpoint("/api/test-enhanced-signals", "Enhanced Signal Engine")
        self.print_result("Enhanced Signal Engine", status, details)
        
        # Test Advanced Features
        status, details = self.test_api_endpoint("/api/test-advanced-features", "Advanced Features")
        self.print_result("Advanced Features", status, details)
        
        # Test Signal Generation
        status, details = self.test_api_endpoint("/api/test-signals", "Signal Generation")
        self.print_result("Signal Generation", status, details)
        
        # Test Position Management
        status, details = self.test_api_endpoint("/api/positions", "Position Management")
        self.print_result("Position Management", status, details)
    
    def check_risk_management(self):
        """ตรวจสอบ Risk Management"""
        self.print_header("RISK MANAGEMENT SYSTEMS")
        
        # Test Pullback Protection
        try:
            response = requests.get(f"{self.base_url}/api/pullback-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.print_result("Pullback Protection", "OK", f"Active: {data.get('active', False)}")
                else:
                    self.print_result("Pullback Protection", "ERROR", "API Error")
            else:
                self.print_result("Pullback Protection", "ERROR", f"HTTP {response.status_code}")
        except:
            self.print_result("Pullback Protection", "WARNING", "Endpoint not available")
        
        # Test Trailing Stops
        try:
            response = requests.get(f"{self.base_url}/api/trailing-status", timeout=5)
            if response.status_code == 200:
                self.print_result("Trailing Stops", "OK", "API Available")
            else:
                self.print_result("Trailing Stops", "WARNING", "Limited functionality")
        except:
            self.print_result("Trailing Stops", "WARNING", "Endpoint not available")
        
        # Test Hedging System
        try:
            response = requests.get(f"{self.base_url}/hedging", timeout=5)
            if response.status_code == 200:
                self.print_result("Hedging System", "OK", "Dashboard Available")
            else:
                self.print_result("Hedging System", "WARNING", "Dashboard unavailable")
        except:
            self.print_result("Hedging System", "WARNING", "System not available")
    
    def check_data_quality(self):
        """ตรวจสอบคุณภาพข้อมูล"""
        self.print_header("DATA QUALITY CHECK")
        
        try:
            response = requests.get(f"{self.base_url}/api/market-data", timeout=10)
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('data', {})
                
                if len(market_data) > 0:
                    self.print_result("Market Data Loading", "OK", f"{len(market_data)} symbols loaded")
                    
                    # ตรวจสอบข้อมูลตัวอย่าง
                    sample_symbol = list(market_data.keys())[0]
                    sample_data = market_data[sample_symbol]
                    
                    required_fields = ['current_price', 'signal', 'strength']
                    missing_fields = [field for field in required_fields if field not in sample_data]
                    
                    if not missing_fields:
                        self.print_result("Data Completeness", "OK", "All required fields present")
                    else:
                        self.print_result("Data Completeness", "WARNING", f"Missing: {missing_fields}")
                    
                    # ตรวจสอบราคา
                    price = sample_data.get('current_price', 0)
                    if price > 0:
                        self.print_result("Price Data", "OK", f"Valid prices (sample: {price})")
                    else:
                        self.print_result("Price Data", "ERROR", "Invalid price data")
                else:
                    self.print_result("Market Data Loading", "ERROR", "No market data available")
            else:
                self.print_result("Market Data Loading", "ERROR", f"HTTP {response.status_code}")
        except Exception as e:
            self.print_result("Market Data Loading", "ERROR", str(e))
    
    def check_mt5_connection(self):
        """ตรวจสอบการเชื่อมต่อ MT5"""
        self.print_header("MT5 CONNECTION")
        
        try:
            response = requests.get(f"{self.base_url}/api/account-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    account_info = data.get('account', {})
                    mt5_connected = data.get('mt5_connected', False)
                    
                    if mt5_connected:
                        self.print_result("MT5 Connection", "OK", "Connected")
                        
                        balance = account_info.get('balance', 0)
                        if balance > 0:
                            self.print_result("Account Balance", "OK", f"${balance:,.2f}")
                        else:
                            self.print_result("Account Balance", "WARNING", "No balance data")
                        
                        server = account_info.get('server', 'Unknown')
                        self.print_result("MT5 Server", "OK", server)
                    else:
                        self.print_result("MT5 Connection", "ERROR", "Not connected")
                else:
                    self.print_result("MT5 Connection", "ERROR", data.get('error', 'Unknown error'))
            else:
                self.print_result("MT5 Connection", "ERROR", f"HTTP {response.status_code}")
        except Exception as e:
            self.print_result("MT5 Connection", "ERROR", str(e))
    
    def run_comprehensive_check(self):
        """รันการตรวจสอบทั้งหมด"""
        print("🚀 FOREX TRADING SYSTEM HEALTH CHECK")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Target URL: {self.base_url}")
        
        # รันการตรวจสอบทั้งหมด
        self.check_core_apis()
        self.check_mt5_connection()
        self.check_data_quality()
        self.check_trading_systems()
        self.check_risk_management()
        
        # สรุปผลลัพธ์
        self.print_summary()
    
    def print_summary(self):
        """สรุปผลลัพธ์"""
        self.print_header("SUMMARY")
        
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)
        
        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
            print(f"📊 Overall Health: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed)")
            
            if success_rate >= 90:
                print("🎉 System Status: EXCELLENT - All systems operational")
            elif success_rate >= 70:
                print("✅ System Status: GOOD - Minor issues detected")
            elif success_rate >= 50:
                print("⚠️ System Status: FAIR - Some components need attention")
            else:
                print("❌ System Status: POOR - Major issues detected")
        else:
            print("⚠️ No automated checks completed")
        
        print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

def main():
    """Main function"""
    print("🔍 FOREX SYSTEM HEALTH CHECKER")
    print("Starting comprehensive system check...")
    
    # สร้าง checker instance
    checker = SystemHealthChecker()
    
    # รันการตรวจสอบ
    try:
        checker.run_comprehensive_check()
    except KeyboardInterrupt:
        print("\n❌ Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()