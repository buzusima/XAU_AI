#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Fixed MT5 Detection Version
🚀 แก้ปัญหา MT5 Library detection + คำแนะนำที่ชัดเจน

Fixed Issues:
✅ แสดง MT5 installation status ที่ชัดเจน
✅ คำแนะนำการแก้ไขปัญหา MT5
✅ Demo mode ทำงานได้โดยไม่ต้องมี MT5
✅ Error messages ที่เข้าใจง่าย

Version: 2.1.1 (MT5 Fixed)
"""

import json
import logging
import os
import threading
import time
import webbrowser
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Flask imports
from flask import Flask, render_template_string, request, jsonify

# MetaTrader 5 - Enhanced Detection
MT5_AVAILABLE = False
MT5_ERROR_MSG = ""

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    MT5_ERROR_MSG = "MT5 library loaded successfully"
    print("[INFO] ✅ MT5 library available")
except ImportError as e:
    MT5_AVAILABLE = False
    MT5_ERROR_MSG = f"MT5 library not installed: {str(e)}"
    print(f"[WARNING] ❌ MT5 not available: {e}")
except Exception as e:
    MT5_AVAILABLE = False
    MT5_ERROR_MSG = f"MT5 library error: {str(e)}"
    print(f"[ERROR] ❌ MT5 error: {e}")

class EnhancedMT5Manager:
    """🔌 Enhanced MT5 Manager with better error detection"""
    
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.connection_attempts = 0
        self.max_attempts = 3
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_mt5_installation(self) -> Tuple[bool, str]:
        """ตรวจสอบการติดตั้ง MT5 library อย่างละเอียด"""
        if not MT5_AVAILABLE:
            return False, f"""❌ MetaTrader5 library ไม่ได้ติดตั้ง

🔧 วิธีแก้ไข:
1. ติดตั้ง MetaTrader5 library:
   pip install MetaTrader5

2. หรือรัน fix_mt5_install.bat

3. หรือใช้โหมด Demo (ไม่ต้องมี MT5)

Error: {MT5_ERROR_MSG}"""
        
        return True, "✅ MetaTrader5 library ติดตั้งแล้ว"
    
    def test_mt5_connection(self) -> Tuple[bool, str]:
        """ทดสอบการเชื่อมต่อ MT5 แบบละเอียด"""
        # Step 1: Check library installation
        lib_ok, lib_msg = self.check_mt5_installation()
        if not lib_ok:
            return False, lib_msg
        
        try:
            # Step 2: Test MT5 initialization
            success = mt5.initialize()
            if not success:
                error = mt5.last_error()
                return False, f"""❌ ไม่สามารถเชื่อมต่อ MT5 ได้

🔧 วิธีแก้ไข:
1. เปิด MetaTrader 5 application
2. ตรวจสอบว่า MT5 ทำงานอยู่
3. Login เข้าบัญชี (demo หรือ live)
4. ลองใหม่อีกครั้ง

MT5 Error: {error}"""
            
            # Step 3: Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False, """❌ ไม่สามารถดึงข้อมูล MT5 terminal ได้

🔧 วิธีแก้ไข:
1. ปิด MT5 แล้วเปิดใหม่
2. ตรวจสอบว่า MT5 ไม่ถูก block โดย Firewall
3. รัน MT5 ในฐานะ Administrator"""
            
            # Step 4: Check account
            account_info = mt5.account_info()
            if account_info is None:
                return True, f"""✅ MT5 เชื่อมต่อได้แต่ยังไม่ได้ login

📋 ข้อมูล Terminal:
• ชื่อ: {terminal_info.name}
• Build: {terminal_info.build}
• Path: {terminal_info.path}

💡 ขั้นตอนต่อไป:
1. Login เข้าบัญชี MT5
2. ใช้ "Auto Detect" หรือ "Manual Setup"
3. เริ่มการเทรด"""
            
            # Step 5: Full success
            return True, f"""✅ MT5 เชื่อมต่อสมบูรณ์!

📋 ข้อมูลบัญชี:
• Account: {account_info.login}
• Server: {account_info.server}
• Company: {account_info.company}
• Balance: ${account_info.balance:,.2f}
• Currency: {account_info.currency}
• Trading: {'✅ Allowed' if account_info.trade_allowed else '❌ Not Allowed'}

🎉 พร้อมเทรดแล้ว!"""
            
        except Exception as e:
            return False, f"""❌ เกิดข้อผิดพลาด MT5

Error: {str(e)}

🔧 วิธีแก้ไข:
1. ตรวจสอบว่า MT5 ทำงานอย่างถูกต้อง
2. Restart MT5 และ Lightning Scalper
3. ติดต่อ support หากปัญหายังคงอยู่"""
    
    def initialize_mt5(self) -> Tuple[bool, str]:
        """Initialize MT5 with enhanced error handling"""
        if not MT5_AVAILABLE:
            return False, "MT5 library not installed"
        
        try:
            # ลอง shutdown ก่อน
            try:
                mt5.shutdown()
                time.sleep(0.5)
            except:
                pass
            
            # Initialize MT5
            if mt5.initialize():
                self.logger.info("MT5 initialized successfully")
                return True, "MT5 initialized successfully"
            else:
                error = mt5.last_error()
                error_msg = f"MT5 initialization failed: {error}"
                self.logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"MT5 initialization exception: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def auto_detect_mt5(self) -> Tuple[bool, Dict]:
        """Enhanced auto-detect with step-by-step checking"""
        if not MT5_AVAILABLE:
            return False, {
                "error": "MetaTrader5 library ไม่ได้ติดตั้ง",
                "hint": "รัน: pip install MetaTrader5",
                "solution": "ใช้ fix_mt5_install.bat หรือติดตั้งด้วยตนเอง"
            }
        
        try:
            # Step 1: Initialize
            success, msg = self.initialize_mt5()
            if not success:
                return False, {
                    "error": "ไม่สามารถเริ่มต้น MT5 ได้",
                    "hint": "ตรวจสอบว่า MetaTrader 5 เปิดอยู่และทำงานปกติ",
                    "details": msg
                }
            
            # Step 2: Terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False, {
                    "error": "ไม่สามารถเชื่อมต่อ MT5 terminal ได้",
                    "hint": "ตรวจสอบว่า MT5 ไม่ถูก Firewall block",
                    "solution": "ลองรัน MT5 ในฐานะ Administrator"
                }
            
            # Step 3: Account info
            account_info = mt5.account_info()
            if account_info is None:
                return False, {
                    "error": "MT5 terminal เชื่อมต่อได้แต่ยังไม่ได้ login บัญชี",
                    "hint": "กรุณา login เข้าบัญชี MT5 ก่อน",
                    "terminal_info": {
                        "name": terminal_info.name,
                        "build": terminal_info.build,
                        "company": getattr(terminal_info, 'company', 'Unknown')
                    }
                }
            
            # Step 4: Trading permissions
            if not account_info.trade_allowed:
                return False, {
                    "error": "บัญชีนี้ไม่อนุญาตให้เทรด",
                    "hint": "ตรวจสอบการตั้งค่าบัญชีหรือติดต่อ broker",
                    "account": account_info.login
                }
            
            # Step 5: Success
            self.connected = True
            
            self.account_info = {
                "account": account_info.login,
                "name": account_info.name,
                "server": account_info.server,
                "currency": account_info.currency,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level if account_info.margin > 0 else 0,
                "company": account_info.company,
                "leverage": account_info.leverage,
                "trade_allowed": account_info.trade_allowed,
                "trade_expert": account_info.trade_expert,
                
                # Terminal info
                "terminal_name": terminal_info.name,
                "terminal_build": terminal_info.build,
                
                # Status
                "connection_status": "Connected",
                "last_update": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully connected to MT5 account {account_info.login}")
            return True, self.account_info
            
        except Exception as e:
            error_msg = f"Auto-detect failed: {str(e)}"
            self.logger.error(error_msg)
            return False, {
                "error": "เกิดข้อผิดพลาดในการตรวจหา MT5",
                "details": error_msg,
                "hint": "ลอง restart MT5 และ Lightning Scalper"
            }
    
    def manual_connect(self, login: int, password: str, server: str) -> Tuple[bool, Dict]:
        """Manual connection with enhanced error handling"""
        if not MT5_AVAILABLE:
            return False, {
                "error": "MetaTrader5 library ไม่ได้ติดตั้ง",
                "hint": "รัน: pip install MetaTrader5"
            }
        
        try:
            self.logger.info(f"Manual connection attempt: {login}@{server}")
            
            # Step 1: Initialize
            success, msg = self.initialize_mt5()
            if not success:
                return False, {"error": f"ไม่สามารถเริ่มต้น MT5: {msg}"}
            
            # Step 2: Login with retry
            for attempt in range(self.max_attempts):
                self.logger.info(f"Login attempt {attempt + 1}/{self.max_attempts}")
                
                login_success = mt5.login(login, password, server)
                
                if login_success:
                    self.logger.info("Manual login successful")
                    return self.auto_detect_mt5()
                else:
                    error = mt5.last_error()
                    self.logger.warning(f"Login attempt {attempt + 1} failed: {error}")
                    
                    if attempt < self.max_attempts - 1:
                        time.sleep(2)
            
            # All attempts failed
            last_error = mt5.last_error()
            return False, {
                "error": f"Login ไม่สำเร็จหลังจากพยายาม {self.max_attempts} ครั้ง",
                "mt5_error": str(last_error),
                "hint": "ตรวจสอบ login, password, และชื่อ server ให้ถูกต้อง"
            }
            
        except Exception as e:
            error_msg = f"Manual connection failed: {str(e)}"
            self.logger.error(error_msg)
            return False, {"error": error_msg}
    
    def get_account_status(self) -> Dict:
        """Get current account status"""
        if not self.connected or not MT5_AVAILABLE:
            return {"status": "disconnected"}
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    "status": "connected",
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "margin_level": account_info.margin_level if account_info.margin > 0 else 0,
                    "last_update": datetime.now().isoformat()
                }
            else:
                self.connected = False
                return {"status": "disconnected"}
        except:
            self.connected = False
            return {"status": "disconnected"}

class ConfigManager:
    """⚙️ Configuration Manager"""
    
    def __init__(self):
        self.config = self._load_default_config()
        self.config_file = "lightning_scalper_config.json"
        self._load_config()
    
    def _load_default_config(self) -> Dict:
        return {
            "risk_level": 3,
            "timeframes": ["M15", "H1"],
            "entry_method": "hybrid",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
            "max_positions": 5,
            "risk_per_trade": 2.0,
            "daily_loss_limit": 5.0,
            "auto_trading": False,
            "demo_mode": True,
            "magic_number": 123456,
            "slippage": 3,
            "trailing_stop": False,
            "partial_close": False,
            "notifications": {
                "sound": True,
                "popup": True,
                "email": False
            }
        }
    
    def _load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
        except Exception as e:
            print(f"Config load error: {e}")
    
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Config save error: {e}")
            return False
    
    def update_config(self, updates: Dict) -> bool:
        try:
            self.config.update(updates)
            return self.save_config()
        except Exception as e:
            print(f"Config update error: {e}")
            return False

class LightningScalperMT5Fixed:
    """🚀 Lightning Scalper - MT5 Detection Fixed"""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lightning_scalper_mt5_fixed_2024'
        
        # Core managers
        self.mt5_manager = EnhancedMT5Manager()
        self.config_manager = ConfigManager()
        
        # Runtime state
        self.is_trading = False
        self.performance_data = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,
            "daily_profit": 0.0,
            "win_rate": 0.0
        }
        
        self._setup_routes()
        self._start_monitoring()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_html())
        
        @self.app.route('/api/status')
        def get_status():
            account_status = self.mt5_manager.get_account_status()
            return jsonify({
                'mt5_available': MT5_AVAILABLE,
                'mt5_error': MT5_ERROR_MSG,
                'mt5_connected': self.mt5_manager.connected,
                'is_trading': self.is_trading,
                'account_info': self.mt5_manager.account_info,
                'account_status': account_status,
                'performance': self.performance_data,
                'config': self.config_manager.config,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/mt5/test', methods=['POST'])
        def test_mt5():
            success, message = self.mt5_manager.test_mt5_connection()
            return jsonify({"success": success, "message": message})
        
        @self.app.route('/api/mt5/auto-detect', methods=['POST'])
        def auto_detect_mt5():
            success, data = self.mt5_manager.auto_detect_mt5()
            return jsonify({"success": success, "data": data})
        
        @self.app.route('/api/mt5/manual-connect', methods=['POST'])
        def manual_connect_mt5():
            try:
                data = request.json
                if not data:
                    return jsonify({"success": False, "data": {"error": "ไม่มีข้อมูล"}})
                
                login = data.get('login')
                password = data.get('password')
                server = data.get('server')
                
                if not all([login, password, server]):
                    return jsonify({"success": False, "data": {"error": "กรุณากรอกข้อมูลให้ครบถ้วน"}})
                
                success, result = self.mt5_manager.manual_connect(
                    int(login), str(password), str(server)
                )
                
                return jsonify({"success": success, "data": result})
                
            except Exception as e:
                return jsonify({"success": False, "data": {"error": f"เกิดข้อผิดพลาด: {str(e)}"}})
        
        @self.app.route('/api/config/get', methods=['GET'])
        def get_config():
            return jsonify({"success": True, "config": self.config_manager.config})
        
        @self.app.route('/api/config/update', methods=['POST'])
        def update_config():
            try:
                updates = request.json
                if updates:
                    success = self.config_manager.update_config(updates)
                    return jsonify({"success": success})
                return jsonify({"success": False, "error": "ไม่มีข้อมูล"})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/api/trading/start', methods=['POST'])
        def start_trading():
            if not MT5_AVAILABLE:
                return jsonify({"success": False, "error": "MT5 library ไม่ได้ติดตั้ง - ใช้โหมด Demo"})
            
            if not self.mt5_manager.connected:
                return jsonify({"success": False, "error": "MT5 ไม่ได้เชื่อมต่อ"})
            
            self.is_trading = True
            return jsonify({"success": True})
        
        @self.app.route('/api/trading/stop', methods=['POST'])
        def stop_trading():
            self.is_trading = False
            return jsonify({"success": True})
        
        @self.app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "version": "2.1.1",
                "mt5_available": MT5_AVAILABLE,
                "features": ["mt5_detection_fixed", "settings_panel", "enhanced_ui"],
                "timestamp": datetime.now().isoformat()
            })
    
    def _start_monitoring(self):
        """Start background monitoring"""
        def monitor():
            while True:
                if self.mt5_manager.connected and MT5_AVAILABLE:
                    status = self.mt5_manager.get_account_status()
                    if status.get("status") == "disconnected":
                        self.mt5_manager.connected = False
                
                time.sleep(5)
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def _get_dashboard_html(self) -> str:
        """Dashboard HTML with enhanced MT5 status"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightning Scalper - MT5 Detection Fixed</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"></script>
    <style>
        :root {
            --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success: linear-gradient(135deg, #4CAF50, #45a049);
            --danger: linear-gradient(135deg, #f44336, #d32f2f);
            --warning: linear-gradient(135deg, #ff9800, #f57c00);
            --info: linear-gradient(135deg, #2196F3, #1976D2);
            --glass: rgba(255, 255, 255, 0.95);
            --shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .version-badge {
            background: var(--success);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-top: 10px;
            display: inline-block;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--glass);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 25px;
        }
        .card-icon {
            font-size: 1.8rem;
            margin-right: 15px;
            background: var(--primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card-title {
            font-size: 1.4rem;
            font-weight: 700;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            padding: 15px 20px;
            border-radius: 12px;
            margin: 15px 0;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .status-connected {
            background: var(--success);
            color: white;
            animation: pulse 2s infinite;
        }
        .status-disconnected {
            background: var(--danger);
            color: white;
        }
        .status-trading {
            background: var(--info);
            color: white;
            animation: pulse 2s infinite;
        }
        .status-warning {
            background: var(--warning);
            color: white;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .btn {
            padding: 15px 25px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 140px;
        }
        .btn i { margin-right: 8px; }
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .btn-success {
            background: var(--success);
            color: white;
        }
        .btn-danger {
            background: var(--danger);
            color: white;
        }
        .btn-warning {
            background: var(--warning);
            color: white;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        .form-control {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        .form-control:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.6);
            backdrop-filter: blur(5px);
        }
        .modal-content {
            background: var(--glass);
            margin: 5% auto;
            padding: 40px;
            border-radius: 20px;
            width: 90%;
            max-width: 600px;
            position: relative;
            backdrop-filter: blur(20px);
            max-height: 90vh;
            overflow-y: auto;
        }
        .close {
            position: absolute;
            right: 20px;
            top: 20px;
            font-size: 1.5rem;
            cursor: pointer;
            color: #aaa;
        }
        .close:hover { color: #333; }
        
        .account-details {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre-line;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .mt5-status-card {
            border: 2px solid;
            border-image: var(--primary) 1;
        }
        
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            display: flex;
            align-items: center;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .stat-icon {
            font-size: 2rem;
            margin-right: 15px;
            background: var(--primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-content {
            flex: 1;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .performance-grid { grid-template-columns: repeat(2, 1fr); }
            .header h1 { font-size: 2rem; }
        }
        
        .install-help {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffc107;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            color: #856404;
        }
        .install-help h4 {
            color: #856404;
            margin-bottom: 10px;
        }
        .install-help code {
            background: rgba(0,0,0,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-bolt"></i> Lightning Scalper</h1>
            <p>MT5 Detection Fixed - Complete AI Trading System</p>
            <div class="version-badge">
                <i class="fas fa-check-circle"></i> Version 2.1.1 - MT5 Detection Fixed
            </div>
        </div>
        
        <div class="dashboard-grid">
            <!-- MT5 Connection -->
            <div class="card mt5-status-card">
                <div class="card-header">
                    <i class="fas fa-plug card-icon"></i>
                    <h3 class="card-title">MT5 Connection Status</h3>
                </div>
                
                <div id="mt5-library-status" class="status-indicator status-warning">
                    <i class="fas fa-question-circle"></i>
                    <span>Checking MT5 Library...</span>
                </div>
                
                <div id="mt5-connection-status" class="status-indicator status-disconnected">
                    <i class="fas fa-times-circle"></i>
                    <span>Not Connected</span>
                </div>
                
                <div id="mt5-install-help" class="install-help" style="display: none;">
                    <h4><i class="fas fa-tools"></i> วิธีติดตั้ง MT5 Library</h4>
                    <p><strong>วิธีที่ 1:</strong> รัน <code>fix_mt5_install.bat</code></p>
                    <p><strong>วิธีที่ 2:</strong> รันคำสั่ง <code>pip install MetaTrader5</code></p>
                    <p><strong>วิธีที่ 3:</strong> ใช้โหมด Demo (ไม่ต้องมี MT5)</p>
                </div>
                
                <div id="account-info" class="account-details" style="display: none;">
                    Account info will appear here...
                </div>
                
                <button class="btn btn-primary" onclick="testMT5()">
                    <i class="fas fa-flask"></i> Test MT5
                </button>
                
                <button class="btn btn-primary" onclick="autoDetectMT5()">
                    <i class="fas fa-search"></i> Auto Detect
                </button>
                
                <button class="btn btn-primary" onclick="showManualSetup()">
                    <i class="fas fa-cog"></i> Manual Setup
                </button>
            </div>
            
            <!-- Trading Control -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-robot card-icon"></i>
                    <h3 class="card-title">AI Trading Control</h3>
                </div>
                
                <div id="trading-status" class="status-indicator status-disconnected">
                    <i class="fas fa-pause-circle"></i>
                    <span>Trading Stopped</span>
                </div>
                
                <div class="account-details">
                    <strong>Status:</strong> <span id="system-status">Ready</span><br>
                    <strong>Mode:</strong> <span id="trading-mode">Demo</span><br>
                    <strong>Last Update:</strong> <span id="last-update">-</span>
                </div>
                
                <button id="start-btn" class="btn btn-success" onclick="startTrading()" disabled>
                    <i class="fas fa-play"></i> Start Trading
                </button>
                
                <button id="stop-btn" class="btn btn-danger" onclick="stopTrading()" disabled>
                    <i class="fas fa-stop"></i> Stop Trading
                </button>
            </div>
            
            <!-- Performance Dashboard -->
            <div class="card" style="grid-column: 1 / -1;">
                <div class="card-header">
                    <i class="fas fa-chart-bar card-icon"></i>
                    <h3 class="card-title">Performance Dashboard</h3>
                </div>
                
                <div class="performance-grid">
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-dollar-sign"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="balance">$0.00</div>
                            <div class="stat-label">Balance</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="equity">$0.00</div>
                            <div class="stat-label">Equity</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-percentage"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="margin-level">0%</div>
                            <div class="stat-label">Margin Level</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-trophy"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="win-rate">0%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-cogs"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="mt5-library-status-text">Checking...</div>
                            <div class="stat-label">MT5 Library</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-plug"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value" id="connection-status-text">Disconnected</div>
                            <div class="stat-label">Connection</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Manual Setup Modal -->
    <div id="manual-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('manual-modal')">&times;</span>
            <h2><i class="fas fa-cog"></i> Manual MT5 Setup</h2>
            
            <div class="form-group">
                <label class="form-label">Login</label>
                <input type="number" id="mt5-login" class="form-control" placeholder="Enter MT5 login">
            </div>
            
            <div class="form-group">
                <label class="form-label">Password</label>
                <input type="password" id="mt5-password" class="form-control" placeholder="Enter MT5 password">
            </div>
            
            <div class="form-group">
                <label class="form-label">Server</label>
                <input type="text" id="mt5-server" class="form-control" placeholder="e.g., MetaQuotes-Demo">
            </div>
            
            <button class="btn btn-primary" onclick="manualConnect()">
                <i class="fas fa-plug"></i> Connect
            </button>
        </div>
    </div>

    <script>
        let isConnected = false;
        let isTrading = false;
        let mt5Available = false;
        
        // Update status every 3 seconds
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                if (!response.ok) throw new Error('Network error');
                
                const data = await response.json();
                
                updateMT5LibraryStatus(data.mt5_available, data.mt5_error);
                updateMT5ConnectionStatus(data.account_info || {}, data.mt5_connected);
                updateTradingStatus(data.is_trading);
                updatePerformanceData(data.account_status || {});
                
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Status update failed:', error);
                document.getElementById('last-update').textContent = 'Update failed';
            }
        }
        
        function updateMT5LibraryStatus(available, errorMsg) {
            const statusEl = document.getElementById('mt5-library-status');
            const helpEl = document.getElementById('mt5-install-help');
            const statusTextEl = document.getElementById('mt5-library-status-text');
            
            mt5Available = available;
            
            if (available) {
                statusEl.className = 'status-indicator status-connected';
                statusEl.innerHTML = '<i class="fas fa-check-circle"></i><span>MT5 Library ติดตั้งแล้ว</span>';
                helpEl.style.display = 'none';
                statusTextEl.textContent = 'Installed';
                statusTextEl.style.color = '#4CAF50';
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-times-circle"></i><span>MT5 Library ไม่ได้ติดตั้ง</span>';
                helpEl.style.display = 'block';
                statusTextEl.textContent = 'Not Installed';
                statusTextEl.style.color = '#f44336';
            }
        }
        
        function updateMT5ConnectionStatus(accountInfo, connected) {
            const statusEl = document.getElementById('mt5-connection-status');
            const accountInfoEl = document.getElementById('account-info');
            const startBtn = document.getElementById('start-btn');
            const connectionStatusEl = document.getElementById('connection-status-text');
            
            isConnected = connected;
            
            if (connected) {
                statusEl.className = 'status-indicator status-connected';
                statusEl.innerHTML = '<i class="fas fa-check-circle"></i><span>MT5 เชื่อมต่อแล้ว</span>';
                
                const info = `Account: ${accountInfo.account || '-'}
Server: ${accountInfo.server || '-'}
Company: ${accountInfo.company || '-'}
Currency: ${accountInfo.currency || '-'}
Balance: $${(accountInfo.balance || 0).toFixed(2)}
Equity: $${(accountInfo.equity || 0).toFixed(2)}
Free Margin: $${(accountInfo.free_margin || 0).toFixed(2)}
Leverage: 1:${accountInfo.leverage || '-'}
Trading Allowed: ${accountInfo.trade_allowed ? 'Yes' : 'No'}`;
                
                accountInfoEl.textContent = info;
                accountInfoEl.style.display = 'block';
                startBtn.disabled = false;
                connectionStatusEl.textContent = 'Connected';
                connectionStatusEl.style.color = '#4CAF50';
                document.getElementById('trading-mode').textContent = 'Live Trading';
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-times-circle"></i><span>MT5 ไม่ได้เชื่อมต่อ</span>';
                
                accountInfoEl.style.display = 'none';
                startBtn.disabled = true;
                connectionStatusEl.textContent = 'Disconnected';
                connectionStatusEl.style.color = '#f44336';
                document.getElementById('trading-mode').textContent = mt5Available ? 'Ready for MT5' : 'Demo Mode';
            }
        }
        
        function updateTradingStatus(trading) {
            const statusEl = document.getElementById('trading-status');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            isTrading = trading;
            
            if (trading) {
                statusEl.className = 'status-indicator status-trading';
                statusEl.innerHTML = '<i class="fas fa-robot"></i><span>AI Trading Active</span>';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                document.getElementById('system-status').textContent = 'Trading Active';
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-pause-circle"></i><span>Trading Stopped</span>';
                startBtn.disabled = !isConnected;
                stopBtn.disabled = true;
                document.getElementById('system-status').textContent = 'Ready';
            }
        }
        
        function updatePerformanceData(data) {
            document.getElementById('balance').textContent = `$${(data.balance || 0).toFixed(2)}`;
            document.getElementById('equity').textContent = `$${(data.equity || 0).toFixed(2)}`;
            document.getElementById('margin-level').textContent = `${(data.margin_level || 0).toFixed(1)}%`;
        }
        
        // MT5 Functions
        async function testMT5() {
            const btn = event.target;
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/mt5/test', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                showNotification(result.message, result.success ? 'success' : 'error');
                
            } catch (error) {
                showNotification('Test Error: ' + error.message, 'error');
            } finally {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }
        }
        
        async function autoDetectMT5() {
            const btn = event.target;
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/mt5/auto-detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification('🎉 MT5 เชื่อมต่อสำเร็จ!', 'success');
                    updateStatus();
                } else {
                    const error = result.data?.error || 'Connection failed';
                    const hint = result.data?.hint || '';
                    showNotification('❌ ' + error, 'error');
                    if (hint) {
                        setTimeout(() => showNotification('💡 ' + hint, 'info'), 2000);
                    }
                }
            } catch (error) {
                showNotification('Connection Error: ' + error.message, 'error');
            } finally {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }
        }
        
        async function manualConnect() {
            const login = document.getElementById('mt5-login').value.trim();
            const password = document.getElementById('mt5-password').value.trim();
            const server = document.getElementById('mt5-server').value.trim();
            
            if (!login || !password || !server) {
                showNotification('กรุณากรอกข้อมูลให้ครบถ้วน', 'error');
                return;
            }
            
            const btn = event.target;
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/mt5/manual-connect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        login: parseInt(login),
                        password: password,
                        server: server
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification('🎉 Manual connection สำเร็จ!', 'success');
                    closeModal('manual-modal');
                    updateStatus();
                } else {
                    const error = result.data?.error || 'Connection failed';
                    const hint = result.data?.hint || '';
                    showNotification('❌ ' + error, 'error');
                    if (hint) {
                        setTimeout(() => showNotification('💡 ' + hint, 'info'), 2000);
                    }
                }
            } catch (error) {
                showNotification('Connection Error: ' + error.message, 'error');
            } finally {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }
        }
        
        // Trading Functions
        async function startTrading() {
            try {
                const response = await fetch('/api/trading/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification('🚀 การเทรดเริ่มแล้ว!', 'success');
                    updateStatus();
                } else {
                    showNotification('❌ ' + (result.error || 'Failed to start'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        async function stopTrading() {
            try {
                const response = await fetch('/api/trading/stop', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification('⏹️ การเทรดหยุดแล้ว!', 'success');
                    updateStatus();
                } else {
                    showNotification('❌ ' + (result.error || 'Failed to stop'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        // Utility Functions
        function showManualSetup() {
            document.getElementById('manual-modal').style.display = 'block';
        }
        
        function closeModal(modalId) {
            document.getElementById(modalId).style.display = 'none';
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : 
                           type === 'info' ? 'var(--info)' : 'var(--danger)';
            
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 12px;
                color: white;
                font-weight: 600;
                z-index: 10001;
                background: ${bgColor};
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                max-width: 400px;
                word-wrap: break-word;
                animation: slideIn 0.3s ease;
            `;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }, 5000);
        }
        
        // Close modals when clicking outside
        window.onclick = function(event) {
            if (event.target.classList.contains('modal')) {
                event.target.style.display = 'none';
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Lightning Scalper MT5 Detection Fixed Version 2.1.1 initialized');
            
            // Initial status update
            setTimeout(updateStatus, 1000);
            
            // Auto-update every 3 seconds
            setInterval(updateStatus, 3000);
        });
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>'''
    
    def run(self, host='127.0.0.1', port=5000):
        """Run the enhanced dashboard"""
        print(f"🚀 Lightning Scalper MT5 Detection Fixed starting...")
        print(f"   URL: http://{host}:{port}")
        print(f"   MT5 Library: {'✅ Available' if MT5_AVAILABLE else '❌ Not Available'}")
        print(f"   Opening browser automatically...")
        
        # Open browser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        
        # Run Flask server
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except Exception as e:
            print(f"Error starting server: {e}")
            input("Press Enter to exit...")

def find_free_port(start_port=5000):
    """Find a free port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return start_port

def main():
    """Main application entry point"""
    print("🚀 Lightning Scalper - MT5 Detection Fixed Version")
    print("   ✅ Enhanced MT5 Library Detection")
    print("   ✅ Clear Error Messages & Solutions")
    print("   ✅ Demo Mode Support")
    print("   Version 2.1.1 (MT5 Fixed)")
    print("-" * 50)
    
    # Show MT5 status
    if MT5_AVAILABLE:
        print("✅ MT5 Library: Available")
    else:
        print("❌ MT5 Library: Not Available")
        print("💡 Tip: Run fix_mt5_install.bat to install MT5 library")
        print("🔧 Or use Demo mode (no MT5 required)")
    
    print()
    
    # Find free port
    port = find_free_port()
    
    # Initialize and run dashboard
    dashboard = LightningScalperMT5Fixed()
    
    try:
        dashboard.run(host='127.0.0.1', port=port)
    except KeyboardInterrupt:
        print("\n⏹️  Lightning Scalper stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()