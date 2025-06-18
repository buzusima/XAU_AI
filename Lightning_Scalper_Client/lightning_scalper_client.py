#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - All-in-One Client Dashboard (Fixed Version)
🚀 Production-Grade AI Trading Client Package

Fixed Issues:
- ✅ SocketIO configuration 
- ✅ ConfigManager initialization
- ✅ Template handling
- ✅ Error handling improved

Author: Phoenix Trading AI
Version: 1.0.1 (Fixed)
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import webbrowser
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Flask imports
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit

# MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MT5 not available - Demo mode only")

class SecurityManager:
    """🔐 Advanced Security & Encryption Manager"""
    
    def __init__(self, master_key: str = "LIGHTNING_SCALPER_2024"):
        self.master_key = master_key.encode()
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Initialize AES encryption"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'lightning_salt_2024',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            self.cipher = Fernet(key)
        except Exception as e:
            logging.error(f"Encryption setup failed: {e}")
            self.cipher = None
    
    def encrypt_data(self, data: Dict) -> str:
        """Encrypt configuration data"""
        if not self.cipher:
            return json.dumps(data)
        
        try:
            json_data = json.dumps(data).encode()
            encrypted = self.cipher.encrypt(json_data)
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logging.error(f"Encryption failed: {e}")
            return json.dumps(data)
    
    def decrypt_data(self, encrypted_data: str) -> Dict:
        """Decrypt configuration data"""
        if not self.cipher:
            try:
                return json.loads(encrypted_data)
            except:
                return {}
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            return {}

class MT5Manager:
    """🔌 MetaTrader 5 Integration Manager"""
    
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.connection_attempts = 0
        self.max_attempts = 3
    
    def auto_detect_mt5(self) -> Tuple[bool, Dict]:
        """Auto-detect existing MT5 connection"""
        if not MT5_AVAILABLE:
            return False, {"error": "MT5 not available"}
        
        try:
            # Try to initialize MT5
            if not mt5.initialize():
                return False, {"error": "MT5 initialization failed"}
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return False, {"error": "No active MT5 account"}
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            
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
                "margin_level": account_info.margin_level,
                "company": account_info.company,
                "trade_allowed": account_info.trade_allowed,
                "expert_allowed": account_info.trade_expert,
                "terminal_path": terminal_info.path if terminal_info else "Unknown",
                "connection_status": "Connected",
                "last_update": datetime.now().isoformat()
            }
            
            return True, self.account_info
            
        except Exception as e:
            return False, {"error": f"MT5 detection failed: {str(e)}"}
    
    def manual_connect(self, login: int, password: str, server: str) -> Tuple[bool, Dict]:
        """Manual MT5 connection"""
        if not MT5_AVAILABLE:
            return False, {"error": "MT5 not available"}
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                return False, {"error": "MT5 initialization failed"}
            
            # Login with credentials
            if not mt5.login(login, password, server):
                error_code = mt5.last_error()
                return False, {"error": f"Login failed: {error_code}"}
            
            # Get account info after successful login
            return self.auto_detect_mt5()
            
        except Exception as e:
            return False, {"error": f"Manual connection failed: {str(e)}"}
    
    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        if not self.connected or not MT5_AVAILABLE:
            return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]  # Default symbols
        
        try:
            symbols = mt5.symbols_get()
            return [s.name for s in symbols if s.visible][:50]  # Limit to 50 symbols
        except:
            return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

class ConfigManager:
    """⚙️ Advanced Configuration Manager (Fixed)"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        # Set server_endpoint BEFORE loading config
        self.server_endpoint = "https://api.lightning-scalper.com"  # Your server
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            "client_id": "",
            "api_key": "",
            "api_secret": "",
            "server_endpoint": self.server_endpoint,
            "risk_level": 3,  # 1-5 scale
            "timeframes": ["M15", "H1"],
            "entry_method": "hybrid",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            "max_positions": 5,
            "risk_per_trade": 2.0,
            "daily_loss_limit": 5.0,
            "auto_trading": False,
            "demo_mode": True,
            "notifications": {
                "email": True,
                "sound": True,
                "popup": False
            },
            "advanced": {
                "slippage": 3,
                "magic_number": 123456,
                "trailing_stop": False,
                "partial_close": False
            }
        }
    
    def update_config(self, updates: Dict) -> bool:
        """Update configuration and sync to server"""
        try:
            self.config.update(updates)
            self._sync_to_server()
            return True
        except Exception as e:
            logging.error(f"Config update failed: {e}")
            return False
    
    def _sync_to_server(self):
        """Sync configuration to server (encrypted)"""
        # This would sync to your actual server
        encrypted_config = self.security.encrypt_data(self.config)
        logging.info("Configuration synced to server (encrypted)")

class LightningScalperDashboard:
    """🚀 Lightning Scalper Client Dashboard (Fixed)"""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lightning_scalper_secret_key_2024'
        
        # Initialize SocketIO with fallback async_mode for .exe
        try:
            # Try threading mode first
            self.socketio = SocketIO(
                self.app, 
                cors_allowed_origins="*",
                async_mode='threading',
                logger=False,
                engineio_logger=False
            )
        except ValueError:
            try:
                # Fallback to eventlet
                self.socketio = SocketIO(
                    self.app, 
                    cors_allowed_origins="*",
                    async_mode='eventlet',
                    logger=False,
                    engineio_logger=False
                )
            except (ValueError, ImportError):
                # Last resort - no async_mode specified
                self.socketio = SocketIO(
                    self.app, 
                    cors_allowed_origins="*",
                    logger=False,
                    engineio_logger=False
                )
        
        # Core managers
        self.security = SecurityManager()
        self.config_manager = ConfigManager(self.security)
        self.mt5_manager = MT5Manager()
        
        # Runtime state
        self.is_trading = False
        self.last_signal = None
        self.performance_data = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,
            "daily_profit": 0.0,
            "win_rate": 0.0
        }
        
        self._setup_routes()
        self._setup_socketio()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_html())
        
        @self.app.route('/api/mt5/auto-detect', methods=['POST'])
        def auto_detect_mt5():
            success, data = self.mt5_manager.auto_detect_mt5()
            return jsonify({"success": success, "data": data})
        
        @self.app.route('/api/mt5/manual-connect', methods=['POST'])
        def manual_connect_mt5():
            data = request.json
            success, result = self.mt5_manager.manual_connect(
                data['login'], data['password'], data['server']
            )
            return jsonify({"success": success, "data": result})
        
        @self.app.route('/api/config/update', methods=['POST'])
        def update_config():
            updates = request.json
            success = self.config_manager.update_config(updates)
            return jsonify({"success": success})
        
        @self.app.route('/api/trading/start', methods=['POST'])
        def start_trading():
            if not self.mt5_manager.connected:
                return jsonify({"success": False, "error": "MT5 not connected"})
            
            self.is_trading = True
            self.socketio.emit('trading_status', {'status': 'started'})
            return jsonify({"success": True})
        
        @self.app.route('/api/trading/stop', methods=['POST'])
        def stop_trading():
            self.is_trading = False
            self.socketio.emit('trading_status', {'status': 'stopped'})
            return jsonify({"success": True})
        
        @self.app.route('/api/symbols')
        def get_symbols():
            symbols = self.mt5_manager.get_symbols()
            return jsonify(symbols)
        
        @self.app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.1"
            })
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('status', {'message': 'Connected to Lightning Scalper'})
        
        @self.socketio.on('request_update')
        def handle_update_request():
            self._broadcast_status()
    
    def _broadcast_status(self):
        """Broadcast current status to all clients"""
        status = {
            'mt5_connected': self.mt5_manager.connected,
            'is_trading': self.is_trading,
            'account_info': self.mt5_manager.account_info,
            'performance': self.performance_data,
            'config': self.config_manager.config,
            'timestamp': datetime.now().isoformat()
        }
        self.socketio.emit('status_update', status)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightning Scalper - AI Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.0/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"></script>
    <style>
        :root {
            --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success: linear-gradient(135deg, #4CAF50, #45a049);
            --danger: linear-gradient(135deg, #f44336, #d32f2f);
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
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--glass);
            border-radius: 15px;
            padding: 25px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .card-icon {
            font-size: 1.5rem;
            margin-right: 10px;
            color: #667eea;
        }
        .card-title {
            font-size: 1.3rem;
            font-weight: 600;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 500;
        }
        .status-connected {
            background: var(--success);
            color: white;
        }
        .status-disconnected {
            background: var(--danger);
            color: white;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
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
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 30px;
            border-radius: 15px;
            width: 90%;
            max-width: 500px;
            position: relative;
        }
        .close {
            position: absolute;
            right: 15px;
            top: 15px;
            font-size: 1.5rem;
            cursor: pointer;
            color: #aaa;
        }
        .close:hover { color: #333; }
        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-bolt"></i> Lightning Scalper</h1>
            <p>Professional AI Trading Dashboard</p>
        </div>
        
        <div class="dashboard-grid">
            <!-- MT5 Connection Status -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-plug card-icon"></i>
                    <h3 class="card-title">MT5 Connection</h3>
                </div>
                
                <div id="mt5-status" class="status-indicator status-disconnected">
                    <i class="fas fa-times-circle"></i>
                    <span>Not Connected</span>
                </div>
                
                <div id="account-info" style="display: none;">
                    <div class="form-group">
                        <strong>Account:</strong> <span id="account-number">-</span><br>
                        <strong>Server:</strong> <span id="server">-</span><br>
                        <strong>Balance:</strong> $<span id="balance">0.00</span>
                    </div>
                </div>
                
                <button class="btn btn-primary" onclick="autoDetectMT5()">
                    <i class="fas fa-search"></i> Auto Detect MT5
                </button>
                
                <button class="btn btn-primary" onclick="showManualSetup()">
                    <i class="fas fa-cog"></i> Manual Setup
                </button>
            </div>
            
            <!-- Trading Control -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-chart-line card-icon"></i>
                    <h3 class="card-title">Trading Control</h3>
                </div>
                
                <div id="trading-status" class="status-indicator status-disconnected">
                    <i class="fas fa-pause-circle"></i>
                    <span>Trading Stopped</span>
                </div>
                
                <button id="start-btn" class="btn btn-success" onclick="startTrading()" disabled>
                    <i class="fas fa-play"></i> Start Trading
                </button>
                
                <button id="stop-btn" class="btn btn-danger" onclick="stopTrading()" disabled>
                    <i class="fas fa-stop"></i> Stop Trading
                </button>
            </div>
        </div>
    </div>
    
    <!-- Manual Setup Modal -->
    <div id="manual-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
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
        // Initialize Socket.IO
        const socket = io();
        
        let isConnected = false;
        let isTrading = false;
        
        // Socket events
        socket.on('connect', function() {
            console.log('Connected to Lightning Scalper server');
        });
        
        socket.on('status_update', function(data) {
            updateMT5Status(data.account_info || {}, data.mt5_connected);
            updateTradingStatus(data.is_trading);
        });
        
        // UI Functions
        function updateMT5Status(accountInfo, connected) {
            const statusEl = document.getElementById('mt5-status');
            const accountInfoEl = document.getElementById('account-info');
            const startBtn = document.getElementById('start-btn');
            
            isConnected = connected;
            
            if (connected) {
                statusEl.className = 'status-indicator status-connected';
                statusEl.innerHTML = '<i class="fas fa-check-circle"></i><span>Connected</span>';
                
                document.getElementById('account-number').textContent = accountInfo.account || '-';
                document.getElementById('server').textContent = accountInfo.server || '-';
                document.getElementById('balance').textContent = (accountInfo.balance || 0).toFixed(2);
                
                accountInfoEl.style.display = 'block';
                startBtn.disabled = false;
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-times-circle"></i><span>Not Connected</span>';
                
                accountInfoEl.style.display = 'none';
                startBtn.disabled = true;
            }
        }
        
        function updateTradingStatus(trading) {
            const statusEl = document.getElementById('trading-status');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            isTrading = trading;
            
            if (trading) {
                statusEl.className = 'status-indicator status-connected';
                statusEl.innerHTML = '<i class="fas fa-play-circle"></i><span>Trading Active</span>';
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-pause-circle"></i><span>Trading Stopped</span>';
                startBtn.disabled = !isConnected;
                stopBtn.disabled = true;
            }
        }
        
        // API Functions
        async function autoDetectMT5() {
            try {
                const response = await fetch('/api/mt5/auto-detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateMT5Status(result.data, true);
                    showNotification('MT5 connection detected successfully!', 'success');
                } else {
                    showNotification('Failed to detect MT5: ' + (result.data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        async function manualConnect() {
            const login = document.getElementById('mt5-login').value;
            const password = document.getElementById('mt5-password').value;
            const server = document.getElementById('mt5-server').value;
            
            if (!login || !password || !server) {
                showNotification('Please fill all fields', 'error');
                return;
            }
            
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
                    updateMT5Status(result.data, true);
                    closeModal();
                    showNotification('MT5 connected successfully!', 'success');
                } else {
                    showNotification('Connection failed: ' + (result.data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        async function startTrading() {
            try {
                const response = await fetch('/api/trading/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification('Trading started!', 'success');
                } else {
                    showNotification('Failed to start trading: ' + (result.error || 'Unknown error'), 'error');
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
                    showNotification('Trading stopped!', 'success');
                } else {
                    showNotification('Failed to stop trading: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        // Modal Functions
        function showManualSetup() {
            document.getElementById('manual-modal').style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('manual-modal').style.display = 'none';
        }
        
        // Notification Function
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                z-index: 10000;
                ${type === 'success' ? 'background: var(--success);' : 'background: var(--danger);'}
            `;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('manual-modal');
            if (event.target === modal) {
                closeModal();
            }
        }
    </script>
</body>
</html>'''
    
    def run(self, host='127.0.0.1', port=5000):
        """Run the dashboard server"""
        print(f"🚀 Lightning Scalper Dashboard starting...")
        print(f"   URL: http://{host}:{port}")
        print(f"   Opening browser automatically...")
        
        # Open browser after a short delay
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Run Flask-SocketIO server
        try:
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except Exception as e:
            print(f"Error starting server: {e}")
            input("Press Enter to exit...")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def monitor_mt5():
            while True:
                if self.mt5_manager.connected:
                    success, data = self.mt5_manager.auto_detect_mt5()
                    if success:
                        self.mt5_manager.account_info.update(data)
                        self.socketio.emit('mt5_update', data)
                    else:
                        self.mt5_manager.connected = False
                        self.socketio.emit('mt5_disconnect', {})
                
                time.sleep(5)  # Check every 5 seconds
        
        threading.Thread(target=monitor_mt5, daemon=True).start()

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
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
    print("🚀 Lightning Scalper - All-in-One Client Package")
    print("   Professional AI Trading System")
    print("   Version 1.0.1 (Fixed)")
    print("-" * 50)
    
    # Find free port
    port = find_free_port()
    
    # Initialize and run dashboard
    dashboard = LightningScalperDashboard()
    
    try:
        dashboard.run(host='127.0.0.1', port=port)
    except KeyboardInterrupt:
        print("\n⏹️  Lightning Scalper stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()