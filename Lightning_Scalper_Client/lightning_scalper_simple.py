#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Simple Version (No SocketIO)
🚀 ทำงานได้แน่นอนใน .exe file

ลบ SocketIO ออก ใช้ Ajax polling แทน
Author: Phoenix Trading AI
Version: 1.0.2 (Simple)
"""

import json
import logging
import os
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

# Flask imports (no SocketIO)
from flask import Flask, render_template_string, request, jsonify

# MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MT5 not available - Demo mode only")

class SecurityManager:
    """🔐 Security Manager"""
    
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

class MT5Manager:
    """🔌 MetaTrader 5 Manager"""
    
    def __init__(self):
        self.connected = False
        self.account_info = {}
    
    def auto_detect_mt5(self) -> Tuple[bool, Dict]:
        """Auto-detect MT5"""
        if not MT5_AVAILABLE:
            return False, {"error": "MT5 not available"}
        
        try:
            if not mt5.initialize():
                return False, {"error": "MT5 initialization failed"}
            
            account_info = mt5.account_info()
            if account_info is None:
                return False, {"error": "No active MT5 account"}
            
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
            if not mt5.initialize():
                return False, {"error": "MT5 initialization failed"}
            
            if not mt5.login(login, password, server):
                error_code = mt5.last_error()
                return False, {"error": f"Login failed: {error_code}"}
            
            return self.auto_detect_mt5()
            
        except Exception as e:
            return False, {"error": f"Manual connection failed: {str(e)}"}

class ConfigManager:
    """⚙️ Configuration Manager"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.server_endpoint = "https://api.lightning-scalper.com"
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            "client_id": "",
            "api_key": "",
            "api_secret": "",
            "server_endpoint": self.server_endpoint,
            "risk_level": 3,
            "timeframes": ["M15", "H1"],
            "entry_method": "hybrid",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            "max_positions": 5,
            "risk_per_trade": 2.0,
            "daily_loss_limit": 5.0,
            "auto_trading": False,
            "demo_mode": True
        }
    
    def update_config(self, updates: Dict) -> bool:
        """Update configuration"""
        try:
            self.config.update(updates)
            return True
        except Exception as e:
            logging.error(f"Config update failed: {e}")
            return False

class LightningScalperDashboard:
    """🚀 Lightning Scalper Dashboard (Simple Version)"""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lightning_scalper_secret_key_2024'
        
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
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_html())
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            return jsonify({
                'mt5_connected': self.mt5_manager.connected,
                'is_trading': self.is_trading,
                'account_info': self.mt5_manager.account_info,
                'performance': self.performance_data,
                'config': self.config_manager.config,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/mt5/test', methods=['POST'])
        def test_mt5():
            """Test MT5 basic functionality"""
            success, message = self.mt5_manager.test_mt5_simple()
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
                    return jsonify({"success": False, "data": {"error": "No data provided"}})
                
                login = data.get('login')
                password = data.get('password') 
                server = data.get('server')
                
                if not all([login, password, server]):
                    return jsonify({"success": False, "data": {"error": "Missing required fields"}})
                
                success, result = self.mt5_manager.manual_connect(
                    int(login), str(password), str(server)
                )
                return jsonify({"success": success, "data": result})
                
            except Exception as e:
                return jsonify({"success": False, "data": {"error": f"Server error: {str(e)}"}})
        
        @self.app.route('/api/config/update', methods=['POST'])
        def update_config():
            try:
                updates = request.json
                if updates:
                    success = self.config_manager.update_config(updates)
                    return jsonify({"success": success})
                return jsonify({"success": False})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/api/trading/start', methods=['POST'])
        def start_trading():
            if not self.mt5_manager.connected:
                return jsonify({"success": False, "error": "MT5 not connected"})
            
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
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.2"
            })
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML template (No SocketIO)"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightning Scalper - AI Trading Dashboard</title>
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
            --shadow-hover: 0 20px 40px rgba(0,0,0,0.15);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary);
            min-height: 100vh;
            color: #333;
            background-attachment: fixed;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            z-index: -1;
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
            position: relative;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
        }
        .header h1 {
            font-size: 3.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-weight: 700;
            background: linear-gradient(45deg, #fff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header p {
            font-size: 1.2rem;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        .header small {
            opacity: 0.7;
            font-size: 0.9rem;
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
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--primary);
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }
        .card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-hover);
        }
        .card:hover::before {
            transform: scaleX(1);
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
            background-clip: text;
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
            position: relative;
            overflow: hidden;
        }
        .status-indicator::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        .status-indicator.pulse::before {
            animation: shimmer 2s infinite;
        }
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        .status-connected {
            background: var(--success);
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        .status-disconnected {
            background: var(--danger);
            color: white;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        .status-trading {
            background: var(--info);
            color: white;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            min-width: 140px;
        }
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        .btn:hover::before {
            left: 100%;
        }
        .btn i { margin-right: 10px; font-size: 1.1rem; }
        .btn-primary {
            background: var(--primary);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        .btn-success {
            background: var(--success);
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        .btn-success:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }
        .btn-danger {
            background: var(--danger);
            color: white;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        .btn-danger:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);
        }
        .btn-secondary {
            background: linear-gradient(135deg, #6c757d, #5a6268);
            color: white;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        .form-group {
            margin-bottom: 25px;
        }
        .form-label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #555;
            font-size: 1rem;
        }
        .form-control {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.9);
        }
        .form-control:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            background: white;
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
            max-width: 500px;
            position: relative;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: var(--shadow);
        }
        .close {
            position: absolute;
            right: 20px;
            top: 20px;
            font-size: 1.5rem;
            cursor: pointer;
            color: #aaa;
            transition: color 0.3s ease;
        }
        .close:hover { color: #333; }
        .account-details {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .auto-refresh {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--primary);
            color: white;
            padding: 12px 18px;
            border-radius: 25px;
            font-size: 0.9rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }
        
        /* New Settings Styles */
        .settings-card {
            grid-column: 1 / -1;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 25px;
        }
        .settings-column {
            display: flex;
            flex-direction: column;
        }
        .risk-slider-container {
            margin: 15px 0;
        }
        .risk-slider {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(to right, #4CAF50, #FFC107, #FF5722);
            outline: none;
            margin: 15px 0;
            -webkit-appearance: none;
            cursor: pointer;
        }
        .risk-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            border: 3px solid #667eea;
            transition: all 0.3s ease;
        }
        .risk-slider::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }
        .risk-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #666;
            margin-top: 8px;
        }
        .risk-labels span {
            font-weight: 500;
        }
        .risk-description {
            background: rgba(102, 126, 234, 0.1);
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9rem;
            border-left: 4px solid #667eea;
        }
        .timeframe-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            padding: 12px 15px;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }
        .checkbox-item:hover {
            background: rgba(102, 126, 234, 0.1);
            transform: translateY(-1px);
        }
        .checkbox-item input {
            margin-right: 8px;
            accent-color: #667eea;
            transform: scale(1.2);
        }
        .checkbox-item input:checked + .checkmark + span {
            color: #667eea;
            font-weight: 600;
        }
        .settings-actions {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid rgba(102, 126, 234, 0.1);
        }
        
        /* Performance Grid */
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
            border-radius: 15px;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.3);
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
            background-clip: text;
        }
        .stat-content {
            flex: 1;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .settings-grid { grid-template-columns: 1fr; }
            .performance-grid { grid-template-columns: repeat(2, 1fr); }
            .header h1 { font-size: 2.5rem; }
            .timeframe-grid { grid-template-columns: repeat(2, 1fr); }
            .settings-actions { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-bolt"></i> Lightning Scalper</h1>
            <p>Professional AI Trading Dashboard</p>
            <small>Simple Version - No WebSocket</small>
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
                
                <div id="account-info" class="account-details" style="display: none;">
                    <strong>Account:</strong> <span id="account-number">-</span><br>
                    <strong>Server:</strong> <span id="server">-</span><br>
                    <strong>Balance:</strong> $<span id="balance">0.00</span><br>
                    <strong>Equity:</strong> $<span id="equity">0.00</span>
                </div>
                
                <button class="btn btn-primary" onclick="testMT5()">
                    <i class="fas fa-flask"></i> Test MT5
                </button>
                
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
                
                <div class="account-details">
                    <strong>Status:</strong> <span id="system-status">Ready</span><br>
                    <strong>Last Update:</strong> <span id="last-update">-</span>
                </div>
                
                <button id="start-btn" class="btn btn-success" onclick="startTrading()" disabled>
                    <i class="fas fa-play"></i> Start Trading
                </button>
                
                <button id="stop-btn" class="btn btn-danger" onclick="stopTrading()" disabled>
                    <i class="fas fa-stop"></i> Stop Trading
                </button>
            </div>
        </div>
        
        <!-- Trading Settings Section -->
        <div class="card settings-card">
            <div class="card-header">
                <i class="fas fa-sliders-h card-icon"></i>
                <h3 class="card-title">AI Trading Settings</h3>
            </div>
            
            <div class="settings-grid">
                <div class="settings-column">
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-shield-alt"></i> Risk Level
                        </label>
                        <div class="risk-slider-container">
                            <input type="range" id="risk-level" class="risk-slider" min="1" max="5" value="3" oninput="updateRiskLevel(this.value)">
                            <div class="risk-labels">
                                <span>Ultra Safe</span>
                                <span>Conservative</span>
                                <span>Balanced</span>
                                <span>Aggressive</span>
                                <span>Max Risk</span>
                            </div>
                            <div id="risk-description" class="risk-description">
                                <strong>Balanced Mode:</strong> 2.0% risk per trade - Recommended for most traders
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-crosshairs"></i> Entry Method
                        </label>
                        <select id="entry-method" class="form-control" onchange="updateEntryMethod(this.value)">
                            <option value="precision">Precision Mode - Wait for perfect setups</option>
                            <option value="speed">Speed Mode - Quick market entries</option>
                            <option value="hybrid" selected>Hybrid Mode - Balanced approach</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-layer-group"></i> Max Positions
                        </label>
                        <input type="number" id="max-positions" class="form-control" value="5" min="1" max="20" onchange="updateMaxPositions(this.value)">
                    </div>
                </div>
                
                <div class="settings-column">
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-clock"></i> Active Timeframes
                        </label>
                        <div class="timeframe-grid">
                            <label class="checkbox-item">
                                <input type="checkbox" value="M1" onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>M1</span>
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" value="M5" onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>M5</span>
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" value="M15" checked onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>M15</span>
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" value="M30" onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>M30</span>
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" value="H1" checked onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>H1</span>
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" value="H4" onchange="updateTimeframes()"> 
                                <span class="checkmark"></span>
                                <span>H4</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-percentage"></i> Risk Per Trade (%)
                        </label>
                        <input type="number" id="risk-per-trade" class="form-control" value="2.0" min="0.1" max="10.0" step="0.1" onchange="updateRiskPerTrade(this.value)">
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">
                            <i class="fas fa-dollar-sign"></i> Daily Loss Limit (%)
                        </label>
                        <input type="number" id="daily-loss-limit" class="form-control" value="5.0" min="1.0" max="20.0" step="0.5" onchange="updateDailyLossLimit(this.value)">
                    </div>
                </div>
            </div>
            
            <div class="settings-actions">
                <button class="btn btn-primary" onclick="saveSettings()">
                    <i class="fas fa-save"></i> Save Settings
                </button>
                <button class="btn btn-secondary" onclick="resetSettings()">
                    <i class="fas fa-undo"></i> Reset to Default
                </button>
            </div>
        </div>
        
        <!-- Performance Dashboard -->
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-bar card-icon"></i>
                <h3 class="card-title">Live Performance</h3>
            </div>
            
            <div class="performance-grid">
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-exchange-alt"></i>
                    </div>
                    <div class="stat-content">
                        <div class="stat-value" id="total-trades">0</div>
                        <div class="stat-label">Total Trades</div>
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
                        <i class="fas fa-calendar-day"></i>
                    </div>
                    <div class="stat-content">
                        <div class="stat-value" id="daily-profit">$0.00</div>
                        <div class="stat-label">Daily Profit</div>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <div class="stat-content">
                        <div class="stat-value" id="total-profit">$0.00</div>
                        <div class="stat-label">Total Profit</div>
                    </div>
                </div>
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

    <!-- Auto Refresh Indicator -->
    <div class="auto-refresh">
        <i class="fas fa-sync-alt"></i> Auto Refresh: ON
    </div>

    <script>
        let isConnected = false;
        let isTrading = false;
        
        // Update status every 5 seconds (reduced from 3)
        async function updateStatus() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
                
                const response = await fetch('/api/status', {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                
                updateMT5Status(data.account_info || {}, data.mt5_connected);
                updateTradingStatus(data.is_trading);
                
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Status update failed:', error);
                    document.getElementById('last-update').textContent = 'Update failed';
                }
            }
        }
        
        // Test MT5 basic functionality
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
                
                if (result.success) {
                    showNotification('MT5 Test: ' + result.message, 'success');
                } else {
                    showNotification('MT5 Test Failed: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('Test Error: ' + error.message, 'error');
            } finally {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }
        }
        
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
                document.getElementById('equity').textContent = (accountInfo.equity || 0).toFixed(2);
                
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
                statusEl.className = 'status-indicator status-trading';
                statusEl.innerHTML = '<i class="fas fa-play-circle"></i><span>Trading Active</span>';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                document.getElementById('system-status').textContent = 'AI Trading Active';
            } else {
                statusEl.className = 'status-indicator status-disconnected';
                statusEl.innerHTML = '<i class="fas fa-pause-circle"></i><span>Trading Stopped</span>';
                startBtn.disabled = !isConnected;
                stopBtn.disabled = true;
                document.getElementById('system-status').textContent = 'Ready';
            }
        }
        
        async function autoDetectMT5() {
            const btn = event.target;
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
            btn.disabled = true;
            
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
                
                const response = await fetch('/api/mt5/auto-detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                const result = await response.json();
                
                if (result.success) {
                    updateMT5Status(result.data, true);
                    showNotification('MT5 connected successfully! 🎉', 'success');
                } else {
                    const errorMsg = result.data?.error || 'Unknown error';
                    showNotification('MT5 Detection Failed: ' + errorMsg, 'error');
                    
                    // Show helpful message
                    if (errorMsg.includes('initialize')) {
                        showNotification('💡 Make sure MT5 is running and you are logged in', 'error');
                    }
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    showNotification('Connection timeout - Please check MT5 and try again', 'error');
                } else {
                    showNotification('Connection Error: ' + error.message, 'error');
                }
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
                showNotification('Please fill all fields', 'error');
                return;
            }
            
            const btn = event.target;
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
            btn.disabled = true;
            
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
                
                const response = await fetch('/api/mt5/manual-connect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        login: parseInt(login),
                        password: password,
                        server: server
                    }),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                const result = await response.json();
                
                if (result.success) {
                    updateMT5Status(result.data, true);
                    closeModal();
                    showNotification('MT5 connected successfully! 🎉', 'success');
                } else {
                    const errorMsg = result.data?.error || 'Unknown error';
                    showNotification('Connection Failed: ' + errorMsg, 'error');
                    
                    // Show helpful messages
                    if (errorMsg.includes('Login failed')) {
                        showNotification('💡 Check your login, password, and server name', 'error');
                    }
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    showNotification('Connection timeout - Please check your details and try again', 'error');
                } else {
                    showNotification('Connection Error: ' + error.message, 'error');
                }
            } finally {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
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
                    updateStatus(); // Immediate status update
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
                    updateStatus(); // Immediate status update
                } else {
                    showNotification('Failed to stop trading: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showNotification('Error: ' + error.message, 'error');
            }
        }
        
        function showManualSetup() {
            document.getElementById('manual-modal').style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('manual-modal').style.display = 'none';
        }
        
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
                transition: all 0.3s ease;
                ${type === 'success' ? 'background: var(--success);' : 'background: var(--danger);'}
            `;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
        // Settings Functions
        const riskDescriptions = {
            1: "Ultra Safe Mode: 0.5% risk per trade - Maximum capital protection",
            2: "Conservative Mode: 1.0% risk per trade - Low risk approach", 
            3: "Balanced Mode: 2.0% risk per trade - Recommended for most traders",
            4: "Aggressive Mode: 3.0% risk per trade - Higher profit potential",
            5: "Max Risk Mode: 5.0% risk per trade - For experienced traders only"
        };
        
        function updateRiskLevel(level) {
            const description = riskDescriptions[level];
            document.getElementById('risk-description').innerHTML = `<strong>${description.split(':')[0]}:</strong> ${description.split(':')[1]}`;
            
            // Update risk per trade automatically
            const riskPercentages = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 5.0};
            document.getElementById('risk-per-trade').value = riskPercentages[level];
            
            saveSettingsAuto();
        }
        
        function updateEntryMethod(method) {
            saveSettingsAuto();
        }
        
        function updateMaxPositions(value) {
            saveSettingsAuto();
        }
        
        function updateRiskPerTrade(value) {
            saveSettingsAuto();
        }
        
        function updateDailyLossLimit(value) {
            saveSettingsAuto();
        }
        
        function updateTimeframes() {
            saveSettingsAuto();
        }
        
        function getTimeframes() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"][value^="M"], input[type="checkbox"][value^="H"]');
            const selected = [];
            checkboxes.forEach(cb => {
                if (cb.checked) selected.push(cb.value);
            });
            return selected;
        }
        
        async function saveSettingsAuto() {
            const settings = {
                risk_level: parseInt(document.getElementById('risk-level').value),
                entry_method: document.getElementById('entry-method').value,
                max_positions: parseInt(document.getElementById('max-positions').value),
                risk_per_trade: parseFloat(document.getElementById('risk-per-trade').value),
                daily_loss_limit: parseFloat(document.getElementById('daily-loss-limit').value),
                timeframes: getTimeframes()
            };
            
            try {
                const response = await fetch('/api/config/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settings)
                });
                
                const result = await response.json();
                if (!result.success) {
                    console.error('Auto-save failed');
                }
            } catch (error) {
                console.error('Auto-save error:', error);
            }
        }
        
        async function saveSettings() {
            await saveSettingsAuto();
            showNotification('Settings saved successfully!', 'success');
        }
        
        function resetSettings() {
            // Reset to default values
            document.getElementById('risk-level').value = 3;
            document.getElementById('entry-method').value = 'hybrid';
            document.getElementById('max-positions').value = 5;
            document.getElementById('risk-per-trade').value = 2.0;
            document.getElementById('daily-loss-limit').value = 5.0;
            
            // Reset timeframes
            const checkboxes = document.querySelectorAll('input[type="checkbox"][value^="M"], input[type="checkbox"][value^="H"]');
            checkboxes.forEach(cb => {
                cb.checked = (cb.value === 'M15' || cb.value === 'H1');
            });
            
            updateRiskLevel(3);
            showNotification('Settings reset to default!', 'success');
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Lightning Scalper Simple Dashboard initialized');
            
            // Initial status update
            setTimeout(updateStatus, 1000);
            
            // Auto-update every 5 seconds (reduced frequency)
            setInterval(updateStatus, 5000);
            
            // Auto-test MT5 on startup (instead of auto-detect)
            setTimeout(testMT5, 3000);
        });
    </script>
</body>
</html>'''
    
    def run(self, host='127.0.0.1', port=5000):
        """Run the dashboard server"""
        print(f"🚀 Lightning Scalper Simple Dashboard starting...")
        print(f"   URL: http://{host}:{port}")
        print(f"   Opening browser automatically...")
        
        # Open browser after a short delay
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        
        # Run Flask server (no SocketIO)
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
    print("🚀 Lightning Scalper - Simple Version")
    print("   No SocketIO - Works with .exe")
    print("   Version 1.0.2 (Simple)")
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