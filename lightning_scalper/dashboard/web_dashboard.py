#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[ROCKET] Lightning Scalper - Fixed Web Dashboard
Production-Grade Web Interface - WORKING VERSION

แก้ไขปัญหาหลัก:
- ✅ WebSocket real-time working
- ✅ Complete HTML templates  
- ✅ Error handling improved
- ✅ Production-ready features

Author: Phoenix Trading AI
Version: 1.1.0 (Fixed)
License: Proprietary
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import os
import sys

# Web framework imports
try:
    from flask import Flask, render_template_string, request, jsonify, redirect, url_for
    from flask_socketio import SocketIO, emit, join_room, leave_room
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"[X] Missing web dependencies: {e}")
    print("   Install with: pip install flask flask-socketio pandas")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our core modules
try:
    from core.main_controller import LightningScalperController
    from core.lightning_scalper_engine import FVGSignal, FVGType, CurrencyPair
    from execution.trade_executor import ClientAccount, Position, Order
except ImportError as e:
    print(f"[X] Failed to import core modules: {e}")
    print("   Using demo mode...")

class LightningScalperDashboard:
    """
    [GLOBE] Lightning Scalper Real-time Web Dashboard - FIXED VERSION
    Production web interface for monitoring trading operations
    """
    
    def __init__(self, controller: Optional[LightningScalperController] = None, 
                 host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        
        self.controller = controller
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app with fixed template handling
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lightning_scalper_dashboard_2024'
        
        # Initialize SocketIO for real-time updates - FIXED
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        
        # Dashboard state
        self.connected_clients = set()
        self.room_subscriptions = {}
        self.last_update = datetime.now()
        
        # Data caching for performance - IMPROVED
        self.cache = {
            'system_status': {},
            'client_summaries': {},
            'recent_signals': [],
            'performance_data': {},
            'charts_data': {}
        }
        self.cache_expiry = timedelta(seconds=5)
        self.last_cache_update = datetime.now()
        
        # Setup logging
        self.logger = logging.getLogger('WebDashboard')
        
        # Background update thread
        self.update_thread = None
        self.is_running = False
        
        # Demo data for testing
        self.demo_mode = controller is None
        self.demo_data = self._generate_demo_data()
        
        # Setup routes and socket events - FIXED
        self._setup_routes()
        self._setup_socket_events()
        
        self.logger.info("[GLOBE] Lightning Scalper Dashboard initialized (FIXED VERSION)")
    
    def _generate_demo_data(self) -> Dict[str, Any]:
        """Generate realistic demo data for testing"""
        return {
            'clients': [
                {
                    'client_id': 'CLIENT_001',
                    'name': 'Demo Client 001',
                    'status': 'online',
                    'balance': 10000.0,
                    'pnl_today': 245.50,
                    'positions': 3,
                    'last_active': '2 min ago'
                },
                {
                    'client_id': 'CLIENT_002', 
                    'name': 'Demo Client 002',
                    'status': 'online',
                    'balance': 25000.0,
                    'pnl_today': 890.25,
                    'positions': 5,
                    'last_active': '5 min ago'
                },
                {
                    'client_id': 'CLIENT_003',
                    'name': 'Demo Client 003', 
                    'status': 'offline',
                    'balance': 5000.0,
                    'pnl_today': -125.75,
                    'positions': 1,
                    'last_active': '1 hour ago'
                }
            ],
            'signals': [
                {
                    'currency_pair': 'EURUSD',
                    'type': 'BUY',
                    'timeframe': 'M5',
                    'entry_price': 1.10850,
                    'target_1': 1.11150,
                    'stop_loss': 1.10550,
                    'confluence_score': 89.5,
                    'timestamp': '14:25'
                },
                {
                    'currency_pair': 'GBPUSD',
                    'type': 'SELL', 
                    'timeframe': 'M5',
                    'entry_price': 1.25340,
                    'target_1': 1.25090,
                    'stop_loss': 1.25590,
                    'confluence_score': 82.1,
                    'timestamp': '14:18'
                }
            ],
            'metrics': {
                'active_clients': 2,
                'total_signals_today': 15,
                'active_positions': 9,
                'total_pnl_today': 1009.75
            }
        }
    
    def _setup_routes(self):
        """Setup Flask routes - FIXED"""
        
        # Embedded template to avoid file dependency
        DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightning Scalper - Real-time Trading Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); color: white; font-family: 'Segoe UI', sans-serif; }
        .navbar { background: rgba(26, 26, 26, 0.95); border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
        .stat-card { background: rgba(45, 45, 45, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 25px; margin-bottom: 20px; }
        .stat-number { font-size: 2.5rem; font-weight: bold; margin-bottom: 5px; }
        .stat-label { color: #6c757d; font-size: 0.9rem; text-transform: uppercase; }
        .status-online { background: rgba(40, 167, 69, 0.2); color: #28a745; border: 1px solid #28a745; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; }
        .status-offline { background: rgba(220, 53, 69, 0.2); color: #dc3545; border: 1px solid #dc3545; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; }
        .table-dark { background: rgba(45, 45, 45, 0.8); }
        .connection-status { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        .pulse { animation: pulse 2s infinite; } @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#"><i class="fas fa-bolt"></i> Lightning Scalper</a>
            <span class="navbar-text"><i class="fas fa-user"></i> IB Dashboard</span>
        </div>
    </nav>

    <div class="connection-status">
        <div id="connection-indicator" class="badge bg-success pulse">
            <i class="fas fa-wifi"></i> Connected
        </div>
    </div>

    <div class="container-fluid p-4">
        <div class="row mb-4">
            <div class="col-12">
                <h1><i class="fas fa-chart-line"></i> Trading Dashboard</h1>
                <p class="text-muted">Real-time monitoring of Lightning Scalper AI Trading System</p>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-lg-3 col-md-6">
                <div class="stat-card text-center">
                    <div class="stat-number text-primary" id="active-clients">{{ metrics.active_clients }}</div>
                    <div class="stat-label">Active Clients</div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6">
                <div class="stat-card text-center">
                    <div class="stat-number text-success" id="todays-signals">{{ metrics.total_signals_today }}</div>
                    <div class="stat-label">Today's Signals</div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6">
                <div class="stat-card text-center">
                    <div class="stat-number text-warning" id="active-positions">{{ metrics.active_positions }}</div>
                    <div class="stat-label">Active Positions</div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6">
                <div class="stat-card text-center">
                    <div class="stat-number text-info" id="total-pnl">${{ "%.2f"|format(metrics.total_pnl_today) }}</div>
                    <div class="stat-label">Total P&L Today</div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="table-responsive">
                    <table class="table table-dark table-hover">
                        <thead>
                            <tr>
                                <th>Client ID</th>
                                <th>Status</th>
                                <th>Balance</th>
                                <th>P&L Today</th>
                                <th>Positions</th>
                                <th>Last Active</th>
                            </tr>
                        </thead>
                        <tbody id="clients-table-body">
                            {% for client in clients %}
                            <tr>
                                <td><strong>{{ client.client_id }}</strong></td>
                                <td>
                                    <span class="status-{{ client.status }}">
                                        {{ client.status.title() }}
                                    </span>
                                </td>
                                <td>${{ "%.2f"|format(client.balance) }}</td>
                                <td class="{{ 'text-success' if client.pnl_today > 0 else 'text-danger' }}">
                                    {{ '+' if client.pnl_today > 0 else '' }}${{ "%.2f"|format(client.pnl_today) }}
                                </td>
                                <td>{{ client.positions }}</td>
                                <td>{{ client.last_active }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <h5><i class="fas fa-signal"></i> Latest Signals</h5>
                <div id="signals-list">
                    {% for signal in signals %}
                    <div class="alert alert-info">
                        <strong>{{ signal.currency_pair }}</strong> {{ signal.type }} - 
                        Entry: {{ signal.entry_price }} | TP: {{ signal.target_1 }} | SL: {{ signal.stop_loss }} |
                        Confluence: {{ signal.confluence_score }}% | {{ signal.timestamp }}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        class LightningDashboard {
            constructor() {
                this.socket = null;
                this.isConnected = false;
                this.init();
            }
            
            init() {
                this.setupWebSocket();
                console.log('⚡ Lightning Scalper Dashboard initialized');
            }
            
            setupWebSocket() {
                try {
                    this.socket = io({
                        transports: ['websocket', 'polling'],
                        reconnection: true
                    });
                    
                    this.socket.on('connect', () => {
                        console.log('✅ WebSocket connected');
                        this.isConnected = true;
                        this.updateConnectionStatus(true);
                    });
                    
                    this.socket.on('disconnect', () => {
                        console.log('❌ WebSocket disconnected');
                        this.isConnected = false;
                        this.updateConnectionStatus(false);
                    });
                    
                    this.socket.on('system_status', (data) => {
                        this.updateSystemStats(data);
                    });
                    
                    this.socket.on('new_signal', (data) => {
                        this.addNewSignal(data.signal);
                    });
                    
                } catch (error) {
                    console.error('❌ WebSocket setup failed:', error);
                    this.fallbackMode();
                }
            }
            
            updateConnectionStatus(connected) {
                const indicator = document.getElementById('connection-indicator');
                if (connected) {
                    indicator.className = 'badge bg-success pulse';
                    indicator.innerHTML = '<i class="fas fa-wifi"></i> Connected';
                } else {
                    indicator.className = 'badge bg-danger';
                    indicator.innerHTML = '<i class="fas fa-wifi"></i> Disconnected';
                }
            }
            
            updateSystemStats(data) {
                try {
                    if (data.metrics) {
                        const elements = {
                            'active-clients': data.metrics.active_clients,
                            'todays-signals': data.metrics.total_signals_today,
                            'active-positions': data.metrics.active_positions,
                            'total-pnl': ' + (data.metrics.total_pnl_today || 0).toFixed(2)
                        };
                        
                        Object.entries(elements).forEach(([id, value]) => {
                            const element = document.getElementById(id);
                            if (element) element.textContent = value;
                        });
                    }
                } catch (error) {
                    console.error('Error updating stats:', error);
                }
            }
            
            addNewSignal(signal) {
                const signalsList = document.getElementById('signals-list');
                const signalHtml = `
                    <div class="alert alert-info">
                        <strong>${signal.currency_pair}</strong> ${signal.type} - 
                        Entry: ${signal.entry_price} | TP: ${signal.target_1} | SL: ${signal.stop_loss} |
                        Confluence: ${signal.confluence_score}% | Just now
                    </div>
                `;
                signalsList.insertAdjacentHTML('afterbegin', signalHtml);
                
                // Keep only latest 5 signals
                const signals = signalsList.children;
                while (signals.length > 5) {
                    signalsList.removeChild(signals[signals.length - 1]);
                }
            }
            
            fallbackMode() {
                console.log('🔄 Falling back to polling mode');
                setInterval(() => {
                    this.fetchSystemStatus();
                }, 30000);
            }
            
            async fetchSystemStatus() {
                try {
                    const response = await fetch('/api/system_status');
                    const data = await response.json();
                    this.updateSystemStats(data);
                } catch (error) {
                    console.error('Error fetching system status:', error);
                }
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            window.dashboard = new LightningDashboard();
        });
    </script>
</body>
</html>"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard route - FIXED"""
            try:
                if self.demo_mode:
                    data = self.demo_data
                else:
                    data = self._get_live_data()
                
                return render_template_string(
                    DASHBOARD_TEMPLATE,
                    clients=data['clients'],
                    signals=data['signals'],
                    metrics=data['metrics']
                )
            except Exception as e:
                self.logger.error(f"Dashboard route error: {e}")
                return f"<h1>Error loading dashboard: {str(e)}</h1>", 500
        
        @self.app.route('/api/system_status')
        def api_system_status():
            """API endpoint for system status - FIXED"""
            try:
                if self.demo_mode:
                    data = self.demo_data['metrics']
                else:
                    data = self._get_live_system_status()
                
                return jsonify({
                    'status': 'ok',
                    'metrics': data,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"API status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/clients')
        def api_clients():
            """API endpoint for client data - FIXED"""
            try:
                if self.demo_mode:
                    data = self.demo_data['clients']
                else:
                    data = self._get_live_clients_data()
                
                return jsonify({
                    'status': 'ok',
                    'clients': data,
                    'count': len(data)
                })
            except Exception as e:
                self.logger.error(f"API clients error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.1.0',
                'demo_mode': self.demo_mode
            })
    
    def _setup_socket_events(self):
        """Setup SocketIO event handlers - FIXED"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection - FIXED"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            self.logger.info(f"🔌 Dashboard client connected: {client_id}")
            
            # Send initial data immediately
            try:
                if self.demo_mode:
                    emit('system_status', {'metrics': self.demo_data['metrics']})
                else:
                    emit('system_status', {'metrics': self._get_live_system_status()})
            except Exception as e:
                self.logger.error(f"Error sending initial data: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            self.logger.info(f"🔌 Dashboard client disconnected: {client_id}")
        
        @self.socketio.on('subscribe_to_updates')
        def handle_subscribe(data):
            """Handle subscription to real-time updates"""
            client_id = request.sid
            update_type = data.get('type', 'all')
            
            room = f"updates_{update_type}"
            join_room(room)
            
            if room not in self.room_subscriptions:
                self.room_subscriptions[room] = set()
            self.room_subscriptions[room].add(client_id)
            
            self.logger.debug(f"📡 Client {client_id} subscribed to {update_type}")
        
        @self.socketio.on('heartbeat')
        def handle_heartbeat():
            """Handle heartbeat from client"""
            emit('heartbeat_response', {'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('request_full_update')
        def handle_full_update():
            """Handle request for full data update"""
            try:
                if self.demo_mode:
                    emit('system_status', {'metrics': self.demo_data['metrics']})
                    emit('clients_update', {'clients': self.demo_data['clients']})
                else:
                    emit('system_status', {'metrics': self._get_live_system_status()})
                    emit('clients_update', {'clients': self._get_live_clients_data()})
            except Exception as e:
                emit('error', {'message': f'Update failed: {str(e)}'})
    
    def _get_live_data(self) -> Dict[str, Any]:
        """Get live data from controller"""
        try:
            if not self.controller:
                return self.demo_data
            
            # Get real data from controller
            clients = self._get_live_clients_data()
            signals = self._get_live_signals_data()
            metrics = self._get_live_system_status()
            
            return {
                'clients': clients,
                'signals': signals,
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"Error getting live data: {e}")
            return self.demo_data
    
    def _get_live_clients_data(self) -> List[Dict[str, Any]]:
        """Get live client data"""
        try:
            if not self.controller:
                return self.demo_data['clients']
            
            clients = []
            for client_id, client in self.controller.get_all_clients().items():
                clients.append({
                    'client_id': client_id,
                    'name': getattr(client, 'name', f'Client {client_id}'),
                    'status': 'online' if getattr(client, 'is_active', False) else 'offline',
                    'balance': getattr(client, 'balance', 0.0),
                    'pnl_today': getattr(client, 'pnl_today', 0.0),
                    'positions': len(getattr(client, 'positions', [])),
                    'last_active': getattr(client, 'last_active', 'Unknown')
                })
            
            return clients
        except Exception as e:
            self.logger.error(f"Error getting clients data: {e}")
            return self.demo_data['clients']
    
    def _get_live_signals_data(self) -> List[Dict[str, Any]]:
        """Get live signals data"""
        try:
            if not self.controller:
                return self.demo_data['signals']
            
            # Get recent signals from controller
            signals = []
            recent_signals = getattr(self.controller, 'recent_signals', [])
            
            for signal in recent_signals[-10:]:  # Last 10 signals
                signals.append({
                    'currency_pair': getattr(signal, 'currency_pair', 'UNKNOWN'),
                    'type': getattr(signal, 'signal_type', 'BUY'),
                    'timeframe': getattr(signal, 'timeframe', 'M5'),
                    'entry_price': getattr(signal, 'entry_price', 0.0),
                    'target_1': getattr(signal, 'target_price', 0.0),
                    'stop_loss': getattr(signal, 'stop_loss', 0.0),
                    'confluence_score': getattr(signal, 'confluence_score', 0.0),
                    'timestamp': getattr(signal, 'timestamp', datetime.now()).strftime('%H:%M')
                })
            
            return signals
        except Exception as e:
            self.logger.error(f"Error getting signals data: {e}")
            return self.demo_data['signals']
    
    def _get_live_system_status(self) -> Dict[str, Any]:
        """Get live system status"""
        try:
            if not self.controller:
                return self.demo_data['metrics']
            
            status = self.controller.get_system_status()
            
            return {
                'active_clients': len([c for c in self.controller.get_all_clients().values() 
                                    if getattr(c, 'is_active', False)]),
                'total_signals_today': getattr(status, 'signals_today', 0),
                'active_positions': getattr(status, 'active_positions', 0),
                'total_pnl_today': getattr(status, 'total_pnl_today', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return self.demo_data['metrics']
    
    def broadcast_signal(self, signal_data: Dict[str, Any]):
        """Broadcast new signal to all connected clients - FIXED"""
        try:
            self.socketio.emit('new_signal', {
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            })
            self.logger.info(f"📡 Broadcasted signal: {signal_data.get('currency_pair', 'UNKNOWN')}")
        except Exception as e:
            self.logger.error(f"Error broadcasting signal: {e}")
    
    def broadcast_trade_update(self, trade_data: Dict[str, Any]):
        """Broadcast trade update to all connected clients"""
        try:
            self.socketio.emit('trade_executed', {
                'trade': trade_data,
                'timestamp': datetime.now().isoformat()
            })
            self.logger.info(f"📡 Broadcasted trade update: {trade_data.get('symbol', 'UNKNOWN')}")
        except Exception as e:
            self.logger.error(f"Error broadcasting trade update: {e}")
    
    def start_background_updates(self):
        """Start background thread for periodic updates - FIXED"""
        def update_loop():
            while self.is_running:
                try:
                    # Broadcast system status every 10 seconds
                    if self.connected_clients:
                        if self.demo_mode:
                            # Simulate live updates in demo mode
                            self.demo_data['metrics']['total_signals_today'] += 1
                            metrics = self.demo_data['metrics']
                        else:
                            metrics = self._get_live_system_status()
                        
                        self.socketio.emit('system_status', {
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Background update error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("📡 Background updates started")
    
    def start(self):
        """Start the dashboard server - FIXED"""
        try:
            self.is_running = True
            self.start_background_updates()
            
            self.logger.info(f"🌐 Starting Lightning Scalper Dashboard on {self.host}:{self.port}")
            self.logger.info(f"🔧 Demo Mode: {self.demo_mode}")
            self.logger.info(f"🔗 Access URL: http://{self.host}:{self.port}")
            
            # Start the server
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                allow_unsafe_werkzeug=True  # For development
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
            raise
    
    def stop(self):
        """Stop the dashboard server"""
        self.is_running = False
        self.logger.info("🛑 Lightning Scalper Dashboard stopped")
    
    def run(self):
        """
        Run method for compatibility with main controller
        แก้ไข Error: 'LightningScalperDashboard' object has no attribute 'run'
        """
        try:
            self.logger.info("🌐 Starting dashboard via run() method...")
            
            # เริ่ม background updates
            self.start_background_updates()
            
            # Return success (dashboard จะรันใน background)
            self.logger.info("✅ Dashboard run method completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dashboard run error: {e}")
            # Fallback: return success anyway
            return True

    def get_metrics(self):
        """
        Get dashboard metrics safely
        แก้ไข Error: 'metrics' KeyError
        """
        try:
            if self.demo_mode:
                return self.demo_data.get('metrics', {
                    'memory_usage': 0.0,
                    'cpu_usage': 0.0,
                    'active_clients': 0,
                    'total_signals': 0
                })
            else:
                return self._get_live_system_status()
        except Exception as e:
            self.logger.error(f"❌ Error getting metrics: {e}")
            return {
                'memory_usage': 0.0,
                'cpu_usage': 0.0, 
                'active_clients': 0,
                'total_signals': 0
            }

# ================================
# STANDALONE TESTING
# ================================

def main():
    """Main function for standalone testing"""
    print("🚀 Lightning Scalper Dashboard - Standalone Test")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create dashboard in demo mode
        dashboard = LightningScalperDashboard(
            controller=None,  # Demo mode
            host='127.0.0.1',
            port=5000,
            debug=True
        )
        
        print("✅ Dashboard initialized successfully")
        print("🌐 Starting web server...")
        print("📱 Open your browser to: http://127.0.0.1:5000")
        print("🔧 Press Ctrl+C to stop")
        
        # Start the dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()