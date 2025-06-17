#!/usr/bin/env python3
"""
🚀 Lightning Scalper - Real-time Web Dashboard
Production-Grade Web Interface for monitoring 80+ clients

Real-time dashboard for monitoring trading signals, client performance,
system status, and risk management across multiple clients.

Features:
- Real-time client monitoring
- Live signal tracking
- Performance analytics
- Risk management dashboard
- System health monitoring
- Interactive charts and graphs

Author: Phoenix Trading AI
Version: 1.0.0
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
    from flask import Flask, render_template, request, jsonify, redirect, url_for
    from flask_socketio import SocketIO, emit, join_room, leave_room
    import plotly.graph_objs as go
    import plotly.utils
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing web dependencies: {e}")
    print("   Install with: pip install flask flask-socketio plotly pandas")
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
    print(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

class LightningScalperDashboard:
    """
    🌐 Lightning Scalper Real-time Web Dashboard
    Production web interface for monitoring trading operations
    """
    
    def __init__(self, controller: Optional[LightningScalperController] = None, 
                 host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        
        self.controller = controller
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'lightning_scalper_dashboard_2024'
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, 
                               cors_allowed_origins="*",
                               async_mode='threading')
        
        # Dashboard state
        self.connected_clients = set()
        self.room_subscriptions = {}
        self.last_update = datetime.now()
        
        # Data caching for performance
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
        
        # Setup routes and socket events
        self._setup_template_filters()
        self._setup_routes()
        self._setup_socket_events()
        
        self.logger.info("🌐 Lightning Scalper Dashboard initialized")
    
    
    def _setup_template_filters(self):
        """Setup custom template filters to handle data safely"""
        
        @self.app.template_filter('safe_round')
        def safe_round(value, digits=2):
            try:
                if value is None:
                    return 0.0
                return round(float(value), digits)
            except (ValueError, TypeError):
                return 0.0
        
        @self.app.template_filter('safe_format_currency')
        def safe_format_currency(value):
            try:
                if value is None:
                    return '$0.00'
                return f'${float(value):,.2f}'
            except (ValueError, TypeError):
                return '$0.00'
        
        @self.app.template_filter('safe_format_percent')
        def safe_format_percent(value):
            try:
                if value is None:
                    return '0%'
                return f'{float(value):.1f}%'
            except (ValueError, TypeError):
                return '0%'
        
        @self.app.template_filter('safe_int')
        def safe_int(value):
            try:
                if value is None:
                    return 0
                return int(float(value))
            except (ValueError, TypeError):
                return 0

    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            try:
                # Get system overview data
                if self.controller:
                    system_status = self.controller.get_system_status()
                    client_count = len(self.controller.client_connections)
                else:
                    system_status = self._get_demo_system_status()
                    client_count = 5  # Demo data
                
                return render_template('dashboard.html',
                                     system_status=system_status,
                                     client_count=client_count,
                                     last_update=datetime.now())
            except Exception as e:
                self.logger.error(f"Error rendering index: {e}")
                return self._render_error_page("Dashboard Error", str(e))
        
        @self.app.route('/clients')
        def clients():
            """Client monitoring page"""
            try:
                clients_data = self._get_clients_data()
                return render_template('clients.html', 
                                     clients=clients_data,
                                     last_update=datetime.now())
            except Exception as e:
                self.logger.error(f"Error rendering clients page: {e}")
                return self._render_error_page("Clients Error", str(e))
        
        @self.app.route('/signals')
        def signals():
            """Signal monitoring page"""
            try:
                signals_data = self._get_signals_data()
                return render_template('signals.html',
                                     signals=signals_data,
                                     last_update=datetime.now())
            except Exception as e:
                self.logger.error(f"Error rendering signals page: {e}")
                return self._render_error_page("Signals Error", str(e))
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics and performance page"""
            try:
                analytics_data = self._get_analytics_data()
                return render_template('analytics.html',
                                     analytics=analytics_data,
                                     last_update=datetime.now())
            except Exception as e:
                self.logger.error(f"Error rendering analytics page: {e}")
                return self._render_error_page("Analytics Error", str(e))
        
        # API Routes
        @self.app.route('/api/system_status')
        def api_system_status():
            """API endpoint for system status"""
            try:
                if self.controller:
                    status = self.controller.get_system_status()
                else:
                    status = self._get_demo_system_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/clients')
        def api_clients():
            """API endpoint for clients data"""
            try:
                clients = self._get_clients_data()
                return jsonify(clients)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/client/<client_id>')
        def api_client_detail(client_id):
            """API endpoint for specific client data"""
            try:
                if self.controller:
                    client_data = self.controller.get_client_status(client_id)
                else:
                    client_data = self._get_demo_client_data(client_id)
                return jsonify(client_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/signals')
        def api_signals():
            """API endpoint for signals data"""
            try:
                signals = self._get_signals_data()
                return jsonify(signals)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance_chart')
        def api_performance_chart():
            """API endpoint for performance charts"""
            try:
                chart_data = self._generate_performance_chart()
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/emergency_stop', methods=['POST'])
        def api_emergency_stop():
            """API endpoint for emergency stop"""
            try:
                if self.controller:
                    reason = request.json.get('reason', 'Manual emergency stop from dashboard')
                    self.controller.emergency_stop(reason)
                    return jsonify({'success': True, 'message': 'Emergency stop activated'})
                else:
                    return jsonify({'success': False, 'message': 'No controller connected'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/resume_trading', methods=['POST'])
        def api_resume_trading():
            """API endpoint for resuming trading"""
            try:
                if self.controller:
                    self.controller.resume_operations()
                    return jsonify({'success': True, 'message': 'Trading resumed'})
                else:
                    return jsonify({'success': False, 'message': 'No controller connected'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _setup_socket_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            self.logger.info(f"🔗 Dashboard client connected: {client_id}")
            
            # Send initial data
            emit('system_status', self._get_cached_system_status())
            emit('client_count', {'count': len(self._get_clients_data())})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            
            # Clean up room subscriptions
            for room in list(self.room_subscriptions.keys()):
                if client_id in self.room_subscriptions[room]:
                    self.room_subscriptions[room].discard(client_id)
                    if not self.room_subscriptions[room]:
                        del self.room_subscriptions[room]
            
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
            
            self.logger.debug(f"📡 Client {client_id} subscribed to {update_type} updates")
        
        @self.socketio.on('unsubscribe_from_updates')
        def handle_unsubscribe(data):
            """Handle unsubscription from updates"""
            client_id = request.sid
            update_type = data.get('type', 'all')
            
            room = f"updates_{update_type}"
            leave_room(room)
            
            if room in self.room_subscriptions:
                self.room_subscriptions[room].discard(client_id)
        
        @self.socketio.on('request_client_details')
        def handle_client_details_request(data):
            """Handle request for specific client details"""
            client_id = data.get('client_id')
            if client_id:
                try:
                    if self.controller:
                        client_data = self.controller.get_client_status(client_id)
                    else:
                        client_data = self._get_demo_client_data(client_id)
                    
                    emit('client_details', {
                        'client_id': client_id,
                        'data': client_data
                    })
                except Exception as e:
                    emit('error', {'message': f"Failed to get client details: {e}"})
    
    def _get_clients_data(self) -> List[Dict[str, Any]]:
        """Get clients data for dashboard"""
        try:
            if self.controller:
                clients = []
                for client_id, connection in self.controller.client_connections.items():
                    try:
                        client_summary = self.controller.get_client_status(client_id)
                        clients.append(client_summary)
                    except Exception as e:
                        self.logger.error(f"Error getting client {client_id} data: {e}")
                return clients
            else:
                # Demo data
                return self._get_demo_clients_data()
        except Exception as e:
            self.logger.error(f"Error getting clients data: {e}")
            return []
    
    def _get_signals_data(self) -> List[Dict[str, Any]]:
        """Get signals data for dashboard"""
        try:
            if self.controller:
                # Get recent signals from controller
                recent_signals = []
                for signal_log in list(self.controller.signal_history)[-20:]:
                    signal = signal_log['signal']
                    recent_signals.append({
                        'id': signal.id,
                        'timestamp': signal_log['timestamp'].isoformat(),
                        'currency_pair': signal.currency_pair.value,
                        'type': signal.fvg_type.value,
                        'timeframe': signal.timeframe,
                        'confluence_score': signal.confluence_score,
                        'entry_price': signal.entry_price,
                        'target_1': signal.target_1,
                        'stop_loss': signal.stop_loss,
                        'session': signal.session,
                        'status': signal.status.value,
                        'priority': signal_log.get('priority', 1)
                    })
                return recent_signals
            else:
                return self._get_demo_signals_data()
        except Exception as e:
            self.logger.error(f"Error getting signals data: {e}")
            return []
    
    def _get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        try:
            if self.controller:
                system_status = self.controller.get_system_status()
                
                # Calculate additional analytics
                total_clients = len(self.controller.client_connections)
                active_positions = len(self.controller.trade_executor.active_positions)
                
                # Get performance data
                performance_chart = self._generate_performance_chart()
                
                return {
                    'system_metrics': system_status['metrics'],
                    'performance_chart': performance_chart,
                    'total_clients': total_clients,
                    'active_positions': active_positions,
                    'risk_metrics': self._calculate_risk_metrics()
                }
            else:
                return self._get_demo_analytics_data()
        except Exception as e:
            self.logger.error(f"Error getting analytics data: {e}")
            return {}
    
    def _generate_performance_chart(self) -> Dict[str, Any]:
        """Generate performance chart data"""
        try:
            # Generate sample performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                end=datetime.now(), freq='H')
            
            # Simulate P&L data
            np.random.seed(42)
            cumulative_pnl = np.cumsum(np.random.normal(2, 50, len(dates)))
            
            # Create Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_pnl,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                title='System Performance (7 Days)',
                xaxis_title='Time',
                yaxis_title='P&L ($)',
                template='plotly_dark',
                height=400
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            self.logger.error(f"Error generating performance chart: {e}")
            return {}
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        try:
            if self.controller:
                system_status = self.controller.get_system_status()
                
                # Calculate risk metrics
                total_pnl = system_status['metrics']['pnl_today']
                max_loss_limit = 5000  # From config
                
                risk_usage = abs(total_pnl) / max_loss_limit * 100 if total_pnl < 0 else 0
                
                return {
                    'daily_risk_usage': min(risk_usage, 100),
                    'max_daily_loss': max_loss_limit,
                    'current_pnl': total_pnl,
                    'risk_status': 'HIGH' if risk_usage > 80 else 'MEDIUM' if risk_usage > 50 else 'LOW'
                }
            else:
                return {
                    'daily_risk_usage': 25.5,
                    'max_daily_loss': 5000,
                    'current_pnl': -1275.50,
                    'risk_status': 'LOW'
                }
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _get_cached_system_status(self) -> Dict[str, Any]:
        """Get cached system status for performance"""
        now = datetime.now()
        
        if now - self.last_cache_update > self.cache_expiry:
            try:
                if self.controller:
                    self.cache['system_status'] = self.controller.get_system_status()
                else:
                    self.cache['system_status'] = self._get_demo_system_status()
                
                self.last_cache_update = now
            except Exception as e:
                self.logger.error(f"Error updating system status cache: {e}")
        
        return self.cache['system_status']
    
    def _get_demo_system_status(self) -> Dict[str, Any]:
        """Get demo system status for testing"""
        return {
            'status': 'RUNNING',
            'metrics': {
                'total_clients': 5,
                'active_clients': 4,
                'connected_clients': 3,
                'uptime_hours': 12.5,
                'signals_today': 47,
                'trades_today': 23,
                'pnl_today': 1234.56,
                'avg_signal_quality': 78.5
            },
            'safety': {
                'emergency_stop': False,
                'global_safety_enabled': True
            },
            'performance': {
                'cache_hit_rate': 0.85,
                'execution_queue_size': 2,
                'thread_pool_active': 8
            }
        }
    
    def _get_demo_clients_data(self) -> List[Dict[str, Any]]:
        """Get demo clients data - แก้ไข structure ให้ตรงกับ template"""
        return [
            {
                'client_id': 'CLIENT_001',
                'name': 'Demo Client 001',
                'email': 'demo1@example.com',
                'is_active': True,
                'is_demo': True,
                'balance': 10000.0,
                'pnl_today': 245.50,
                'active_positions': 3,
                'last_active': '2 min ago',
                'connection_status': 'online',
                'account_type': 'demo'
            },
            {
                'client_id': 'CLIENT_002',
                'name': 'Demo Client 002', 
                'email': 'demo2@example.com',
                'is_active': True,
                'is_demo': False,
                'balance': 25000.0,
                'pnl_today': 890.25,
                'active_positions': 5,
                'last_active': '5 min ago',
                'connection_status': 'online',
                'account_type': 'live'
            },
            {
                'client_id': 'CLIENT_003',
                'name': 'Demo Client 003',
                'email': 'demo3@example.com', 
                'is_active': False,
                'is_demo': True,
                'balance': 5000.0,
                'pnl_today': -125.75,
                'active_positions': 1,
                'last_active': '1 hour ago',
                'connection_status': 'offline',
                'account_type': 'demo'
            },
            {
                'client_id': 'CLIENT_004',
                'name': 'Demo Client 004',
                'email': 'demo4@example.com',
                'is_active': True,
                'is_demo': False, 
                'balance': 50000.0,
                'pnl_today': 1340.0,
                'active_positions': 2,
                'last_active': '1 min ago',
                'connection_status': 'online',
                'account_type': 'live'
            },
            {
                'client_id': 'CLIENT_005',
                'name': 'Demo Client 005',
                'email': 'demo5@example.com',
                'is_active': True,
                'is_demo': True,
                'balance': 15000.0,
                'pnl_today': 678.90,
                'active_positions': 4,
                'last_active': '3 min ago',
                'connection_status': 'online',
                'account_type': 'demo'
            }
        ]
    
    def _get_demo_client_data(self, client_id: str) -> Dict[str, Any]:
        """Get demo data for specific client"""
        demo_clients = self._get_demo_clients_data()
        for client in demo_clients:
            if client['client_id'] == client_id:
                return client
        
        # Return default demo data if client not found
        return {
            'client_id': client_id,
            'connection': {
                'is_active': False,
                'auto_trading': False,
                'mt5_connected': False
            },
            'trading_summary': {
                'account_info': {'balance': 0, 'equity': 0},
                'pnl': {'daily': 0, 'weekly': 0, 'monthly': 0},
                'active_positions': 0
            }
        }
    
    def _get_demo_signals_data(self) -> List[Dict[str, Any]]:
        """Get demo signals data"""
        return [
            {
                'id': 'FVG_DEMO_001',
                'timestamp': '14:23',
                'currency_pair': 'EURUSD',
                'type': 'BULLISH',
                'timeframe': 'M15',
                'confluence_score': 78.5,
                'entry_price': 1.10450,
                'target_1': 1.10680,
                'stop_loss': 1.10220,
                'risk_reward_ratio': 1.5,
                'session': 'London',
                'status': 'ACTIVE',
                'priority': 4
            },
            {
                'id': 'FVG_DEMO_002',
                'timestamp': '14:18',
                'currency_pair': 'GBPUSD',
                'type': 'BEARISH',
                'timeframe': 'M5',
                'confluence_score': 82.1,
                'entry_price': 1.25340,
                'target_1': 1.25090,
                'stop_loss': 1.25590,
                'risk_reward_ratio': 1.8,
                'session': 'London',
                'status': 'FILLED',
                'priority': 3
            }
        ]
    
    def _get_demo_analytics_data(self) -> Dict[str, Any]:
        """Get demo analytics data"""
        return {
            'system_metrics': self._get_demo_system_status()['metrics'],
            'performance_chart': self._generate_performance_chart(),
            'total_clients': 5,
            'active_positions': 7,
            'risk_metrics': self._calculate_risk_metrics()
        }
    
    def _render_error_page(self, title: str, message: str) -> str:
        """Render error page"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lightning Scalper - {title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }}
                .error {{ background: #ff4444; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>⚠️ {title}</h1>
            <div class="error">
                <h3>Error Details:</h3>
                <p>{message}</p>
                <a href="/" style="color: #88ff88;">← Back to Dashboard</a>
            </div>
        </body>
        </html>
        """
    
    def start_background_updates(self):
        """Start background thread for real-time updates"""
        def update_loop():
            while self.is_running:
                try:
                    # Update cached data
                    self._get_cached_system_status()
                    
                    # Emit updates to subscribed clients
                    if self.connected_clients:
                        # System status updates
                        self.socketio.emit('system_status', 
                                         self.cache['system_status'],
                                         room='updates_system')
                        
                        # Client updates
                        clients_data = self._get_clients_data()
                        self.socketio.emit('clients_update',
                                         {'clients': clients_data},
                                         room='updates_clients')
                        
                        # Signals updates
                        signals_data = self._get_signals_data()
                        self.socketio.emit('signals_update',
                                         {'signals': signals_data[-5:]},  # Last 5 signals
                                         room='updates_signals')
                    
                    # Sleep for next update
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in background update loop: {e}")
                    time.sleep(10)  # Longer sleep on error
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("📡 Background update thread started")
    
    def run(self, **kwargs):
        """Run the dashboard server"""
        try:
            self.is_running = True
            
            # Start background updates
            self.start_background_updates()
            
            # Subscribe to controller events if available
            if self.controller:
                self._subscribe_to_controller_events()
            
            self.logger.info(f"🌐 Starting Lightning Scalper Dashboard on {self.host}:{self.port}")
            self.logger.info(f"🔗 Dashboard URL: http://{self.host}:{self.port}")
            
            # Run the Flask-SocketIO server
            self.socketio.run(self.app, 
                            host=self.host, 
                            port=self.port,
                            debug=self.debug,
                            **kwargs)
                            
        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
            raise
    
    def _subscribe_to_controller_events(self):
        """Subscribe to controller events for real-time updates"""
        if not self.controller:
            return
        
        def on_signal_generated(event_type, data):
            """Handle signal generated event"""
            self.socketio.emit('new_signal', {
                'signal': {
                    'id': data['signal'].id,
                    'timestamp': datetime.now().isoformat(),
                    'currency_pair': data['signal'].currency_pair.value,
                    'type': data['signal'].fvg_type.value,
                    'confluence_score': data['signal'].confluence_score
                }
            }, room='updates_signals')
        
        def on_execution_complete(event_type, data):
            """Handle execution complete event"""
            self.socketio.emit('execution_update', {
                'client_id': data['client_id'],
                'signal_id': data['signal_id'],
                'result': data['result']
            }, room='updates_executions')
        
        def on_emergency_stop(event_type, data):
            """Handle emergency stop event"""
            self.socketio.emit('emergency_alert', {
                'message': f"🚨 EMERGENCY STOP: {data['reason']}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Subscribe to events
        self.controller.subscribe_to_events('signal_generated', on_signal_generated)
        self.controller.subscribe_to_events('execution_complete', on_execution_complete)
        self.controller.subscribe_to_events('emergency_stop', on_emergency_stop)
        
        self.logger.info("📡 Subscribed to controller events")
    
    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("🛑 Dashboard stopped")

def create_dashboard_app(controller: Optional[LightningScalperController] = None,
                        host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> LightningScalperDashboard:
    """Factory function to create dashboard app"""
    return LightningScalperDashboard(controller=controller, host=host, port=port, debug=debug)

# Demo/Testing functionality
def run_standalone_demo():
    """Run dashboard in standalone demo mode"""
    print("🌐 Lightning Scalper Dashboard - Demo Mode")
    print("=" * 50)
    print("🔗 Dashboard will be available at: http://localhost:5000")
    print("📊 Demo data will be displayed (no real trading)")
    print("⌨️ Press Ctrl+C to stop")
    print("=" * 50)
    
    # Create dashboard without controller (demo mode)
    dashboard = create_dashboard_app(controller=None, debug=True)
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    run_standalone_demo()