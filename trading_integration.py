# 🎯 ENHANCED TRADING SYSTEM INTEGRATION
# Integration code สำหรับเชื่อมต่อ Intelligent Trailing Stop กับระบบหลัก

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List
import json
from flask import Flask, request, jsonify
# Import the trailing stop system
from trailing_stop_system import TrailingStopManager, IntelligentTrailingStop

class EnhancedTradingSystemWithTrailing:
    """
    🎯 Enhanced Trading System with Intelligent Trailing Stop
    """
    
    def __init__(self, main_dashboard):
        """Initialize Enhanced Trading System"""
        self.main_dashboard = main_dashboard
        
        # Initialize Trailing Stop Manager
        self.trailing_manager = TrailingStopManager()
        
        # Trailing Stop Settings (สามารถปรับได้)
        self.trailing_enabled = True
        self.trailing_update_interval = 10  # seconds
        self.trailing_profiles = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE']
        self.current_trailing_profile = 'MODERATE'
        
        # Threading for background trailing stop updates
        self.trailing_thread = None
        self.trailing_running = False
        
        # Statistics
        self.trailing_stats = {
            'total_sl_updates': 0,
            'breakeven_protections': 0,
            'profit_secured': 0.0,
            'positions_saved': 0
        }
        
        # Initialize trailing system
        self.setup_trailing_system()
        
    def setup_trailing_system(self):
        """🎯 Setup and configure trailing stop system"""
        try:
            # Set initial profile
            self.trailing_manager.trailing_system.set_trailing_profile(self.current_trailing_profile)
            
            # Configure settings
            self.trailing_manager.enabled = self.trailing_enabled
            self.trailing_manager.update_interval = self.trailing_update_interval
            
            print(f"✅ Trailing Stop System Initialized")
            print(f"   Profile: {self.current_trailing_profile}")
            print(f"   Update Interval: {self.trailing_update_interval}s")
            print(f"   Enabled: {self.trailing_enabled}")
            
            # Start background thread if enabled
            if self.trailing_enabled:
                self.start_trailing_thread()
                
        except Exception as e:
            print(f"❌ Error setting up trailing system: {str(e)}")
    
    def start_trailing_thread(self):
        """🚀 Start background thread for trailing stop updates"""
        if self.trailing_thread and self.trailing_thread.is_alive():
            return
        
        self.trailing_running = True
        self.trailing_thread = threading.Thread(target=self._trailing_loop, daemon=True)
        self.trailing_thread.start()
        print("🚀 Trailing Stop Background Thread Started")
    
    def stop_trailing_thread(self):
        """⏹️ Stop background trailing thread"""
        self.trailing_running = False
        if self.trailing_thread:
            self.trailing_thread.join(timeout=5)
        print("⏹️ Trailing Stop Background Thread Stopped")
    
    def _trailing_loop(self):
        """🔄 Background loop for trailing stop updates"""
        while self.trailing_running:
            try:
                if self.trailing_enabled and hasattr(self.main_dashboard, 'open_positions'):
                    # Get current positions
                    positions = self._get_open_positions()
                    
                    if positions:
                        # Get market data
                        market_data = self._get_market_data_for_trailing()
                        
                        if market_data:
                            # Process trailing stops
                            self._process_trailing_stops(positions, market_data)
                
                time.sleep(self.trailing_update_interval)
                
            except Exception as e:
                print(f"❌ Error in trailing loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _get_open_positions(self) -> List[Dict]:
        """📊 Get current open positions from MT5"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_dict = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'price_open': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'volume': pos.volume,
                    'profit': pos.profit,
                    'time': pos.time
                }
                position_list.append(position_dict)
            
            return position_list
            
        except Exception as e:
            print(f"❌ Error getting positions: {str(e)}")
            return []
    
    def _get_market_data_for_trailing(self) -> Dict:
        """📊 Get market data for trailing calculations"""
        try:
            if not hasattr(self.main_dashboard, 'live_data'):
                return {}
            
            market_data = {}
            
            # Get data from main dashboard
            for symbol, data in self.main_dashboard.live_data.items():
                if data and 'bid' in data and 'ask' in data:
                    market_data[symbol] = {
                        'bid': data['bid'],
                        'ask': data['ask'],
                        'atr': data.get('atr', 0.001),  # Use ATR from technical analysis
                        'spread': data.get('spread', 0.0001)
                    }
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error getting market data: {str(e)}")
            return {}
    
    def _process_trailing_stops(self, positions: List[Dict], market_data: Dict):
        """🎯 Process trailing stops for all positions"""
        try:
            # Process through trailing manager
            results = self.trailing_manager.process_trailing_stops(positions, market_data)
            
            if results.get('success') and results['results']['updates_needed'] > 0:
                # Execute the SL updates
                updates_executed = 0
                
                for update in results['results']['updates']:
                    success = self._execute_sl_modification(
                        update['ticket'], 
                        update['new_sl'], 
                        update['symbol']
                    )
                    
                    if success:
                        updates_executed += 1
                        self.trailing_stats['total_sl_updates'] += 1
                        
                        if update['reason'] == 'BREAKEVEN_PROTECTION':
                            self.trailing_stats['breakeven_protections'] += 1
                
                # Log results
                if updates_executed > 0:
                    print(f"🎯 Trailing Stop Updates: {updates_executed}/{results['results']['updates_needed']}")
                    print(f"   Breakeven Moves: {results['results']['breakeven_moves']}")
                    print(f"   Trail Moves: {results['results']['trailing_moves']}")
            
        except Exception as e:
            print(f"❌ Error processing trailing stops: {str(e)}")
    
    def _execute_sl_modification(self, ticket: int, new_sl: float, symbol: str) -> bool:
        """🎯 Execute Stop Loss modification in MT5"""
        try:
            # Get current position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": pos.tp,  # Keep existing TP
                "magic": pos.magic,
                "comment": f"Trailing SL - {self.current_trailing_profile}",
            }
            
            # Execute modification
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ SL Updated: {symbol} #{ticket} → SL: {new_sl:.5f}")
                return True
            else:
                print(f"❌ SL Update Failed: {symbol} #{ticket} - {result.comment}")
                return False
                
        except Exception as e:
            print(f"❌ Error executing SL modification: {str(e)}")
            return False
    
    def set_trailing_profile(self, profile: str) -> bool:
        """🎯 Change trailing stop profile"""
        if profile in self.trailing_profiles:
            self.current_trailing_profile = profile
            self.trailing_manager.trailing_system.set_trailing_profile(profile)
            print(f"🎯 Trailing Profile Changed: {profile}")
            return True
        return False
    
    def toggle_trailing_stops(self, enabled: bool) -> bool:
        """🎯 Enable/Disable trailing stops"""
        self.trailing_enabled = enabled
        self.trailing_manager.enabled = enabled
        
        if enabled:
            self.start_trailing_thread()
            print("🟢 Trailing Stops: ENABLED")
        else:
            self.stop_trailing_thread()
            print("🔴 Trailing Stops: DISABLED")
        
        return True
    
    def get_trailing_dashboard_data(self) -> Dict:
        """📊 Get data for trailing stop dashboard"""
        try:
            # Get statistics from trailing manager
            stats = self.trailing_manager.get_statistics()
            
            # Get current positions with trailing info
            positions = self._get_open_positions()
            market_data = self._get_market_data_for_trailing()
            
            position_details = []
            
            for pos in positions:
                symbol = pos['symbol']
                ticket = pos['ticket']
                
                if symbol in market_data:
                    # Calculate trailing info for this position
                    trail_info = self.trailing_manager.trailing_system.calculate_dynamic_trailing_stop(
                        pos, market_data[symbol]
                    )
                    
                    position_details.append({
                        'ticket': ticket,
                        'symbol': symbol,
                        'type': 'BUY' if pos['type'] == 0 else 'SELL',
                        'entry_price': pos['price_open'],
                        'current_sl': pos['sl'],
                        'profit': pos['profit'],
                        'trail_info': trail_info
                    })
            
            return {
                'enabled': self.trailing_enabled,
                'profile': self.current_trailing_profile,
                'statistics': {
                    **stats,
                    'system_stats': self.trailing_stats
                },
                'positions': position_details,
                'thread_status': 'RUNNING' if self.trailing_running else 'STOPPED'
            }
            
        except Exception as e:
            print(f"❌ Error getting dashboard data: {str(e)}")
            return {'error': str(e)}

# 🎯 API Integration for Web Dashboard
def add_trailing_stop_routes(app, enhanced_system):
    """🌐 Add trailing stop routes to Flask app"""
    
    @app.route('/api/trailing-stops/status')
    def get_trailing_status():
        """Get trailing stop status"""
        try:
            data = enhanced_system.get_trailing_dashboard_data()
            return {'success': True, 'data': data}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @app.route('/api/trailing-stops/toggle', methods=['POST'])
    def toggle_trailing():
        """Toggle trailing stops on/off"""
        try:
            data = request.get_json()
            enabled = data.get('enabled', False)
            success = enhanced_system.toggle_trailing_stops(enabled)
            return {'success': success}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @app.route('/api/trailing-stops/profile', methods=['POST'])
    def set_trailing_profile():
        """Set trailing stop profile"""
        try:
            data = request.get_json()
            profile = data.get('profile', 'MODERATE')
            success = enhanced_system.set_trailing_profile(profile)
            return {'success': success, 'profile': profile}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @app.route('/api/trailing-stops/manual-update', methods=['POST'])
    def manual_trailing_update():
        """Manually trigger trailing stop update"""
        try:
            positions = enhanced_system._get_open_positions()
            market_data = enhanced_system._get_market_data_for_trailing()
            
            if positions and market_data:
                enhanced_system._process_trailing_stops(positions, market_data)
                return {'success': True, 'message': 'Trailing stops updated'}
            else:
                return {'success': False, 'message': 'No positions or market data'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

# 🎯 Integration Instructions
def integrate_with_existing_system():
    """
    📖 Integration Instructions
    
    1. ในไฟล์ mt5_forex_connector.py เพิ่ม:
       from trading_integration import EnhancedTradingSystemWithTrailing, add_trailing_stop_routes
    
    2. ใน __init__ ของ EnhancedSmartAutoTradingDashboard:
       self.enhanced_trading = EnhancedTradingSystemWithTrailing(self)
    
    3. ใน setup_routes():
       add_trailing_stop_routes(self.app, self.enhanced_trading)
    
    4. ใน graceful_shutdown():
       if hasattr(self, 'enhanced_trading'):
           self.enhanced_trading.stop_trailing_thread()
    
    5. เพิ่ม JavaScript ใน Dashboard สำหรับ Trailing Stop Controls
    """
    print("📖 Integration Instructions printed above")

if __name__ == "__main__":
    integrate_with_existing_system()