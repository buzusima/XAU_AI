# 🎯 ENHANCED INTELLIGENT TRAILING STOP SYSTEM
# Professional Forex Auto Trading Dashboard - Trailing Stop Module
# ระบบ Trailing Stop อัจฉริยะที่ปรับตัวตามสภาวะตลาดและความผันผวน

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class IntelligentTrailingStop:
    """
    🎯 Intelligent Trailing Stop System
    
    Features:
    - Dynamic trailing distance based on ATR
    - Breakeven protection
    - Trend-following trailing logic
    - Support/Resistance aware trailing
    - Volume-based trailing adjustment
    - Multiple trailing modes (CONSERVATIVE, MODERATE, AGGRESSIVE)
    """
    
    def __init__(self):
        """Initialize Intelligent Trailing Stop System"""
        self.logger = logging.getLogger(__name__)
        
        # 📊 Trailing Stop Profiles
        self.trailing_profiles = {
            'CONSERVATIVE': {
                'initial_distance_atr': 2.5,      # Initial distance = 2.5 * ATR
                'min_trail_distance_atr': 1.5,    # Minimum trailing = 1.5 * ATR
                'breakeven_trigger_atr': 1.0,     # Move to BE when profit = 1.0 * ATR
                'trail_step_atr': 0.3,            # Trail every 0.3 * ATR price move
                'acceleration_factor': 0.02,       # Slow acceleration
                'max_acceleration': 0.2           # Max acceleration
            },
            'MODERATE': {
                'initial_distance_atr': 2.0,      # Initial distance = 2.0 * ATR
                'min_trail_distance_atr': 1.2,    # Minimum trailing = 1.2 * ATR
                'breakeven_trigger_atr': 0.8,     # Move to BE when profit = 0.8 * ATR
                'trail_step_atr': 0.25,           # Trail every 0.25 * ATR price move
                'acceleration_factor': 0.03,       # Moderate acceleration
                'max_acceleration': 0.25          # Max acceleration
            },
            'AGGRESSIVE': {
                'initial_distance_atr': 1.5,      # Initial distance = 1.5 * ATR
                'min_trail_distance_atr': 0.8,    # Minimum trailing = 0.8 * ATR
                'breakeven_trigger_atr': 0.5,     # Move to BE when profit = 0.5 * ATR
                'trail_step_atr': 0.2,            # Trail every 0.2 * ATR price move
                'acceleration_factor': 0.04,       # Fast acceleration
                'max_acceleration': 0.3           # Max acceleration
            }
        }
        
        # 🎯 Current Settings
        self.current_profile = 'MODERATE'
        self.enabled = True
        self.breakeven_protection = True
        self.support_resistance_aware = True
        
        # 📈 Position Tracking
        self.position_states = {}  # Track each position's trailing state
        
    def calculate_dynamic_trailing_stop(self, position: Dict, market_data: Dict, 
                                      support_resistance: Optional[Dict] = None) -> Dict:
        """
        🎯 Calculate Dynamic Trailing Stop
        
        Args:
            position: Position information
            market_data: Current market data with technical indicators
            support_resistance: Optional S/R levels
            
        Returns:
            Dict with trailing stop information
        """
        try:
            symbol = position['symbol']
            ticket = position['ticket']
            position_type = position['type']  # 0=BUY, 1=SELL
            entry_price = position['price_open']
            current_price = market_data.get('bid' if position_type == 0 else 'ask', entry_price)
            current_sl = position.get('sl', 0)
            
            # Get ATR for dynamic calculations
            atr = market_data.get('atr', 0.001)
            if atr <= 0:
                atr = abs(current_price - entry_price) * 0.01  # Fallback ATR
                
            # Get trailing profile settings
            profile = self.trailing_profiles[self.current_profile]
            
            # Initialize position state if not exists
            if ticket not in self.position_states:
                self.position_states[ticket] = {
                    'highest_price': current_price if position_type == 0 else entry_price,
                    'lowest_price': current_price if position_type == 1 else entry_price,
                    'last_trail_price': entry_price,
                    'breakeven_activated': False,
                    'trail_count': 0,
                    'acceleration_factor': profile['acceleration_factor']
                }
            
            state = self.position_states[ticket]
            
            # 📊 Calculate current profit in ATR terms
            if position_type == 0:  # BUY position
                profit_pips = (current_price - entry_price)
                state['highest_price'] = max(state['highest_price'], current_price)
                reference_price = state['highest_price']
            else:  # SELL position
                profit_pips = (entry_price - current_price)
                state['lowest_price'] = min(state['lowest_price'], current_price)
                reference_price = state['lowest_price']
            
            profit_atr = profit_pips / atr
            
            # 🎯 Breakeven Protection Logic
            new_sl = current_sl
            should_update = False
            trail_reason = "NO_UPDATE"
            
            # Check if we should move to breakeven
            if (not state['breakeven_activated'] and 
                profit_atr >= profile['breakeven_trigger_atr'] and 
                self.breakeven_protection):
                
                new_sl = entry_price + (0.0001 if position_type == 0 else -0.0001)  # Small buffer
                state['breakeven_activated'] = True
                should_update = True
                trail_reason = "BREAKEVEN_PROTECTION"
                
            # 🚀 Dynamic Trailing Logic
            elif state['breakeven_activated'] or profit_atr >= profile['breakeven_trigger_atr']:
                
                # Calculate dynamic trailing distance
                base_distance = profile['min_trail_distance_atr'] * atr
                
                # Acceleration based on profit
                acceleration = min(
                    state['acceleration_factor'] * state['trail_count'],
                    profile['max_acceleration']
                )
                
                # Adjust distance based on market volatility
                volatility_multiplier = min(2.0, max(0.5, atr / 0.001))  # Adjust for pair volatility
                dynamic_distance = base_distance * (1 - acceleration) * volatility_multiplier
                
                # 📈 Support/Resistance Awareness
                if self.support_resistance_aware and support_resistance:
                    dynamic_distance = self._adjust_for_support_resistance(
                        dynamic_distance, current_price, position_type, 
                        support_resistance, atr
                    )
                
                # Calculate new trailing stop
                if position_type == 0:  # BUY position
                    calculated_sl = reference_price - dynamic_distance
                    
                    # Only trail up, never down
                    if calculated_sl > current_sl:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_UP"
                        state['trail_count'] += 1
                        
                else:  # SELL position
                    calculated_sl = reference_price + dynamic_distance
                    
                    # Only trail down, never up
                    if current_sl == 0 or calculated_sl < current_sl:
                        new_sl = calculated_sl
                        should_update = True
                        trail_reason = "TRAILING_DOWN"
                        state['trail_count'] += 1
            
            # 📊 Calculate trail statistics
            trail_info = {
                'should_update': should_update,
                'new_sl': new_sl,
                'current_sl': current_sl,
                'trail_reason': trail_reason,
                'profit_atr': round(profit_atr, 2),
                'trail_distance_atr': round((abs(new_sl - current_price) / atr), 2) if new_sl != current_sl else 0,
                'breakeven_activated': state['breakeven_activated'],
                'trail_count': state['trail_count'],
                'acceleration_factor': round(state['acceleration_factor'], 4),
                'profile': self.current_profile
            }
            
            return trail_info
            
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop for {position.get('symbol', 'UNKNOWN')}: {str(e)}")
            return {
                'should_update': False,
                'error': str(e),
                'trail_reason': 'ERROR'
            }
    
    def _adjust_for_support_resistance(self, base_distance: float, current_price: float, 
                                     position_type: int, sr_levels: Dict, atr: float) -> float:
        """
        🎯 Adjust trailing distance based on Support/Resistance levels
        
        Args:
            base_distance: Base trailing distance
            current_price: Current market price
            position_type: 0=BUY, 1=SELL
            sr_levels: Support/Resistance levels
            atr: Average True Range value
            
        Returns:
            Adjusted trailing distance
        """
        try:
            if position_type == 0:  # BUY position - look for support below
                supports = sr_levels.get('support_levels', [])
                nearest_support = None
                
                for support in supports:
                    if support < current_price:
                        if nearest_support is None or support > nearest_support:
                            nearest_support = support
                
                if nearest_support:
                    distance_to_support = current_price - nearest_support
                    # If support is very close, increase trailing distance slightly
                    if distance_to_support < base_distance * 1.5:
                        return min(base_distance * 1.3, distance_to_support * 0.8)
                        
            else:  # SELL position - look for resistance above
                resistances = sr_levels.get('resistance_levels', [])
                nearest_resistance = None
                
                for resistance in resistances:
                    if resistance > current_price:
                        if nearest_resistance is None or resistance < nearest_resistance:
                            nearest_resistance = resistance
                
                if nearest_resistance:
                    distance_to_resistance = nearest_resistance - current_price
                    # If resistance is very close, increase trailing distance slightly
                    if distance_to_resistance < base_distance * 1.5:
                        return min(base_distance * 1.3, distance_to_resistance * 0.8)
            
            return base_distance
            
        except Exception:
            return base_distance
    
    def update_trailing_stops_batch(self, positions: List[Dict], market_data_batch: Dict,
                                  support_resistance_batch: Optional[Dict] = None) -> Dict:
        """
        🎯 Update trailing stops for multiple positions in batch
        
        Args:
            positions: List of open positions
            market_data_batch: Market data for all symbols
            support_resistance_batch: Optional S/R data for all symbols
            
        Returns:
            Dict with update results
        """
        results = {
            'total_positions': len(positions),
            'updates_needed': 0,
            'breakeven_moves': 0,
            'trailing_moves': 0,
            'errors': 0,
            'updates': []
        }
        
        for position in positions:
            try:
                symbol = position['symbol']
                ticket = position['ticket']
                
                # Get market data for this symbol
                market_data = market_data_batch.get(symbol, {})
                if not market_data:
                    continue
                
                # Get S/R data if available
                sr_data = support_resistance_batch.get(symbol) if support_resistance_batch else None
                
                # Calculate trailing stop
                trail_info = self.calculate_dynamic_trailing_stop(position, market_data, sr_data)
                
                if trail_info.get('should_update', False):
                    results['updates_needed'] += 1
                    
                    # Count update types
                    if trail_info['trail_reason'] == 'BREAKEVEN_PROTECTION':
                        results['breakeven_moves'] += 1
                    elif 'TRAILING' in trail_info['trail_reason']:
                        results['trailing_moves'] += 1
                    
                    # Add to updates list
                    results['updates'].append({
                        'ticket': ticket,
                        'symbol': symbol,
                        'new_sl': trail_info['new_sl'],
                        'reason': trail_info['trail_reason'],
                        'profit_atr': trail_info['profit_atr'],
                        'trail_distance_atr': trail_info['trail_distance_atr']
                    })
                    
            except Exception as e:
                results['errors'] += 1
                self.logger.error(f"Error processing trailing stop for position {position.get('ticket', 'unknown')}: {str(e)}")
        
        return results
    
    def set_trailing_profile(self, profile_name: str) -> bool:
        """
        🎯 Set trailing stop profile
        
        Args:
            profile_name: CONSERVATIVE, MODERATE, or AGGRESSIVE
            
        Returns:
            Success status
        """
        if profile_name in self.trailing_profiles:
            self.current_profile = profile_name
            self.logger.info(f"Trailing stop profile changed to: {profile_name}")
            return True
        return False
    
    def get_trailing_statistics(self) -> Dict:
        """
        📊 Get trailing stop statistics
        
        Returns:
            Dictionary with trailing stop statistics
        """
        active_positions = len(self.position_states)
        breakeven_count = sum(1 for state in self.position_states.values() if state['breakeven_activated'])
        total_trails = sum(state['trail_count'] for state in self.position_states.values())
        
        return {
            'active_positions': active_positions,
            'breakeven_protected': breakeven_count,
            'total_trail_moves': total_trails,
            'current_profile': self.current_profile,
            'enabled': self.enabled,
            'breakeven_protection': self.breakeven_protection,
            'support_resistance_aware': self.support_resistance_aware,
            'average_trails_per_position': round(total_trails / max(1, active_positions), 2)
        }
    
    def cleanup_closed_positions(self, open_tickets: List[int]) -> int:
        """
        🧹 Clean up tracking data for closed positions
        
        Args:
            open_tickets: List of currently open position tickets
            
        Returns:
            Number of cleaned up positions
        """
        cleaned = 0
        tickets_to_remove = []
        
        for ticket in self.position_states:
            if ticket not in open_tickets:
                tickets_to_remove.append(ticket)
        
        for ticket in tickets_to_remove:
            del self.position_states[ticket]
            cleaned += 1
        
        return cleaned

# 🎯 Integration class for main trading system
class TrailingStopManager:
    """
    🎯 Trailing Stop Manager for integration with main trading system
    """
    
    def __init__(self):
        """Initialize Trailing Stop Manager"""
        self.trailing_system = IntelligentTrailingStop()
        self.enabled = True
        self.update_interval = 5  # seconds
        self.last_update = datetime.now()
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'breakeven_saves': 0,
            'trail_moves': 0,
            'positions_protected': 0
        }
        
    def should_update_trailing_stops(self) -> bool:
        """Check if it's time to update trailing stops"""
        return (datetime.now() - self.last_update).total_seconds() >= self.update_interval
    
    def process_trailing_stops(self, positions: List[Dict], market_data: Dict) -> Dict:
        """
        🎯 Main method to process trailing stops for all positions
        
        Args:
            positions: List of open positions
            market_data: Current market data
            
        Returns:
            Processing results
        """
        if not self.enabled or not positions:
            return {'success': False, 'reason': 'disabled_or_no_positions'}
        
        # Update trailing stops
        results = self.trailing_system.update_trailing_stops_batch(positions, market_data)
        
        # Update statistics
        self.stats['total_updates'] += results['updates_needed']
        self.stats['breakeven_saves'] += results['breakeven_moves']
        self.stats['trail_moves'] += results['trailing_moves']
        
        # Clean up closed positions
        open_tickets = [pos['ticket'] for pos in positions]
        cleaned = self.trailing_system.cleanup_closed_positions(open_tickets)
        
        self.last_update = datetime.now()
        
        return {
            'success': True,
            'results': results,
            'cleaned_positions': cleaned,
            'statistics': self.get_statistics()
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive trailing stop statistics"""
        system_stats = self.trailing_system.get_trailing_statistics()
        
        return {
            **system_stats,
            'manager_stats': self.stats,
            'enabled': self.enabled,
            'update_interval': self.update_interval,
            'last_update': self.last_update.isoformat()
        }

# 🎯 Usage Example and Testing
def example_usage():
    """
    📖 Example usage of the Intelligent Trailing Stop System
    """
    print("🎯 INTELLIGENT TRAILING STOP SYSTEM - EXAMPLE")
    print("=" * 50)
    
    # Initialize system
    trailing_manager = TrailingStopManager()
    
    # Example position data
    example_positions = [
        {
            'ticket': 12345,
            'symbol': 'EURUSD.c',
            'type': 0,  # BUY
            'price_open': 1.0850,
            'sl': 1.0820,
            'tp': 1.0920
        },
        {
            'ticket': 12346,
            'symbol': 'GBPUSD.c',
            'type': 1,  # SELL
            'price_open': 1.2650,
            'sl': 1.2680,
            'tp': 1.2580
        }
    ]
    
    # Example market data
    example_market_data = {
        'EURUSD.c': {
            'bid': 1.0885,
            'ask': 1.0887,
            'atr': 0.0025
        },
        'GBPUSD.c': {
            'bid': 1.2615,
            'ask': 1.2617,
            'atr': 0.0030
        }
    }
    
    # Process trailing stops
    results = trailing_manager.process_trailing_stops(example_positions, example_market_data)
    
    print(f"✅ Processing Results:")
    print(f"   Updates Needed: {results['results']['updates_needed']}")
    print(f"   Breakeven Moves: {results['results']['breakeven_moves']}")
    print(f"   Trailing Moves: {results['results']['trailing_moves']}")
    
    # Show statistics
    stats = trailing_manager.get_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Active Positions: {stats['active_positions']}")
    print(f"   Breakeven Protected: {stats['breakeven_protected']}")
    print(f"   Current Profile: {stats['current_profile']}")
    
    print("\n🎯 Trailing Stop System Ready!")

if __name__ == "__main__":
    example_usage()