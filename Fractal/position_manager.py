import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from trading_core import TradingConfig

@dataclass
class Position:
    """Position data structure"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    open_price: float
    open_time: datetime
    tp: float = 0.0
    sl: float = 0.0
    profit: float = 0.0
    recovery_level: int = 0
    is_recovery: bool = False
    original_signal: str = ""

@dataclass
class RecoveryGroup:
    """Recovery group tracking"""
    group_id: str
    direction: int  # 0=BUY, 1=SELL
    positions: List[Position]
    total_volume: float = 0.0
    total_profit: float = 0.0
    avg_price: float = 0.0
    target_profit: float = 0.0
    current_level: int = 0
    max_level: int = 3
    last_signal_time: datetime = None

class PositionManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions: Dict[int, Position] = {}
        self.recovery_groups: Dict[str, RecoveryGroup] = {}
        
        # Exit speed multipliers by timeframe
        self.exit_speed_multipliers = {
            "M1": {"FAST": 0.5, "MEDIUM": 1.0, "SLOW": 2.0},
            "M5": {"FAST": 0.6, "MEDIUM": 1.2, "SLOW": 2.5},
            "M15": {"FAST": 0.8, "MEDIUM": 1.5, "SLOW": 3.0},
            "M30": {"FAST": 1.0, "MEDIUM": 2.0, "SLOW": 4.0},
            "H1": {"FAST": 1.2, "MEDIUM": 2.5, "SLOW": 5.0},
            "H4": {"FAST": 2.0, "MEDIUM": 4.0, "SLOW": 8.0},
            "D1": {"FAST": 3.0, "MEDIUM": 6.0, "SLOW": 12.0}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def update_positions(self):
        """Update positions from MT5"""
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            positions = []
        
        # Update existing positions
        current_tickets = {pos.ticket for pos in positions}
        
        # Remove closed positions
        closed_tickets = set(self.positions.keys()) - current_tickets
        for ticket in closed_tickets:
            self._on_position_closed(ticket)
        
        # Update or add positions
        for pos in positions:
            if pos.ticket in self.positions:
                self._update_position(pos)
            else:
                self._add_new_position(pos)
        
        # Update recovery groups
        self._update_recovery_groups()
    
    def _add_new_position(self, mt5_pos):
        """Add new position to tracking"""
        position = Position(
            ticket=mt5_pos.ticket,
            symbol=mt5_pos.symbol,
            type=mt5_pos.type,
            volume=mt5_pos.volume,
            open_price=mt5_pos.price_open,
            open_time=datetime.fromtimestamp(mt5_pos.time),
            tp=mt5_pos.tp,
            sl=mt5_pos.sl,
            profit=mt5_pos.profit
        )
        
        self.positions[mt5_pos.ticket] = position
        self.logger.info(f"Added position: {mt5_pos.ticket}")
    
    def _update_position(self, mt5_pos):
        """Update existing position"""
        if mt5_pos.ticket in self.positions:
            pos = self.positions[mt5_pos.ticket]
            pos.profit = mt5_pos.profit
            pos.tp = mt5_pos.tp
            pos.sl = mt5_pos.sl
    
    def _on_position_closed(self, ticket: int):
        """Handle position closure"""
        if ticket in self.positions:
            pos = self.positions[ticket]
            self.logger.info(f"Position closed: {ticket}, Profit: {pos.profit}")
            
            # Update recovery group if needed
            self._remove_from_recovery_group(pos)
            del self.positions[ticket]
    
    def calculate_take_profit(self, direction: int, recovery_positions: List[Position] = None) -> float:
        """Calculate take profit based on recovery positions"""
        if not recovery_positions or len(recovery_positions) == 0:
            # First position - use standard TP
            return self._get_base_tp_points()
        
        # Calculate TP for recovery group
        total_volume = sum(pos.volume for pos in recovery_positions)
        target_profit_usd = self._calculate_target_profit_usd(recovery_positions[0].volume)
        
        # Get current price
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return self._get_base_tp_points()
        
        current_price = symbol_info.bid if direction == 1 else symbol_info.ask
        
        # Calculate required TP in points to achieve target profit
        point_value = self._get_point_value()
        required_points = target_profit_usd / (total_volume * point_value)
        
        # Apply spread buffer
        spread_buffer = self._calculate_spread_buffer()
        tp_points = required_points + spread_buffer
        
        # Apply exit speed modifier
        speed_modifier = self._get_exit_speed_modifier()
        tp_points *= speed_modifier
        
        return max(tp_points, 10)  # Minimum 10 points TP
    
    def _get_base_tp_points(self) -> float:
        """Get base TP points for first position"""
        speed_modifier = self._get_exit_speed_modifier()
        return self.config.tp_first * speed_modifier
    
    def _calculate_target_profit_usd(self, base_volume: float) -> float:
        """Calculate target profit in USD for recovery"""
        point_value = self._get_point_value()
        return self.config.tp_first * base_volume * point_value
    
    def _get_point_value(self) -> float:
        """Get point value for XAUUSD (typically $0.01 per 0.01 lot)"""
        return 0.01  # XAUUSD standard point value per 0.01 lot
    
    def _get_exit_speed_modifier(self) -> float:
        """Get exit speed modifier based on timeframe and setting"""
        tf = self.config.primary_tf
        speed_names = ["FAST", "MEDIUM", "SLOW"]
        speed_name = speed_names[self.config.exit_speed]
        
        multipliers = self.exit_speed_multipliers.get(tf, self.exit_speed_multipliers["M15"])
        return multipliers.get(speed_name, 1.0)
    
    def _calculate_spread_buffer(self) -> int:
        """Calculate spread buffer for TP"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 5
        
        current_spread = symbol_info.spread
        
        if self.config.spread_mode == 0:  # AUTO
            return int(current_spread * 1.5) + 2
        elif self.config.spread_mode == 1:  # FIXED
            return self.config.spread_buffer
        elif self.config.spread_mode == 2:  # SMART
            return int(current_spread * 1.2) + 1
        else:  # NONE
            return 0
    
    def check_recovery_needed(self, position: Position) -> bool:
        """Check if position needs recovery"""
        if position.profit >= 0:
            return False
        
        # Calculate loss in points
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return False
        
        current_price = symbol_info.bid if position.type == 0 else symbol_info.ask
        price_diff = abs(current_price - position.open_price)
        loss_points = price_diff / symbol_info.point
        
        return loss_points >= self.config.recovery_price
    
    def get_recovery_lot_size(self, original_volume: float, recovery_level: int) -> float:
        """Calculate lot size for recovery position"""
        multiplier = self.config.martingale ** recovery_level
        return round(original_volume * multiplier, 2)
    
    def can_add_recovery(self, group_id: str) -> bool:
        """Check if can add recovery position to group"""
        if group_id not in self.recovery_groups:
            return True
        
        group = self.recovery_groups[group_id]
        return group.current_level < self.config.max_recovery
    
    def create_recovery_group(self, original_position: Position) -> str:
        """Create recovery group for position"""
        group_id = f"{original_position.type}_{original_position.open_time.strftime('%Y%m%d_%H%M%S')}"
        
        group = RecoveryGroup(
            group_id=group_id,
            direction=original_position.type,
            positions=[original_position],
            total_volume=original_position.volume,
            target_profit=self._calculate_target_profit_usd(original_position.volume),
            max_level=self.config.max_recovery
        )
        
        self.recovery_groups[group_id] = group
        original_position.recovery_level = 0
        
        self.logger.info(f"Created recovery group: {group_id}")
        return group_id
    
    def add_to_recovery_group(self, group_id: str, position: Position):
        """Add position to recovery group"""
        if group_id in self.recovery_groups:
            group = self.recovery_groups[group_id]
            group.positions.append(position)
            group.total_volume += position.volume
            group.current_level += 1
            
            position.recovery_level = group.current_level
            position.is_recovery = True
            
            self.logger.info(f"Added to recovery group {group_id}: Level {group.current_level}")
    
    def _update_recovery_groups(self):
        """Update recovery group statistics"""
        for group_id, group in self.recovery_groups.items():
            group.total_profit = sum(pos.profit for pos in group.positions)
            
            # Calculate weighted average price
            total_volume = sum(pos.volume for pos in group.positions)
            if total_volume > 0:
                weighted_sum = sum(pos.open_price * pos.volume for pos in group.positions)
                group.avg_price = weighted_sum / total_volume
            
            group.total_volume = total_volume
    
    def _remove_from_recovery_group(self, position: Position):
        """Remove position from recovery group when closed"""
        for group_id, group in list(self.recovery_groups.items()):
            if position in group.positions:
                group.positions.remove(position)
                
                # If group is empty, remove it
                if not group.positions:
                    del self.recovery_groups[group_id]
                    self.logger.info(f"Recovery group completed: {group_id}")
                break
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        total_profit = sum(pos.profit for pos in self.positions.values())
        buy_positions = [pos for pos in self.positions.values() if pos.type == 0]
        sell_positions = [pos for pos in self.positions.values() if pos.type == 1]
        
        return {
            "total_positions": len(self.positions),
            "buy_positions": len(buy_positions),
            "sell_positions": len(sell_positions),
            "total_profit": total_profit,
            "recovery_groups": len(self.recovery_groups),
            "buy_profit": sum(pos.profit for pos in buy_positions),
            "sell_profit": sum(pos.profit for pos in sell_positions)
        }
    
    def get_recovery_status(self) -> Dict:
        """Get recovery groups status"""
        status = {}
        for group_id, group in self.recovery_groups.items():
            status[group_id] = {
                "direction": "BUY" if group.direction == 0 else "SELL",
                "level": group.current_level,
                "max_level": group.max_level,
                "total_volume": group.total_volume,
                "total_profit": group.total_profit,
                "target_profit": group.target_profit,
                "avg_price": group.avg_price,
                "positions_count": len(group.positions)
            }
        return status
    
    def emergency_close_all(self) -> List[int]:
        """Emergency close all positions"""
        closed_tickets = []
        
        for ticket in list(self.positions.keys()):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "type": mt5.ORDER_TYPE_SELL if self.positions[ticket].type == 0 else mt5.ORDER_TYPE_BUY,
                "volume": self.positions[ticket].volume,
                "symbol": self.config.symbol,
                "deviation": 20,
                "magic": 234000,
                "comment": "Emergency close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed_tickets.append(ticket)
                self.logger.info(f"Emergency closed position: {ticket}")
        
        return closed_tickets