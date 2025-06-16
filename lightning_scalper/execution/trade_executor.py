import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
import uuid
from collections import deque
import json

# Import our FVG Engine
from core.lightning_scalper_engine import FVGSignal, FVGType, EnhancedFVGDetector, CurrencyPair

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Individual order structure"""
    id: str
    client_id: str
    symbol: str
    direction: TradeDirection
    order_type: OrderType
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    swap: float = 0.0
    pnl: float = 0.0
    signal_id: Optional[str] = None
    notes: str = ""

@dataclass
class Position:
    """Position management structure"""
    id: str
    client_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    remaining_quantity: float
    status: PositionStatus
    open_time: datetime
    close_time: Optional[datetime] = None
    
    # Targets and stops
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    stop_loss: float = None
    
    # Partial fills tracking
    partial_closes: List[Dict] = field(default_factory=list)
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_swap: float = 0.0
    
    # Metadata
    signal_id: Optional[str] = None
    original_signal: Optional[FVGSignal] = None
    execution_notes: str = ""

@dataclass
class ClientAccount:
    """Client account management"""
    client_id: str
    account_number: str
    broker: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    
    # Risk limits
    max_daily_loss: float
    max_weekly_loss: float
    max_monthly_loss: float
    max_positions: int
    max_lot_size: float
    
    # Current status
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    active_positions: int = 0
    is_trading_enabled: bool = True
    
    # Trading preferences
    risk_per_trade: float = 0.02  # 2% per trade
    preferred_pairs: List[str] = field(default_factory=list)
    trading_sessions: List[str] = field(default_factory=list)

class TradeExecutor:
    """
    Lightning Scalper Trade Execution Engine
    Handles automatic trade execution, position management, and risk control
    """
    
    def __init__(self):
        self.clients: Dict[str, ClientAccount] = {}
        self.active_positions: Dict[str, Position] = {}
        self.order_history: deque = deque(maxlen=10000)
        self.execution_queue: deque = deque()
        
        # Risk management
        self.global_risk_enabled = True
        self.max_simultaneous_trades = 50
        self.emergency_stop = False
        
        # Execution parameters
        self.slippage_tolerance = 2.0  # pips
        self.execution_timeout = 30  # seconds
        self.max_retries = 3
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'avg_slippage': 0.0,
            'total_volume': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        self.execution_thread = None
        self.is_running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradeExecutor')
        
        # Initialize broker connections (placeholder)
        self.broker_connections = {}
    
    def register_client(self, client_account: ClientAccount) -> bool:
        """Register a new client account"""
        try:
            with self.lock:
                self.clients[client_account.client_id] = client_account
                self.logger.info(f"Client {client_account.client_id} registered successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to register client {client_account.client_id}: {e}")
            return False
    
    def start_execution_engine(self):
        """Start the trade execution engine"""
        if not self.is_running:
            self.is_running = True
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()
            self.logger.info("Trade Execution Engine started")
    
    def stop_execution_engine(self):
        """Stop the trade execution engine"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join()
        self.logger.info("Trade Execution Engine stopped")
    
    def _execution_loop(self):
        """Main execution loop"""
        while self.is_running:
            try:
                if self.execution_queue and not self.emergency_stop:
                    execution_request = self.execution_queue.popleft()
                    self._process_execution_request(execution_request)
                else:
                    time.sleep(0.1)  # Brief pause if no orders
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(1)
    
    def execute_fvg_signal(self, signal: FVGSignal, client_id: str, custom_lot_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute trade based on FVG signal
        """
        if client_id not in self.clients:
            return {"success": False, "error": "Client not found"}
        
        client = self.clients[client_id]
        
        # Pre-execution risk checks
        risk_check = self._pre_execution_risk_check(client, signal)
        if not risk_check['allowed']:
            return {"success": False, "error": risk_check['reason']}
        
        # Calculate position size
        lot_size = self._calculate_optimal_lot_size(client, signal, custom_lot_size)
        if lot_size <= 0:
            return {"success": False, "error": "Invalid lot size calculated"}
        
        # Create execution request
        execution_request = {
            'type': 'FVG_SIGNAL',
            'client_id': client_id,
            'signal': signal,
            'lot_size': lot_size,
            'timestamp': datetime.now(),
            'request_id': str(uuid.uuid4())
        }
        
        # Add to execution queue
        with self.lock:
            self.execution_queue.append(execution_request)
        
        self.logger.info(f"FVG signal queued for execution - Client: {client_id}, Signal: {signal.id}")
        
        return {
            "success": True, 
            "request_id": execution_request['request_id'],
            "estimated_lot_size": lot_size,
            "estimated_risk": lot_size * abs(signal.entry_price - signal.stop_loss) * 100000  # Rough pip value
        }
    
    def _process_execution_request(self, request: Dict[str, Any]):
        """Process individual execution request"""
        try:
            request_type = request['type']
            
            if request_type == 'FVG_SIGNAL':
                self._execute_fvg_signal_internal(request)
            elif request_type == 'CLOSE_POSITION':
                self._close_position_internal(request)
            elif request_type == 'MODIFY_POSITION':
                self._modify_position_internal(request)
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing execution request: {e}")
    
    def _execute_fvg_signal_internal(self, request: Dict[str, Any]):
        """Internal FVG signal execution"""
        client_id = request['client_id']
        signal = request['signal']
        lot_size = request['lot_size']
        
        client = self.clients[client_id]
        
        # Determine trade direction
        direction = TradeDirection.BUY if signal.fvg_type == FVGType.BULLISH else TradeDirection.SELL
        
        # Create primary order
        order = Order(
            id=str(uuid.uuid4()),
            client_id=client_id,
            symbol=signal.currency_pair.value,
            direction=direction,
            order_type=OrderType.LIMIT,  # Use limit order for better fills
            quantity=lot_size,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            signal_id=signal.id
        )
        
        # Execute the order
        execution_result = self._execute_order(order, client)
        
        if execution_result['success']:
            # Create position
            position = Position(
                id=str(uuid.uuid4()),
                client_id=client_id,
                symbol=signal.currency_pair.value,
                direction=direction,
                entry_price=execution_result['fill_price'],
                quantity=lot_size,
                remaining_quantity=lot_size,
                status=PositionStatus.OPEN,
                open_time=datetime.now(),
                target_1=signal.target_1,
                target_2=signal.target_2,
                target_3=signal.target_3,
                stop_loss=signal.stop_loss,
                signal_id=signal.id,
                original_signal=signal
            )
            
            # Store position
            with self.lock:
                self.active_positions[position.id] = position
                client.active_positions += 1
            
            # Set up automatic target orders
            self._setup_target_orders(position, client)
            
            self.logger.info(f"Position opened successfully - {position.id}")
            
        else:
            self.logger.error(f"Failed to execute order: {execution_result['error']}")
    
    def _execute_order(self, order: Order, client: ClientAccount) -> Dict[str, Any]:
        """
        Execute individual order (placeholder for broker integration)
        """
        try:
            # Simulate order execution (replace with actual broker API calls)
            execution_start = time.time()
            
            # Simulate market conditions
            current_price = order.price
            slippage = np.random.uniform(-self.slippage_tolerance, self.slippage_tolerance) * 0.0001
            fill_price = current_price + slippage
            
            # Simulate execution delay
            time.sleep(np.random.uniform(0.1, 0.5))
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.fill_time = datetime.now()
            order.commission = self._calculate_commission(order, client)
            
            # Update statistics
            execution_time = time.time() - execution_start
            with self.lock:
                self.execution_stats['total_executions'] += 1
                self.execution_stats['successful_executions'] += 1
                self.execution_stats['avg_execution_time'] = (
                    (self.execution_stats['avg_execution_time'] * (self.execution_stats['total_executions'] - 1) + execution_time) /
                    self.execution_stats['total_executions']
                )
                self.execution_stats['avg_slippage'] = (
                    (self.execution_stats['avg_slippage'] * (self.execution_stats['total_executions'] - 1) + abs(slippage)) /
                    self.execution_stats['total_executions']
                )
                self.execution_stats['total_volume'] += order.quantity
            
            # Store in history
            self.order_history.append(order)
            
            return {
                'success': True,
                'order_id': order.id,
                'fill_price': fill_price,
                'slippage': slippage,
                'execution_time': execution_time
            }
            
        except Exception as e:
            # Update failure statistics
            with self.lock:
                self.execution_stats['total_executions'] += 1
                self.execution_stats['failed_executions'] += 1
            
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _setup_target_orders(self, position: Position, client: ClientAccount):
        """Set up automatic target orders for position"""
        try:
            targets = [
                (position.target_1, 0.5),  # Close 50% at target 1
                (position.target_2, 0.3),  # Close 30% at target 2
                (position.target_3, 0.2)   # Close 20% at target 3
            ]
            
            for target_price, close_ratio in targets:
                if target_price:
                    close_quantity = position.quantity * close_ratio
                    
                    # Create target order
                    target_order = Order(
                        id=str(uuid.uuid4()),
                        client_id=client.client_id,
                        symbol=position.symbol,
                        direction=TradeDirection.SELL if position.direction == TradeDirection.BUY else TradeDirection.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=close_quantity,
                        price=target_price,
                        notes=f"Target order for position {position.id}"
                    )
                    
                    # Queue target order for monitoring
                    self._add_pending_order(target_order, position.id)
            
            # Set up stop loss order
            if position.stop_loss:
                stop_order = Order(
                    id=str(uuid.uuid4()),
                    client_id=client.client_id,
                    symbol=position.symbol,
                    direction=TradeDirection.SELL if position.direction == TradeDirection.BUY else TradeDirection.BUY,
                    order_type=OrderType.STOP,
                    quantity=position.remaining_quantity,
                    price=position.stop_loss,
                    notes=f"Stop loss for position {position.id}"
                )
                
                self._add_pending_order(stop_order, position.id)
                
        except Exception as e:
            self.logger.error(f"Failed to setup target orders for position {position.id}: {e}")
    
    def _add_pending_order(self, order: Order, position_id: str):
        """Add order to pending orders for monitoring"""
        # This would integrate with broker's order management system
        # For now, we'll simulate with a simple structure
        pass
    
    def _pre_execution_risk_check(self, client: ClientAccount, signal: FVGSignal) -> Dict[str, Any]:
        """Comprehensive pre-execution risk check"""
        
        # Check if trading is enabled
        if not client.is_trading_enabled:
            return {'allowed': False, 'reason': 'Trading disabled for client'}
        
        # Check global emergency stop
        if self.emergency_stop:
            return {'allowed': False, 'reason': 'Emergency stop activated'}
        
        # Check daily loss limit
        if client.daily_pnl <= -client.max_daily_loss:
            return {'allowed': False, 'reason': 'Daily loss limit reached'}
        
        # Check weekly loss limit
        if client.weekly_pnl <= -client.max_weekly_loss:
            return {'allowed': False, 'reason': 'Weekly loss limit reached'}
        
        # Check monthly loss limit
        if client.monthly_pnl <= -client.max_monthly_loss:
            return {'allowed': False, 'reason': 'Monthly loss limit reached'}
        
        # Check maximum positions
        if client.active_positions >= client.max_positions:
            return {'allowed': False, 'reason': 'Maximum positions limit reached'}
        
        # Check margin requirements
        estimated_margin = self._estimate_margin_requirement(client, signal)
        if estimated_margin > client.free_margin:
            return {'allowed': False, 'reason': 'Insufficient margin'}
        
        # Check signal quality threshold
        if signal.confluence_score < 60:  # Minimum quality threshold
            return {'allowed': False, 'reason': 'Signal quality below threshold'}
        
        return {'allowed': True, 'reason': 'All checks passed'}
    
    def _calculate_optimal_lot_size(self, client: ClientAccount, signal: FVGSignal, custom_lot_size: Optional[float] = None) -> float:
        """Calculate optimal lot size based on risk management"""
        
        if custom_lot_size:
            return min(custom_lot_size, client.max_lot_size)
        
        # Risk per trade calculation
        account_risk = client.balance * client.risk_per_trade
        
        # Calculate pip value
        pip_value = 0.0001 if 'JPY' not in signal.currency_pair.value else 0.01
        
        # Calculate distance to stop loss in pips
        if signal.fvg_type == FVGType.BULLISH:
            stop_distance = abs(signal.entry_price - signal.stop_loss)
        else:
            stop_distance = abs(signal.stop_loss - signal.entry_price)
        
        stop_distance_pips = stop_distance / pip_value
        
        if stop_distance_pips <= 0:
            return 0.0
        
        # Calculate lot size
        pip_cost = 10.0  # $10 per pip for 1 lot (standard account)
        max_loss = stop_distance_pips * pip_cost
        
        if max_loss <= 0:
            return 0.0
        
        lot_size = account_risk / max_loss
        
        # Apply position size factor from signal quality
        lot_size *= signal.position_size_factor
        
        # Ensure within limits
        lot_size = max(0.01, min(lot_size, client.max_lot_size))
        
        return round(lot_size, 2)
    
    def _estimate_margin_requirement(self, client: ClientAccount, signal: FVGSignal) -> float:
        """Estimate margin requirement for trade"""
        # Simplified margin calculation (replace with broker-specific logic)
        leverage = 100  # Assume 1:100 leverage
        base_currency_value = 100000  # Standard lot value
        estimated_lot_size = 0.1  # Conservative estimate
        
        return (base_currency_value * estimated_lot_size) / leverage
    
    def _calculate_commission(self, order: Order, client: ClientAccount) -> float:
        """Calculate commission for order"""
        # Example commission structure (replace with broker-specific)
        commission_per_lot = 7.0  # $7 per lot
        return order.quantity * commission_per_lot
    
    def close_position(self, position_id: str, close_ratio: float = 1.0, reason: str = "Manual close") -> Dict[str, Any]:
        """Close position (full or partial)"""
        
        if position_id not in self.active_positions:
            return {"success": False, "error": "Position not found"}
        
        position = self.active_positions[position_id]
        close_quantity = position.remaining_quantity * close_ratio
        
        # Create close order
        close_order = Order(
            id=str(uuid.uuid4()),
            client_id=position.client_id,
            symbol=position.symbol,
            direction=TradeDirection.SELL if position.direction == TradeDirection.BUY else TradeDirection.BUY,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            price=0,  # Market order
            notes=f"Close position {position_id} - {reason}"
        )
        
        # Queue for execution
        close_request = {
            'type': 'CLOSE_POSITION',
            'position_id': position_id,
            'close_order': close_order,
            'close_ratio': close_ratio,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        with self.lock:
            self.execution_queue.append(close_request)
        
        return {"success": True, "message": f"Position close queued - {close_quantity} lots"}
    
    def _close_position_internal(self, request: Dict[str, Any]):
        """Internal position closing logic"""
        position_id = request['position_id']
        close_order = request['close_order']
        close_ratio = request['close_ratio']
        
        if position_id not in self.active_positions:
            return
        
        position = self.active_positions[position_id]
        client = self.clients[position.client_id]
        
        # Execute close order
        execution_result = self._execute_order(close_order, client)
        
        if execution_result['success']:
            # Update position
            close_quantity = position.remaining_quantity * close_ratio
            position.remaining_quantity -= close_quantity
            
            # Calculate P&L for this close
            if position.direction == TradeDirection.BUY:
                pnl = (execution_result['fill_price'] - position.entry_price) * close_quantity * 100000
            else:
                pnl = (position.entry_price - execution_result['fill_price']) * close_quantity * 100000
            
            pnl -= close_order.commission  # Subtract commission
            
            position.realized_pnl += pnl
            
            # Record partial close
            position.partial_closes.append({
                'quantity': close_quantity,
                'price': execution_result['fill_price'],
                'pnl': pnl,
                'timestamp': datetime.now(),
                'reason': request['reason']
            })
            
            # Update client P&L
            client.daily_pnl += pnl
            client.weekly_pnl += pnl
            client.monthly_pnl += pnl
            
            # Check if position fully closed
            if position.remaining_quantity <= 0.001:  # Account for floating point precision
                position.status = PositionStatus.CLOSED
                position.close_time = datetime.now()
                client.active_positions -= 1
                
                # Remove from active positions
                with self.lock:
                    del self.active_positions[position_id]
                
                self.logger.info(f"Position {position_id} fully closed - P&L: {position.realized_pnl:.2f}")
            else:
                position.status = PositionStatus.PARTIAL_CLOSE
                self.logger.info(f"Position {position_id} partially closed - Remaining: {position.remaining_quantity}")
    
    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive client summary"""
        if client_id not in self.clients:
            return {"error": "Client not found"}
        
        client = self.clients[client_id]
        
        # Get client's active positions
        client_positions = [pos for pos in self.active_positions.values() if pos.client_id == client_id]
        
        # Calculate unrealized P&L
        total_unrealized = sum(pos.unrealized_pnl for pos in client_positions)
        
        return {
            "client_id": client_id,
            "account_info": {
                "balance": client.balance,
                "equity": client.balance + total_unrealized,
                "margin": client.margin,
                "free_margin": client.free_margin,
                "margin_level": client.margin_level
            },
            "pnl": {
                "daily": client.daily_pnl,
                "weekly": client.weekly_pnl,
                "monthly": client.monthly_pnl,
                "unrealized": total_unrealized
            },
            "risk_status": {
                "daily_limit_used": abs(client.daily_pnl) / client.max_daily_loss * 100,
                "weekly_limit_used": abs(client.weekly_pnl) / client.max_weekly_loss * 100,
                "monthly_limit_used": abs(client.monthly_pnl) / client.max_monthly_loss * 100,
                "positions_used": client.active_positions / client.max_positions * 100
            },
            "active_positions": len(client_positions),
            "trading_enabled": client.is_trading_enabled
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution engine statistics"""
        with self.lock:
            stats = self.execution_stats.copy()
        
        # Calculate success rate
        if stats['total_executions'] > 0:
            stats['success_rate'] = (stats['successful_executions'] / stats['total_executions']) * 100
        else:
            stats['success_rate'] = 0
        
        stats['active_positions'] = len(self.active_positions)
        stats['registered_clients'] = len(self.clients)
        stats['queue_size'] = len(self.execution_queue)
        
        return stats
    
    def emergency_stop_all(self, reason: str = "Emergency stop"):
        """Emergency stop all trading activities"""
        self.emergency_stop = True
        
        # Close all active positions
        for position_id in list(self.active_positions.keys()):
            self.close_position(position_id, 1.0, f"Emergency stop: {reason}")
        
        # Disable trading for all clients
        for client in self.clients.values():
            client.is_trading_enabled = False
        
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def resume_trading(self):
        """Resume trading after emergency stop"""
        self.emergency_stop = False
        
        # Re-enable trading for clients (manual review required)
        for client in self.clients.values():
            client.is_trading_enabled = True
        
        self.logger.info("Trading resumed after emergency stop")

# Usage Example and Integration Test
if __name__ == "__main__":
    print("🚀 Lightning Scalper Trade Execution Engine")
    print("=" * 60)
    
    # Initialize trade executor
    executor = TradeExecutor()
    
    # Create sample client
    sample_client = ClientAccount(
        client_id="CLIENT_001",
        account_number="12345678",
        broker="MetaTrader5",
        currency="USD",
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        margin_level=0.0,
        max_daily_loss=200.0,    # $200 daily loss limit
        max_weekly_loss=500.0,   # $500 weekly loss limit
        max_monthly_loss=1500.0, # $1500 monthly loss limit
        max_positions=5,         # Max 5 positions
        max_lot_size=1.0,        # Max 1 lot per trade
        preferred_pairs=["EURUSD", "GBPUSD", "USDJPY"],
        trading_sessions=["London", "NewYork"]
    )
    
    # Register client
    if executor.register_client(sample_client):
        print(f"✅ Client {sample_client.client_id} registered successfully")
    
    # Start execution engine
    executor.start_execution_engine()
    print("✅ Execution engine started")
    
    # Create sample FVG signal for testing
    from core.lightning_scalper_engine import FVGSignal, FVGType, CurrencyPair, MarketCondition, FVGStatus
    
    sample_signal = FVGSignal(
        id="FVG_TEST_001",
        timestamp=datetime.now(),
        timeframe="M5",
        currency_pair=CurrencyPair.EURUSD,
        fvg_type=FVGType.BULLISH,
        high=1.1050,
        low=1.1030,
        gap_size=0.0020,
        gap_percentage=0.18,
        confluence_score=85.0,
        market_condition=MarketCondition.TRENDING_UP,
        session="London",
        status=FVGStatus.ACTIVE,
        entry_price=1.1045,
        target_1=1.1065,
        target_2=1.1075,
        target_3=1.1085,
        stop_loss=1.1025,
        risk_reward_ratio=1.5,
        position_size_factor=1.2,
        urgency_level=4,
        atr_ratio=1.1,
        volume_strength=25.0,
        momentum_score=18.0,
        structure_score=42.0,
        tags=["session_London", "condition_TRENDING_UP", "high_volume"]
    )
    
    # Execute FVG signal
    execution_result = executor.execute_fvg_signal(sample_signal, "CLIENT_001")
    
    if execution_result['success']:
        print(f"🎯 FVG Signal execution queued successfully")
        print(f"   Request ID: {execution_result['request_id']}")
        print(f"   Estimated Lot Size: {execution_result['estimated_lot_size']}")
        print(f"   Estimated Risk: ${execution_result['estimated_risk']:.2f}")
    else:
        print(f"❌ FVG Signal execution failed: {execution_result['error']}")
    
    # Wait for execution
    time.sleep(2)
    
    # Get client summary
    client_summary = executor.get_client_summary("CLIENT_001")
    print(f"\n📊 Client Summary:")
    print(f"   Balance: ${client_summary['account_info']['balance']:.2f}")
    print(f"   Equity: ${client_summary['account_info']['equity']:.2f}")
    print(f"   Daily P&L: ${client_summary['pnl']['daily']:.2f}")
    print(f"   Active Positions: {client_summary['active_positions']}")
    print(f"   Trading Enabled: {client_summary['trading_enabled']}")
    
    # Get execution statistics
    exec_stats = executor.get_execution_statistics()
    print(f"\n📈 Execution Statistics:")
    print(f"   Total Executions: {exec_stats['total_executions']}")
    print(f"   Success Rate: {exec_stats['success_rate']:.1f}%")
    print(f"   Average Execution Time: {exec_stats['avg_execution_time']:.3f}s")
    print(f"   Active Positions: {exec_stats['active_positions']}")
    print(f"   Registered Clients: {exec_stats['registered_clients']}")
    
    print("\n✅ Trade Execution Engine Ready for Production!")
    print("🎯 Next Step: Broker Integration (MT4/MT5 Bridge)")
    
    # Stop execution engine
    executor.stop_execution_engine()