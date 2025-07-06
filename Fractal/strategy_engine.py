import MetaTrader5 as mt5
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import json

from trading_core import XAUUSDTradingCore, TradingConfig
from position_manager import PositionManager, Position, RecoveryGroup
from order_executor import OrderExecutor, OrderType, OrderResult
from risk_manager import RiskManager, RiskLevel

class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"
    NO_SIGNAL = "NO_SIGNAL"

@dataclass
class EngineStatus:
    """Engine status information"""
    state: EngineState
    uptime: float = 0.0
    last_update: datetime = None
    total_trades: int = 0
    successful_trades: int = 0
    current_positions: int = 0
    total_pnl: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    restrictions: List[str] = None
    last_signal: str = ""
    last_trade: datetime = None
    errors: List[str] = None

class StrategyEngine:
    def __init__(self, config: TradingConfig = None):
        # Initialize components
        self.config = config or TradingConfig()
        self.trading_core = XAUUSDTradingCore(self.config)
        self.position_manager = PositionManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.position_manager)
        self.risk_manager = RiskManager(self.config, self.position_manager)
        
        # Engine state
        self.state = EngineState.STOPPED
        self.start_time = None
        self.last_update = None
        self.running = False
        
        # Threading
        self.main_thread = None
        self.update_interval = 1.0  # seconds
        self.signal_check_interval = 5.0  # seconds
        self.last_signal_check = None
        
        # Event handlers
        self.event_handlers = {
            'on_trade_opened': [],
            'on_trade_closed': [],
            'on_signal_detected': [],
            'on_risk_alert': [],
            'on_error': [],
            'on_state_changed': []
        }
        
        # Performance tracking
        self.engine_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_closed': 0,
            'recovery_triggered': 0,
            'errors_occurred': 0,
            'uptime_seconds': 0
        }
        
        # Signal history
        self.signal_history = []
        self.trade_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """Start the trading engine"""
        try:
            self.logger.info("Starting XAUUSD Trading Engine...")
            self.state = EngineState.STARTING
            self._notify_state_change()
            
            # Initialize MT5 connection
            if not self.trading_core.initialize_mt5():
                self.state = EngineState.ERROR
                self._notify_state_change()
                return False
            
            # Validate configuration
            if not self._validate_config():
                self.state = EngineState.ERROR
                self._notify_state_change()
                return False
            
            # Start main trading loop
            self.running = True
            self.start_time = datetime.now()
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            self.state = EngineState.RUNNING
            self._notify_state_change()
            
            self.logger.info("Trading Engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start engine: {e}")
            self.state = EngineState.ERROR
            self._notify_state_change()
            return False
    
    def stop(self):
        """Stop the trading engine"""
        self.logger.info("Stopping Trading Engine...")
        self.running = False
        self.state = EngineState.STOPPED
        self._notify_state_change()
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        self.logger.info("Trading Engine stopped")
    
    def pause(self):
        """Pause trading (keep monitoring but don't trade)"""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            self._notify_state_change()
            self.logger.info("Trading Engine paused")
    
    def resume(self):
        """Resume trading from pause"""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            self._notify_state_change()
            self.logger.info("Trading Engine resumed")
    
    def emergency_stop(self):
        """Emergency stop - close all positions and stop trading"""
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        self.state = EngineState.EMERGENCY_STOP
        self._notify_state_change()
        
        # Close all positions
        closed_positions = self.order_executor.emergency_close_all()
        self.logger.info(f"Emergency closed {len(closed_positions)} positions")
        
        # Stop engine
        self.stop()
        
        # Trigger emergency risk shutdown
        self.risk_manager.emergency_risk_shutdown()
    
    def update_config(self, new_config: Dict):
        """Update configuration during runtime"""
        try:
            # Update trading core config
            self.trading_core.update_config(new_config)
            
            # Update risk limits if provided
            risk_limits = {k: v for k, v in new_config.items() 
                          if k.startswith(('daily_', 'weekly_', 'monthly_', 'max_'))}
            if risk_limits:
                self.risk_manager.update_risk_limits(risk_limits)
            
            self.logger.info(f"Configuration updated: {len(new_config)} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            self._notify_error(f"Config update error: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        self.logger.info("Main trading loop started")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Update all components
                self._update_components()
                
                # Check trading conditions
                if self.state == EngineState.RUNNING:
                    self._process_trading_logic()
                
                # Update engine statistics
                self._update_engine_stats()
                
                # Sleep for remaining interval time
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                self._notify_error(f"Main loop error: {e}")
                self.engine_stats['errors_occurred'] += 1
                
                # If too many errors, stop engine
                if self.engine_stats['errors_occurred'] > 10:
                    self.logger.critical("Too many errors, stopping engine")
                    self.emergency_stop()
                    break
    
    def _update_components(self):
        """Update all engine components"""
        # Update positions
        self.position_manager.update_positions()
        
        # Update risk metrics
        self.risk_manager.update_metrics()
        
        # Update last update time
        self.last_update = datetime.now()
    
    def _process_trading_logic(self):
        """Main trading logic processing"""
        # Check if signal analysis is due
        now = datetime.now()
        if (self.last_signal_check is None or 
            (now - self.last_signal_check).total_seconds() >= self.signal_check_interval):
            
            self._analyze_and_process_signals()
            self.last_signal_check = now
        
        # Check recovery opportunities
        self._check_recovery_opportunities()
        
        # Monitor existing positions
        self._monitor_positions()
    
    def _analyze_and_process_signals(self):
        """Analyze market signals and process trading decisions"""
        try:
            # Check if trading is allowed
            trading_allowed, restrictions = self.risk_manager.check_trading_allowed()
            if not trading_allowed:
                self.logger.info(f"Trading restricted: {restrictions}")
                return
            
            # Check trading conditions
            conditions = self.trading_core.check_trading_conditions()
            if not conditions.get("can_trade", False):
                self.logger.info(f"Trading conditions not met: {conditions.get('reason')}")
                return
            
            # Analyze entry signals
            signals = self.trading_core.analyze_entry_signals()
            
            if not signals:
                return
            
            # Process signals
            for signal_type, signal_data in signals.items():
                if signal_type in ["BUY", "SELL"]:
                    self._process_entry_signal(signal_type, signal_data)
                    
        except Exception as e:
            self.logger.error(f"Signal analysis error: {e}")
            self._notify_error(f"Signal analysis error: {e}")
    
    def _process_entry_signal(self, signal_type: str, signal_data: Dict):
        """Process entry signal"""
        try:
            # Check anti-hedge logic
            if not self._check_anti_hedge(signal_type):
                self.logger.info(f"Anti-hedge: {signal_type} signal blocked")
                return
            
            # Calculate position size
            base_volume = self.config.lot_size
            
            # Validate order size with risk manager
            valid, adjusted_volume, message = self.risk_manager.validate_order_size(
                base_volume, signal_type
            )
            
            if not valid:
                self.logger.warning(f"Order size validation failed: {message}")
                return
            
            # Calculate take profit
            tp_points = self.position_manager.calculate_take_profit(
                0 if signal_type == "BUY" else 1
            )
            
            # Execute order
            order_type = OrderType.MARKET_BUY if signal_type == "BUY" else OrderType.MARKET_SELL
            comment = f"{signal_type} Signal - RSI: {signal_data.get('rsi', 0):.1f}"
            
            result = self.order_executor.execute_market_order(
                order_type=order_type,
                volume=adjusted_volume,
                tp_points=tp_points,
                comment=comment
            )
            
            if result.success:
                self._on_trade_opened(result, signal_type, signal_data)
            else:
                self.logger.error(f"Failed to execute {signal_type} order: {result.error_msg}")
                self._notify_error(f"Order execution failed: {result.error_msg}")
            
        except Exception as e:
            self.logger.error(f"Entry signal processing error: {e}")
            self._notify_error(f"Entry signal error: {e}")
    
    def _check_anti_hedge(self, signal_type: str) -> bool:
        """Check anti-hedge logic"""
        current_positions = self.position_manager.positions
        
        for position in current_positions.values():
            # If we have a BUY position and signal is SELL (or vice versa), block
            if ((position.type == 0 and signal_type == "SELL") or 
                (position.type == 1 and signal_type == "BUY")):
                return False
        
        return True
    
    def _check_recovery_opportunities(self):
        """Check and execute recovery opportunities"""
        try:
            for ticket, position in self.position_manager.positions.items():
                # Check if recovery is needed
                if self.position_manager.check_recovery_needed(position):
                    
                    # Check if position is already in recovery group
                    group_id = None
                    for gid, group in self.position_manager.recovery_groups.items():
                        if position in group.positions:
                            group_id = gid
                            break
                    
                    # Create new recovery group if needed
                    if group_id is None:
                        group_id = self.position_manager.create_recovery_group(position)
                    
                    # Check if can add recovery
                    if self.position_manager.can_add_recovery(group_id):
                        # Execute recovery order
                        recovery_result = self.order_executor.execute_recovery_order(
                            position, group_id
                        )
                        
                        if recovery_result.success:
                            self.engine_stats['recovery_triggered'] += 1
                            self.logger.info(f"Recovery order executed for position {ticket}")
                        else:
                            self.logger.warning(f"Recovery failed for position {ticket}: {recovery_result.error_msg}")
                            
        except Exception as e:
            self.logger.error(f"Recovery check error: {e}")
            self._notify_error(f"Recovery error: {e}")
    
    def _monitor_positions(self):
        """Monitor existing positions for management"""
        try:
            # Update dynamic TP if enabled
            if self.config.dynamic_tp:
                for group_id, group in self.position_manager.recovery_groups.items():
                    if len(group.positions) > 1:  # Only for recovery groups
                        # Recalculate TP for group
                        new_tp = self.position_manager.calculate_take_profit(
                            group.direction, group.positions
                        )
                        
                        # Update TP for all positions in group
                        for position in group.positions:
                            self.order_executor.modify_position_tp(position.ticket, new_tp)
                            
        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")
    
    def _on_trade_opened(self, result: OrderResult, signal_type: str, signal_data: Dict):
        """Handle trade opened event"""
        trade_info = {
            "ticket": result.ticket,
            "type": signal_type,
            "volume": result.volume,
            "price": result.price,
            "timestamp": datetime.now(),
            "signal_data": signal_data
        }
        
        self.trade_history.append(trade_info)
        self.engine_stats['trades_executed'] += 1
        
        # Notify event handlers
        for handler in self.event_handlers['on_trade_opened']:
            try:
                handler(trade_info)
            except Exception as e:
                self.logger.error(f"Event handler error: {e}")
        
        self.logger.info(f"Trade opened: {signal_type} {result.volume} lots at {result.price}")
    
    def _validate_config(self) -> bool:
        """Validate trading configuration"""
        if self.config.lot_size <= 0:
            self.logger.error("Invalid lot size")
            return False
        
        if not (20 <= self.config.rsi_down <= 50):
            self.logger.error("Invalid RSI_DOWN range")
            return False
        
        if not (50 <= self.config.rsi_up <= 80):
            self.logger.error("Invalid RSI_UP range")
            return False
        
        if self.config.rsi_down >= self.config.rsi_up:
            self.logger.error("RSI_DOWN must be less than RSI_UP")
            return False
        
        return True
    
    def _update_engine_stats(self):
        """Update engine statistics"""
        if self.start_time:
            self.engine_stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
    
    def _notify_state_change(self):
        """Notify state change to event handlers"""
        for handler in self.event_handlers['on_state_changed']:
            try:
                handler(self.state)
            except Exception as e:
                self.logger.error(f"State change handler error: {e}")
    
    def _notify_error(self, error_msg: str):
        """Notify error to event handlers"""
        for handler in self.event_handlers['on_error']:
            try:
                handler(error_msg)
            except Exception as e:
                self.logger.error(f"Error handler error: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler"""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    def get_status(self) -> EngineStatus:
        """Get current engine status"""
        # Get risk report
        risk_report = self.risk_manager.get_risk_report()
        
        # Get position summary
        position_summary = self.position_manager.get_position_summary()
        
        # Get execution stats
        execution_stats = self.order_executor.get_execution_stats()
        
        uptime = self.engine_stats['uptime_seconds'] if self.start_time else 0
        
        return EngineStatus(
            state=self.state,
            uptime=uptime,
            last_update=self.last_update,
            total_trades=execution_stats.get('total_orders', 0),
            successful_trades=execution_stats.get('successful_orders', 0),
            current_positions=position_summary.get('total_positions', 0),
            total_pnl=position_summary.get('total_profit', 0),
            risk_level=risk_report.get('risk_level', 'low'),
            restrictions=risk_report.get('restrictions', []),
            last_signal=self.signal_history[-1]['type'] if self.signal_history else "",
            last_trade=self.trade_history[-1]['timestamp'] if self.trade_history else None,
            errors=[f"Error count: {self.engine_stats['errors_occurred']}"]
        )
    
    def get_detailed_status(self) -> Dict:
        """Get detailed engine status for UI"""
        status = self.get_status()
        
        return {
            "engine": {
                "state": status.state.value,
                "uptime": status.uptime,
                "last_update": status.last_update.isoformat() if status.last_update else None,
                "stats": self.engine_stats
            },
            "trading": {
                "total_trades": status.total_trades,
                "successful_trades": status.successful_trades,
                "current_positions": status.current_positions,
                "total_pnl": status.total_pnl,
                "last_trade": status.last_trade.isoformat() if status.last_trade else None
            },
            "risk": self.risk_manager.get_risk_report(),
            "positions": self.position_manager.get_position_summary(),
            "recovery": self.position_manager.get_recovery_status(),
            "execution": self.order_executor.get_execution_stats(),
            "config": self.config.to_dict()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            "signals_generated": self.engine_stats['signals_generated'],
            "trades_executed": self.engine_stats['trades_executed'],
            "recovery_triggered": self.engine_stats['recovery_triggered'],
            "success_rate": (self.engine_stats['trades_executed'] / 
                           max(1, self.engine_stats['signals_generated'])) * 100,
            "uptime_hours": self.engine_stats['uptime_seconds'] / 3600,
            "errors_per_hour": (self.engine_stats['errors_occurred'] / 
                              max(1, self.engine_stats['uptime_seconds'] / 3600)),
            "signal_history": self.signal_history[-50:],  # Last 50 signals
            "trade_history": self.trade_history[-50:]     # Last 50 trades
        }
    
    def save_state(self, filepath: str):
        """Save engine state to file"""
        try:
            state_data = {
                "config": self.config.to_dict(),
                "stats": self.engine_stats,
                "trade_history": [
                    {**trade, "timestamp": trade["timestamp"].isoformat()} 
                    for trade in self.trade_history
                ],
                "signal_history": self.signal_history,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.info(f"Engine state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """Load engine state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration
            self.config.update_from_dict(state_data.get("config", {}))
            
            # Restore statistics
            self.engine_stats.update(state_data.get("stats", {}))
            
            # Restore histories
            self.signal_history = state_data.get("signal_history", [])
            
            # Restore trade history with datetime conversion
            trade_history = state_data.get("trade_history", [])
            for trade in trade_history:
                if "timestamp" in trade:
                    trade["timestamp"] = datetime.fromisoformat(trade["timestamp"])
            self.trade_history = trade_history
            
            self.logger.info(f"Engine state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False